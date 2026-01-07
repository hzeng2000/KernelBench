import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_gemm_scale_bn_kernel(M: int, K: int, N: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def gemm_scale_bn_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        scale: T.Tensor((N,), dtype),
        running_mean: T.Tensor((N,), "float32"),
        running_var: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_m = by * block_M
            start_n = bx * block_N

            # Allocate shared memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            
            # Allocate local accumulator
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            
            # Initialize accumulator
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = 0.0
            
            # Main computation loop
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                # Load A tile to shared memory
                for i, j in T.Parallel(block_M, block_K):
                    global_i = start_m + i
                    global_j = k * block_K + j
                    if global_i < M and global_j < K:
                        A_shared[i, j] = A[global_i, global_j]
                    else:
                        A_shared[i, j] = 0.0
                
                # Load B tile to shared memory
                for i, j in T.Parallel(block_N, block_K):
                    global_i = start_n + i
                    global_j = k * block_K + j
                    if global_i < N and global_j < K:
                        B_shared[i, j] = B[global_i, global_j]
                    else:
                        B_shared[i, j] = 0.0
                
                # Compute GEMM tile
                for i, j, kk in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += T.cast(A_shared[i, kk], "float32") * T.cast(B_shared[j, kk], "float32")
            
            # Apply bias, scale, and batch norm
            for i, j in T.Parallel(block_M, block_N):
                global_i = start_m + i
                global_j = start_n + j
                
                if global_i < M and global_j < N:
                    # Add bias
                    val = C_local[i, j] + T.cast(bias[global_j], "float32")
                    
                    # Apply scale
                    val = val * T.cast(scale[global_j], "float32")
                    
                    # Apply batch norm
                    mean = running_mean[global_j]
                    var = running_var[global_j]
                    eps = 1e-5
                    val = (val - mean) / T.sqrt(var + eps)
                    
                    C[global_i, global_j] = T.cast(val, dtype)

    return tilelang.compile(gemm_scale_bn_kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self._kernel_cache = {}
        
        # Initialize running stats for inference
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))

    def _get_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_scale_bn_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and in fp16
        x = x.contiguous().half()
        
        M, K = x.shape
        N = self.gemm.out_features
        
        # Get kernel
        kernel = self._get_kernel(M, K, N, "float16")
        
        # Get weight in correct format
        weight = self.gemm.weight.contiguous().half()
        
        # Get bias
        bias = self.gemm.bias.contiguous().half() if self.gemm.bias is not None else torch.zeros(N, dtype=torch.float16, device=x.device)
        
        # Get scale
        scale = self.scale.contiguous().half()
        
        # Use running stats for inference
        if not self.training:
            running_mean = self.bn.running_mean
            running_var = self.bn.running_var
        else:
            # For training, compute stats on the fly (simplified)
            with torch.no_grad():
                # Compute output without BN first to get stats
                temp_out = torch.nn.functional.linear(x, weight, bias) * scale
                running_mean = temp_out.mean(dim=0)
                running_var = temp_out.var(dim=0, unbiased=False)
                # Update running stats
                self.bn.running_mean.copy_(self.bn.momentum * running_mean + (1 - self.bn.momentum) * self.bn.running_mean)
                self.bn.running_var.copy_(self.bn.momentum * running_var + (1 - self.bn.momentum) * self.bn.running_var)
        
        # Run fused kernel
        output = kernel(x, weight, bias, scale, running_mean, running_var)
        
        # Apply BN weight and bias if they exist
        if self.bn.affine:
            output = output * self.bn.weight.unsqueeze(0) + self.bn.bias.unsqueeze(0)
        
        return output