import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_linear_relu_div_kernel(
    M: int, 
    K: int, 
    N: int, 
    block_M: int = 64, 
    block_N: int = 64, 
    block_K: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    
    @T.prim_func
    def fused_linear_relu_div_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
        Divisor: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype, accum=True)
            
            start_m = by * block_M
            start_n = bx * block_N
            
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                k_start = k * block_K
                
                # Load A tile to shared memory
                for i, j in T.Parallel(block_M, block_K):
                    if start_m + i < M and k_start + j < K:
                        A_shared[i, j] = A[start_m + i, k_start + j]
                    else:
                        A_shared[i, j] = T.cast(0, dtype)
                
                # Load B tile to shared memory (transposed)
                for i, j in T.Parallel(block_K, block_N):
                    if k_start + i < K and start_n + j < N:
                        B_shared[i, j] = B[start_n + j, k_start + i]
                    else:
                        B_shared[i, j] = T.cast(0, dtype)
                
                # Compute matmul tile
                for i, j, k_inner in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += T.cast(A_shared[i, k_inner], "float32") * T.cast(B_shared[k_inner, j], "float32")
            
            # Apply ReLU and division, then store result
            divisor_val = Divisor[0]
            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < M and start_n + j < N:
                    relu_val = T.max(T.cast(C_local[i, j], "float32"), 0.0)
                    div_val = relu_val / divisor_val
                    C[start_m + i, start_n + j] = T.cast(div_val, dtype)

    return tilelang.compile(fused_linear_relu_div_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = nn.Parameter(torch.tensor([divisor], dtype=torch.float32), requires_grad=False)
        self._kernel_cache = {}

    def _get_kernel(self, M: int, K: int, N: int):
        key = (M, K, N)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_linear_relu_div_kernel(M, K, N)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure weight is contiguous and in correct dtype
        weight = self.linear.weight.data.t().contiguous().half()
        bias = self.linear.bias.data.half() if self.linear.bias is not None else None
        
        # Get input shape
        original_shape = x.shape
        M = x.numel() // x.size(-1)
        K = x.size(-1)
        N = weight.size(1)
        
        # Flatten input
        x_flat = x.view(M, K).half().contiguous()
        
        # Get kernel
        kernel = self._get_kernel(M, K, N)
        
        # Allocate output
        output = torch.empty(M, N, dtype=torch.float16, device=x.device)
        
        # Run fused kernel
        kernel(x_flat, weight, output, self.divisor)
        
        # Add bias if present
        if bias is not None:
            output = output + bias.unsqueeze(0)
        
        # Reshape output
        new_shape = list(original_shape[:-1]) + [N]
        return output.view(new_shape)