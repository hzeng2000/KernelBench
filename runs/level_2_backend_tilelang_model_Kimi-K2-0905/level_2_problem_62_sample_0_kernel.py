import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_matmul_gn_leakyrelu_kernel(M: int, N: int, K: int, num_groups: int, eps: float = 1e-5, negative_slope: float = 0.01,
                                           block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        gn_weight: T.Tensor((N,), dtype),
        gn_bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_m = by * block_M
            start_n = bx * block_N
            
            # Shared memory for tile
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            # Initialize C_local to zero
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.cast(0.0, dtype)
            
            # Compute mean and variance for group norm
            group_size = N // num_groups
            mean_shared = T.alloc_shared((block_M, num_groups), "float32")
            var_shared = T.alloc_shared((block_M, num_groups), "float32")
            
            # Compute mean per group
            for g in range(num_groups):
                for m in range(block_M):
                    if start_m + m < M:
                        sum_val = T.cast(0.0, "float32")
                        for n in range(group_size):
                            if start_n + g * group_size + n < N:
                                sum_val += T.cast(C_local[m, g * group_size + n], "float32")
                        mean_shared[m, g] = sum_val / T.cast(group_size, "float32")
            
            # Compute variance per group
            for g in range(num_groups):
                for m in range(block_M):
                    if start_m + m < M:
                        var_val = T.cast(0.0, "float32")
                        for n in range(group_size):
                            if start_n + g * group_size + n < N:
                                diff = T.cast(C_local[m, g * group_size + n], "float32") - mean_shared[m, g]
                                var_val += diff * diff
                        var_shared[m, g] = var_val / T.cast(group_size, "float32")
            
            # Main matmul loop
            for k in range(0, K, block_K):
                # Load A tile
                for i, j in T.Parallel(block_M, block_K):
                    if start_m + i < M and k + j < K:
                        A_shared[i, j] = A[start_m + i, k + j]
                    else:
                        A_shared[i, j] = T.cast(0.0, dtype)
                
                # Load B tile
                for i, j in T.Parallel(block_N, block_K):
                    if start_n + i < N and k + j < K:
                        B_shared[i, j] = B[start_n + i, k + j]
                    else:
                        B_shared[i, j] = T.cast(0.0, dtype)
                
                # Compute matmul
                for i, j in T.Parallel(block_M, block_N):
                    for kk in range(block_K):
                        C_local[i, j] += A_shared[i, kk] * B_shared[j, kk]
            
            # Add bias, apply group norm and leaky relu
            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < M and start_n + j < N:
                    # Add bias
                    val = C_local[i, j] + bias[start_n + j]
                    
                    # Group norm
                    g = j // group_size
                    mean = mean_shared[i, g]
                    var = var_shared[i, g]
                    normalized = (T.cast(val, "float32") - mean) / T.sqrt(var + T.cast(eps, "float32"))
                    val = T.cast(normalized, dtype) * gn_weight[start_n + j] + gn_bias[start_n + j]
                    
                    # Leaky ReLU
                    if val < T.cast(0.0, dtype):
                        val = val * T.cast(negative_slope, dtype)
                    
                    # Element-wise add (x + x)
                    C[start_m + i, start_n + j] = val + val
    
    return tilelang.compile(fused_kernel, out_idx=[5], target="cuda")


def build_elementwise_add_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def elementwise_add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                C[y, x] = A[y, x] + B[y, x]

    return tilelang.compile(elementwise_add_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        
        self._kernel_cache = {}

    def _get_fused_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = ("fused", M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_matmul_gn_leakyrelu_kernel(
                M, N, K, self.num_groups, self.eps, self.negative_slope, dtype=tl_dtype
            )
        return self._kernel_cache[key]

    def _get_add_kernel(self, M: int, N: int, tl_dtype: str):
        key = ("add", M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_elementwise_add_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get original shape
        original_shape = x.shape
        batch_size = x.size(0)
        
        # Convert to fp16
        x = x.half()
        
        # Linear layer (matmul + bias)
        weight = self.fc.weight.half()
        bias = self.fc.bias.half()
        
        # Get group norm parameters
        gn_weight = self.gn.weight.half()
        gn_bias = self.gn.bias.half()
        
        # Prepare for kernel
        M = batch_size
        K = self.input_size
        N = self.hidden_size
        
        kernel = self._get_fused_kernel(M, N, K, "float16")
        
        # Allocate output
        output = torch.empty(M, N, dtype=torch.float16, device=x.device)
        
        # Run fused kernel
        kernel(x, weight, bias, gn_weight, gn_bias, output)
        
        # Element-wise add (x + x)
        add_kernel = self._get_add_kernel(M, N, "float16")
        final_output = torch.empty_like(output)
        add_kernel(output, output, final_output)
        
        return final_output.float()