import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_matmul_relu_div_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    divisor: float,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    threads: int = 128,
    dtype: str = "float16"
):
    
    @T.prim_func
    def fused_matmul_relu_div_kernel(
        A: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        bias: T.Tensor((out_features,), dtype),
        divisor: T.Tensor((), dtype),
        C: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_M), T.ceildiv(out_features, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(W[by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K], W_shared)
                
                with T.Parallel(block_M, block_N):
                    for kk in T.serial(block_K):
                        C_local[T.thread_y, T.thread_x] += A_shared[T.thread_y, kk] * W_shared[T.thread_x, kk]
            
            with T.Parallel(block_M, block_N):
                i = bx * block_M + T.thread_y
                j = by * block_N + T.thread_x
                if i < batch_size and j < out_features:
                    temp = C_local[T.thread_y, T.thread_x] + bias[j]
                    temp = T.max(temp, T.cast(0.0, dtype))
                    C[i, j] = temp / divisor[()]
    
    return tilelang.compile(fused_matmul_relu_div_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model that fuses matrix multiplication, ReLU, and division into a single TileLang kernel.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self._kernel_cache = {}
    
    def _get_kernel(self, batch_size: int, in_features: int, out_features: int, divisor: float, tl_dtype: str):
        key = (batch_size, in_features, out_features, divisor, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_matmul_relu_div_kernel(batch_size, in_features, out_features, divisor, dtype=tl_dtype)
        return self._kernel_cache[key]
    
    def forward(self, x):
        x_c = x.contiguous()
        weight_c = self.linear.weight.contiguous()
        bias_c = self.linear.bias.contiguous()
        divisor_tensor = torch.tensor(self.divisor, dtype=torch.float16, device=x.device)
        
        batch_size, in_features = x_c.shape
        out_features = weight_c.shape[0]
        
        kernel = self._get_kernel(batch_size, in_features, out_features, self.divisor, "float16")
        C = kernel(x_c, weight_c, bias_c, divisor_tensor)
        
        return C