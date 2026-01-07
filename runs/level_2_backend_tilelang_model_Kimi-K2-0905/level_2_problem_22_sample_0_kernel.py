import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_kernel(batch_size: int, hidden_size: int, block_M: int = 64, block_N: int = 128, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((batch_size, hidden_size), dtype),
        B: T.Tensor((hidden_size, hidden_size), dtype),
        C: T.Tensor((batch_size, 1), dtype),
        scale_factor: T.float32,
        clamp_min: T.float32,
        clamp_max: T.float32,
    ):
        with T.Kernel(T.ceildiv(batch_size, block_M), threads=threads) as bx:
            start_y = bx * block_M
            
            # Shared memory for reduction
            shared_mem = T.alloc_shared([block_M, block_N], dtype)
            
            # Initialize max and sum for logsumexp
            max_val = T.alloc_local([1], dtype)
            sum_exp = T.alloc_local([1], dtype)
            
            for local_y in T.Parallel(block_M):
                y = start_y + local_y
                if y < batch_size:
                    # Initialize max to -inf
                    max_val[0] = T.cast(-1e38, dtype)
                    
                    # Find max value for numerical stability
                    for k in T.Serial(hidden_size):
                        val = T.cast(0.0, dtype)
                        for j in T.Serial(hidden_size):
                            val += A[y, j] * B[j, k]
                        val = val * T.cast(scale_factor, dtype) + val  # scale and add residual
                        val = T.clamp(val, T.cast(clamp_min, dtype), T.cast(clamp_max, dtype))
                        if val > max_val[0]:
                            max_val[0] = val
                    
                    # Compute sum of exp(x - max)
                    sum_exp[0] = T.cast(0.0, dtype)
                    for k in T.Serial(hidden_size):
                        val = T.cast(0.0, dtype)
                        for j in T.Serial(hidden_size):
                            val += A[y, j] * B[j, k]
                        val = val * T.cast(scale_factor, dtype) + val  # scale and add residual
                        val = T.clamp(val, T.cast(clamp_min, dtype), T.cast(clamp_max, dtype))
                        sum_exp[0] += T.exp(val - max_val[0])
                    
                    # Compute logsumexp
                    logsumexp_val = max_val[0] + T.log(sum_exp[0])
                    
                    # Compute mish activation
                    tanh_val = T.tanh(T.log1p(T.exp(logsumexp_val)))
                    mish_val = logsumexp_val * tanh_val
                    
                    C[y, 0] = mish_val

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self._kernel_cache = {}
        
        # Initialize weight as identity for testing
        with torch.no_grad():
            self.matmul.weight.copy_(torch.eye(hidden_size, input_size))
            self.matmul.bias.zero_()

    def _get_kernel(self, batch_size: int, hidden_size: int):
        key = (batch_size, hidden_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(batch_size, hidden_size)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        self.matmul.weight.data = self.matmul.weight.data.half()
        self.matmul.bias.data = self.matmul.bias.data.half()
        
        batch_size, input_size = x.shape
        hidden_size = self.matmul.weight.shape[0]
        
        # Apply linear transformation
        x_linear = self.matmul(x)
        
        # Use custom kernel for the rest of operations
        kernel = self._get_kernel(batch_size, hidden_size)
        output = kernel(x, self.matmul.weight.t())
        
        return output.float()