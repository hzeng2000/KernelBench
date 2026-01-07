import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_kernel(batch_size: int, out_features: int, block_M: int = 64, block_N: int = 128, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((batch_size, out_features), dtype),
        B: T.Tensor((out_features,), dtype),
        C: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(out_features, block_N), T.ceildiv(batch_size, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < batch_size and x < out_features:
                    val = A[y, x] + B[x]
                    
                    # Swish: sigmoid(x) * x
                    sigmoid = 1.0 / (1.0 + T.exp(-val))
                    val = sigmoid * val
                    
                    # Tanh
                    val = T.tanh(val)
                    
                    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    gelu_c = math.sqrt(2.0 / math.pi)
                    tanh_arg = gelu_c * (val + 0.044715 * val * val * val)
                    gelu_val = 0.5 * val * (1.0 + T.tanh(tanh_arg))
                    val = gelu_val
                    
                    # Hardtanh: clamp(x, -1, 1)
                    if val < -1.0:
                        val = -1.0
                    elif val > 1.0:
                        val = 1.0
                    
                    C[y, x] = val

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self._kernel_cache = {}
        self.out_features = out_features

    def _get_kernel(self, batch_size: int, out_features: int, tl_dtype: str):
        key = (batch_size, out_features, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(batch_size, out_features, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.matmul(x)
        
        batch_size = x.shape[0]
        kernel = self._get_kernel(batch_size, self.out_features, "float16")
        
        x_fp16 = x.half()
        add_value_fp16 = self.add_value.half()
        
        x_out = kernel(x_fp16, add_value_fp16)
        
        return x_out.float()