import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_gelu_groupnorm_kernel(N: int, C: int, H: int, W: int, num_groups: int, dtype: str = "float16"):
    group_size = C // num_groups
    
    @T.prim_func
    def gelu_groupnorm_kernel(
        X: T.Tensor((N, C, H, W), dtype),
        weight: T.Tensor((C,), dtype),
        bias: T.Tensor((C,), dtype),
        Y: T.Tensor((N, C, H, W), dtype),
    ):
        with T.Kernel(N, H, W, threads=C) as (n, h, w):
            gelu_x = T.alloc_buffer((C,), dtype, scope="local")
            
            for c in T.serial(C):
                x_val = X[n, c, h, w]
                gelu_val = 0.5 * x_val * (1 + T.tanh(T.sqrt(2 / 3.141592653589793) * (x_val + 0.044715 * x_val * x_val * x_val)))
                gelu_x[c] = gelu_val
            
            for g in T.serial(num_groups):
                sum_gelu = T.alloc_buffer((), dtype, scope="local", init=0)
                sum_sq_gelu = T.alloc_buffer((), dtype, scope="local", init=0)
                
                for c in T.serial(group_size):
                    c_idx = g * group_size + c
                    sum_gelu[()] += gelu_x[c_idx]
                    sum_sq_gelu[()] += gelu_x[c_idx] * gelu_x[c_idx]
                
                mean = sum_gelu[()] / group_size
                var = sum_sq_gelu[()] / group_size - mean * mean
                
                for c in T.serial(group_size):
                    c_idx = g * group_size + c
                    normalized = (gelu_x[c_idx] - mean) / T.sqrt(var + 1e-5)
                    Y[n, c_idx, h, w] = normalized * weight[c_idx] + bias[c_idx]
    
    return tilelang.compile(gelu_groupnorm_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self._kernel_cache = {}
        self.num_groups = num_groups

    def _get_kernel(self, N: int, C: int, H: int, W: int, tl_dtype: str):
        key = (N, C, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gelu_groupnorm_kernel(N, C, H, W, self.num_groups, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = x.contiguous()
        N, C, H, W = x.shape
        kernel = self._get_kernel(N, C, H, W, "float16")
        y = torch.empty_like(x, dtype=torch.float16)
        kernel(x, self.group_norm.weight.to(torch.float16), self.group_norm.bias.to(torch.float16), y)
        return y