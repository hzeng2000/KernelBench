import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_layernorm_gelu_scale_kernel(B: int, C: int, DD: int, HH: int, WW: int, eps: float, scaling_factor: float, dtype: str = "float16"):
    
    @T.prim_func
    def fused_layernorm_gelu_scale_kernel(
        X: T.Tensor((B, C, DD, HH, WW), dtype),
        Y: T.Tensor((B, C, DD, HH, WW), dtype),
        eps: T.float32,
        scaling_factor: T.float32,
    ):
        with T.Kernel(B, DD, HH, WW, threads=1) as (b, dd, hh, ww):
            sum_x = T.alloc((1,), "float32", scope="local")
            sum_sq = T.alloc((1,), "float32", scope="local")
            sum_x[0] = 0.0
            sum_sq[0] = 0.0
            with T.serial(C) as c:
                val = X[b, c, dd, hh, ww]
                sum_x[0] += val
                sum_sq[0] += val * val
            mean = sum_x[0] / C
            var = sum_sq[0] / C - mean * mean
            with T.Parallel(C) as c:
                x = X[b, c, dd, hh, ww]
                normalized = (x - mean) / T.sqrt(var + eps)
                # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                gelu_val = 0.5 * normalized * (1 + T.tanh(0.7978845608028654 * (normalized + 0.044715 * normalized * normalized * normalized)))
                Y[b, c, dd, hh, ww] = gelu_val * scaling_factor

    return tilelang.compile(fused_layernorm_gelu_scale_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by a fused LayerNorm + GELU + scaling kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.eps = eps
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, DD: int, HH: int, WW: int, eps: float, scaling_factor: float, tl_dtype: str):
        key = (B, C, DD, HH, WW, eps, scaling_factor, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_layernorm_gelu_scale_kernel(B, C, DD, HH, WW, eps, scaling_factor, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv_transpose(x)
        x_c = x.contiguous()
        B, C, DD, HH, WW = x_c.shape
        kernel = self._get_kernel(B, C, DD, HH, WW, self.eps, self.scaling_factor, "float16")
        y = kernel(x_c, self.eps, self.scaling_factor)
        return y