import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_relu_group_norm_kernel(B: int, C: int, D: int, H: int, W: int, G: int, eps: float = 1e-5, dtype: str = "float16"):
    channels_per_group = C // G
    num_elements = channels_per_group * D * H * W

    @T.prim_func
    def fused_relu_group_norm_kernel(
        X: T.Tensor((B, C, D, H, W), dtype),
        weight: T.Tensor((C,), dtype),
        bias: T.Tensor((C,), dtype),
        Y: T.Tensor((B, C, D, H, W), dtype),
    ):
        temp = T.alloc((B, C, D, H, W), dtype)
        with T.Parallel(B, C, D, H, W):
            temp[T.axis(0), T.axis(1), T.axis(2), T.axis(3), T.axis(4)] = T.max(0, X[T.axis(0), T.axis(1), T.axis(2), T.axis(3), T.axis(4)])

        sum_val = T.alloc((B, G), dtype)
        sum_sq = T.alloc((B, G), dtype)
        with T.serial(B):
            with T.serial(G):
                sum_val[T.axis("b"), T.axis("g")] = T.sum(temp[T.axis("b"), T.axis("g") * channels_per_group:(T.axis("g") + 1) * channels_per_group, :, :, :], axis=[1, 2, 3, 4])
                sum_sq[T.axis("b"), T.axis("g")] = T.sum(temp[T.axis("b"), T.axis("g") * channels_per_group:(T.axis("g") + 1) * channels_per_group, :, :, :] ** 2, axis=[1, 2, 3, 4])

        mean = sum_val / num_elements
        var = sum_sq / num_elements - mean ** 2

        with T.Parallel(B, C, D, H, W):
            b = T.axis(0)
            c = T.axis(1)
            d = T.axis(2)
            h = T.axis(3)
            w = T.axis(4)
            g = c // channels_per_group
            val = temp[b, c, d, h, w]
            Y[b, c, d, h, w] = (val - mean[b, g]) / T.sqrt(var[b, g] + eps) * weight[c] + bias[c]

    return tilelang.compile(fused_relu_group_norm_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, D: int, H: int, W: int, G: int, tl_dtype: str):
        key = (B, C, D, H, W, G, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_relu_group_norm_kernel(B, C, D, H, W, G, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, D, H, W = x.shape
        G = self.group_norm.num_groups
        kernel = self._get_kernel(B, C, D, H, W, G, "float16")
        x = kernel(x, self.group_norm.weight, self.group_norm.bias)
        return x