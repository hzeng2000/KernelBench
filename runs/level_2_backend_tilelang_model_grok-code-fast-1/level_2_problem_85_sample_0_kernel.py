import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_scale_maxpool_clamp_kernel(batch: int, out_c: int, h: int, w: int, maxpool_k: int, block_size: int = 128, threads: int = 128, dtype: str = "float16"):
    oh = h // maxpool_k
    ow = w // maxpool_k
    total_elements = batch * out_c * oh * ow

    @T.prim_func
    def fused_scale_maxpool_clamp_kernel(
        Input: T.Tensor((batch, out_c, h, w), dtype),
        Scale: T.Tensor((out_c, 1, 1), dtype),
        Output: T.Tensor((batch, out_c, oh, ow), dtype),
        clamp_min: T.float32,
        clamp_max: T.float32,
    ):
        with T.Kernel(T.ceildiv(total_elements, block_size), threads=threads) as i:
            b = i // (out_c * oh * ow)
            oc = (i % (out_c * oh * ow)) // (oh * ow)
            oy = (i % (oh * ow)) // ow
            ox = i % ow

            if b < batch and oc < out_c and oy < oh and ox < ow:
                max_val = T.min_value(dtype)
                for ky in T.serial(maxpool_k):
                    for kx in T.serial(maxpool_k):
                        iy = oy * maxpool_k + ky
                        ix = ox * maxpool_k + kx
                        val = Input[b, oc, iy, ix] * Scale[oc, 0, 0]
                        max_val = T.max(max_val, val)
                Output[b, oc, oy, ox] = T.clamp(max_val, clamp_min, clamp_max)

    return tilelang.compile(fused_scale_maxpool_clamp_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs convolution, group normalization, and a fused scaling + max pooling + clamping kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self._kernel_cache = {}

    def _get_kernel(self, batch: int, out_c: int, h: int, w: int, maxpool_k: int, tl_dtype: str):
        key = (batch, out_c, h, w, maxpool_k, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_scale_maxpool_clamp_kernel(batch, out_c, h, w, maxpool_k, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        x = self.group_norm(x)
        # x is now (batch, out_c, h, w), where h and w are after conv
        batch, out_c, h, w = x.shape
        kernel = self._get_kernel(batch, out_c, h, w, self.maxpool_kernel_size, "float16")
        x_c = x.contiguous()
        scale_c = self.scale.contiguous()
        output = kernel(x_c, scale_c, self.clamp_min, self.clamp_max)
        return output