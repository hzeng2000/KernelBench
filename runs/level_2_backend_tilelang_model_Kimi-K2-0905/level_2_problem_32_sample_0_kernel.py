import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv_scale_min_kernel(
    batch_size: int,
    out_channels: int,
    out_height: int,
    out_width: int,
    kernel_size: int,
    in_channels: int,
    stride: int = 1,
    padding: int = 1,
    block_M: int = 8,
    block_N: int = 16,
    block_K: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def conv_scale_min_kernel(
        X: T.Tensor((batch_size, in_channels, out_height + 2 * padding, out_width + 2 * padding), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        Out: T.Tensor((batch_size, 1, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_N), T.ceildiv(out_height, block_M), batch_size, threads=threads) as (bx, by, bz):
            start_x = bx * block_N
            start_y = by * block_M
            start_z = bz

            local_accum = T.alloc_fragment((block_M, block_N), dtype, "local")
            min_accum = T.alloc_fragment((block_M, block_N), dtype, "local")
            
            for local_y, local_x in T.Parallel(block_M, block_N):
                local_accum[local_y, local_x] = T.cast(0, dtype)
                min_accum[local_y, local_x] = T.cast(1e10, dtype)

            for kc in range(in_channels):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        for local_y, local_x in T.Parallel(block_M, block_N):
                            y = start_y + local_y
                            x = start_x + local_x
                            if y < out_height and x < out_width:
                                in_y = y * stride + kh - padding
                                in_x = x * stride + kw - padding
                                if 0 <= in_y < out_height + 2 * padding and 0 <= in_x < out_width + 2 * padding:
                                    val = X[start_z, kc, in_y, in_x]
                                    for oc in range(out_channels):
                                        w_val = W[oc, kc, kh, kw]
                                        conv_val = val * w_val
                                        if oc == 0:
                                            min_accum[local_y, local_x] = conv_val
                                        else:
                                            min_accum[local_y, local_x] = T.min(min_accum[local_y, local_x], conv_val)

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                if y < out_height and x < out_width:
                    Out[start_z, 0, y, x] = min_accum[local_y, local_x] * T.cast(2.0, dtype)

    return tilelang.compile(conv_scale_min_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.scale_factor = scale_factor
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, out_channels: int, out_height: int, out_width: int, in_channels: int, kernel_size: int):
        key = (batch_size, out_channels, out_height, out_width, in_channels, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_scale_min_kernel(
                batch_size, out_channels, out_height, out_width, kernel_size, in_channels
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        out_height = height
        out_width = width
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        # Pad input manually
        pad = kernel_size // 2
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0)

        kernel = self._get_kernel(batch_size, out_channels, out_height, out_width, in_channels, kernel_size)
        weight = self.conv.weight.half()
        output = kernel(x_padded.half(), weight)

        return output