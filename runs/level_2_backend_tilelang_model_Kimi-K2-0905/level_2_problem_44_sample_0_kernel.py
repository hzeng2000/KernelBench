import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv_transpose_kernel(
    batch: int, in_c: int, out_c: int, in_h: int, in_w: int,
    out_h: int, out_w: int, k_h: int, k_w: int,
    stride_h: int, stride_w: int, pad_h: int, pad_w: int,
    out_pad_h: int, out_pad_w: int, threads: int = 256,
    block_h: int = 8, block_w: int = 16, dtype: str = "float16"
):
    @T.prim_func
    def conv_transpose_kernel(
        Input: T.Tensor((batch, in_c, in_h, in_w), dtype),
        Weight: T.Tensor((in_c, out_c, k_h, k_w), dtype),
        Output: T.Tensor((batch, out_c, out_h, out_w), dtype),
    ):
        with T.Kernel(T.ceildiv(out_w, block_w), T.ceildiv(out_h, block_h), batch * out_c, threads=threads) as (bx, by, bz):
            # Block indices
            b = bz // out_c
            oc = bz % out_c
            oh = by * block_h
            ow = bx * block_w

            # Allocate shared memory for input tile
            shared_in = T.alloc_shared((in_c, in_h, in_w), dtype)
            # Allocate shared memory for weight tile
            shared_w = T.alloc_shared((in_c, out_c, k_h, k_w), dtype)

            # Load input tile
            for ic in T.Parallel(in_c):
                for ih in range(in_h):
                    for iw in range(in_w):
                        shared_in[ic, ih, iw] = Input[b, ic, ih, iw]

            # Load weight tile
            for ic in T.Parallel(in_c):
                for oc_inner in range(out_c):
                    for kh in range(k_h):
                        for kw in range(k_w):
                            shared_w[ic, oc_inner, kh, kw] = Weight[ic, oc_inner, kh, kw]

            # Compute output tile
            for local_oh in T.Parallel(block_h):
                for local_ow in range(block_w):
                    o_h = oh + local_oh
                    o_w = ow + local_ow
                    if o_h < out_h and o_w < out_w:
                        acc = T.cast(0.0, dtype)
                        for ic in range(in_c):
                            for kh in range(k_h):
                                for kw in range(k_w):
                                    # Compute input coordinates
                                    i_h = o_h + pad_h - kh * stride_h - out_pad_h
                                    i_w = o_w + pad_w - kw * stride_w - out_pad_w
                                    if i_h >= 0 and i_h < in_h and i_w >= 0 and i_w < in_w:
                                        acc += shared_in[ic, i_h, i_w] * shared_w[ic, oc, kh, kw]
                        Output[b, oc, o_h, o_w] = acc

    return tilelang.compile(conv_transpose_kernel, out_idx=[2], target="cuda")


def build_fused_multiply_global_pool_kernel(
    batch: int, channels: int, height: int, width: int,
    multiplier: float, threads: int = 256,
    block_c: int = 64, dtype: str = "float16"
):
    @T.prim_func
    def fused_kernel(
        Input: T.Tensor((batch, channels, height, width), dtype),
        Output: T.Tensor((batch, channels, 1, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(channels, block_c), batch, threads=threads) as (bx, by):
            b = by
            c_start = bx * block_c

            # Allocate shared memory for reduction
            shared_sum = T.alloc_shared((block_c,), dtype)

            # Initialize shared memory
            for local_c in T.Parallel(block_c):
                shared_sum[local_c] = T.cast(0.0, dtype)

            # Compute sum over spatial dimensions
            for h in range(height):
                for w in range(width):
                    for local_c in T.Parallel(block_c):
                        c = c_start + local_c
                        if c < channels:
                            val = Input[b, c, h, w] * T.cast(multiplier, dtype)
                            shared_sum[local_c] += val

            # Write output
            for local_c in T.Parallel(block_c):
                c = c_start + local_c
                if c < channels:
                    # First global average pooling
                    avg1 = shared_sum[local_c] / T.cast(height * width, dtype)
                    # Second global average pooling (on 1x1 spatial dims, so it's identity)
                    Output[b, c, 0, 0] = avg1

    return tilelang.compile(fused_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self._kernel_cache = {}

    def _get_conv_transpose_kernel(self, batch, in_c, out_c, in_h, in_w, out_h, out_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w):
        key = (batch, in_c, out_c, in_h, in_w, out_h, out_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose_kernel(
                batch, in_c, out_c, in_h, in_w, out_h, out_w, k_h, k_w,
                stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w
            )
        return self._kernel_cache[key]

    def _get_fused_kernel(self, batch, channels, height, width):
        key = (batch, channels, height, width, self.multiplier)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_multiply_global_pool_kernel(
                batch, channels, height, width, self.multiplier
            )
        return self._kernel_cache[key]

    def forward(self, x):
        # Get conv transpose weight
        weight = self.conv_transpose.weight  # Shape: (in_c, out_c, k_h, k_w)
        
        # Compute output spatial dimensions
        batch, in_c, in_h, in_w = x.shape
        out_c = self.conv_transpose.out_channels
        k_h, k_w = self.conv_transpose.kernel_size
        stride_h, stride_w = self.conv_transpose.stride
        pad_h, pad_w = self.conv_transpose.padding
        out_pad_h, out_pad_w = self.conv_transpose.output_padding
        
        out_h = (in_h - 1) * stride_h - 2 * pad_h + k_h + out_pad_h
        out_w = (in_w - 1) * stride_w - 2 * pad_w + k_w + out_pad_w
        
        # Convert to fp16
        x_fp16 = x.half()
        weight_fp16 = weight.half()
        
        # Run custom conv transpose kernel
        kernel = self._get_conv_transpose_kernel(
            batch, in_c, out_c, in_h, in_w, out_h, out_w, k_h, k_w,
            stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w
        )
        conv_out = kernel(x_fp16, weight_fp16)
        
        # Run fused multiply + global pooling kernel
        fused_kernel = self._get_fused_kernel(batch, out_c, out_h, out_w)
        output = fused_kernel(conv_out)
        
        return output