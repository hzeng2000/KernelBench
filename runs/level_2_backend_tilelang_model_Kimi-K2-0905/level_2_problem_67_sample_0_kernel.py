import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv_gelu_gavg_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    block_H: int = 8,
    block_W: int = 8,
    block_OC: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    pad = kernel_size // 2
    out_height = height
    out_width = width

    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, in_channels, height, width), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        B: T.Tensor((out_channels,), dtype),
        Out: T.Tensor((batch_size, out_channels), dtype)
    ):
        # Grid: (out_channels/block_OC, out_width/block_W, out_height/block_H, batch_size)
        with T.Kernel(T.ceildiv(out_channels, block_OC), T.ceildiv(out_width, block_W),
                      T.ceildiv(out_height, block_H), batch_size, threads=threads) as (oc_b, w_b, h_b, n):
            # Allocate shared memory for input tile and weights tile
            shared_X = T.alloc_shared((block_H + kernel_size - 1, block_W + kernel_size - 1, in_channels), dtype)
            shared_W = T.alloc_shared((block_OC, in_channels, kernel_size, kernel_size), dtype)
            shared_B = T.alloc_shared((block_OC,), dtype)

            # Load weight tile
            for ko in T.Parallel(block_OC):
                oc = oc_b * block_OC + ko
                if oc < out_channels:
                    shared_B[ko] = B[oc]
                    for ic in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                shared_W[ko, ic, kh, kw] = W[oc, ic, kh, kw]

            # Compute output for this tile
            for ko in T.Parallel(block_OC):
                oc = oc_b * block_OC + ko
                if oc < out_channels:
                    sum_val = T.alloc_fragment((1,), "float32", 0.0)
                    for ic in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                for hi in T.unroll(block_H):
                                    h = h_b * block_H + hi
                                    if h < out_height:
                                        for wi in T.unroll(block_W):
                                            w = w_b * block_W + wi
                                            if w < out_width:
                                                ih = h + kh - pad
                                                iw = w + kw - pad
                                                if 0 <= ih < height and 0 <= iw < width:
                                                    val = X[n, ic, ih, iw] * shared_W[ko, ic, kh, kw]
                                                    sum_val[0] += T.cast(val, "float32")

                    # Apply bias
                    sum_val[0] += T.cast(shared_B[ko], "float32")
                    # Apply GELU
                    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    x_fp32 = sum_val[0]
                    x_cubed = x_fp32 * x_fp32 * x_fp32
                    tanh_arg = T.cast(0.7978845608, "float32") * (x_fp32 + T.cast(0.044715, "float32") * x_cubed)
                    # tanh approximation
                    tanh_val = tanh_arg
                    tanh2 = tanh_val * tanh_val
                    tanh4 = tanh2 * tanh2
                    tanh6 = tanh4 * tanh2
                    tanh_approx = tanh_val * (T.cast(1.0, "float32") + tanh2 * (T.cast(0.13333334, "float32") + tanh2 * T.cast(0.053968254, "float32")))
                    gelu_val = T.cast(0.5, "float32") * x_fp32 * (T.cast(1.0, "float32") + tanh_approx)

                    # Global average pooling: accumulate into Out
                    # We do reduction across spatial dimensions here
                    # We'll accumulate partial sums and divide at the end
                    if h_b == 0 and w_b == 0:
                        Out[n, oc] = T.cast(gelu_val, dtype)
                    else:
                        Out[n, oc] += T.cast(gelu_val, dtype)

    return tilelang.compile(kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self._kernel_cache = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def _get_kernel(self, batch_size: int, height: int, width: int):
        key = (batch_size, height, width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_gelu_gavg_kernel(
                batch_size, self.in_channels, self.out_channels, height, width, self.kernel_size
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x = x.contiguous()
        # Ensure weights and bias are in fp16
        w = self.conv.weight.to(torch.float16)
        b = self.conv.bias.to(torch.float16)
        kernel = self._get_kernel(batch_size, height, width)
        out = kernel(x, w, b)
        # Final division for global average pooling
        out = out / (height * width)
        return out