import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def build_fused_conv3d_activation_kernel(
    batch_size: int,
    out_channels: int,
    out_depth: int,
    out_height: int,
    out_width: int,
    in_channels: int,
    kernel_size: int,
    block_C: int = 8,
    block_D: int = 4,
    block_H: int = 8,
    block_W: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    pad_d = (kernel_size - 1) // 2
    pad_h = (kernel_size - 1) // 2
    pad_w = (kernel_size - 1) // 2
    
    @T.prim_func
    def fused_conv3d_activation_kernel(
        Input: T.Tensor((batch_size, in_channels, out_depth + 2 * pad_d, out_height + 2 * pad_h, out_width + 2 * pad_w), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels, 1, 1, 1), dtype),
        Output: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_W),
            T.ceildiv(out_height, block_H),
            T.ceildiv(out_depth, block_D),
            T.ceildiv(out_channels, block_C),
            batch_size,
            threads=threads
        ) as (bx, by, bz, bc, bb):
            start_w = bx * block_W
            start_h = by * block_H
            start_d = bz * block_D
            start_c = bc * block_C
            start_b = bb

            for local_c, local_d, local_h, local_w in T.Parallel(block_C, block_D, block_H, block_W):
                c = start_c + local_c
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w

                if c < out_channels and d < out_depth and h < out_height and w < out_width:
                    acc = T.cast(0.0, dtype)
                    for ic in range(in_channels):
                        for kd in range(kernel_size):
                            for kh in range(kernel_size):
                                for kw in range(kernel_size):
                                    in_d = d + kd - pad_d
                                    in_h = h + kh - pad_h
                                    in_w = w + kw - pad_w
                                    if 0 <= in_d < out_depth + 2 * pad_d and 0 <= in_h < out_height + 2 * pad_h and 0 <= in_w < out_width + 2 * pad_w:
                                        acc += Input[start_b, ic, in_d, in_h, in_w] * Weight[c, ic, kd, kh, kw]
                    
                    # Apply ReLU
                    relu_out = T.max(acc, T.cast(0.0, dtype))
                    # Apply LeakyReLU
                    leaky_out = T.where(relu_out > T.cast(0.0, dtype), relu_out, relu_out * T.cast(0.01, dtype))
                    # Apply GELU approximation
                    gelu_out = leaky_out * T.cast(0.5, dtype) * (T.cast(1.0, dtype) + T.tanh(T.cast(0.7978845608, dtype) * (leaky_out + T.cast(0.044715, dtype) * leaky_out * leaky_out * leaky_out)))
                    # Apply Sigmoid
                    sig_out = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-gelu_out))
                    # Add bias
                    Output[start_b, c, d, h, w] = sig_out + Bias[c, 0, 0, 0]

    return tilelang.compile(fused_conv3d_activation_kernel, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def _get_kernel(self, batch_size: int, out_depth: int, out_height: int, out_width: int):
        key = (batch_size, out_depth, out_height, out_width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv3d_activation_kernel(
                batch_size, self.out_channels, out_depth, out_height, out_width,
                self.in_channels, self.kernel_size, dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch_size, _, depth, height, width = x.shape
        out_depth = depth
        out_height = height
        out_width = width
        
        kernel = self._get_kernel(batch_size, out_depth, out_height, out_width)
        
        weight_fp16 = self.conv.weight.to(torch.float16)
        bias_fp16 = self.bias.to(torch.float16)
        x_fp16 = x.to(torch.float16)
        
        output = kernel(x_fp16, weight_fp16, bias_fp16)
        
        return output.to(torch.float32)