import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv_multiply_activation_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    block_size: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1
    
    @T.prim_func
    def fused_kernel(
        Input: T.Tensor((batch_size, in_channels, height, width), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        Multiplier: T.Tensor((out_channels, 1, 1), dtype),
        Output: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_size), T.ceildiv(out_height, block_size), batch_size * out_channels, threads=threads) as (bx, by, bz):
            # Thread indices
            tx = T.thread_idx("threadIdx.x")
            ty = T.thread_idx("threadIdx.y")
            
            # Block indices
            b = bz // out_channels
            oc = bz % out_channels
            oh_start = by * block_size
            ow_start = bx * block_size
            
            # Shared memory for input tile
            shared_input = T.alloc_shared((block_size + kernel_size - 1, block_size + kernel_size - 1), dtype)
            # Shared memory for weight tile
            shared_weight = T.alloc_shared((in_channels, kernel_size, kernel_size), dtype)
            
            # Load weights to shared memory
            for ic in T.Parallel(in_channels):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        shared_weight[ic, kh, kw] = Weight[oc, ic, kh, kw]
            
            # Process output tile
            for oh in range(oh_start, T.min(oh_start + block_size, out_height)):
                for ow in range(ow_start, T.min(ow_start + block_size, out_width)):
                    # Load input tile to shared memory
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            ih = oh + kh
                            iw = ow + kw
                            if ih < height and iw < width:
                                for ic in T.Parallel(in_channels):
                                    shared_input[kh, kw] = Input[b, ic, ih, iw]
                    
                    # Compute convolution
                    acc = T.alloc_fragment((1,), dtype, 0)
                    for ic in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                ih = oh + kh
                                iw = ow + kw
                                if ih < height and iw < width:
                                    acc[0] += shared_weight[ic, kh, kw] * shared_input[kh, kw]
                    
                    # Add bias
                    acc[0] += Bias[oc]
                    
                    # Multiply by learnable scalar
                    acc[0] *= Multiplier[oc, 0, 0]
                    
                    # Apply LeakyReLU
                    leaky_relu_slope = T.float16(0.01)
                    acc[0] = T.max(acc[0], acc[0] * leaky_relu_slope)
                    
                    # Apply GELU approximation
                    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    pi = T.float16(3.14159265359)
                    sqrt_2_over_pi = T.sqrt(T.float16(2.0) / pi)
                    x_cubed = acc[0] * acc[0] * acc[0]
                    tanh_arg = sqrt_2_over_pi * (acc[0] + T.float16(0.044715) * x_cubed)
                    tanh_result = T.tanh(tanh_arg)
                    gelu_result = T.float16(0.5) * acc[0] * (T.float16(1.0) + tanh_result)
                    
                    Output[b, oc, oh, ow] = gelu_result

    return tilelang.compile(fused_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self._kernel_cache = {}
        
    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int):
        key = (batch_size, in_channels, out_channels, height, width, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_multiply_activation_kernel(
                batch_size, in_channels, out_channels, height, width, kernel_size
            )
        return self._kernel_cache[key]

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        
        # Convert to fp16
        x_fp16 = x.half()
        weight_fp16 = self.conv.weight.half()
        bias_fp16 = self.conv.bias.half()
        multiplier_fp16 = self.multiplier.half()
        
        # Get kernel
        kernel = self._get_kernel(batch_size, in_channels, out_channels, height, width, kernel_size)
        
        # Allocate output tensor
        out_height = height - kernel_size + 1
        out_width = width - kernel_size + 1
        output = torch.empty(batch_size, out_channels, out_height, out_width, dtype=torch.float16, device=x.device)
        
        # Run fused kernel
        kernel(x_fp16, weight_fp16, bias_fp16, multiplier_fp16, output)
        
        return output.float()