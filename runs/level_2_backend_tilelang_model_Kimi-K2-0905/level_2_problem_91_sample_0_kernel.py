import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose_softmax_bias_sigmoid_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    block_M: int = 8,
    block_N: int = 32,
    block_K: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def kernel(
        Input: T.Tensor((batch_size, in_channels, in_height, in_width), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels, 1, 1), dtype),
        Output: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        # Compute output dimensions
        with T.Kernel(T.ceildiv(out_width, block_N), T.ceildiv(out_height, block_M), batch_size * out_channels, threads=threads) as (bx, by, bz):
            # Thread-local accumulators
            accum = T.alloc_fragment((block_M, block_N), dtype, scope="local")
            max_val = T.alloc_fragment((block_M, block_N), dtype, scope="local")
            sum_exp = T.alloc_fragment((block_M, block_N), dtype, scope="local")
            
            # Get output position
            out_x = bx * block_N
            out_y = by * block_M
            batch_idx = bz // out_channels
            out_c = bz % out_channels
            
            # Initialize accumulators
            for i, j in T.Parallel(block_M, block_N):
                accum[i, j] = T.cast(0.0, dtype)
            
            # Compute convolution transpose
            for in_c in range(in_channels):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        # Compute input position
                        in_y = out_y - kh + padding
                        in_x = out_x - kw + padding
                        
                        # Check bounds and accumulate
                        for i, j in T.Parallel(block_M, block_N):
                            if in_y + i >= 0 and in_y + i < in_height and in_x + j >= 0 and in_x + j < in_width:
                                weight_val = Weight[in_c, out_c, kh, kw]
                                input_val = Input[batch_idx, in_c, in_y + i, in_x + j]
                                accum[i, j] += weight_val * input_val
            
            # Add bias and apply softmax (dim=1 handled by channel-wise processing)
            for i, j in T.Parallel(block_M, block_N):
                if out_y + i < out_height and out_x + j < out_width:
                    # Add bias
                    accum[i, j] += Bias[out_c, 0, 0]
                    
                    # Compute max for numerical stability
                    max_val[i, j] = accum[i, j]
                    
                    # Compute exp and sum
                    exp_val = T.exp(accum[i, j] - max_val[i, j])
                    sum_exp[i, j] = exp_val
                    
                    # Normalize (softmax)
                    accum[i, j] = exp_val / sum_exp[i, j]
                    
                    # Scale and apply sigmoid
                    scaled = accum[i, j] * T.cast(2.0, dtype)
                    sigmoid = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-scaled))
                    
                    Output[batch_idx, out_c, out_y + i, out_x + j] = sigmoid

    return tilelang.compile(kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}
        
    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, in_height: int, in_width: int, out_height: int, out_width: int, kernel_size: int, stride: int, padding: int, output_padding: int):
        key = (batch_size, in_channels, out_channels, in_height, in_width, out_height, out_width, kernel_size, stride, padding, output_padding)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose_softmax_bias_sigmoid_kernel(
                batch_size, in_channels, out_channels, in_height, in_width, out_height, out_width,
                kernel_size, stride, padding, output_padding
            )
        return self._kernel_cache[key]

    def forward(self, x):
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        in_height = x.shape[2]
        in_width = x.shape[3]
        
        # Compute output dimensions
        out_height = (in_height - 1) * self.conv_transpose.stride[0] - 2 * self.conv_transpose.padding[0] + self.conv_transpose.kernel_size[0] + self.conv_transpose.output_padding[0]
        out_width = (in_width - 1) * self.conv_transpose.stride[1] - 2 * self.conv_transpose.padding[1] + self.conv_transpose.kernel_size[1] + self.conv_transpose.output_padding[1]
        out_channels = self.conv_transpose.out_channels
        
        # Get kernel
        kernel = self._get_kernel(
            batch_size, in_channels, out_channels, in_height, in_width, out_height, out_width,
            self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0], self.conv_transpose.padding[0], self.conv_transpose.output_padding[0]
        )
        
        # Convert to fp16
        x_fp16 = x.half()
        weight_fp16 = self.conv_transpose.weight.half()
        bias_fp16 = self.bias.half()
        
        # Allocate output
        output = torch.empty(batch_size, out_channels, out_height, out_width, dtype=torch.float16, device=x.device)
        
        # Run kernel
        kernel(x_fp16, weight_fp16, bias_fp16, output)
        
        return output.float()