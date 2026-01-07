import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv_transpose_kernel(
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
    block_N: int = 8,
    block_K: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    # Compute output dimensions
    weight_shape = (in_channels, out_channels, kernel_size, kernel_size)
    
    @T.prim_func
    def fused_conv_transpose_kernel(
        X: T.Tensor((batch_size, in_channels, in_height, in_width), dtype),
        W: T.Tensor(weight_shape, dtype),
        bias: T.Tensor((out_channels, 1, 1), dtype),
        scaling_factor: T.Tensor((1,), dtype),
        Y: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        # Cache scaling factor value
        scale_val = scaling_factor[0]
        
        with T.Kernel(
            T.ceildiv(out_width, block_N),
            T.ceildiv(out_height, block_M),
            batch_size * out_channels,
            threads=threads
        ) as (bx, by, bz):
            # Calculate output position
            out_x = bx * block_N + T.thread_binding(0, block_N, thread="threadIdx.x")
            out_y = by * block_M + T.thread_binding(0, block_M, thread="threadIdx.y")
            batch_idx = bz // out_channels
            out_c = bz % out_channels
            
            if out_x < out_width and out_y < out_height and batch_idx < batch_size:
                # Compute corresponding input position for transposed conv
                in_x_origin = out_x - output_padding + padding
                in_y_origin = out_y - output_padding + padding
                
                acc = T.alloc_fragment((1,), dtype, scope="local")
                acc[0] = T.cast(0.0, dtype)
                
                # Perform convolution
                for in_c in T.serial(in_channels):
                    for ky in T.serial(kernel_size):
                        for kx in T.serial(kernel_size):
                            # Calculate input position
                            in_x = (in_x_origin - kx) // stride
                            in_y = (in_y_origin - ky) // stride
                            
                            # Check bounds
                            if (in_x >= 0 and in_x < in_width and 
                                in_y >= 0 and in_y < in_height and
                                (in_x_origin - kx) % stride == 0 and
                                (in_y_origin - ky) % stride == 0):
                                
                                weight_val = W[in_c, out_c, ky, kx]
                                input_val = X[batch_idx, in_c, in_y, in_x]
                                acc[0] += input_val * weight_val
                
                # Add bias
                acc[0] += bias[out_c, 0, 0]
                
                # Apply clamping, scaling, clamping, and division
                # Clamp to [0, 1]
                clamped1 = T.max(T.min(acc[0], T.cast(1.0, dtype)), T.cast(0.0, dtype))
                # Scale
                scaled = clamped1 * scale_val
                # Clamp again
                clamped2 = T.max(T.min(scaled, T.cast(scale_val, dtype)), T.cast(0.0, dtype))
                # Divide
                result = clamped2 / scale_val
                
                Y[batch_idx, out_c, out_y, out_x] = result

    return tilelang.compile(fused_conv_transpose_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}
        
    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, 
                   in_height: int, in_width: int, out_height: int, out_width: int,
                   kernel_size: int, stride: int, padding: int, output_padding: int):
        key = (batch_size, in_channels, out_channels, in_height, in_width, out_height, out_width,
               kernel_size, stride, padding, output_padding)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_transpose_kernel(
                batch_size, in_channels, out_channels, in_height, in_width, out_height, out_width,
                kernel_size, stride, padding, output_padding
            )
        return self._kernel_cache[key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, in_height, in_width = x.shape
        out_channels = self.conv_transpose.out_channels
        kernel_size = self.conv_transpose.kernel_size[0]
        stride = self.conv_transpose.stride[0]
        padding = self.conv_transpose.padding[0]
        output_padding = self.conv_transpose.output_padding[0]
        
        # Calculate output dimensions
        out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding
        out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding
        
        # Get kernel
        kernel = self._get_kernel(batch_size, in_channels, out_channels, 
                                 in_height, in_width, out_height, out_width,
                                 kernel_size, stride, padding, output_padding)
        
        # Convert to fp16
        x_fp16 = x.half()
        weight_fp16 = self.conv_transpose.weight.half()
        bias_fp16 = self.bias.half()
        scale_fp16 = torch.tensor([self.scaling_factor], dtype=torch.float16, device=x.device)
        
        # Run kernel
        output = kernel(x_fp16, weight_fp16, bias_fp16, scale_fp16)
        
        return output