import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def build_fused_transpose_conv_mish_add_hardtanh_scale_kernel(
    batch_size: int, 
    out_channels: int, 
    out_height: int, 
    out_width: int,
    block_M: int = 16,
    block_N: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
        AddVal: T.Tensor((1,), dtype),
        Scale: T.Tensor((1,), dtype),
        Y: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_N), T.ceildiv(out_height, block_M), batch_size * out_channels, threads=threads) as (bx, by, bz):
            start_x = bx * block_N
            start_y = by * block_M
            start_z = bz

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                z = start_z

                if y < out_height and x < out_width:
                    val = X[z, y, x]
                    
                    # Mish activation: x * tanh(softplus(x))
                    softplus = T.log1p(T.exp(val))
                    tanh_sp = T.tanh(softplus)
                    mish_val = val * tanh_sp
                    
                    # Add value
                    add_val = mish_val + AddVal[0]
                    
                    # Hardtanh activation: clamp between -1 and 1
                    hardtanh_val = T.max(T.min(add_val, T.cast(1.0, dtype)), T.cast(-1.0, dtype))
                    
                    # Scale
                    scaled_val = hardtanh_val * Scale[0]
                    
                    Y[z, y, x] = scaled_val

    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.register_buffer('add_value', torch.tensor([add_value], dtype=torch.float16))
        self.register_buffer('scale', torch.tensor([scale], dtype=torch.float16))
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, out_channels: int, out_height: int, out_width: int):
        key = (batch_size, out_channels, out_height, out_width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_transpose_conv_mish_add_hardtanh_scale_kernel(
                batch_size, out_channels, out_height, out_width
            )
        return self._kernel_cache[key]

    def forward(self, x):
        # Perform transposed convolution
        x = self.conv_transpose(x)
        
        # Get output dimensions
        batch_size, channels, height, width = x.shape
        
        # Ensure contiguous and fp16
        x = x.contiguous().half()
        
        # Get or create kernel
        kernel = self._get_kernel(batch_size, channels, height, width)
        
        # Apply fused kernel
        output = kernel(x, self.add_value, self.scale)
        
        return output