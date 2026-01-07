import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gelu_scale_kernel(gX: cute.Tensor, gY: cute.Tensor, scaling_factor: float): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx
    total_elems = gX.size()

    if thread_idx >= total_elems:
        return

    x = gX.flat[thread_idx]
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = 0.7978845608028654  # sqrt(2/pi)
    x_cubed = x * x * x
    tanh_arg = sqrt_2_pi * (x + 0.044715 * x_cubed)
    tanh_val = (cute.exp(2 * tanh_arg) - 1) / (cute.exp(2 * tanh_arg) + 1)  # tanh approximation
    gelu_val = 0.5 * x * (1 + tanh_val)
    gY.flat[thread_idx] = gelu_val * scaling_factor

@cute.jit
def gelu_scale_host(mX: cute.Tensor, mY: cute.Tensor, scaling_factor: float):
    total_elems = mX.size()
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    gelu_scale_kernel(mX, mY, scaling_factor).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, layer normalization, fused GELU activation and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        self.compiled = {}

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv_transpose(x)
        x = self.layer_norm(x)
        # Fused GELU and scaling
        x = x.contiguous().cuda()
        y = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(gelu_scale_host, mX, mY, self.scaling_factor)
            self.compiled[key] = compiled
        compiled(mX, mY, self.scaling_factor)
        return y