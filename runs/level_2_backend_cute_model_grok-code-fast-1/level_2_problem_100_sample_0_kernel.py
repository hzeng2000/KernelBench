import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def clamp_divide_kernel(gA: cute.Tensor, gC: cute.Tensor, min_value: float, divisor: float): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    shape = gA.shape
    total_elems = 1
    for s in shape:
        total_elems *= s

    if thread_idx >= total_elems:
        return

    # Flatten index to 1D
    idx = thread_idx
    coords = []
    for i in range(len(shape)):
        coords.append(idx % shape[i])
        idx //= shape[i]

    a_val = gA[tuple(coords)]
    c_val = max(a_val, min_value) / divisor
    gC[tuple(coords)] = c_val

@cute.jit
def clamp_divide_host(mA: cute.Tensor, mC: cute.Tensor, min_value: float, divisor: float):
    shape = mA.shape
    total_elems = 1
    for s in shape:
        total_elems *= s

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    clamp_divide_kernel(mA, mC, min_value, divisor).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant, with clamp and divide fused into a custom CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused clamp and divide
        x = x.contiguous().cuda()
        C = torch.empty_like(x)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(clamp_divide_host, mA, mC, self.min_value, self.divisor)
            self.compiled[key] = compiled

        compiled(mA, mC, self.min_value, self.divisor)
        return C