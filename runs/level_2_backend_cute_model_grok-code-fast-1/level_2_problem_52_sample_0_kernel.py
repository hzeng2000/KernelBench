import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def activation_kernel(gX: cute.Tensor, gY: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    b, c, h, w = gX.shape
    total = b * c * h * w
    if thread_idx >= total:
        return

    bi = thread_idx // (c * h * w)
    rem = thread_idx % (c * h * w)
    ci = rem // (h * w)
    rem2 = rem % (h * w)
    hi = rem2 // w
    wi = rem2 % w

    val = gX[bi, ci, hi, wi]
    soft = cute.math.log(1 + cute.math.exp(val))
    t = cute.math.tanh(soft)
    gY[bi, ci, hi, wi] = val * t

@cute.jit
def activation_host(mX: cute.Tensor, mY: cute.Tensor):
    b, c, h, w = mX.shape
    total = b * c * h * w

    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)

    activation_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = cutlass.ops.Conv2d(in_channels, out_channels, [kernel_size, kernel_size], stride=[1, 1], padding=[kernel_size // 2, kernel_size // 2])
        self.bn = cutlass.ops.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.contiguous()
        y = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(activation_host, mX, mY)
            self.compiled[key] = compiled

        compiled(mX, mY)
        x = y
        x = self.bn(x)
        return x