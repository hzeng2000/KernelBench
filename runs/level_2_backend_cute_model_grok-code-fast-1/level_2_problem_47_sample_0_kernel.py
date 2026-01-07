import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def mish_tanh_kernel(gA: cute.Tensor, gC: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    N, K, OD, OH, OW = gA.shape
    total_elems = N * K * OD * OH * OW

    if thread_idx >= total_elems:
        return

    n = thread_idx // (K * OD * OH * OW)
    rem = thread_idx % (K * OD * OH * OW)
    k = rem // (OD * OH * OW)
    rem = rem % (OD * OH * OW)
    od = rem // (OH * OW)
    rem = rem % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    val = gA[n, k, od, oh, ow]
    softplus = cute.log(1.0 + cute.exp(val))
    mish = val * cute.tanh(softplus)
    gC[n, k, od, oh, ow] = cute.tanh(mish)

@cute.jit
def mish_tanh_host(mA: cute.Tensor, mC: cute.Tensor):
    N, K, OD, OH, OW = mA.shape
    total_elems = N * K * OD * OH * OW

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    mish_tanh_kernel(mA, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        N, K, OD, OH, OW = x.shape
        x = x.contiguous().cuda()
        C = torch.empty_like(x)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(mish_tanh_host, mA, mC)
            self.compiled[key] = compiled

        compiled(mA, mC)
        return C