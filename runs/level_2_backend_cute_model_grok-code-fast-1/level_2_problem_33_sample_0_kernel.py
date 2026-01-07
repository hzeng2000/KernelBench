import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def scale_kernel(gC: cute.Tensor, gScale: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    m, n = gC.shape
    total_elems = m * n
    if thread_idx >= total_elems:
        return

    ni = thread_idx % n  
    mi = thread_idx // n  

    gC[mi, ni] *= gScale[ni]

@cute.jit
def scale_host(mC: cute.Tensor, mScale: cute.Tensor):
    M, N = mC.shape

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    scale_kernel(mC, mScale).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.randn(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.compiled_gemm = None
        self.compiled_scale = {}

    def forward(self, x):
        batch_size, in_f = x.shape
        out_f = self.gemm_weight.shape[0]
        x = x.contiguous().cuda()
        C = torch.empty((batch_size, out_f), dtype=x.dtype, device=x.device)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.gemm_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(self.gemm_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        if self.compiled_gemm is None:
            self.compiled_gemm = cute.compile(cute.ops.gemm, mA, mB, mC, bias=mBias)

        self.compiled_gemm(mA, mB, mC, bias=mBias)

        # now scale
        mScale = from_dlpack(self.scale, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))

        key = (C.dtype,)
        compiled_scale = self.compiled_scale.get(key)
        if compiled_scale is None:
            compiled_scale = cute.compile(scale_host, mC, mScale)
            self.compiled_scale[key] = compiled_scale

        compiled_scale(mC, mScale)

        # now bn
        C = self.bn(C)
        return C