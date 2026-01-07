import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def global_avg_add_bias_kernel(gX: cute.Tensor, gBias: cute.Tensor, gZ: cute.Tensor, H: int, W: int):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x

    B = gX.shape[0]
    C = gX.shape[1]
    b_c = bidx
    b = b_c // C
    c = b_c % C

    s_sum = cute.shared_memory(float, (bdim,))
    s_sum[tidx] = 0.0

    total_elems = H * W
    for i in range(tidx, total_elems, bdim):
        hi = i // W
        wi = i % W
        s_sum[tidx] += gX[b, c, hi, wi]
    cute.sync()

    step = bdim // 2
    while step > 0:
        if tidx < step:
            s_sum[tidx] += s_sum[tidx + step]
        cute.sync()
        step //= 2

    if tidx == 0:
        avg = s_sum[0] / total_elems
        gZ[b, c, 0, 0] = avg + gBias[c, 0, 0]

@cute.jit
def global_avg_add_bias_host(mX: cute.Tensor, mBias: cute.Tensor, mZ: cute.Tensor, H: int, W: int):
    B = mX.shape[0]
    C = mX.shape[1]
    threads_per_block = 256
    grid_x = B * C
    global_avg_add_bias_kernel(mX, mBias, mZ, H, W).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def logsumexp_mul_kernel(gZ: cute.Tensor, gResult: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x

    B = gZ.shape[0]
    C = gZ.shape[1]
    b = bidx
    c = tidx

    s_exp = cute.shared_memory(float, (bdim,))
    if c < C:
        s_exp[tidx] = exp(gZ[b, c, 0, 0])
    else:
        s_exp[tidx] = 0.0
    cute.sync()

    step = bdim // 2
    while step > 0:
        if tidx < step:
            s_exp[tidx] += s_exp[tidx + step]
        cute.sync()
        step //= 2

    if tidx == 0:
        gResult[b, 0] = log(s_exp[0]) * 10.0

@cute.jit
def logsumexp_mul_host(mZ: cute.Tensor, mResult: cute.Tensor):
    B = mZ.shape[0]
    threads_per_block = 128
    grid_x = B
    logsumexp_mul_kernel(mZ, mResult).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, H, W = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        Z = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mZ = from_dlpack(Z, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key1 = (x.dtype,)
        compiled1 = self.compiled.get(('avg', key1))
        if compiled1 is None:
            compiled1 = cute.compile(global_avg_add_bias_host, mX, mBias, mZ, H, W)
            self.compiled[('avg', key1)] = compiled1
        compiled1(mX, mBias, mZ)

        result = torch.empty((B, 1), dtype=x.dtype, device=x.device)
        mResult = from_dlpack(result, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key2 = (x.dtype,)
        compiled2 = self.compiled.get(('logsum', key2))
        if compiled2 is None:
            compiled2 = cute.compile(logsumexp_mul_host, mZ, mResult)
            self.compiled[('logsum', key2)] = compiled2
        compiled2(mZ, mResult)
        return result