import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def elementwise_mul_kernel(gA: cute.Tensor, scale: float, gC: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    thread_idx = bidx * bdim + tidx
    if thread_idx < gA.size():
        gC[thread_idx] = gA[thread_idx] * scale

@cute.jit
def elementwise_mul_host(mA: cute.Tensor, scale: float, mC: cute.Tensor):
    total = mA.size()
    threads = 256
    grid_x = cute.ceil_div(total, threads)
    elementwise_mul_kernel(mA, scale, mC).launch(grid=(grid_x, 1, 1), block=(threads, 1, 1))

@cute.kernel
def fused_add_bias_mul_kernel(gA: cute.Tensor, gBias: cute.Tensor, scale2: float, gC: cute.Tensor, batch_size: int, out_c: int, d: int, h: int, w: int):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    thread_idx = bidx * bdim + tidx
    total = batch_size * out_c * d * h * w
    if thread_idx < total:
        total_per_batch = out_c * d * h * w
        batch_idx = thread_idx // total_per_batch
        rem = thread_idx % total_per_batch
        c_idx = rem // (d * h * w)
        rem2 = rem % (d * h * w)
        d_idx = rem2 // (h * w)
        h_idx = (rem2 % (h * w)) // w
        w_idx = rem2 % w
        val = gA[batch_idx, c_idx, d_idx, h_idx, w_idx]
        bias_val = gBias[c_idx, 0, 0, 0]
        gC[batch_idx, c_idx, d_idx, h_idx, w_idx] = (val + bias_val) * scale2

@cute.jit
def fused_add_bias_mul_host(mA: cute.Tensor, mBias: cute.Tensor, scale2: float, mC: cute.Tensor, batch_size: int, out_c: int, d: int, h: int, w: int):
    total = batch_size * out_c * d * h * w
    threads = 256
    grid_x = cute.ceil_div(total, threads)
    fused_add_bias_mul_kernel(mA, mBias, scale2, mC, batch_size, out_c, d, h, w).launch(grid=(grid_x, 1, 1), block=(threads, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        # Custom elementwise multiply by scale1
        A = x.contiguous()
        C = torch.empty_like(A)
        mA = from_dlpack(A.view(-1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C.view(-1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        key = (A.dtype,)
        compiled_mul = self.compiled.get('mul', {}).get(key)
        if compiled_mul is None:
            compiled_mul = cute.compile(elementwise_mul_host, mA, self.scale1.item(), mC)
            self.compiled.setdefault('mul', {})[key] = compiled_mul
        compiled_mul(mA, self.scale1.item(), mC)
        x = C.view_as(A)
        x = self.avg_pool(x)
        # Custom fused add bias and multiply by scale2
        A = x.contiguous()
        C = torch.empty_like(A)
        batch_size, out_c, d, h, w = A.shape
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        key2 = (A.dtype, batch_size, out_c, d, h, w)
        compiled_fused = self.compiled.get('fused', {}).get(key2)
        if compiled_fused is None:
            compiled_fused = cute.compile(fused_add_bias_mul_host, mA, mBias, self.scale2.item(), mC, batch_size, out_c, d, h, w)
            self.compiled.setdefault('fused', {})[key2] = compiled_fused
        compiled_fused(mA, mBias, self.scale2.item(), mC, batch_size, out_c, d, h, w)
        x = C
        return x