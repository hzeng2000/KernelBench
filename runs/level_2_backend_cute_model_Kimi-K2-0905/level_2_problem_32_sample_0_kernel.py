import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_scale_min_kernel(
    gInput: cute.Tensor,
    gWeight: cute.Tensor,
    gBias: cute.Tensor,
    gOutput: cute.Tensor,
    scale: float,
    batch_size: int,
    in_c: int,
    out_c: int,
    h: int,
    w: int,
    k: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_x = bidy * bdimy + tidy
    out_y = bidz * bdimz + tidz

    if out_x < h and out_y < w:
        min_val = float('inf')
        for oc in range(out_c):
            acc = 0.0
            for ic in range(in_c):
                for kh in range(k):
                    for kw in range(k):
                        in_h = out_x + kh - k // 2
                        in_w = out_y + kw - k // 2
                        if 0 <= in_h < h and 0 <= in_w < w:
                            inp = gInput[bidx, ic, in_h, in_w]
                            wgt = gWeight[oc, ic, kh, kw]
                            acc += inp * wgt
            acc += gBias[oc]
            acc *= scale
            if acc < min_val:
                min_val = acc
        gOutput[bidx, 0, out_x, out_y] = min_val

@cute.jit
def conv_scale_min_host(
    mInput: cute.Tensor,
    mWeight: cute.Tensor,
    mBias: cute.Tensor,
    mOutput: cute.Tensor,
    scale: float,
    batch_size: int,
    in_c: int,
    out_c: int,
    h: int,
    w: int,
    k: int
):
    threads_per_block = 16
    grid_x = batch_size
    grid_y = cute.ceil_div(h, threads_per_block)
    grid_z = cute.ceil_div(w, threads_per_block)

    conv_scale_min_kernel(
        mInput, mWeight, mBias, mOutput, scale,
        batch_size, in_c, out_c, h, w, k
    ).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(1, threads_per_block, threads_per_block)
    )


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_c, h, w = x.shape
        x = x.contiguous().cuda()
        
        output = torch.empty((batch_size, 1, h, w), dtype=x.dtype, device=x.device)
        
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mWeight = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_scale_min_host,
                mInput, mWeight, mBias, mOutput,
                self.scale_factor,
                batch_size, in_c, self.out_channels, h, w, self.kernel_size
            )
            self.compiled[key] = compiled
        
        compiled(
            mInput, mWeight, mBias, mOutput,
            self.scale_factor,
            batch_size, in_c, self.out_channels, h, w, self.kernel_size
        )
        return output