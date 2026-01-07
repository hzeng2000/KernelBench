import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_transpose_conv_bias_clamp_kernel(
    gInput: cute.Tensor,
    gWeight: cute.Tensor,
    gBias: cute.Tensor,
    gOutput: cute.Tensor,
    batch_size: int,
    in_h: int,
    in_w: int,
    in_c: int,
    out_h: int,
    out_w: int,
    out_c: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    out_pad_h: int,
    out_pad_w: int,
    scale: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    out_n = bidz
    out_c_idx = bidy * bdimy + tidy
    hw_idx = bidx * bdimx + tidx
    
    if out_n >= batch_size or out_c_idx >= out_c or hw_idx >= out_h * out_w:
        return
    
    out_h_idx = hw_idx // out_w
    out_w_idx = hw_idx % out_w
    
    acc = 0.0
    
    for in_c_idx in range(in_c):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                in_h_idx = out_h_idx * stride_h - pad_h + kh
                in_w_idx = out_w_idx * stride_w - pad_w + kw
                
                if in_h_idx >= 0 and in_h_idx < in_h and in_w_idx >= 0 and in_w_idx < in_w:
                    inp_val = gInput[out_n, in_c_idx, in_h_idx, in_w_idx]
                    w_val = gWeight[out_c_idx, in_c_idx, kernel_h - 1 - kh, kernel_w - 1 - kw]
                    acc += inp_val * w_val
    
    acc += gBias[out_c_idx, 0, 0]
    
    acc = cute.min(cute.max(acc, 0.0), 1.0)
    acc = acc * scale
    acc = cute.min(cute.max(acc, 0.0), 1.0)
    acc = acc / scale
    
    gOutput[out_n, out_c_idx, out_h_idx, out_w_idx] = acc

@cute.jit
def fused_transpose_conv_bias_clamp_host(
    mInput: cute.Tensor,
    mWeight: cute.Tensor,
    mBias: cute.Tensor,
    mOutput: cute.Tensor,
    batch_size: int,
    in_h: int,
    in_w: int,
    in_c: int,
    out_h: int,
    out_w: int,
    out_c: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    out_pad_h: int,
    out_pad_w: int,
    scale: float
):
    threads_per_block = 256
    elems_per_block = threads_per_block
    
    total_hw = out_h * out_w
    total_c = out_c
    
    grid_x = cute.ceil_div(total_hw, elems_per_block)
    grid_y = cute.ceil_div(total_c, 1)
    grid_z = batch_size
    
    fused_transpose_conv_bias_clamp_kernel(
        mInput, mWeight, mBias, mOutput,
        batch_size, in_h, in_w, in_c, out_h, out_w, out_c,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, scale
    ).launch(grid=(grid_x, grid_y, grid_z), block=(elems_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.compiled = {}

    def forward(self, x):
        batch_size, in_c, in_h, in_w = x.shape
        out_h = (in_h - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_w = (in_w - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        output = torch.empty(batch_size, self.out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
        
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_transpose_conv_bias_clamp_host,
                mInput, mWeight, mBias, mOutput,
                batch_size, in_h, in_w, in_c, out_h, out_w, self.out_channels,
                self.kernel_size, self.kernel_size, self.stride, self.stride,
                self.padding, self.padding, self.output_padding, self.output_padding,
                self.scaling_factor
            )
            self.compiled[key] = compiled
        
        compiled(mInput, mWeight, mBias, mOutput,
                 batch_size, in_h, in_w, in_c, out_h, out_w, self.out_channels,
                 self.kernel_size, self.kernel_size, self.stride, self.stride,
                 self.padding, self.padding, self.output_padding, self.output_padding,
                 self.scaling_factor)
        
        return output