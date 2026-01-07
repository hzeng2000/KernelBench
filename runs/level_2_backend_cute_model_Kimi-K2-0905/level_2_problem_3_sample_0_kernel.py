import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose3d_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    out_pad_d: int, out_pad_h: int, out_pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz
    
    if out_x < out_w and out_y < out_h and out_z < out_d:
        for n in range(batch_size):
            for oc in range(out_channels):
                acc = 0.0
                for ic in range(in_channels):
                    for kd in range(kernel_d):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                in_z = (out_z + pad_d - kd * 1) // stride_d
                                in_y = (out_y + pad_h - kh * 1) // stride_h
                                in_x = (out_x + pad_w - kw * 1) // stride_w
                                
                                if (out_z + pad_d - kd) % stride_d == 0 and \
                                   (out_y + pad_h - kh) % stride_h == 0 and \
                                   (out_x + pad_w - kw) % stride_w == 0 and \
                                   in_z >= 0 and in_z < in_d and \
                                   in_y >= 0 and in_y < in_h and \
                                   in_x >= 0 and in_x < in_w:
                                    acc += gInput[n, ic, in_z, in_y, in_x] * gWeight[ic, oc, kd, kh, kw]
                gOutput[n, oc, out_z, out_y, out_x] = acc

@cute.kernel
def add_bias_kernel(gX: cute.Tensor, bias: float, gOut: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    
    idx = bidx * bdim + tidx
    total = gX.shape[0] * gX.shape[1] * gX.shape[2] * gX.shape[3] * gX.shape[4]
    
    if idx < total:
        gOut.data[idx] = gX.data[idx] + bias

@cute.kernel
def layer_norm_kernel(gX: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor, gOut: cute.Tensor, eps: float):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    
    n = gX.shape[0]
    c = gX.shape[1]
    d = gX.shape[2]
    h = gX.shape[3]
    w = gX.shape[4]
    
    for i in range(bidx * bdim + tidx, n * d * h * w, bdim * cute.arch.grid_dim().x):
        ni = i // (d * h * w)
        rem = i % (d * h * w)
        di = rem // (h * w)
        rem = rem % (h * w)
        hi = rem // w
        wi = rem % w
        
        mean = gMean[ni, di, hi, wi]
        var = gVar[ni, di, hi, wi]
        
        for ci in range(c):
            val = gX[ni, ci, di, hi, wi]
            gOut[ni, ci, di, hi, wi] = (val - mean) / cute.sqrt(var + eps)

@cute.kernel
def avg_pool3d_kernel(gX: cute.Tensor, gOut: cute.Tensor, kernel_d: int, kernel_h: int, kernel_w: int):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    
    n = gOut.shape[0]
    c = gOut.shape[1]
    out_d = gOut.shape[2]
    out_h = gOut.shape[3]
    out_w = gOut.shape[4]
    in_d = gX.shape[2]
    in_h = gX.shape[3]
    in_w = gX.shape[4]
    
    for i in range(bidx * bdim + tidx, n * c * out_d * out_h * out_w, bdim * cute.arch.grid_dim().x):
        ni = i // (c * out_d * out_h * out_w)
        rem = i % (c * out_d * out_h * out_w)
        ci = rem // (out_d * out_h * out_w)
        rem = rem % (out_d * out_h * out_w)
        od = rem // (out_h * out_w)
        rem = rem % (out_h * out_w)
        oh = rem // out_w
        ow = rem % out_w
        
        sum_val = 0.0
        count = 0
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    id_ = od * kernel_d + kd
                    ih = oh * kernel_h + kh
                    iw = ow * kernel_w + kw
                    if id_ < in_d and ih < in_h and iw < in_w:
                        sum_val += gX[ni, ci, id_, ih, iw]
                        count += 1
        gOut[ni, ci, od, oh, ow] = sum_val / count

@cute.kernel
def gelu_kernel(gX: cute.Tensor, gOut: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    
    idx = bidx * bdim + tidx
    total = gX.shape[0] * gX.shape[1] * gX.shape[2] * gX.shape[3] * gX.shape[4]
    
    if idx < total:
        val = gX.data[idx]
        gOut.data[idx] = 0.5 * val * (1.0 + cute.tanh(0.7978845608 * (val + 0.044715 * val * val * val)))

@cute.jit
def fused_conv_transpose_add_norm_pool_gelu(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: float, mOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    out_pad_d: int, out_pad_h: int, out_pad_w: int,
    pool_kernel_d: int, pool_kernel_h: int, pool_kernel_w: int
):
    threads = 256
    blocks = (out_w * out_h * out_d + threads - 1) // threads
    
    temp1 = cute.empty((batch_size, out_channels, out_d, out_h, out_w), dtype=mInput.dtype)
    conv_transpose3d_kernel(mInput, mWeight, temp1,
                           batch_size, in_channels, out_channels,
                           in_d, in_h, in_w, out_d, out_h, out_w,
                           kernel_d, kernel_h, kernel_w,
                           stride_d, stride_h, stride_w,
                           pad_d, pad_h, pad_w,
                           out_pad_d, out_pad_h, out_pad_w).launch(
        grid=(blocks, 1, 1), block=(threads, 1, 1))
    
    temp2 = cute.empty_like(temp1)
    add_bias_kernel(temp1, mBias, temp2).launch(
        grid=(blocks, 1, 1), block=(threads, 1, 1))
    
    mean = cute.empty((batch_size, out_d, out_h, out_w))
    var = cute.empty((batch_size, out_d, out_h, out_w))
    
    for i in range(batch_size * out_d * out_h * out_w):
        ni = i // (out_d * out_h * out_w)
        rem = i % (out_d * out_h * out_w)
        di = rem // (out_h * out_w)
        rem = rem % (out_h * out_w)
        hi = rem // out_w
        wi = rem % out_w
        
        sum_val = 0.0
        for ci in range(out_channels):
            sum_val += temp2[ni, ci, di, hi, wi]
        mean[ni, di, hi, wi] = sum_val / out_channels
        
        sum_sq = 0.0
        for ci in range(out_channels):
            diff = temp2[ni, ci, di, hi, wi] - mean[ni, di, hi, wi]
            sum_sq += diff * diff
        var[ni, di, hi, wi] = sum_sq / out_channels
    
    temp3 = cute.empty_like(temp2)
    layer_norm_kernel(temp2, mean, var, temp3, 1e-5).launch(
        grid=(blocks, 1, 1), block=(threads, 1, 1))
    
    pool_out_d = out_d // pool_kernel_d
    pool_out_h = out_h // pool_kernel_h
    pool_out_w = out_w // pool_kernel_w
    temp4 = cute.empty((batch_size, out_channels, pool_out_d, pool_out_h, pool_out_w))
    
    pool_blocks = (batch_size * out_channels * pool_out_d * pool_out_h * pool_out_w + threads - 1) // threads
    avg_pool3d_kernel(temp3, temp4, pool_kernel_d, pool_kernel_h, pool_kernel_w).launch(
        grid=(pool_blocks, 1, 1), block=(threads, 1, 1))
    
    gelu_kernel(temp4, mOutput).launch(
        grid=(pool_blocks, 1, 1), block=(threads, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm_shape = norm_shape
        self.pool_kernel_size = pool_kernel_size
        self.compiled = {}
        
    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight
        bias = self.sum_weight.item()
        
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        out_channels = weight.shape[1]
        kernel_d, kernel_h, kernel_w = self.conv_transpose.kernel_size
        stride_d, stride_h, stride_w = self.conv_transpose.stride
        pad_d, pad_h, pad_w = self.conv_transpose.padding
        out_pad_d, out_pad_h, out_pad_w = self.conv_transpose.output_padding
        
        out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d
        out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h
        out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w
        
        pool_kernel_d, pool_kernel_h, pool_kernel_w = self.pool_kernel_size
        pool_out_d = out_d // pool_kernel_d
        pool_out_h = out_h // pool_kernel_h
        pool_out_w = out_w // pool_kernel_w
        
        output = torch.empty((batch_size, out_channels, pool_out_d, pool_out_h, pool_out_w), dtype=x.dtype, device=x.device)
        
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_conv_transpose_add_norm_pool_gelu, 
                                   mInput, mWeight, bias, mOutput,
                                   batch_size, in_channels, out_channels,
                                   in_d, in_h, in_w, out_d, out_h, out_w,
                                   kernel_d, kernel_h, kernel_w,
                                   stride_d, stride_h, stride_w,
                                   pad_d, pad_h, pad_w,
                                   out_pad_d, out_pad_h, out_pad_w,
                                   pool_kernel_d, pool_kernel_h, pool_kernel_w)
            self.compiled[key] = compiled
        
        compiled(mInput, mWeight, bias, mOutput,
                batch_size, in_channels, out_channels,
                in_d, in_h, in_w, out_d, out_h, out_w,
                kernel_d, kernel_h, kernel_w,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                out_pad_d, out_pad_h, out_pad_w,
                pool_kernel_d, pool_kernel_h, pool_kernel_w)
        
        return output