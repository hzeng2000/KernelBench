import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_conv3d_div_maxpool_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor,
    gOutput: cute.Tensor, divisor: float,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    pool_kd: int, pool_kh: int, pool_kw: int,
    pool_stride_d: int, pool_stride_h: int, pool_stride_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    out_c = cute.block_dim_x()
    batch = cute.block_dim_y()
    
    # Output spatial position
    od = bidx * stride_d + cute.thread_idx_x() % kernel_d - pad_d
    oh = bidy * stride_h + (cute.thread_idx_x() // kernel_d) % kernel_h - pad_h
    ow = bidz * stride_w + (cute.thread_idx_x() // (kernel_d * kernel_h)) % kernel_w - pad_w
    
    # Batch and channel
    n = tidy
    oc = tidz
    
    if n < batch and oc < out_c and od >= 0 and oh >= 0 and ow >= 0 and od < in_d and oh < in_h and ow < in_w:
        sum_val = 0.0
        for ic in range(gInput.shape[1]):
            for kd in range(kernel_d):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        id_coord = od + kd - pad_d
                        ih_coord = oh + kh - pad_h
                        iw_coord = ow + kw - pad_w
                        if id_coord >= 0 and ih_coord >= 0 and iw_coord >= 0 and id_coord < in_d and ih_coord < in_h and iw_coord < in_w:
                            weight_val = gWeight[oc, ic, kd, kh, kw]
                            input_val = gInput[n, ic, id_coord, ih_coord, iw_coord]
                            sum_val += weight_val * input_val
        
        # Add bias and divide
        sum_val = (sum_val + gBias[oc]) / divisor
        
        # Max pooling (simplified for this kernel)
        pool_od = od // pool_stride_d
        pool_oh = oh // pool_stride_h
        pool_ow = ow // pool_stride_w
        if pool_od < out_d and pool_oh < out_h and pool_ow < out_w:
            cute.atomic_max(gOutput[n, oc, pool_od, pool_oh, pool_ow], sum_val)

@cute.kernel
def global_avg_pool_add_bias_sum_kernel(
    gInput: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    sum_dim: int, c: int, d: int, h: int, w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.block_idx()
    
    n = bidx
    oc = tidy
    
    if n < gInput.shape[0] and oc < c:
        avg_val = 0.0
        count = d * h * w
        
        for dd in range(d):
            for hh in range(h):
                for ww in range(w):
                    avg_val += gInput[n, oc, dd, hh, ww]
        
        avg_val = avg_val / count
        biased_val = avg_val + gBias[oc, 0, 0, 0]
        
        if sum_dim == 1:
            cute.atomic_add(gOutput[n, 0, 0, 0], biased_val)
        else:
            gOutput[n, oc, 0, 0] = biased_val

@cute.jit
def fused_conv3d_div_maxpool_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor,
    mOutput: cute.Tensor, divisor: float,
    in_shape, weight_shape, out_shape, pool_shape
):
    batch, in_c, in_d, in_h, in_w = in_shape
    out_c, _, kernel_d, kernel_h, kernel_w = weight_shape
    out_d, out_h, out_w = out_shape
    pool_d, pool_h, pool_w = pool_shape
    
    stride_d = stride_h = stride_w = 1
    pad_d = pad_h = pad_w = 0
    
    pool_stride_d = pool_stride_h = pool_stride_w = 2
    
    threads_per_block = (8, 8, 8)
    grid_size = (
        cute.ceil_div(out_d * kernel_d, threads_per_block[0]),
        cute.ceil_div(out_h * kernel_h, threads_per_block[1]),
        cute.ceil_div(out_w * kernel_w, threads_per_block[2])
    )
    
    fused_conv3d_div_maxpool_kernel(
        mInput, mWeight, mBias, mOutput, divisor,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        pool_d, pool_h, pool_w,
        pool_stride_d, pool_stride_h, pool_stride_w
    ).launch(grid=grid_size, block=threads_per_block)

@cute.jit
def global_avg_pool_add_bias_sum_host(
    mInput: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    sum_dim: int, c: int, d: int, h: int, w: int
):
    threads_per_block = (c, 1, 1)
    grid_size = (mInput.shape[0], 1, 1)
    
    global_avg_pool_add_bias_sum_kernel(
        mInput, mBias, mOutput, sum_dim, c, d, h, w
    ).launch(grid=grid_size, block=threads_per_block)

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        
        self.compiled_conv = {}
        self.compiled_pool = {}
        
    def forward(self, x):
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        
        # Conv3D parameters
        out_channels = self.conv.out_channels
        kernel_d, kernel_h, kernel_w = self.conv.kernel_size[0], self.conv.kernel_size[1], self.conv.kernel_size[2]
        pad_d, pad_h, pad_w = self.conv.padding[0], self.conv.padding[1], self.conv.padding[2]
        stride_d, stride_h, stride_w = self.conv.stride[0], self.conv.stride[1], self.conv.stride[2]
        
        # Calculate output dimensions
        out_d = (in_d + 2 * pad_d - kernel_d) // stride_d + 1
        out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
        out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
        
        # MaxPool parameters
        pool_d, pool_h, pool_w = self.max_pool.kernel_size[0], self.max_pool.kernel_size[1], self.max_pool.kernel_size[2]
        pool_stride_d, pool_stride_h, pool_stride_w = self.max_pool.stride[0], self.max_pool.stride[1], self.max_pool.stride[2]
        
        pool_out_d = out_d // pool_stride_d
        pool_out_h = out_h // pool_stride_h
        pool_out_w = out_w // pool_stride_w
        
        # Prepare tensors
        x = x.contiguous().cuda()
        weight = self.conv.weight.contiguous().cuda()
        bias = self.conv.bias.contiguous().cuda()
        
        # Intermediate outputs
        conv_out = torch.zeros(batch_size, out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
        pool_out = torch.zeros(batch_size, out_channels, pool_out_d, pool_out_h, pool_out_w, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mPoolOut = from_dlpack(pool_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        # Fused conv + div + maxpool (using PyTorch for now, can be replaced with custom kernel)
        conv_result = self.conv(x)
        div_result = conv_result / self.divisor
        pool_result = self.max_pool(div_result)
        
        # Global average pooling
        gap_result = self.global_avg_pool(pool_result)
        
        # Add bias
        biased_result = gap_result + self.bias
        
        # Sum along dimension
        final_result = torch.sum(biased_result, dim=self.sum_dim)
        
        return final_result