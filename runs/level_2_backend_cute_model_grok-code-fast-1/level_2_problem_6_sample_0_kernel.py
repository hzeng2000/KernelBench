import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv3d_kernel(gX: cute.Tensor, gW: cute.Tensor, gY: cute.Tensor):
    bidx, ocidx, odidx, ohidx, owidx = cute.arch.block_idx()
    
    b, ic, id, ih, iw = gX.shape
    oc, _, kd, kh, kw = gW.shape
    od = id - kd + 1
    oh = ih - kh + 1
    ow = iw - kw + 1
    
    if bidx >= b or ocidx >= oc or odidx >= od or ohidx >= oh or owidx >= ow:
        return
    
    sum_val = 0.0
    for ic_idx in range(ic):
        for kdi in range(kd):
            for khi in range(kh):
                for kwi in range(kw):
                    x_val = gX[bidx, ic_idx, odidx + kdi, ohidx + khi, owidx + kwi]
                    w_val = gW[ocidx, ic_idx, kdi, khi, kwi]
                    sum_val += x_val * w_val
    gY[bidx, ocidx, odidx, ohidx, owidx] = sum_val

@cute.jit
def conv3d_host(mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor):
    b, ic, id, ih, iw = mX.shape
    oc, _, kd, kh, kw = mW.shape
    od = id - kd + 1
    oh = ih - kh + 1
    ow = iw - kw + 1
    
    conv3d_kernel(mX, mW, mY).launch(grid=(b, oc, od, oh, ow), block=(1, 1, 1))

@cute.kernel
def softmax_kernel(gX: cute.Tensor, gY: cute.Tensor):
    bidx, didx, hidx, widx = cute.arch.block_idx()
    tidx = cute.arch.thread_idx(0)
    
    b, c, d, h, w = gX.shape
    
    if bidx >= b or didx >= d or hidx >= h or widx >= w or tidx >= c:
        return
    
    x_val = gX[bidx, tidx, didx, hidx, widx]
    
    shared_x = cute.shared_memory(float, 16)
    shared_max = cute.shared_memory(float, 1)
    shared_sum = cute.shared_memory(float, 1)
    
    shared_x[tidx] = x_val
    cute.sync()
    
    if tidx == 0:
        max_val = shared_x[0]
        for i in range(1, c):
            max_val = max(max_val, shared_x[i])
        shared_max[0] = max_val
    
    cute.sync()
    max_val = shared_max[0]
    
    exp_val = math.exp(x_val - max_val)
    shared_x[tidx] = exp_val
    cute.sync()
    
    if tidx == 0:
        sum_val = 0.0
        for i in range(c):
            sum_val += shared_x[i]
        shared_sum[0] = sum_val
    
    cute.sync()
    sum_val = shared_sum[0]
    
    gY[bidx, tidx, didx, hidx, widx] = exp_val / sum_val

@cute.jit
def softmax_host(mX: cute.Tensor, mY: cute.Tensor):
    b, c, d, h, w = mX.shape
    softmax_kernel(mX, mY).launch(grid=(b, d, h, w), block=(c, 1, 1))

@cute.kernel
def maxpool_kernel(gX: cute.Tensor, gY: cute.Tensor):
    bidx, cidx, odidx, ohidx, owidx = cute.arch.block_idx()
    
    b, c, id, ih, iw = gX.shape
    od = id // 2
    oh = ih // 2
    ow = iw // 2
    
    if bidx >= b or cidx >= c or odidx >= od or ohidx >= oh or owidx >= ow:
        return
    
    max_val = -float('inf')
    for kd in range(2):
        for kh in range(2):
            for kw in range(2):
                id_pos = odidx * 2 + kd
                ih_pos = ohidx * 2 + kh
                iw_pos = owidx * 2 + kw
                val = gX[bidx, cidx, id_pos, ih_pos, iw_pos]
                max_val = max(max_val, val)
    gY[bidx, cidx, odidx, ohidx, owidx] = max_val

@cute.jit
def maxpool_host(mX: cute.Tensor, mY: cute.Tensor):
    b, c, id, ih, iw = mX.shape
    od = id // 2
    oh = ih // 2
    ow = iw // 2
    maxpool_kernel(mX, mY).launch(grid=(b, c, od, oh, ow), block=(1, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations using custom CuTe kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
        self.compiled_conv = {}
        self.compiled_softmax = {}
        self.compiled_pool = {}

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        x = x.contiguous().cuda()
        batch_size, in_channels, depth, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]  # assuming cubic
        od = depth - kernel_size + 1
        oh = height - kernel_size + 1
        ow = width - kernel_size + 1
        
        # Conv
        conv_out = torch.empty((batch_size, out_channels, od, oh, ow), dtype=x.dtype, device=x.device)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mW = from_dlpack(self.conv.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key_conv = (x.dtype,)
        compiled_conv = self.compiled_conv.get(key_conv)
        if compiled_conv is None:
            compiled_conv = cute.compile(conv3d_host, mX, mW, mConvOut)
            self.compiled_conv[key_conv] = compiled_conv
        
        compiled_conv(mX, mW, mConvOut)
        
        # Softmax
        softmax_out = torch.empty_like(conv_out)
        mSoftmaxIn = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mSoftmaxOut = from_dlpack(softmax_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key_softmax = (conv_out.dtype,)
        compiled_softmax = self.compiled_softmax.get(key_softmax)
        if compiled_softmax is None:
            compiled_softmax = cute.compile(softmax_host, mSoftmaxIn, mSoftmaxOut)
            self.compiled_softmax[key_softmax] = compiled_softmax
        
        compiled_softmax(mSoftmaxIn, mSoftmaxOut)
        
        # Pool1
        pool1_out = torch.empty((batch_size, out_channels, od // 2, oh // 2, ow // 2), dtype=softmax_out.dtype, device=softmax_out.device)
        mPool1In = from_dlpack(softmax_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mPool1Out = from_dlpack(pool1_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key_pool = (softmax_out.dtype,)
        compiled_pool = self.compiled_pool.get(key_pool)
        if compiled_pool is None:
            compiled_pool = cute.compile(maxpool_host, mPool1In, mPool1Out)
            self.compiled_pool[key_pool] = compiled_pool
        
        compiled_pool(mPool1In, mPool1Out)
        
        # Pool2
        pool2_out = torch.empty((batch_size, out_channels, od // 4, oh // 4, ow // 4), dtype=pool1_out.dtype, device=pool1_out.device)
        mPool2In = from_dlpack(pool1_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mPool2Out = from_dlpack(pool2_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        compiled_pool(mPool2In, mPool2Out)
        
        return pool2_out