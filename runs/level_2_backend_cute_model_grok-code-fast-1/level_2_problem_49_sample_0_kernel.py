import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def softmax_kernel(gX: cute.Tensor, gY: cute.Tensor, C: int):
    s = cute.arch.block_idx(0)
    c = cute.arch.thread_idx(0)
    if c >= C:
        return
    smem = cute.shared_tensor(cute.float32, (C,))
    x = gX[s, c]
    smem[c] = x
    cute.sync()
    # reduction for max
    stride = 1
    while stride < C:
        if c % (2 * stride) == 0 and c + stride < C:
            smem[c] = cute.max(smem[c], smem[c + stride])
        stride *= 2
        cute.sync()
    max_val = smem[0]
    cute.sync()
    # exp
    exp_val = cute.exp(x - max_val)
    smem[c] = exp_val
    cute.sync()
    # sum
    stride = 1
    while stride < C:
        if c % (2 * stride) == 0 and c + stride < C:
            smem[c] += smem[c + stride]
        stride *= 2
        cute.sync()
    sum_val = smem[0]
    cute.sync()
    gY[s, c] = exp_val / sum_val

@cute.jit
def softmax_host(mX: cute.Tensor, mY: cute.Tensor, C: int):
    S = mX.shape[0]
    softmax_kernel(mX, mY, C).launch(grid=(S, 1, 1), block=(C, 1, 1))

@cute.kernel
def sigmoid_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_idx = bidx * bdim + tidx
    total = gX.size()
    if thread_idx < total:
        gY[thread_idx] = 1.0 / (1.0 + cute.exp(-gX[thread_idx]))

@cute.jit
def sigmoid_host(mX: cute.Tensor, mY: cute.Tensor):
    total = mX.size()
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    sigmoid_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, applies custom Softmax and Sigmoid using CuTe kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.out_channels = out_channels
        self.compiled = {}

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv_transpose(x)
        B, C, Dd, Hh, Ww = x.shape
        S = B * Dd * Hh * Ww
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(S, C)
        y_flat = torch.empty_like(x_flat)
        
        mX = from_dlpack(x_flat, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mY = from_dlpack(y_flat, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key_softmax = (C, x_flat.dtype)
        compiled_softmax = self.compiled.get(key_softmax)
        if compiled_softmax is None:
            compiled_softmax = cute.compile(softmax_host, mX, mY, C)
            self.compiled[key_softmax] = compiled_softmax
        
        compiled_softmax(mX, mY, C)
        
        y = y_flat.view(B, Dd, Hh, Ww, C).permute(0, 4, 1, 2, 3).contiguous()
        y_flat_sig = y.contiguous().view(-1)
        z_flat = torch.empty_like(y_flat_sig)
        
        mX_sig = from_dlpack(y_flat_sig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mZ = from_dlpack(z_flat, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key_sigmoid = ('sigmoid', y_flat_sig.dtype)
        compiled_sigmoid = self.compiled.get(key_sigmoid)
        if compiled_sigmoid is None:
            compiled_sigmoid = cute.compile(sigmoid_host, mX_sig, mZ)
            self.compiled[key_sigmoid] = compiled_sigmoid
        
        compiled_sigmoid(mX_sig, mZ)
        
        z = z_flat.view_as(y)
        return z