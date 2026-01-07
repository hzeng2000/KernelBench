import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_bias_relu_kernel(gX: cute.Tensor, gBias: cute.Tensor, gOut: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    batch_size, out_features = gX.shape
    total_elems = batch_size * out_features

    if thread_idx >= total_elems:
        return

    bi = thread_idx // out_features  
    fi = thread_idx % out_features  

    x_val = gX[bi, fi]
    bias_val = gBias[fi]

    gOut[bi, fi] = max(0.0, x_val + bias_val)

@cute.jit
def fused_bias_relu_host(mX: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor):
    batch_size, out_features = mX.shape
    threads_per_block = 256
    total_elems = batch_size * out_features
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_bias_relu_kernel(mX, mBias, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using PyTorch's nn.Linear, 
    then fuses bias addition and ReLU into a single custom CuTe kernel.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        x = self.gemm(x)
        batch_size, out_features = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_bias_relu_host, mX, mBias, mOut)
            self.compiled[key] = compiled

        compiled(mX, mBias, mOut)
        return out