import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_conv3d_gn_min_clamp_dropout_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    gMean: cute.Tensor, gVar: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor,
    min_val: float, max_val: float, dropout_prob: float, seed: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    N, C_out, D_out, H_out, W_out = gY.shape
    C_in, _, Kd, Kh, Kw = gW.shape
    
    # Compute output position
    w_out = (bidx * bdimx + tidx) % W_out
    h_out = ((bidx * bdimx + tidx) // W_out) % H_out
    d_out = ((bidx * bdimx + tidx) // (W_out * H_out)) % D_out
    c_out = ((bidx * bdimx + tidx) // (W_out * H_out * D_out)) % C_out
    n = (bidx * bdimx + tidx) // (W_out * H_out * D_out * C_out)
    
    if n < N and c_out < C_out and d_out < D_out and h_out < H_out and w_out < W_out:
        # Compute convolution
        sum_val = 0.0
        if gB:
            sum_val = gB[c_out]
        
        for c_in in range(C_in):
            for kd in range(Kd):
                for kh in range(Kh):
                    for kw in range(Kw):
                        d_in = d_out + kd
                        h_in = h_out + kh
                        w_in = w_out + kw
                        if d_in < D_out + Kd - 1 and h_in < H_out + Kh - 1 and w_in < W_out + Kw - 1:
                            x_val = gX[n, c_in, d_in, h_in, w_in]
                            w_val = gW[c_out, c_in, kd, kh, kw]
                            sum_val += x_val * w_val
        
        # GroupNorm
        group_size = C_out // gMean.shape[0]
        group_idx = c_out // group_size
        
        mean = gMean[n, group_idx, d_out, h_out, w_out]
        var = gVar[n, group_idx, d_out, h_out, w_out]
        gamma = gGamma[c_out]
        beta = gBeta[c_out]
        
        norm_val = (sum_val - mean) / cute.sqrt(var + 1e-5)
        norm_val = norm_val * gamma + beta
        
        # Min and clamp
        norm_val = cute.min(norm_val, min_val)
        norm_val = cute.clamp(norm_val, min_val, max_val)
        
        # Dropout
        rng_state = seed + (n * C_out * D_out * H_out * W_out + 
                           c_out * D_out * H_out * W_out + 
                           d_out * H_out * W_out + h_out * W_out + w_out)
        rng_val = cute.cast_float(rng_state * 1664525 + 1013904223)
        dropout_mask = rng_val > dropout_prob
        
        if dropout_mask:
            norm_val = norm_val / (1.0 - dropout_prob)
        else:
            norm_val = 0.0
        
        gY[n, c_out, d_out, h_out, w_out] = norm_val

@cute.jit
def fused_conv3d_gn_min_clamp_dropout_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    mMean: cute.Tensor, mVar: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor,
    min_val: float, max_val: float, dropout_prob: float, seed: int
):
    N, C_out, D_out, H_out, W_out = mY.shape
    total_threads = N * C_out * D_out * H_out * W_out
    
    threads_per_block = 256
    blocks = cute.ceil_div(total_threads, threads_per_block)
    
    fused_conv3d_gn_min_clamp_dropout_kernel(
        mX, mW, mB, mY, mMean, mVar, mGamma, mBeta,
        min_val, max_val, dropout_prob, seed
    ).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        self.groups = groups
        self.seed = 42
        
        # Pre-compute normalization parameters
        self.register_buffer('gamma', torch.ones(out_channels))
        self.register_buffer('beta', torch.zeros(out_channels))
        
        self.compiled = {}
        
    def forward(self, x):
        # Get conv weights and bias
        w = self.conv.weight
        b = self.conv.bias
        
        # Compute conv output shape
        N, C_in, D_in, H_in, W_in = x.shape
        C_out = w.shape[0]
        D_out = D_in - w.shape[2] + 1
        H_out = H_in - w.shape[3] + 1
        W_out = W_in - w.shape[4] + 1
        
        # Compute GroupNorm statistics
        conv_out = self.conv(x)
        gn_out = self.norm(conv_out)
        mean = conv_out.mean(dim=[2, 3, 4], keepdim=True)
        var = conv_out.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        
        # Reshape for group norm
        mean = mean.reshape(N, self.groups, C_out // self.groups, D_out, H_out, W_out).mean(dim=2)
        var = var.reshape(N, self.groups, C_out // self.groups, D_out, H_out, W_out).mean(dim=2)
        
        # Prepare output tensor
        y = torch.empty(N, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mW = from_dlpack(w.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mB = from_dlpack(b.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mMean = from_dlpack(mean.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mVar = from_dlpack(var.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mGamma = from_dlpack(self.gamma.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(self.beta.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        # Compile and launch kernel
        key = (x.dtype, y.shape)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_conv3d_gn_min_clamp_dropout_host,
                mX, mW, mB, mY, mMean, mVar, mGamma, mBeta,
                self.min_value, self.max_value, self.dropout_p, self.seed
            )
            self.compiled[key] = compiled
        
        compiled(mX, mW, mB, mY, mMean, mVar, mGamma, mBeta,
                self.min_value, self.max_value, self.dropout_p, self.seed)
        
        return y