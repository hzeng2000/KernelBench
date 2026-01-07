import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_activation_bn_kernel(
    input: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    running_mean: cute.Tensor,
    running_var: cute.Tensor,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    output: cute.Tensor,
    eps: float,
    N: int, C: int, H: int, W: int,
    K: int, R: int, S: int,
    P: int, Q: int
):
    # Thread indices
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    # Global thread ID
    tid = tidx + tidy * bdimx + tidz * bdimx * bdimy
    total_threads = bdimx * bdimy * bdimz
    
    # Output dimensions
    out_hw = P * Q
    out_chw = K * out_hw
    
    # Each thread processes multiple output elements
    for idx in range(tid, N * out_chw, total_threads):
        n = idx // out_chw
        k = (idx // out_hw) % K
        pq = idx % out_hw
        p = pq // Q
        q = pq % Q
        
        # Compute convolution for this output pixel
        sum_val = 0.0
        for c in range(C):
            for r in range(R):
                for s in range(S):
                    h = p + r
                    w = q + s
                    if h < H and w < W:
                        input_val = input[n, c, h, w]
                        weight_val = weight[k, c, r, s]
                        sum_val += input_val * weight_val
        
        # Add bias
        if bias.shape[0] > 0:
            sum_val += bias[k]
        
        # Apply activation: tanh(softplus(x)) * x
        # softplus(x) = log(1 + exp(x))
        softplus_val = math.log(1.0 + math.exp(sum_val))
        tanh_val = math.tanh(softplus_val)
        activated_val = tanh_val * sum_val
        
        # Batch normalization
        mean = running_mean[k]
        var = running_var[k]
        bn_val = (activated_val - mean) / math.sqrt(var + eps)
        bn_val = bn_val * gamma[k] + beta[k]
        
        output[n, k, p, q] = bn_val

@cute.jit
def conv_activation_bn_host(
    input: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    running_mean: cute.Tensor,
    running_var: cute.Tensor,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    output: cute.Tensor,
    eps: float
):
    N, C, H, W = input.shape
    K, C, R, S = weight.shape
    P = H - R + 1
    Q = W - S + 1
    
    threads_per_block = 256
    total_elements = N * K * P * Q
    grid_size = (total_elements + threads_per_block - 1) // threads_per_block
    
    conv_activation_bn_kernel(
        input, weight, bias, running_mean, running_var, gamma, beta, output, eps,
        N, C, H, W, K, R, S, P, Q
    ).launch(grid=(grid_size, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps
        self.compiled = None
        
    def forward(self, x):
        # Get weights and parameters
        weight = self.conv.weight
        bias = self.conv.bias if self.conv.bias is not None else torch.empty(0, device=x.device)
        
        # Get batch norm parameters
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        
        # Prepare tensors
        x = x.contiguous().cuda()
        N, C, H, W = x.shape
        K = weight.shape[0]
        R, S = weight.shape[2], weight.shape[3]
        P, Q = H - R + 1, W - S + 1
        
        output = torch.empty(N, K, P, Q, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningMean = from_dlpack(running_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningVar = from_dlpack(running_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGamma = from_dlpack(gamma, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(beta, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        # Compile if not already compiled
        if self.compiled is None:
            self.compiled = cute.compile(conv_activation_bn_host, mInput, mWeight, mBias, 
                                       mRunningMean, mRunningVar, mGamma, mBeta, mOutput, self.eps)
        
        # Launch kernel
        self.compiled(mInput, mWeight, mBias, mRunningMean, mRunningVar, mGamma, mBeta, mOutput, self.eps)
        
        return output