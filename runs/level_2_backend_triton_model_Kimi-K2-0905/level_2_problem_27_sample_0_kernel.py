import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_hardswish_gn_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    B, C_in, C_out, D, H, W, KD, KH, KW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Compute output sizes
    D_out = (D + 2 * pad_d - KD) // stride_d + 1
    H_out = (H + 2 * pad_h - KH) // stride_h + 1
    W_out = (W + 2 * pad_w - KW) // stride_w + 1
    
    total_out = B * C_out * D_out * H_out * W_out
    mask = offsets < total_out
    
    # Compute indices
    tmp = offsets
    w_out = tmp % W_out
    tmp = tmp // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    d_out = tmp % D_out
    tmp = tmp // D_out
    c_out = tmp % C_out
    b = tmp // C_out
    
    # Compute input start indices
    d_start = d_out * stride_d - pad_d
    h_start = h_out * stride_h - pad_h
    w_start = w_out * stride_w - pad_w
    
    # Compute group norm parameters
    group_size = C_out // groups
    group_idx = c_out // group_size
    
    # Compute conv output
    acc = 0.0
    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                d_in = d_start + kd
                h_in = h_start + kh
                w_in = w_start + kw
                if d_in >= 0 and d_in < D and h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                    for c_in in range(C_in):
                        x_idx = b * C_in * D * H * W + c_in * D * H * W + d_in * H * W + h_in * W + w_in
                        w_idx = c_out * C_in * KD * KH * KW + c_in * KD * KH * KW + kd * KH * KW + kh * KW + kw
                        x_val = tl.load(x_ptr + x_idx)
                        w_val = tl.load(w_ptr + w_idx)
                        acc += x_val * w_val
    
    # Add bias
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_out)
        acc += b_val
    
    # HardSwish
    acc = acc * tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0) / 6.0
    
    # Store intermediate for group norm
    out_idx = b * C_out * D_out * H_out * W_out + c_out * D_out * H_out * W_out + d_out * H_out * W_out + h_out * W_out + w_out
    tl.store(out_ptr + out_idx, acc, mask=mask)


@triton.jit
def group_norm_reduce_kernel(
    out_ptr, mean_ptr, var_ptr,
    B, C, D, H, W,
    groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    group_size = C // groups
    total_spatial = D * H * W
    
    # Compute indices
    tmp = pid
    group = tmp % groups
    b = tmp // groups
    
    c_start = group * group_size
    c_end = c_start + group_size
    
    # Compute mean
    sum_val = 0.0
    count = 0
    for c in range(c_start, c_end):
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w
                    val = tl.load(out_ptr + idx)
                    sum_val += val
                    count += 1
    mean = sum_val / count
    tl.store(mean_ptr + pid, mean)
    
    # Compute variance
    sum_sq = 0.0
    for c in range(c_start, c_end):
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w
                    val = tl.load(out_ptr + idx)
                    sum_sq += (val - mean) * (val - mean)
    var = sum_sq / count
    tl.store(var_ptr + pid, var)


@triton.jit
def group_norm_apply_kernel(
    out_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, final_ptr,
    B, C, D, H, W,
    groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total = B * C * D * H * W
    mask = offsets < total
    
    # Compute indices
    tmp = offsets
    w = tmp % W
    tmp = tmp // W
    h = tmp % H
    tmp = tmp // H
    d = tmp % D
    tmp = tmp // D
    c = tmp % C
    b = tmp // C
    
    group_size = C // groups
    group = c // group_size
    reduce_idx = b * groups + group
    
    mean = tl.load(mean_ptr + reduce_idx)
    var = tl.load(var_ptr + reduce_idx)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w
    val = tl.load(out_ptr + idx)
    
    normalized = (val - mean) * inv_std
    
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + c)
        normalized = normalized * weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + c)
        normalized = normalized + bias
    
    tl.store(final_ptr + idx, normalized, mask=mask)


@triton.jit
def mean_pool_kernel(
    final_ptr, out_ptr,
    B, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total = B * C
    mask = offsets < total
    
    # Compute indices
    tmp = offsets
    c = tmp % C
    b = tmp // C
    
    # Compute mean
    sum_val = 0.0
    count = D * H * W
    for d in range(D):
        for h in range(H):
            for w in range(W):
                idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w
                val = tl.load(final_ptr + idx)
                sum_val += val
    mean = sum_val / count
    
    out_idx = b * C + c
    tl.store(out_ptr + out_idx, mean, mask=mask)


def triton_fused_conv_hardswish_gn_mean(x, weight, bias, gn_weight, gn_bias, groups, eps=1e-5):
    B, C_in, D, H, W = x.shape
    C_out, _, KD, KH, KW = weight.shape
    
    # Compute output sizes (assuming stride=1, padding=0 for simplicity)
    D_out = D - KD + 1
    H_out = H - KH + 1
    W_out = W - KW + 1
    
    # Allocate intermediate buffer
    intermediate = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Launch conv+swish kernel
    total_threads = B * C_out * D_out * H_out * W_out
    BLOCK_SIZE = 128
    grid = lambda meta: ((total_threads + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Allocate buffers for group norm reduction
    mean_buffer = torch.empty(B * groups, device=x.device, dtype=x.dtype)
    var_buffer = torch.empty(B * groups, device=x.device, dtype=x.dtype)
    
    # First kernel: conv + hardswish
    fused_conv_hardswish_gn_mean_kernel[grid](
        x, weight, bias, intermediate,
        mean_buffer, var_buffer, gn_weight, gn_bias,
        B, C_in, C_out, D, H, W, KD, KH, KW,
        1, 1, 1, 0, 0, 0,
        groups, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Second kernel: group norm reduce
    grid2 = lambda meta: (B * groups,)
    group_norm_reduce_kernel[grid2](
        intermediate, mean_buffer, var_buffer,
        B, C_out, D_out, H_out, W_out,
        groups, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Third kernel: group norm apply
    total_threads3 = B * C_out * D_out * H_out * W_out
    grid3 = lambda meta: ((total_threads3 + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    group_norm_apply_kernel[grid3](
        intermediate, mean_buffer, var_buffer, gn_weight, gn_bias, intermediate,
        B, C_out, D_out, H_out, W_out,
        groups, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Fourth kernel: mean pool
    out = torch.empty(B, C_out, device=x.device, dtype=x.dtype)
    grid4 = lambda meta: ((B * C_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mean_pool_kernel[grid4](
        intermediate, out,
        B, C_out, D_out, H_out, W_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups

    def forward(self, x):
        return triton_fused_conv_hardswish_gn_mean(
            x, self.weight, self.bias,
            self.group_norm.weight, self.group_norm.bias,
            self.num_groups
        )