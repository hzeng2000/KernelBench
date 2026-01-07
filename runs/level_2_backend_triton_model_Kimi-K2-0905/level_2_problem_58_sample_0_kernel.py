import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_logsumexp_hardswish_sub_clamp_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, H, W,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    numel = B * D * H * W
    for i in range(pid * BLOCK_SIZE, numel, BLOCK_SIZE * tl.num_programs(0)):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        b_idx = offsets // (D * H * W)
        dhw_idx = offsets % (D * H * W)
        d_idx = dhw_idx // (H * W)
        hw_idx = dhw_idx % (H * W)
        
        # Compute channel offset base
        base = b_idx * stride_b + d_idx * stride_d + (hw_idx // W) * stride_h + (hw_idx % W) * stride_w
        
        # LogSumExp over channels
        max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
        for c in range(C):
            off = base + c * stride_c
            val = tl.load(x_ptr + off, mask=mask, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
        
        # Compute exp and sum
        sum_exp = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
        for c in range(C):
            off = base + c * stride_c
            val = tl.load(x_ptr + off, mask=mask, other=float('-inf'))
            exp_val = tl.exp(val - max_val)
            sum_exp += exp_val
        
        logsumexp = max_val + tl.log(sum_exp)
        
        # HardSwish
        sigmoid_arg = logsumexp + 3.0
        sigmoid = tl.sigmoid(sigmoid_arg)
        hardswish = logsumexp * sigmoid / 6.0
        
        # Subtract bias
        bias = tl.load(bias_ptr)
        subbed = hardswish - bias
        
        # Clamp
        clamped = tl.clamp(subbed, -1.0, 1.0)
        
        # Store result
        out_off = b_idx * (D * H * W) + dhw_idx
        tl.store(out_ptr + out_off, clamped, mask=mask)


def triton_fused_ops(x, bias):
    B, C, D, H, W = x.shape
    out = torch.empty(B, 1, D, H, W, dtype=x.dtype, device=x.device)
    numel = B * D * H * W
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_logsumexp_hardswish_sub_clamp_kernel[grid](
        x, bias, out,
        B, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


@triton.jit
def conv_transpose3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_c, in_d, in_h, in_w,
    out_c, out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    output_stride_b, output_stride_c, output_stride_d, output_stride_h, output_stride_w,
    weight_stride_outc, weight_stride_inc, weight_stride_kd, weight_stride_kh, weight_stride_kw,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    numel = batch_size * out_c * out_d * out_h * out_w
    for idx in range(pid * BLOCK_SIZE, numel, BLOCK_SIZE * tl.num_programs(0)):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        
        # Compute output indices
        b = offsets // (out_c * out_d * out_h * out_w)
        rem = offsets % (out_c * out_d * out_h * out_w)
        oc = rem // (out_d * out_h * out_w)
        rem = rem % (out_d * out_h * out_w)
        od = rem // (out_h * out_w)
        rem = rem % (out_h * out_w)
        oh = rem // out_w
        ow = rem % out_w
        
        # Compute input region
        id_start = od * stride_d - pad_d
        ih_start = oh * stride_h - pad_h
        iw_start = ow * stride_w - pad_w
        
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for ic in range(in_c):
            for kd in range(kernel_d):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        id_val = id_start + kd
                        ih_val = ih_start + kh
                        iw_val = iw_start + kw
                        
                        in_bounds = (id_val >= 0) & (id_val < in_d) & (ih_val >= 0) & (ih_val < in_h) & (iw_val >= 0) & (iw_val < in_w)
                        
                        in_idx = b * input_stride_b + ic * input_stride_c + id_val * input_stride_d + ih_val * input_stride_h + iw_val * input_stride_w
                        w_idx = oc * weight_stride_outc + ic * weight_stride_inc + kd * weight_stride_kd + kh * weight_stride_kh + kw * weight_stride_kw
                        
                        in_val = tl.load(input_ptr + in_idx, mask=mask & in_bounds, other=0.0)
                        w_val = tl.load(weight_ptr + w_idx)
                        acc += in_val * w_val
        
        out_idx = b * output_stride_b + oc * output_stride_c + od * output_stride_d + oh * output_stride_h + ow * output_stride_w
        tl.store(output_ptr + out_idx, acc, mask=mask)


def triton_conv_transpose3d(input, weight, bias, stride, padding):
    batch_size, in_c, in_d, in_h, in_w = input.shape
    out_c, _, kernel_d, kernel_h, kernel_w = weight.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    
    out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d
    out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h
    out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w
    
    output = torch.empty(batch_size, out_c, out_d, out_h, out_w, dtype=input.dtype, device=input.device)
    
    numel = batch_size * out_c * out_d * out_h * out_w
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    
    conv_transpose3d_kernel[grid](
        input, weight, bias, output,
        batch_size, in_c, in_d, in_h, in_w,
        out_c, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3), input.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x):
        x = triton_conv_transpose3d(x, self.conv_transpose.weight, self.conv_transpose.bias, self.conv_transpose.stride, self.conv_transpose.padding)
        x = triton_fused_ops(x, self.bias)
        return x