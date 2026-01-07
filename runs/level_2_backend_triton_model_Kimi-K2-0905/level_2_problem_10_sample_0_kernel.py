import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_conv_maxpool_hardtanh_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, in_h, in_w,
    out_h, out_w, kernel_size, stride, padding,
    pool_kernel, pool_stride, pool_out_h, pool_out_w,
    hardtanh_min, hardtanh_max,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    numel = batch_size * out_channels * pool_out_h * pool_out_w
    for i in range(pid, numel, tl.num_programs(0)):
        b = i // (out_channels * pool_out_h * pool_out_w)
        c = (i // (pool_out_h * pool_out_w)) % out_channels
        ph = (i // pool_out_w) % pool_out_h
        pw = i % pool_out_w

        h_start = ph * pool_stride
        h_end = h_start + pool_kernel
        w_start = pw * pool_stride
        w_end = w_start + pool_kernel

        max_val = float('-inf')
        for oh in range(h_start, h_end):
            for ow in range(w_start, w_end):
                acc = 0.0
                for ic in range(in_channels):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            ih = oh + kh - padding
                            iw = ow + kw - padding
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                inp_idx = b * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw
                                w_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw
                                inp_val = tl.load(input_ptr + inp_idx)
                                w_val = tl.load(weight_ptr + w_idx)
                                acc += inp_val * w_val
                if bias_ptr:
                    bias_val = tl.load(bias_ptr + c)
                    acc += bias_val
                acc = tl.clamp(acc, hardtanh_min, hardtanh_max)
                max_val = tl.maximum(max_val, acc)
        out_idx = b * out_channels * pool_out_h * pool_out_w + c * pool_out_h * pool_out_w + ph * pool_out_w + pw
        tl.store(output_ptr + out_idx, max_val)


@triton.jit
def fused_mean_tanh_kernel(
    input_ptr, output_ptr,
    batch_size, channels, h, w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    numel = batch_size * channels
    for i in range(pid, numel, tl.num_programs(0)):
        b = i // channels
        c = i % channels
        sum_val = 0.0
        for j in range(h * w):
            idx = b * channels * h * w + c * h * w + j
            val = tl.load(input_ptr + idx)
            sum_val += val
        mean_val = sum_val / (h * w)
        tanh_val = tl.tanh(mean_val)
        out_idx = b * channels + c
        tl.store(output_ptr + out_idx, tanh_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels = self.conv_transpose.out_channels
        kernel_size = self.conv_transpose.kernel_size[0]
        stride = self.conv_transpose.stride[0]
        padding = self.conv_transpose.padding[0]
        out_h = (in_h - 1) * stride - 2 * padding + kernel_size
        out_w = (in_w - 1) * stride - 2 * padding + kernel_size
        pool_out_h = (out_h - self.maxpool_kernel_size) // self.maxpool_stride + 1
        pool_out_w = (out_w - self.maxpool_kernel_size) // self.maxpool_stride + 1

        x = x.contiguous()
        weight = self.conv_transpose.weight.data.contiguous()
        bias = self.conv_transpose.bias
        if bias is not None:
            bias = bias.data.contiguous()

        # Allocate intermediate tensor after conv_transpose + maxpool + hardtanh
        inter = torch.empty(batch_size, out_channels, pool_out_h, pool_out_w, device=x.device, dtype=x.dtype)

        numel = batch_size * out_channels * pool_out_h * pool_out_w
        BLOCK_SIZE = 128
        grid = lambda meta: (min(tl.cdiv(numel, meta['BLOCK_SIZE']), 1024),)

        fused_transpose_conv_maxpool_hardtanh_kernel[grid](
            x, weight, bias, inter,
            batch_size, in_channels, out_channels, in_h, in_w,
            out_h, out_w, kernel_size, stride, padding,
            self.maxpool_kernel_size, self.maxpool_stride, pool_out_h, pool_out_w,
            self.hardtanh_min, self.hardtanh_max,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Allocate output tensor after mean + tanh
        out = torch.empty(batch_size, out_channels, device=x.device, dtype=x.dtype)
        numel2 = batch_size * out_channels
        grid2 = lambda meta: (min(tl.cdiv(numel2, meta['BLOCK_SIZE']), 1024),)

        fused_mean_tanh_kernel[grid2](
            inter, out,
            batch_size, out_channels, pool_out_h, pool_out_w,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return out.unsqueeze(-1).unsqueeze(-1)