import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    depth, height, width,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_depth, out_height, out_width,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    oc_start = pid_oc * BLOCK_C
    dhw_start = pid_dhw * BLOCK_DHW

    oc_offsets = oc_start + tl.arange(0, BLOCK_C)
    dhw_offsets = dhw_start + tl.arange(0, BLOCK_DHW)

    mask_oc = oc_offsets < out_channels
    mask_dhw = dhw_offsets < (out_depth * out_height * out_width)

    for b in range(batch_size):
        for idx in range(BLOCK_DHW):
            if dhw_offsets[idx] < (out_depth * out_height * out_width):
                d_idx = dhw_offsets[idx] // (out_height * out_width)
                hw_idx = dhw_offsets[idx] % (out_height * out_width)
                h_idx = hw_idx // out_width
                w_idx = hw_idx % out_width

                in_d_start = d_idx * stride_d - pad_d
                in_h_start = h_idx * stride_h - pad_h
                in_w_start = w_idx * stride_w - pad_w

                acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

                for ic in range(0, in_channels, BLOCK_C):
                    ic_offsets = ic + tl.arange(0, BLOCK_C)
                    mask_ic = ic_offsets < in_channels

                    for kd in range(kernel_d):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                in_d = in_d_start + kd
                                in_h = in_h_start + kh
                                in_w = in_w_start + kw

                                if in_d >= 0 and in_d < depth and in_h >= 0 and in_h < height and in_w >= 0 and in_w < width:
                                    in_idx = b * in_channels * depth * height * width + \
                                             ic_offsets * depth * height * width + \
                                             in_d * height * width + in_h * width + in_w
                                    w_idx = oc_offsets * in_channels * kernel_d * kernel_h * kernel_w + \
                                            ic_offsets * kernel_d * kernel_h * kernel_w + \
                                            kd * kernel_h * kernel_w + kh * kernel_w + kw

                                    in_val = tl.load(input_ptr + in_idx, mask=mask_ic, other=0.0)
                                    w_val = tl.load(weight_ptr + w_idx, mask=mask_oc[:, None] & mask_ic, other=0.0)
                                    acc += tl.sum(in_val * w_val, axis=1)

                out_idx = b * out_channels * out_depth * out_height * out_width + \
                          oc_offsets * out_depth * out_height * out_width + \
                          dhw_offsets[idx]
                tl.store(output_ptr + out_idx, acc, mask=mask_oc)


@triton.jit
def maxpool3d_kernel(
    input_ptr, output_ptr,
    batch_size, channels, depth, height, width,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    out_depth, out_height, out_width,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    c_start = pid_c * BLOCK_C
    dhw_start = pid_dhw * BLOCK_DHW

    c_offsets = c_start + tl.arange(0, BLOCK_C)
    dhw_offsets = dhw_start + tl.arange(0, BLOCK_DHW)

    mask_c = c_offsets < channels
    mask_dhw = dhw_offsets < (out_depth * out_height * out_width)

    for b in range(batch_size):
        for idx in range(BLOCK_DHW):
            if dhw_offsets[idx] < (out_depth * out_height * out_width):
                d_idx = dhw_offsets[idx] // (out_height * out_width)
                hw_idx = dhw_offsets[idx] % (out_height * out_width)
                h_idx = hw_idx // out_width
                w_idx = hw_idx % out_width

                out_d_start = d_idx * stride_d
                out_h_start = h_idx * stride_h
                out_w_start = w_idx * stride_w

                max_val = tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)

                for kd in range(kernel_d):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            in_d = out_d_start + kd
                            in_h = out_h_start + kh
                            in_w = out_w_start + kw

                            if in_d < depth and in_h < height and in_w < width:
                                in_idx = b * channels * depth * height * width + \
                                         c_offsets * depth * height * width + \
                                         in_d * height * width + in_h * width + in_w
                                val = tl.load(input_ptr + in_idx, mask=mask_c, other=float('-inf'))
                                max_val = tl.maximum(max_val, val)

                out_idx = b * channels * out_depth * out_height * out_width + \
                          c_offsets * out_depth * out_height * out_width + \
                          dhw_offsets[idx]
                tl.store(output_ptr + out_idx, max_val, mask=mask_c)


@triton.jit
def logsumexp_relu_kernel(
    input_ptr, output_ptr,
    batch_size, channels, depth, height, width,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    dhw_start = pid_dhw * BLOCK_DHW
    dhw_offsets = dhw_start + tl.arange(0, BLOCK_DHW)
    mask_dhw = dhw_offsets < (depth * height * width)

    for b in range(batch_size):
        for idx in range(BLOCK_DHW):
            if dhw_offsets[idx] < (depth * height * width):
                max_val = tl.full((1,), float('-inf'), dtype=tl.float32)

                # Find max for numerical stability
                for c in range(0, channels, BLOCK_C):
                    c_offsets = c + tl.arange(0, BLOCK_C)
                    mask_c = c_offsets < channels
                    in_idx = b * channels * depth * height * width + \
                             c_offsets * depth * height * width + \
                             dhw_offsets[idx]
                    val = tl.load(input_ptr + in_idx, mask=mask_c, other=float('-inf'))
                    max_val = tl.maximum(max_val, tl.max(val))

                # Compute exp(sum)
                sum_exp = tl.full((1,), 0.0, dtype=tl.float32)
                for c in range(0, channels, BLOCK_C):
                    c_offsets = c + tl.arange(0, BLOCK_C)
                    mask_c = c_offsets < channels
                    in_idx = b * channels * depth * height * width + \
                             c_offsets * depth * height * width + \
                             dhw_offsets[idx]
                    val = tl.load(input_ptr + in_idx, mask=mask_c, other=0.0)
                    sum_exp += tl.sum(tl.exp(val - max_val))

                logsumexp = tl.log(sum_exp) + max_val
                relu_out = tl.maximum(logsumexp, 0.0)

                out_idx = b * 1 * depth * height * width + dhw_offsets[idx]
                tl.store(output_ptr + out_idx, relu_out)


def triton_conv3d(input, weight, bias, stride, padding):
    batch_size, in_channels, depth, height, width = input.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    out_depth = (depth + 2 * pad_d - kernel_d) // stride_d + 1
    out_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_w) // stride_w + 1

    output = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=input.device, dtype=input.dtype)

    BLOCK_C = 32
    BLOCK_DHW = 64

    grid = (batch_size, (out_channels + BLOCK_C - 1) // BLOCK_C, (out_depth * out_height * out_width + BLOCK_DHW - 1) // BLOCK_DHW)

    conv3d_kernel[grid](
        input, weight, bias, output,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_depth, out_height, out_width,
        BLOCK_C=BLOCK_C, BLOCK_DHW=BLOCK_DHW
    )
    return output


def triton_maxpool3d(input, kernel_size, stride):
    batch_size, channels, depth, height, width = input.shape
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride

    out_depth = depth // stride_d
    out_height = height // stride_h
    out_width = width // stride_w

    output = torch.empty(batch_size, channels, out_depth, out_height, out_width, device=input.device, dtype=input.dtype)

    BLOCK_C = 32
    BLOCK_DHW = 64

    grid = (batch_size, (channels + BLOCK_C - 1) // BLOCK_C, (out_depth * out_height * out_width + BLOCK_DHW - 1) // BLOCK_DHW)

    maxpool3d_kernel[grid](
        input, output,
        batch_size, channels, depth, height, width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        out_depth, out_height, out_width,
        BLOCK_C=BLOCK_C, BLOCK_DHW=BLOCK_DHW
    )
    return output


def triton_logsumexp_relu(input):
    batch_size, channels, depth, height, width = input.shape
    output = torch.empty(batch_size, 1, depth, height, width, device=input.device, dtype=input.dtype)

    BLOCK_DHW = 64

    grid = (batch_size, (depth * height * width + BLOCK_DHW - 1) // BLOCK_DHW)

    logsumexp_relu_kernel[grid](
        input, output,
        batch_size, channels, depth, height, width,
        BLOCK_C=32, BLOCK_DHW=BLOCK_DHW
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = triton_conv3d(x, self.weight, self.bias, (self.stride, self.stride, self.stride), (self.padding, self.padding, self.padding))
        x = triton_maxpool3d(x, (2, 2, 2), (2, 2, 2))
        x = triton_logsumexp_relu(x)
        return x