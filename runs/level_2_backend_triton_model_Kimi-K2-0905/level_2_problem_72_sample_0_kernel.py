import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    in_d, in_h, in_w, out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_elements = batch_size * out_channels * out_d * out_h * out_w
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Compute 5D indices from linear offset
    n = offsets // (out_channels * out_d * out_h * out_w)
    rem = offsets % (out_channels * out_d * out_h * out_w)
    c_out = rem // (out_d * out_h * out_w)
    rem = rem % (out_d * out_h * out_w)
    d = rem // (out_h * out_w)
    rem = rem % (out_h * out_w)
    h = rem // out_w
    w = rem % out_w

    # Compute input spatial indices
    in_d_start = d * stride_d - pad_d
    in_h_start = h * stride_h - pad_h
    in_w_start = w * stride_w - pad_w

    acc = 0.0
    for kc in range(in_channels):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    in_d_idx = in_d_start + kd
                    in_h_idx = in_h_start + kh
                    in_w_idx = in_w_start + kw
                    if in_d_idx >= 0 and in_d_idx < in_d and in_h_idx >= 0 and in_h_idx < in_h and in_w_idx >= 0 and in_w_idx < in_w:
                        in_offset = n * in_channels * in_d * in_h * in_w + kc * in_d * in_h * in_w + in_d_idx * in_h * in_w + in_h_idx * in_w + in_w_idx
                        w_offset = c_out * in_channels * kernel_d * kernel_h * kernel_w + kc * kernel_d * kernel_h * kernel_w + kd * kernel_h * kernel_w + kh * kernel_w + kw
                        in_val = tl.load(input_ptr + in_offset)
                        w_val = tl.load(weight_ptr + w_offset)
                        acc += in_val * w_val
    if bias_ptr:
        b_val = tl.load(bias_ptr + c_out)
        acc += b_val
    tl.store(output_ptr + offsets, acc, mask=mask)


@triton.jit
def batch_norm3d_kernel(
    x_ptr, out_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, eps,
    num_features, spatial_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_elements = num_features * spatial_size
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    f = offsets // spatial_size
    s = offsets % spatial_size

    mean = tl.load(mean_ptr + f)
    var = tl.load(var_ptr + f)
    weight = tl.load(weight_ptr + f)
    bias = tl.load(bias_ptr + f)

    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out_val = weight * (x_val - mean) / tl.sqrt(var + eps) + bias
    tl.store(out_ptr + offsets, out_val, mask=mask)


@triton.jit
def avg_pool3d_kernel(
    x_ptr, out_ptr,
    batch_size, channels, in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_elements = batch_size * channels * out_d * out_h * out_w
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    n = offsets // (channels * out_d * out_h * out_w)
    rem = offsets % (channels * out_d * out_h * out_w)
    c = rem // (out_d * out_h * out_w)
    rem = rem % (out_d * out_h * out_w)
    d = rem // (out_h * out_w)
    rem = rem % (out_h * out_w)
    h = rem // out_w
    w = rem % out_w

    d_start = d * kernel_d
    h_start = h * kernel_h
    w_start = w * kernel_w

    acc = 0.0
    count = 0
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                in_d_idx = d_start + kd
                in_h_idx = h_start + kh
                in_w_idx = w_start + kw
                if in_d_idx < in_d and in_h_idx < in_h and in_w_idx < in_w:
                    in_offset = n * channels * in_d * in_h * in_w + c * in_d * in_h * in_w + in_d_idx * in_h * in_w + in_h_idx * in_w + in_w_idx
                    val = tl.load(x_ptr + in_offset)
                    acc += val
                    count += 1
    out_val = acc / count if count > 0 else 0.0
    tl.store(out_ptr + offsets, out_val, mask=mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding):
    batch_size, in_channels, in_d, in_h, in_w = x.shape
    out_channels = weight.shape[1]
    kernel_d, kernel_h, kernel_w = weight.shape[2:]
    stride_d, stride_h, stride_w = stride, stride, stride
    pad_d, pad_h, pad_w = padding, padding, padding

    out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d
    out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h
    out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w

    output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)
    num_elements = output.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    conv_transpose3d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


def triton_batch_norm3d(x, running_mean, running_var, weight, bias, eps=1e-5):
    num_features = x.shape[1]
    spatial_size = x.shape[2] * x.shape[3] * x.shape[4]
    output = torch.empty_like(x)
    num_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    batch_norm3d_kernel[grid](
        x, output, running_mean, running_var, weight, bias, eps,
        num_features, spatial_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


def triton_avg_pool3d(x, kernel_size):
    batch_size, channels, in_d, in_h, in_w = x.shape
    kernel_d, kernel_h, kernel_w = kernel_size, kernel_size, kernel_size
    out_d = in_d // kernel_d
    out_h = in_h // kernel_h
    out_w = in_w // kernel_w

    output = torch.empty(batch_size, channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)
    num_elements = output.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    avg_pool3d_kernel[grid](
        x, output,
        batch_size, channels, in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = triton_conv_transpose3d(x, self.conv_transpose.weight, self.conv_transpose.bias, self.conv_transpose.stride[0], self.conv_transpose.padding[0])
        x = triton_batch_norm3d(x, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.weight, self.batch_norm.bias)
        x = triton_avg_pool3d(x, 2)
        x = triton_avg_pool3d(x, 2)
        return x