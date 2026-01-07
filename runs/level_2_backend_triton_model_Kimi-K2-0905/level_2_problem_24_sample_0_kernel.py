import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, D, H, W,
    kernel_size, stride, padding,
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    if pid_b >= batch_size or pid_o >= out_channels or pid_d >= D or pid_h >= H or pid_w >= W:
        return

    acc = 0.0
    for ic in range(in_channels):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    in_d = pid_d - padding + kd
                    in_h = pid_h - padding + kh
                    in_w = pid_w - padding + kw
                    if 0 <= in_d < D and 0 <= in_h < H and 0 <= in_w < W:
                        in_idx = pid_b * in_channels * D * H * W + ic * D * H * W + in_d * H * W + in_h * W + in_w
                        w_idx = pid_o * in_channels * kernel_size * kernel_size * kernel_size + ic * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw
                        in_val = tl.load(input_ptr + in_idx)
                        w_val = tl.load(weight_ptr + w_idx)
                        acc += in_val * w_val

    if bias_ptr is not None:
        b_val = tl.load(bias_ptr + pid_o)
        acc += b_val

    out_idx = pid_b * out_channels * D * H * W + pid_o * D * H * W + pid_d * H * W + pid_h * W + pid_w
    tl.store(output_ptr + out_idx, acc)


@triton.jit
def min_and_softmax_kernel(
    input_ptr, output_ptr,
    batch_size, out_channels, D, H, W, dim,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    if pid_b >= batch_size or pid_o >= out_channels or pid_h >= H or pid_w >= W:
        return

    min_val = float('inf')
    for d in range(D):
        idx = pid_b * out_channels * D * H * W + pid_o * D * H * W + d * H * W + pid_h * W + pid_w
        val = tl.load(input_ptr + idx)
        if val < min_val:
            min_val = val

    out_idx = pid_b * out_channels * H * W + pid_o * H * W + pid_h * W + pid_w
    tl.store(output_ptr + out_idx, min_val)

    # Softmax along channel dimension
    max_val = float('-inf')
    for o in range(out_channels):
        idx = pid_b * out_channels * H * W + o * H * W + pid_h * W + pid_w
        val = tl.load(output_ptr + idx)
        if val > max_val:
            max_val = val

    sum_exp = 0.0
    for o in range(out_channels):
        idx = pid_b * out_channels * H * W + o * H * W + pid_h * W + pid_w
        val = tl.load(output_ptr + idx)
        exp_val = tl.exp(val - max_val)
        sum_exp += exp_val
        tl.store(output_ptr + idx, exp_val)

    for o in range(out_channels):
        idx = pid_b * out_channels * H * W + o * H * W + pid_h * W + pid_w
        val = tl.load(output_ptr + idx)
        tl.store(output_ptr + idx, val / sum_exp)


def triton_conv3d_min_softmax(input_tensor, weight, bias, dim):
    batch_size, in_channels, D, H, W = input_tensor.shape
    out_channels, _, kernel_size, _, _ = weight.shape
    padding = 1
    stride = 1

    output = torch.empty(batch_size, out_channels, D, H, W, device=input_tensor.device, dtype=input_tensor.dtype)
    grid = (batch_size, out_channels, D, H, W)
    conv3d_kernel[grid](
        input_tensor, weight, bias, output,
        batch_size, in_channels, out_channels, D, H, W,
        kernel_size, stride, padding,
        BLOCK_C=1, BLOCK_D=1, BLOCK_H=1, BLOCK_W=1
    )

    final_output = torch.empty(batch_size, out_channels, H, W, device=input_tensor.device, dtype=input_tensor.dtype)
    grid2 = (batch_size, out_channels, H, W)
    min_and_softmax_kernel[grid2](
        output, final_output,
        batch_size, out_channels, D, H, W, dim,
        BLOCK_H=1, BLOCK_W=1
    )

    return final_output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return triton_conv3d_min_softmax(x, weight, bias, self.dim)


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]