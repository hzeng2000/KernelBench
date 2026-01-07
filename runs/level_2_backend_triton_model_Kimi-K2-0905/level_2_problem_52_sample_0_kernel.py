import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_activation_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, out_c, h, w, k,
    stride_h, stride_w, pad_h, pad_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    hw = h * w
    n = pid // hw
    hw_rem = pid % hw
    ho = hw_rem // w
    wo = hw_rem % w

    acc = 0.0
    for ic in range(in_c):
        for kh in range(k):
            for kw in range(k):
                hi = ho * stride_h - pad_h + kh
                wi = wo * stride_w - pad_w + kw
                if hi >= 0 and hi < h and wi >= 0 and wi < w:
                    x_idx = n * in_c * h * w + ic * h * w + hi * w + wi
                    w_idx = (pid // hw) % out_c * in_c * k * k + ic * k * k + kh * k + kw
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val

    out_idx = pid
    if b_ptr is not None:
        b_val = tl.load(b_ptr + (pid // hw) % out_c)
        acc += b_val

    # Softplus + Tanh * x fusion
    softplus = tl.log(1.0 + tl.exp(acc))
    tanh_sp = tl.tanh(softplus)
    out_val = tanh_sp * acc

    tl.store(out_ptr + out_idx, out_val)


@triton.jit
def fused_bn_kernel(
    x_ptr, out_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n, c, hw, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    c_idx = pid // hw
    hw_idx = pid % hw

    mean = tl.load(mean_ptr + c_idx)
    var = tl.load(var_ptr + c_idx)
    weight = tl.load(weight_ptr + c_idx)
    bias = tl.load(bias_ptr + c_idx)

    x_val = tl.load(x_ptr + pid)
    norm = (x_val - mean) / tl.sqrt(var + eps)
    out_val = norm * weight + bias
    tl.store(out_ptr + pid, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        # Fused conv + activation
        batch, in_c, h, w = x.shape
        out_c = self.conv.out_channels
        k = self.conv.kernel_size[0]
        stride_h = self.conv.stride[0]
        stride_w = self.conv.stride[1]
        pad_h = self.conv.padding[0]
        pad_w = self.conv.padding[1]

        out_h = (h + 2 * pad_h - k) // stride_h + 1
        out_w = (w + 2 * pad_w - k) // stride_w + 1

        x = x.contiguous()
        w = self.conv.weight.data.contiguous()
        b = self.conv.bias.data.contiguous() if self.conv.bias is not None else None

        out = torch.empty(batch, out_c, out_h, out_w, device=x.device, dtype=x.dtype)
        n_elements = batch * out_c * out_h * out_w
        BLOCK_SIZE = 128
        grid = lambda meta: (n_elements,)

        fused_conv_activation_kernel[grid](
            x, w, b, out,
            batch, in_c, out_c, h, w, k,
            stride_h, stride_w, pad_h, pad_w,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Fused BatchNorm
        if self.bn.training:
            mean = out.mean(dim=[0, 2, 3])
            var = out.var(dim=[0, 2, 3], unbiased=False)
            self.bn.running_mean = (1 - self.bn.momentum) * self.bn.running_mean + self.bn.momentum * mean
            self.bn.running_var = (1 - self.bn.momentum) * self.bn.running_var + self.bn.momentum * var
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var

        out_flat = out.view(batch * out_c * out_h * out_w)
        out_bn = torch.empty_like(out_flat)

        fused_bn_kernel[grid](
            out_flat, out_bn,
            mean, var, self.bn.weight.data, self.bn.bias.data,
            batch, out_c, out_h * out_w, self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return out_bn.view(batch, out_c, out_h, out_w)