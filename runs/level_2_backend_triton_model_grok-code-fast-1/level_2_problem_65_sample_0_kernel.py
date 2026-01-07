import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avgpool_sigmoid_sum_kernel(
    x_ptr,
    out_ptr,
    B, C, H, W, ks,
    OH, OW,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)
    
    sum_val = 0.0
    for di in range(ks):
        for dj in range(ks):
            ih = oh * ks + di
            iw = ow * ks + dj
            if ih < H and iw < W:
                idx = b * (C * H * W) + c * (H * W) + ih * W + iw
                val = tl.load(x_ptr + idx)
                sum_val += val
    avg = sum_val / (ks * ks)
    sigmoid_val = tl.sigmoid(avg)
    tl.atomic_add(out_ptr + b, sigmoid_val)


def triton_avgpool_sigmoid_sum(x: torch.Tensor, ks: int):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    B, C, H, W = x.shape
    OH = H // ks
    OW = W // ks
    out = torch.zeros(B, dtype=x.dtype, device=x.device)
    grid = (B, C, OH, OW)
    avgpool_sigmoid_sum_kernel[grid](x, out, B, C, H, W, ks, OH, OW, BLOCK_SIZE=1)
    return out


class ModelNew(nn.Module):
    """
    This model performs a convolution, then fuses average pooling, sigmoid, and sum using a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_ks = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        return triton_avgpool_sigmoid_sum(x, self.pool_ks)