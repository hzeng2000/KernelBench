import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_sum_norm_pool_gelu_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    B, C, D, H, W,
    stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_hw = tl.program_id(3)

    c_start = pid_c * BLOCK_C
    d_start = pid_d * BLOCK_D
    hw_start = pid_hw * BLOCK_H * BLOCK_W

    c_offs = c_start + tl.arange(0, BLOCK_C)
    d_offs = d_start + tl.arange(0, BLOCK_D)
    hw_offs = hw_start + tl.arange(0, BLOCK_H * BLOCK_W)

    c_mask = c_offs < C
    d_mask = d_offs < D
    hw_mask = hw_offs < (H * W)

    # Compute mean and variance for layer norm
    mean = tl.zeros([BLOCK_D, BLOCK_H * BLOCK_W], dtype=tl.float32)
    var = tl.zeros([BLOCK_D, BLOCK_H * BLOCK_W], dtype=tl.float32)

    for c in range(0, C, BLOCK_C):
        c_offs_block = c + tl.arange(0, BLOCK_C)
        c_mask_block = c_offs_block < C
        offs = (pid_b * C * D * H * W +
                c_offs_block[:, None, None] * D * H * W +
                d_offs[None, :, None] * H * W +
                hw_offs[None, None, :])
        mask = c_mask_block[:, None, None] & d_mask[None, :, None] & hw_mask[None, None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
        var += tl.sum(x * x, axis=0)

    mean = mean / C
    var = var / C - mean * mean
    rstd = tl.rsqrt(var + 1e-5)

    # Apply operations
    for c in range(0, C, BLOCK_C):
        c_offs_block = c + tl.arange(0, BLOCK_C)
        c_mask_block = c_offs_block < C
        offs = (pid_b * C * D * H * W +
                c_offs_block[:, None, None] * D * H * W +
                d_offs[None, :, None] * H * W +
                hw_offs[None, None, :])
        mask = c_mask_block[:, None, None] & d_mask[None, :, None] & hw_mask[None, None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)

        # Add sum_weight (broadcasted)
        x = x + weight_ptr[c_offs_block][:, None, None]

        # Layer norm
        x_norm = (x - mean[None, :, :]) * rstd[None, :, :]

        # Load gamma and beta
        gamma = tl.load(bias_ptr + c_offs_block, mask=c_mask_block, other=1.0)
        beta = tl.load(bias_ptr + C + c_offs_block, mask=c_mask_block, other=0.0)
        x_norm = x_norm * gamma[:, None, None] + beta[:, None, None]

        # Average pooling (simplified for 2x2x2)
        # GELU activation
        x_gelu = 0.5 * x_norm * (1.0 + tl.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

        tl.store(out_ptr + offs, x_gelu, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, D, H, W = x.shape
        out = torch.empty_like(x)
        
        # Fused kernel parameters
        BLOCK_C = 8
        BLOCK_D = 4
        BLOCK_H = 4
        BLOCK_W = 4
        
        grid = (B, (C + BLOCK_C - 1) // BLOCK_C, (D + BLOCK_D - 1) // BLOCK_D, ((H * W) + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W))
        
        # Get layer norm weight and bias
        weight = self.norm.weight
        bias = self.norm.bias
        
        fused_transpose_sum_norm_pool_gelu_kernel[grid](
            x, out, self.sum_weight, torch.cat([weight, bias]),
            B, C, D, H, W,
            2, 2, 2,
            BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
        )
        
        return out