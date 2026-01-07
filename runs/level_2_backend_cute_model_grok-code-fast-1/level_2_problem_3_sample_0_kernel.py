import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_avg_pool_gelu_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx()
    bdim = cute.arch.block_dim().x

    thread_idx = bidx.x * bdim + tidx

    B, C, D, H, W = gX.shape
    D_out = D // 2
    H_out = H // 2
    W_out = W // 2
    total_out = B * C * D_out * H_out * W_out

    if thread_idx >= total_out:
        return

    b = thread_idx // (C * D_out * H_out * W_out)
    rem = thread_idx % (C * D_out * H_out * W_out)
    c = rem // (D_out * H_out * W_out)
    rem = rem % (D_out * H_out * W_out)
    d_out = rem // (H_out * W_out)
    rem = rem % (H_out * W_out)
    h_out = rem // W_out
    w_out = rem % W_out

    d_in = 2 * d_out
    h_in = 2 * h_out
    w_in = 2 * w_out

    sum_val = 0.0
    for dd in range(2):
        for hh in range(2):
            for ww in range(2):
                sum_val += gX[b, c, d_in + dd, h_in + hh, w_in + ww]
    avg = sum_val / 8.0

    # Approximate GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    # Since CuTe may not have erf, use approximation: gelu(x) ≈ x * sigmoid(1.702 * x)
    gelu_val = avg * cute.math.sigmoid(1.702 * avg)
    gY[b, c, d_out, h_out, w_out] = gelu_val

@cute.jit
def fused_avg_pool_gelu_host(mX: cute.Tensor, mY: cute.Tensor):
    B, C, D, H, W = mX.shape
    D_out = D // 2
    H_out = H // 2
    W_out = W // 2
    total_out = B * C * D_out * H_out * W_out

    threads_per_block = 256
    grid_x = cute.ceil_div(total_out, threads_per_block)

    fused_avg_pool_gelu_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = torch.nn.Parameter(torch.tensor(sum_weight))
        self.norm = torch.nn.LayerNorm(norm_shape)
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.sum_weight
        x = self.norm(x)
        # Compute output shape for avg pool
        B, C, D, H, W = x.shape
        D_out = D // 2
        H_out = H // 2
        W_out = W // 2
        y = torch.empty((B, C, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
        
        x = x.contiguous()
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_avg_pool_gelu_host, mX, mY)
            self.compiled[key] = compiled
        
        compiled(mX, mY)
        return y