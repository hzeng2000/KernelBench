import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_bias_scale_sigmoid_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gS: cute.Tensor, gY: cute.Tensor,
    N: int, C: int, H: int, W: int, K: int, R: int, S: int, P: int, Q: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_id = tidz * bdimx * bdimy + tidy * bdimx + tidx
    block_id = bidz * cute.arch.grid_dim().x * cute.arch.grid_dim().y + bidy * cute.arch.grid_dim().x + bidx

    total_threads = bdimx * bdimy * bdimz
    total_blocks = cute.arch.grid_dim().x * cute.arch.grid_dim().y * cute.arch.grid_dim().z

    for n in range(N):
        for k in range(K):
            for p in range(P):
                for q in range(Q):
                    elem_idx = n * K * P * Q + k * P * Q + p * Q + q
                    if elem_idx % total_threads == thread_id and elem_idx // total_threads % total_blocks == block_id:
                        acc = 0.0
                        for c in range(C):
                            for r in range(R):
                                for s in range(S):
                                    h_idx = p + r
                                    w_idx = q + s
                                    if h_idx < H and w_idx < W:
                                        acc += gX[n, c, h_idx, w_idx] * gW[k, c, r, s]
                        acc += gB[k, 0, 0]
                        acc *= gS[k, 0, 0]
                        gY[n, k, p, q] = 1.0 / (1.0 + cute.exp(-acc))

@cute.kernel
def group_norm_kernel(
    gX: cute.Tensor, gY: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor,
    N: int, G: int, C: int, H: int, W: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_id = tidz * bdimx * bdimy + tidy * bdimx + tidx
    block_id = bidz * cute.arch.grid_dim().x * cute.arch.grid_dim().y + bidy * cute.arch.grid_dim().x + bidx

    total_threads = bdimx * bdimy * bdimz
    total_blocks = cute.arch.grid_dim().x * cute.arch.grid_dim().y * cute.arch.grid_dim().z

    channels_per_group = C // G

    for n in range(N):
        for g in range(G):
            elem_idx = n * G + g
            if elem_idx % total_threads == thread_id and elem_idx // total_threads % total_blocks == block_id:
                mean = 0.0
                count = 0
                for c in range(g * channels_per_group, (g + 1) * channels_per_group):
                    for h in range(H):
                        for w in range(W):
                            mean += gX[n, c, h, w]
                            count += 1
                mean /= count

                var = 0.0
                for c in range(g * channels_per_group, (g + 1) * channels_per_group):
                    for h in range(H):
                        for w in range(W):
                            diff = gX[n, c, h, w] - mean
                            var += diff * diff
                var /= count
                inv_std = cute.rsqrt(var + 1e-5)

                for c in range(g * channels_per_group, (g + 1) * channels_per_group):
                    for h in range(H):
                        for w in range(W):
                            gY[n, c, h, w] = (gX[n, c, h, w] - mean) * inv_std

@cute.jit
def conv_bias_scale_sigmoid_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mS: cute.Tensor, mY: cute.Tensor,
    N: int, C: int, H: int, W: int, K: int, R: int, S: int, P: int, Q: int
):
    threads_per_block = 256
    total_elems = N * K * P * Q
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    conv_bias_scale_sigmoid_kernel(mX, mW, mB, mS, mY, N, C, H, W, K, R, S, P, Q).launch(
        grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1)
    )

@cute.jit
def group_norm_host(
    mX: cute.Tensor, mY: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor,
    N: int, G: int, C: int, H: int, W: int
):
    threads_per_block = 256
    total_groups = N * G
    grid_x = cute.ceil_div(total_groups, threads_per_block)

    group_norm_kernel(mX, mY, mMean, mVar, N, G, C, H, W).launch(
        grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1)
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.compiled = {}

    def forward(self, x):
        N, C, H, W = x.shape
        K = self.out_channels
        R = S = self.kernel_size
        P = Q = H - R + 1

        x = x.contiguous().cuda()
        conv_out = torch.empty((N, K, P, Q), dtype=x.dtype, device=x.device)
        norm_out = torch.empty_like(conv_out)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(self.conv_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mS = from_dlpack(self.scale, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mNormOut = from_dlpack(norm_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        mean = torch.zeros((N, self.num_groups), dtype=x.dtype, device=x.device)
        var = torch.zeros_like(mean)
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled_conv = cute.compile(conv_bias_scale_sigmoid_host, mX, mW, mB, mS, mConvOut, N, C, H, W, K, R, S, P, Q)
            compiled_norm = cute.compile(group_norm_host, mConvOut, mNormOut, mMean, mVar, N, self.num_groups, K, P, Q)
            self.compiled[key] = (compiled_conv, compiled_norm)
            compiled = (compiled_conv, compiled_norm)

        compiled[0](mX, mW, mB, mS, mConvOut, N, C, H, W, K, R, S, P, Q)
        compiled[1](mConvOut, mNormOut, mMean, mVar, N, self.num_groups, K, P, Q)

        return norm_out