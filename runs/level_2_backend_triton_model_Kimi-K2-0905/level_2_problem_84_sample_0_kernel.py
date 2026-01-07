import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_bn_scale_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    scale_ptr,
    M, N, K,
    eps,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w_ptrs = w_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_x = (rm[:, None] < M) & (rk[None, :] < K - k)
        mask_w = (rk[:, None] < K - k) & (rn[None, :] < N)
        x_val = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_val = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(x_val, w_val)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b_val = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
    acc += b_val[None, :]

    # BatchNorm inference: (x - mean) / sqrt(var + eps) * gamma + beta
    mean = tl.load(running_mean_ptr + rn, mask=rn < N, other=0.0)
    var = tl.load(running_var_ptr + rn, mask=rn < N, other=0.0)
    gamma = tl.load(weight_ptr + rn, mask=rn < N, other=1.0)
    beta = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
    bn_out = (acc - mean[None, :]) / tl.sqrt(var[None, :] + eps) * gamma[None, :] + beta[None, :]

    # Scale
    scale_val = tl.load(scale_ptr)
    scaled = bn_out * scale_val

    # Online softmax
    row_max = tl.max(scaled, axis=1)
    row_max = tl.where(rm < M, row_max, float('-inf'))
    exp = tl.exp(scaled - row_max[:, None])
    row_sum = tl.sum(exp, axis=1)
    softmax_out = exp / row_sum[:, None]

    # Store output
    out_ptrs = out_ptr + (rm[:, None] * stride_outm + rn[None, :] * stride_outn)
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, softmax_out, mask=mask_out)


def fused_gemm_bn_scale_softmax(x, w, b, running_mean, running_var, weight, bias, scale):
    M, K = x.shape
    N = w.shape[0]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    gemm_bn_scale_softmax_kernel[grid](
        x, w, b, out,
        running_mean, running_var, weight, bias,
        scale,
        M, N, K,
        1e-5,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        w = self.gemm.weight
        b = self.gemm.bias
        rm = self.bn.running_mean
        rv = self.bn.running_var
        bn_w = self.bn.weight
        bn_b = self.bn.bias
        return fused_gemm_bn_scale_softmax(x, w, b, rm, rv, bn_w, bn_b, self.scale)