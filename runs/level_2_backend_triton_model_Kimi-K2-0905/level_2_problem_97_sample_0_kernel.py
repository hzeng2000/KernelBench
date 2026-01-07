import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_bn_bias_div_swish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    add_bias_ptr, div_val_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w_ptrs = w_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_x = (rm[:, None] < M) & (rk[None, :] < K)
        mask_w = (rk[:, None] < K) & (rn[None, :] < N)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # BN: normalize
    mean = tl.load(running_mean_ptr + rn, mask=rn < N, other=0.0)
    var = tl.load(running_var_ptr + rn, mask=rn < N, other=1.0)
    gamma = tl.load(weight_ptr + rn, mask=rn < N, other=1.0)
    beta = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)

    x_bn = (acc - mean[None, :]) / tl.sqrt(var[None, :] + eps)
    x_bn = gamma[None, :] * x_bn + beta[None, :]

    # Add bias
    add_bias = tl.load(add_bias_ptr)
    x_bn = x_bn + add_bias

    # Divide
    div_val = tl.load(div_val_ptr)
    x_bn = x_bn / div_val

    # Swish
    sig = tl.sigmoid(x_bn)
    out = x_bn * sig

    out_ptrs = out_ptr + (rm[:, None] * stride_outm + rn[None, :] * stride_outn)
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, out, mask=mask_out)


def fused_matmul_bn_bias_div_swish(x, w, b, bn_weight, bn_bias, running_mean, running_var, add_bias, div_val, eps):
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()

    M, K = x.shape
    K, N = w.shape

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fused_matmul_bn_bias_div_swish_kernel[grid](
        x, w, b, out,
        running_mean, running_var, bn_weight, bn_bias,
        add_bias, div_val,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        eps=eps,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x):
        w = self.matmul.weight.T
        b = self.matmul.bias
        return fused_matmul_bn_bias_div_swish(
            x, w, b,
            self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var,
            self.bias, torch.tensor(self.divide_value, device=x.device), self.bn.eps
        )