import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_div_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_bn,
    stride_outm, stride_outn,
    divisor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if b_ptr is not None:
        b = tl.load(b_ptr + rn * stride_bn, mask=rn < N, other=0.0)
        acc += b[None, :]

    acc = acc / divisor

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_acc = acc
    pi = 3.141592653589793
    coeff = tl.sqrt(2.0 / pi)
    x_cubed = x_acc * x_acc * x_acc
    tanh_arg = coeff * (x_acc + 0.044715 * x_cubed)
    tanh_val = tl.tanh(tanh_arg)
    out = 0.5 * x_acc * (1.0 + tanh_val)

    out_ptrs = out_ptr + (rm[:, None] * stride_outm + rn[None, :] * stride_outn)
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, out, mask=mask_out)


def triton_linear_div_gelu(x, weight, bias, divisor):
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    M, K = x.shape
    K_w, N = weight.shape
    assert K == K_w
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = lambda META: ((M + META['BLOCK_M'] - 1) // META['BLOCK_M'],
                         (N + META['BLOCK_N'] - 1) // META['BLOCK_N'])

    linear_div_gelu_kernel[grid](
        x, weight, bias, out,
        M, K, N,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        bias.stride(0) if bias is not None else 0,
        out.stride(0), out.stride(1),
        divisor,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses linear, division, and GELU into a single Triton kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        return triton_linear_div_gelu(x, self.linear.weight, self.linear.bias, self.divisor)