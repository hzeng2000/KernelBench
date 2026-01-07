import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_dropout_softmax_kernel(
    x_ptr, w_ptr, out_ptr,
    dropout_seed, dropout_p,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    # Apply dropout
    dropout_offs = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * N + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :])
    dropout_mask = dropout_offs < M * N
    rand_vals = tl.rand(dropout_seed, dropout_offs)
    dropout_scale = 1.0 / (1.0 - dropout_p)
    acc = tl.where(rand_vals > dropout_p, acc * dropout_scale, 0.0)

    # Softmax along rows
    row_max = tl.max(acc, axis=1)[:, None]
    acc = acc - row_max
    acc = tl.exp(acc)
    row_sum = tl.sum(acc, axis=1)[:, None]
    acc = acc / row_sum

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn, acc, mask=out_mask)


def triton_matmul_dropout_softmax(x: torch.Tensor, w: torch.Tensor, dropout_p: float):
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32
    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w

    out = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = lambda META: ((M + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M'],
                         (N + META['BLOCK_SIZE_N'] - 1) // META['BLOCK_SIZE_N'])

    seed = torch.randint(0, 2**31, (1,)).item()

    matmul_dropout_softmax_kernel[grid](
        x, w, out,
        seed, dropout_p,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.dropout_p = dropout_p
        nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5.0)))

    def forward(self, x):
        return triton_matmul_dropout_softmax(x, self.weight.t(), self.dropout_p)