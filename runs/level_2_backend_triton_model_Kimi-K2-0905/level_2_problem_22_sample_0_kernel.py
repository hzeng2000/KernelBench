import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, input_size, hidden_size,
    scale_factor, clamp_min, clamp_max,
    stride_xb, stride_xm,
    stride_wb, stride_wn,
    stride_outb, stride_outn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xb)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wb + offs_n[None, :] * stride_wn)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, input_size, BLOCK_SIZE_K):
        mask_x = (offs_m[:, None] < batch_size) & (offs_k[None, :] < input_size)
        mask_w = (offs_k[:, None] < input_size) & (offs_n[None, :] < hidden_size)
        
        x_chunk = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_chunk = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc += tl.dot(x_chunk, w_chunk)
        
        x_ptrs += BLOCK_SIZE_K * stride_xb
        w_ptrs += BLOCK_SIZE_K * stride_wb
    
    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n, mask=offs_n < hidden_size, other=0.0)
        acc = acc + b[None, :]
    
    acc = acc * scale_factor
    acc = acc + acc
    acc = tl.clamp(acc, clamp_min, clamp_max)
    
    row_max = tl.max(acc, axis=1)
    row_sum = tl.sum(tl.exp(acc - row_max[:, None]), axis=1)
    lse = row_max + tl.log(row_sum)
    
    mish_in = lse[:, None]
    mish_out = mish_in * tl.tanh(tl.softplus(mish_in))
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_outn
    tl.store(out_ptrs, mish_out, mask=offs_m[:, None] < batch_size)


def triton_forward(x, w, b, scale_factor, clamp_min, clamp_max):
    assert x.is_cuda and w.is_cuda
    batch_size, input_size = x.shape
    hidden_size, _ = w.shape
    
    out = torch.empty((batch_size, 1), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    grid = ((batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, 1)
    
    fused_kernel[grid](
        x, w, b, out,
        batch_size, input_size, hidden_size,
        scale_factor, clamp_min, clamp_max,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        return triton_forward(x, self.matmul.weight, self.matmul.bias,
                              self.scale_factor, self.clamp_min, self.clamp_max)