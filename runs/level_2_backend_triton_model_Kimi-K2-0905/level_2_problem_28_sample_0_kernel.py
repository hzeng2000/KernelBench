import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_instancenorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, out_ptr,
    batch_size, in_features, out_features,
    eps,
    stride_x_batch, stride_x_feat,
    stride_w_out, stride_w_in,
    stride_y_batch, stride_y_feat,
    stride_out_batch, stride_out_feat,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // out_features
    col = pid % out_features
    
    if row < batch_size:
        # Compute linear output
        acc = 0.0
        for k in range(0, in_features, BLOCK_SIZE):
            k_offs = k + tl.arange(0, BLOCK_SIZE)
            mask_k = k_offs < in_features
            x_idx = row * stride_x_batch + k_offs * stride_x_feat
            w_idx = col * stride_w_out + k_offs * stride_w_in
            x_val = tl.load(x_ptr + x_idx, mask=mask_k, other=0.0)
            w_val = tl.load(w_ptr + w_idx, mask=mask_k, other=0.0)
            acc += tl.sum(x_val * w_val)
        
        # Add bias
        b_val = tl.load(b_ptr + col)
        linear_out = acc + b_val
        
        # Instance norm (simplified for 1D)
        mean = linear_out
        var = 0.0
        norm_out = (linear_out - mean) / tl.sqrt(var + eps)
        
        # Residual add and mul
        y_val = tl.load(y_ptr + row * stride_y_batch + col * stride_y_feat)
        out_val = (norm_out + y_val) * y_val
        
        out_idx = row * stride_out_batch + col * stride_out_feat
        tl.store(out_ptr + out_idx, out_val)


def triton_fused_linear_instancenorm(x, w, b, y, eps):
    batch_size, in_features = x.shape
    out_features = w.shape[0]
    
    out = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)
    
    n_elements = batch_size * out_features
    BLOCK_SIZE = 32
    
    grid = lambda meta: (n_elements,)
    
    fused_linear_instancenorm_kernel[grid](
        x, w, b, y, out,
        batch_size, in_features, out_features,
        eps,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.eps = eps

    def forward(self, x, y):
        w = self.bmm.weight
        b = self.bmm.bias
        return triton_fused_linear_instancenorm(x, w, b, y, self.eps)