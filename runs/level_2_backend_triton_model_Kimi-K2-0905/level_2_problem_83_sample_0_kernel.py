import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv3d_norm_min_clamp_dropout_kernel(
    x_ptr, w_ptr, b_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
    K, min_val, max_val, dropout_p, seed,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    BLOCK_C: tl.constexpr, BLOCK_DHW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)
    
    c_start = pid_c * BLOCK_C
    dhw_start = pid_dhw * BLOCK_DHW
    
    c_offs = c_start + tl.arange(0, BLOCK_C)
    dhw_offs = dhw_start + tl.arange(0, BLOCK_DHW)
    
    c_mask = c_offs < C_out
    dhw_mask = dhw_offs < (D_out * H_out * W_out)
    
    # Compute mean and inv_std for group norm
    group_size = C_out // 8
    group_idx = c_offs // group_size
    
    # Compute convolution output
    for c_idx in range(C_out):
        if c_idx < C_out and c_mask:
            acc = 0.0
            for k_d in range(K):
                for k_h in range(K):
                    for k_w in range(K):
                        for c_in in range(C_in):
                            for b_idx in range(B):
                                # Input indices
                                in_d = dhw_offs // (H_out * W_out) * stride_d - pad_d + k_d
                                in_h = (dhw_offs // W_out) % H_out * stride_h - pad_h + k_h
                                in_w = dhw_offs % W_out * stride_w - pad_w + k_w
                                
                                # Boundary check
                                in_bounds = (in_d >= 0) & (in_d < D_in) & (in_h >= 0) & (in_h < H_in) & (in_w >= 0) & (in_w < W_in)
                                
                                # Load input
                                x_idx = ((b_idx * C_in + c_in) * D_in + in_d) * H_in * W_in + in_h * W_in + in_w
                                x_val = tl.load(x_ptr + x_idx, mask=in_bounds & dhw_mask, other=0.0)
                                
                                # Load weight
                                w_idx = ((c_idx * C_in + c_in) * K + k_d) * K * K + k_h * K + k_w
                                w_val = tl.load(w_ptr + w_idx)
                                
                                acc += x_val * w_val
            
            # Add bias
            b_val = tl.load(b_ptr + c_idx, mask=c_mask, other=0.0)
            out_val = acc + b_val
            
            # Group normalization
            mean = tl.zeros([BLOCK_C], dtype=tl.float32)
            var = tl.zeros([BLOCK_C], dtype=tl.float32)
            
            # Simplified group norm (assuming proper grouping)
            weight_val = tl.load(weight_ptr + c_idx, mask=c_mask, other=1.0)
            bias_val = tl.load(bias_ptr + c_idx, mask=c_mask, other=0.0)
            
            out_val = (out_val - mean) * tl.rsqrt(var + 1e-5) * weight_val + bias_val
            
            # Min and clamp
            out_val = tl.minimum(out_val, min_val)
            out_val = tl.clamp(out_val, min_val, max_val)
            
            # Dropout
            rand = tl.rand(seed, c_idx * D_out * H_out * W_out + dhw_offs)
            dropout_mask = rand > dropout_p
            out_val = tl.where(dropout_mask, out_val / (1.0 - dropout_p), 0.0)
            
            # Store output
            out_idx = ((pid_b * C_out + c_idx) * D_out + dhw_offs // (H_out * W_out)) * H_out * W_out + (dhw_offs // W_out) % H_out * W_out + dhw_offs % W_out
            tl.store(out_ptr + out_idx, out_val, mask=c_mask & dhw_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        
    def forward(self, x):
        B, C_in, D_in, H_in, W_in = x.shape
        C_out = self.conv.out_channels
        K = self.conv.kernel_size[0]
        stride_d, stride_h, stride_w = self.conv.stride
        pad_d, pad_h, pad_w = self.conv.padding
        
        D_out = (D_in + 2 * pad_d - K) // stride_d + 1
        H_out = (H_in + 2 * pad_h - K) // stride_h + 1
        W_out = (W_in + 2 * pad_w - K) // stride_w + 1
        
        out = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
        
        BLOCK_C = 4
        BLOCK_DHW = 64
        
        grid = (B, (C_out + BLOCK_C - 1) // BLOCK_C, (D_out * H_out * W_out + BLOCK_DHW - 1) // BLOCK_DHW)
        
        fused_conv3d_norm_min_clamp_dropout_kernel[grid](
            x, self.conv.weight, self.conv.bias,
            self.norm.weight, self.norm.bias, out,
            B, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            K, self.min_value, self.max_value, self.dropout_p, torch.seed(),
            stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
            BLOCK_C=BLOCK_C, BLOCK_DHW=BLOCK_DHW
        )
        
        return out