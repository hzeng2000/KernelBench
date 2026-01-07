import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose3d_kernel(
    batch_size, in_channels, out_channels, 
    in_d, in_h, in_w, out_d, out_h, out_w,
    kernel_size, stride, padding, output_padding,
    block_d=4, block_h=4, block_w=4, threads=256, dtype="float16"
):
    
    @T.prim_func
    def conv_transpose3d_kernel(
        Input: T.Tensor((batch_size, in_channels, in_d, in_h, in_w), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size, out_channels, out_d, out_h, out_w), dtype),
    ):
        with T.Kernel(T.ceildiv(out_w, block_w), T.ceildiv(out_h, block_h), 
                     T.ceildiv(out_d, block_d), out_channels, batch_size, threads=threads) as (bx, by, bz, c, n):
            
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            
            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                
                if d < out_d and h < out_h and w < out_w and c < out_channels and n < batch_size:
                    acc = 0.0
                    
                    for ic in range(in_channels):
                        for kd in range(kernel_size):
                            for kh in range(kernel_size):
                                for kw in range(kernel_size):
                                    # Calculate input coordinates
                                    in_d_coord = (d + padding - kd * 1 - output_padding) // stride
                                    in_h_coord = (h + padding - kh * 1 - output_padding) // stride
                                    in_w_coord = (w + padding - kw * 1 - output_padding) // stride
                                    
                                    # Check if coordinates are valid
                                    if (in_d_coord >= 0 and in_d_coord < in_d and
                                        in_h_coord >= 0 and in_h_coord < in_h and
                                        in_w_coord >= 0 and in_w_coord < in_w and
                                        (d + padding - kd - output_padding) % stride == 0 and
                                        (h + padding - kh - output_padding) % stride == 0 and
                                        (w + padding - kw - output_padding) % stride == 0):
                                        
                                        acc += Input[n, ic, in_d_coord, in_h_coord, in_w_coord] * Weight[ic, c, kd, kh, kw]
                    
                    Output[n, c, d, h, w] = acc + Bias[c]

    return tilelang.compile(conv_transpose3d_kernel, out_idx=[3], target="cuda")


def build_activation_fused_kernel(
    batch_size, channels, d, h, w,
    block_d=4, block_h=4, block_w=4, threads=256, dtype="float16"
):
    
    @T.prim_func
    def activation_fused_kernel(
        Input: T.Tensor((batch_size, channels, d, h, w), dtype),
        Output: T.Tensor((batch_size, channels, d, h, w), dtype),
    ):
        with T.Kernel(T.ceildiv(w, block_w), T.ceildiv(h, block_h), 
                     T.ceildiv(d, block_d), channels, batch_size, threads=threads) as (bx, by, bz, c, n):
            
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            
            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                d_coord = start_d + local_d
                h_coord = start_h + local_h
                w_coord = start_w + local_w
                
                if d_coord < d and h_coord < h and w_coord < w and c < channels and n < batch_size:
                    val = Input[n, c, d_coord, h_coord, w_coord]
                    
                    # Softmax along channel dimension (dim=1)
                    # For numerical stability, subtract max
                    max_val = val
                    exp_sum = 0.0
                    
                    # Compute max for softmax
                    for cc in range(channels):
                        if cc == c:
                            max_val = T.max(max_val, Input[n, cc, d_coord, h_coord, w_coord])
                    
                    # Compute exp and sum
                    for cc in range(channels):
                        exp_val = T.exp(Input[n, cc, d_coord, h_coord, w_coord] - max_val)
                        if cc == c:
                            exp_sum += exp_val
                    
                    # Apply softmax
                    softmax_val = T.exp(val - max_val) / exp_sum
                    
                    # Apply sigmoid
                    sigmoid_val = 1.0 / (1.0 + T.exp(-softmax_val))
                    
                    Output[n, c, d_coord, h_coord, w_coord] = sigmoid_val

    return tilelang.compile(activation_fused_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                               stride=stride, padding=padding, 
                                               output_padding=output_padding, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self._kernel_cache = {}
        
    def _get_conv_kernel(self, batch_size, in_d, in_h, in_w, out_d, out_h, out_w):
        key = (batch_size, in_d, in_h, in_w, out_d, out_h, out_w)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose3d_kernel(
                batch_size, self.in_channels, self.out_channels,
                in_d, in_h, in_w, out_d, out_h, out_w,
                self.kernel_size, self.stride, self.padding, self.output_padding
            )
        return self._kernel_cache[key]
    
    def _get_activation_kernel(self, batch_size, channels, d, h, w):
        key = (batch_size, channels, d, h, w)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_activation_fused_kernel(
                batch_size, channels, d, h, w
            )
        return self._kernel_cache[key]
        
    def forward(self, x):
        batch_size = x.shape[0]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        
        # Calculate output dimensions
        out_d = (in_d - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_h = (in_h - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_w = (in_w - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        # Get weight and bias
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        
        # Convert to fp16
        x_fp16 = x.half()
        weight_fp16 = weight.half()
        bias_fp16 = bias.half()
        
        # Conv transpose kernel
        conv_kernel = self._get_conv_kernel(batch_size, in_d, in_h, in_w, out_d, out_h, out_w)
        conv_out = conv_kernel(x_fp16, weight_fp16, bias_fp16)
        
        # Activation fused kernel (softmax + sigmoid)
        activation_kernel = self._get_activation_kernel(batch_size, self.out_channels, out_d, out_h, out_w)
        final_out = activation_kernel(conv_out)
        
        return final_out