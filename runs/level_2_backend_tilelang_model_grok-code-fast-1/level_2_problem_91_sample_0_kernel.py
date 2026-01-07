import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_softmax_bias_scale_sigmoid_kernel(B: int, C: int, H: int, W: int, block_B: int = 1, block_H: int = 1, block_W: int = 1, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_softmax_bias_scale_sigmoid_kernel(
        X: T.Tensor((B, C, H, W), dtype),
        Bias: T.Tensor((C, 1, 1), dtype),
        Scale: T.Tensor((), dtype),
        Y: T.Tensor((B, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(H, block_H), T.ceildiv(W, block_W), threads=threads) as (bx, by, bz):
            b = bx * block_B
            h = by * block_H
            w = bz * block_W
            
            shared = T.alloc_shared((C,), dtype)
            tid = T.get_thread_binding()
            shared[tid] = X[b, tid, h, w]
            T.sync()
            
            max_val = T.reduce(T.max, T.min_value(dtype), lambda i: shared[i])
            exp_val = T.exp(shared[tid] - max_val)
            sum_exp = T.reduce(T.add, T.cast(0.0, dtype), lambda i: T.exp(shared[i] - max_val))
            softmax_val = exp_val / sum_exp
            y = softmax_val + Bias[tid, 0, 0]
            y = y * Scale[()]
            y = 1.0 / (1.0 + T.exp(-y))
            Y[b, tid, h, w] = y

    return tilelang.compile(fused_softmax_bias_scale_sigmoid_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then applies a fused kernel for softmax, bias add, scaling, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, H: int, W: int, tl_dtype: str):
        key = (B, C, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_softmax_bias_scale_sigmoid_kernel(B, C, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.half()
        bias = self.bias.half()
        scale = torch.tensor(self.scaling_factor, dtype=torch.float16, device=x.device)
        B, C, H, W = x.shape
        kernel = self._get_kernel(B, C, H, W, "float16")
        y = kernel(x, bias, scale)
        return y.float()  # Cast back to float to match original output dtype