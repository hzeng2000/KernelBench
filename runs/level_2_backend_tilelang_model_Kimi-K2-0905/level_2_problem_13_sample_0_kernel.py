import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_kernel(B: int, C: int, D: int, H: int, W: int, block_C: int = 64, block_HW: int = 16, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((B, C, D, H, W), dtype),
        Bias: T.Tensor((1, C, 1, 1, 1), dtype),
        Out: T.Tensor((B, C, 1, H, W), dtype),
        Scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(C, block_C), T.ceildiv(H, block_HW), T.ceildiv(W, block_HW), B, threads=threads) as (bc, bh, bw, bb):
            start_c = bc * block_C
            start_h = bh * block_HW
            start_w = bw * block_HW

            for local_c, local_h, local_w in T.Parallel(block_C, block_HW, block_HW):
                c = start_c + local_c
                h = start_h + local_h
                w = start_w + local_w

                if c < C and h < H and w < W:
                    # Mean pooling over depth
                    sum_val = T.allocate([1], dtype, "local")
                    sum_val[0] = T.cast(0.0, dtype)
                    
                    for d in range(D):
                        sum_val[0] = sum_val[0] + X[bb, c, d, h, w]
                    
                    mean_val = sum_val[0] / T.cast(D, dtype)
                    
                    # Add bias
                    biased = mean_val + Bias[0, c, 0, 0, 0]
                    
                    # Store intermediate result for softmax
                    Out[bb, c, 0, h, w] = biased

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


def build_softmax_kernel(B: int, C: int, H: int, W: int, block_C: int = 64, block_HW: int = 16, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def softmax_kernel(
        X: T.Tensor((B, C, 1, H, W), dtype),
        Out: T.Tensor((B, C, 1, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(H, block_HW), T.ceildiv(W, block_HW), B, threads=threads) as (bh, bw, bb):
            start_h = bh * block_HW
            start_w = bw * block_HW

            # Find max for numerical stability
            max_val = T.allocate([1], dtype, "local")
            max_val[0] = T.cast(-1e9, dtype)
            
            for c in range(C):
                for local_h, local_w in T.Parallel(block_HW, block_HW):
                    h = start_h + local_h
                    w = start_w + local_w
                    if h < H and w < W:
                        if X[bb, c, 0, h, w] > max_val[0]:
                            max_val[0] = X[bb, c, 0, h, w]

            # Compute exp and sum
            sum_exp = T.allocate([1], dtype, "local")
            sum_exp[0] = T.cast(0.0, dtype)
            
            for c in range(C):
                for local_h, local_w in T.Parallel(block_HW, block_HW):
                    h = start_h + local_h
                    w = start_w + local_w
                    if h < H and w < W:
                        exp_val = T.exp(X[bb, c, 0, h, w] - max_val[0])
                        sum_exp[0] = sum_exp[0] + exp_val

            # Compute softmax
            for c in range(C):
                for local_h, local_w in T.Parallel(block_HW, block_HW):
                    h = start_h + local_h
                    w = start_w + local_w
                    if h < H and w < W:
                        exp_val = T.exp(X[bb, c, 0, h, w] - max_val[0])
                        Out[bb, c, 0, h, w] = exp_val / sum_exp[0]

    return tilelang.compile(softmax_kernel, out_idx=[1], target="cuda")


def build_tanh_scale_kernel(B: int, C: int, H: int, W: int, block_C: int = 64, block_HW: int = 16, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def tanh_scale_kernel(
        X: T.Tensor((B, C, 1, H, W), dtype),
        Out: T.Tensor((B, C, 1, H, W), dtype),
        Scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(C, block_C), T.ceildiv(H, block_HW), T.ceildiv(W, block_HW), B, threads=threads) as (bc, bh, bw, bb):
            start_c = bc * block_C
            start_h = bh * block_HW
            start_w = bw * block_HW

            for local_c, local_h, local_w in T.Parallel(block_C, block_HW, block_HW):
                c = start_c + local_c
                h = start_h + local_h
                w = start_w + local_w

                if c < C and h < H and w < W:
                    tanh_val = T.tanh(X[bb, c, 0, h, w])
                    Out[bb, c, 0, h, w] = tanh_val * T.cast(Scale, dtype)

    return tilelang.compile(tanh_scale_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self._fused_kernel_cache = {}
        self._softmax_kernel_cache = {}
        self._tanh_scale_kernel_cache = {}

    def _get_fused_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = (B, C, D, H, W, tl_dtype)
        if key not in self._fused_kernel_cache:
            self._fused_kernel_cache[key] = build_fused_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._fused_kernel_cache[key]

    def _get_softmax_kernel(self, B: int, C: int, H: int, W: int, tl_dtype: str):
        key = (B, C, H, W, tl_dtype)
        if key not in self._softmax_kernel_cache:
            self._softmax_kernel_cache[key] = build_softmax_kernel(B, C, H, W, dtype=tl_dtype)
        return self._softmax_kernel_cache[key]

    def _get_tanh_scale_kernel(self, B: int, C: int, H: int, W: int, tl_dtype: str):
        key = (B, C, H, W, tl_dtype)
        if key not in self._tanh_scale_kernel_cache:
            self._tanh_scale_kernel_cache[key] = build_tanh_scale_kernel(B, C, H, W, dtype=tl_dtype)
        return self._tanh_scale_kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvTranspose3d
        x = self.conv_transpose(x)
        
        B, C, D, H, W = x.shape
        
        # Convert to FP16 for TileLang kernels
        x_fp16 = x.half()
        bias_fp16 = self.bias.half()
        
        # Fused kernel: mean pool + bias add
        fused_kernel = self._get_fused_kernel(B, C, D, H, W, "float16")
        intermediate = fused_kernel(x_fp16, bias_fp16)
        
        # Softmax kernel
        softmax_kernel = self._get_softmax_kernel(B, C, H, W, "float16")
        softmax_out = softmax_kernel(intermediate)
        
        # Tanh + scale kernel
        tanh_scale_kernel = self._get_tanh_scale_kernel(B, C, H, W, "float16")
        final_out = tanh_scale_kernel(softmax_out, self.scaling_factor)
        
        return final_out