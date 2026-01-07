import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_relu_bias_kernel(B: int, O: int, H: int, W: int, block_B: int = 1, block_O: int = 4, block_HW: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def relu_bias_kernel(
        A: T.Tensor((B, O, H, W), dtype),
        bias: T.Tensor((O,), dtype),
        C: T.Tensor((B, O, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(O, block_O), T.ceildiv(B, block_B), T.ceildiv(H * W, block_HW), threads=threads) as (bx, by, bz):
            start_o = bx * block_O
            start_b = by * block_B
            start_hw = bz * block_HW

            for local_b, local_o, local_hw in T.Parallel(block_B, block_O, block_HW):
                b = start_b + local_b
                o = start_o + local_o
                hw = start_hw + local_hw
                y = hw // W
                x = hw % W

                if b < B and o < O and y < H and x < W:
                    C[b, o, y, x] = T.max(A[b, o, y, x], T.float16(0)) + bias[o]

    return tilelang.compile(relu_bias_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies ReLU, and adds a bias term.
    The ReLU and bias addition are fused into a custom TileLang kernel for speedup.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.kernel_size = kernel_size
        self._kernel_cache = {}

    def _get_kernel(self, B: int, O: int, H: int, W: int, tl_dtype: str):
        key = (B, O, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_relu_bias_kernel(B, O, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv(x)
        # Compute output height and width
        H_out = x.size(2)
        W_out = x.size(3)
        B, O = x.size(0), x.size(1)
        
        # Ensure FP16
        x = x.half()
        bias = self.bias.view(O).half()
        
        kernel = self._get_kernel(B, O, H_out, W_out, "float16")
        x = kernel(x, bias)
        
        return x