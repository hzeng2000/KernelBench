import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(B: int, C: int, D: int, H: int, W: int, block_B: int = 1, block_C: int = 128, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((B, C, D, H, W), dtype),
        scale_factor: T.Tensor((), "float32"),
        running_mean: T.Tensor((C,), "float32"),
        running_var: T.Tensor((C,), "float32"),
        gamma: T.Tensor((C,), "float32"),
        beta: T.Tensor((C,), "float32"),
        eps: T.Tensor((), "float32"),
        Y: T.Tensor((B, C, 1, 1, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(C, block_C), threads=threads) as (bx_b, bx_c):
            b = bx_b * block_B + T.thread_binding(0, block_B, "threadIdx.y")
            c = bx_c * block_C + T.thread_binding(0, block_C, "threadIdx.x")
            
            if b < B and c < C:
                Y[b, c, 0, 0, 0] = T.reduce(
                    lambda d, h, w: (X[b, c, d, h, w] * scale_factor[()] - running_mean[c]) / T.sqrt(running_var[c] + eps[()]) * gamma[c] + beta[c],
                    T.reduce_axis(0, D, "d"),
                    T.reduce_axis(0, H, "h"),
                    T.reduce_axis(0, W, "w"),
                    init=0.0,
                    op=T.Add
                ) / (D * H * W)
    
    return tilelang.compile(fused_kernel, out_idx=[7], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, then fuses scaling, batch normalization, 
    and global average pooling into a single TileLang kernel for speedup, optimized for FP16.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size).half()
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum).half()
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = (B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.half()
        x = self.conv_transpose(x)
        B, C, D, H, W = x.shape
        kernel = self._get_kernel(B, C, D, H, W, "float16")
        scale_tensor = torch.tensor(self.scale_factor, dtype=torch.float32, device=x.device)
        eps_tensor = torch.tensor(self.batch_norm.eps, dtype=torch.float32, device=x.device)
        y = kernel(x, scale_tensor, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.weight, self.batch_norm.bias, eps_tensor)
        return y