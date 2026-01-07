import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_swish_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def swish_kernel(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < M and x < N:
                    softplus = T.log(1 + T.exp(A[y, x]))
                    sigmoid = T.tanh(softplus)
                    C[y, x] = A[y, x] * sigmoid

    return tilelang.compile(swish_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_swish_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv(x)
        x_c = x.contiguous()
        
        # Get original shape for reshaping output
        original_shape = x_c.shape
        
        x_c = x_c.view(-1, x_c.size(-1))

        M, N = x_c.shape
        kernel = self._get_kernel(M, N, "float16")
        x_half = x_c.to(torch.float16)
        C = kernel(x_half)
        x = C.view(original_shape).to(x.dtype)
        
        x = self.bn(x)
        return x


batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda().half()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]