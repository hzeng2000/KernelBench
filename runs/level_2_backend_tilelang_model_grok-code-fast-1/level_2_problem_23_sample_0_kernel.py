import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_mean_kernel(B: int, C: int, D: int, H: int, W: int, dtype: str = "float16"):
    num = C * D * H * W

    @T.prim_func
    def mean_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        C_out: T.Tensor((B,), dtype),
    ):
        with T.Kernel(B, threads=128) as bx:
            ty = T.thread_axis("threadIdx.x")
            partial_sum = T.shared(dtype, (128,))
            sum_val = T.alloc_var(dtype, ())
            sum_val[()] = T.cast(0, dtype)
            for i in T.serial(T.ceildiv(num, 128)):
                idx = ty + i * 128
                if idx < num:
                    c = idx // (D * H * W)
                    rem = idx % (D * H * W)
                    d = rem // (H * W)
                    rem2 = rem % (H * W)
                    h = rem2 // W
                    w = rem2 % W
                    sum_val[()] += A[bx, c, d, h, w]
            partial_sum[ty] = sum_val[()]
            T.sync()
            # Tree reduction
            for stride in [64, 32, 16, 8, 4, 2, 1]:
                if ty < stride:
                    partial_sum[ty] += partial_sum[ty + stride]
                T.sync()
            if ty == 0:
                C_out[bx] = partial_sum[0] / T.cast(num, dtype)

    return tilelang.compile(mean_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = (B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_mean_kernel(B, C, D, H, W, tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x_c = x.contiguous().half()
        B, C, D, H, W = x_c.shape
        kernel = self._get_kernel(B, C, D, H, W, "float16")
        out = kernel(x_c)
        return out.float()