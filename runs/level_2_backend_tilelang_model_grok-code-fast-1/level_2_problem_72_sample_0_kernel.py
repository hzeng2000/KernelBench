import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_avg_pool_3d_kernel(dtype: str = "float16"):
    @T.prim_func
    def avg_pool_3d_kernel(
        A: T.Tensor((64, 16, 63, 63, 63), dtype),
        C: T.Tensor((64, 16, 15, 15, 15), dtype),
    ):
        with T.Kernel(64, 16, 15, 15, 15, threads=256) as (bb, bc, bd, bh, bw):
            b = bb
            c = bc
            d_out = bd
            h_out = bh
            w_out = bw
            sum_val = T.float16(0.0)
            for kd in T.serial(4):
                for kh in T.serial(4):
                    for kw in T.serial(4):
                        sum_val = sum_val + A[b, c, d_out * 4 + kd, h_out * 4 + kh, w_out * 4 + kw]
            C[b, c, d_out, h_out, w_out] = sum_val / T.float16(64.0)

    return tilelang.compile(avg_pool_3d_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization, 
    and a fused average pooling (equivalent to two average pooling layers).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool_kernel = build_avg_pool_3d_kernel("float16")

    def forward(self, x):
        x = x.half()
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.avg_pool_kernel(x)
        return x