import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_batch_norm_spatial_center_kernel(N: int, C: int, D: int, H: int, W: int, eps: float, block_N: int = 1, block_C: int = 1, block_D: int = 4, block_H: int = 8, block_W: int = 8, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        x: T.Tensor((N, C, D, H, W), dtype),
        running_mean: T.Tensor((C,), dtype),
        running_var: T.Tensor((C,), dtype),
        weight: T.Tensor((C,), dtype),
        bias: T.Tensor((C,), dtype),
        y: T.Tensor((N, C, D, H, W), dtype),
    ):
        temp = T.alloc((N, C, D, H, W), dtype)
        mean_spatial = T.alloc((N, C, 1, 1, 1), dtype)
        
        # BatchNorm into temp
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(C, block_C), T.ceildiv(D, block_D), T.ceildiv(H, block_H), T.ceildiv(W, block_W), threads=threads) as (bx, by, bz, bh, bw):
            start_n = bx * block_N
            start_c = by * block_C
            start_d = bz * block_D
            start_h = bh * block_H
            start_w = bw * block_W
            
            for local_n, local_c, local_d, local_h, local_w in T.Parallel(block_N, block_C, block_D, block_H, block_W):
                n = start_n + local_n
                c = start_c + local_c
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                
                if n < N and c < C and d < D and h < H and w < W:
                    temp[n, c, d, h, w] = (x[n, c, d, h, w] - running_mean[c]) / T.sqrt(running_var[c] + eps) * weight[c] + bias[c]
        
        # Compute spatial mean
        num_spatial = D * H * W
        with T.Kernel(N, C, threads=threads) as (bx, by):
            T.reduce(T.Parallel(num_spatial), lambda i: temp[bx, by, i // (H * W), (i % (H * W)) // W, i % W], T.sum, init=0, result=mean_spatial[bx, by, 0, 0, 0])
            mean_spatial[bx, by, 0, 0, 0] = mean_spatial[bx, by, 0, 0, 0] / num_spatial
        
        # Subtract mean
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(C, block_C), T.ceildiv(D, block_D), T.ceildiv(H, block_H), T.ceildiv(W, block_W), threads=threads) as (bx, by, bz, bh, bw):
            start_n = bx * block_N
            start_c = by * block_C
            start_d = bz * block_D
            start_h = bh * block_H
            start_w = bw * block_W
            
            for local_n, local_c, local_d, local_h, local_w in T.Parallel(block_N, block_C, block_D, block_H, block_W):
                n = start_n + local_n
                c = start_c + local_c
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                
                if n < N and c < C and d < D and h < H and w < W:
                    y[n, c, d, h, w] = temp[n, c, d, h, w] - mean_spatial[n, c, 0, 0, 0]
    
    return tilelang.compile(fused_kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    """
    A 3D convolutional transpose layer followed by fused Batch Normalization and spatial mean subtraction using TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        # Output shape after conv_transpose: N=16, C=32, D=31, H=63, W=63
        self.kernel = build_fused_batch_norm_spatial_center_kernel(16, 32, 31, 63, 63, self.batch_norm.eps, dtype="float16")

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.half()
        running_mean = self.batch_norm.running_mean.half()
        running_var = self.batch_norm.running_var.half()
        weight = self.batch_norm.weight.half()
        bias = self.batch_norm.bias.half()
        y = self.kernel(x, running_mean, running_var, weight, bias)
        return y