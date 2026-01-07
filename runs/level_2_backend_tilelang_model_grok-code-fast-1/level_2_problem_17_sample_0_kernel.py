import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_instance_norm_divide_kernel(batch: int, out_c: int, h: int, w: int, block_B: int = 1, block_OC: int = 1, block_H: int = 8, block_W: int = 8, threads: int = 128, dtype: str = "float16"):
    eps = 1e-5

    @T.prim_func
    def fused_instance_norm_divide_kernel(
        A: T.Tensor((batch, out_c, h, w), dtype),
        Gamma: T.Tensor((out_c,), dtype),
        Beta: T.Tensor((out_c,), dtype),
        divide_by: T.Tensor((1,), dtype),
        Y: T.Tensor((batch, out_c, h, w), dtype),
    ):
        mean = T.alloc((batch, out_c), dtype)
        var = T.alloc((batch, out_c), dtype)

        with T.Kernel(T.ceildiv(batch, block_B), T.ceildiv(out_c, block_OC), T.ceildiv(h, block_H), T.ceildiv(w, block_W), threads=threads) as (bx, by, bz, bw):
            # Compute mean and var
            for b, oc in T.grid(batch, out_c):
                mean[b, oc] = T.reduce(T.sum, [2, 3], A[b, oc, :, :]) / (h * w)
                var[b, oc] = T.reduce(T.sum, [2, 3], A[b, oc, :, :] * A[b, oc, :, :]) / (h * w) - mean[b, oc] * mean[b, oc]

            # Normalization and divide
            start_b = bx * block_B
            start_oc = by * block_OC
            start_h = bz * block_H
            start_w = bw * block_W

            for local_b, local_oc, local_h, local_w in T.Parallel(block_B, block_OC, block_H, block_W):
                b = start_b + local_b
                oc = start_oc + local_oc
                oh = start_h + local_h
                ow = start_w + local_w

                if b < batch and oc < out_c and oh < h and ow < w:
                    Y[b, oc, oh, ow] = ((A[b, oc, oh, ow] - mean[b, oc]) / T.sqrt(var[b, oc] + eps) * Gamma[oc] + Beta[oc]) / divide_by[0]

    return tilelang.compile(fused_instance_norm_divide_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).half()
        self.instance_norm = nn.InstanceNorm2d(out_channels).half()
        self.divide_by = divide_by
        self.gamma = self.instance_norm.weight
        self.beta = self.instance_norm.bias
        self._kernel_cache = {}

    def _get_kernel(self, batch: int, out_c: int, h: int, w: int, tl_dtype: str):
        key = (batch, out_c, h, w, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_instance_norm_divide_kernel(batch, out_c, h, w, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.half()
        x = self.conv(x)
        divide_by_tensor = torch.tensor([self.divide_by], dtype=torch.float16, device=x.device)
        kernel = self._get_kernel(x.shape[0], x.shape[1], x.shape[2], x.shape[3], "float16")
        y = kernel(x, self.gamma, self.beta, divide_by_tensor)
        return y