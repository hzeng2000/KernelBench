import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv3d_hardswish_kernel(
    B: int, IC: int, ID: int, IH: int, IW: int, OC: int, OD: int, OH: int, OW: int, KD: int, KH: int, KW: int,
    block_B: int = 1, block_OC: int = 16, block_OD: int = 1, block_OH: int = 1, block_OW: int = 32, threads: int = 32, dtype: str = "float16"
):
    
    @T.prim_func
    def conv3d_hardswish_kernel(
        Input: T.Tensor((B, IC, ID, IH, IW), dtype),
        Weight: T.Tensor((OC, IC, KD, KH, KW), dtype),
        Bias: T.Tensor((OC,), dtype),
        Output: T.Tensor((B, OC, OD, OH, OW), dtype),
    ):
        with T.Kernel(
            T.ceildiv(OW, block_OW), T.ceildiv(OH, block_OH), T.ceildiv(OD, block_OD), T.ceildiv(OC, block_OC), T.ceildiv(B, block_B), 
            threads=threads
        ) as (bx, by, bz, boc, bb):
            start_ow = bx * block_OW
            start_oh = by * block_OH
            start_od = bz * block_OD
            start_oc = boc * block_OC
            start_b = bb * block_B

            for local_ow, local_oh, local_od, local_oc, local_b in T.Parallel(block_OW, block_OH, block_OD, block_OC, block_B):
                ow = start_ow + local_ow
                oh = start_oh + local_oh
                od = start_od + local_od
                oc = start_oc + local_oc
                b = start_b + local_b

                if ow < OW and oh < OH and od < OD and oc < OC and b < B:
                    reduce_ic = T.reduce_axis(0, IC)
                    reduce_kd = T.reduce_axis(0, KD)
                    reduce_kh = T.reduce_axis(0, KH)
                    reduce_kw = T.reduce_axis(0, KW)
                    
                    conv_sum = T.reduce_sum(
                        Input[b, reduce_ic, od + reduce_kd, oh + reduce_kh, ow + reduce_kw] *
                        Weight[oc, reduce_ic, reduce_kd, reduce_kh, reduce_kw],
                        axis=[reduce_ic, reduce_kd, reduce_kh, reduce_kw]
                    ) + Bias[oc]
                    
                    # HardSwish: x * relu6(x + 3) / 6
                    x = conv_sum
                    relu6 = T.min(T.max(x + T.float16(3.0), T.float16(0.0)), T.float16(6.0))
                    hardswish = x * relu6 / T.float16(6.0)
                    
                    Output[b, oc, od, oh, ow] = hardswish

    return tilelang.compile(conv3d_hardswish_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs:
    1. Conv3D + HardSwish fused in TileLang kernel
    2. GroupNorm  
    3. Mean pooling across spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self._kernel_cache = {}

    def _get_kernel(self, B: int, IC: int, ID: int, IH: int, IW: int, OC: int, OD: int, OH: int, OW: int, KD: int, KH: int, KW: int, tl_dtype: str):
        key = (B, IC, ID, IH, IW, OC, OD, OH, OW, KD, KH, KW, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv3d_hardswish_kernel(B, IC, ID, IH, IW, OC, OD, OH, OW, KD, KH, KW, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x_c = x.contiguous().half()
        w_c = self.conv.weight.contiguous().half()
        b_c = self.conv.bias.contiguous().half()
        
        B, IC, ID, IH, IW = x_c.shape
        OC, _, KD, KH, KW = w_c.shape
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        
        kernel = self._get_kernel(B, IC, ID, IH, IW, OC, OD, OH, OW, KD, KH, KW, "float16")
        x = kernel(x_c, w_c, b_c)
        
        x = self.group_norm(x.float())  # Convert back to float for GroupNorm
        x = torch.mean(x, dim=[2, 3, 4])  # Mean over spatial dims → (B, C)
        return x