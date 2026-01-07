import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv_bn_scale_kernel(batch, in_c, h, w, out_c, kh, kw, oh, ow, block_B=1, block_OC=16, block_OH=8, block_OW=8, threads=128, dtype="float16"):
    
    @T.prim_func
    def conv_bn_scale_kernel(
        Input: T.Tensor((batch, in_c, h, w), dtype),
        Weight: T.Tensor((out_c, in_c, kh, kw), dtype),
        Bias: T.Tensor((out_c,), dtype),
        RunningMean: T.Tensor((out_c,), "float32"),
        RunningVar: T.Tensor((out_c,), "float32"),
        WeightBN: T.Tensor((out_c,), "float32"),
        BiasBN: T.Tensor((out_c,), "float32"),
        Eps: T.float32,
        ScalingFactor: T.float32,
        Output: T.Tensor((batch, out_c, oh, ow), dtype),
    ):
        with T.Kernel(T.ceildiv(batch, block_B), T.ceildiv(out_c, block_OC), T.ceildiv(oh, block_OH), T.ceildiv(ow, block_OW), threads=threads) as (bb, boc, boh, bow):
            start_b = bb * block_B
            start_oc = boc * block_OC
            start_oh = boh * block_OH
            start_ow = bow * block_OW

            for local_b, local_oc, local_oh, local_ow in T.Parallel(block_B, block_OC, block_OH, block_OW):
                b = start_b + local_b
                oc = start_oc + local_oc
                oy = start_oh + local_oh
                ox = start_ow + local_ow

                if b < batch and oc < out_c and oy < oh and ox < ow:
                    conv_sum = T.cast(0.0, dtype)
                    for ic in range(in_c):
                        for ky in range(kh):
                            for kx in range(kw):
                                if oy + ky < h and ox + kx < w:
                                    conv_sum += Input[b, ic, oy + ky, ox + kx] * Weight[oc, ic, ky, kx]
                    conv_sum += Bias[oc]

                    mean = RunningMean[oc]
                    var = RunningVar[oc]
                    inv_std = 1.0 / T.sqrt(var + Eps)
                    bn_weight = WeightBN[oc]
                    bn_bias = BiasBN[oc]
                    normalized = (conv_sum - mean) * inv_std * bn_weight + bn_bias
                    Output[b, oc, oy, ox] = normalized * ScalingFactor

    return tilelang.compile(conv_bn_scale_kernel, out_idx=[9], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model that fuses convolution, batch normalization, and scaling into a single TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

        # Compute output shape using a dummy forward pass
        dummy = torch.zeros(1, in_channels, 128, 128)
        out = self.conv(dummy)
        batch_size, out_c, oh, ow = out.shape
        self.kernel = build_conv_bn_scale_kernel(batch_size, in_channels, 128, 128, out_channels, kernel_size, kernel_size, oh, ow, dtype="float16")

    def forward(self, x):
        # Ensure input is FP16
        x = x.half()
        return self.kernel(x, self.conv.weight.half(), self.conv.bias.half(), self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, self.bn.eps, self.scaling_factor)