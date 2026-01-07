import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose3d_bn_avgpool_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    depth: int,
    height: int,
    width: int,
    kernel_size: int,
    stride: int,
    padding: int,
    block_d: int = 4,
    block_h: int = 4,
    block_w: int = 4,
    threads: int = 128,
    dtype: str = "float16"
):
    out_depth = (depth - 1) * stride - 2 * padding + kernel_size
    out_height = (height - 1) * stride - 2 * padding + kernel_size
    out_width = (width - 1) * stride - 2 * padding + kernel_size
    
    pooled_depth = out_depth // 2 // 2
    pooled_height = out_height // 2 // 2
    pooled_width = out_width // 2 // 2

    @T.prim_func
    def kernel(
        Input: T.Tensor((batch_size, in_channels, depth, height, width), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        RunningMean: T.Tensor((out_channels,), "float32"),
        RunningVar: T.Tensor((out_channels,), "float32"),
        Gamma: T.Tensor((out_channels,), dtype),
        Beta: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size, out_channels, pooled_depth, pooled_height, pooled_width), dtype),
    ):
        with T.Kernel(
            T.ceildiv(pooled_width, block_w),
            T.ceildiv(pooled_height, block_h),
            T.ceildiv(pooled_depth, block_d),
            batch_size,
            out_channels,
            threads=threads
        ) as (bx, by, bz, n, c):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d

            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                pd = start_d + local_d
                ph = start_h + local_h
                pw = start_w + local_w

                if pd < pooled_depth and ph < pooled_height and pw < pooled_width:
                    # Average pooling over 2x2x2 regions (two avgpool layers)
                    sum_val = T.alloc_fragment((1,), dtype, 0)
                    count = 0
                    
                    for dz in range(2):
                        for hz in range(2):
                            for wz in range(2):
                                od = pd * 4 + dz * 2
                                oh = ph * 4 + hz * 2
                                ow = pw * 4 + wz * 2
                                
                                # ConvTranspose3d computation
                                conv_val = T.alloc_fragment((1,), dtype, 0)
                                for ic in range(in_channels):
                                    for kd in range(kernel_size):
                                        for kh in range(kernel_size):
                                            for kw in range(kernel_size):
                                                # Input indices
                                                in_d = od + padding - kd
                                                in_h = oh + padding - kh
                                                in_w = ow + padding - kw
                                                
                                                # Check bounds and stride
                                                if (in_d % stride == 0 and
                                                    in_h % stride == 0 and
                                                    in_w % stride == 0):
                                                    in_d = in_d // stride
                                                    in_h = in_h // stride
                                                    in_w = in_w // stride
                                                    
                                                    if (in_d >= 0 and in_d < depth and
                                                        in_h >= 0 and in_h < height and
                                                        in_w >= 0 and in_w < width):
                                                        conv_val[0] += (
                                                            Input[n, ic, in_d, in_h, in_w] *
                                                            Weight[ic, c, kd, kh, kw]
                                                        )
                                
                                # Add bias
                                conv_val[0] += Bias[c]
                                
                                # Batch normalization
                                bn_val = (conv_val[0] - RunningMean[c]) / T.sqrt(RunningVar[c] + 1e-5)
                                bn_val = bn_val * Gamma[c] + Beta[c]
                                
                                sum_val[0] += bn_val
                                count += 1
                    
                    Output[n, c, pd, ph, pw] = sum_val[0] / count

    return tilelang.compile(kernel, out_idx=[6], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)
        
        self._kernel_cache = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def _get_kernel(self, batch_size: int, depth: int, height: int, width: int):
        key = (batch_size, depth, height, width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose3d_bn_avgpool_kernel(
                batch_size=batch_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                depth=depth,
                height=height,
                width=width,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().half()
        batch_size, in_channels, depth, height, width = x.shape
        
        # Get kernel
        kernel = self._get_kernel(batch_size, depth, height, width)
        
        # Get batch norm parameters
        running_mean = self.batch_norm.running_mean.float()
        running_var = self.batch_norm.running_var.float()
        gamma = self.batch_norm.weight.half()
        beta = self.batch_norm.bias.half()
        
        # Get conv weight and bias
        weight = self.conv_transpose.weight.transpose(0, 1).half()  # TileLang expects (in_c, out_c, ...)
        bias = self.conv_transpose.bias.half()
        
        # Compute output
        output = kernel(x, weight, bias, running_mean, running_var, gamma, beta)
        
        return output.float()