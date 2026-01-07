import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_activation_bn_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    eps: float = 1e-5,
    block_M: int = 8,
    block_N: int = 8,
    block_K: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1
    
    @T.prim_func
    def conv_activation_bn_kernel(
        X: T.Tensor((batch_size, in_channels, height, width), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        B: T.Tensor((out_channels,), dtype),
        running_mean: T.Tensor((out_channels,), "float32"),
        running_var: T.Tensor((out_channels,), "float32"),
        gamma: T.Tensor((out_channels,), dtype),
        beta: T.Tensor((out_channels,), dtype),
        Y: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_N), T.ceildiv(out_height, block_M), out_channels, batch_size, threads=threads) as (bx, by, co, b):
            start_x = bx * block_N
            start_y = by * block_M
            
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                
                if y < out_height and x < out_width:
                    acc = T.alloc_fragment((1,), dtype, scope="local")
                    acc[0] = T.cast(0.0, dtype)
                    
                    for ci in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                acc[0] += X[b, ci, y + kh, x + kw] * W[co, ci, kh, kw]
                    
                    acc[0] += B[co]
                    
                    # Activation: multiply(tanh(softplus(x)), x)
                    softplus = T.log(T.cast(1.0, dtype) + T.exp(acc[0]))
                    tanh_val = T.tanh(softplus)
                    activated = tanh_val * acc[0]
                    
                    # BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
                    mean = T.cast(running_mean[co], dtype)
                    var = T.cast(running_var[co], dtype)
                    bn = (activated - mean) / T.sqrt(var + T.cast(eps, dtype)) * gamma[co] + beta[co]
                    
                    Y[b, co, y, x] = bn

    return tilelang.compile(conv_activation_bn_kernel, out_idx=[6], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self._kernel_cache = {}
        self.eps = eps

    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int):
        key = (batch_size, in_channels, out_channels, height, width, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_activation_bn_kernel(
                batch_size, in_channels, out_channels, height, width, kernel_size, eps=self.eps
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        
        # Ensure weights and inputs are in fp16
        x_fp16 = x.half()
        weight_fp16 = self.conv.weight.half()
        bias_fp16 = self.conv.bias.half()
        
        # Get BatchNorm parameters
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight.half()
        beta = self.bn.bias.half()
        
        kernel = self._get_kernel(batch_size, in_channels, out_channels, height, width, kernel_size)
        output = kernel(
            x_fp16, weight_fp16, bias_fp16,
            running_mean, running_var, gamma, beta
        )
        
        return output.float()