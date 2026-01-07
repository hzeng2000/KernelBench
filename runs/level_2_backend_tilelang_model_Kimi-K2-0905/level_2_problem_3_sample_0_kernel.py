import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose3d_kernel(
    batch_size, in_channels, out_channels, 
    in_depth, in_height, in_width,
    out_depth, out_height, out_width,
    kernel_size, stride, padding, output_padding, groups=1
):
    k_d, k_h, k_w = kernel_size
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding
    
    @T.prim_func
    def conv_transpose3d_kernel(
        Input: T.Tensor((batch_size, in_channels, in_depth, in_height, in_width), "float16"),
        Weight: T.Tensor((in_channels, out_channels // groups, k_d, k_h, k_w), "float16"),
        Output: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), "float16"),
    ):
        with T.Kernel(batch_size, out_channels, out_depth, out_height, out_width, threads=128) as (n, oc, od, oh, ow):
            acc = T.alloc_fragment((1,), "float32", scope="local")
            acc[0] = 0.0
            
            for ic in range(in_channels):
                for kd in range(k_d):
                    for kh in range(k_h):
                        for kw in range(k_w):
                            in_d = od + pad_d - kd * stride_d + out_pad_d
                            in_h = oh + pad_h - kh * stride_h + out_pad_h
                            in_w = ow + pad_w - kw * stride_w + out_pad_w
                            
                            if in_d >= 0 and in_d < in_depth and in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width:
                                acc[0] += T.cast(Input[n, ic, in_d, in_h, in_w], "float32") * T.cast(Weight[ic, oc // groups, kd, kh, kw], "float32")
            
            Output[n, oc, od, oh, ow] = T.cast(acc[0], "float16")
    
    return tilelang.compile(conv_transpose3d_kernel, out_idx=[2], target="cuda")


def build_fused_add_layernorm_gelu_kernel(batch_size, channels, depth, height, width, eps=1e-5):
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, channels, depth, height, width), "float16"),
        Weight: T.Tensor((channels,), "float16"),
        Bias: T.Tensor((channels,), "float16"),
        AddVal: T.Tensor((1,), "float16"),
        Output: T.Tensor((batch_size, channels, depth, height, width), "float16"),
    ):
        with T.Kernel(batch_size, channels, depth, height, width, threads=128) as (n, c, d, h, w):
            val = T.cast(X[n, c, d, h, w], "float32") + T.cast(AddVal[0], "float32")
            
            # Compute mean across channels for this spatial location
            mean = T.alloc_fragment((1,), "float32", scope="local")
            mean[0] = 0.0
            for cc in range(channels):
                mean[0] += T.cast(X[n, cc, d, h, w], "float32")
            mean[0] /= channels
            
            # Compute variance
            var = T.alloc_fragment((1,), "float32", scope="local")
            var[0] = 0.0
            for cc in range(channels):
                diff = T.cast(X[n, cc, d, h, w], "float32") - mean[0]
                var[0] += diff * diff
            var[0] /= channels
            
            # Normalize
            normalized = (val - mean[0]) / T.sqrt(var[0] + eps)
            
            # Scale and shift
            out_val = normalized * T.cast(Weight[c], "float32") + T.cast(Bias[c], "float32")
            
            # GELU activation
            gelu_out = 0.5 * out_val * (1.0 + T.tanh(0.7978845608 * (out_val + 0.044715 * out_val * out_val * out_val)))
            
            Output[n, c, d, h, w] = T.cast(gelu_out, "float16")
    
    return tilelang.compile(fused_kernel, out_idx=[4], target="cuda")


def build_avgpool3d_kernel(batch_size, channels, in_depth, in_height, in_width, kernel_size):
    k_d, k_h, k_w = kernel_size
    out_depth = in_depth // k_d
    out_height = in_height // k_h
    out_width = in_width // k_w
    
    @T.prim_func
    def avgpool3d_kernel(
        Input: T.Tensor((batch_size, channels, in_depth, in_height, in_width), "float16"),
        Output: T.Tensor((batch_size, channels, out_depth, out_height, out_width), "float16"),
    ):
        with T.Kernel(batch_size, channels, out_depth, out_height, out_width, threads=128) as (n, c, od, oh, ow):
            acc = T.alloc_fragment((1,), "float32", scope="local")
            acc[0] = 0.0
            
            for kd in range(k_d):
                for kh in range(k_h):
                    for kw in range(k_w):
                        in_d = od * k_d + kd
                        in_h = oh * k_h + kh
                        in_w = ow * k_w + kw
                        acc[0] += T.cast(Input[n, c, in_d, in_h, in_w], "float32")
            
            acc[0] /= (k_d * k_h * k_w)
            Output[n, c, od, oh, ow] = T.cast(acc[0], "float16")
    
    return tilelang.compile(avgpool3d_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight, dtype=torch.float16))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        
        self._kernel_cache = {}
        
    def _get_conv_kernel(self, batch_size, in_channels, out_channels, in_depth, in_height, in_width, out_depth, out_height, out_width, kernel_size, stride, padding, output_padding):
        key = ("conv", batch_size, in_channels, out_channels, in_depth, in_height, in_width, out_depth, out_height, out_width, kernel_size, stride, padding, output_padding)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose3d_kernel(
                batch_size, in_channels, out_channels, 
                in_depth, in_height, in_width,
                out_depth, out_height, out_width,
                kernel_size, stride, padding, output_padding
            )
        return self._kernel_cache[key]
    
    def _get_fused_kernel(self, batch_size, channels, depth, height, width):
        key = ("fused", batch_size, channels, depth, height, width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_add_layernorm_gelu_kernel(batch_size, channels, depth, height, width)
        return self._kernel_cache[key]
    
    def _get_pool_kernel(self, batch_size, channels, in_depth, in_height, in_width, kernel_size):
        key = ("pool", batch_size, channels, in_depth, in_height, in_width, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_avgpool3d_kernel(batch_size, channels, in_depth, in_height, in_width, kernel_size)
        return self._kernel_cache[key]
    
    def forward(self, x):
        # ConvTranspose3D
        x = self.conv_transpose(x)
        
        # Get dimensions
        batch_size, channels, depth, height, width = x.shape
        
        # Add + LayerNorm + GELU (fused)
        weight = self.norm.weight.data.to(torch.float16)
        bias = self.norm.bias.data.to(torch.float16)
        add_val = self.sum_weight
        
        fused_kernel = self._get_fused_kernel(batch_size, channels, depth, height, width)
        x = fused_kernel(x, weight, bias, add_val)
        
        # Average Pooling
        pool_kernel_size = self.avg_pool.kernel_size
        pool_kernel = self._get_pool_kernel(batch_size, channels, depth, height, width, pool_kernel_size)
        x = pool_kernel(x)
        
        return x