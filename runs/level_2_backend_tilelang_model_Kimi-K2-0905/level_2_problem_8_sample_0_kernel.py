import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_conv3d_div_maxpool_kernel(
    batch_size: int, 
    in_channels: int, 
    out_channels: int, 
    depth: int, 
    height: int, 
    width: int,
    kernel_size: int = 3,
    pool_size: int = 2,
    divisor: float = 2.0,
    block_D: int = 4,
    block_H: int = 4,
    block_W: int = 4,
    threads: int = 128,
    dtype: str = "float16"
):
    
    out_depth = depth - kernel_size + 1
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1
    
    pooled_depth = out_depth // pool_size
    pooled_height = out_height // pool_size
    pooled_width = out_width // pool_size
    
    @T.prim_func
    def fused_kernel(
        Input: T.Tensor((batch_size, in_channels, depth, height, width), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype),
        Output: T.Tensor((batch_size, out_channels, pooled_depth, pooled_height, pooled_width), dtype),
    ):
        with T.Kernel(T.ceildiv(pooled_width, block_W), T.ceildiv(pooled_height, block_H), 
                     T.ceildiv(pooled_depth, block_D), out_channels, batch_size, threads=threads) as (bx, by, bz, oc, b):
            start_w = bx * block_W * pool_size
            start_h = by * block_H * pool_size
            start_d = bz * block_D * pool_size
            
            for local_d, local_h, local_w in T.Parallel(block_D, block_H, block_W):
                d = start_d + local_d * pool_size
                h = start_h + local_h * pool_size
                w = start_w + local_w * pool_size
                
                if d < out_depth and h < out_height and w < out_width:
                    max_val = T.min_value(dtype)
                    
                    for pd in range(pool_size):
                        for ph in range(pool_size):
                            for pw in range(pool_size):
                                conv_sum = T.cast(0.0, dtype)
                                
                                for ic in range(in_channels):
                                    for kd in range(kernel_size):
                                        for kh in range(kernel_size):
                                            for kw in range(kernel_size):
                                                in_d = d + pd + kd
                                                in_h = h + ph + kh
                                                in_w = w + pw + kw
                                                if in_d < depth and in_h < height and in_w < width:
                                                    conv_sum += Input[b, ic, in_d, in_h, in_w] * Weight[oc, ic, kd, kh, kw]
                                
                                conv_div = conv_sum / T.cast(divisor, dtype)
                                max_val = T.max(max_val, conv_div)
                    
                    out_d = (d // pool_size)
                    out_h = (h // pool_size)
                    out_w = (w // pool_size)
                    if out_d < pooled_depth and out_h < pooled_height and out_w < pooled_width:
                        Output[b, oc, out_d, out_h, out_w] = max_val

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


def build_global_avg_pool_add_sum_kernel(
    batch_size: int,
    channels: int,
    depth: int,
    height: int,
    width: int,
    sum_dim: int,
    block_C: int = 16,
    threads: int = 128,
    dtype: str = "float16"
):
    
    @T.prim_func
    def global_avg_kernel(
        Input: T.Tensor((batch_size, channels, depth, height, width), dtype),
        Bias: T.Tensor((channels, 1, 1, 1), dtype),
        Output: T.Tensor((batch_size, channels), dtype),
    ):
        with T.Kernel(T.ceildiv(channels, block_C), batch_size, threads=threads) as (bc, b):
            start_c = bc * block_C
            
            for local_c in T.Parallel(block_C):
                c = start_c + local_c
                if c < channels:
                    sum_val = T.cast(0.0, dtype)
                    count = 0
                    
                    for d in range(depth):
                        for h in range(height):
                            for w in range(width):
                                sum_val += Input[b, c, d, h, w]
                                count += 1
                    
                    avg_val = sum_val / T.cast(count, dtype)
                    biased_val = avg_val + Bias[c, 0, 0, 0]
                    
                    if sum_dim == 1:
                        Output[b, c] = biased_val
                    else:
                        Output[b, 0] = biased_val

    return tilelang.compile(global_avg_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.pool_size = pool_size[0]
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self._kernel_cache = {}
        
    def _get_fused_kernel(self, batch_size: int, in_channels: int, out_channels: int, 
                         depth: int, height: int, width: int):
        key = ("fused", batch_size, in_channels, out_channels, depth, height, width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv3d_div_maxpool_kernel(
                batch_size, in_channels, out_channels, depth, height, width,
                kernel_size=3, pool_size=self.pool_size, divisor=self.divisor
            )
        return self._kernel_cache[key]
    
    def _get_pool_kernel(self, batch_size: int, channels: int, depth: int, height: int, width: int):
        key = ("pool", batch_size, channels, depth, height, width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_global_avg_pool_add_sum_kernel(
                batch_size, channels, depth, height, width, self.sum_dim
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, depth, height, width = x.shape
        out_channels = self.conv.out_channels
        
        # Fused conv3d + div + maxpool
        fused_kernel = self._get_fused_kernel(batch_size, in_channels, out_channels, depth, height, width)
        weight = self.conv.weight.half()
        
        out_depth = (depth - 3 + 1) // self.pool_size
        out_height = (height - 3 + 1) // self.pool_size
        out_width = (width - 3 + 1) // self.pool_size
        
        pooled_output = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, 
                                   dtype=torch.float16, device=x.device)
        
        fused_kernel(x.half(), weight, pooled_output)
        
        # Global average pooling + bias addition + sum
        pool_kernel = self._get_pool_kernel(batch_size, out_channels, out_depth, out_height, out_width)
        bias_expanded = self.bias.half().expand(out_channels, out_depth, out_height, out_width)
        
        final_output = torch.empty(batch_size, out_channels, dtype=torch.float16, device=x.device)
        pool_kernel(pooled_output, self.bias.half(), final_output)
        
        if self.sum_dim == 1:
            return final_output.sum(dim=1)
        else:
            return final_output