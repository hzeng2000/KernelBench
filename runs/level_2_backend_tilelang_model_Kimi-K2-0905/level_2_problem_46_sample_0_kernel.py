import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_conv_sub_tanh_sub_pool_kernel(
    batch_size: int, in_channels: int, out_channels: int, height: int, width: int,
    kernel_size: int, pool_size: int, subtract1: float, subtract2: float,
    block_size: int = 16, threads: int = 256, dtype: str = "float16"
):
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1
    pooled_height = out_height // pool_size
    pooled_width = out_width // pool_size
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_channels, height, width), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        B: T.Tensor((out_channels,), dtype),
        Y: T.Tensor((batch_size, out_channels, pooled_height, pooled_width), dtype),
    ):
        with T.Kernel(T.ceildiv(pooled_width, block_size), T.ceildiv(pooled_height, block_size), out_channels, batch_size, threads=threads) as (bx, by, co, n):
            start_x = bx * block_size
            start_y = by * block_size
            
            for local_y, local_x in T.Parallel(block_size, block_size):
                py = start_y + local_y
                px = start_x + local_x
                
                if py < pooled_height and px < pooled_width:
                    sum_val = T.cast(0.0, dtype)
                    
                    # Average pooling over pool_size x pool_size region
                    for ky in range(pool_size):
                        for kx in range(pool_size):
                            oy = py * pool_size + ky
                            ox = px * pool_size + kx
                            
                            # Conv2d at this position
                            conv_val = T.cast(0.0, dtype)
                            for ci in range(in_channels):
                                for kh in range(kernel_size):
                                    for kw in range(kernel_size):
                                        ih = oy + kh
                                        iw = ox + kw
                                        conv_val += X[n, ci, ih, iw] * W[co, ci, kh, kw]
                            conv_val += B[co]
                            
                            # Subtract subtract1, tanh, subtract subtract2
                            intermediate = conv_val - T.cast(subtract1, dtype)
                            intermediate = T.tanh(intermediate)
                            intermediate = intermediate - T.cast(subtract2, dtype)
                            
                            sum_val += intermediate
                    
                    # Average
                    Y[n, co, py, px] = sum_val / T.cast(pool_size * pool_size, dtype)
    
    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.kernel_size_pool = kernel_size_pool
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, height: int, width: int, tl_dtype: str):
        key = (batch_size, in_channels, out_channels, height, width, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_sub_tanh_sub_pool_kernel(
                batch_size, in_channels, out_channels, height, width,
                self.conv.kernel_size[0], self.kernel_size_pool,
                self.subtract1_value, self.subtract2_value, dtype=tl_dtype
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel = self._get_kernel(batch_size, in_channels, out_channels, height, width, "float16")
        
        # Get conv weights and bias
        weight = self.conv.weight.half().contiguous()
        bias = self.conv.bias.half().contiguous() if self.conv.bias is not None else torch.zeros(out_channels, dtype=torch.float16, device=x.device)
        
        # Compute output shape
        out_height = height - self.conv.kernel_size[0] + 1
        out_width = width - self.conv.kernel_size[1] + 1
        pooled_height = out_height // self.kernel_size_pool
        pooled_width = out_width // self.kernel_size_pool
        
        # Allocate output tensor
        output = torch.empty(batch_size, out_channels, pooled_height, pooled_width, dtype=torch.float16, device=x.device)
        
        # Run fused kernel
        kernel(x.half(), weight, bias, output)
        
        return output