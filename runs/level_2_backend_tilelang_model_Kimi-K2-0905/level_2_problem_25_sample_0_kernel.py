import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv_min_tanh_kernel(batch_size: int, out_channels: int, height: int, width: int, kernel_size: int, block_M: int = 8, block_N: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def conv_min_tanh_kernel(
        Input: T.Tensor((batch_size, out_channels, height, width), dtype),
        Output: T.Tensor((batch_size, 1, height, width), dtype),
    ):
        with T.Kernel(T.ceildiv(width, block_N), T.ceildiv(height, block_M), batch_size, threads=threads) as (bx, by, bz):
            start_x = bx * block_N
            start_y = by * block_M
            batch_idx = bz

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < height and x < width:
                    min_val = T.min_value(dtype)
                    for c in range(out_channels):
                        val = Input[batch_idx, c, y, x]
                        if val < min_val:
                            min_val = val
                    
                    # Apply tanh twice
                    tanh1 = T.tanh(min_val)
                    tanh2 = T.tanh(tanh1)
                    
                    Output[batch_idx, 0, y, x] = tanh2

    return tilelang.compile(conv_min_tanh_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, out_channels: int, height: int, width: int, tl_dtype: str):
        key = (batch_size, out_channels, height, width, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_min_tanh_kernel(batch_size, out_channels, height, width, 3, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv(x)
        
        batch_size, out_channels, height, width = x.shape
        kernel = self._get_kernel(batch_size, out_channels, height, width, "float16")
        
        x_fp16 = x.half()
        output = kernel(x_fp16)
        
        return output