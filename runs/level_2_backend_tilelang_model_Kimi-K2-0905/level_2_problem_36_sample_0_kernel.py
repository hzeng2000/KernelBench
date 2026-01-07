import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(batch_size: int, out_channels: int, out_height: int, out_width: int, block_M: int = 16, block_N: int = 16, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
        Bias: T.Tensor((1, 1, 1), dtype),
        Out: T.Tensor((batch_size, 1, 1, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_N), batch_size, threads=threads) as (bx, by):
            start_x = bx * block_N
            batch_idx = by

            # Shared memory for reduction
            shared_min = T.alloc_shared((block_M, block_N), dtype)
            shared_sum = T.alloc_shared((block_N,), dtype)

            # Initialize shared memory
            for local_x in T.Parallel(block_N):
                shared_sum[local_x] = T.cast(0, dtype)

            # Compute min and sum in parallel
            for local_y, local_x in T.Parallel(out_channels, block_N):
                x = start_x + local_x
                if x < out_width:
                    # Find min across channels
                    min_val = X[batch_idx, 0, 0, x]
                    for c in range(1, out_channels):
                        val = X[batch_idx, c, 0, x]
                        if val < min_val:
                            min_val = val
                    
                    # GELU activation
                    gelu_val = T.cast(0.5, dtype) * min_val * (T.cast(1.0, dtype) + T.erf(T.cast(0.70710678, dtype) * min_val))
                    
                    # Add bias
                    gelu_val = gelu_val + Bias[0, 0, 0]
                    
                    Out[batch_idx, 0, 0, x] = gelu_val

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, out_channels: int, out_height: int, out_width: int, tl_dtype: str):
        key = (batch_size, out_channels, out_height, out_width, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(batch_size, out_channels, out_height, out_width, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        # ConvTranspose2d
        x = self.conv_transpose(x)
        
        # Get dimensions
        batch_size, out_channels, out_height, out_width = x.shape
        
        # Convert to fp16
        x = x.half()
        bias_fp16 = self.bias.half()
        
        # Get kernel
        kernel = self._get_kernel(batch_size, out_channels, out_height, out_width, "float16")
        
        # Allocate output
        out = torch.empty(batch_size, 1, 1, out_width, dtype=torch.float16, device=x.device)
        
        # Run fused kernel
        kernel(x, bias_fp16, out)
        
        return out.float()