import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_add_hardswish_kernel(total_size: int, block_size: int = 1024, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_add_hardswish_kernel(
        A: T.Tensor((total_size,), dtype),
        B: T.Tensor((total_size,), dtype),
        C: T.Tensor((total_size,), dtype),
    ):
        with T.Kernel(T.ceildiv(total_size, block_size), threads=threads) as bx:
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < total_size:
                    temp = A[idx] + B[idx]
                    relu6 = T.max(T.cast(0, dtype), T.min(T.cast(6, dtype), temp + T.cast(3, dtype)))
                    hardswish = temp * relu6 / T.cast(6, dtype)
                    C[idx] = hardswish

    return tilelang.compile(fused_add_hardswish_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, fuses add and HardSwish with a custom TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}

    def _get_kernel(self, total_size: int, tl_dtype: str):
        key = (total_size, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_add_hardswish_kernel(total_size, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added, of shape (batch_size, out_channels, D_out, H_out, W_out).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D_out, H_out, W_out) after fused add and HardSwish.
        """
        x = self.conv_transpose(x)
        # Ensure tensors are contiguous and in FP16
        x = x.contiguous().half()
        add_input = add_input.contiguous().half()
        # Flatten to 1D
        total_size = x.numel()
        kernel = self._get_kernel(total_size, "float16")
        output = kernel(x.flatten(), add_input.flatten())
        return output.view(x.shape)