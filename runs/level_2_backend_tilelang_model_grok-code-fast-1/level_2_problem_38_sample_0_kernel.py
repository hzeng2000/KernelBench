import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_clamp_softmax_scale_kernel(B: int, C: int, S: int, block_S: int = 128, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_clamp_softmax_scale_kernel(
        X: T.Tensor((B, C, S), dtype),
        clamp_min: T.float32,
        clamp_max: T.float32,
        Scale: T.Tensor((1, C, 1, 1, 1), dtype),
        Out: T.Tensor((B, C, S), dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * C, threads=threads) as (bx, by):
            b = by // C
            c = by % C
            start_s = bx * block_S
            
            shared_max = T.shared(threads, dtype)
            shared_sum = T.shared(threads, dtype)
            
            # Compute local max
            local_max = T.min_value(dtype)
            with T.serial(block_S) as local_s:
                s = start_s + local_s
                if s < S:
                    val = T.clamp(X[b, c, s], clamp_min, clamp_max)
                    local_max = T.max(local_max, val)
            
            shared_max[T.thread_id()] = local_max
            T.sync()
            
            # Reduce shared_max
            with T.serial(T.log2(threads)) as i:
                offset = threads >> (i + 1)
                if T.thread_id() < offset:
                    shared_max[T.thread_id()] = T.max(shared_max[T.thread_id()], shared_max[T.thread_id() + offset])
            
            T.sync()
            global_max = shared_max[0]
            
            # Compute local sum
            local_sum = T.cast(0, dtype)
            with T.serial(block_S) as local_s:
                s = start_s + local_s
                if s < S:
                    val = T.clamp(X[b, c, s], clamp_min, clamp_max)
                    local_sum += T.exp(val - global_max)
            
            shared_sum[T.thread_id()] = local_sum
            T.sync()
            
            # Reduce shared_sum
            with T.serial(T.log2(threads)) as i:
                offset = threads >> (i + 1)
                if T.thread_id() < offset:
                    shared_sum[T.thread_id()] += shared_sum[T.thread_id() + offset]
            
            T.sync()
            global_sum = shared_sum[0]
            
            # Compute output
            with T.serial(block_S) as local_s:
                s = start_s + local_s
                if s < S:
                    val = T.clamp(X[b, c, s], clamp_min, clamp_max)
                    Out[b, c, s] = T.exp(val - global_max) / global_sum * Scale[0, c, 0, 0, 0]
    
    return tilelang.compile(fused_clamp_softmax_scale_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs average pooling, 3D transposed convolution, and fused clamp + spatial softmax + scale multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, S: int, tl_dtype: str):
        key = (B, C, S, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_clamp_softmax_scale_kernel(B, C, S, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)  # flatten spatial dims
        S = d * h * w
        kernel = self._get_kernel(b, c, S, "float16")
        x = kernel(x, self.clamp_min, self.clamp_max, self.scale)
        x = x.view(b, c, d, h, w)
        return x