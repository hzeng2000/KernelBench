import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def build_conv_transpose3d_kernel(
    batch, in_c, out_c, depth, height, width,
    kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w, out_d, out_h, out_w,
    block_d=4, block_h=4, block_w=4, block_c=16,
    threads=256, dtype="float16"
):
    @T.prim_func
    def conv_transpose3d_kernel(
        Input: T.Tensor((batch, in_c, depth, height, width), dtype),
        Weight: T.Tensor((in_c, out_c, kernel_d, kernel_h, kernel_w), dtype),
        Output: T.Tensor((batch, out_c, out_d, out_h, out_w), dtype),
    ):
        with T.Kernel(T.ceildiv(out_w, block_w), T.ceildiv(out_h, block_h), T.ceildiv(out_d, block_d), T.ceildiv(out_c, block_c), batch, threads=threads) as (bx, by, bz, bc, bb):
            out_x = bx * block_w
            out_y = by * block_h
            out_z = bz * block_d
            out_c_idx = bc * block_c
            batch_idx = bb

            for local_c, local_z, local_y, local_x in T.Parallel(block_c, block_d, block_h, block_w):
                c = out_c_idx + local_c
                z = out_z + local_z
                y = out_y + local_y
                x = out_x + local_x

                if c < out_c and z < out_d and y < out_h and x < out_w and batch_idx < batch:
                    acc = T.cast(0.0, dtype)
                    for in_c_idx in range(in_c):
                        for k_d in range(kernel_d):
                            for k_h in range(kernel_h):
                                for k_w in range(kernel_w):
                                    in_z = (z + pad_d - k_d) // stride_d
                                    in_y = (y + pad_h - k_h) // stride_h
                                    in_x = (x + pad_w - k_w) // stride_w
                                    if (z + pad_d - k_d) % stride_d == 0 and (y + pad_h - k_h) % stride_h == 0 and (x + pad_w - k_w) % stride_w == 0:
                                        if in_z >= 0 and in_z < depth and in_y >= 0 and in_y < height and in_x >= 0 and in_x < width:
                                            acc += Input[batch_idx, in_c_idx, in_z, in_y, in_x] * Weight[in_c_idx, c, k_d, k_h, k_w]
                    Output[batch_idx, c, z, y, x] = acc

    return tilelang.compile(conv_transpose3d_kernel, out_idx=[2], target="cuda")

def build_maxpool3d_sum_kernel(
    batch, channels, depth, height, width, pool1_kernel=2, pool2_kernel=3,
    block_d=4, block_h=4, block_w=4, threads=256, dtype="float16"
):
    out_d1 = depth // pool1_kernel
    out_h1 = height // pool1_kernel
    out_w1 = width // pool1_kernel
    out_d2 = out_d1 // pool2_kernel
    out_h2 = out_h1 // pool2_kernel
    out_w2 = out_w1 // pool2_kernel

    @T.prim_func
    def maxpool3d_sum_kernel(
        Input: T.Tensor((batch, channels, depth, height, width), dtype),
        Output: T.Tensor((batch, 1, out_d2, out_h2, out_w2), dtype),
    ):
        with T.Kernel(T.ceildiv(out_w2, block_w), T.ceildiv(out_h2, block_h), T.ceildiv(out_d2, block_d), batch, threads=threads) as (bx, by, bz, bb):
            out_x = bx * block_w
            out_y = by * block_h
            out_z = bz * block_d
            batch_idx = bb

            for local_z, local_y, local_x in T.Parallel(block_d, block_h, block_w):
                z = out_z + local_z
                y = out_y + local_y
                x = out_x + local_x

                if z < out_d2 and y < out_h2 and x < out_w2 and batch_idx < batch:
                    max_val = T.cast(-1e9, dtype)
                    for c in range(channels):
                        pool1_max = T.cast(-1e9, dtype)
                        for k_d1 in range(pool1_kernel):
                            for k_h1 in range(pool1_kernel):
                                for k_w1 in range(pool1_kernel):
                                    in_z1 = z * pool2_kernel * pool1_kernel + k_d1 * pool2_kernel
                                    in_y1 = y * pool2_kernel * pool1_kernel + k_h1 * pool2_kernel
                                    in_x1 = x * pool2_kernel * pool1_kernel + k_w1 * pool2_kernel
                                    if in_z1 < depth and in_y1 < height and in_x1 < width:
                                        pool2_max = T.cast(-1e9, dtype)
                                        for k_d2 in range(pool2_kernel):
                                            for k_h2 in range(pool2_kernel):
                                                for k_w2 in range(pool2_kernel):
                                                    in_z2 = in_z1 + k_d2
                                                    in_y2 = in_y1 + k_h2
                                                    in_x2 = in_x1 + k_w2
                                                    if in_z2 < depth and in_y2 < height and in_x2 < width:
                                                        val = Input[batch_idx, c, in_z2, in_y2, in_x2]
                                                        pool2_max = T.max(pool2_max, val)
                                        pool1_max = T.max(pool1_max, pool2_max)
                        max_val = T.max(max_val, pool1_max)
                    Output[batch_idx, 0, z, y, x] = max_val

    return tilelang.compile(maxpool3d_sum_kernel, out_idx=[1], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *self.kernel_size, dtype=torch.float16))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self._kernel_cache = {}

    def _get_conv_transpose_kernel(self, batch, depth, height, width):
        key = ("conv_transpose", batch, depth, height, width)
        if key not in self._kernel_cache:
            out_d = (depth - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
            out_h = (height - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
            out_w = (width - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2]
            self._kernel_cache[key] = build_conv_transpose3d_kernel(
                batch, self.in_channels, self.out_channels, depth, height, width,
                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                self.stride[0], self.stride[1], self.stride[2],
                self.padding[0], self.padding[1], self.padding[2],
                out_d, out_h, out_w
            )
        return self._kernel_cache[key]

    def _get_maxpool_sum_kernel(self, batch, channels, depth, height, width):
        key = ("maxpool_sum", batch, channels, depth, height, width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_maxpool3d_sum_kernel(
                batch, channels, depth, height, width
            )
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.contiguous().half()
        batch, _, depth, height, width = x.shape
        
        conv_kernel = self._get_conv_transpose_kernel(batch, depth, height, width)
        x = conv_kernel(x, self.weight)
        
        pool_sum_kernel = self._get_maxpool_sum_kernel(batch, self.out_channels, x.shape[2], x.shape[3], x.shape[4])
        x = pool_sum_kernel(x)
        
        return x