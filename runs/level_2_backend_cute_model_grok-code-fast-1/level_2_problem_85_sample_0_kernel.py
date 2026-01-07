import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def scale_kernel(gA: cute.Tensor, gScale: cute.Tensor, gC: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    B, C, H, W = gA.shape
    total = B * C * H * W

    if thread_idx < total:
        b = thread_idx // (C * H * W)
        c = (thread_idx // (H * W)) % C
        h = (thread_idx // W) % H
        w = thread_idx % W

        gC[b, c, h, w] = gA[b, c, h, w] * gScale[c, 0, 0]

@cute.jit
def scale_host(mA: cute.Tensor, mScale: cute.Tensor, mC: cute.Tensor):
    B, C, H, W = mA.shape
    total = B * C * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)

    scale_kernel(mA, mScale, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def clamp_kernel(gA: cute.Tensor, gC: cute.Tensor, clamp_min: float, clamp_max: float): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    B, C, Hp, Wp = gA.shape
    total = B * C * Hp * Wp

    if thread_idx < total:
        b = thread_idx // (C * Hp * Wp)
        c = (thread_idx // (Hp * Wp)) % C
        hp = (thread_idx // Wp) % Hp
        wp = thread_idx % Wp

        val = gA[b, c, hp, wp]
        gC[b, c, hp, wp] = max(min(val, clamp_max), clamp_min)

@cute.jit
def clamp_host(mA: cute.Tensor, mC: cute.Tensor, clamp_min: float, clamp_max: float):
    B, C, Hp, Wp = mA.shape
    total = B * C * Hp * Wp
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)

    clamp_kernel(mA, mC, clamp_min, clamp_max).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = torch.nn.GroupNorm(num_groups, out_channels)
        self.scale = torch.nn.Parameter(torch.ones(scale_shape))
        self.maxpool = torch.nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.compiled_scale = {}
        self.compiled_clamp = {}

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        # Custom scale
        B, C, H, W = x.shape
        x = x.contiguous().cuda()
        scale_tensor = self.scale.contiguous().cuda()
        x_scaled = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mScale = from_dlpack(scale_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mC_scale = from_dlpack(x_scaled, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled_scale.get(key)
        if compiled is None:
            compiled = cute.compile(scale_host, mA, mScale, mC_scale)
            self.compiled_scale[key] = compiled

        compiled(mA, mScale, mC_scale)
        x = x_scaled

        x = self.maxpool(x)
        # Custom clamp
        B, C, Hp, Wp = x.shape
        x = x.contiguous().cuda()
        x_clamped = torch.empty((B, C, Hp, Wp), dtype=x.dtype, device=x.device)

        mA_clamp = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC_clamp = from_dlpack(x_clamped, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        compiled_clamp = self.compiled_clamp.get(key)
        if compiled_clamp is None:
            compiled_clamp = cute.compile(clamp_host, mA_clamp, mC_clamp, self.clamp_min, self.clamp_max)
            self.compiled_clamp[key] = compiled_clamp

        compiled_clamp(mA_clamp, mC_clamp, self.clamp_min, self.clamp_max)
        x = x_clamped
        return x