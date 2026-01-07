import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def avg_pool3d_kernel(gX: cute.Tensor, gY: cute.Tensor, kernel_size: int):
    bidx, tidy, tidx = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    batch = bidx
    c = tidy
    hw = tidx
    
    B, C, D, H, W = gX.shape
    D_out = D // kernel_size
    H_out = H // kernel_size
    W_out = W // kernel_size
    
    if batch < B and c < C and hw < D_out * H_out * W_out:
        d_out = hw // (H_out * W_out)
        rem = hw % (H_out * W_out)
        h_out = rem // W_out
        w_out = rem % W_out
        
        sum_val = 0.0
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    d_in = d_out * kernel_size + kd
                    h_in = h_out * kernel_size + kh
                    w_in = w_out * kernel_size + kw
                    sum_val += gX[batch, c, d_in, h_in, w_in]
        
        gY[batch, c, d_out, h_out, w_out] = sum_val / (kernel_size * kernel_size * kernel_size)

@cute.kernel
def conv_transpose3d_kernel(gX: cute.Tensor, gW: cute.Tensor, gY: cute.Tensor):
    bidx, tidy, tidx = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    batch = bidx
    c_out = tidy
    hw = tidx
    
    B, C_in, D_in, H_in, W_in = gX.shape
    C_out, _, K, _, _ = gW.shape
    D_out = D_in * 2
    H_out = H_in * 2
    W_out = W_in * 2
    
    if batch < B and c_out < C_out and hw < D_out * H_out * W_out:
        d_out = hw // (H_out * W_out)
        rem = hw % (H_out * W_out)
        h_out = rem // W_out
        w_out = rem % W_out
        
        sum_val = 0.0
        for c_in in range(C_in):
            for k in range(K):
                for kh in range(K):
                    for kw in range(K):
                        d_in = (d_out + padding - k) // stride
                        h_in = (h_out + padding - kh) // stride
                        w_in = (w_out + padding - kw) // stride
                        if (d_out + padding - k) % stride == 0 and (h_out + padding - kh) % stride == 0 and (w_out + padding - kw) % stride == 0:
                            if d_in >= 0 and d_in < D_in and h_in >= 0 and h_in < H_in and w_in >= 0 and w_in < W_in:
                                sum_val += gX[batch, c_in, d_in, h_in, w_in] * gW[c_out, c_in, k, kh, kw]
        
        gY[batch, c_out, d_out, h_out, w_out] = sum_val

@cute.kernel
def clamp_softmax_scale_kernel(gX: cute.Tensor, gY: cute.Tensor, clamp_min: float, clamp_max: float, gScale: cute.Tensor):
    bidx, tidy, tidx = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    batch = bidx
    c = tidy
    hw = tidx
    
    B, C, D, H, W = gX.shape
    spatial_size = D * H * W
    
    if batch < B and c < C and hw < spatial_size:
        d = hw // (H * W)
        rem = hw % (H * W)
        h = rem // W
        w = rem % W
        
        val = gX[batch, c, d, h, w]
        val = min(max(val, clamp_min), clamp_max)
        gX[batch, c, d, h, w] = val
    
    cute.arch.sync_threads()
    
    if batch < B and c < C and hw == 0:
        max_val = gX[batch, c, 0, 0, 0]
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    max_val = max(max_val, gX[batch, c, d, h, w])
        
        sum_exp = 0.0
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    exp_val = exp(gX[batch, c, d, h, w] - max_val)
                    gX[batch, c, d, h, w] = exp_val
                    sum_exp += exp_val
        
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    gY[batch, c, d, h, w] = (gX[batch, c, d, h, w] / sum_exp) * gScale[0, c, 0, 0, 0]

@cute.jit
def avg_pool3d_host(mX: cute.Tensor, mY: cute.Tensor, kernel_size: int):
    B, C, _, H, W = mX.shape
    D_out = mX.shape[2] // kernel_size
    H_out = H // kernel_size
    W_out = W // kernel_size
    
    threads_per_block = 256
    total_spatial = D_out * H_out * W_out
    grid_x = B
    grid_y = C
    grid_z = cute.ceil_div(total_spatial, threads_per_block)
    
    avg_pool3d_kernel(mX, mY, kernel_size).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, 1, 1))

@cute.jit
def conv_transpose3d_host(mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor, padding: int, stride: int):
    B, C_in, D_in, H_in, W_in = mX.shape
    C_out, _, K, _, _ = mW.shape
    D_out = D_in * 2
    H_out = H_in * 2
    W_out = W_in * 2
    
    threads_per_block = 256
    total_spatial = D_out * H_out * W_out
    grid_x = B
    grid_y = C_out
    grid_z = cute.ceil_div(total_spatial, threads_per_block)
    
    conv_transpose3d_kernel(mX, mW, mY).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, 1, 1))

@cute.jit
def clamp_softmax_scale_host(mX: cute.Tensor, mY: cute.Tensor, clamp_min: float, clamp_max: float, mScale: cute.Tensor):
    B, C, D, H, W = mX.shape
    spatial_size = D * H * W
    
    threads_per_block = spatial_size
    grid_x = B
    grid_y = C
    grid_z = 1
    
    clamp_softmax_scale_kernel(mX, mY, clamp_min, clamp_max, mScale).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.compiled = {}

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.contiguous().cuda()
        
        # Average pooling
        D_out = D // self.pool_kernel_size
        H_out = H // self.pool_kernel_size
        W_out = W // self.pool_kernel_size
        pooled = torch.empty((B, C, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mPooled = from_dlpack(pooled, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key = (x.dtype, 'avg_pool')
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(avg_pool3d_host, mX, mPooled, self.pool_kernel_size)
            self.compiled[key] = compiled
        compiled(mX, mPooled, self.pool_kernel_size)
        
        # Conv transpose (using PyTorch for now due to complexity)
        conv_out = self.conv_transpose(pooled)
        
        # Clamp + softmax + scale fusion
        B, C_out, D_conv, H_conv, W_conv = conv_out.shape
        output = torch.empty_like(conv_out)
        
        mConv = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mScale = from_dlpack(self.scale, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key = (conv_out.dtype, 'clamp_softmax_scale')
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(clamp_softmax_scale_host, mConv, mOutput, self.clamp_min, self.clamp_max, mScale)
            self.compiled[key] = compiled
        compiled(mConv, mOutput, self.clamp_min, self.clamp_max, mScale)
        
        return output