import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_post_gemm_kernel(gX: cute.Tensor, gGNW: cute.Tensor, gGNB: cute.Tensor, gMW: cute.Tensor, gY: cute.Tensor, num_groups: int, eps: float):
    batch_idx = cute.arch.block_idx().x
    group_idx = cute.arch.block_idx().y
    tid = cute.arch.thread_idx().x
    
    out_features = gX.shape[1]
    group_size = out_features // num_groups
    idx = group_idx * group_size + tid
    
    shared_x = cute.shared_memory(float, (group_size,))
    shared_x[tid] = gX[batch_idx, idx]
    cute.sync()
    
    # Reduction for sum
    s = 1
    while s < group_size:
        if tid % (2 * s) == 0:
            shared_x[tid] += shared_x[tid + s]
        cute.sync()
        s *= 2
    sum_val = shared_x[0]
    
    shared_sq = cute.shared_memory(float, (group_size,))
    shared_sq[tid] = shared_x[tid] * shared_x[tid]
    cute.sync()
    
    # Reduction for sum_sq
    s = 1
    while s < group_size:
        if tid % (2 * s) == 0:
            shared_sq[tid] += shared_sq[tid + s]
        cute.sync()
        s *= 2
    sum_sq = shared_sq[0]
    
    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    
    x_norm = (shared_x[tid] - mean) / cute.sqrt(var + eps)
    y = x_norm * gGNW[idx] + gGNB[idx]
    
    # Swish
    sig = 1.0 / (1.0 + cute.exp(-y))
    y = y * sig
    
    # Multiply
    y = y * gMW[idx]
    
    # Swish again
    sig = 1.0 / (1.0 + cute.exp(-y))
    y = y * sig
    
    gY[batch_idx, idx] = y

@cute.jit
def fused_post_gemm_host(mX: cute.Tensor, mGNW: cute.Tensor, mGNB: cute.Tensor, mMW: cute.Tensor, mY: cute.Tensor, num_groups: int, eps: float):
    batch_size = mX.shape[0]
    out_features = mX.shape[1]
    group_size = out_features // num_groups
    grid = (batch_size, num_groups, 1)
    block = (group_size, 1, 1)
    fused_post_gemm_kernel(mX, mGNW, mGNB, mMW, mY, num_groups, eps).launch(grid=grid, block=block)

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.compiled = {}

    def forward(self, x):
        x = self.gemm(x)
        M, N = x.shape
        x = x.contiguous().cuda()
        gnw = self.group_norm.weight.contiguous().cuda()
        gnb = self.group_norm.bias.contiguous().cuda()
        mw = self.multiply_weight.contiguous().cuda()
        y = torch.empty((M, N), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mGNW = from_dlpack(gnw, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGNB = from_dlpack(gnb, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mMW = from_dlpack(mw, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_post_gemm_host, mX, mGNW, mGNB, mMW, mY, num_groups, 1e-5)
            self.compiled[key] = compiled
        
        compiled(mX, mGNW, mGNB, mMW, mY, num_groups, 1e-5)
        return y