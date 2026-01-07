import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def group_norm_hardtanh_kernel(gX: cute.Tensor, gGN_w: cute.Tensor, gGN_b: cute.Tensor, gY: cute.Tensor, eps, min_val, max_val, num_groups):
    batch_idx = cute.arch.block_idx(0)
    group_idx = cute.arch.block_idx(1)
    thread_idx = cute.arch.thread_idx(0)
    group_size = gX.shape[1] // num_groups
    start_n = group_idx * group_size
    value = gX[batch_idx, start_n + thread_idx]
    layout = cute.Layout((group_size,), (1,))
    collective_tensor = cute.Tensor(value, layout)
    sum_x = cute.reduce(cute.sum, collective_tensor)
    sum_sq = cute.reduce(cute.sum, value * value, collective_tensor)
    mean = sum_x / group_size
    var = sum_sq / group_size - mean * mean
    normalized = (value - mean) / cute.sqrt(var + eps)
    gn_w = gGN_w[start_n + thread_idx]
    gn_b = gGN_b[start_n + thread_idx]
    y = normalized * gn_w + gn_b
    y = cute.max(min_val, cute.min(max_val, y))
    gY[batch_idx, start_n + thread_idx] = y

@cute.jit
def group_norm_hardtanh_host(mX: cute.Tensor, mGN_w: cute.Tensor, mGN_b: cute.Tensor, mY: cute.Tensor, eps, min_val, max_val, num_groups):
    batch_size = mX.shape[0]
    group_size = mX.shape[1] // num_groups
    grid = (batch_size, num_groups, 1)
    block = (group_size, 1, 1)
    group_norm_hardtanh_kernel(mX, mGN_w, mGN_b, mY, eps, min_val, max_val, num_groups).launch(grid=grid, block=block)

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.gn_weight = torch.nn.Parameter(torch.randn(out_features))
        self.gn_bias = torch.nn.Parameter(torch.randn(out_features))
        self.gemm_op = cutlass.Gemm(element=cutlass.DataType.f32, layout_a=cutlass.LayoutType.RowMajor, layout_b=cutlass.LayoutType.ColumnMajor, layout_c=cutlass.LayoutType.RowMajor)
        self.compiled = {}

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        C = self.bias.unsqueeze(0).expand(batch_size, -1).contiguous()
        self.gemm_op.run(x, self.weight.T, C)
        x = C
        x = x.contiguous()
        Y = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mGN_w = from_dlpack(self.gn_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGN_b = from_dlpack(self.gn_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mY = from_dlpack(Y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        eps = 1e-5
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(group_norm_hardtanh_host, mX, mGN_w, mGN_b, mY, eps, self.hardtanh_min, self.hardtanh_max, self.num_groups)
            self.compiled[key] = compiled
        compiled(mX, mGN_w, mGN_b, mY, eps, self.hardtanh_min, self.hardtanh_max, self.num_groups)
        return Y