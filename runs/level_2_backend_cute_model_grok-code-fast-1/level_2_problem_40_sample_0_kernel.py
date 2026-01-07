import torch
import torch.nn as nn
import cutlass
from cutlass import ops, LayoutType, DataType, epilogue

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        A = x.contiguous().cuda()
        B = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        scaling = self.scaling_factor + 1.0
        epilogue_fn = epilogue.LinearCombination(alpha=scaling, beta=scaling)
        gemm_op = ops.Gemm(
            element_A=DataType.f32,
            element_B=DataType.f32,
            element_C=DataType.f32,
            element_D=DataType.f32,
            layout_A=LayoutType.RowMajor,
            layout_B=LayoutType.ColumnMajor,
            layout_C=LayoutType.RowMajor,
            element_accumulator=DataType.f32,
            epilogue=epilogue_fn,
            swizzling_functor=cutlass.IdentitySwizzle,
        )
        C = torch.empty(A.shape[0], B.shape[0], dtype=A.dtype, device=A.device)
        gemm_op(A, B, C, bias=bias)
        return C