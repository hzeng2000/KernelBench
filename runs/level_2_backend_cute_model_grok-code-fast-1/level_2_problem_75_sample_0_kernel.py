import torch
import torch.nn as nn
import cutlass

class ModelNew(nn.Module):
    """
    Optimized Model that performs a GEMM using CUTLASS, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.final_bias = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        B, K = x.shape
        N = self.weight.shape[0]
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty((B, N), dtype=x.dtype, device=x.device)
        cutlass.gemm(x, weight.t(), out)
        out = out + bias
        x = self.group_norm(out)
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = x + self.final_bias
        return x