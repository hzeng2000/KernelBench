import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Swish + Multiply + Swish
fused_swish_multiply_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_swish_multiply_swish_kernel(const float* x, const float* multiply_weight, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float s1 = 1.0f / (1.0f + expf(-val));
        float temp = val * s1;
        temp = temp * multiply_weight[idx % multiply_weight.numel()];  // Handle broadcasting
        float s2 = 1.0f / (1.0f + expf(-temp));
        out[idx] = temp * s2;
    }
}

torch::Tensor fused_swish_multiply_swish_cuda(torch::Tensor x, torch::Tensor multiply_weight) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_swish_multiply_swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), multiply_weight.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_swish_multiply_swish_cpp_source = (
    "torch::Tensor fused_swish_multiply_swish_cuda(torch::Tensor x, torch::Tensor multiply_weight);"
)

# Compile the inline CUDA code for fused operations
fused_op = load_inline(
    name="fused_swish_multiply_swish",
    cpp_sources=fused_swish_multiply_swish_cpp_source,
    cuda_sources=fused_swish_multiply_swish_source,
    functions=["fused_swish_multiply_swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs GEMM, GroupNorm, and fused Swish + Multiply + Swish operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.fused_op = fused_op

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = self.gemm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.group_norm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.fused_op.fused_swish_multiply_swish_cuda(x, self.multiply_weight)
        return x