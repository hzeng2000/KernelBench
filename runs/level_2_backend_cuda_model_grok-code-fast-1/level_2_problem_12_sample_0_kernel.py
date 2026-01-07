import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + scale + LeakyReLU
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* x, const float* weight, const float* bias, float multiplier, float negative_slope, float* out, int batch_size, int in_features, int out_features) {
    int batch = blockIdx.x;
    int out_f = blockIdx.y;
    if (batch >= batch_size || out_f >= out_features) return;
    
    float sum = bias[out_f];
    for (int in_f = 0; in_f < in_features; ++in_f) {
        sum += x[batch * in_features + in_f] * weight[out_f * in_features + in_f];
    }
    sum *= multiplier;
    out[batch * out_features + out_f] = sum > 0 ? sum : sum * negative_slope;
}

torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float multiplier, float negative_slope) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);
    auto out = torch::empty({batch_size, out_features}, x.options());
    
    dim3 blocks(batch_size, out_features);
    fused_kernel<<<blocks, 1>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), multiplier, negative_slope, out.data_ptr<float>(), batch_size, in_features, out_features);
    
    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float multiplier, float negative_slope);"
)

# Compile the inline CUDA code for the fused operation
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs fused GEMM + scale + LeakyReLU in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.fused_forward = fused_op

    def forward(self, x):
        return self.fused_forward.fused_forward_cuda(x, self.gemm.weight, self.gemm.bias, self.multiplier, self.negative_slope)