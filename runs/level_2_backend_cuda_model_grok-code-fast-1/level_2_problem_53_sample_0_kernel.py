import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, scaling, hardtanh, and GELU
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_kernel(const float* x, const float* W, const float* b, float* out, int batch_size, int in_features, int out_features, float scaling_factor, float hardtanh_min, float hardtanh_max) {
    int batch = blockIdx.x;
    int out_f = blockIdx.y;
    if (batch >= batch_size || out_f >= out_features) return;
    
    float sum = b[out_f];
    for (int j = 0; j < in_features; j++) {
        sum += x[batch * in_features + j] * W[out_f * in_features + j];
    }
    sum *= scaling_factor;
    sum = fmaxf(fminf(sum, hardtanh_max), hardtanh_min);
    float gelu = 0.5f * sum * (1.0f + erff(sum / sqrtf(2.0f)));
    out[batch * out_features + out_f] = gelu;
}

torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b, float scaling_factor, float hardtanh_min, float hardtanh_max) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = W.size(0);
    auto out = torch::zeros({batch_size, out_features}, x.options());
    
    dim3 blocks(batch_size, out_features);
    fused_kernel<<<blocks, 1>>>(x.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max);
    
    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b, float scaling_factor, float hardtanh_min, float hardtanh_max);"
)

# Compile the inline CUDA code for the fused operation
fused_module = load_inline(
    name="fused_module",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs fused GEMM, scaling, hardtanh, and GELU in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        # Initialize weights and biases as parameters
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.randn(out_features))
        self.fused_forward = fused_module

    def forward(self, x):
        return self.fused_forward.fused_forward_cuda(x, self.W, self.b, self.scaling_factor, self.hardtanh_min, self.hardtanh_max)