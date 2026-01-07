import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for pooling and GELU
pool_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}

__global__ void pool_gelu_kernel(const float* x, float* y, int batch_size, int out_features) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    float sum = 0.0f;
    for (int i = 0; i < out_features; i++) {
        sum += x[batch * out_features + i];
    }
    float mean = sum / out_features;
    y[batch] = gelu(mean);
}

torch::Tensor pool_gelu_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int out_features = x.size(1);
    auto y = torch::zeros({batch_size, 1}, x.options());
    pool_gelu_kernel<<<batch_size, 1>>>(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, out_features);
    return y;
}
"""

pool_gelu_cpp_source = (
    "torch::Tensor pool_gelu_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for pooling and GELU
pool_gelu = load_inline(
    name="pool_gelu",
    cpp_sources=pool_gelu_cpp_source,
    cuda_sources=pool_gelu_source,
    functions=["pool_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for residual add
residual_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void residual_add_kernel(const float* y, const float* original_x, float* out, int batch_size, int in_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_features;
    if (idx < total) {
        int batch = idx / in_features;
        out[idx] = y[batch] + original_x[idx];
    }
}

torch::Tensor residual_add_cuda(torch::Tensor y, torch::Tensor original_x) {
    int batch_size = y.size(0);
    int in_features = original_x.size(1);
    auto out = torch::zeros_like(original_x);
    int total = batch_size * in_features;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    residual_add_kernel<<<num_blocks, block_size>>>(y.data_ptr<float>(), original_x.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features);
    return out;
}
"""

residual_add_cpp_source = (
    "torch::Tensor residual_add_cuda(torch::Tensor y, torch::Tensor original_x);"
)

# Compile the inline CUDA code for residual add
residual_add = load_inline(
    name="residual_add",
    cpp_sources=residual_add_cpp_source,
    cuda_sources=residual_add_source,
    functions=["residual_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.pool_gelu = pool_gelu
        self.residual_add = residual_add

    def forward(self, x):
        original_x = x.clone().detach()
        # Gemm
        x = self.gemm(x)
        # Subtract
        x = x - self.subtract
        # GlobalAvgPool, LogSumExp, GELU
        x = self.pool_gelu.pool_gelu_cuda(x)
        # ResidualAdd
        x = self.residual_add.residual_add_cuda(x, original_x)
        return x