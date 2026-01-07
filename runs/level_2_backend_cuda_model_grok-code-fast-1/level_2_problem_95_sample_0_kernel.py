import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activations: Swish -> Tanh -> GELU -> Hardtanh
fused_activations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_activations_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Swish: sigmoid(val) * val
        val = (1.0f / (1.0f + expf(-val))) * val;
        // Tanh
        val = tanhf(val);
        // GELU: 0.5 * val * (1 + erf(val / sqrt(2)))
        val = 0.5f * val * (1.0f + erff(val * 0.7071067811865476f));
        // Hardtanh: clamp to [-1, 1]
        val = fmaxf(-1.0f, fminf(1.0f, val));
        out[idx] = val;
    }
}

torch::Tensor fused_activations_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_activations_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_activations_cpp_source = (
    "torch::Tensor fused_activations_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for fused activations
fused_activations = load_inline(
    name="fused_activations",
    cpp_sources=fused_activations_cpp_source,
    cuda_sources=fused_activations_source,
    functions=["fused_activations_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, adds a value, and applies fused Swish, Tanh, GELU, and Hardtanh activation functions using a custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.fused_activations = fused_activations

    def forward(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = self.fused_activations.fused_activations_cuda(x)
        return x