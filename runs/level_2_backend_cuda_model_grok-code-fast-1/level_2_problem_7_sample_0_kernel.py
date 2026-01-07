import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activations and bias addition
fused_activations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_activations_kernel(const float* x, const float* bias, float* out, int size, int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / spatial) % C;
        float val = x[idx];
        // ReLU
        val = fmaxf(0.0f, val);
        // LeakyReLU
        val = val > 0.0f ? val : 0.01f * val;
        // GELU approximation
        float sqrt_2_over_pi = sqrtf(2.0f / 3.141592653589793f);
        float coeff = 0.044715f;
        float tanh_arg = sqrt_2_over_pi * (val + coeff * val * val * val);
        val = 0.5f * val * (1.0f + tanhf(tanh_arg));
        // Sigmoid
        val = 1.0f / (1.0f + expf(-val));
        // Add bias
        val += bias[c];
        out[idx] = val;
    }
}

torch::Tensor fused_activations_cuda(torch::Tensor x, torch::Tensor bias) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int spatial = D * H * W;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_activations_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), size, C, spatial);
    return out;
}
"""

fused_activations_cpp_source = (
    "torch::Tensor fused_activations_cuda(torch::Tensor x, torch::Tensor bias);"
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
    Optimized Model that performs a 3D convolution, then applies fused ReLU, LeakyReLU, GELU, Sigmoid activations, and bias addition in a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_activations = fused_activations

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activations.fused_activations_cuda(x, self.bias)
        return x