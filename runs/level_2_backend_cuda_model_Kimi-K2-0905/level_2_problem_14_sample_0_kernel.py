import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + div + sum + scale
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16

__global__ void fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int batch_size,
    int input_size,
    int hidden_size,
    float scale_div,
    float scale_mul) {

    // Each thread computes one element of the output (after sum)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            float dot = 0.0f;
            for (int k = 0; k < input_size; k++) {
                dot += x[row * input_size + k] * weight[j * input_size + k];
            }
            sum += dot * scale_div;
        }
        out[row] = sum * scale_mul;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    float scaling_factor) {

    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);

    auto out = torch::zeros({batch_size, 1}, x.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        2.0f,
        scaling_factor);

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that fuses matrix multiplication, division, summation, and scaling into a single CUDA kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(x, self.weight, self.scaling_factor)


def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]