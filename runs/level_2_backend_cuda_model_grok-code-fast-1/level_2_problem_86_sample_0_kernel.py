import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear, divide, and GELU
fused_linear_div_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define M_PI 3.14159265358979323846

__global__ void fused_linear_div_gelu_kernel(const float* x, const float* weight, const float* bias, float divisor, float* out, int batch_size, int input_size, int output_size) {
    int batch = blockIdx.x;
    int out_idx = blockIdx.y;
    if (batch >= batch_size || out_idx >= output_size) return;

    float sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
        sum += x[batch * input_size + i] * weight[out_idx * input_size + i];
    }
    sum += bias[out_idx];
    sum /= divisor;

    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x_val = sum;
    float coeff = sqrtf(2.0f / M_PI);
    float tanh_arg = coeff * (x_val + 0.044715f * x_val * x_val * x_val);
    sum = 0.5f * x_val * (1.0f + tanhf(tanh_arg));

    out[batch * output_size + out_idx] = sum;
}

torch::Tensor fused_linear_div_gelu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float divisor) {
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int output_size = weight.size(0);
    auto out = torch::zeros({batch_size, output_size}, x.options());

    dim3 blocks(batch_size, output_size);
    fused_linear_div_gelu_kernel<<<blocks, 1>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), divisor, out.data_ptr<float>(), batch_size, input_size, output_size);

    return out;
}
"""

fused_linear_div_gelu_cpp_source = (
    "torch::Tensor fused_linear_div_gelu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float divisor);"
)

# Compile the inline CUDA code for fused linear, divide, and GELU
fused_linear_div_gelu = load_inline(
    name="fused_linear_div_gelu",
    cpp_sources=fused_linear_div_gelu_cpp_source,
    cuda_sources=fused_linear_div_gelu_source,
    functions=["fused_linear_div_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    A model that performs a fused matrix multiplication, divides by a scalar, and applies GELU activation using a custom CUDA operator.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_op = fused_linear_div_gelu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.fused_op.fused_linear_div_gelu_cuda(x, self.linear.weight, self.linear.bias, self.divisor)