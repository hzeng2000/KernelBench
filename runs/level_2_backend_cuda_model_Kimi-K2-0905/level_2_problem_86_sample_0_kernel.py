import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + div + gelu
fused_linear_div_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_linear_div_gelu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int input_size, int output_size, float divisor) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[row * input_size + i] * weight[col * input_size + i];
        }
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Divide by divisor
        sum /= divisor;
        
        // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x = sum;
        float x3 = x * x * x;
        float tanh_arg = 0.7978845608f * (x + 0.044715f * x3);
        
        // Fast tanh approximation
        float tanh_val;
        if (tanh_arg > 5.0f) {
            tanh_val = 1.0f;
        } else if (tanh_arg < -5.0f) {
            tanh_val = -1.0f;
        } else {
            float exp2x = expf(2.0f * tanh_arg);
            tanh_val = (exp2x - 1.0f) / (exp2x + 1.0f);
        }
        
        float gelu = 0.5f * x * (1.0f + tanh_val);
        output[row * output_size + col] = gelu;
    }
}

torch::Tensor fused_linear_div_gelu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size((output_size + block_size.x - 1) / block_size.x,
                   (batch_size + block_size.y - 1) / block_size.y);
    
    fused_linear_div_gelu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, input_size, output_size, divisor);
    
    return output;
}
"""

fused_linear_div_gelu_cpp_source = (
    "torch::Tensor fused_linear_div_gelu_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_linear_div_gelu_cpp_source,
    cuda_sources=fused_linear_div_gelu_source,
    functions=["fused_linear_div_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    A model that performs a fused matrix multiplication, division by scalar, and GELU activation.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.fused_ops.fused_linear_div_gelu_cuda(
            x, self.linear.weight, self.linear.bias, self.divisor
        )