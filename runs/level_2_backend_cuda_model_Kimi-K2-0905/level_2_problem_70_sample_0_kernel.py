import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused gemm + sigmoid + scaling + residual add
fused_gemm_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void fused_gemm_activation_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int input_size, int hidden_size,
    float scaling_factor) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < hidden_size) {
        float sum = 0.0f;
        
        // Compute matrix multiplication
        for (int i = 0; i < input_size; i++) {
            sum += input[row * input_size + i] * weight[col * input_size + i];
        }
        
        // Add bias if it exists
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Apply sigmoid
        float sigmoid_val = 1.0f / (1.0f + expf(-sum));
        
        // Apply scaling and residual add
        output[row * hidden_size + col] = sum + sigmoid_val * scaling_factor;
    }
}

torch::Tensor fused_gemm_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, hidden_size}, input.options());
    
    dim3 blockDim(16, 16);
    dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                 (batch_size + blockDim.y - 1) / blockDim.y);
    
    fused_gemm_activation_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, input_size, hidden_size,
        scaling_factor);
    
    return output;
}
"""

fused_gemm_activation_cpp_source = (
    "torch::Tensor fused_gemm_activation_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "float scaling_factor);"
)

# Compile the inline CUDA code
fused_gemm_activation = load_inline(
    name="fused_gemm_activation",
    cpp_sources=fused_gemm_activation_cpp_source,
    cuda_sources=fused_gemm_activation_source,
    functions=["fused_gemm_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd" with fused CUDA kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_gemm_activation

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        return self.fused_op.fused_gemm_activation_cuda(x, self.weight, self.bias, self.scaling_factor)