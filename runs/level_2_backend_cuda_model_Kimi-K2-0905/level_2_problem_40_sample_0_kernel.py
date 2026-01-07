import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + scaling + residual
fused_matmul_scale_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void fused_matmul_scale_residual_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    float scaling_factor) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        
        // Add bias if not null
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Apply scaling and residual connection
        float residual = sum;
        sum = sum * scaling_factor + residual;
        
        output[row * out_features + col] = sum;
    }
}

torch::Tensor fused_matmul_scale_residual_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    fused_matmul_scale_residual_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_features, out_features,
        scaling_factor);
    
    return output;
}
"""

fused_matmul_scale_residual_cpp_source = (
    "torch::Tensor fused_matmul_scale_residual_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "float scaling_factor);"
)

# Compile the inline CUDA code
fused_matmul_scale_residual = load_inline(
    name="fused_matmul_scale_residual",
    cpp_sources=fused_matmul_scale_residual_cpp_source,
    cuda_sources=fused_matmul_scale_residual_source,
    functions=["fused_matmul_scale_residual_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul + scaling + residual.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.fused_op = fused_matmul_scale_residual

    def forward(self, x):
        return self.fused_op.fused_matmul_scale_residual_cuda(
            x, self.weight, self.bias, self.scaling_factor)

batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]