import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + scaling + hardtanh + GELU
fused_gemm_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_gemm_activation_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    float scaling_factor, float hardtanh_min, float hardtanh_max) {
    
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        
        // Compute GEMM
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        
        // Add bias if exists
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Apply scaling
        sum *= scaling_factor;
        
        // Apply hardtanh
        sum = fmaxf(fminf(sum, hardtanh_max), hardtanh_min);
        
        // Apply GELU
        output[row * out_features + col] = gelu(sum);
    }
}

torch::Tensor fused_gemm_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, float hardtanh_min, float hardtanh_max) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    
    fused_gemm_activation_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_features, out_features,
        scaling_factor, hardtanh_min, hardtanh_max);
    
    return output;
}
"""

fused_gemm_activation_cpp_source = """
torch::Tensor fused_gemm_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, float hardtanh_min, float hardtanh_max);
"""

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
    Optimized Model that performs a fused GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_op = fused_gemm_activation

    def forward(self, x):
        return self.fused_op.fused_gemm_activation_cuda(
            x, self.weight, self.bias,
            self.scaling_factor, self.hardtanh_min, self.hardtanh_max
        )

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]