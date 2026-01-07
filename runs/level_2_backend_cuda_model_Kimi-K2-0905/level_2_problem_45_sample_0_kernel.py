import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused linear + sigmoid
fused_linear_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sigmoid_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int input_size, int output_size) {
    
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
        // Sigmoid activation
        output[row * output_size + col] = 1.0f / (1.0f + expf(-sum));
    }
}

torch::Tensor fused_linear_sigmoid_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size((output_size + block_size.x - 1) / block_size.x,
                   (batch_size + block_size.y - 1) / block_size.y);
    
    fused_linear_sigmoid_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, input_size, output_size);
    
    return output;
}
"""

fused_linear_sigmoid_cpp_source = (
    "torch::Tensor fused_linear_sigmoid_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Custom CUDA kernel for linear + online logsumexp
fused_linear_logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_linear_logsumexp_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int input_size, int output_size) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size) {
        float max_val = -FLT_MAX;
        
        // First pass: compute max for numerical stability
        for (int col = 0; col < output_size; col++) {
            float sum = 0.0f;
            for (int i = 0; i < input_size; i++) {
                sum += input[row * input_size + i] * weight[col * input_size + i];
            }
            if (bias != nullptr) {
                sum += bias[col];
            }
            max_val = fmaxf(max_val, sum);
        }
        
        // Second pass: compute exp and sum
        float sum_exp = 0.0f;
        for (int col = 0; col < output_size; col++) {
            float sum = 0.0f;
            for (int i = 0; i < input_size; i++) {
                sum += input[row * input_size + i] * weight[col * input_size + i];
            }
            if (bias != nullptr) {
                sum += bias[col];
            }
            sum_exp += expf(sum - max_val);
        }
        
        output[row] = logf(sum_exp) + max_val;
    }
}

torch::Tensor fused_linear_logsumexp_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    
    fused_linear_logsumexp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, input_size, output_size);
    
    return output;
}
"""

fused_linear_logsumexp_cpp_source = (
    "torch::Tensor fused_linear_logsumexp_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_linear_sigmoid = load_inline(
    name="fused_linear_sigmoid",
    cpp_sources=fused_linear_sigmoid_cpp_source,
    cuda_sources=fused_linear_sigmoid_source,
    functions=["fused_linear_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

fused_linear_logsumexp = load_inline(
    name="fused_linear_logsumexp",
    cpp_sources=fused_linear_logsumexp_cpp_source,
    cuda_sources=fused_linear_logsumexp_source,
    functions=["fused_linear_logsumexp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_linear_sigmoid = fused_linear_sigmoid
        self.fused_linear_logsumexp = fused_linear_logsumexp

    def forward(self, x):
        x = self.fused_linear_sigmoid.fused_linear_sigmoid_cuda(x, self.linear1.weight, self.linear1.bias)
        x = self.fused_linear_logsumexp.fused_linear_logsumexp_cuda(x, self.linear2.weight, self.linear2.bias)
        return x

batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, output_size]