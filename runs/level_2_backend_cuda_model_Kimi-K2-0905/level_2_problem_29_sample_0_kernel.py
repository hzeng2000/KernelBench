import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + mish + mish
fused_linear_mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float mish_activation(float x) {
    return x * tanhf(fmaxf(0.0f, x) + logf(1.0f + expf(-fabsf(x))));
}

__global__ void fused_linear_mish_mish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Apply Mish twice
        sum = mish_activation(sum);
        sum = mish_activation(sum);
        
        output[row * out_features + col] = sum;
    }
}

torch::Tensor fused_linear_mish_mish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size((out_features + block_size.x - 1) / block_size.x,
                   (batch_size + block_size.y - 1) / block_size.y);
    
    fused_linear_mish_mish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_features, out_features);
    
    return output;
}
"""

fused_linear_mish_mish_cpp_source = (
    "torch::Tensor fused_linear_mish_mish_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_linear_mish_mish = load_inline(
    name="fused_linear_mish_mish",
    cpp_sources=fused_linear_mish_mish_cpp_source,
    cuda_sources=fused_linear_mish_mish_source,
    functions=["fused_linear_mish_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_op = fused_linear_mish_mish

    def forward(self, x):
        return self.fused_op.fused_linear_mish_mish_cuda(
            x, self.linear.weight, self.linear.bias)

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features]