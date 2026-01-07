import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused linear + GELU + softmax
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_linear_gelu_softmax_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        
        // Matrix multiplication
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        
        // Add bias
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // GELU activation
        float gelu_out = 0.5f * sum * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
        
        // Store intermediate result for softmax
        output[row * out_features + col] = gelu_out;
    }
}

__global__ void softmax_kernel(float* output, int batch_size, int out_features) {
    int row = blockIdx.x;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < out_features; i++) {
        max_val = fmaxf(max_val, output[row * out_features + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < out_features; i++) {
        output[row * out_features + i] = expf(output[row * out_features + i] - max_val);
        sum += output[row * out_features + i];
    }
    
    // Normalize
    for (int i = 0; i < out_features; i++) {
        output[row * out_features + i] /= sum;
    }
}

torch::Tensor fused_linear_gelu_softmax_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 block_size(256);
    dim3 grid_size((out_features + block_size.x - 1) / block_size.x, batch_size);
    
    fused_linear_gelu_softmax_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_features, out_features);
    
    // Launch softmax kernel
    softmax_kernel<<<batch_size, 1>>>(output.data_ptr<float>(), batch_size, out_features);
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_linear_gelu_softmax_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_linear_gelu_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.fused_ops = fused_ops
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_ops.fused_linear_gelu_softmax_cuda(x, self.weight, self.bias)