import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused linear + instance norm + residual operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_linear_instancenorm_kernel(
    const float* input, const float* weight, const float* bias,
    const float* residual, float* output,
    int batch_size, int in_features, int out_features,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int batch_idx = idx / out_features;
        int feature_idx = idx % out_features;
        
        // Compute linear transformation
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[batch_idx * in_features + i] * weight[feature_idx * in_features + i];
        }
        sum += bias[feature_idx];
        
        // Store intermediate result for instance norm
        output[idx] = sum;
    }
    
    __syncthreads();
    
    // Compute mean and variance for instance norm (per sample)
    if (idx < batch_size) {
        float mean = 0.0f;
        for (int i = 0; i < out_features; i++) {
            mean += output[idx * out_features + i];
        }
        mean /= out_features;
        
        float var = 0.0f;
        for (int i = 0; i < out_features; i++) {
            float diff = output[idx * out_features + i] - mean;
            var += diff * diff;
        }
        var /= out_features;
        
        // Apply instance normalization and residual operations
        for (int i = 0; i < out_features; i++) {
            int output_idx = idx * out_features + i;
            float normalized = (output[output_idx] - mean) / sqrtf(var + eps);
            float residual_val = residual[output_idx];
            output[output_idx] = normalized * residual_val + residual_val;
        }
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor residual, float eps) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;
    
    fused_linear_instancenorm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        residual.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_features, out_features, eps);
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "torch::Tensor residual, float eps);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.fused_ops = fused_ops
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, y):
        return self.fused_ops.fused_ops_cuda(x, self.weight, self.bias, y, self.eps)