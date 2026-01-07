import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GEMM + Scale + BatchNorm
fused_gemm_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define BLOCK_SIZE 256

__global__ void fused_gemm_scale_bn_kernel(
    const float* input, const float* weight, const float* bias,
    const float* scale, const float* bn_weight, const float* bn_bias,
    const float* running_mean, const float* running_var,
    float* output, int batch_size, int in_features, int out_features,
    float eps) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        sum *= scale[col];
        
        float mean = running_mean[col];
        float var = running_var[col];
        float normalized = (sum - mean) / sqrtf(var + eps);
        
        output[row * out_features + col] = normalized * bn_weight[col] + bn_bias[col];
    }
}

torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor scale, torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor running_mean, torch::Tensor running_var, float eps) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 blockSize(16, 16);
    dim3 gridSize((out_features + blockSize.x - 1) / blockSize.x,
                  (batch_size + blockSize.y - 1) / blockSize.y);
    
    fused_gemm_scale_bn_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        scale.data_ptr<float>(), bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(), running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_features, out_features, eps);
    
    return output;
}
"""

fused_gemm_scale_bn_cpp_source = """
torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor scale, torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor running_mean, torch::Tensor running_var, float eps);
"""

# Compile the inline CUDA code
fused_gemm_scale_bn = load_inline(
    name="fused_gemm_scale_bn",
    cpp_sources=fused_gemm_scale_bn_cpp_source,
    cuda_sources=fused_gemm_scale_bn_source,
    functions=["fused_gemm_scale_bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.eps = eps
        self.momentum = momentum
        self.fused_op = fused_gemm_scale_bn

    def forward(self, x):
        return self.fused_op.fused_gemm_scale_bn_cuda(
            x, self.weight, self.bias, self.scale,
            self.bn_weight, self.bn_bias,
            self.running_mean, self.running_var, self.eps)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scale_shape]