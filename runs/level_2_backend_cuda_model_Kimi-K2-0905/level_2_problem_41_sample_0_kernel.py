import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GEMM + BatchNorm + GELU + ReLU
fused_gemm_bn_gelu_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 256

__global__ void fused_gemm_bn_gelu_relu_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias,
    const float* bn_mean, const float* bn_var,
    float* output, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        
        // GEMM
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        sum += bias[col];
        
        // BatchNorm
        float bn_scale = bn_weight[col] / sqrtf(bn_var[col] + 1e-5f);
        float bn_shift = bn_bias[col] - bn_scale * bn_mean[col];
        sum = bn_scale * sum + bn_shift;
        
        // GELU
        float gelu = 0.5f * sum * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
        
        // ReLU
        output[row * out_features + col] = fmaxf(0.0f, gelu);
    }
}

torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size((out_features + 15) / 16, (batch_size + 15) / 16);
    
    fused_gemm_bn_gelu_relu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features);
    
    return output;
}
"""

fused_gemm_bn_gelu_relu_cpp_source = (
    "torch::Tensor fused_gemm_bn_gelu_relu_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "torch::Tensor bn_weight, torch::Tensor bn_bias,"
    "torch::Tensor bn_mean, torch::Tensor bn_var);"
)

# Compile the inline CUDA code
fused_gemm_bn_gelu_relu = load_inline(
    name="fused_gemm_bn_gelu_relu",
    cpp_sources=fused_gemm_bn_gelu_relu_cpp_source,
    cuda_sources=fused_gemm_bn_gelu_relu_source,
    functions=["fused_gemm_bn_gelu_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        
        self.register_buffer('bn_mean', torch.zeros(out_features))
        self.register_buffer('bn_var', torch.ones(out_features))
        
        self.fused_op = fused_gemm_bn_gelu_relu

    def forward(self, x):
        return self.fused_op.fused_gemm_bn_gelu_relu_cuda(
            x, self.weight, self.bias,
            self.bn_weight, self.bn_bias,
            self.bn_mean, self.bn_var)

batch_size = 16384
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]