import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused matmul + batchnorm + bias + div + swish
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float divide_value,
    int batch_size,
    int in_features,
    int out_features,
    float bn_eps)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        // Matmul
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        // BatchNorm
        float bn_out = (sum - bn_mean[col]) / sqrtf(bn_var[col] + bn_eps);
        bn_out = bn_out * bn_weight[col] + bn_bias[col];
        
        // Bias + Div + Swish
        bn_out = bn_out + bias[0];
        bn_out = bn_out / divide_value;
        float sigmoid = 1.0f / (1.0f + expf(-bn_out));
        output[row * out_features + col] = bn_out * sigmoid;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    float divide_value)
{
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    const float bn_eps = 1e-5f;
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    
    fused_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        divide_value,
        batch_size,
        in_features,
        out_features,
        bn_eps);
    
    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    float divide_value);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divide_value = divide_value
        
        # Linear weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.linear_bias = nn.Parameter(torch.zeros(out_features))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('bn_mean', torch.zeros(out_features))
        self.register_buffer('bn_var', torch.ones(out_features))
        
        # Additional bias
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self.fused_op = fused_op
        
    def forward(self, x):
        return self.fused_op.fused_op_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.bn_weight, 
            self.bn_bias, 
            self.bn_mean, 
            self.bn_var, 
            self.divide_value
        )

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]