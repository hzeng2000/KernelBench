import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear, subtract, multiply, and ReLU
fused_linear_sub_mul_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sub_mul_relu_kernel(const float* x, const float* weight, const float* bias, float* out, int batch_size, int in_features, int out_features, float subtract_value, float multiply_value) {
    int batch = blockIdx.x;
    int out_f = blockIdx.y;
    if (batch >= batch_size || out_f >= out_features) return;
    
    float sum = bias[out_f];
    for (int in_f = 0; in_f < in_features; ++in_f) {
        sum += x[batch * in_features + in_f] * weight[out_f * in_features + in_f];
    }
    sum = sum - subtract_value;
    sum = sum * multiply_value;
    out[batch * out_features + out_f] = fmaxf(sum, 0.0f);
}

torch::Tensor fused_linear_sub_mul_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float subtract_value, float multiply_value) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    auto out = torch::zeros({batch_size, out_features}, x.options());
    
    dim3 blocks(batch_size, out_features);
    fused_linear_sub_mul_relu_kernel<<<blocks, 1>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, out_features, subtract_value, multiply_value);
    
    return out;
}
"""

fused_linear_sub_mul_relu_cpp_source = (
    "torch::Tensor fused_linear_sub_mul_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float subtract_value, float multiply_value);"
)

# Compile the inline CUDA code for fused operation
fused_op = load_inline(
    name="fused_linear_sub_mul_relu",
    cpp_sources=fused_linear_sub_mul_relu_cpp_source,
    cuda_sources=fused_linear_sub_mul_relu_source,
    functions=["fused_linear_sub_mul_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs fused matrix multiplication, subtraction, multiplication, and ReLU activation in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_linear_sub_mul_relu_cuda(x, self.linear.weight, self.linear.bias, self.subtract_value, self.multiply_value)