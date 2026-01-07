import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused tanh, scaling, and bias addition
fused_tanh_scale_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_tanh_scale_bias_kernel(const float* x, float* out, float scale, const float* bias, int batch, int out_channels, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * h * w;
    if (idx < total) {
        int oc = (idx / (h * w)) % out_channels;
        float val = x[idx];
        out[idx] = tanhf(val) * scale + bias[oc];
    }
}

torch::Tensor fused_tanh_scale_bias_cuda(torch::Tensor x, float scale, torch::Tensor bias) {
    auto batch = x.size(0);
    auto out_channels = x.size(1);
    auto h = x.size(2);
    auto w = x.size(3);
    auto out = torch::empty_like(x);
    int total = x.numel();
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    fused_tanh_scale_bias_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), scale, bias.data_ptr<float>(), batch, out_channels, h, w);
    return out;
}
"""

fused_tanh_scale_bias_cpp_source = (
    "torch::Tensor fused_tanh_scale_bias_cuda(torch::Tensor x, float scale, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused tanh, scaling, and bias addition
fused_tanh_scale_bias = load_inline(
    name="fused_tanh_scale_bias",
    cpp_sources=fused_tanh_scale_bias_cpp_source,
    cuda_sources=fused_tanh_scale_bias_source,
    functions=["fused_tanh_scale_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that performs a convolution, applies fused tanh+scaling+bias, and then max-pools.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.fused_tanh_scale_bias = fused_tanh_scale_bias

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        # Fused tanh, scaling, and bias addition
        x = self.fused_tanh_scale_bias.fused_tanh_scale_bias_cuda(x, self.scaling_factor, self.bias)
        # Max-pooling
        x = self.max_pool(x)
        return x