import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for bias subtraction and tanh
bias_sub_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void bias_sub_tanh_kernel(float* x, const float* bias, int N, int C, int H, int W, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int nchw = idx;
        int w = nchw % W;
        nchw /= W;
        int h = nchw % H;
        nchw /= H;
        int c = nchw % C;
        nchw /= C;
        int n = nchw;
        x[idx] = tanhf(x[idx] - bias[c]);
    }
}

torch::Tensor bias_sub_tanh_cuda(torch::Tensor x, torch::Tensor bias) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto total = x.numel();

    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    bias_sub_tanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), N, C, H, W, total);

    return x;
}
"""

bias_sub_tanh_cpp_source = (
    "torch::Tensor bias_sub_tanh_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for bias subtraction and tanh
bias_sub_tanh = load_inline(
    name="bias_sub_tanh",
    cpp_sources=bias_sub_tanh_cpp_source,
    cuda_sources=bias_sub_tanh_source,
    functions=["bias_sub_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.bias_sub_tanh = bias_sub_tanh

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bias_sub_tanh.bias_sub_tanh_cuda(x, self.bias)
        return x