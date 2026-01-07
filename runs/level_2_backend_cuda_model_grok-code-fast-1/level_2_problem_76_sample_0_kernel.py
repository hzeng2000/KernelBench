import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias addition and ReLU
fused_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_relu_kernel(const float* x, const float* bias, float* out, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx < total) {
        int feature = idx % out_features;
        out[idx] = fmaxf(0.0f, x[idx] + bias[feature]);
    }
}

torch::Tensor fused_bias_relu_cuda(torch::Tensor x, torch::Tensor bias) {
    auto batch_size = x.size(0);
    auto out_features = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (x.numel() + block_size - 1) / block_size;

    fused_bias_relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, out_features);

    return out;
}
"""

fused_bias_relu_cpp_source = (
    "torch::Tensor fused_bias_relu_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused bias addition and ReLU
fused_bias_relu = load_inline(
    name="fused_bias_relu",
    cpp_sources=fused_bias_relu_cpp_source,
    cuda_sources=fused_bias_relu_source,
    functions=["fused_bias_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, then applies fused bias addition and ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_bias_relu = fused_bias_relu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.fused_bias_relu.fused_bias_relu_cuda(x, self.bias)
        return x