import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for linear transformation
linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_kernel(const float* x, const float* W, const float* b, float* y, int batch, int in_features, int out_features) {
    int b_idx = blockIdx.x;
    int o = blockIdx.y;
    if (b_idx < batch && o < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += x[b_idx * in_features + i] * W[o * in_features + i];
        }
        y[b_idx * out_features + o] = sum + b[o];
    }
}

torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b) {
    auto batch = x.size(0);
    auto in_features = x.size(1);
    auto out_features = W.size(0);
    auto y = torch::zeros({batch, out_features}, x.options());
    dim3 blocks(batch, out_features);
    linear_kernel<<<blocks, 1>>>(x.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), batch, in_features, out_features);
    return y;
}
"""

linear_cpp_source = "torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b);"

# Compile the inline CUDA code for linear
linear = load_inline(
    name="linear",
    cpp_sources=linear_cpp_source,
    cuda_sources=linear_source,
    functions=["linear_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel1(const float* x, float* max_val, float* sum_exp, int batch, int out_features) {
    int b = blockIdx.x;
    if (b < batch) {
        float maxv = -INFINITY;
        for (int o = 0; o < out_features; o++) {
            maxv = fmaxf(maxv, x[b * out_features + o]);
        }
        max_val[b] = maxv;
        float sum = 0.0f;
        for (int o = 0; o < out_features; o++) {
            sum += expf(x[b * out_features + o] - maxv);
        }
        sum_exp[b] = sum;
    }
}

__global__ void softmax_kernel2(float* x, const float* max_val, const float* sum_exp, int batch, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * out_features) {
        int b = idx / out_features;
        x[idx] = expf(x[idx] - max_val[b]) / sum_exp[b];
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto batch = x.size(0);
    auto out_features = x.size(1);
    auto max_val = torch::zeros(batch, x.options());
    auto sum_exp = torch::zeros(batch, x.options());
    softmax_kernel1<<<batch, 1>>>(x.data_ptr<float>(), max_val.data_ptr<float>(), sum_exp.data_ptr<float>(), batch, out_features);
    int total = batch * out_features;
    const int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    softmax_kernel2<<<num_blocks, block_size>>>(x.data_ptr<float>(), max_val.data_ptr<float>(), sum_exp.data_ptr<float>(), batch, out_features);
    return x;
}
"""

softmax_cpp_source = "torch::Tensor softmax_cuda(torch::Tensor x);"

# Compile the inline CUDA code for softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.randn(out_features))
        self.dropout = nn.Dropout(dropout_p)
        self.linear = linear
        self.softmax = softmax

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.linear.linear_cuda(x, self.W, self.b)
        x = self.dropout(x)
        x = self.softmax.softmax_cuda(x)
        return x