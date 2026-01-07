import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear and sum
fused_linear_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sum_kernel(const float* x, const float* v, float bias, float* out, int batch_size, int in_features) {
    int i = blockIdx.x;
    if (i >= batch_size) return;
    float sum = 0.0f;
    for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
        sum += x[i * in_features + k] * v[k];
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[i] = sdata[0] + bias;
    }
}

torch::Tensor fused_linear_sum_cuda(torch::Tensor x, torch::Tensor v, float bias) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    auto out = torch::zeros({batch_size, 1}, x.options());
    const int block_size = 256;
    const int num_blocks = batch_size;
    fused_linear_sum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), v.data_ptr<float>(), bias, out.data_ptr<float>(), batch_size, in_features);
    return out;
}
"""

fused_linear_sum_cpp_source = (
    "torch::Tensor fused_linear_sum_cuda(torch::Tensor x, torch::Tensor v, float bias);"
)

# Compile the inline CUDA code for fused linear and sum
fused_linear_sum = load_inline(
    name="fused_linear_sum",
    cpp_sources=fused_linear_sum_cpp_source,
    cuda_sources=fused_linear_sum_source,
    functions=["fused_linear_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a sequence of operations with fused linear and sum using custom CUDA.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.sum_weight = self.linear.weight.sum(dim=0)
        self.sum_bias = self.linear.bias.sum()
        self.fused_linear_sum = fused_linear_sum

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.fused_linear_sum.fused_linear_sum_cuda(x, self.sum_weight, self.sum_bias)  # (batch_size, 1)
        x = torch.max(x, dim=1, keepdim=True)[0]  # (batch_size, 1)
        x = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True)  # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True)  # (batch_size, 1)
        return x