import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min and subtract
min_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_subtract_kernel(const float* x, float c, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fminf(x[idx] - c, 0.0f);
    }
}

torch::Tensor min_subtract_cuda(torch::Tensor x, float c) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    min_subtract_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), c, out.data_ptr<float>(), size);

    return out;
}
"""

min_subtract_cpp_source = (
    "torch::Tensor min_subtract_cuda(torch::Tensor x, float c);"
)

# Compile the inline CUDA code for min and subtract
min_subtract = load_inline(
    name="min_subtract",
    cpp_sources=min_subtract_cpp_source,
    cuda_sources=min_subtract_source,
    functions=["min_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies minimum, and subtracts a constant using a fused CUDA kernel for min and subtract.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.min_subtract = min_subtract

    def forward(self, x):
        x = self.linear(x)
        x = self.min_subtract.min_subtract_cuda(x, self.constant.item())
        return x