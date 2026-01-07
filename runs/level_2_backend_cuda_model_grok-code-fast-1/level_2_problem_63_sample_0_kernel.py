import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ReLU and division
relu_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_div_kernel(const float* x, float* out, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = fmaxf(0.0f, val) / divisor;
    }
}

torch::Tensor relu_div_cuda(torch::Tensor x, float divisor) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_div_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), divisor, size);

    return out;
}
"""

relu_div_cpp_source = (
    "torch::Tensor relu_div_cuda(torch::Tensor x, float divisor);"
)

# Compile the inline CUDA code for fused ReLU and division
relu_div = load_inline(
    name="relu_div",
    cpp_sources=relu_div_cpp_source,
    cuda_sources=relu_div_source,
    functions=["relu_div_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies fused ReLU and division.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self.relu_div = relu_div

    def forward(self, x):
        x = self.linear(x)
        return self.relu_div.relu_div_cuda(x, self.divisor)