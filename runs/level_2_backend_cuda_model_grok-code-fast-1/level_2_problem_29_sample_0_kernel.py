import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused mish applied twice
mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float softplus(float x) {
    if (x > 0.0f) {
        return x + logf(1.0f + expf(-x));
    } else {
        return logf(1.0f + expf(x));
    }
}

__device__ float mish(float x) {
    float sp = softplus(x);
    return x * tanhf(sp);
}

__global__ void mish_mish_kernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        float mish1 = mish(x);
        float mish2 = mish(mish1);
        out[idx] = mish2;
    }
}

torch::Tensor mish_mish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

mish_mish_cpp_source = (
    "torch::Tensor mish_mish_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for fused mish mish
mish_mish = load_inline(
    name="mish_mish",
    cpp_sources=mish_mish_cpp_source,
    cuda_sources=mish_mish_source,
    functions=["mish_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication and applies fused Mish twice.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mish_mish = mish_mish

    def forward(self, x):
        x = self.linear(x)
        x = self.mish_mish.mish_mish_cuda(x)
        return x