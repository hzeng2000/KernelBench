import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused subtract and Mish activation
fused_sub_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_sub_mish_kernel(const float* in, float* out, float sub1, float sub2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = in[idx] - sub1 - sub2;
        float sp = logf(1.0f + expf(val));
        float tanh_sp = tanhf(sp);
        out[idx] = val * tanh_sp;
    }
}

torch::Tensor fused_sub_mish_cuda(torch::Tensor x, float sub1, float sub2) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_sub_mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), sub1, sub2, size);

    return out;
}
"""

fused_sub_mish_cpp_source = (
    "torch::Tensor fused_sub_mish_cuda(torch::Tensor x, float sub1, float sub2);"
)

# Compile the inline CUDA code for fused subtract and Mish
fused_sub_mish = load_inline(
    name="fused_sub_mish",
    cpp_sources=fused_sub_mish_cpp_source,
    cuda_sources=fused_sub_mish_source,
    functions=["fused_sub_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, then fused subtract and Mish activation using custom CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.fused_sub_mish = fused_sub_mish

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_sub_mish.fused_sub_mish_cuda(x, self.subtract_value_1, self.subtract_value_2)
        return x