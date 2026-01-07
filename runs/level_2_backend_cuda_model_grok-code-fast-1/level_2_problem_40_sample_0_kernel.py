import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scale and add
fused_scale_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_scale_add_kernel(const float* a, float* out, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * scale + a[idx];
    }
}

torch::Tensor fused_scale_add_cuda(torch::Tensor a, float scale) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_scale_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), scale, size);

    return out;
}
"""

fused_scale_add_cpp_source = (
    "torch::Tensor fused_scale_add_cuda(torch::Tensor a, float scale);"
)

# Compile the inline CUDA code for fused scale and add
fused_scale_add = load_inline(
    name="fused_scale_add",
    cpp_sources=fused_scale_add_cpp_source,
    cuda_sources=fused_scale_add_source,
    functions=["fused_scale_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, scaling, and residual addition.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_scale_add = fused_scale_add

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.fused_scale_add.fused_scale_add_cuda(x, self.scaling_factor)
        return x