import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* x, const float* weight_sum, float* out, float scaling_factor, int batch_size, int input_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += x[i * input_size + j] * weight_sum[j];
        }
        out[i] = (scaling_factor / 2.0f) * sum;
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight_sum, float scaling_factor) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto out = torch::zeros({batch_size, 1}, x.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), weight_sum.data_ptr<float>(), out.data_ptr<float>(), scaling_factor, batch_size, input_size);

    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight_sum, float scaling_factor);"
)

# Compile the inline CUDA code for the fused operation
fused = load_inline(
    name="fused",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs the fused operation using a custom CUDA kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_sum = torch.sum(self.weight, dim=0)
        self.scaling_factor = scaling_factor
        self.fused = fused

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.fused.fused_forward_cuda(x, self.weight_sum, self.scaling_factor)