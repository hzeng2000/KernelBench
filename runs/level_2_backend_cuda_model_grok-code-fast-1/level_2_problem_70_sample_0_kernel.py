import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused sigmoid, scaling, and residual add
fused_sigmoid_scale_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_sigmoid_scale_residual_kernel(float* x, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float sig = 1.0f / (1.0f + expf(-val));
        x[idx] = sig * scale + val;
    }
}

torch::Tensor fused_sigmoid_scale_residual_cuda(torch::Tensor x, float scale) {
    auto size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_sigmoid_scale_residual_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), scale, size);
    return x;
}
"""

fused_sigmoid_scale_residual_cpp_source = (
    "torch::Tensor fused_sigmoid_scale_residual_cuda(torch::Tensor x, float scale);"
)

# Compile the inline CUDA code for fused operation
fused_op = load_inline(
    name="fused_sigmoid_scale_residual",
    cpp_sources=fused_sigmoid_scale_residual_cpp_source,
    cuda_sources=fused_sigmoid_scale_residual_source,
    functions=["fused_sigmoid_scale_residual_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd" with fused CUDA operator.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = self.gemm(x)
        return self.fused_op.fused_sigmoid_scale_residual_cuda(x, self.scaling_factor)