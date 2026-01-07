import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias add, hardtanh, and mish
fused_bias_hardtanh_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bias_hardtanh_mish_kernel(const float* x, const float* bias, float* out, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx < total) {
        int f = idx % out_features;
        float val = x[idx] + bias[f];
        // hardtanh: clamp to [-1, 1]
        val = fmaxf(-1.0f, fminf(1.0f, val));
        // mish: x * tanh(softplus(x))
        float sp = log1pf(expf(val));  // softplus
        val = val * tanhf(sp);
        out[idx] = val;
    }
}

torch::Tensor fused_bias_hardtanh_mish_cuda(torch::Tensor x, torch::Tensor bias) {
    auto batch_size = x.size(0);
    auto out_features = x.size(1);
    auto out = torch::empty_like(x);
    int total = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    fused_bias_hardtanh_mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, out_features);
    return out;
}
"""

fused_bias_hardtanh_mish_cpp_source = (
    "torch::Tensor fused_bias_hardtanh_mish_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused bias add, hardtanh, and mish
fused_ops = load_inline(
    name="fused_bias_hardtanh_mish",
    cpp_sources=fused_bias_hardtanh_mish_cpp_source,
    cuda_sources=fused_bias_hardtanh_mish_source,
    functions=["fused_bias_hardtanh_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that performs a GEMM, fused BiasAdd+Hardtanh+Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.fused_ops.fused_bias_hardtanh_mish_cuda(x, self.bias)
        x = self.groupnorm(x)
        return x