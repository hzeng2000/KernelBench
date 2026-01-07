import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min, clamp, and dropout
fused_clamp_min_dropout_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void clamp_min_dropout_kernel(const float* x, float* out, float min_val, float max_val, float dropout_p, bool is_training, int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        val = fminf(val, min_val);
        val = fmaxf(fminf(val, max_val), min_val);
        if (is_training) {
            curandState state;
            curand_init(seed, idx, 0, &state);
            float rand = curand_uniform(&state);
            if (rand < dropout_p) {
                out[idx] = 0.0f;
            } else {
                out[idx] = val / (1.0f - dropout_p);
            }
        } else {
            out[idx] = val;
        }
    }
}

torch::Tensor clamp_min_dropout_cuda(torch::Tensor x, float min_val, float max_val, float dropout_p, bool is_training) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    unsigned long long seed = 12345;  // Fixed seed for reproducibility; can be randomized if needed

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    clamp_min_dropout_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), min_val, max_val, dropout_p, is_training, size, seed);

    return out;
}
"""

fused_clamp_min_dropout_cpp_source = (
    "torch::Tensor clamp_min_dropout_cuda(torch::Tensor x, float min_val, float max_val, float dropout_p, bool is_training);"
)

# Compile the inline CUDA code for fused clamp, min, and dropout
fused_clamp_min_dropout = load_inline(
    name="fused_clamp_min_dropout",
    cpp_sources=fused_clamp_min_dropout_cpp_source,
    cuda_sources=fused_clamp_min_dropout_source,
    functions=["clamp_min_dropout_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcurand"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies Group Normalization, and fused min, clamp, dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.fused_clamp_min_dropout = fused_clamp_min_dropout
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.fused_clamp_min_dropout.clamp_min_dropout_cuda(x, self.min_value, self.max_value, self.dropout_p, self.training)
        return x