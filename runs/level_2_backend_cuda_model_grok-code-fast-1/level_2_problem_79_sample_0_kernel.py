import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused clamp, multiply, and max reduction
fused_clamp_mul_max_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_clamp_mul_max_kernel(const float* x, const float* multiplier, float* out, float clamp_min, float clamp_max, int batch, int out_channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = depth * height * width;
    int total = batch * total_spatial;
    if (idx >= total) return;
    int b = idx / total_spatial;
    int spatial_idx = idx % total_spatial;
    int d = spatial_idx / (height * width);
    int h = (spatial_idx % (height * width)) / width;
    int w = spatial_idx % width;
    float max_val = -INFINITY;
    for (int c = 0; c < out_channels; c++) {
        int x_idx = ((b * out_channels + c) * depth + d) * height * width + h * width + w;
        float val = x[x_idx];
        val = fmaxf(fminf(val, clamp_max), clamp_min);
        val *= multiplier[c];
        if (val > max_val) max_val = val;
    }
    out[idx] = max_val;
}

torch::Tensor fused_clamp_mul_max_cuda(torch::Tensor x, torch::Tensor multiplier, float clamp_min, float clamp_max) {
    auto batch = x.size(0);
    auto out_channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);
    auto out = torch::zeros({batch, depth, height, width}, x.options());
    int total = batch * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    fused_clamp_mul_max_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), multiplier.data_ptr<float>(), out.data_ptr<float>(), clamp_min, clamp_max, batch, out_channels, depth, height, width);
    return out;
}
"""

fused_clamp_mul_max_cpp_source = (
    "torch::Tensor fused_clamp_mul_max_cuda(torch::Tensor x, torch::Tensor multiplier, float clamp_min, float clamp_max);"
)

# Compile the inline CUDA code for fused clamp, multiply, and max
fused_clamp_mul_max = load_inline(
    name="fused_clamp_mul_max",
    cpp_sources=fused_clamp_mul_max_cpp_source,
    cuda_sources=fused_clamp_mul_max_source,
    functions=["fused_clamp_mul_max_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_clamp_mul_max = fused_clamp_mul_max

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.instance_norm(x)
        x = self.fused_clamp_mul_max.fused_clamp_mul_max_cuda(x, self.multiplier, self.clamp_min, self.clamp_max)
        return x