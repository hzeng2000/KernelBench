import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: tanh, hardswish, residual add, and logsumexp
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float hardswish(float x) {
    return x * min(max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_kernel(const float* x_conv, const float* x_norm, float* out, int batch, int out_c, int h, int w) {
    int n = blockIdx.x;
    int ij = blockIdx.y * blockDim.x + threadIdx.x;
    int i = ij / w;
    int j = ij % w;
    if (n >= batch || i >= h || j >= w) return;

    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    for(int c = 0; c < out_c; c++){
        int idx = ((n * out_c + c) * h + i) * w + j;
        float val = x_conv[idx] + hardswish(tanhf(x_norm[idx]));
        if (val > max_val){
            sum_exp = sum_exp * expf(max_val - val) + 1.0f;
            max_val = val;
        } else {
            sum_exp += expf(val - max_val);
        }
    }
    int out_idx = (n * h + i) * w + j;
    out[out_idx] = max_val + logf(sum_exp);
}

torch::Tensor fused_forward_cuda(torch::Tensor x_conv, torch::Tensor x_norm) {
    auto batch = x_conv.size(0);
    auto out_c = x_conv.size(1);
    auto h = x_conv.size(2);
    auto w = x_conv.size(3);
    auto out = torch::zeros({batch, 1, h, w}, x_conv.options());

    dim3 blocks(batch, (h * w + 255) / 256);
    dim3 threads(256);

    fused_kernel<<<blocks, threads>>>(x_conv.data_ptr<float>(), x_norm.data_ptr<float>(), out.data_ptr<float>(), batch, out_c, h, w);

    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_forward_cuda(torch::Tensor x_conv, torch::Tensor x_norm);"
)

# Compile the inline CUDA code for fused operations
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
    Optimized Model that performs a convolution, applies Group Normalization, and then fused Tanh, HardSwish, 
    Residual Addition, and LogSumExp in a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.fused = fused

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        # Fused Tanh, HardSwish, Residual Addition, and LogSumExp
        return self.fused.fused_forward_cuda(x_conv, x_norm)