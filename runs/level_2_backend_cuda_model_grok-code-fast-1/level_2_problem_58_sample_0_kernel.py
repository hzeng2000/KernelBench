import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-conv operations: logsumexp, hardswish, subtract bias, clamp
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_post_conv_kernel(const float* x, const float* bias, float* out, int batch, int channels, int depth, int height, int width) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int h = blockIdx.z;
    int w = threadIdx.x;
    
    if (b >= batch || d >= depth || h >= height || w >= width) return;
    
    // Compute logsumexp over channels
    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
        int idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        max_val = fmaxf(max_val, x[idx]);
    }
    
    float sum_exp = 0.0f;
    for (int c = 0; c < channels; ++c) {
        int idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        sum_exp += __expf(x[idx] - max_val);
    }
    
    float lse = max_val + __logf(sum_exp);
    
    // HardSwish: x * sigmoid(x + 3) / 6
    float sig = 1.0f / (1.0f + __expf(-(lse + 3.0f)));
    float hs = lse * sig / 6.0f;
    
    // Subtract bias
    hs -= bias[0];
    
    // Clamp
    hs = fmaxf(-1.0f, fminf(1.0f, hs));
    
    // Output
    int out_idx = ((b * 1 + 0) * depth + d) * height * width + h * width + w;
    out[out_idx] = hs;
}

torch::Tensor fused_post_conv_cuda(torch::Tensor x, torch::Tensor bias) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);
    
    auto out = torch::zeros({batch, 1, depth, height, width}, x.options());
    
    dim3 grid(batch, depth, height);
    dim3 block(width);
    
    fused_post_conv_kernel<<<grid, block>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch, channels, depth, height, width);
    
    return out;
}
"""

fused_post_conv_cpp_source = (
    "torch::Tensor fused_post_conv_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused post-conv operations
fused_post_conv = load_inline(
    name="fused_post_conv",
    cpp_sources=fused_post_conv_cpp_source,
    cuda_sources=fused_post_conv_source,
    functions=["fused_post_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by fused custom CUDA operations for LogSumExp, HardSwish, subtraction, and clamp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.fused_post_conv = fused_post_conv

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_post_conv.fused_post_conv_cuda(x, self.bias)
        return x