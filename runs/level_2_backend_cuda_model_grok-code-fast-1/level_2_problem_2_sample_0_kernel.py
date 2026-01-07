import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-processing
fused_post_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_kernel(const float* out, const float* bias, float* result, float scaling_factor, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;
    
    int c = (idx / (H * W)) % C;
    float val = out[idx] + bias[c];
    val = fmaxf(0.0f, fminf(val, 1.0f));
    val *= scaling_factor;
    val = fmaxf(0.0f, fminf(val, 1.0f));
    val /= scaling_factor;
    result[idx] = val;
}

torch::Tensor fused_post_cuda(torch::Tensor out, torch::Tensor bias, float scaling_factor) {
    auto B = out.size(0);
    auto C = out.size(1);
    auto H = out.size(2);
    auto W = out.size(3);
    auto result = torch::empty_like(out);
    
    int total = B * C * H * W;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_post_kernel<<<num_blocks, block_size>>>(out.data_ptr<float>(), bias.data_ptr<float>(), result.data_ptr<float>(), scaling_factor, B, C, H, W);
    
    return result;
}
"""

fused_post_cpp_source = (
    "torch::Tensor fused_post_cuda(torch::Tensor out, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code for fused post-processing
fused_post = load_inline(
    name="fused_post",
    cpp_sources=fused_post_cpp_source,
    cuda_sources=fused_post_source,
    functions=["fused_post_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution and fused post-processing with a custom CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        self.fused_post = fused_post

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_post.fused_post_cuda(x, self.bias, self.scaling_factor)
        return x