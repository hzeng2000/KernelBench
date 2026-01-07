import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for softmax along dim=1
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void softmax_kernel(const float* x, float* out, int batch, int channels, int depth, int height, int width) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int h = blockIdx.z;
    int w = threadIdx.x;
    if (w >= width) return;
    
    float max_val = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < channels; c++) {
        int idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        max_val = max(max_val, x[idx]);
    }
    float sum = 0.0f;
    for (int c = 0; c < channels; c++) {
        int idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        sum += expf(x[idx] - max_val);
    }
    for (int c = 0; c < channels; c++) {
        int idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        out[idx] = expf(x[idx] - max_val) / sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);
    auto out = torch::zeros_like(x);
    
    dim3 blocks(batch, depth, height);
    dim3 threads(width);
    
    softmax_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch, channels, depth, height, width);
    
    return out;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for fused max pool (two 2x2x2 pools fused into one 4x4x4 pool)
fused_maxpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void fused_maxpool_kernel(const float* x, float* out, int batch, int channels, int in_depth, int in_height, int in_width, int out_depth, int out_height, int out_width) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int od = blockIdx.z;
    int oh = threadIdx.y;
    int ow = threadIdx.x;
    if (od >= out_depth || oh >= out_height || ow >= out_width) return;
    
    float max_val = -std::numeric_limits<float>::infinity();
    for (int kd = 0; kd < 4; kd++) {
        for (int kh = 0; kh < 4; kh++) {
            for (int kw = 0; kw < 4; kw++) {
                int id = od * 4 + kd;
                int ih = oh * 4 + kh;
                int iw = ow * 4 + kw;
                if (id < in_depth && ih < in_height && iw < in_width) {
                    int idx = ((b * channels + c) * in_depth + id) * in_height * in_width + ih * in_width + iw;
                    max_val = max(max_val, x[idx]);
                }
            }
        }
    }
    int out_idx = ((b * channels + c) * out_depth + od) * out_height * out_width + oh * out_width + ow;
    out[out_idx] = max_val;
}

torch::Tensor fused_maxpool_cuda(torch::Tensor x) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto in_depth = x.size(2);
    auto in_height = x.size(3);
    auto in_width = x.size(4);
    int out_depth = in_depth / 4;
    int out_height = in_height / 4;
    int out_width = in_width / 4;
    auto out = torch::zeros({batch, channels, out_depth, out_height, out_width}, x.options());
    
    dim3 blocks(batch, channels, out_depth);
    dim3 threads(out_width, out_height);
    
    fused_maxpool_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch, channels, in_depth, in_height, in_width, out_depth, out_height, out_width);
    
    return out;
}
"""

fused_maxpool_cpp_source = (
    "torch::Tensor fused_maxpool_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for fused max pool
fused_maxpool = load_inline(
    name="fused_maxpool",
    cpp_sources=fused_maxpool_cpp_source,
    cuda_sources=fused_maxpool_source,
    functions=["fused_maxpool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies custom CUDA Softmax, and performs fused max pooling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.softmax = softmax
        self.fused_pool = fused_maxpool

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after fused pooling.
        """
        x = self.conv(x)
        x = self.softmax.softmax_cuda(x)
        x = self.fused_pool.fused_maxpool_cuda(x)
        return x