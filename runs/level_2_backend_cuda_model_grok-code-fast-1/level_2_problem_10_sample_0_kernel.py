import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused maxpool and hardtanh
fused_maxpool_hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_maxpool_hardtanh_kernel(const float* input, float* output, int batch, int channels, int height, int width, int kernel_h, int kernel_w, int stride_h, int stride_w, float min_val, float max_val, int out_height, int out_width) {
    int b = blockIdx.z / channels;
    int c = blockIdx.z % channels;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (oh >= out_height || ow >= out_width) return;
    int ih_start = oh * stride_h;
    int iw_start = ow * stride_w;
    float maxv = -INFINITY;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = ih_start + kh;
            int iw = iw_start + kw;
            if (ih < height && iw < width) {
                int idx = ((b * channels + c) * height + ih) * width + iw;
                maxv = fmaxf(maxv, input[idx]);
            }
        }
    }
    maxv = fminf(fmaxf(maxv, min_val), max_val);
    int out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
    output[out_idx] = maxv;
}

torch::Tensor fused_maxpool_hardtanh_cuda(torch::Tensor input, int kernel_h, int kernel_w, int stride_h, int stride_w, float min_val, float max_val) {
    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_height = (height - kernel_h) / stride_h + 1;
    int out_width = (width - kernel_w) / stride_w + 1;
    auto output = torch::zeros({batch, channels, out_height, out_width}, input.options());
    dim3 block(16, 16);
    dim3 grid((out_width + 15) / 16, (out_height + 15) / 16, batch * channels);
    fused_maxpool_hardtanh_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch, channels, height, width, kernel_h, kernel_w, stride_h, stride_w, min_val, max_val, out_height, out_width);
    return output;
}
"""

fused_maxpool_hardtanh_cpp_source = (
    "torch::Tensor fused_maxpool_hardtanh_cuda(torch::Tensor input, int kernel_h, int kernel_w, int stride_h, int stride_w, float min_val, float max_val);"
)

# Compile the inline CUDA code for fused maxpool and hardtanh
fused_maxpool_hardtanh = load_inline(
    name="fused_maxpool_hardtanh",
    cpp_sources=fused_maxpool_hardtanh_cpp_source,
    cuda_sources=fused_maxpool_hardtanh_source,
    functions=["fused_maxpool_hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for mean and tanh
mean_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void mean_tanh_kernel(const float* input, float* output, int batch, int channels, int height, int width) {
    extern __shared__ float sdata[];
    int b = blockIdx.x / channels;
    int c = blockIdx.x % channels;
    int tid = threadIdx.x;
    int num_elements = height * width;
    sdata[tid] = 0.0f;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int h = i / width;
        int w = i % width;
        int idx = ((b * channels + c) * height + h) * width + w;
        sdata[tid] += input[idx];
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float mean = sdata[0] / num_elements;
        output[b * channels + c] = tanhf(mean);
    }
}

torch::Tensor mean_tanh_cuda(torch::Tensor input) {
    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    auto output = torch::zeros({batch, channels, 1, 1}, input.options());
    int num_blocks = batch * channels;
    int block_size = 256;
    mean_tanh_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch, channels, height, width);
    return output;
}
"""

mean_tanh_cpp_source = (
    "torch::Tensor mean_tanh_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for mean and tanh
mean_tanh = load_inline(
    name="mean_tanh",
    cpp_sources=mean_tanh_cpp_source,
    cuda_sources=mean_tanh_source,
    functions=["mean_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, followed by fused maxpool+hardtanh, and fused mean+tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_maxpool_hardtanh = fused_maxpool_hardtanh
        self.mean_tanh = mean_tanh
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_maxpool_hardtanh.fused_maxpool_hardtanh_cuda(x, self.maxpool_kernel_size, self.maxpool_kernel_size, self.maxpool_stride, self.maxpool_stride, self.hardtanh_min, self.hardtanh_max)
        x = self.mean_tanh.mean_tanh_cuda(x)
        return x