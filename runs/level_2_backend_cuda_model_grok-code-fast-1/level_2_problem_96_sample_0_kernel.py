import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for global average pooling and clamping
global_avg_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_clamp_kernel(const float* x, float* out, int batch, int channels, int d, int h, int w, float clamp_min, float clamp_max) {
    int bc = blockIdx.x;
    int b = bc / channels;
    int c = bc % channels;
    int numel = d * h * w;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < numel; i += blockDim.x) {
        int idx_d = i / (h * w);
        int idx_h = (i / w) % h;
        int idx_w = i % w;
        sum += x[((b * channels + c) * d + idx_d) * h * w + idx_h * w + idx_w];
    }
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float avg = sdata[0] / numel;
        out[bc] = fminf(fmaxf(avg, clamp_min), clamp_max);
    }
}

torch::Tensor global_avg_clamp_cuda(torch::Tensor x, float clamp_min, float clamp_max) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto d = x.size(2);
    auto h = x.size(3);
    auto w = x.size(4);
    auto out = torch::zeros({batch, channels, 1, 1, 1}, x.options());
    int num_bc = batch * channels;
    const int block_size = 256;
    const int num_blocks = num_bc;
    size_t shared_mem = block_size * sizeof(float);
    global_avg_clamp_kernel<<<num_blocks, block_size, shared_mem>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch, channels, d, h, w, clamp_min, clamp_max);
    return out;
}
"""

global_avg_clamp_cpp_source = (
    "torch::Tensor global_avg_clamp_cuda(torch::Tensor x, float clamp_min, float clamp_max);"
)

# Compile the inline CUDA code for global average pooling and clamping
global_avg_clamp = load_inline(
    name="global_avg_clamp",
    cpp_sources=global_avg_clamp_cpp_source,
    cuda_sources=global_avg_clamp_source,
    functions=["global_avg_clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed 3D convolution (with scale absorbed into weights), max pooling, 
    fused global average pooling and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Absorb scale into conv weights and bias
        self.conv_transpose.weight.data *= scale
        self.conv_transpose.bias.data *= scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_clamp = global_avg_clamp
        self.clamp_min = 0.0
        self.clamp_max = 1.0

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.global_avg_clamp.global_avg_clamp_cuda(x, self.clamp_min, self.clamp_max)
        return x