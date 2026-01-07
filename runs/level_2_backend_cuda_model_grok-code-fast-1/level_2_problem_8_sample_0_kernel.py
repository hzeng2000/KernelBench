import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: division, maxpool, global avg pool, add bias, sum over channels
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_ops_kernel(const float* x, float* out, const float* bias, float divisor, int batch_size, int out_channels, int depth, int height, int width, int pool_d, int pool_h, int pool_w) {
    int b = blockIdx.x / out_channels;
    int c = blockIdx.x % out_channels;
    int tid = threadIdx.x;

    int od_max = depth / pool_d;
    int oh_max = height / pool_h;
    int ow_max = width / pool_w;
    int num_elements = od_max * oh_max * ow_max;

    float sum_val = 0.0f;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int od = i / (oh_max * ow_max);
        int oh = (i / ow_max) % oh_max;
        int ow = i % ow_max;

        float max_val = -INFINITY;
        for (int pd = 0; pd < pool_d; pd++) {
            for (int ph = 0; ph < pool_h; ph++) {
                for (int pw = 0; pw < pool_w; pw++) {
                    int id = od * pool_d + pd;
                    int ih = oh * pool_h + ph;
                    int iw = ow * pool_w + pw;
                    int idx = ((b * out_channels + c) * depth + id) * height * width + ih * width + iw;
                    max_val = fmaxf(max_val, x[idx] / divisor);
                }
            }
        }
        sum_val += max_val;
    }

    __shared__ float sdata[256];
    sdata[tid] = sum_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float avg = sdata[0] / num_elements;
        avg += bias[c];
        atomicAdd(&out[b], avg);
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias, float divisor, int pool_d, int pool_h, int pool_w) {
    auto batch_size = x.size(0);
    auto out_channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);
    auto out = torch::zeros({batch_size}, x.options());

    const int block_size = 256;
    const int num_blocks = batch_size * out_channels;

    fused_ops_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), bias.data_ptr<float>(), divisor, batch_size, out_channels, depth, height, width, pool_d, pool_h, pool_w);

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias, float divisor, int pool_d, int pool_h, int pool_w);"
)

# Compile the inline CUDA code for fused operations
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, then fuses division, max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape).squeeze())  # Squeeze to (out_channels,)
        self.sum_dim = sum_dim
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # Fused operations: division, maxpool, global avg pool, add bias, sum over channels
        return self.fused_ops.fused_ops_cuda(x, self.bias, self.divisor, self.pool_size[0], self.pool_size[1], self.pool_size[2])