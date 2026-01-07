import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused global average pooling, bias addition, log-sum-exp, and multiplication
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sum_spatial_kernel(const float* x, float* sums, int batch, int channels, int H, int W) {
    int batch_idx = blockIdx.x / channels;
    int channel_idx = blockIdx.x % channels;
    int spatial_size = H * W;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum_val = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int h = i / W;
        int w = i % W;
        sum_val += x[((batch_idx * channels + channel_idx) * H + h) * W + w];
    }
    sdata[tid] = sum_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&sums[batch_idx * channels + channel_idx], sdata[0]);
    }
}

__global__ void compute_logsumexp_kernel(const float* sums, const float* bias, float* out, int batch, int channels, int HW) {
    int batch_idx = blockIdx.x;
    float max_val = -INFINITY;
    for (int c = 0; c < channels; c++) {
        float val = sums[batch_idx * channels + c] / HW + bias[c];
        if (val > max_val) max_val = val;
    }
    float sum_exp = 0.0f;
    for (int c = 0; c < channels; c++) {
        float val = sums[batch_idx * channels + c] / HW + bias[c];
        sum_exp += expf(val - max_val);
    }
    out[batch_idx] = (logf(sum_exp) + max_val) * 10.0f;
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias) {
    int batch = x.size(0);
    int channels = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int HW = H * W;
    auto sums = torch::zeros({batch, channels}, x.options());
    int num_blocks = batch * channels;
    int block_size = 256;
    size_t shared_mem = block_size * sizeof(float);
    reduce_sum_spatial_kernel<<<num_blocks, block_size, shared_mem>>>(x.data_ptr<float>(), sums.data_ptr<float>(), batch, channels, H, W);
    auto out = torch::zeros({batch, 1}, x.options());
    compute_logsumexp_kernel<<<batch, 1>>>(sums.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch, channels, HW);
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias);"
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
    Optimized Model that performs a transposed convolution, then fused global average pooling, bias addition, log-sum-exp, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias)
        return x