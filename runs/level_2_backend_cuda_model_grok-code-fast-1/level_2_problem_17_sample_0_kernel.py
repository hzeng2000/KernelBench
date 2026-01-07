import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused instance normalization and division
fused_instance_norm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void compute_stats_kernel(const float* x, float* mean, float* var, int batch, int channels, int height, int width, float eps) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    int n = height * width;
    extern __shared__ float sdata[];
    float* ssum = sdata;
    float* ssum_sq = sdata + blockDim.x;
    ssum[tid] = 0.0f;
    ssum_sq[tid] = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        int h = i / width;
        int w = i % width;
        float val = x[((b * channels + c) * height + h) * width + w];
        ssum[tid] += val;
        ssum_sq[tid] += val * val;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            ssum_sq[tid] += ssum_sq[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum_x = ssum[0];
        float sum_sq = ssum_sq[0];
        float m = sum_x / n;
        float v = sum_sq / n - m * m;
        mean[b * channels + c] = m;
        var[b * channels + c] = v;
    }
}

__global__ void apply_norm_kernel(const float* x, float* out, const float* mean, const float* var, int batch, int channels, int height, int width, float divide_by, float eps) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z;
    int w = threadIdx.x;
    int idx = ((b * channels + c) * height + h) * width + w;
    float m = mean[b * channels + c];
    float v = var[b * channels + c];
    float std = sqrtf(v + eps);
    float val = x[idx];
    out[idx] = (val - m) / (std * divide_by);
}

torch::Tensor fused_instance_norm_div_cuda(torch::Tensor x, float divide_by, float eps = 1e-5) {
    int batch = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    auto mean = torch::zeros({batch, channels}, x.options());
    auto var = torch::zeros({batch, channels}, x.options());
    auto out = torch::zeros_like(x);
    dim3 grid_stats(batch, channels);
    int block_size = 256;
    int shared_size = 2 * block_size * sizeof(float);
    compute_stats_kernel<<<grid_stats, block_size, shared_size>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), batch, channels, height, width, eps);
    dim3 grid_apply(batch, channels, height);
    dim3 block_apply(width);
    apply_norm_kernel<<<grid_apply, block_apply>>>(x.data_ptr<float>(), out.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), batch, channels, height, width, divide_by, eps);
    return out;
}
"""

fused_instance_norm_div_cpp_source = (
    "torch::Tensor fused_instance_norm_div_cuda(torch::Tensor x, float divide_by, float eps = 1e-5);"
)

# Compile the inline CUDA code for fused instance normalization and division
fused_instance_norm_div = load_inline(
    name="fused_instance_norm_div",
    cpp_sources=fused_instance_norm_div_cpp_source,
    cuda_sources=fused_instance_norm_div_source,
    functions=["fused_instance_norm_div_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution and applies fused Instance Normalization with division by a constant using a custom CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.fused_norm_div = fused_instance_norm_div

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_norm_div.fused_instance_norm_div_cuda(x, self.divide_by)
        return x