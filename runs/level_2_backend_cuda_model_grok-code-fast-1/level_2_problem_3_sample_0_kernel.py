import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for computing mean and std for layer norm with add
compute_stats_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_stats_kernel(const float* x, float sum_weight, float* sum_out, float* sumsq_out, int batch, int channels, int depth, int height, int width) {
    int b = blockIdx.x / channels;
    int c = blockIdx.x % channels;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int spatial_size = depth * height * width;
    int elements_per_thread = (spatial_size + block_size - 1) / block_size;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = tid * elements_per_thread + i;
        if (idx < spatial_size) {
            int d = idx / (height * width);
            int h = (idx % (height * width)) / width;
            int w = idx % width;
            float val = x[((b * channels + c) * depth + d) * height * width + h * width + w] + sum_weight;
            local_sum += val;
            local_sumsq += val * val;
        }
    }

    // Reduce within block
    __shared__ float shared_sum[1024];
    __shared__ float shared_sumsq[1024];
    shared_sum[tid] = local_sum;
    shared_sumsq[tid] = local_sumsq;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sumsq[tid] += shared_sumsq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&sum_out[blockIdx.x], shared_sum[0]);
        atomicAdd(&sumsq_out[blockIdx.x], shared_sumsq[0]);
    }
}

__global__ void normalize_kernel(const float* x, float sum_weight, const float* mean, const float* std, const float* weight, const float* bias, float* out, int batch, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch * channels * depth * height * width;
    if (idx < total_size) {
        int b = idx / (channels * depth * height * width);
        int c = (idx % (channels * depth * height * width)) / (depth * height * width);
        float val = x[idx] + sum_weight;
        val = (val - mean[b * channels + c]) / (std[b * channels + c] + 1e-5f);
        val = val * weight[c] + bias[c];
        out[idx] = val;
    }
}
"""

compute_stats_cpp_source = """
void compute_stats_kernel(const float* x, float sum_weight, float* sum_out, float* sumsq_out, int batch, int channels, int depth, int height, int width);
void normalize_kernel(const float* x, float sum_weight, const float* mean, const float* std, const float* weight, const float* bias, float* out, int batch, int channels, int depth, int height, int width);
"""

# Compile the inline CUDA code for add_norm
add_norm_ext = load_inline(
    name="add_norm",
    cpp_sources=compute_stats_cpp_source,
    cuda_sources=compute_stats_source,
    functions=["compute_stats_kernel", "normalize_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for avgpool + gelu
avgpool_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void avgpool_gelu_kernel(const float* x, float* out, int batch, int channels, int in_depth, int in_height, int in_width, int out_depth, int out_height, int out_width) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int od = blockIdx.z / (out_height * out_width);
    int oh = (blockIdx.z % (out_height * out_width)) / out_width;
    int ow = blockIdx.z % out_width;

    float sum = 0.0f;
    for (int kd = 0; kd < 2; ++kd) {
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                int id = od * 2 + kd;
                int ih = oh * 2 + kh;
                int iw = ow * 2 + kw;
                sum += x[((b * channels + c) * in_depth + id) * in_height * in_width + ih * in_width + iw];
            }
        }
    }
    float avg = sum / 8.0f;
    // GELU: 0.5 * x * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
    float x_val = avg;
    float coeff = sqrtf(2.0f / 3.141592653589793f);
    float inner = coeff * (x_val + 0.044715f * x_val * x_val * x_val);
    float gelu_val = 0.5f * x_val * (1.0f + tanhf(inner));
    out[((b * channels + c) * out_depth + od) * out_height * out_width + oh * out_width + ow] = gelu_val;
}
"""

avgpool_gelu_cpp_source = (
    "void avgpool_gelu_kernel(const float* x, float* out, int batch, int channels, int in_depth, int in_height, int in_width, int out_depth, int out_height, int out_width);"
)

# Compile the inline CUDA code for avgpool_gelu
avgpool_gelu_ext = load_inline(
    name="avgpool_gelu",
    cpp_sources=avgpool_gelu_cpp_source,
    cuda_sources=avgpool_gelu_source,
    functions=["avgpool_gelu_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by a fused add + layer normalization, fused average pooling + GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.batch_size = 32  # Assuming fixed for kernel launches
        self.out_channels = out_channels
        self.depth, self.height, self.width = 32, 32, 32  # Output of conv
        self.out_depth, self.out_height, self.out_width = 16, 32, 32  # Output of pool

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused add + norm
        batch, channels, depth, height, width = x.shape
        sum_out = torch.zeros(batch * channels, dtype=torch.float32, device=x.device)
        sumsq_out = torch.zeros(batch * channels, dtype=torch.float32, device=x.device)
        block_size = 1024
        num_blocks = batch * channels
        add_norm_ext.compute_stats_kernel(
            x.data_ptr<float>(), self.sum_weight.item(), sum_out.data_ptr<float>(), sumsq_out.data_ptr<float>(),
            batch, channels, depth, height, width, block_size=block_size, grid=(num_blocks,)
        )
        N = depth * height * width
        mean = sum_out / N
        var = sumsq_out / N - mean * mean
        std = torch.sqrt(var + 1e-5)
        out_norm = torch.zeros_like(x)
        total_size = x.numel()
        num_blocks_norm = (total_size + block_size - 1) // block_size
        add_norm_ext.normalize_kernel(
            x.data_ptr<float>(), self.sum_weight.item(), mean.data_ptr<float>(), std.data_ptr<float>(),
            self.norm.weight.data_ptr<float>(), self.norm.bias.data_ptr<float>(), out_norm.data_ptr<float>(),
            batch, channels, depth, height, width, block_size=block_size, grid=(num_blocks_norm,)
        )
        x = out_norm
        # Fused avgpool + gelu
        out_pool = torch.zeros(batch, channels, self.out_depth, self.out_height, self.out_width, dtype=torch.float32, device=x.device)
        avgpool_gelu_ext.avgpool_gelu_kernel(
            x.data_ptr<float>(), out_pool.data_ptr<float>(), batch, channels, depth, height, width, self.out_depth, self.out_height, self.out_width,
            block_size=1, grid=(batch, channels, self.out_depth * self.out_height * self.out_width)
        )
        return out_pool