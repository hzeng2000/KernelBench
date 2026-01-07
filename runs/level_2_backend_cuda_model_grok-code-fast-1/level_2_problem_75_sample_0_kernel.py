import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define TILE_SIZE for the linear kernel
TILE_SIZE = 16

# Custom CUDA kernel for Linear (GEMM)
linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void linear_kernel(const float* weight, const float* x, const float* bias, float* out, int batch, int in_features, int out_features) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    float sum = 0.0f;
    for (int t = 0; t < (in_features + 16 - 1) / 16; ++t) {
        if (row < out_features && t * 16 + tx < in_features) {
            As[ty][tx] = weight[row * in_features + t * 16 + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        if (col < batch && t * 16 + ty < in_features) {
            Bs[ty][tx] = x[col * in_features + t * 16 + ty];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < 16; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < out_features && col < batch) {
        out[col * out_features + row] = sum + bias[row];
    }
}

torch::Tensor custom_linear_cuda(torch::Tensor weight, torch::Tensor x, torch::Tensor bias) {
    auto batch = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);
    auto out = torch::zeros({batch, out_features}, x.options());
    dim3 blockDim(16, 16);
    dim3 gridDim((batch + 16 - 1) / 16, (out_features + 16 - 1) / 16);
    linear_kernel<<<gridDim, blockDim>>>(weight.data_ptr<float>(), x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch, in_features, out_features);
    return out;
}
"""

linear_cpp_source = "torch::Tensor custom_linear_cuda(torch::Tensor weight, torch::Tensor x, torch::Tensor bias);"

custom_linear = load_inline(
    name="custom_linear",
    cpp_sources=linear_cpp_source,
    cuda_sources=linear_source,
    functions=["custom_linear_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for GroupNorm
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void group_norm_kernel(const float* x, float* out, const float* weight, const float* bias, int N, int C, int G, float eps) {
    int n = blockIdx.x;
    int g = blockIdx.y;
    int group_size = C / G;
    int start = g * group_size;
    int idx = threadIdx.x;
    if (n < N && g < G && idx < group_size) {
        __shared__ float shared_x[16];
        shared_x[idx] = x[n * C + start + idx];
        __syncthreads();
        if (idx == 0) {
            float mean = 0.0f;
            for (int i = 0; i < group_size; ++i) {
                mean += shared_x[i];
            }
            mean /= group_size;
            float var = 0.0f;
            for (int i = 0; i < group_size; ++i) {
                float diff = shared_x[i] - mean;
                var += diff * diff;
            }
            var /= group_size;
            shared_x[0] = mean;
            shared_x[1] = var;
        }
        __syncthreads();
        float mean = shared_x[0];
        float var = shared_x[1];
        float val = shared_x[idx];
        val = (val - mean) / sqrtf(var + eps);
        val = val * weight[start + idx] + bias[start + idx];
        out[n * C + start + idx] = val;
    }
}

torch::Tensor custom_group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto out = torch::zeros_like(x);
    int group_size = C / num_groups;
    dim3 blockDim(group_size);
    dim3 gridDim(N, num_groups);
    float eps = 1e-5f;
    group_norm_kernel<<<gridDim, blockDim>>>(x.data_ptr<float>(), out.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), N, C, num_groups, eps);
    return out;
}
"""

group_norm_cpp_source = "torch::Tensor custom_group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups);"

custom_group_norm = load_inline(
    name="custom_group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["custom_group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for fused min and add
min_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void min_add_kernel(const float* x, float* out, float bias, int batch, int out_features) {
    int b = blockIdx.x;
    if (b >= batch) return;
    __shared__ float shared_min[256];
    int tid = threadIdx.x;
    float min_val = FLT_MAX;
    for (int i = tid; i < out_features; i += blockDim.x) {
        min_val = fminf(min_val, x[b * out_features + i]);
    }
    shared_min[tid] = min_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[b] = shared_min[0] + bias;
    }
}

torch::Tensor custom_min_add_cuda(torch::Tensor x, torch::Tensor bias) {
    auto batch = x.size(0);
    auto out_features = x.size(1);
    auto out = torch::zeros({batch, 1}, x.options());
    int block_size = 256;
    int num_blocks = batch;
    min_add_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), bias.item<float>(), batch, out_features);
    return out.squeeze(1).unsqueeze(1);  // to match keepdim=True shape
}
"""

min_add_cpp_source = "torch::Tensor custom_min_add_cuda(torch::Tensor x, torch::Tensor bias);"

custom_min_add = load_inline(
    name="custom_min_add",
    cpp_sources=min_add_cpp_source,
    cuda_sources=min_add_source,
    functions=["custom_min_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.custom_linear = custom_linear
        self.custom_group_norm = custom_group_norm
        self.custom_min_add = custom_min_add

    def forward(self, x):
        x = self.custom_linear.custom_linear_cuda(self.gemm.weight, x, self.gemm.bias)
        x = self.custom_group_norm.custom_group_norm_cuda(x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups)
        x = self.custom_min_add.custom_min_add_cuda(x, self.bias)
        return x