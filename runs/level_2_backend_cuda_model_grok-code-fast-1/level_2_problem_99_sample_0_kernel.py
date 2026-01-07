import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for linear + GELU
linear_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void linear_gelu_kernel(const float* x, const float* w, const float* b, float* out, int batch, int in_f, int out_f) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y;
    if (batch_idx >= batch || out_idx >= out_f) return;

    __shared__ float s_sum[256];
    int tid = threadIdx.x;
    s_sum[tid] = 0.0f;
    for (int i = tid; i < in_f; i += blockDim.x) {
        s_sum[tid] += x[batch_idx * in_f + i] * w[out_idx * in_f + i];
    }
    __syncthreads();
    // reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = s_sum[0] + b[out_idx];
        // GELU
        out[batch_idx * out_f + out_idx] = 0.5f * sum * (1.0f + erff(sum / sqrtf(2.0f)));
    }
}

torch::Tensor linear_gelu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int batch = x.size(0);
    int in_f = x.size(1);
    int out_f = w.size(0);
    auto out = torch::zeros({batch, out_f}, x.options());
    dim3 blocks(batch, out_f);
    dim3 threads(256);
    linear_gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), batch, in_f, out_f);
    return out;
}
"""

# Define the custom CUDA kernel for softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(const float* in, float* out, int batch, int out_f) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;
    // find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < out_f; i += blockDim.x) {
        max_val = fmaxf(max_val, in[batch_idx * out_f + i]);
    }
    __shared__ float s_max[256];
    s_max[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = s_max[0];
    // compute exp and sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < out_f; i += blockDim.x) {
        float val = expf(in[batch_idx * out_f + i] - max_val);
        out[batch_idx * out_f + i] = val;
        sum_exp += val;
    }
    __shared__ float s_sum[256];
    s_sum[threadIdx.x] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_exp = s_sum[0];
    // normalize
    for (int i = threadIdx.x; i < out_f; i += blockDim.x) {
        out[batch_idx * out_f + i] /= sum_exp;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    int batch = x.size(0);
    int out_f = x.size(1);
    auto out = torch::zeros_like(x);
    dim3 blocks(batch);
    dim3 threads(256);
    softmax_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch, out_f);
    return out;
}
"""

cpp_source = (
    "torch::Tensor linear_gelu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);"
    "torch::Tensor softmax_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=linear_gelu_source + softmax_source,
    functions=["linear_gelu_cuda", "softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA operators for linear + GELU and softmax.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear_gelu = custom_ops
        self.softmax = custom_ops

    def forward(self, x):
        x = self.linear_gelu.linear_gelu_cuda(x, self.linear.weight, self.linear.bias)
        x = self.softmax.softmax_cuda(x)
        return x