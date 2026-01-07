import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + sigmoid
fused_linear_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sigmoid_kernel(const float* x, const float* weight, const float* bias, float* out, int batch_size, int input_size, int hidden_size) {
    int b = blockIdx.x;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch_size || h >= hidden_size) return;
    float sum = bias[h];
    for (int i = 0; i < input_size; i++) {
        sum += x[b * input_size + i] * weight[h * input_size + i];
    }
    out[b * hidden_size + h] = 1.0f / (1.0f + expf(-sum));
}

torch::Tensor fused_linear_sigmoid_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);
    auto out = torch::zeros({batch_size, hidden_size}, x.options());
    dim3 blockDim(256);
    dim3 gridDim(batch_size, (hidden_size + 255) / 256);
    fused_linear_sigmoid_kernel<<<gridDim, blockDim>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, input_size, hidden_size);
    return out;
}
"""

fused_linear_sigmoid_cpp_source = (
    "torch::Tensor fused_linear_sigmoid_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused linear + sigmoid
fused_linear_sigmoid = load_inline(
    name="fused_linear_sigmoid",
    cpp_sources=fused_linear_sigmoid_cpp_source,
    cuda_sources=fused_linear_sigmoid_source,
    functions=["fused_linear_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for fused linear + logsumexp
fused_linear_logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_logsumexp_kernel(const float* hidden, const float* weight, const float* bias, float* out, int batch_size, int hidden_size, int output_size) {
    int b = blockIdx.x;
    extern __shared__ float s_temp[];
    int o = threadIdx.x;
    if (o >= output_size) return;
    float sum = bias[o];
    for (int h = 0; h < hidden_size; h++) {
        sum += hidden[b * hidden_size + h] * weight[o * hidden_size + h];
    }
    s_temp[o] = sum;
    __syncthreads();
    // reduce for max
    __shared__ float s_max[1024];
    s_max[o] = s_temp[o];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (o < stride) {
            s_max[o] = fmaxf(s_max[o], s_max[o + stride]);
        }
        __syncthreads();
    }
    float max_val = s_max[0];
    __syncthreads();
    // compute exp
    s_temp[o] = expf(s_temp[o] - max_val);
    __syncthreads();
    // reduce for sum
    __shared__ float s_sum[1024];
    s_sum[o] = s_temp[o];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (o < stride) {
            s_sum[o] += s_sum[o + stride];
        }
        __syncthreads();
    }
    if (o == 0) {
        out[b] = logf(s_sum[0]) + max_val;
    }
}

torch::Tensor fused_linear_logsumexp_cuda(torch::Tensor hidden, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = hidden.size(0);
    int hidden_size = hidden.size(1);
    int output_size = weight.size(0);
    auto out = torch::zeros({batch_size}, hidden.options());
    dim3 blockDim(output_size);
    dim3 gridDim(batch_size);
    size_t shared_mem = output_size * sizeof(float) * 2;
    fused_linear_logsumexp_kernel<<<gridDim, blockDim, shared_mem>>>(hidden.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, hidden_size, output_size);
    return out;
}
"""

fused_linear_logsumexp_cpp_source = (
    "torch::Tensor fused_linear_logsumexp_cuda(torch::Tensor hidden, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused linear + logsumexp
fused_linear_logsumexp = load_inline(
    name="fused_linear_logsumexp",
    cpp_sources=fused_linear_logsumexp_cpp_source,
    cuda_sources=fused_linear_logsumexp_source,
    functions=["fused_linear_logsumexp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs fused matrix multiplication + Sigmoid,
    and fused matrix multiplication + LogSumExp over features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_linear_sigmoid = fused_linear_sigmoid
        self.fused_linear_logsumexp = fused_linear_logsumexp

    def forward(self, x):
        x = self.fused_linear_sigmoid.fused_linear_sigmoid_cuda(x, self.linear1.weight, self.linear1.bias)
        x = self.fused_linear_logsumexp.fused_linear_logsumexp_cuda(x, self.linear2.weight, self.linear2.bias)
        return x