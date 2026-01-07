import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + sum + max + mean + logsumexp + logsumexp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features) {

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (batch_idx >= batch_size) return;

    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = &shared_mem[blockDim.x];

    float local_sum = 0.0f;

    // Linear layer: compute row of output
    for (int j = tid; j < out_features; j += stride) {
        float val = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            val += input[batch_idx * in_features + i] * weight[j * in_features + i];
        }
        if (bias != nullptr) {
            val += bias[j];
        }
        local_sum += val;
    }

    // Block-level reduction for sum
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    float sum_val = shared_sum[0];

    // Max (redundant here since sum is scalar, but kept for compatibility)
    float max_val = sum_val;

    // Mean (also redundant)
    float mean_val = max_val;

    // First LogSumExp: log(exp(mean_val)) = mean_val
    float lse1 = mean_val;

    // Second LogSumExp: log(exp(lse1)) = lse1
    float lse2 = lse1;

    if (tid == 0) {
        output[batch_idx] = lse2;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::zeros({batch_size, 1}, input.options());

    const int threads = 256;
    const int blocks = batch_size;
    const int shared_mem_size = 2 * threads * sizeof(float);

    fused_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
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
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(x, self.weight, self.bias)