import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + logsumexp + double leaky_relu + double gelu
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* max_vals = shared_mem;
    float* sum_vals = &shared_mem[blockDim.x];

    if (row >= batch_size) return;

    // Compute linear output for this row
    float local_max = -1e38f;
    for (int col = tid; col < out_features; col += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < in_features; k++) {
            acc += input[row * in_features + k] * weight[col * in_features + k];
        }
        if (bias != nullptr) {
            acc += bias[col];
        }
        // First LeakyReLU
        acc = fmaxf(acc, 0.0f) + 0.01f * fminf(acc, 0.0f);
        // Second LeakyReLU
        acc = fmaxf(acc, 0.0f) + 0.01f * fminf(acc, 0.0f);
        // First GELU
        float gelu1 = 0.5f * acc * (1.0f + tanhf(0.7978845608f * (acc + 0.044715f * acc * acc * acc)));
        // Second GELU
        float gelu2 = 0.5f * gelu1 * (1.0f + tanhf(0.7978845608f * (gelu1 + 0.044715f * gelu1 * gelu1 * gelu1)));
        
        // Update max for logsumexp
        if (gelu2 > local_max) local_max = gelu2;
        // Store intermediate in shared memory for reduction
        shared_mem[col] = gelu2;
    }

    // Block-wide reduction to find max
    max_vals[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (max_vals[tid + stride] > max_vals[tid]) {
                max_vals[tid] = max_vals[tid + stride];
            }
        }
        __syncthreads();
    }
    float global_max = max_vals[0];
    __syncthreads();

    // Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int col = tid; col < out_features; col += blockDim.x) {
        float val = shared_mem[col];
        local_sum += expf(val - global_max);
    }
    sum_vals[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_vals[tid] += sum_vals[tid + stride];
        }
        __syncthreads();
    }
    float global_sum = sum_vals[0];

    // Compute logsumexp
    if (tid == 0) {
        output[row] = logf(global_sum) + global_max;
    }
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_ops = fused_ops

    def forward(self, x):
        weight = self.linear.weight
        bias = self.linear.bias
        batch_size = x.size(0)
        output = torch.empty(batch_size, 1, device=x.device, dtype=x.dtype)
        fused_ops.fused_forward_cuda(x, weight, bias, output, batch_size, x.size(1), weight.size(0))
        return output