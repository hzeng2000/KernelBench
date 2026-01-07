import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused linear + group norm + leaky_relu + add
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_groups,
    float eps,
    float negative_slope) {

    int tid = threadIdx.x;
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= hidden_size) return;

    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sum_sq = &shared_mem[num_groups];

    // Compute linear output
    float sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
        sum += input[row * input_size + i] * weight[col * input_size + i];
    }
    if (bias != nullptr) {
        sum += bias[col];
    }

    // Group normalization
    int group_size = hidden_size / num_groups;
    int group_id = col / group_size;
    int group_start = group_id * group_size;

    // Compute mean for group
    __shared__ float group_mean[512];
    __shared__ float group_var[512];

    atomicAdd(&shared_sum[group_id], sum);
    atomicAdd(&shared_sum_sq[group_id], sum * sum);

    __syncthreads();

    if (tid == 0) {
        group_mean[group_id] = shared_sum[group_id] / group_size;
        float mean = group_mean[group_id];
        group_var[group_id] = shared_sum_sq[group_id] / group_size - mean * mean;
    }

    __syncthreads();

    float mean = group_mean[group_id];
    float var = group_var[group_id];
    float std = sqrtf(var + eps);

    // Normalize, scale, shift
    float normalized = (sum - mean) / std;
    float scaled_shifted = normalized * gamma[col] + beta[col];

    // Leaky ReLU
    float activated = scaled_shifted > 0 ? scaled_shifted : scaled_shifted * negative_slope;

    // Element-wise add (x + x)
    output[row * hidden_size + col] = activated + activated;
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps,
    float negative_slope) {

    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::zeros({batch_size, hidden_size}, input.options());

    dim3 block(BLOCK_SIZE);
    dim3 grid(batch_size, (hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    size_t shared_mem_size = 2 * num_groups * sizeof(float);

    fused_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        num_groups,
        eps,
        negative_slope);

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda("
    "torch::Tensor input, "
    "torch::Tensor weight, "
    "torch::Tensor bias, "
    "torch::Tensor gamma, "
    "torch::Tensor beta, "
    "int num_groups, "
    "float eps, "
    "float negative_slope);"
)

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
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.fused_ops = fused_ops
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        weight = self.fc.weight
        bias = self.fc.bias
        gamma = self.gn.weight
        beta = self.gn.bias
        return self.fused_ops.fused_ops_cuda(
            x, weight, bias, gamma, beta,
            self.num_groups, self.eps, self.negative_slope
        )