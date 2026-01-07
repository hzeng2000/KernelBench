import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + GroupNorm + HardTanh
fused_gemm_gn_ht_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void fused_gemm_gn_ht_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    float hardtanh_min,
    float hardtanh_max,
    float eps
) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int gid = threadIdx.y;
    int group_size = out_features / num_groups;

    extern __shared__ float shared_mem[];
    float* shared_mean = shared_mem;
    float* shared_var = &shared_mem[blockDim.y];
    float* shared_out = &shared_mem[2 * blockDim.y];

    if (row >= batch_size) return;

    // Compute GEMM for this row
    for (int col = gid; col < out_features; col += blockDim.y) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        if (bias != nullptr) {
            sum += bias[col];
        }
        shared_out[gid] = sum;

        // Compute group statistics
        int group_id = col / group_size;
        int group_start = group_id * group_size;
        int group_end = (group_id + 1) * group_size;

        // Compute mean
        __syncthreads();
        float mean = 0.0f;
        for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
            mean += shared_out[i];
        }
        mean /= group_size;

        // Compute variance
        float var = 0.0f;
        for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
            float diff = shared_out[i] - mean;
            var += diff * diff;
        }
        var /= group_size;

        // Normalize
        float std = sqrtf(var + eps);
        float normalized = (shared_out[gid - group_start] - mean) / std;

        // Scale and shift
        float gn_out = gamma[col] * normalized + beta[col];

        // Apply HardTanh
        gn_out = fminf(fmaxf(gn_out, hardtanh_min), hardtanh_max);

        output[row * out_features + col] = gn_out;
    }
}

torch::Tensor fused_gemm_gn_ht_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float hardtanh_min,
    float hardtanh_max,
    float eps
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::zeros({batch_size, out_features}, input.options());

    const dim3 blocks(batch_size);
    const dim3 threads(BLOCK_SIZE, 1);
    const int shared_mem_size = (2 * num_groups + out_features) * sizeof(float);

    fused_gemm_gn_ht_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        hardtanh_min,
        hardtanh_max,
        eps
    );

    return output;
}
"""

fused_gemm_gn_ht_cpp_source = (
    "torch::Tensor fused_gemm_gn_ht_cuda("
    "torch::Tensor input, "
    "torch::Tensor weight, "
    "torch::Tensor bias, "
    "torch::Tensor gamma, "
    "torch::Tensor beta, "
    "int num_groups, "
    "float hardtanh_min, "
    "float hardtanh_max, "
    "float eps);"
)

# Compile the inline CUDA code
fused_gemm_gn_ht = load_inline(
    name="fused_gemm_gn_ht",
    cpp_sources=fused_gemm_gn_ht_cpp_source,
    cuda_sources=fused_gemm_gn_ht_source,
    functions=["fused_gemm_gn_ht_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.eps = 1e-5
        self.fused_op = fused_gemm_gn_ht

    def forward(self, x):
        weight = self.gemm.weight
        bias = self.gemm.bias
        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        
        return self.fused_op.fused_gemm_gn_ht_cuda(
            x, weight, bias, gamma, beta,
            self.group_norm.num_groups,
            self.hardtanh_min,
            self.hardtanh_max,
            self.eps
        )