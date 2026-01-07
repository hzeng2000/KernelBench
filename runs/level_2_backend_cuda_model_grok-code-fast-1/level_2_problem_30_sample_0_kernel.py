import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm and HardTanh
group_norm_hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void group_norm_stats_kernel(const float* x, float* means, float* vars, int N, int C, int G, int group_size, float eps) {
    int n = blockIdx.x;
    int g = blockIdx.y;
    int tid = threadIdx.x;
    if (n >= N || g >= G) return;

    extern __shared__ float sdata[];
    float* ssum = sdata;
    float* ssumsq = sdata + blockDim.x;

    ssum[tid] = 0.0f;
    ssumsq[tid] = 0.0f;

    for (int c = tid; c < group_size; c += blockDim.x) {
        int idx = n * C + g * group_size + c;
        float val = x[idx];
        ssum[tid] += val;
        ssumsq[tid] += val * val;
    }

    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            ssumsq[tid] += ssumsq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = ssum[0] / group_size;
        float var = (ssumsq[0] / group_size) - mean * mean;
        means[n * G + g] = mean;
        vars[n * G + g] = var;
    }
}

__global__ void group_norm_norm_kernel(const float* x, float* out, const float* means, const float* vars, const float* weight, const float* bias, int N, int C, int G, int group_size, float min_val, float max_val, float eps) {
    int n = blockIdx.x;
    int g = blockIdx.y;
    int c_in_group = threadIdx.x;
    if (n >= N || g >= G || c_in_group >= group_size) return;

    int c = g * group_size + c_in_group;
    int idx = n * C + c;
    float mean = means[n * G + g];
    float var = vars[n * G + g];
    float std = sqrtf(var + eps);
    float val = (x[idx] - mean) / std;
    val = val * weight[c] + bias[c];
    val = fmaxf(fminf(val, max_val), min_val);
    out[idx] = val;
}

torch::Tensor group_norm_hardtanh_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups, float min_val, float max_val) {
    int N = x.size(0);
    int C = x.size(1);
    int G = num_groups;
    int group_size = C / G;
    float eps = 1e-5f;

    auto means = torch::zeros({N, G}, x.options());
    auto vars = torch::zeros({N, G}, x.options());
    auto out = torch::zeros_like(x);

    dim3 blocks_stats(N, G);
    int threads_stats = 256;
    int shared_size = 2 * threads_stats * sizeof(float);
    group_norm_stats_kernel<<<blocks_stats, threads_stats, shared_size>>>(x.data_ptr<float>(), means.data_ptr<float>(), vars.data_ptr<float>(), N, C, G, group_size, eps);

    dim3 blocks_norm(N, G);
    int threads_norm = group_size;
    group_norm_norm_kernel<<<blocks_norm, threads_norm>>>(x.data_ptr<float>(), out.data_ptr<float>(), means.data_ptr<float>(), vars.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), N, C, G, group_size, min_val, max_val, eps);

    return out;
}
"""

group_norm_hardtanh_cpp_source = (
    "torch::Tensor group_norm_hardtanh_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups, float min_val, float max_val);"
)

# Compile the inline CUDA code for fused GroupNorm and HardTanh
fused_group_norm_hardtanh = load_inline(
    name="fused_group_norm_hardtanh",
    cpp_sources=group_norm_hardtanh_cpp_source,
    cuda_sources=group_norm_hardtanh_source,
    functions=["group_norm_hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs GEMM, applies fused Group Normalization and HardTanh.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.fused_op = fused_group_norm_hardtanh
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.fused_op.group_norm_hardtanh_cuda(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.hardtanh_min, self.hardtanh_max)
        return x