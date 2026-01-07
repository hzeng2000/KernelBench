import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GEMM + GroupNorm + Swish fusion
fused_gemm_norm_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 256

__global__ void group_norm_swish_kernel(
    float* out, const float* weight, const float* bias,
    int batch_size, int out_features, int num_groups, int group_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;
    
    int batch_idx = idx / out_features;
    int feature_idx = idx % out_features;
    int group_idx = feature_idx / group_size;
    
    // Compute group mean and variance
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int group_start = batch_idx * out_features + group_idx * group_size;
    
    for (int i = 0; i < group_size; i++) {
        float val = out[group_start + i];
        sum += val;
        sum_sq += val * val;
    }
    
    float mean = sum / group_size;
    float var = (sum_sq / group_size) - (mean * mean);
    float std = sqrtf(var + 1e-5f);
    
    // Normalize, apply weight and bias, then swish
    float normalized = (out[idx] - mean) / std;
    float weighted = normalized * weight[feature_idx] + bias[feature_idx];
    out[idx] = weighted / (1.0f + expf(-weighted));
}

torch::Tensor fused_gemm_norm_swish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gn_weight, torch::Tensor gn_bias,
    int num_groups) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    int group_size = out_features / num_groups;
    
    auto out = torch::zeros({batch_size, out_features}, input.options());
    
    // GEMM using cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                out.data_ptr<float>(), out_features);
    
    // Launch GroupNorm + Swish kernel
    int total_threads = batch_size * out_features;
    int num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    group_norm_swish_kernel<<<num_blocks, BLOCK_SIZE>>>(
        out.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        batch_size, out_features, num_groups, group_size);
    
    cublasDestroy(handle);
    return out;
}
"""

fused_gemm_norm_swish_cpp_source = """
torch::Tensor fused_gemm_norm_swish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gn_weight, torch::Tensor gn_bias,
    int num_groups);
"""

# Custom CUDA kernel for element-wise multiply + swish
multiply_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void multiply_swish_kernel(
    float* x, const float* weight, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] * weight[idx % (size / gridDim.x)];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        x[idx] = val * sigmoid;
    }
}

torch::Tensor multiply_swish_cuda(torch::Tensor x, torch::Tensor weight) {
    auto size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    multiply_swish_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), size);
    
    return x;
}
"""

multiply_swish_cpp_source = """
torch::Tensor multiply_swish_cuda(torch::Tensor x, torch::Tensor weight);
"""

# Compile the inline CUDA code
fused_gemm_norm_swish = load_inline(
    name="fused_gemm_norm_swish",
    cpp_sources=fused_gemm_norm_swish_cpp_source,
    cuda_sources=fused_gemm_norm_swish_source,
    functions=["fused_gemm_norm_swish_cuda"],
    verbose=True,
    extra_cflags=["-lcublas"],
    extra_ldflags=["-lcublas"],
)

multiply_swish = load_inline(
    name="multiply_swish",
    cpp_sources=multiply_swish_cpp_source,
    cuda_sources=multiply_swish_source,
    functions=["multiply_swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.zeros(out_features))
        self.group_norm_weight = nn.Parameter(torch.ones(out_features))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_features))
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.num_groups = num_groups
        
        self.fused_gemm_norm_swish = fused_gemm_norm_swish
        self.multiply_swish = multiply_swish

    def forward(self, x):
        x = self.fused_gemm_norm_swish.fused_gemm_norm_swish_cuda(
            x, self.gemm_weight, self.gemm_bias,
            self.group_norm_weight, self.group_norm_bias,
            self.num_groups)
        x = self.multiply_swish.multiply_swish_cuda(x, self.multiply_weight)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]