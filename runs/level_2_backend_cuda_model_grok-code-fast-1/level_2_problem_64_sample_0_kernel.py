import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused logsumexp and activations
fused_logsumexp_activations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}

__global__ void fused_logsumexp_activations_kernel(const float* input, float* output, int batch_size, int out_features) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    const float* row = input + batch * out_features;
    float* out = output + batch;
    
    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }
    
    // Reduce max within block
    extern __shared__ float s_data[];
    float* s_max = s_data;
    s_max[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = s_max[0];
    
    // Compute sum exp(x - max)
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
        sum_exp += expf(row[i] - max_val);
    }
    
    float* s_sum = s_data;
    s_sum[threadIdx.x] = sum_exp;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_exp = s_sum[0];
    
    // logsumexp
    float lse = max_val + logf(sum_exp);
    
    // Apply activations: only thread 0 does it
    if (threadIdx.x == 0) {
        float val = lse;
        // LeakyReLU twice
        val = (val > 0.0f) ? val : 0.01f * val;
        val = (val > 0.0f) ? val : 0.01f * val;
        // GELU twice
        val = gelu(val);
        val = gelu(val);
        out[0] = val;
    }
}

torch::Tensor fused_logsumexp_activations_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    const int block_size = 1024;
    const int num_blocks = batch_size;
    size_t shared_mem_size = block_size * sizeof(float);
    
    fused_logsumexp_activations_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_features);
    
    return output;
}
"""

fused_logsumexp_activations_cpp_source = (
    "torch::Tensor fused_logsumexp_activations_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_logsumexp_activations = load_inline(
    name="fused_logsumexp_activations",
    cpp_sources=fused_logsumexp_activations_cpp_source,
    cuda_sources=fused_logsumexp_activations_source,
    functions=["fused_logsumexp_activations_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication (Gemm), followed by fused LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations using a custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_op = fused_logsumexp_activations

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        # Fused LogSumExp + activations
        x = self.fused_op.fused_logsumexp_activations_cuda(x)
        return x