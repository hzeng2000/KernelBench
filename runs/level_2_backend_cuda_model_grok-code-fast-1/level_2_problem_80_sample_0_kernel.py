import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused max, mean subtraction, and GELU
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_kernel(const float* x, float* out, int batch_size, int out_features) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;

    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = &sdata[blockDim.x];

    int tid = threadIdx.x;
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (int j = tid; j < out_features; j += blockDim.x) {
        float val = x[batch * out_features + j];
        local_max = fmaxf(local_max, val);
        local_sum += val;
    }

    s_max[tid] = local_max;
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_val = s_max[0];
        float mean_val = s_sum[0] / out_features;
        float diff = max_val - mean_val;
        // GELU approximation
        float sqrt_2_pi = sqrtf(2.0f / M_PI);
        float x3 = diff * diff * diff;
        float inner = sqrt_2_pi * (diff + 0.044715f * x3);
        float gelu_val = 0.5f * diff * (1.0f + tanhf(inner));
        out[batch] = gelu_val;
    }
}

torch::Tensor fused_max_mean_sub_gelu_cuda(torch::Tensor x, int out_features) {
    auto batch_size = x.size(0);
    auto out = torch::zeros({batch_size, 1}, x.options());

    const int block_size = 256;
    const int num_blocks = batch_size;
    size_t shared_mem_size = 2 * block_size * sizeof(float);

    fused_kernel<<<num_blocks, block_size, shared_mem_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, out_features);

    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_max_mean_sub_gelu_cuda(torch::Tensor x, int out_features);"
)

# Compile the inline CUDA code for the fused operation
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_max_mean_sub_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs GEMM, followed by fused max, mean subtraction, and GELU.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.out_features = out_features
        self.max_dim = max_dim
        self.fused = fused_op

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        x = self.gemm(x)
        x = self.fused.fused_max_mean_sub_gelu_cuda(x, self.out_features)
        return x