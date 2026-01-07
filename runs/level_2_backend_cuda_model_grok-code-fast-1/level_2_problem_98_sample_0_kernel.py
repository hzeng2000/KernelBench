import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused avgpool, gelu, scale, and max
fused_pool_gelu_scale_max_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void fused_pool_gelu_scale_max_kernel(const float* x, float* out, float scale_factor, int batch_size, int out_features, int pool_size) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    int num_pooled = out_features / pool_size;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    if (tid < num_pooled) {
        // compute avg
        float sum = 0.0f;
        for (int i = 0; i < pool_size; i++) {
            sum += x[batch * out_features + tid * pool_size + i];
        }
        float avg = sum / pool_size;
        // gelu approximation
        float coeff = sqrtf(2.0f / M_PI);
        float tanh_arg = coeff * (avg + 0.044715f * avg * avg * avg);
        float gelu_val = 0.5f * avg * (1.0f + tanhf(tanh_arg));
        // scale
        gelu_val *= scale_factor;
        sdata[tid] = gelu_val;
    } else {
        sdata[tid] = -INFINITY;
    }
    __syncthreads();
    // reduce max
    for (int s = num_pooled / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[batch] = sdata[0];
    }
}

torch::Tensor fused_pool_gelu_scale_max_cuda(torch::Tensor x, float scale_factor, int pool_size) {
    int batch_size = x.size(0);
    int out_features = x.size(1);
    auto out = torch::zeros({batch_size}, x.options());
    int num_pooled = out_features / pool_size;
    int block_size = num_pooled;
    int shared_size = block_size * sizeof(float);
    fused_pool_gelu_scale_max_kernel<<<batch_size, block_size, shared_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), scale_factor, batch_size, out_features, pool_size);
    return out;
}
"""

fused_pool_gelu_scale_max_cpp_source = (
    "torch::Tensor fused_pool_gelu_scale_max_cuda(torch::Tensor x, float scale_factor, int pool_size);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_pool_gelu_scale_max",
    cpp_sources=fused_pool_gelu_scale_max_cpp_source,
    cuda_sources=fused_pool_gelu_scale_max_source,
    functions=["fused_pool_gelu_scale_max_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused avgpool, gelu, scale, and max in custom CUDA.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.fused_op = fused_op
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x = self.matmul(x)
        return self.fused_op.fused_pool_gelu_scale_max_cuda(x, self.scale_factor, self.pool_kernel_size)