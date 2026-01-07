import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused maxpool, sum, and scale
fused_maxpool_sum_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_maxpool_sum_scale_kernel(const float* x, float* y, float scale, int batch_size, int in_features) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_pairs = in_features / 2;
    
    float local_sum = 0.0f;
    for (int j = tid; j < num_pairs; j += num_threads) {
        int idx = batch * in_features + 2 * j;
        float m = fmaxf(x[idx], x[idx + 1]);
        local_sum += m;
    }
    
    extern __shared__ float sdata[];
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        y[batch] = sdata[0] * scale;
    }
}

torch::Tensor fused_maxpool_sum_scale_cuda(torch::Tensor x, float scale) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    auto y = torch::zeros({batch_size}, x.options());
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = block_size * sizeof(float);
    
    fused_maxpool_sum_scale_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), scale, batch_size, in_features
    );
    
    return y;
}
"""

fused_maxpool_sum_scale_cpp_source = (
    "torch::Tensor fused_maxpool_sum_scale_cuda(torch::Tensor x, float scale);"
)

# Compile the inline CUDA code for fused maxpool, sum, and scale
fused_op = load_inline(
    name="fused_maxpool_sum_scale",
    cpp_sources=fused_maxpool_sum_scale_cpp_source,
    cuda_sources=fused_maxpool_sum_scale_source,
    functions=["fused_maxpool_sum_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs matrix multiplication, fused max pooling, sum, and scaling using custom CUDA.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x = self.matmul(x)
        return self.fused_op.fused_maxpool_sum_scale_cuda(x, self.scale_factor)