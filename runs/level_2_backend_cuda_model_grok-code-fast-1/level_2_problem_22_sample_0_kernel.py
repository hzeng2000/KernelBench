import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-matmul operations: scaling, doubling, clamping, logsumexp, and mish multiplication
fused_post_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_post_kernel(const float* x, float* out, int batch_size, int hidden_size, float scale_factor, float clamp_min, float clamp_max) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    int tid = threadIdx.x;
    float sum_exp = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[batch * hidden_size + i];
        val = val * scale_factor * 2.0f;
        val = fmaxf(fminf(val, clamp_max), clamp_min);
        sum_exp += expf(val);
    }
    
    __shared__ float shared[1024];
    shared[tid] = sum_exp;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float logsumexp_val = logf(shared[0]);
        float softplus = logf(1.0f + expf(logsumexp_val));
        float mish_val = logsumexp_val * tanhf(softplus);
        out[batch] = logsumexp_val * mish_val;
    }
}

torch::Tensor fused_post_cuda(torch::Tensor x, float scale_factor, float clamp_min, float clamp_max) {
    auto batch_size = x.size(0);
    auto hidden_size = x.size(1);
    auto out = torch::zeros({batch_size, 1}, x.options());
    
    const int block_size = 1024;
    const int num_blocks = batch_size;
    
    fused_post_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, hidden_size, scale_factor, clamp_min, clamp_max);
    
    return out;
}
"""

fused_post_cpp_source = (
    "torch::Tensor fused_post_cuda(torch::Tensor x, float scale_factor, float clamp_min, float clamp_max);"
)

# Compile the inline CUDA code for fused post-matmul operations
fused_post = load_inline(
    name="fused_post",
    cpp_sources=fused_post_cpp_source,
    cuda_sources=fused_post_source,
    functions=["fused_post_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication, then fuses the scaling, doubling, clamping, LogSumExp, and Mish activation into a custom CUDA kernel.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_post = fused_post

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.matmul(x)
        return self.fused_post.fused_post_cuda(x, self.scale_factor, self.clamp_min, self.clamp_max)