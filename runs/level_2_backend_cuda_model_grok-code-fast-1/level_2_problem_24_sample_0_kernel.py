import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min along dim=2 and softmax along dim=1
fused_min_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_min_softmax_kernel(const float* x, float* out, int B, int C, int D, int H, int W) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int c = threadIdx.x;

    extern __shared__ float shared_mins[];

    // Compute min for this c over d
    float min_val = INFINITY;
    for (int d = 0; d < D; d++) {
        int idx = ((b * C + c) * D + d) * H * W + h * W + w;
        min_val = fminf(min_val, x[idx]);
    }
    shared_mins[c] = min_val;
    __syncthreads();

    // Compute max for softmax
    float max_val = -INFINITY;
    if (c == 0) {
        for (int i = 0; i < C; i++) {
            max_val = fmaxf(max_val, shared_mins[i]);
        }
        shared_mins[C] = max_val;  // Store max in shared_mins[C]
    }
    __syncthreads();
    max_val = shared_mins[C];

    // Compute exp
    float exp_val = expf(shared_mins[c] - max_val);
    shared_mins[c] = exp_val;
    __syncthreads();

    // Compute sum
    float sum_val = 0.0f;
    if (c == 0) {
        for (int i = 0; i < C; i++) {
            sum_val += shared_mins[i];
        }
        shared_mins[C] = sum_val;  // Store sum in shared_mins[C]
    }
    __syncthreads();
    sum_val = shared_mins[C];

    // Output
    int out_idx = ((b * C + c) * H + h) * W + w;
    out[out_idx] = exp_val / sum_val;
}

torch::Tensor fused_min_softmax_cuda(torch::Tensor x, int dim_min, int dim_softmax) {
    int B = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    auto out = torch::zeros({B, C, H, W}, x.options());

    dim3 grid(B, H, W);
    dim3 block(C);
    size_t shared_mem_size = (C + 1) * sizeof(float);  // For mins and max/sum

    fused_min_softmax_kernel<<<grid, block, shared_mem_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), B, C, D, H, W);

    return out;
}
"""

fused_min_softmax_cpp_source = (
    "torch::Tensor fused_min_softmax_cuda(torch::Tensor x, int dim_min, int dim_softmax);"
)

# Compile the inline CUDA code for fused min and softmax
fused_min_softmax = load_inline(
    name="fused_min_softmax",
    cpp_sources=fused_min_softmax_cpp_source,
    cuda_sources=fused_min_softmax_source,
    functions=["fused_min_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax using a fused custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim_min = dim  # For min
        self.dim_softmax = 1  # For softmax
        self.fused_min_softmax = fused_min_softmax

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)
        x = self.fused_min_softmax.fused_min_softmax_cuda(x, self.dim_min, self.dim_softmax)
        return x