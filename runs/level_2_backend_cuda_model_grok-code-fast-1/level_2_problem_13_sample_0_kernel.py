import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: mean pool over depth, bias add, softmax over channels, tanh, scaling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_ops_kernel(const float* input, const float* bias, float* output, int B, int C, int D, int H, int W, float scale) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int c = threadIdx.x;
    
    if (b >= B || h >= H || w >= W || c >= C) return;
    
    // Compute mean over depth dimension
    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        int idx = ((b * C + c) * D + d) * H * W + h * W + w;
        sum += input[idx];
    }
    float mean_val = sum / static_cast<float>(D);
    
    // Add bias
    mean_val += bias[c];
    
    // Shared memory for softmax computation
    extern __shared__ float shared_vals[];
    shared_vals[c] = mean_val;
    __syncthreads();
    
    // Compute max for softmax
    if (c == 0) {
        float max_val = shared_vals[0];
        for (int i = 1; i < C; ++i) {
            max_val = fmaxf(max_val, shared_vals[i]);
        }
        shared_vals[C] = max_val;  // Store max in shared_vals[C]
    }
    __syncthreads();
    
    // Compute exp
    float max_val = shared_vals[C];
    float exp_val = expf(shared_vals[c] - max_val);
    shared_vals[c] = exp_val;
    __syncthreads();
    
    // Compute sum of exp
    if (c == 0) {
        float sum_exp = 0.0f;
        for (int i = 0; i < C; ++i) {
            sum_exp += shared_vals[i];
        }
        shared_vals[C + 1] = sum_exp;  // Store sum in shared_vals[C+1]
    }
    __syncthreads();
    
    // Compute softmax
    float sum_exp = shared_vals[C + 1];
    float softmax_val = shared_vals[c] / sum_exp;
    
    // Apply tanh and scaling
    float result = tanhf(softmax_val) * scale;
    
    // Output index: (B, C, 1, H, W)
    int out_idx = ((b * C + c) * 1 * H * W) + h * W + w;
    output[out_idx] = result;
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float scale) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    auto output = torch::zeros({B, C, 1, H, W}, input.options());
    
    dim3 blocks(B, H, W);
    dim3 threads(C);
    size_t shared_mem_size = (C + 2) * sizeof(float);  // For shared_vals, max, and sum
    
    fused_ops_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        B, C, D, H, W, scale
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float scale);"
)

# Compile the inline CUDA code for fused operations
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
    """
    Optimized Model that performs a series of operations:
    1. Transposed 3D convolution (unchanged)
    2-6. Fused: Mean pooling (across depth), Addition, Softmax (across channels), Tanh activation, Scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))  # Broadcastable bias over channels
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)                            # (B, C, D, H, W)
        x = self.fused_ops.fused_ops_cuda(x, self.bias.squeeze(0).squeeze(2).squeeze(3).squeeze(4), self.scaling_factor)  # Fused operations
        return x