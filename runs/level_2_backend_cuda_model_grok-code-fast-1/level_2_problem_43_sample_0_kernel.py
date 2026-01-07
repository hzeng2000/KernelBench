import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused logsumexp and ReLU
logsumexp_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void logsumexp_relu_kernel(const float* input, float* output, int B, int C, int D, int H, int W) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int h = blockIdx.z;
    int w = threadIdx.x;
    if (b >= B || d >= D || h >= H || w >= W) return;

    // Compute max over channels for numerical stability
    float max_val = -INFINITY;
    for (int c = 0; c < C; ++c) {
        int idx = ((b * C + c) * D + d) * H * W + h * W + w;
        max_val = fmaxf(max_val, input[idx]);
    }

    // Compute sum of exp(x - max_val)
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        int idx = ((b * C + c) * D + d) * H * W + h * W + w;
        sum_exp += expf(input[idx] - max_val);
    }

    // Compute logsumexp and apply ReLU
    float lse = logf(sum_exp) + max_val;
    int out_idx = ((b * 1 + 0) * D + d) * H * W + h * W + w;
    output[out_idx] = fmaxf(0.0f, lse);
}

torch::Tensor logsumexp_relu_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    int B = sizes[0], C = sizes[1], D = sizes[2], H = sizes[3], W = sizes[4];
    auto output = torch::zeros({B, 1, D, H, W}, input.options());

    dim3 grid(B, D, H);
    dim3 block(W);

    logsumexp_relu_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), B, C, D, H, W);

    return output;
}
"""

logsumexp_relu_cpp_source = (
    "torch::Tensor logsumexp_relu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for fused logsumexp and ReLU
logsumexp_relu = load_inline(
    name="logsumexp_relu",
    cpp_sources=logsumexp_relu_cpp_source,
    cuda_sources=logsumexp_relu_source,
    functions=["logsumexp_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, max pooling, fused logsumexp and ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.logsumexp_relu = logsumexp_relu

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, 1, depth', height', width')
        """
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.logsumexp_relu.logsumexp_relu_cuda(x)
        return x