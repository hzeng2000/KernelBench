import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ReLU and GroupNorm
fused_relu_group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_sum_kernel(const float* x, float* sum, float* sumsq, int N, int C, int D, int H, int W, int G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D * H * W;
    if (idx >= total) return;
    int n = idx / (C * D * H * W);
    int c = (idx / (D * H * W)) % C;
    int g = c / (C / G);
    float val = fmaxf(x[idx], 0.0f);
    atomicAdd(&sum[n * G + g], val);
    atomicAdd(&sumsq[n * G + g], val * val);
}

__global__ void normalize_kernel(const float* x, float* out, const float* mean, const float* var, const float* weight, const float* bias, int N, int C, int D, int H, int W, int G, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D * H * W;
    if (idx >= total) return;
    int n = idx / (C * D * H * W);
    int c = (idx / (D * H * W)) % C;
    int g = c / (C / G);
    float m = mean[n * G + g];
    float v = var[n * G + g];
    float val = fmaxf(x[idx], 0.0f);
    val = (val - m) / sqrtf(v + eps);
    val = val * weight[c] + bias[c];
    out[idx] = val;
}

torch::Tensor fused_relu_group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);
    auto G = num_groups;
    int channels_per_group = C / G;
    int num_elements_per_group = channels_per_group * D * H * W;
    auto sum = torch::zeros({N * G}, x.options());
    auto sumsq = torch::zeros({N * G}, x.options());
    int total_elements = N * C * D * H * W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    compute_sum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), sum.data_ptr<float>(), sumsq.data_ptr<float>(), N, C, D, H, W, G);
    auto mean = torch::zeros({N * G}, x.options());
    auto var = torch::zeros({N * G}, x.options());
    for (int i = 0; i < N * G; ++i) {
        float s = sum.index({i}).item<float>();
        float ss = sumsq.index({i}).item<float>();
        float m = s / num_elements_per_group;
        float v = ss / num_elements_per_group - m * m;
        mean.index_put_({i}, m);
        var.index_put_({i}, v);
    }
    auto out = torch::zeros_like(x);
    normalize_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), N, C, D, H, W, G, eps);
    return out;
}
"""

fused_relu_group_norm_cpp_source = (
    "torch::Tensor fused_relu_group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps);"
)

# Compile the inline CUDA code for fused ReLU and GroupNorm
fused_relu_group_norm = load_inline(
    name="fused_relu_group_norm",
    cpp_sources=fused_relu_group_norm_cpp_source,
    cuda_sources=fused_relu_group_norm_source,
    functions=["fused_relu_group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed 3D convolution, applies fused ReLU and GroupNorm using custom CUDA.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.fused_relu_group_norm = fused_relu_group_norm
        self.eps = 1e-5

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.fused_relu_group_norm.fused_relu_group_norm_cuda(x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups, self.eps)
        return x