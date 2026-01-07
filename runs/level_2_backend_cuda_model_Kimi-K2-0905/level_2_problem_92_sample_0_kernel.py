import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused conv + group norm + tanh + hardswish + residual + logsumexp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__device__ inline float hard_swish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_conv_gn_tanh_hswish_res_logsumexp_kernel(
    const float* input, const float* weight, const float* bias,
    const float* gamma, const float* beta,
    float* output, float* workspace,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int groups,
    int out_height, int out_width, int group_size, float eps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_height * out_width;
    if (idx >= total_threads) return;

    int n = idx / (out_channels * out_height * out_width);
    int c = (idx / (out_height * out_width)) % out_channels;
    int h = (idx / out_width) % out_height;
    int w = idx % out_width;

    // Convolution
    float conv_val = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_h = h + kh;
                int in_w = w + kw;
                if (in_h < height && in_w < width) {
                    int in_idx = n * in_channels * height * width + ic * height * width + in_h * width + in_w;
                    int weight_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                    conv_val += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }
    if (bias != nullptr) {
        conv_val += bias[c];
    }

    // Group Normalization
    int group = c / group_size;
    float mean = 0.0f;
    float var = 0.0f;

    // Compute mean
    for (int gc = group * group_size; gc < (group + 1) * group_size; ++gc) {
        if (gc < out_channels) {
            mean += conv_val;
        }
    }
    mean /= group_size;

    // Compute variance
    for (int gc = group * group_size; gc < (group + 1) * group_size; ++gc) {
        if (gc < out_channels) {
            float diff = conv_val - mean;
            var += diff * diff;
        }
    }
    var /= group_size;

    // Normalize
    float norm_val = (conv_val - mean) / sqrtf(var + eps);
    norm_val = norm_val * gamma[c] + beta[c];

    // Tanh
    float tanh_val = tanhf(norm_val);

    // HardSwish
    float hswish_val = hard_swish(tanh_val);

    // Residual
    float res_val = conv_val + hswish_val;

    // Store intermediate for logsumexp
    workspace[idx] = res_val;

    __syncthreads();

    // LogSumExp reduction along channel dimension
    if (c == 0) {
        float max_val = -INFINITY;
        for (int cc = 0; cc < out_channels; ++cc) {
            int ws_idx = n * out_channels * out_height * out_width + cc * out_height * out_width + h * out_width + w;
            max_val = fmaxf(max_val, workspace[ws_idx]);
        }

        float sum_exp = 0.0f;
        for (int cc = 0; cc < out_channels; ++cc) {
            int ws_idx = n * out_channels * out_height * out_width + cc * out_height * out_width + h * out_width + w;
            sum_exp += expf(workspace[ws_idx] - max_val);
        }

        int out_idx = n * 1 * out_height * out_width + h * out_width + w;
        output[out_idx] = logf(sum_exp) + max_val;
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int groups, float eps) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_height = height;
    const int out_width = width;

    const int group_size = out_channels / groups;

    auto output = torch::zeros({batch_size, 1, out_height, out_width}, input.options());
    auto workspace = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int total_threads = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fused_conv_gn_tanh_hswish_res_logsumexp_kernel<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(), workspace.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width, kernel_size, groups,
        out_height, out_width, group_size, eps);

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int groups, float eps);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.fused_ops = fused_ops

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        return self.fused_ops.fused_forward_cuda(x, weight, bias, gamma, beta, self.groups, self.eps)