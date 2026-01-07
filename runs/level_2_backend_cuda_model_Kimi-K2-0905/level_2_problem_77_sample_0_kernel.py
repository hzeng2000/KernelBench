import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ConvTranspose3d + scale + ReLU (as activation) + BatchNorm3d + GlobalAvgPool3d
fused_conv_transpose_bn_gap_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_bn_gap_kernel(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, const float* gamma, const float* beta,
    float scale_factor, float eps,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    float* output) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_d * out_h * out_w;
    if (idx >= total_threads) return;

    int tmp = idx;
    int n = tmp / (out_channels * out_d * out_h * out_w);
    tmp %= (out_channels * out_d * out_h * out_w);
    int c = tmp / (out_d * out_h * out_w);
    tmp %= (out_d * out_h * out_w);
    int z = tmp / (out_h * out_w);
    tmp %= (out_h * out_w);
    int y = tmp / out_w;
    int x = tmp % out_w;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_z = z - kd;
                    int in_y = y - kh;
                    int in_x = x - kw;

                    if (in_z >= 0 && in_z < in_d && in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                        int in_idx = n * in_channels * in_d * in_h * in_w +
                                     ic * in_d * in_h * in_w +
                                     in_z * in_h * in_w +
                                     in_y * in_w +
                                     in_x;

                        int weight_idx = c * in_channels * kernel_d * kernel_h * kernel_w +
                                         ic * kernel_d * kernel_h * kernel_w +
                                         kd * kernel_h * kernel_w +
                                         kh * kernel_w +
                                         kw;

                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c];
    }

    sum *= scale_factor;

    // BatchNorm inference: (x - mean) / sqrt(var + eps) * gamma + beta
    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = rsqrtf(var + eps);
    float bn_val = (sum - mean) * inv_std * gamma[c] + beta[c];

    // ReLU activation
    bn_val = fmaxf(bn_val, 0.0f);

    // Global average pooling: accumulate and divide by volume later
    atomicAdd(&output[n * out_channels + c], bn_val / (out_d * out_h * out_w));
}

torch::Tensor fused_conv_transpose_bn_gap_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor gamma, torch::Tensor beta,
    float scale_factor, float eps) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_d = input.size(2);
    const int in_h = input.size(3);
    const int in_w = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    const int out_d = in_d + kernel_d - 1;
    const int out_h = in_h + kernel_h - 1;
    const int out_w = in_w + kernel_w - 1;

    auto output = torch::zeros({batch_size, out_channels}, input.options());

    const int total_threads = batch_size * out_channels * out_d * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    fused_conv_transpose_bn_gap_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        scale_factor, eps,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        output.data_ptr<float>());

    return output;
}
"""

fused_conv_transpose_bn_gap_cpp_source = (
    "torch::Tensor fused_conv_transpose_bn_gap_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor gamma, torch::Tensor beta,"
    "float scale_factor, float eps);"
)

fused_conv_transpose_bn_gap = load_inline(
    name="fused_conv_transpose_bn_gap",
    cpp_sources=fused_conv_transpose_bn_gap_cpp_source,
    cuda_sources=fused_conv_transpose_bn_gap_source,
    functions=["fused_conv_transpose_bn_gap_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.fused_op = fused_conv_transpose_bn_gap

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        return self.fused_op.fused_conv_transpose_bn_gap_cuda(
            x, weight, bias, running_mean, running_var, gamma, beta,
            self.scale_factor, self.batch_norm.eps
        ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)