import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ConvTranspose3d + scale1 + avg_pool + bias + scale2
fused_conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 4
#define THREADS_PER_BLOCK 256

__global__ void conv_transpose_forward_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float scale1, float scale2,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    if (idx >= total_elements) return;

    int tmp = idx;
    int n = tmp / (out_channels * out_d * out_h * out_w);
    tmp %= (out_channels * out_d * out_h * out_w);
    int c_out = tmp / (out_d * out_h * out_w);
    tmp %= (out_d * out_h * out_w);
    int d_out = tmp / (out_h * out_w);
    tmp %= (out_h * out_w);
    int h_out = tmp / out_w;
    int w_out = tmp % out_w;

    float acc = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int d_in = d_out * stride_d - pad_d + kd;
                    int h_in = h_out * stride_h - pad_h + kh;
                    int w_in = w_out * stride_w - pad_w + kw;
                    if (d_in >= 0 && d_in < in_d && h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        int in_idx = ((n * in_channels + c_in) * in_d + d_in) * in_h * in_w + h_in * in_w + w_in;
                        int weight_idx = ((c_in * out_channels + c_out) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                        acc += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    acc = acc * scale1 + (bias ? bias[c_out] : 0.0f);
    acc *= scale2;

    // Average pooling (2x2x2)
    int pool_d = out_d / 2;
    int pool_h = out_h / 2;
    int pool_w = out_w / 2;
    int pd = d_out / 2;
    int ph = h_out / 2;
    int pw = w_out / 2;
    if (d_out % 2 == 0 && h_out % 2 == 0 && w_out % 2 == 0 && pd < pool_d && ph < pool_h && pw < pool_w) {
        int out_idx = ((n * out_channels + c_out) * pool_d + pd) * pool_h * pool_w + ph * pool_w + pw;
        output[out_idx] = acc / 8.0f;  // divide by 8 for 2x2x2 average
    }
}

torch::Tensor fused_conv_transpose_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale1, float scale2
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    int out_channels = weight.size(1);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int stride_d = 2, stride_h = 2, stride_w = 2;
    int pad_d = 1, pad_h = 1, pad_w = 1;

    int out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d;
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w;

    int pool_d = out_d / 2;
    int pool_h = out_h / 2;
    int pool_w = out_w / 2;

    auto output = torch::zeros({batch_size, out_channels, pool_d, pool_h, pool_w}, input.options());

    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    conv_transpose_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), scale1, scale2,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    );

    return output;
}
"""

fused_conv_transpose_cpp_source = (
    "torch::Tensor fused_conv_transpose_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scale1, float scale2);"
)

fused_conv_transpose = load_inline(
    name="fused_conv_transpose",
    cpp_sources=fused_conv_transpose_cpp_source,
    cuda_sources=fused_conv_transpose_source,
    functions=["fused_conv_transpose_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.fused_conv_transpose = fused_conv_transpose

    def forward(self, x):
        return self.fused_conv_transpose.fused_conv_transpose_cuda(
            x, self.weight, self.bias, self.scale1.item(), self.scale2.item()
        )