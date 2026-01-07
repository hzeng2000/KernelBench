import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ConvTranspose3d + scale + maxpool3d + global_avg_pool3d + clamp
fused_conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

__global__ void fused_conv_transpose_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float scale,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int maxpool_kd, int maxpool_kh, int maxpool_kw,
    float clamp_min, float clamp_max) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_d * out_h * out_w;

    if (idx < total_threads) {
        int tmp = idx;
        int n = tmp / (out_channels * out_d * out_h * out_w);
        tmp %= (out_channels * out_d * out_h * out_w);
        int c = tmp / (out_d * out_h * out_w);
        tmp %= (out_d * out_h * out_w);
        int d = tmp / (out_h * out_w);
        tmp %= (out_h * out_w);
        int h = tmp / out_w;
        int w = tmp % out_w;

        float val = 0.0f;

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_d; ++kd) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int in_d_idx = (d + pad_d - kd) / stride_d;
                        int in_h_idx = (h + pad_h - kh) / stride_h;
                        int in_w_idx = (w + pad_w - kw) / stride_w;

                        if ((d + pad_d - kd) % stride_d != 0) continue;
                        if ((h + pad_h - kh) % stride_h != 0) continue;
                        if ((w + pad_w - kw) % stride_w != 0) continue;

                        if (in_d_idx >= 0 && in_d_idx < in_d &&
                            in_h_idx >= 0 && in_h_idx < in_h &&
                            in_w_idx >= 0 && in_w_idx < in_w) {

                            int in_idx = n * in_channels * in_d * in_h * in_w +
                                         ic * in_d * in_h * in_w +
                                         in_d_idx * in_h * in_w +
                                         in_h_idx * in_w +
                                         in_w_idx;

                            int weight_idx = c * in_channels * kernel_d * kernel_h * kernel_w +
                                             ic * kernel_d * kernel_h * kernel_w +
                                             kd * kernel_h * kernel_w +
                                             kh * kernel_w +
                                             kw;

                            val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            val += bias[c];
        }

        val *= scale;

        // MaxPool3d (kernel=2, stride=2, padding=0)
        int pd = d / maxpool_kd;
        int ph = h / maxpool_kh;
        int pw = w / maxpool_kw;
        if (d % maxpool_kd == 0 && h % maxpool_kh == 0 && w % maxpool_kw == 0) {
            int out_idx = n * out_channels * (out_d/maxpool_kd) * (out_h/maxpool_kh) * (out_w/maxpool_kw) +
                          c * (out_d/maxpool_kd) * (out_h/maxpool_kh) * (out_w/maxpool_kw) +
                          pd * (out_h/maxpool_kh) * (out_w/maxpool_kw) +
                          ph * (out_w/maxpool_kw) +
                          pw;
            output[out_idx] = val;
        }
    }
}

__global__ void global_avg_pool_clamp_kernel(
    const float* input, float* output,
    int batch_size, int channels, int spatial_size,
    float clamp_min, float clamp_max) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels) {
        int n = idx / channels;
        int c = idx % channels;

        float sum = 0.0f;
        for (int i = 0; i < spatial_size; ++i) {
            sum += input[n * channels * spatial_size + c * spatial_size + i];
        }
        float avg = sum / spatial_size;
        avg = fmaxf(fminf(avg, clamp_max), clamp_min);

        output[n * channels + c] = avg;
    }
}

torch::Tensor fused_conv_transpose_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale, int stride, int padding, int maxpool_kernel_size,
    float clamp_min, float clamp_max) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_d = input.size(2);
    const auto in_h = input.size(3);
    const auto in_w = input.size(4);

    const auto out_channels = weight.size(0);
    const auto kernel_d = weight.size(2);
    const auto kernel_h = weight.size(3);
    const auto kernel_w = weight.size(4);

    const int out_d = (in_d - 1) * stride - 2 * padding + kernel_d;
    const int out_h = (in_h - 1) * stride - 2 * padding + kernel_h;
    const int out_w = (in_w - 1) * stride - 2 * padding + kernel_w;

    auto pooled_d = out_d / maxpool_kernel_size;
    auto pooled_h = out_h / maxpool_kernel_size;
    auto pooled_w = out_w / maxpool_kernel_size;

    auto pooled = torch::zeros({batch_size, out_channels, pooled_d, pooled_h, pooled_w}, input.options());

    const int total_threads = batch_size * out_channels * out_d * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    fused_conv_transpose_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        pooled.data_ptr<float>(), scale,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride, stride, stride,
        padding, padding, padding,
        maxpool_kernel_size, maxpool_kernel_size, maxpool_kernel_size,
        clamp_min, clamp_max);

    CUDA_CHECK(cudaGetLastError());

    // Global average pooling: reduce spatial dims to 1x1x1
    auto global_pooled = torch::zeros({batch_size, out_channels, 1, 1, 1}, input.options());

    const int spatial_size = pooled_d * pooled_h * pooled_w;
    const int gap_threads = batch_size * out_channels;
    const int gap_blocks = (gap_threads + block_size - 1) / block_size;

    global_avg_pool_clamp_kernel<<<gap_blocks, block_size>>>(
        pooled.data_ptr<float>(), global_pooled.data_ptr<float>(),
        batch_size, out_channels, spatial_size,
        clamp_min, clamp_max);

    CUDA_CHECK(cudaGetLastError());

    return global_pooled;
}
"""

fused_conv_transpose_cpp_source = """
torch::Tensor fused_conv_transpose_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale, int stride, int padding, int maxpool_kernel_size,
    float clamp_min, float clamp_max);
"""

fused_ops = load_inline(
    name="fused_conv_transpose",
    cpp_sources=fused_conv_transpose_cpp_source,
    cuda_sources=fused_conv_transpose_source,
    functions=["fused_conv_transpose_forward"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.fused_ops = fused_ops

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        return self.fused_ops.fused_conv_transpose_forward(
            x, weight, bias, self.scale,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max
        ).view(x.size(0), -1)