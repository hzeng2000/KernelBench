import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for transposed convolution + maxpool + hardtanh + mean + tanh fusion
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 16

__global__ void fused_conv_transpose_maxpool_hardtanh_mean_tanh_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp_output, float* temp_pooled, float* temp_activated,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int pool_kernel, int pool_stride,
    float hardtanh_min, float hardtanh_max) {

    int b = blockIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || c >= out_channels || h >= out_height) return;

    int w = threadIdx.y; // reuse threadIdx.y for width iteration
    for (int out_w = w; out_w < out_width; out_w += blockDim.y) {

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_h = (h + padding - kh);
                    int in_w = (out_w + padding - kw);
                    if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                        float inp_val = input[b * in_channels * in_height * in_width +
                                              ic * in_height * in_width +
                                              in_h * in_width + in_w];
                        float w_val = weight[c * in_channels * kernel_size * kernel_size +
                                             ic * kernel_size * kernel_size +
                                             kh * kernel_size + kw];
                        sum += inp_val * w_val;
                    }
                }
            }
        }
        if (bias) sum += bias[c];
        temp_output[b * out_channels * out_height * out_width +
                    c * out_height * out_width +
                    h * out_width + out_w] = sum;
    }
}

__global__ void maxpool_kernel(const float* input, float* output,
                               int batch_size, int channels, int in_height, int in_width,
                               int out_height, int out_width, int kernel_size, int stride) {

    int b = blockIdx.z;
    int c = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || c >= channels || h >= out_height) return;

    int w = threadIdx.y;
    for (int out_w = w; out_w < out_width; out_w += blockDim.y) {
        float max_val = -1e20f;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_h = h * stride + kh;
                int in_w = out_w * stride + kw;
                if (in_h < in_height && in_w < in_width) {
                    float val = input[b * channels * in_height * in_width +
                                      c * in_height * in_width +
                                      in_h * in_width + in_w];
                    if (val > max_val) max_val = val;
                }
            }
        }
        output[b * channels * out_height * out_width +
               c * out_height * out_width +
               h * out_width + out_w] = max_val;
    }
}

__global__ void hardtanh_kernel(float* data, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        data[idx] = val;
    }
}

__global__ void mean_tanh_kernel(const float* input, float* output,
                                 int batch_size, int channels, int height, int width) {
    int b = blockIdx.x;
    int c = blockIdx.y;

    if (b >= batch_size || c >= channels) return;

    float sum = 0.0f;
    int size = height * width;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            sum += input[b * channels * height * width +
                         c * height * width +
                         h * width + w];
        }
    }
    float mean = sum / size;
    output[b * channels + c] = tanhf(mean);
}

torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                 int kernel_size, int stride, int padding,
                                 int pool_kernel, int pool_stride,
                                 float hardtanh_min, float hardtanh_max) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(0);

    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;

    const int pooled_height = out_height / pool_stride;
    const int pooled_width = out_width / pool_stride;

    auto temp_output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    auto temp_pooled = torch::zeros({batch_size, out_channels, pooled_height, pooled_width}, input.options());
    auto temp_activated = torch::zeros({batch_size, out_channels, pooled_height, pooled_width}, input.options());
    auto output = torch::zeros({batch_size, out_channels, 1, 1}, input.options());

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  batch_size);

    fused_conv_transpose_maxpool_hardtanh_mean_tanh_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), temp_output.data_ptr<float>(), temp_pooled.data_ptr<float>(), temp_activated.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_size, stride, padding,
        pool_kernel, pool_stride,
        hardtanh_min, hardtanh_max);

    dim3 poolBlockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 poolGridSize((pooled_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      out_channels,
                      batch_size);

    maxpool_kernel<<<poolGridSize, poolBlockSize>>>(
        temp_output.data_ptr<float>(), temp_pooled.data_ptr<float>(),
        batch_size, out_channels, out_height, out_width,
        pooled_height, pooled_width, pool_kernel, pool_stride);

    int activation_size = batch_size * out_channels * pooled_height * pooled_width;
    int activationThreads = 256;
    int activationBlocks = (activation_size + activationThreads - 1) / activationThreads;

    hardtanh_kernel<<<activationBlocks, activationThreads>>>(
        temp_pooled.data_ptr<float>(), activation_size, hardtanh_min, hardtanh_max);

    dim3 meanGridSize(batch_size, out_channels);
    mean_tanh_kernel<<<meanGridSize, 1>>>(
        temp_pooled.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, out_channels, pooled_height, pooled_width);

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                 int kernel_size, int stride, int padding,
                                 int pool_kernel, int pool_stride,
                                 float hardtanh_min, float hardtanh_max);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_ops = fused_ops

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        return self.fused_ops.fused_forward_cuda(
            x, weight, bias,
            self.conv_transpose.kernel_size[0],
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.maxpool_kernel_size,
            self.maxpool_stride,
            self.hardtanh_min,
            self.hardtanh_max
        )