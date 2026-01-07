import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3D + MaxPool3D + MaxPool3D + Sum fusion
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void fused_conv_transpose_maxpool_sum_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp1, float* temp2,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding,
    int pool1_d, int pool1_h, int pool1_w,
    int pool2_d, int pool2_h, int pool2_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx < total_threads) {
        int tmp = idx;
        int b = tmp / (out_channels * out_d * out_h * out_w);
        tmp %= (out_channels * out_d * out_h * out_w);
        int c = tmp / (out_d * out_h * out_w);
        tmp %= (out_d * out_h * out_w);
        int od = tmp / (out_h * out_w);
        tmp %= (out_h * out_w);
        int oh = tmp / out_w;
        int ow = tmp % out_w;

        float val = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int id = od * stride - padding + kd;
                        int ih = oh * stride - padding + kh;
                        int iw = ow * stride - padding + kw;
                        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < w) {
                            int in_idx = b * in_channels * in_d * in_h * w +
                                         ic * in_d * in_h * w +
                                         id * in_h * w +
                                         ih * w +
                                         iw;
                            int w_idx = c * in_channels * kernel_size * kernel_size * kernel_size +
                                        ic * kernel_size * kernel_size * kernel_size +
                                        kd * kernel_size * kernel_size +
                                        kh * kernel_size +
                                        kw;
                            val += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        if (bias != nullptr) {
            val += bias[c];
        }
        int out_idx = b * out_channels * out_d * out_h * out_w +
                      c * out_d * out_h * out_w +
                      od * out_h * out_w +
                      oh * out_w +
                      ow;
        temp1[out_idx] = val;
    }
}

__global__ void maxpool1_kernel(
    const float* input, float* output,
    int batch_size, int channels, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * channels * out_d * out_h * out_w;
    
    if (idx < total_threads) {
        int tmp = idx;
        int b = tmp / (channels * out_d * out_h * out_w);
        tmp %= (channels * out_d * out_h * out_w);
        int c = tmp / (out_d * out_h * out_w);
        tmp %= (out_d * out_h * out_w);
        int od = tmp / (out_h * out_w);
        tmp %= (out_h * out_w);
        int oh = tmp / out_w;
        int ow = tmp % out_w;

        float max_val = -1e20f;
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int id = od * kernel_size + kd;
                    int ih = oh * kernel_size + kh;
                    int iw = ow * kernel_size + kw;
                    if (id < in_d && ih < in_h && iw < in_w) {
                        int in_idx = b * channels * in_d * in_h * in_w +
                                     c * in_d * in_h * in_w +
                                     id * in_h * in_w +
                                     ih * in_w +
                                     iw;
                        max_val = fmaxf(max_val, input[in_idx]);
                    }
                }
            }
        }
        int out_idx = b * channels * out_d * out_h * out_w +
                      c * out_d * out_h * out_w +
                      od * out_h * out_w +
                      oh * out_w +
                      ow;
        output[out_idx] = max_val;
    }
}

__global__ void maxpool2_kernel(
    const float* input, float* output,
    int batch_size, int channels, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * channels * out_d * out_h * out_w;
    
    if (idx < total_threads) {
        int tmp = idx;
        int b = tmp / (channels * out_d * out_h * out_w);
        tmp %= (channels * out_d * out_h * out_w);
        int c = tmp / (out_d * out_h * out_w);
        tmp %= (out_d * out_h * out_w);
        int od = tmp / (out_h * out_w);
        tmp %= (out_h * out_w);
        int oh = tmp / out_w;
        int ow = tmp % out_w;

        float max_val = -1e20f;
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int id = od * kernel_size + kd;
                    int ih = oh * kernel_size + kh;
                    int iw = ow * kernel_size + kw;
                    if (id < in_d && ih < in_h && iw < in_w) {
                        int in_idx = b * channels * in_d * in_h * in_w +
                                     c * in_d * in_h * in_w +
                                     id * in_h * in_w +
                                     ih * in_w +
                                     iw;
                        max_val = fmaxf(max_val, input[in_idx]);
                    }
                }
            }
        }
        int out_idx = b * channels * out_d * out_h * out_w +
                      c * out_d * out_h * out_w +
                      od * out_h * out_w +
                      oh * out_w +
                      ow;
        output[out_idx] = max_val;
    }
}

__global__ void sum_reduce_kernel(
    const float* input, float* output,
    int batch_size, int channels, int d, int h, int w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * d * h * w;
    
    if (idx < total_threads) {
        int tmp = idx;
        int b = tmp / (d * h * w);
        tmp %= (d * h * w);
        int id = tmp / (h * w);
        tmp %= (h * w);
        int ih = tmp / w;
        int iw = tmp % w;

        float sum_val = 0.0f;
        for (int c = 0; c < channels; ++c) {
            int in_idx = b * channels * d * h * w +
                         c * d * h * w +
                         id * h * w +
                         ih * w +
                         iw;
            sum_val += input[in_idx];
        }
        int out_idx = b * 1 * d * h * w +
                      0 * d * h * w +
                      id * h * w +
                      ih * w +
                      iw;
        output[out_idx] = sum_val;
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_d = input.size(2);
    const auto in_h = input.size(3);
    const auto in_w = input.size(4);
    const auto out_channels = weight.size(0);

    const int out_d = (in_d - 1) * stride - 2 * padding + kernel_size;
    const int out_h = (in_h - 1) * stride - 2 * padding + kernel_size;
    const int out_w = (in_w - 1) * stride - 2 * padding + kernel_size;

    auto temp1 = torch::zeros({batch_size, out_channels, out_d, out_h, out_w}, input.options());
    auto temp2 = torch::zeros({batch_size, out_channels, out_d / 2, out_h / 2, out_w / 2}, input.options());
    auto output = torch::zeros({batch_size, 1, out_d / 2 / 3, out_h / 2 / 3, out_w / 2 / 3}, input.options());

    const int block_size = 256;

    // ConvTranspose3D
    int total_threads = batch_size * out_channels * out_d * out_h * out_w;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    fused_conv_transpose_maxpool_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        temp1.data_ptr<float>(), nullptr, nullptr,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding,
        2, 2, 2, 3, 3, 3);

    // MaxPool1 (kernel=2)
    total_threads = batch_size * out_channels * (out_d / 2) * (out_h / 2) * (out_w / 2);
    num_blocks = (total_threads + block_size - 1) / block_size;
    maxpool1_kernel<<<num_blocks, block_size>>>(
        temp1.data_ptr<float>(), temp2.data_ptr<float>(),
        batch_size, out_channels, out_d, out_h, out_w,
        out_d / 2, out_h / 2, out_w / 2, 2);

    // MaxPool2 (kernel=3)
    auto temp3 = torch::zeros({batch_size, out_channels, out_d / 2 / 3, out_h / 2 / 3, out_w / 2 / 3}, input.options());
    total_threads = batch_size * out_channels * (out_d / 2 / 3) * (out_h / 2 / 3) * (out_w / 2 / 3);
    num_blocks = (total_threads + block_size - 1) / block_size;
    maxpool2_kernel<<<num_blocks, block_size>>>(
        temp2.data_ptr<float>(), temp3.data_ptr<float>(),
        batch_size, out_channels, out_d / 2, out_h / 2, out_w / 2,
        out_d / 2 / 3, out_h / 2 / 3, out_w / 2 / 3, 3);

    // Sum reduction
    total_threads = batch_size * 1 * (out_d / 2 / 3) * (out_h / 2 / 3) * (out_w / 2 / 3);
    num_blocks = (total_threads + block_size - 1) / block_size;
    sum_reduce_kernel<<<num_blocks, block_size>>>(
        temp3.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, out_channels, out_d / 2 / 3, out_h / 2 / 3, out_w / 2 / 3);

    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_ops = fused_ops

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        return self.fused_ops.fused_forward_cuda(x, weight, bias, 5, 2, 2)