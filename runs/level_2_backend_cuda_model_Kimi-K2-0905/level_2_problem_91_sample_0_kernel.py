import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d + Softmax + Bias + Scale + Sigmoid fusion
fused_conv_transpose_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void conv_transpose_softmax_bias_scale_sigmoid_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp, float* max_vals, float* sum_vals,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, int out_pad_h, int out_pad_w,
    float scaling_factor) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_channels * out_h * out_w;
    if (idx >= total_size) return;

    int n = idx / (out_channels * out_h * out_w);
    int c = (idx / (out_h * out_w)) % out_channels;
    int h = (idx / out_w) % out_h;
    int w = idx % out_w;

    float val = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_h_idx = h + pad_h - kh * stride_h - out_pad_h;
                int in_w_idx = w + pad_w - kw * stride_w - out_pad_w;
                if (in_h_idx % stride_h == 0 && in_w_idx % stride_w == 0) {
                    in_h_idx /= stride_h;
                    in_w_idx /= stride_w;
                    if (in_h_idx >= 0 && in_h_idx < in_h && in_w_idx >= 0 && in_w_idx < in_w) {
                        int in_idx = n * in_channels * in_h * in_w + ic * in_h * in_w + in_h_idx * in_w + in_w_idx;
                        int weight_idx = c * in_channels * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw;
                        val += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    int out_idx = n * out_channels * out_h * out_w + c * out_h * out_w + h * out_w + w;
    temp[out_idx] = val + (bias ? bias[c] : 0.0f);
}

__global__ void softmax_reduce_max_kernel(
    const float* temp, float* max_vals,
    int batch_size, int out_channels, int out_h, int out_w) {

    int n = blockIdx.x;
    int hw = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float shared_max[];
    float max_val = -INFINITY;

    for (int c = tid; c < out_channels; c += blockDim.x) {
        int idx = n * out_channels * out_h * out_w + c * out_h * out_w + hw;
        float v = temp[idx];
        if (v > max_val) max_val = v;
    }
    shared_max[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid])
                shared_max[tid] = shared_max[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_vals[n * out_h * out_w + hw] = shared_max[0];
    }
}

__global__ void softmax_exp_sum_kernel(
    const float* temp, const float* max_vals, float* sum_vals,
    int batch_size, int out_channels, int out_h, int out_w) {

    int n = blockIdx.x;
    int hw = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float shared_sum[];
    float sum_val = 0.0f;

    float max_val = max_vals[n * out_h * out_w + hw];

    for (int c = tid; c < out_channels; c += blockDim.x) {
        int idx = n * out_channels * out_h * out_w + c * out_h * out_w + hw;
        float v = expf(temp[idx] - max_val);
        temp[idx] = v;
        sum_val += v;
    }
    shared_sum[tid] = sum_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum_vals[n * out_h * out_w + hw] = shared_sum[0];
    }
}

__global__ void softmax_div_scale_sigmoid_kernel(
    float* temp, const float* sum_vals, float* output,
    int batch_size, int out_channels, int out_h, int out_w,
    float scaling_factor) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_channels * out_h * out_w;
    if (idx >= total_size) return;

    int n = idx / (out_channels * out_h * out_w);
    int c = (idx / (out_h * out_w)) % out_channels;
    int hw = (idx / out_w) % out_h * out_w + idx % out_w;

    float sum_val = sum_vals[n * out_h * out_w + hw];
    float v = temp[idx] / sum_val;
    v = v * scaling_factor;
    output[idx] = 1.0f / (1.0f + expf(-v));
}

torch::Tensor fused_conv_transpose_softmax_bias_scale_sigmoid_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_h, int stride_w, int pad_h, int pad_w, int out_pad_h, int out_pad_w,
    float scaling_factor) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_h = input.size(2);
    const auto in_w = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    const auto out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    const auto out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    auto temp = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    auto max_vals = torch::zeros({batch_size, out_h, out_w}, input.options());
    auto sum_vals = torch::zeros({batch_size, out_h, out_w}, input.options());

    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_h * out_w + threads - 1) / threads;

    conv_transpose_softmax_bias_scale_sigmoid_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), temp.data_ptr<float>(),
        max_vals.data_ptr<float>(), sum_vals.data_ptr<float>(),
        batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w,
        scaling_factor);

    const int reduce_threads = 256;
    const dim3 reduce_blocks(batch_size, out_h * out_w);
    const int shared_size = reduce_threads * sizeof(float);

    softmax_reduce_max_kernel<<<reduce_blocks, reduce_threads, shared_size>>>(
        temp.data_ptr<float>(), max_vals.data_ptr<float>(),
        batch_size, out_channels, out_h, out_w);

    softmax_exp_sum_kernel<<<reduce_blocks, reduce_threads, shared_size>>>(
        temp.data_ptr<float>(), max_vals.data_ptr<float>(), sum_vals.data_ptr<float>(),
        batch_size, out_channels, out_h, out_w);

    softmax_div_scale_sigmoid_kernel<<<blocks, threads>>>(
        temp.data_ptr<float>(), sum_vals.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, out_channels, out_h, out_w, scaling_factor);

    return output;
}
"""

fused_conv_transpose_activation_cpp_source = """
torch::Tensor fused_conv_transpose_softmax_bias_scale_sigmoid_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_h, int stride_w, int pad_h, int pad_w, int out_pad_h, int out_pad_w,
    float scaling_factor);
"""

fused_conv_transpose_activation = load_inline(
    name="fused_conv_transpose_activation",
    cpp_sources=fused_conv_transpose_activation_cpp_source,
    cuda_sources=fused_conv_transpose_activation_source,
    functions=["fused_conv_transpose_softmax_bias_scale_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor
        self.fused_op = fused_conv_transpose_activation

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_softmax_bias_scale_sigmoid_cuda(
            x, self.weight, self.bias,
            self.stride, self.stride, self.padding, self.padding,
            self.output_padding, self.output_padding, self.scaling_factor)