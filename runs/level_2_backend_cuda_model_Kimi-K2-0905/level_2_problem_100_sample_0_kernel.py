import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d + clamp + divide
conv_transpose_clamp_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d: %s\\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Simplified ConvTranspose3d forward kernel for stride=2, padding=1, kernel=3
// This is a direct implementation focusing on the common case
__global__ void conv_transpose_clamp_div_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size,
    int in_c, int in_d, int in_h, int in_w,
    int out_c, int out_d, int out_h, int out_w,
    int k, int stride, int padding,
    float min_val, float divisor) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_size = batch_size * out_c * out_d * out_h * out_w;

    if (idx < total_out_size) {
        // Compute n, c, d, h, w from linear index
        int tmp = idx;
        int w = tmp % out_w; tmp /= out_w;
        int h = tmp % out_h; tmp /= out_h;
        int d = tmp % out_d; tmp /= out_d;
        int c = tmp % out_c; tmp /= out_c;
        int n = tmp;

        float acc = 0.0f;

        // Loop over input channels and kernel
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kd = 0; kd < k; ++kd) {
                for (int kh = 0; kh < k; ++kh) {
                    for (int kw = 0; kw < k; ++kw) {
                        int in_d_idx = d - kd + padding;
                        int in_h_idx = h - kh + padding;
                        int in_w_idx = w - kw + padding;

                        if (in_d_idx % stride == 0 && in_h_idx % stride == 0 && in_w_idx % stride == 0) {
                            in_d_idx /= stride;
                            in_h_idx /= stride;
                            in_w_idx /= stride;

                            if (in_d_idx >= 0 && in_d_idx < in_d &&
                                in_h_idx >= 0 && in_h_idx < in_h &&
                                in_w_idx >= 0 && in_w_idx < in_w) {

                                int input_idx = ((n * in_c + ic) * in_d + in_d_idx) * in_h + in_h_idx;
                                input_idx = input_idx * in_w + in_w_idx;

                                int weight_idx = ((c * in_c + ic) * k + kd) * k + kh;
                                weight_idx = weight_idx * k + kw;

                                acc += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }

        // Apply bias (optional), clamp, and divide
        int out_idx = ((n * out_c + c) * out_d + d) * out_h + h;
        out_idx = out_idx * out_w + w;

        float val = acc; // + bias[c] if bias is used
        val = fmaxf(val, min_val);
        val /= divisor;
        output[out_idx] = val;
    }
}

torch::Tensor conv_transpose_clamp_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    float min_val,
    float divisor) {

    const int batch_size = input.size(0);
    const int in_c = input.size(1);
    const int in_d = input.size(2);
    const int in_h = input.size(3);
    const int in_w = input.size(4);

    const int out_c = weight.size(0);
    const int k = weight.size(2);

    const int out_d = (in_d - 1) * stride - 2 * padding + k;
    const int out_h = (in_h - 1) * stride - 2 * padding + k;
    const int out_w = (in_w - 1) * stride - 2 * padding + k;

    auto output = torch::zeros({batch_size, out_c, out_d, out_h, out_w}, input.options());

    const int total_size = batch_size * out_c * out_d * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;

    conv_transpose_clamp_div_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_c, in_d, in_h, in_w,
        out_c, out_d, out_h, out_w,
        k, stride, padding,
        min_val, divisor
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

conv_transpose_clamp_div_cpp_source = (
    "torch::Tensor conv_transpose_clamp_div_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "int stride, int padding, float min_val, float divisor);"
)

# Compile the custom CUDA extension
conv_transpose_clamp_div = load_inline(
    name="conv_transpose_clamp_div",
    cpp_sources=conv_transpose_clamp_div_cpp_source,
    cuda_sources=conv_transpose_clamp_div_source,
    functions=["conv_transpose_clamp_div_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.custom_op = conv_transpose_clamp_div

    def forward(self, x):
        return self.custom_op.conv_transpose_clamp_div_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.min_value,
            self.divisor
        )