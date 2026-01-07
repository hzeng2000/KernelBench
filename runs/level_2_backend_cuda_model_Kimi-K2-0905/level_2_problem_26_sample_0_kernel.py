import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3D + Add + HardSwish fusion
conv_transpose_add_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 8

__global__ void conv_transpose_add_hardswish_kernel(
    const float* input, const float* weight, const float* bias,
    const float* add_input, float* output,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_x >= out_w || out_y >= out_h || out_z >= out_d) return;

    int b = blockIdx.w;
    int out_c = threadIdx.w;

    float sum = 0.0f;

    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int kz = 0; kz < kernel_d; ++kz) {
            for (int ky = 0; ky < kernel_h; ++ky) {
                for (int kx = 0; kx < kernel_w; ++kx) {
                    int in_z = (out_z + pad_d - kz * stride_d) / stride_d;
                    int in_y = (out_y + pad_h - ky * stride_h) / stride_h;
                    int in_x = (out_x + pad_w - kx * stride_w) / stride_w;

                    if (in_z >= 0 && in_z < in_d && in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                        int input_idx = ((b * in_channels + in_c) * in_d + in_z) * in_h * in_w + in_y * in_w + in_x;
                        int weight_idx = ((out_c * in_channels + in_c) * kernel_d + kz) * kernel_h * kernel_w + ky * kernel_w + kx;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    int out_idx = ((b * out_channels + out_c) * out_d + out_z) * out_h * out_w + out_y * out_w + out_x;
    float val = sum + bias[out_c];
    val += add_input[out_idx];

    // HardSwish: x * relu6(x + 3) / 6
    float relu6_val = fminf(fmaxf(val + 3.0f, 0.0f), 6.0f);
    output[out_idx] = val * relu6_val * (1.0f / 6.0f);
}
"""

conv_transpose_add_hardswish_cpp_source = """
torch::Tensor conv_transpose_add_hardswish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor add_input,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w);
"""

conv_transpose_add_hardswish = load_inline(
    name="conv_transpose_add_hardswish",
    cpp_sources=conv_transpose_add_hardswish_cpp_source,
    cuda_sources=conv_transpose_add_hardswish_source,
    functions=["conv_transpose_add_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride_d, self.stride_h, self.stride_w = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.pad_d, self.pad_h, self.pad_w = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.out_pad_d, self.out_pad_h, self.out_pad_w = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.fused_op = conv_transpose_add_hardswish

    def forward(self, x, add_input):
        batch_size = x.size(0)
        in_channels = x.size(1)
        in_d, in_h, in_w = x.size(2), x.size(3), x.size(4)
        out_channels = self.conv_transpose.out_channels
        out_d = (in_d - 1) * self.stride_d - 2 * self.pad_d + self.kernel_d + self.out_pad_d
        out_h = (in_h - 1) * self.stride_h - 2 * self.pad_h + self.kernel_h + self.out_pad_h
        out_w = (in_w - 1) * self.stride_w - 2 * self.pad_w + self.kernel_w + self.out_pad_w

        output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

        kernel_d, kernel_h, kernel_w = self.conv_transpose.kernel_size
        self.kernel_d, self.kernel_h, self.kernel_w = kernel_d, kernel_h, kernel_w

        return self.fused_op.conv_transpose_add_hardswish_cuda(
            x, self.conv_transpose.weight, self.bias.squeeze(), add_input,
            self.stride_d, self.stride_h, self.stride_w,
            self.pad_d, self.pad_h, self.pad_w,
            self.out_pad_d, self.out_pad_h, self.out_pad_w
        )