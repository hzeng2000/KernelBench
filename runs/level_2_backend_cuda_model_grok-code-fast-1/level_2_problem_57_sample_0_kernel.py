import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv2d + relu + hardswish
conv_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_relu_hardswish_kernel(const float* input, const float* weight, float* output, int batch, int in_c, int out_c, int in_h, int in_w, int k, int out_h, int out_w) {
    int b = blockIdx.z / out_c;
    int oc = blockIdx.z % out_c;
    int oh = blockIdx.x;
    int ow = blockIdx.y;
    if (oh < out_h && ow < out_w) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    sum += input[b * in_c * in_h * in_w + ic * in_h * in_w + ih * in_w + iw] * 
                           weight[oc * in_c * k * k + ic * k * k + kh * k + kw];
                }
            }
        }
        // relu
        sum = max(0.0f, sum);
        // hardswish
        float temp = (sum + 3.0f) / 6.0f;
        temp = min(1.0f, max(0.0f, temp));
        sum *= temp;
        output[b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = sum;
    }
}

torch::Tensor conv_relu_hardswish_cuda(torch::Tensor input, torch::Tensor weight) {
    int batch = input.size(0);
    int in_c = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_c = weight.size(0);
    int k = weight.size(2);
    int out_h = in_h - k + 1;
    int out_w = in_w - k + 1;
    auto output = torch::zeros({batch, out_c, out_h, out_w}, input.options());
    dim3 grid(out_h, out_w, batch * out_c);
    conv_relu_hardswish_kernel<<<grid, 1>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch, in_c, out_c, in_h, in_w, k, out_h, out_w);
    return output;
}
"""

conv_relu_hardswish_cpp_source = (
    "torch::Tensor conv_relu_hardswish_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for fused conv + relu + hardswish
conv_relu_hardswish = load_inline(
    name="conv_relu_hardswish",
    cpp_sources=conv_relu_hardswish_cpp_source,
    cuda_sources=conv_relu_hardswish_source,
    functions=["conv_relu_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA operator fusing conv2d, relu, and hardswish.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_relu_hardswish = conv_relu_hardswish

    def forward(self, x):
        return self.conv_relu_hardswish.conv_relu_hardswish_cuda(x, self.weight)