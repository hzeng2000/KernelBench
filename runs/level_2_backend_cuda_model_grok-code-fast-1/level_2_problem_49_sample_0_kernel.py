import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused softmax (dim=1) and sigmoid
fused_softmax_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_softmax_sigmoid_kernel(const float* input, float* output, int batch, int channels, int depth, int height, int width) {
    int spatial_size = depth * height * width;
    int total_spatial = batch * spatial_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_spatial) return;

    int b = idx / spatial_size;
    int spatial_idx = idx % spatial_size;
    int d = spatial_idx / (height * width);
    int h = (spatial_idx % (height * width)) / width;
    int w = spatial_idx % width;

    // Compute max over channels for numerical stability
    float max_val = -INFINITY;
    for (int c = 0; c < channels; ++c) {
        int input_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        max_val = fmaxf(max_val, input[input_idx]);
    }

    // Compute sum of exp
    float sum_exp = 0.0f;
    for (int c = 0; c < channels; ++c) {
        int input_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        float val = input[input_idx];
        sum_exp += expf(val - max_val);
    }

    // Compute softmax and apply sigmoid
    for (int c = 0; c < channels; ++c) {
        int input_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        int output_idx = input_idx;
        float val = input[input_idx];
        float softmax_val = expf(val - max_val) / sum_exp;
        float sigmoid_val = 1.0f / (1.0f + expf(-softmax_val));
        output[output_idx] = sigmoid_val;
    }
}

torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto output = torch::zeros_like(input);

    int spatial_size = depth * height * width;
    int total_threads = batch * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    fused_softmax_sigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch, channels, depth, height, width);

    return output;
}
"""

fused_softmax_sigmoid_cpp_source = (
    "torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for fused softmax and sigmoid
fused_softmax_sigmoid = load_inline(
    name="fused_softmax_sigmoid",
    cpp_sources=fused_softmax_sigmoid_cpp_source,
    cuda_sources=fused_softmax_sigmoid_source,
    functions=["fused_softmax_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, applies fused Softmax (dim=1) and Sigmoid using a custom CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.fused_softmax_sigmoid = fused_softmax_sigmoid

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.fused_softmax_sigmoid.fused_softmax_sigmoid_cuda(x)
        return x