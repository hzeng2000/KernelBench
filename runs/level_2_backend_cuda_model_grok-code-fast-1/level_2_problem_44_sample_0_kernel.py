import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scaled global average pooling
scaled_gap_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scaled_gap_kernel(const float* x, float* out, float multiplier, int H, int W, int B, int C) {
    int bc = blockIdx.x;
    int b = bc / C;
    int c = bc % C;
    int numel = H * W;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < numel; i += blockDim.x) {
        int h = i / W;
        int w = i % W;
        sum += x[((b * C + c) * H + h) * W + w] * multiplier;
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[bc] = sdata[0] / numel;
    }
}

torch::Tensor scaled_gap_cuda(torch::Tensor x, float multiplier) {
    auto B = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto out = torch::zeros({B, C, 1, 1}, x.options());
    const int block_size = 256;
    const int num_blocks = B * C;
    scaled_gap_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), multiplier, H, W, B, C);
    return out;
}
"""

scaled_gap_cpp_source = (
    "torch::Tensor scaled_gap_cuda(torch::Tensor x, float multiplier);"
)

# Compile the inline CUDA code for scaled global average pooling
scaled_gap = load_inline(
    name="scaled_gap",
    cpp_sources=scaled_gap_cpp_source,
    cuda_sources=scaled_gap_source,
    functions=["scaled_gap_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then fused scaled global average pooling (replacing multiply and first GAP), 
    and skips the second GAP as it is a no-op on 1x1 tensors.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self.scaled_gap = scaled_gap

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.scaled_gap.scaled_gap_cuda(x, self.multiplier)
        return x