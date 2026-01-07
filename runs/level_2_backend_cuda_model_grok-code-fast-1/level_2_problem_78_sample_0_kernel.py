import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for maxpool3d with kernel_size=3, stride=3, padding=0 followed by sum over dim=1
maxpool_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void maxpool_sum_kernel(const float* input, float* output, int batch, int channels, int d1, int h1, int w1, int d2, int h2, int w2) {
    int idx = blockIdx.x;
    int b = idx / (d2 * h2 * w2);
    int od = (idx / (h2 * w2)) % d2;
    int oh = (idx / w2) % h2;
    int ow = idx % w2;
    int c = threadIdx.x;

    int id_start = od * 3;
    int ih_start = oh * 3;
    int iw_start = ow * 3;

    float max_val = -std::numeric_limits<float>::infinity();
    for (int dd = 0; dd < 3; ++dd) {
        for (int dh = 0; dh < 3; ++dh) {
            for (int dw = 0; dw < 3; ++dw) {
                int id = id_start + dd;
                int ih = ih_start + dh;
                int iw = iw_start + dw;
                if (id < d1 && ih < h1 && iw < w1) {
                    int input_idx = ((b * channels + c) * d1 + id) * h1 * w1 + ih * w1 + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }

    __shared__ float s_max[64];
    s_max[threadIdx.x] = max_val;
    __syncthreads();

    for (int s = 32; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_max[threadIdx.x] += s_max[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int output_idx = (b * d2 + od) * h2 * w2 + oh * w2 + ow;
        output[output_idx] = s_max[0];
    }
}

torch::Tensor maxpool_sum_cuda(torch::Tensor input) {
    int batch = input.size(0);
    int channels = input.size(1);
    int d1 = input.size(2);
    int h1 = input.size(3);
    int w1 = input.size(4);
    int d2 = (d1 - 3) / 3 + 1;
    int h2 = (h1 - 3) / 3 + 1;
    int w2 = (w1 - 3) / 3 + 1;
    auto output = torch::zeros({batch, 1, d2, h2, w2}, input.options());

    int num_blocks = batch * d2 * h2 * w2;
    const int block_size = channels;

    maxpool_sum_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch, channels, d1, h1, w1, d2, h2, w2);

    return output;
}
"""

maxpool_sum_cpp_source = (
    "torch::Tensor maxpool_sum_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for maxpool and sum
maxpool_sum = load_inline(
    name="maxpool_sum",
    cpp_sources=maxpool_sum_cpp_source,
    cuda_sources=maxpool_sum_source,
    functions=["maxpool_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by one max pooling layer and a fused max pooling + sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.maxpool_sum = maxpool_sum

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.maxpool_sum.maxpool_sum_cuda(x)
        return x