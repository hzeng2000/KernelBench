import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused 4x4x4 average pooling
fused_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_4x4x4_kernel(const float* input, float* output, int N, int C, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D_out * H_out * W_out;
    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c = (idx / (W_out * H_out * D_out)) % C;
    int n = idx / (W_out * H_out * D_out * C);

    float sum = 0.0f;
    for (int kd = 0; kd < 4; ++kd) {
        for (int kh = 0; kh < 4; ++kh) {
            for (int kw = 0; kw < 4; ++kw) {
                int d_in = d_out * 4 + kd;
                int h_in = h_out * 4 + kh;
                int w_in = w_out * 4 + kw;
                int input_idx = ((n * C + c) * D_in + d_in) * H_in * W_in + h_in * W_in + w_in;
                sum += input[input_idx];
            }
        }
    }
    output[idx] = sum / 64.0f;
}

torch::Tensor fused_avg_pool_cuda(torch::Tensor input) {
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    int D_out = D_in / 4;
    int H_out = H_in / 4;
    int W_out = W_in / 4;
    auto output = torch::zeros({N, C, D_out, H_out, W_out}, input.options());

    int total_elements = N * C * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    avg_pool_4x4x4_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, C, D_in, H_in, W_in, D_out, H_out, W_out);

    return output;
}
"""

fused_avg_pool_cpp_source = (
    "torch::Tensor fused_avg_pool_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for fused average pooling
fused_avg_pool = load_inline(
    name="fused_avg_pool",
    cpp_sources=fused_avg_pool_cpp_source,
    cuda_sources=fused_avg_pool_source,
    functions=["fused_avg_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization, 
    and a fused average pooling layer that combines two 2x2x2 average pools into one 4x4x4.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_avg_pool = fused_avg_pool

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.fused_avg_pool.fused_avg_pool_cuda(x)
        return x