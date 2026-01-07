import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for transposed convolution + global average pooling + bias + logsumexp + sum + multiply
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_transpose_conv_gap_bias_lse_sum_mul_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int out_h, int out_w, int kernel_size, float multiplier) {

    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;

    int tid = threadIdx.x;
    int batch = blockIdx.z;
    int out_c = blockIdx.y;

    int out_h_idx = blockIdx.x / out_w;
    int out_w_idx = blockIdx.x % out_w;

    if (out_h_idx >= out_h || out_w_idx >= out_w) return;

    float sum = 0.0f;
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_h_idx = out_h_idx + kh;
                int in_w_idx = out_w_idx + kw;
                if (in_h_idx < in_h && in_w_idx < in_w) {
                    int in_idx = ((batch * in_channels + in_c) * in_h + in_h_idx) * in_w + in_w_idx;
                    int weight_idx = ((out_c * in_channels + in_c) * kernel_size + kh) * kernel_size + kw;
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Global average pooling: average over spatial dims
    float gap_val = sum / (out_h * out_w);

    // Add bias
    float biased = gap_val + bias[out_c];

    // Store in shared memory for reduction
    shared_data[tid] = biased;

    __syncthreads();

    // Compute logsumexp across channels (dim=1)
    // Each block handles one batch and one out_c
    // We reduce across threads in block (assumes blockDim.x == out_channels)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < out_channels) {
            float a = shared_data[tid];
            float b = shared_data[tid + stride];
            float max_val = fmaxf(a, b);
            shared_data[tid] = max_val + logf(expf(a - max_val) + expf(b - max_val));
        }
        __syncthreads();
    }

    if (tid == 0) {
        float lse = shared_data[0];
        // Sum across spatial dims (already 1x1)
        float summed = lse;
        // Multiply
        output[batch] = summed * multiplier;
    }
}

torch::Tensor fused_transpose_conv_gap_bias_lse_sum_mul_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, float multiplier) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(0);
    int out_h = in_h;
    int out_w = in_w;

    auto output = torch::zeros({batch_size}, input.options());

    const int threads = out_channels;
    const int blocks_x = out_h * out_w;
    const int blocks_y = out_channels;
    const int blocks_z = batch_size;

    size_t shared_size = threads * sizeof(float);

    fused_transpose_conv_gap_bias_lse_sum_mul_kernel<<<dim3(blocks_x, blocks_y, blocks_z), threads, shared_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, in_h, in_w,
        out_channels, out_h, out_w, kernel_size, multiplier);

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_transpose_conv_gap_bias_lse_sum_mul_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, float multiplier);"
)

fused_op = load_inline(
    name="fused_transpose_conv",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_transpose_conv_gap_bias_lse_sum_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op
        self.kernel_size = kernel_size

    def forward(self, x):
        weight = self.conv_transpose.weight
        return self.fused_op.fused_transpose_conv_gap_bias_lse_sum_mul_cuda(
            x, weight, self.bias.squeeze(), self.kernel_size, 10.0
        )

batch_size = 16
in_channels = 64
out_channels = 128
height = width = 512
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]