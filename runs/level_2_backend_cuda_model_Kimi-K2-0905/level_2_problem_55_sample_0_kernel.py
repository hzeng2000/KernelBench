import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused linear + maxpool + sum + scale
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK_SIZE 256

__global__ void fused_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    int kernel_size, float scale_factor) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / out_features;
    int col = tid % out_features;

    if (row < batch_size && col < out_features) {
        // Linear layer: matmul + bias
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        if (bias != nullptr) {
            sum += bias[col];
        }

        // MaxPool1d (kernel_size=2, stride=2)
        // Since out_features is even and kernel_size=2, we pool adjacent pairs
        if (col % 2 == 0 && col + 1 < out_features) {
            float max_val = fmaxf(sum, __ldg(&output[row * out_features + col + 1]));
            output[row * out_features + col / 2] = max_val;
        }

        // Store intermediate result for maxpool
        __syncthreads();
    }
}

__global__ void reduce_scale_kernel(
    const float* pooled, float* final_output, int batch_size, int reduced_features,
    float scale_factor) {

    int row = blockIdx.x;
    if (row < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < reduced_features; i++) {
            sum += pooled[row * reduced_features + i];
        }
        final_output[row] = sum * scale_factor;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, float scale_factor) {

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    const int reduced_features = out_features / kernel_size;

    auto pooled = torch::zeros({batch_size, reduced_features}, input.options());
    auto output = torch::zeros({batch_size}, input.options());

    const int total_threads = batch_size * out_features;
    const int num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fused_kernel<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        pooled.data_ptr<float>(), batch_size, in_features, out_features,
        kernel_size, scale_factor);

    reduce_scale_kernel<<<batch_size, 1>>>(
        pooled.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, reduced_features, scale_factor);

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, float scale_factor);"
)

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(x, self.weight, self.bias, self.kernel_size, self.scale_factor)