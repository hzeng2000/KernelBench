import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + scale + residual + clamp + logsumexp + mish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp, float* temp2,
    int batch_size, int input_size, int hidden_size,
    float scale_factor, float clamp_min, float clamp_max) {

    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < batch_size && col < hidden_size) {
        // Matmul: compute dot product of input[row] with weight[col]
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[row * input_size + i] * weight[col * input_size + i];
        }
        if (bias != nullptr) {
            sum += bias[col];
        }

        // Scale
        sum *= scale_factor;

        // Residual (x + x)
        sum += sum;

        // Clamp
        sum = fmaxf(fminf(sum, clamp_max), clamp_min);

        // Store in temp for logsumexp
        temp[row * hidden_size + col] = sum;
    }

    __syncthreads();

    // Compute logsumexp per row
    if (col == 0 && row < batch_size) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < hidden_size; i++) {
            max_val = fmaxf(max_val, temp[row * hidden_size + i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            sum_exp += expf(temp[row * hidden_size + i] - max_val);
        }

        float logsumexp_val = logf(sum_exp) + max_val;

        // Compute Mish activation: x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x))
        float softplus = logf(1.0f + expf(logsumexp_val));
        float mish_val = logsumexp_val * tanhf(softplus);

        output[row] = mish_val;
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale_factor, float clamp_min, float clamp_max) {

    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);

    auto temp = torch::zeros({batch_size, hidden_size}, input.options());
    auto output = torch::zeros({batch_size, 1}, input.options());

    const int threads = 256;
    const int blocks = batch_size;

    fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        temp.data_ptr<float>(),
        nullptr,
        batch_size, input_size, hidden_size,
        scale_factor, clamp_min, clamp_max
    );

    return output;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_forward_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "float scale_factor, float clamp_min, float clamp_max);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_forward_cuda(
            x, self.matmul.weight, self.matmul.bias,
            self.scale_factor, self.clamp_min, self.clamp_max
        )