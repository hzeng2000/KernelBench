import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + sigmoid + sum
fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void fused_linear_sigmoid_sum_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int input_size, int hidden_size) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < batch_size) {
        extern __shared__ float shared_data[];
        float* partial_sums = shared_data;
        
        // Compute partial sums for this row
        float sum = 0.0f;
        for (int i = tid; i < hidden_size; i += blockDim.x) {
            float acc = 0.0f;
            for (int j = 0; j < input_size; j++) {
                acc += input[row * input_size + j] * weight[i * input_size + j];
            }
            if (bias != nullptr) {
                acc += bias[i];
            }
            // Apply sigmoid
            acc = 1.0f / (1.0f + expf(-acc));
            partial_sums[i] = acc;
            sum += acc;
        }
        
        __syncthreads();
        
        // Reduction to compute sum
        for (int stride = hidden_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride && tid + stride < hidden_size) {
                partial_sums[tid] += partial_sums[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            output[row] = partial_sums[0];
        }
    }
}

torch::Tensor fused_linear_sigmoid_sum_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    const int shared_mem_size = hidden_size * sizeof(float);
    
    fused_linear_sigmoid_sum_kernel<<<batch_size, BLOCK_SIZE, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );
    
    return output;
}
"""

fused_linear_sigmoid_sum_cpp_source = (
    "torch::Tensor fused_linear_sigmoid_sum_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_linear_sigmoid_sum = load_inline(
    name="fused_linear_sigmoid_sum",
    cpp_sources=fused_linear_sigmoid_sum_cpp_source,
    cuda_sources=fused_linear_sigmoid_sum_source,
    functions=["fused_linear_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses linear, sigmoid, and sum operations into a single CUDA kernel.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fused_op = fused_linear_sigmoid_sum

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        return self.fused_op.fused_linear_sigmoid_sum_cuda(
            x, self.linear.weight, self.linear.bias
        )