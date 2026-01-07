import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused sigmoid and sum
sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_sum_kernel(const float* input, float* output, int batch_size, int hidden_size) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = input[batch * hidden_size + i];
        sum += 1.0f / (1.0f + expf(-val));
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch] = sdata[0];
    }
}

torch::Tensor sigmoid_sum_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int hidden_size = input.size(1);
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    const int block_size = 1024;
    const int num_blocks = batch_size;
    int shared_size = block_size * sizeof(float);
    
    sigmoid_sum_kernel<<<num_blocks, block_size, shared_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, hidden_size
    );
    
    return output;
}
"""

sigmoid_sum_cpp_source = (
    "torch::Tensor sigmoid_sum_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for sigmoid and sum
sigmoid_sum = load_inline(
    name="sigmoid_sum",
    cpp_sources=sigmoid_sum_cpp_source,
    cuda_sources=sigmoid_sum_source,
    functions=["sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies sigmoid, and sums the result using a custom fused CUDA kernel for sigmoid and sum.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.sigmoid_sum = sigmoid_sum

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)
        x = self.sigmoid_sum.sigmoid_sum_cuda(x)
        return x