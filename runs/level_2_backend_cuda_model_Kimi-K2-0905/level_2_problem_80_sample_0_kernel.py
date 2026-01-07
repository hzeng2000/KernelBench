import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GEMM + Max + SubMean + GELU
fused_gemm_max_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define BLOCK_SIZE 256

__global__ void fused_gemm_max_submean_gelu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        // GEMM computation
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Store intermediate result in shared memory for max reduction
        extern __shared__ float shared_data[];
        shared_data[threadIdx.x] = sum;
        __syncthreads();
        
        // Max reduction within block
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride && col + stride < out_features) {
                shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        
        float max_val = shared_data[0];
        __syncthreads();
        
        // Subtract mean
        shared_data[threadIdx.x] = sum;
        __syncthreads();
        
        // Sum reduction for mean calculation
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride && col + stride < out_features) {
                shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        float mean = shared_data[0] / out_features;
        float x = sum - mean;
        
        // GELU activation
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        
        // Fast tanh approximation
        float tanh_val;
        if (tanh_arg > 5.0f) {
            tanh_val = 1.0f;
        } else if (tanh_arg < -5.0f) {
            tanh_val = -1.0f;
        } else {
            float exp_val = expf(2.0f * tanh_arg);
            tanh_val = (exp_val - 1.0f) / (exp_val + 1.0f);
        }
        
        output[row * out_features + col] = 0.5f * x * (1.0f + tanh_val);
    }
}

torch::Tensor fused_gemm_max_submean_gelu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    
    fused_gemm_max_submean_gelu_kernel<<<grid, block, BLOCK_SIZE * sizeof(float)>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_gemm_max_gelu_cpp_source = """
torch::Tensor fused_gemm_max_submean_gelu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_gemm_max_gelu_cpp_source,
    cuda_sources=fused_gemm_max_gelu_source,
    functions=["fused_gemm_max_submean_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a fused GEMM, max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        return self.fused_ops.fused_gemm_max_submean_gelu_cuda(
            x, self.gemm.weight, self.gemm.bias
        )