import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused linear + dropout + softmax
fused_linear_dropout_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

__global__ void fused_linear_dropout_softmax_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* dropout_mask,
    int batch_size, int in_features, int out_features,
    float dropout_p, float scale) {
    
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        // Compute linear output
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Apply dropout
        curandState state;
        curand_init(1234, row * out_features + col, 0, &state);
        float rand_val = curand_uniform(&state);
        float dropout_scale = (rand_val > dropout_p) ? scale : 0.0f;
        dropout_mask[row * out_features + col] = dropout_scale;
        sum *= dropout_scale;
        
        // Store intermediate result for softmax
        output[row * out_features + col] = sum;
    }
}

__global__ void softmax_kernel(float* output, int batch_size, int out_features) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int col = 0; col < out_features; col++) {
        max_val = fmaxf(max_val, output[row * out_features + col]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int col = threadIdx.x; col < out_features; col += blockDim.x) {
        float val = expf(output[row * out_features + col] - max_val);
        output[row * out_features + col] = val;
        sum += val;
    }
    
    // Block-level reduction for sum
    __shared__ float shared_sum[BLOCK_SIZE];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Normalize
    if (threadIdx.x == 0) {
        sum = shared_sum[0];
        for (int col = 0; col < out_features; col++) {
            output[row * out_features + col] /= sum;
        }
    }
}

torch::Tensor fused_linear_dropout_softmax_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float dropout_p) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    auto dropout_mask = torch::zeros({batch_size, out_features}, input.options());
    
    float scale = 1.0f / (1.0f - dropout_p);
    
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    fused_linear_dropout_softmax_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), dropout_mask.data_ptr<float>(),
        batch_size, in_features, out_features, dropout_p, scale);
    
    softmax_kernel<<<grid, block>>>(output.data_ptr<float>(), batch_size, out_features);
    
    return output;
}
"""

fused_linear_dropout_softmax_cpp_source = """
torch::Tensor fused_linear_dropout_softmax_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float dropout_p);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_linear_dropout_softmax",
    cpp_sources=fused_linear_dropout_softmax_cpp_source,
    cuda_sources=fused_linear_dropout_softmax_source,
    functions=["fused_linear_dropout_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.fused_ops = fused_ops
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return self.fused_ops.fused_linear_dropout_softmax_cuda(
            x, self.weight, self.bias, self.dropout_p)


def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, dropout_p]