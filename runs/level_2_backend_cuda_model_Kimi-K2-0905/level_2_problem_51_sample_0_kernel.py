import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void fused_gemm_subtract_kernel(
    const float* input, const float* weight, const float* bias, const float* subtract_vec,
    float* output, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        if (bias != nullptr) {
            sum += bias[col];
        }
        sum -= subtract_vec[col];
        output[row * out_features + col] = sum;
    }
}

__global__ void global_avg_pool_kernel(
    const float* input, float* output, int batch_size, int out_features) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < out_features; i++) {
            sum += input[row * out_features + i];
        }
        output[row] = sum / out_features;
    }
}

__global__ void logsumexp_kernel(
    const float* input, float* output, int batch_size) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size) {
        float max_val = input[row];
        float sum = 0.0f;
        sum += expf(input[row] - max_val);
        output[row] = logf(sum) + max_val;
    }
}

__global__ void gelu_kernel(
    const float* input, float* output, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float tanh_input = 0.7978845608f * (x + 0.044715f * x * x * x);
        float tanh_val = tanhf(tanh_input);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

__global__ void residual_add_kernel(
    const float* residual, float* output, int batch_size, int in_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * in_features) {
        output[idx] = output[idx] + residual[idx];
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract_vec) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 blockDim(16, 16);
    dim3 gridDim((out_features + blockDim.x - 1) / blockDim.x,
                   (batch_size + blockDim.y - 1) / blockDim.y);
    
    fused_gemm_subtract_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        subtract_vec.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_features, out_features);
    
    auto pooled = torch::zeros({batch_size}, input.options());
    global_avg_pool_kernel<<<(batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        output.data_ptr<float>(), pooled.data_ptr<float>(), batch_size, out_features);
    
    auto logsumexp_out = torch::zeros({batch_size}, input.options());
    logsumexp_kernel<<<(batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        pooled.data_ptr<float>(), logsumexp_out.data_ptr<float>(), batch_size);
    
    auto gelu_out = torch::zeros({batch_size}, input.options());
    gelu_kernel<<<(batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        logsumexp_out.data_ptr<float>(), gelu_out.data_ptr<float>(), batch_size);
    
    auto final_out = torch::zeros({batch_size, in_features}, input.options());
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_features; j++) {
            final_out[i][j] = gelu_out[i].item<float>();
        }
    }
    
    residual_add_kernel<<<(batch_size * in_features + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        input.data_ptr<float>(), final_out.data_ptr<float>(), batch_size, in_features);
    
    return final_out;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract_vec);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_forward_cuda(x, self.gemm.weight, self.gemm.bias, self.subtract)

batch_size = 2048
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]