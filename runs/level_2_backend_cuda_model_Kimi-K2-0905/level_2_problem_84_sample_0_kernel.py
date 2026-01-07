import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GEMM + BN + Scale + Softmax
fused_gemm_bn_scale_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void fused_gemm_bn_scale_softmax_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias, const float* bn_mean, const float* bn_var,
    const float scale, float* output,
    int batch_size, int in_features, int out_features, float bn_eps) {

    extern __shared__ float shared_mem[];
    float* max_vals = shared_mem;
    float* sum_vals = &shared_mem[blockDim.x];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row < batch_size) {
        // GEMM for this row
        float gemm_result = 0.0f;
        for (int i = tid; i < in_features; i += blockDim.x) {
            gemm_result += input[row * in_features + i] * weight[i];
        }
        
        // Reduce within block
        __shared__ float block_sum;
        if (tid == 0) block_sum = 0.0f;
        __syncthreads();
        
        atomicAdd(&block_sum, gemm_result);
        __syncthreads();
        
        if (tid == 0) {
            // Add bias
            block_sum += bias[blockIdx.y];
            
            // BatchNorm
            float bn_val = (block_sum - bn_mean[blockIdx.y]) / sqrtf(bn_var[blockIdx.y] + bn_eps);
            block_sum = bn_val * bn_weight[blockIdx.y] + bn_bias[blockIdx.y];
            
            // Scale
            block_sum *= scale;
            
            // Store for softmax
            output[row * out_features + blockIdx.y] = block_sum;
        }
    }
    
    __syncthreads();
    
    // Softmax - compute max
    if (row < batch_size) {
        float max_val = -CUDART_INF_F;
        for (int i = tid; i < out_features; i += blockDim.x) {
            max_val = fmaxf(max_val, output[row * out_features + i]);
        }
        max_vals[tid] = max_val;
        __syncthreads();
        
        // Reduce max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + stride]);
            }
            __syncthreads();
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        float max_val_reduced = max_vals[0];
        for (int i = tid; i < out_features; i += blockDim.x) {
            float exp_val = expf(output[row * out_features + i] - max_val_reduced);
            output[row * out_features + i] = exp_val;
            sum += exp_val;
        }
        sum_vals[tid] = sum;
        __syncthreads();
        
        // Reduce sum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sum_vals[tid] += sum_vals[tid + stride];
            }
            __syncthreads();
        }
        
        // Normalize
        float sum_reduced = sum_vals[0];
        for (int i = tid; i < out_features; i += blockDim.x) {
            output[row * out_features + i] /= sum_reduced;
        }
    }
}

torch::Tensor fused_gemm_bn_scale_softmax_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,
    torch::Tensor scale) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    const int block_size = 256;
    const int shared_mem_size = 2 * block_size * sizeof(float);
    
    dim3 grid(batch_size, out_features);
    
    fused_gemm_bn_scale_softmax_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(), bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        scale.item<float>(), output.data_ptr<float>(),
        batch_size, in_features, out_features, 1e-5f);
    
    return output;
}
"""

fused_gemm_bn_scale_softmax_cpp_source = (
    "torch::Tensor fused_gemm_bn_scale_softmax_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,"
    "torch::Tensor scale);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_gemm_bn_scale_softmax_cpp_source,
    cuda_sources=fused_gemm_bn_scale_softmax_source,
    functions=["fused_gemm_bn_scale_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        # Extract parameters
        weight = self.gemm.weight
        bias = self.gemm.bias
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        
        # Call fused kernel
        return self.fused_ops.fused_gemm_bn_scale_softmax_cuda(
            x, weight, bias, bn_weight, bn_bias, bn_mean, bn_var, self.scale
        )