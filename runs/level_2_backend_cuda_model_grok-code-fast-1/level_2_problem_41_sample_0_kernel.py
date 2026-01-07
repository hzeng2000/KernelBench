import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm + GELU + ReLU
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__global__ void fused_batchnorm_gelu_relu_kernel(
    const float* x, const float* weight, const float* bias, 
    float* running_mean, float* running_var, 
    float* out, int batch_size, int out_features, float momentum, float eps, bool training
) {
    int feature = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    extern __shared__ float sdata[];
    float* ssum = sdata;
    float* ssumsq = sdata + block_size;
    
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int i = tid; i < batch_size; i += block_size) {
        float val = x[feature * batch_size + i];
        local_sum += val;
        local_sumsq += val * val;
    }
    ssum[tid] = local_sum;
    ssumsq[tid] = local_sumsq;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            ssumsq[tid] += ssumsq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float mean = ssum[0] / batch_size;
        float mean_sq = ssumsq[0] / batch_size;
        float var = mean_sq - mean * mean;
        float inv_std = rsqrtf(var + eps);
        ssum[0] = mean;
        ssum[1] = inv_std;
        if (training) {
            running_mean[feature] = momentum * running_mean[feature] + (1 - momentum) * mean;
            running_var[feature] = momentum * running_var[feature] + (1 - momentum) * var;
        }
    }
    __syncthreads();
    
    float mean = ssum[0];
    float inv_std = ssum[1];
    float gamma = weight[feature];
    float beta = bias[feature];
    
    for (int i = tid; i < batch_size; i += block_size) {
        float val = x[feature * batch_size + i];
        float x_hat = (val - mean) * inv_std;
        float normed = x_hat * gamma + beta;
        float gelued = gelu(normed);
        out[feature * batch_size + i] = fmaxf(gelued, 0.0f);
    }
}

torch::Tensor fused_batchnorm_gelu_relu_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    float momentum, float eps, bool training
) {
    auto batch_size = x.size(0);
    auto out_features = x.size(1);
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = out_features;
    size_t shared_size = 2 * block_size * sizeof(float);
    
    fused_batchnorm_gelu_relu_kernel<<<num_blocks, block_size, shared_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        out.data_ptr<float>(), batch_size, out_features, momentum, eps, training
    );
    
    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_batchnorm_gelu_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, float momentum, float eps, bool training);"
)

# Compile the inline CUDA code for fused BatchNorm + GELU + ReLU
fused_op = load_inline(
    name="fused_batchnorm_gelu_relu",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_batchnorm_gelu_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs GEMM, then fused BatchNorm + GELU + ReLU in a single custom CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.fused_op.fused_batchnorm_gelu_relu_cuda(
            x, self.batch_norm.weight, self.batch_norm.bias,
            self.batch_norm.running_mean, self.batch_norm.running_var,
            self.batch_norm.momentum, self.batch_norm.eps, self.training
        )
        return x