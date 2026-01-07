import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused batch norm, scale, and softmax
bn_scale_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void bn_scale_softmax_kernel(const float* input, float* output, const float* running_mean, const float* running_var, const float* weight, const float* bias, float scale, float eps, int dim) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int row_start = bid * dim;
    extern __shared__ float s_data[];

    // First, apply bn and scale, find max
    float local_max = -INFINITY;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = input[row_start + i];
        val = (val - running_mean[i]) / sqrtf(running_var[i] + eps) * weight[i] + bias[i];
        val = val * scale;
        output[row_start + i] = val;  // temporarily store bn_scaled
        local_max = max(local_max, val);
    }
    s_data[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = max(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    float row_max = s_data[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = expf(output[row_start + i] - row_max);
        output[row_start + i] = val;
        local_sum += val;
    }
    s_data[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = s_data[tid] + s_data[tid + s];
        }
        __syncthreads();
    }
    float row_sum = s_data[0];

    // Divide
    for (int i = tid; i < dim; i += blockDim.x) {
        output[row_start + i] /= row_sum;
    }
}

torch::Tensor bn_scale_softmax_cuda(torch::Tensor input, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, float scale, float eps) {
    int batch_size = input.size(0);
    int dim = input.size(1);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int shared_size = block_size * sizeof(float);
    const int num_blocks = batch_size;

    bn_scale_softmax_kernel<<<num_blocks, block_size, shared_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        scale, eps, dim
    );

    return output;
}
"""

bn_scale_softmax_cpp_source = (
    "torch::Tensor bn_scale_softmax_cuda(torch::Tensor input, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, float scale, float eps);"
)

# Compile the inline CUDA code for fused bn, scale, softmax
bn_scale_softmax = load_inline(
    name="bn_scale_softmax",
    cpp_sources=bn_scale_softmax_cpp_source,
    cuda_sources=bn_scale_softmax_source,
    functions=["bn_scale_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication (Gemm), fused Batch Normalization, scaling, and Softmax using custom CUDA.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.bn_scale_softmax = bn_scale_softmax

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.bn_scale_softmax.bn_scale_softmax_cuda(x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, self.scale.item(), self.bn.eps)
        return x