import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BN inference + post operations
fused_bn_post_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bn_post_kernel(const float* x, const float* running_mean, const float* running_var, const float* weight, const float* bias_bn, float eps, const float* bias, float divide_value, float* out, int batch, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * features;
    if (idx < total) {
        int f = idx % features;
        float val = x[idx];
        float mean = running_mean[f];
        float var = running_var[f];
        float gamma = weight[f];
        float beta = bias_bn[f];
        val = (val - mean) / sqrtf(var + eps) * gamma + beta;
        val += bias[0];
        val /= divide_value;
        float sig = 1.0f / (1.0f + expf(-val));
        out[idx] = val * sig;
    }
}

torch::Tensor fused_bn_post_cuda(torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias_bn, float eps, torch::Tensor bias, float divide_value) {
    auto batch = x.size(0);
    auto features = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (x.numel() + block_size - 1) / block_size;

    fused_bn_post_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(), weight.data_ptr<float>(), bias_bn.data_ptr<float>(), eps, bias.data_ptr<float>(), divide_value, out.data_ptr<float>(), batch, features);

    return out;
}
"""

fused_bn_post_cpp_source = (
    "torch::Tensor fused_bn_post_cuda(torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias_bn, float eps, torch::Tensor bias, float divide_value);"
)

# Compile the inline CUDA code for fused BN + post
fused_bn_post = load_inline(
    name="fused_bn_post",
    cpp_sources=fused_bn_post_cpp_source,
    cuda_sources=fused_bn_post_source,
    functions=["fused_bn_post_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs matrix multiplication, fused batch normalization inference + bias addition + division + Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.fused_bn_post = fused_bn_post

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_bn_post.fused_bn_post_cuda(x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, self.bn.eps, self.bias, self.divide_value)
        return x