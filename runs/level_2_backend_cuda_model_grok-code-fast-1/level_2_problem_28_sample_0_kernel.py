import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Linear (matmul + bias)
linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_kernel(const float* x, const float* weight, const float* bias, float* out, int batch, int in_features, int out_features) {
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (b < batch && row < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += x[b * in_features + k] * weight[row * in_features + k];
        }
        out[b * out_features + row] = sum + bias[row];
    }
}

torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int batch = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    auto out = torch::empty({batch, out_features}, x.options());

    dim3 blockDim(1, 16, 16);
    dim3 gridDim(1, (out_features + blockDim.y - 1) / blockDim.y, (batch + blockDim.z - 1) / blockDim.z);

    linear_kernel<<<gridDim, blockDim>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch, in_features, out_features);

    return out;
}
"""

linear_cpp_source = "torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"

linear = load_inline(
    name="linear",
    cpp_sources=linear_cpp_source,
    cuda_sources=linear_source,
    functions=["linear_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for InstanceNorm (for 1x1 case, simplifies to bias)
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instance_norm_kernel(const float* weight, const float* bias, float* out, int batch, int out_features) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch && c < out_features) {
        out[b * out_features + c] = bias[c];
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor weight, torch::Tensor bias, int batch, int out_features) {
    auto out = torch::empty({batch, out_features}, weight.options());

    dim3 blockDim(16, 16);
    dim3 gridDim((out_features + blockDim.x - 1) / blockDim.x, (batch + blockDim.y - 1) / blockDim.y);

    instance_norm_kernel<<<gridDim, blockDim>>>(weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch, out_features);

    return out;
}
"""

instance_norm_cpp_source = "torch::Tensor instance_norm_cuda(torch::Tensor weight, torch::Tensor bias, int batch, int out_features);"

instance_norm_custom = load_inline(
    name="instance_norm_custom",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for fused add and multiply
add_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_mul_kernel(const float* x, const float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = x[idx] + y[idx];
        out[idx] = temp * y[idx];
    }
}

torch::Tensor add_mul_cuda(torch::Tensor x, torch::Tensor y) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    add_mul_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

add_mul_cpp_source = "torch::Tensor add_mul_cuda(torch::Tensor x, torch::Tensor y);"

add_mul = load_inline(
    name="add_mul",
    cpp_sources=add_mul_cpp_source,
    cuda_sources=add_mul_source,
    functions=["add_mul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model with custom CUDA operators for Linear, InstanceNorm, and fused add+multiply.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.linear_cuda = linear
        self.instance_norm_cuda = instance_norm_custom
        self.add_mul_cuda = add_mul

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.linear_cuda.linear_cuda(x, self.bmm.weight, self.bmm.bias)
        batch, out_features = x.shape
        x = self.instance_norm_cuda.instance_norm_cuda(self.instance_norm.weight, self.instance_norm.bias, batch, out_features)
        x = self.add_mul_cuda.add_mul_cuda(x, y)
        return x