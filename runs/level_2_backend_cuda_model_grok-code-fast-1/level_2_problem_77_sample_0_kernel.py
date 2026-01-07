import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scale, batch norm, and global average pooling
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_scale_bn_global_avg_kernel(
    const float* x, float* out, 
    float scale_factor, 
    const float* running_mean, const float* running_var, const float* weight, const float* bias, 
    float eps, 
    int batch_size, int out_channels, int depth, int height, int width
) {
    extern __shared__ float s_sum[];
    int tid = threadIdx.x;
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    if (batch >= batch_size || channel >= out_channels) return;

    float mean_val = running_mean[channel];
    float var_val = running_var[channel];
    float gamma = weight[channel];
    float beta = bias[channel];
    float inv_std = 1.0f / sqrtf(var_val + eps);

    int num_elements = depth * height * width;
    int elements_per_thread = (num_elements + blockDim.x - 1) / blockDim.x;
    float local_sum = 0.0f;

    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = tid * elements_per_thread + i;
        if (idx < num_elements) {
            int d = idx / (height * width);
            int h = (idx % (height * width)) / width;
            int w = idx % width;
            int global_idx = ((batch * out_channels + channel) * depth + d) * height * width + h * width + w;
            float val = x[global_idx];
            val = (val * scale_factor - mean_val) * inv_std * gamma + beta;
            local_sum += val;
        }
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[batch * out_channels + channel] = s_sum[0] / num_elements;
    }
}

torch::Tensor fused_scale_bn_global_avg_cuda(
    torch::Tensor x, float scale_factor, 
    torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, 
    float eps
) {
    auto batch_size = x.size(0);
    auto out_channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);
    auto out = torch::zeros({batch_size, out_channels, 1, 1, 1}, x.options());

    dim3 block(256);
    dim3 grid(batch_size, out_channels);
    size_t shared_mem = 256 * sizeof(float);

    fused_scale_bn_global_avg_kernel<<<grid, block, shared_mem>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), 
        scale_factor, 
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        eps, 
        batch_size, out_channels, depth, height, width
    );

    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_scale_bn_global_avg_cuda(torch::Tensor x, float scale_factor, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, float eps);"
)

# Compile the inline CUDA code for the fused operation
fused_op = load_inline(
    name="fused_scale_bn_global_avg",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_scale_bn_global_avg_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, then fuses scaling, batch normalization, 
    and global average pooling into a single custom CUDA kernel for speedup.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_scale_bn_global_avg_cuda(
            x, self.scale_factor, 
            self.batch_norm.running_mean, self.batch_norm.running_var, 
            self.batch_norm.weight, self.batch_norm.bias, 
            self.batch_norm.eps
        )
        return x