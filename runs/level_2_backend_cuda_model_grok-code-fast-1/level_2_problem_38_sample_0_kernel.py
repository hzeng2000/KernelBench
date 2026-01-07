import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused clamp, spatial softmax, and scale
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_clamp_softmax_scale_kernel(const float* x, float* out, int b, int c, int spatial_size, float clamp_min, float clamp_max, const float* scale) {
    int bc = blockIdx.x;
    int batch = bc / c;
    int ch = bc % c;
    float s = scale[ch];

    __shared__ float shared_max[1024];
    __shared__ float shared_sum[1024];

    // First pass: compute max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int idx = batch * c * spatial_size + ch * spatial_size + i;
        float val = fmaxf(fminf(x[idx], clamp_max), clamp_min);
        local_max = fmaxf(local_max, val);
    }
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    // Reduce max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float global_max = shared_max[0];

    // Second pass: compute sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int idx = batch * c * spatial_size + ch * spatial_size + i;
        float val = fmaxf(fminf(x[idx], clamp_max), clamp_min);
        local_sum += expf(val - global_max);
    }
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float global_sum = shared_sum[0];

    // Third pass: compute output
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int idx = batch * c * spatial_size + ch * spatial_size + i;
        float val = fmaxf(fminf(x[idx], clamp_max), clamp_min);
        out[idx] = expf(val - global_max) / global_sum * s;
    }
}

torch::Tensor fused_clamp_softmax_scale_cuda(torch::Tensor x, double clamp_min, double clamp_max, torch::Tensor scale) {
    auto b = x.size(0);
    auto c = x.size(1);
    auto d = x.size(2);
    auto h = x.size(3);
    auto w = x.size(4);
    auto spatial_size = d * h * w;
    auto out = torch::zeros_like(x);

    const int block_size = 1024;
    const int num_blocks = b * c;

    fused_clamp_softmax_scale_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), b, c, spatial_size, (float)clamp_min, (float)clamp_max, scale.data_ptr<float>());

    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_clamp_softmax_scale_cuda(torch::Tensor x, double clamp_min, double clamp_max, torch::Tensor scale);"
)

# Compile the inline CUDA code for fused operation
fused_op = load_inline(
    name="fused_clamp_softmax_scale",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_clamp_softmax_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs average pooling, 3D transposed convolution, and fused clamp + spatial softmax + scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = self.fused_op.fused_clamp_softmax_scale_cuda(x, self.clamp_min, self.clamp_max, self.scale)
        return x