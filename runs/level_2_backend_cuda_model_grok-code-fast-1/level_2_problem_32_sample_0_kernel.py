import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scale and min reduction
fused_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_scale_min_kernel(const float* x, float* out, float scale, int batch, int channels, int height, int width) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];
    
    int idx = ((b * channels + tid) * height + h) * width + w;
    if (tid < channels) {
        sdata[tid] = x[idx] * scale;
    } else {
        sdata[tid] = FLT_MAX;
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        int out_idx = ((b * 1 + 0) * height + h) * width + w;
        out[out_idx] = sdata[0];
    }
}

torch::Tensor fused_scale_min_cuda(torch::Tensor x, float scale_factor) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out = torch::empty({batch, 1, height, width}, x.options());
    
    dim3 grid(batch, height, width);
    dim3 block(channels);
    int shared_mem = channels * sizeof(float);
    
    fused_scale_min_kernel<<<grid, block, shared_mem>>>(x.data_ptr<float>(), out.data_ptr<float>(), scale_factor, batch, channels, height, width);
    
    return out;
}
"""

fused_scale_min_cpp_source = (
    "torch::Tensor fused_scale_min_cuda(torch::Tensor x, float scale_factor);"
)

# Compile the inline CUDA code for fused scale and min
fused_scale_min = load_inline(
    name="fused_scale_min",
    cpp_sources=fused_scale_min_cpp_source,
    cuda_sources=fused_scale_min_source,
    functions=["fused_scale_min_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.fused_scale_min = fused_scale_min

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = self.fused_scale_min.fused_scale_min_cuda(x, self.scale_factor)
        return x