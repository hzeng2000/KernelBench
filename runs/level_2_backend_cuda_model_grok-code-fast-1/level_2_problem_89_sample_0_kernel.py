import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused softmax, subtract, swish, and max
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_softmax_subtract_swish_max_kernel(const float* x, const float* subtract, float* out, int batch, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = depth * height * width;
    int total_elements = batch * total_spatial;
    if (idx >= total_elements) return;
    
    int b = idx / total_spatial;
    int spatial_idx = idx % total_spatial;
    int d = spatial_idx / (height * width);
    int h = (spatial_idx % (height * width)) / width;
    int w = spatial_idx % width;
    
    int x_offset_base = b * channels * depth * height * width + d * height * width + h * width + w;
    int out_offset = b * depth * height * width + d * height * width + h * width + w;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int c = 0; c < channels; c++) {
        float val = x[x_offset_base + c * depth * height * width];
        if (val > max_val) max_val = val;
    }
    
    // Compute sum of exp
    float sum_exp = 0.0f;
    for (int c = 0; c < channels; c++) {
        float val = x[x_offset_base + c * depth * height * width];
        sum_exp += expf(val - max_val);
    }
    
    // Compute max of swish(softmax - subtract)
    float max_result = -INFINITY;
    for (int c = 0; c < channels; c++) {
        float val = x[x_offset_base + c * depth * height * width];
        float softmax_val = expf(val - max_val) / sum_exp;
        float y = softmax_val - subtract[c];
        float sig = 1.0f / (1.0f + expf(-y));
        float swish_val = y * sig;
        if (swish_val > max_result) max_result = swish_val;
    }
    
    out[out_offset] = max_result;
}

torch::Tensor fused_softmax_subtract_swish_max_cuda(torch::Tensor x, torch::Tensor subtract) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);
    auto out = torch::zeros({batch, 1, depth, height, width}, x.options());
    
    int total_threads = batch * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;
    
    fused_softmax_subtract_swish_max_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), subtract.data_ptr<float>(), out.data_ptr<float>(), batch, channels, depth, height, width);
    
    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_softmax_subtract_swish_max_cuda(torch::Tensor x, torch::Tensor subtract);"
)

# Compile the inline CUDA code for the fused operation
fused_op = load_inline(
    name="fused_softmax_subtract_swish_max",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_softmax_subtract_swish_max_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that performs a sequence of operations:
        - ConvTranspose3d
        - MaxPool3d
        - Fused Softmax, Subtract, Swish, Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))  # Assuming subtraction is element-wise across channels
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = self.fused_op.fused_softmax_subtract_swish_max_cuda(x, self.subtract)
        return x