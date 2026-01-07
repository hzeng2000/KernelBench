import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused softmax, bias add, scaling, and sigmoid
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_softmax_bias_scale_sigmoid_kernel(const float* input, const float* bias, float scale, float* output, int batch, int channels, int height, int width) {
    int block_id = blockIdx.x;
    int b = block_id / (height * width);
    int hw = block_id % (height * width);
    int h = hw / width;
    int w = hw % width;
    int c = threadIdx.x;
    
    if (c >= channels) return;
    
    int idx = ((b * channels + c) * height + h) * width + w;
    float val = input[idx];
    
    // Shared memory for max and sum reduction
    extern __shared__ float shared[];
    float* smax = shared;
    float* ssum = shared + channels;
    
    // Load to shared for max
    smax[c] = val;
    __syncthreads();
    
    // Reduction for max
    for (int stride = channels / 2; stride > 0; stride /= 2) {
        if (c < stride) {
            smax[c] = max(smax[c], smax[c + stride]);
        }
        __syncthreads();
    }
    float max_val = smax[0];
    
    // Compute exp(val - max)
    float exp_val = expf(val - max_val);
    ssum[c] = exp_val;
    __syncthreads();
    
    // Reduction for sum
    for (int stride = channels / 2; stride > 0; stride /= 2) {
        if (c < stride) {
            ssum[c] += ssum[c + stride];
        }
        __syncthreads();
    }
    float sum_exp = ssum[0];
    
    // Softmax
    float softmax_val = exp_val / sum_exp;
    
    // Add bias (bias is (channels, 1, 1), so bias[c] is bias[c][0][0])
    softmax_val += bias[c];
    
    // Scale
    softmax_val *= scale;
    
    // Sigmoid
    softmax_val = 1.0f / (1.0f + expf(-softmax_val));
    
    // Write output
    output[idx] = softmax_val;
}

torch::Tensor fused_softmax_bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor bias, float scale) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros_like(input);
    
    dim3 block(channels);
    dim3 grid(batch * height * width);
    size_t shared_mem_size = 2 * channels * sizeof(float);
    
    fused_softmax_bias_scale_sigmoid_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), scale, output.data_ptr<float>(), 
        batch, channels, height, width
    );
    
    return output;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_softmax_bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor bias, float scale);"
)

# Compile the inline CUDA code for the fused operation
fused_op = load_inline(
    name="fused_softmax_bias_scale_sigmoid",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_softmax_bias_scale_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then applies a fused custom CUDA operator for softmax, bias add, scaling, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_softmax_bias_scale_sigmoid_cuda(x, self.bias, self.scaling_factor)
        return x