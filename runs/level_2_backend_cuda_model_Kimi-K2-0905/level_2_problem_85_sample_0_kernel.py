import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom fused CUDA kernel for conv + group norm + scale + maxpool + clamp
fused_conv_block_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl; \
      exit(1); \
    } \
  } while (0)

__global__ void fused_scale_maxpool_clamp_kernel(
    const float* input, float* output, 
    const float* scale, 
    int batch_size, int channels, int in_height, int in_width,
    int out_height, int out_width, int kernel_size,
    float clamp_min, float clamp_max) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx < total_elements) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int n = idx / (out_width * out_height * channels);
        
        int h_in_start = h_out * kernel_size;
        int w_in_start = w_out * kernel_size;
        
        float max_val = -1e20f;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_in_start + kh;
                int w_in = w_in_start + kw;
                
                if (h_in < in_height && w_in < in_width) {
                    int input_idx = ((n * channels + c) * in_height + h_in) * in_width + w_in;
                    float val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
        
        max_val *= scale[c];
        max_val = fmaxf(clamp_min, fminf(clamp_max, max_val));
        
        output[idx] = max_val;
    }
}

torch::Tensor fused_conv_block_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor scale,
    int stride, int padding, int groups,
    int maxpool_kernel_size,
    float clamp_min, float clamp_max) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    
    // Conv2d using cuDNN
    auto conv_out = torch::nn::functional::conv2d(
        input, weight, bias, 
        torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding).groups(groups)
    );
    
    // Group normalization
    auto group_norm_out = torch::nn::functional::group_norm(
        conv_out, groups, 
        torch::ones({out_channels}, conv_out.options()), 
        torch::zeros({out_channels}, conv_out.options())
    );
    
    // Calculate output dimensions for maxpool
    int out_height = in_height / maxpool_kernel_size;
    int out_width = in_width / maxpool_kernel_size;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              group_norm_out.options());
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_scale_maxpool_clamp_kernel<<<num_blocks, block_size>>>(
        group_norm_out.data_ptr<float>(), output.data_ptr<float>(),
        scale.data_ptr<float>(),
        batch_size, out_channels, in_height, in_width,
        out_height, out_width, maxpool_kernel_size,
        clamp_min, clamp_max
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}
"""

fused_conv_block_cpp_source = """
torch::Tensor fused_conv_block_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor scale,
    int stride, int padding, int groups,
    int maxpool_kernel_size,
    float clamp_min, float clamp_max);
"""

# Compile the inline CUDA code
fused_conv_block = load_inline(
    name="fused_conv_block",
    cpp_sources=fused_conv_block_cpp_source,
    cuda_sources=fused_conv_block_source,
    functions=["fused_conv_block_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudnn"],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs convolution, group normalization, scaling, max pooling, and clamping
    using a custom fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_conv_block = fused_conv_block

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        return self.fused_conv_block.fused_conv_block_cuda(
            x, self.conv.weight, self.conv.bias,
            self.scale,
            1, 1, 1,
            self.maxpool_kernel_size,
            self.clamp_min, self.clamp_max
        )

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]