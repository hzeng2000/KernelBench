import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d + ReLU + GroupNorm fusion
conv_transpose_relu_gn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

__global__ void conv_transpose_relu_gn_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* running_mean, float* running_var,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int groups, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx < total_elements) {
        // Calculate output indices
        int tmp = idx;
        int w = tmp % out_w;
        tmp /= out_w;
        int h = tmp % out_h;
        tmp /= out_h;
        int d = tmp % out_d;
        tmp /= out_d;
        int c = tmp % out_channels;
        int b = tmp / out_channels;
        
        // Calculate group
        int group = c / (out_channels / groups);
        int channels_per_group = out_channels / groups;
        
        float sum = 0.0f;
        
        // ConvTranspose3d computation
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_d; kd++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int in_d_idx = d + kd - kernel_d / 2;
                        int in_h_idx = h + kh - kernel_h / 2;
                        int in_w_idx = w + kw - kernel_w / 2;
                        
                        if (in_d_idx >= 0 && in_d_idx < in_d &&
                            in_h_idx >= 0 && in_h_idx < in_h &&
                            in_w_idx >= 0 && in_w_idx < in_w) {
                            
                            int input_idx = ((b * in_channels + ic) * in_d + in_d_idx) * in_h * in_w + in_h_idx * in_w + in_w_idx;
                            int weight_idx = ((c * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        // ReLU activation
        sum = fmaxf(0.0f, sum);
        
        // Store intermediate result for group norm
        output[idx] = sum;
    }
}

__global__ void group_norm_kernel(
    float* output, const float* input,
    int batch_size, int out_channels, int out_d, int out_h, int out_w,
    int groups, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx < total_elements) {
        // Calculate output indices
        int tmp = idx;
        int w = tmp % out_w;
        tmp /= out_w;
        int h = tmp % out_h;
        tmp /= out_h;
        int d = tmp % out_d;
        tmp /= out_d;
        int c = tmp % out_channels;
        int b = tmp / out_channels;
        
        // Calculate group
        int group = c / (out_channels / groups);
        int channels_per_group = out_channels / groups;
        
        // Compute mean for this group
        float sum = 0.0f;
        int count = 0;
        
        for (int gc = group * channels_per_group; gc < (group + 1) * channels_per_group; gc++) {
            for (int gd = 0; gd < out_d; gd++) {
                for (int gh = 0; gh < out_h; gh++) {
                    for (int gw = 0; gw < out_w; gw++) {
                        int group_idx = ((b * out_channels + gc) * out_d + gd) * out_h * out_w + gh * out_w + gw;
                        sum += input[group_idx];
                        count++;
                    }
                }
            }
        }
        
        float mean = sum / count;
        
        // Compute variance for this group
        float var_sum = 0.0f;
        for (int gc = group * channels_per_group; gc < (group + 1) * channels_per_group; gc++) {
            for (int gd = 0; gd < out_d; gd++) {
                for (int gh = 0; gh < out_h; gh++) {
                    for (int gw = 0; gw < out_w; gw++) {
                        int group_idx = ((b * out_channels + gc) * out_d + gd) * out_h * out_w + gh * out_w + gw;
                        float diff = input[group_idx] - mean;
                        var_sum += diff * diff;
                    }
                }
            }
        }
        
        float variance = var_sum / count;
        float std_dev = sqrtf(variance + eps);
        
        // Normalize
        output[idx] = (input[idx] - mean) / std_dev;
    }
}

torch::Tensor conv_transpose_relu_gn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int groups) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_d = input.size(2);
    const int in_h = input.size(3);
    const int in_w = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_d = kernel_size;
    const int kernel_h = kernel_size;
    const int kernel_w = kernel_size;
    
    const int out_d = in_d;
    const int out_h = in_h;
    const int out_w = in_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_d, out_h, out_w}, input.options());
    auto intermediate = torch::zeros_like(output);
    
    const int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_transpose_relu_gn_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        intermediate.data_ptr<float>(), nullptr, nullptr,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w, groups, 1e-5f);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    group_norm_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), intermediate.data_ptr<float>(),
        batch_size, out_channels, out_d, out_h, out_w, groups, 1e-5f);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return output;
}
"""

conv_transpose_relu_gn_cpp_source = """
torch::Tensor conv_transpose_relu_gn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int groups);
"""

# Compile the inline CUDA code
conv_transpose_relu_gn = load_inline(
    name="conv_transpose_relu_gn",
    cpp_sources=conv_transpose_relu_gn_cpp_source,
    cuda_sources=conv_transpose_relu_gn_source,
    functions=["conv_transpose_relu_gn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization
    using a fused custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.kernel_size = kernel_size
        self.groups = groups
        self.fused_op = conv_transpose_relu_gn

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        return self.fused_op.conv_transpose_relu_gn_cuda(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.kernel_size, self.groups
        )