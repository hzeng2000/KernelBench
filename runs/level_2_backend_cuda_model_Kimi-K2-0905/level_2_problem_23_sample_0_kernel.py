import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv3D + GroupNorm + Mean
fused_conv3d_gn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define BLOCK_SIZE 8

__global__ void fused_conv3d_gn_mean_kernel(
    const float* input, const float* weight, const float* bias,
    const float* gamma, const float* beta,
    float* output, float* mean_output,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int W,
    int out_d, int out_h, int out_w,
    int kernel_size, int num_groups) {

    int b = blockIdx.z;
    int g = blockIdx.y;
    int out_c_start = g * (out_channels / num_groups);
    int out_c_end = (g + 1) * (out_channels / num_groups);
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = (blockIdx.y * blockDim.y + threadIdx.y) % out_h;
    int out_z = (blockIdx.y * blockDim.y + threadIdx.y) / out_h;
    
    if (out_z >= out_d || out_y >= out_h) return;
    
    int out_idx = b * out_channels * out_d * out_h * out_w + 
                  out_c_start * out_d * out_h * out_w +
                  out_z * out_h * out_w + out_y * out_w + out_x;
    
    float group_sum = 0.0f;
    float group_sq_sum = 0.0f;
    int group_count = 0;
    
    for (int out_c = out_c_start; out_c < out_c_end; out_c++) {
        if (out_x < out_w) {
            float conv_sum = 0.0f;
            
            for (int in_c = 0; in_c < in_channels; in_c++) {
                for (int kz = 0; kz < kernel_size; kz++) {
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            int in_z = out_z + kz;
                            int in_y = out_y + ky;
                            int in_x = out_x + kx;
                            
                            if (in_z < in_d && in_y < in_h && in_x < W) {
                                int in_idx = b * in_channels * in_d * in_h * W +
                                           in_c * in_d * in_h * W +
                                           in_z * in_h * W + in_y * W + in_x;
                                int weight_idx = out_c * in_channels * kernel_size * kernel_size * kernel_size +
                                               in_c * kernel_size * kernel_size * kernel_size +
                                               kz * kernel_size * kernel_size + ky * kernel_size + kx;
                                conv_sum += input[in_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            if (bias != nullptr) {
                conv_sum += bias[out_c];
            }
            
            int out_linear_idx = b * out_channels * out_d * out_h * out_w +
                               out_c * out_d * out_h * out_w +
                               out_z * out_h * out_w + out_y * out_w + out_x;
            output[out_linear_idx] = conv_sum;
            
            float gn_gamma = gamma[out_c];
            float gn_beta = beta[out_c];
            
            // Compute group statistics
            group_sum += conv_sum;
            group_sq_sum += conv_sum * conv_sum;
            group_count++;
        }
    }
    
    // Compute group norm
    if (group_count > 0 && out_x < out_w) {
        float group_mean = group_sum / group_count;
        float group_var = (group_sq_sum / group_count) - (group_mean * group_mean);
        float group_std = sqrtf(group_var + 1e-5f);
        
        for (int out_c = out_c_start; out_c < out_c_end; out_c++) {
            int out_linear_idx = b * out_channels * out_d * out_h * out_w +
                               out_c * out_d * out_h * out_w +
                               out_z * out_h * out_w + out_y * out_w + out_x;
            if (out_x < out_w) {
                float normalized = (output[out_linear_idx] - group_mean) / group_std;
                float gn_gamma = gamma[out_c];
                float gn_beta = beta[out_c];
                output[out_linear_idx] = normalized * gn_gamma + gn_beta;
                
                // Accumulate for final mean
                atomicAdd(&mean_output[b], output[out_linear_idx] / (out_channels * out_d * out_h * out_w));
            }
        }
    }
}

torch::Tensor fused_conv3d_gn_mean_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int num_groups) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_d = input.size(2);
    const auto in_h = input.size(3);
    const auto in_w = input.size(4);
    
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    
    const auto out_d = in_d - kernel_size + 1;
    const auto out_h = in_h - kernel_size + 1;
    const auto out_w = in_w - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_d, out_h, out_w}, input.options());
    auto mean_output = torch::zeros({batch_size}, input.options());
    
    dim3 blockSize(8, 8);
    dim3 gridSize((out_w + blockSize.x - 1) / blockSize.x,
                  (out_d * out_h + blockSize.y - 1) / blockSize.y,
                  batch_size * num_groups);
    
    fused_conv3d_gn_mean_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(), mean_output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_size, num_groups);
    
    return mean_output;
}
"""

fused_conv3d_gn_mean_cpp_source = """
torch::Tensor fused_conv3d_gn_mean_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int num_groups);
"""

# Compile the inline CUDA code
fused_conv3d_gn_mean = load_inline(
    name="fused_conv3d_gn_mean",
    cpp_sources=fused_conv3d_gn_mean_cpp_source,
    cuda_sources=fused_conv3d_gn_mean_source,
    functions=["fused_conv3d_gn_mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_conv3d_gn_mean = fused_conv3d_gn_mean
        self.num_groups = num_groups
        
    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        
        return self.fused_conv3d_gn_mean.fused_conv3d_gn_mean_cuda(
            x, weight, bias, gamma, beta, self.num_groups)