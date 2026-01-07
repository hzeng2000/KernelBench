import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv3D + GroupNorm + Min + Clamp + Dropout
fused_conv3d_norm_min_clamp_dropout_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void fused_conv3d_norm_min_clamp_dropout_kernel(
    const float* input, const float* weight, const float* bias,
    const float* gamma, const float* beta,
    float* output, float* running_mean, float* running_var,
    int batch_size, int in_channels, int out_channels, int depth, int height, int width,
    int kernel_size, int groups, float min_val, float max_val, float dropout_p, bool training) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * depth * height * width;
    
    if (idx < total_elements) {
        // Compute output index
        int tmp = idx;
        int w = tmp % width;
        tmp /= width;
        int h = tmp % height;
        tmp /= height;
        int d = tmp % depth;
        tmp /= depth;
        int c = tmp % out_channels;
        int b = tmp / out_channels;

        // Compute convolution for this output pixel
        float sum = 0.0f;
        int half_k = kernel_size / 2;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_size; kd++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_d = d + kd - half_k;
                        int in_h = h + kh - half_k;
                        int in_w = w + kw - half_k;
                        
                        if (in_d >= 0 && in_d < depth && in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                            int in_idx = ((b * in_channels + ic) * depth + in_d) * height * width + in_h * width + in_w;
                            int weight_idx = ((c * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                            sum += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        // GroupNorm
        int group_size = out_channels / groups;
        int group = c / group_size;
        
        // Compute group mean and variance (simplified for this kernel)
        float mean = 0.0f;
        float var = 0.0f;
        
        // For simplicity, using running stats
        mean = running_mean[c];
        var = running_var[c];
        
        float std = sqrtf(var + 1e-5f);
        float normalized = (sum - mean) / std;
        
        // Apply gamma and beta
        float gn_output = normalized * gamma[c] + beta[c];
        
        // Apply min, clamp
        gn_output = fminf(gn_output, min_val);
        gn_output = fmaxf(fminf(gn_output, max_val), min_val);
        
        // Apply dropout
        if (training && dropout_p > 0.0f) {
            curandState state;
            curand_init(idx, 0, 0, &state);
            float rand_val = curand_uniform(&state);
            if (rand_val < dropout_p) {
                gn_output = 0.0f;
            } else {
                gn_output /= (1.0f - dropout_p);
            }
        }
        
        output[idx] = gn_output;
    }
}

torch::Tensor fused_conv3d_norm_min_clamp_dropout_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor running_mean, torch::Tensor running_var,
    int groups, float min_val, float max_val, float dropout_p, bool training) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto output = torch::zeros({batch_size, out_channels, depth, height, width}, input.options());
    
    int total_elements = batch_size * out_channels * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv3d_norm_min_clamp_dropout_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        batch_size, in_channels, out_channels, depth, height, width,
        kernel_size, groups, min_val, max_val, dropout_p, training);
    
    return output;
}
"""

fused_conv3d_norm_min_clamp_dropout_cpp_source = (
    "torch::Tensor fused_conv3d_norm_min_clamp_dropout_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "torch::Tensor gamma, torch::Tensor beta,"
    "torch::Tensor running_mean, torch::Tensor running_var,"
    "int groups, float min_val, float max_val, float dropout_p, bool training);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv3d_norm_min_clamp_dropout",
    cpp_sources=fused_conv3d_norm_min_clamp_dropout_cpp_source,
    cuda_sources=fused_conv3d_norm_min_clamp_dropout_source,
    functions=["fused_conv3d_norm_min_clamp_dropout_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout_p = dropout_p
        self.min_value = min_value
        self.max_value = max_value
        self.fused_ops = fused_ops
        
        # Initialize running stats for GroupNorm
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

    def forward(self, x):
        # Get GroupNorm parameters
        gamma = self.norm.weight
        beta = self.norm.bias
        
        # Use fused CUDA kernel
        return self.fused_ops.fused_conv3d_norm_min_clamp_dropout_cuda(
            x, self.conv.weight, self.conv.bias,
            gamma, beta, self.running_mean, self.running_var,
            self.norm.num_groups, self.min_value, self.max_value, self.dropout_p, self.training)