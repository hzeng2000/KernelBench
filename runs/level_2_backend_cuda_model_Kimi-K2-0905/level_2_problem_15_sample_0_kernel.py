import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d + BatchNorm3d + Mean subtraction fusion
conv_transpose_bn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

__global__ void fused_conv_transpose_bn_mean_kernel(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, const float* gamma, const float* beta,
    float* output, float* temp_output,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx < total_elements) {
        int tmp = idx;
        int n = tmp / (out_channels * out_d * out_h * out_w);
        tmp %= (out_channels * out_d * out_h * out_w);
        int c = tmp / (out_d * out_h * out_w);
        tmp %= (out_d * out_h * out_w);
        int d = tmp / (out_h * out_w);
        tmp %= (out_h * out_w);
        int h = tmp / out_w;
        int w = tmp % out_w;
        
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_d; kd++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int in_d_idx = (d - kd + pad_d) / stride_d;
                        int in_h_idx = (h - kh + pad_h) / stride_h;
                        int in_w_idx = (w - kw + pad_w) / stride_w;
                        
                        if ((d - kd + pad_d) % stride_d == 0 && 
                            (h - kh + pad_h) % stride_h == 0 && 
                            (w - kw + pad_w) % stride_w == 0 &&
                            in_d_idx >= 0 && in_d_idx < in_d &&
                            in_h_idx >= 0 && in_h_idx < in_h &&
                            in_w_idx >= 0 && in_w_idx < in_w) {
                            
                            int input_idx = n * in_channels * in_d * in_h * in_w +
                                          ic * in_d * in_h * in_w +
                                          in_d_idx * in_h * in_w +
                                          in_h_idx * in_w +
                                          in_w_idx;
                            
                            int weight_idx = ic * out_channels * kernel_d * kernel_h * kernel_w +
                                           c * kernel_d * kernel_h * kernel_w +
                                           kd * kernel_h * kernel_w +
                                           kh * kernel_w +
                                           kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        // Batch normalization
        float bn_val = (sum - running_mean[c]) / sqrtf(running_var[c] + eps);
        bn_val = bn_val * gamma[c] + beta[c];
        
        temp_output[idx] = bn_val;
    }
}

__global__ void compute_spatial_mean_kernel(
    const float* input, float* output,
    int batch_size, int channels, int depth, int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels;
    
    if (idx < total_elements) {
        int n = idx / channels;
        int c = idx % channels;
        
        float sum = 0.0f;
        int spatial_size = depth * height * width;
        
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int input_idx = n * channels * depth * height * width +
                                  c * depth * height * width +
                                  d * height * width +
                                  h * width +
                                  w;
                    sum += input[input_idx];
                }
            }
        }
        
        float mean = sum / spatial_size;
        
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int output_idx = n * channels * depth * height * width +
                                   c * depth * height * width +
                                   d * height * width +
                                   h * width +
                                   w;
                    output[output_idx] = input[output_idx] - mean;
                }
            }
        }
    }
}

torch::Tensor fused_conv_transpose_bn_mean_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor gamma, torch::Tensor beta,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    float eps) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);
    
    auto out_channels = weight.size(1);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    auto out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d;
    auto out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h;
    auto out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w;
    
    auto temp_output = torch::zeros({batch_size, out_channels, out_d, out_h, out_w}, input.options());
    auto output = torch::zeros_like(temp_output);
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_transpose_bn_mean_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        temp_output.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w, eps);
    
    int mean_elements = batch_size * out_channels;
    int mean_blocks = (mean_elements + block_size - 1) / block_size;
    
    compute_spatial_mean_kernel<<<mean_blocks, block_size>>>(
        temp_output.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, out_channels, out_d, out_h, out_w);
    
    return output;
}
"""

conv_transpose_bn_mean_cpp_source = """
torch::Tensor fused_conv_transpose_bn_mean_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor gamma, torch::Tensor beta,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=conv_transpose_bn_mean_cpp_source,
    cuda_sources=conv_transpose_bn_mean_source,
    functions=["fused_conv_transpose_bn_mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_ops = fused_ops
        
        # Store parameters for CUDA kernel
        self.stride_d = stride
        self.stride_h = stride
        self.stride_w = stride
        self.pad_d = padding
        self.pad_h = padding
        self.pad_w = padding

    def forward(self, x):
        return self.fused_ops.fused_conv_transpose_bn_mean_cuda(
            x, self.conv_transpose.weight, self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.empty(0),
            self.batch_norm.running_mean, self.batch_norm.running_var,
            self.batch_norm.weight, self.batch_norm.bias,
            self.stride_d, self.stride_h, self.stride_w,
            self.pad_d, self.pad_h, self.pad_w,
            self.batch_norm.eps
        )