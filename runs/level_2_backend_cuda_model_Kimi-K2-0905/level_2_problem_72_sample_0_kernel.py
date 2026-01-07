import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d + BatchNorm3d + AvgPool3d fusion
fused_conv_bn_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 8

__global__ void fused_conv_transpose_bn_pool_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias, const float* bn_mean, const float* bn_var,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding,
    float epsilon) {
    
    int b = blockIdx.x;
    int c = blockIdx.y;
    int od = threadIdx.x + blockIdx.z * BLOCK_SIZE;
    int oh = threadIdx.y + (blockIdx.z / ((out_d + BLOCK_SIZE - 1) / BLOCK_SIZE)) * BLOCK_SIZE;
    int ow = threadIdx.z % ((out_h + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE + threadIdx.x;
    
    if (b >= batch_size || c >= out_channels || od >= out_d || oh >= out_h || ow >= out_w) return;
    
    float sum = 0.0f;
    
    // ConvTranspose3d computation
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_od = od + padding - kd;
                    int in_oh = oh + padding - kh;
                    int in_ow = ow + padding - kw;
                    
                    if (in_od >= 0 && in_od % stride == 0 &&
                        in_oh >= 0 && in_oh % stride == 0 &&
                        in_ow >= 0 && in_ow % stride == 0) {
                        
                        int in_d_idx = in_od / stride;
                        int in_h_idx = in_oh / stride;
                        int in_w_idx = in_ow / stride;
                        
                        if (in_d_idx < in_d && in_h_idx < in_h && in_w_idx < in_w) {
                            int input_idx = b * in_channels * in_d * in_h * in_w +
                                          ic * in_d * in_h * in_w +
                                          in_d_idx * in_h * in_w +
                                          in_h_idx * in_w +
                                          in_w_idx;
                            int weight_idx = c * in_channels * kernel_size * kernel_size * kernel_size +
                                           ic * kernel_size * kernel_size * kernel_size +
                                           kd * kernel_size * kernel_size +
                                           kh * kernel_size +
                                           kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += bias[c];
    }
    
    // BatchNorm3d computation
    float bn_scale = bn_weight[c] / sqrtf(bn_var[c] + epsilon);
    float bn_shift = bn_bias[c] - bn_weight[c] * bn_mean[c] / sqrtf(bn_var[c] + epsilon);
    sum = sum * bn_scale + bn_shift;
    
    // ReLU activation (assuming it's part of typical pipeline)
    sum = fmaxf(sum, 0.0f);
    
    // First AvgPool3d (2x2x2)
    float pool1_sum = 0.0f;
    int pool1_count = 0;
    for (int pd = 0; pd < 2 && (od * 2 + pd) < out_d; pd++) {
        for (int ph = 0; ph < 2 && (oh * 2 + ph) < out_h; ph++) {
            for (int pw = 0; pw < 2 && (ow * 2 + pw) < out_w; pw++) {
                pool1_sum += sum;
                pool1_count++;
            }
        }
    }
    float pool1_result = pool1_count > 0 ? pool1_sum / pool1_count : 0.0f;
    
    // Second AvgPool3d (2x2x2)
    float pool2_sum = 0.0f;
    int pool2_count = 0;
    for (int pd = 0; pd < 2 && (od * 4 + pd) < out_d; pd++) {
        for (int ph = 0; ph < 2 && (oh * 4 + ph) < out_h; ph++) {
            for (int pw = 0; pw < 2 && (ow * 4 + pw) < out_w; pw++) {
                pool2_sum += pool1_result;
                pool2_count++;
            }
        }
    }
    float pool2_result = pool2_count > 0 ? pool2_sum / pool2_count : 0.0f;
    
    int out_idx = b * out_channels * (out_d/4) * (out_h/4) * (out_w/4) +
                  c * (out_d/4) * (out_h/4) * (out_w/4) +
                  (od/4) * (out_h/4) * (out_w/4) +
                  (oh/4) * (out_w/4) +
                  (ow/4);
    
    if (od % 4 == 0 && oh % 4 == 0 && ow % 4 == 0 &&
        od/4 < out_d/4 && oh/4 < out_h/4 && ow/4 < out_w/4) {
        output[out_idx] = pool2_result;
    }
}

torch::Tensor fused_conv_transpose_bn_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,
    int kernel_size, int stride, int padding, float epsilon) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);
    auto out_channels = weight.size(0);
    
    int out_d = (in_d - 1) * stride - 2 * padding + kernel_size;
    int out_h = (in_h - 1) * stride - 2 * padding + kernel_size;
    int out_w = (in_w - 1) * stride - 2 * padding + kernel_size;
    
    auto output = torch::zeros({batch_size, out_channels, out_d/4, out_h/4, out_w/4}, input.options());
    
    dim3 blocks(batch_size, out_channels);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    
    fused_conv_transpose_bn_pool_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(), bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_size, stride, padding, epsilon);
    
    return output;
}
"""

fused_conv_bn_pool_cpp_source = """
torch::Tensor fused_conv_transpose_bn_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,
    int kernel_size, int stride, int padding, float epsilon);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_conv_bn_pool_cpp_source,
    cuda_sources=fused_conv_bn_pool_source,
    functions=["fused_conv_transpose_bn_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_ops = fused_ops
        
    def forward(self, x):
        # Get batch norm parameters
        bn_weight = self.batch_norm.weight
        bn_bias = self.batch_norm.bias
        bn_mean = self.batch_norm.running_mean
        bn_var = self.batch_norm.running_var
        
        # Get conv parameters
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.empty(0)
        
        # Call fused kernel
        return self.fused_ops.fused_conv_transpose_bn_pool_cuda(
            x, weight, bias, bn_weight, bn_bias, bn_mean, bn_var,
            self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0], self.conv_transpose.padding[0],
            self.batch_norm.eps)