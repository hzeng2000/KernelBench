import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_transpose_conv_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* workspace,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx < total_size) {
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
                        int in_d_idx = d * stride_d - pad_d + kd;
                        int in_h_idx = h * stride_h - pad_h + kh;
                        int in_w_idx = w * stride_w - pad_w + kw;
                        
                        if (in_d_idx >= 0 && in_d_idx < in_d &&
                            in_h_idx >= 0 && in_h_idx < in_h &&
                            in_w_idx >= 0 && in_w_idx < in_w) {
                            
                            int input_idx = n * in_channels * in_d * in_h * in_w +
                                          ic * in_d * in_h * in_w +
                                          in_d_idx * in_h * in_w +
                                          in_h_idx * in_w +
                                          in_w_idx;
                            
                            int weight_idx = c * in_channels * kernel_d * kernel_h * kernel_w +
                                           ic * kernel_d * kernel_h * kernel_w +
                                           kd * kernel_h * kernel_w +
                                           kh * kernel_w +
                                           kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        int out_idx = n * out_channels * out_d * out_h * out_w +
                     c * out_d * out_h * out_w +
                     d * out_h * out_w +
                     h * out_w +
                     w;
        
        float conv_result = sum + bias[c];
        float original = conv_result;
        float temp = conv_result + original;
        temp = temp * original;
        temp = temp + original;
        
        output[out_idx] = temp;
    }
}

torch::Tensor fused_transpose_conv_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    auto out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
    auto out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    auto out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_d, out_h, out_w}, input.options());
    
    int total_size = batch_size * out_channels * out_d * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_transpose_conv_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), nullptr,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w);
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_transpose_conv_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int stride_d, int stride_h, int stride_w,"
    "int pad_d, int pad_h, int pad_w,"
    "int out_pad_d, int out_pad_h, int out_pad_w);"
)

fused_transpose_conv = load_inline(
    name="fused_transpose_conv",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_transpose_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_transpose_conv

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.bias.squeeze()
        
        stride_d, stride_h, stride_w = self.conv_transpose.stride[0], self.conv_transpose.stride[1], self.conv_transpose.stride[2]
        pad_d, pad_h, pad_w = self.conv_transpose.padding[0], self.conv_transpose.padding[1], self.conv_transpose.padding[2]
        out_pad_d, out_pad_h, out_pad_w = self.conv_transpose.output_padding[0], self.conv_transpose.output_padding[1], self.conv_transpose.output_padding[2]
        
        return self.fused_op.fused_transpose_conv_cuda(
            x, weight, bias,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            out_pad_d, out_pad_h, out_pad_w)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]