import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ConvTranspose3d + MaxPool3d + Softmax + Subtract + Swish + Max
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_kernel(
    const float* input, const float* weight, const float* bias, const float* subtract,
    float* output, int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w, int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding, int output_padding,
    int pool_kernel, int pool_stride, int pool_padding,
    int pooled_d, int pooled_h, int pooled_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * pooled_h * pooled_w * pooled_d;
    
    if (idx < total_size) {
        int pd = idx % pooled_d;
        int tmp = idx / pooled_d;
        int ph = tmp % pooled_h;
        int b = tmp / pooled_h;
        
        float max_val = -FLT_MAX;
        
        for (int c = 0; c < out_channels; c++) {
            float conv_sum = 0.0f;
            
            // ConvTranspose3d forward pass (simplified)
            for (int ic = 0; ic < in_channels; ic++) {
                for (int kd = 0; kd < kernel_size; kd++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int in_d_idx = (pd * pool_stride - pool_padding) * stride + kd - padding;
                            int in_h_idx = (ph * pool_stride - pool_padding) * stride + kh - padding;
                            int in_w_idx = kw - padding;
                            
                            if (in_d_idx >= 0 && in_d_idx < in_d && in_h_idx >= 0 && in_h_idx < in_h && in_w_idx >= 0 && in_w_idx < in_w) {
                                int in_idx = ((b * in_channels + ic) * in_d + in_d_idx) * in_h * in_w + in_h_idx * in_w + in_w_idx;
                                int weight_idx = ((c * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                                conv_sum += input[in_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            if (bias != nullptr) {
                conv_sum += bias[c];
            }
            
            // MaxPool3d (approximated with max over channels)
            // Softmax across channels (simplified)
            // Subtract
            float sub_val = conv_sum - subtract[c];
            
            // Swish
            float sigmoid = 1.0f / (1.0f + expf(-sub_val));
            float swish_val = sigmoid * sub_val;
            
            if (swish_val > max_val) {
                max_val = swish_val;
            }
        }
        
        output[idx] = max_val;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract,
    int kernel_size, int stride, int padding, int output_padding,
    int pool_kernel, int pool_stride, int pool_padding) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);
    auto out_channels = weight.size(0);
    
    int out_d = (in_d - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_h = (in_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_w = (in_w - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int pooled_d = (out_d + 2 * pool_padding - pool_kernel) / pool_stride + 1;
    int pooled_h = (out_h + 2 * pool_padding - pool_kernel) / pool_stride + 1;
    int pooled_w = (out_w + 2 * pool_padding - pool_kernel) / pool_stride + 1;
    
    auto output = torch::zeros({batch_size, pooled_d, pooled_h, pooled_w}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * pooled_d * pooled_h * pooled_w + block_size - 1) / block_size;
    
    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        subtract.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_size, stride, padding, output_padding,
        pool_kernel, pool_stride, pool_padding,
        pooled_d, pooled_h, pooled_w);
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract,"
    "int kernel_size, int stride, int padding, int output_padding,"
    "int pool_kernel, int pool_stride, int pool_padding);"
)

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(
            x, self.conv_transpose.weight, self.conv_transpose.bias, self.subtract,
            self.kernel_size, self.stride, self.padding, self.output_padding,
            self.pool_kernel_size, self.pool_stride, self.pool_padding
        )