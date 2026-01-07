import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d + Softmax + Sigmoid fusion
conv_transpose_softmax_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void conv_transpose_softmax_sigmoid_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp_max, float* temp_sum,
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
        
        float sum = 0.0f;
        
        // Compute convolution transpose
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_d; kd++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int in_d_idx = (d + pad_d - kd * 1 + out_pad_d) / stride_d;
                        int in_h_idx = (h + pad_h - kh * 1 + out_pad_h) / stride_h;
                        int in_w_idx = (w + pad_w - kw * 1 + out_pad_w) / stride_w;
                        
                        if (in_d_idx >= 0 && in_d_idx < in_d && 
                            in_h_idx >= 0 && in_h_idx < in_h && 
                            in_w_idx >= 0 && in_w_idx < in_w &&
                            (d + pad_d - kd * 1 + out_pad_d) % stride_d == 0 &&
                            (h + pad_h - kh * 1 + out_pad_h) % stride_h == 0 &&
                            (w + pad_w - kw * 1 + out_pad_w) % stride_w == 0) {
                            
                            int input_idx = ((b * in_channels + ic) * in_d + in_d_idx) * in_h * in_w + in_h_idx * in_w + in_w_idx;
                            int weight_idx = ((c * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        output[idx] = sum;
        
        // Store for softmax reduction
        int spatial_idx = ((b * out_d + d) * out_h + h) * out_w + w;
        int channel_idx = c;
        int softmax_idx = spatial_idx * out_channels + channel_idx;
        
        atomicMax(&temp_max[spatial_idx], sum);
        __syncthreads();
        
        float exp_val = expf(sum - temp_max[spatial_idx]);
        atomicAdd(&temp_sum[spatial_idx], exp_val);
        __syncthreads();
        
        float softmax_val = exp_val / temp_sum[spatial_idx];
        output[idx] = 1.0f / (1.0f + expf(-softmax_val));
    }
}

torch::Tensor conv_transpose_softmax_sigmoid_cuda(
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
    auto temp_max = torch::full({batch_size * out_d * out_h * out_w}, -FLT_MAX, input.options());
    auto temp_sum = torch::zeros({batch_size * out_d * out_h * out_w}, input.options());
    
    const int block_size = 256;
    int total_size = batch_size * out_channels * out_d * out_h * out_w;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    conv_transpose_softmax_sigmoid_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), temp_max.data_ptr<float>(), temp_sum.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w);
    
    return output;
}
"""

conv_transpose_softmax_sigmoid_cpp_source = """
torch::Tensor conv_transpose_softmax_sigmoid_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w);
"""

# Compile the inline CUDA code
conv_transpose_softmax_sigmoid = load_inline(
    name="conv_transpose_softmax_sigmoid",
    cpp_sources=conv_transpose_softmax_sigmoid_cpp_source,
    cuda_sources=conv_transpose_softmax_sigmoid_source,
    functions=["conv_transpose_softmax_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.fused_op = conv_transpose_softmax_sigmoid
        
    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.Tensor()
        
        stride_d, stride_h, stride_w = self.conv_transpose.stride[0], self.conv_transpose.stride[1], self.conv_transpose.stride[2]
        pad_d, pad_h, pad_w = self.conv_transpose.padding[0], self.conv_transpose.padding[1], self.conv_transpose.padding[2]
        out_pad_d, out_pad_h, out_pad_w = self.conv_transpose.output_padding[0], self.conv_transpose.output_padding[1], self.conv_transpose.output_padding[2]
        
        return self.fused_op.conv_transpose_softmax_sigmoid_cuda(
            x, weight, bias,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            out_pad_d, out_pad_h, out_pad_w)