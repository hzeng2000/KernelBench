import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv2d + avgpool + sigmoid + sum
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_conv_avgpool_sigmoid_sum_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width, int kernel_size, int pool_size) {
    
    int batch_idx = blockIdx.z;
    int out_c = blockIdx.y;
    int out_h = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_w_base = threadIdx.y * TILE_SIZE;
    
    if (out_h >= out_height) return;
    
    float pool_sum = 0.0f;
    int pool_count = 0;
    
    for (int ph = 0; ph < pool_size && (out_h * pool_size + ph) < (in_height - kernel_size + 1); ph++) {
        for (int pw = 0; pw < pool_size && (out_w_base + pw) < (in_width - kernel_size + 1); pw++) {
            float conv_sum = 0.0f;
            
            for (int in_c = 0; in_c < in_channels; in_c++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_idx = ((batch_idx * in_channels + in_c) * in_height + (out_h * pool_size + ph + kh)) * in_width + (out_w_base + pw + kw);
                        int weight_idx = ((out_c * in_channels + in_c) * kernel_size + kh) * kernel_size + kw;
                        conv_sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
            
            if (bias != nullptr) {
                conv_sum += bias[out_c];
            }
            
            float sigmoid_val = 1.0f / (1.0f + expf(-conv_sum));
            pool_sum += sigmoid_val;
            pool_count++;
        }
    }
    
    if (pool_count > 0) {
        float avg_val = pool_sum / pool_count;
        int out_idx = ((batch_idx * out_channels + out_c) * out_height + out_h) * out_width + out_w_base / TILE_SIZE;
        atomicAdd(&output[batch_idx], avg_val);
    }
}

torch::Tensor fused_conv_avgpool_sigmoid_sum_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int pool_size) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto out_height = (in_height - kernel_size + 1) / pool_size;
    auto out_width = (in_width - kernel_size + 1) / pool_size;
    
    auto output = torch::zeros({batch_size}, input.options());
    
    dim3 blocks((out_height + TILE_SIZE - 1) / TILE_SIZE, out_channels, batch_size);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    fused_conv_avgpool_sigmoid_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width, kernel_size, pool_size);
    
    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_conv_avgpool_sigmoid_sum_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "int kernel_size, int pool_size);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_conv_avgpool_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self.fused_op = fused_op
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        return self.fused_op.fused_conv_avgpool_sigmoid_sum_cuda(
            x, self.conv.weight, self.conv.bias, self.kernel_size, self.pool_kernel_size)