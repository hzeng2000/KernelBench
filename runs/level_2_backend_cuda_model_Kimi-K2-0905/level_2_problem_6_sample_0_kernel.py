import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv3D + Softmax + MaxPool3D
fused_conv_softmax_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 8

__global__ void conv3d_kernel(
    const float* input, const float* weight, const float* bias,
    float* conv_out, int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w, int out_d, int out_h, int out_w,
    int kernel_size) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.w;
    
    if (out_x >= out_w || out_y >= out_h || out_z >= out_d || b >= batch_size) return;
    
    int kernel_radius = kernel_size / 2;
    float sum = 0.0f;
    
    for (int oc = 0; oc < out_channels; oc++) {
        float val = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kz = 0; kz < kernel_size; kz++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int in_z = out_z + kz - kernel_radius;
                        int in_y = out_y + ky - kernel_radius;
                        int in_x = out_x + kx - kernel_radius;
                        
                        if (in_z >= 0 && in_z < in_d && in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                            int in_idx = b * in_channels * in_d * in_h * in_w +
                                        ic * in_d * in_h * in_w +
                                        in_z * in_h * in_w +
                                        in_y * in_w +
                                        in_x;
                            int weight_idx = oc * in_channels * kernel_size * kernel_size * kernel_size +
                                            ic * kernel_size * kernel_size * kernel_size +
                                            kz * kernel_size * kernel_size +
                                            ky * kernel_size +
                                            kx;
                            val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        if (bias != nullptr) {
            val += bias[oc];
        }
        int out_idx = b * out_channels * out_d * out_h * out_w +
                     oc * out_d * out_h * out_w +
                     out_z * out_h * out_w +
                     out_y * out_w +
                     out_x;
        conv_out[out_idx] = val;
    }
}

__global__ void softmax_kernel(
    float* conv_out, int batch_size, int channels, int d, int h, int w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * d * h * w;
    
    if (idx >= total_size) return;
    
    int b = idx / (d * h * w);
    int rem = idx % (d * h * w);
    int z = rem / (h * w);
    rem = rem % (h * w);
    int y = rem / w;
    int x = rem % w;
    
    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
        int offset = b * channels * d * h * w + c * d * h * w + z * h * w + y * w + x;
        max_val = fmaxf(max_val, conv_out[offset]);
    }
    
    float sum = 0.0f;
    for (int c = 0; c < channels; c++) {
        int offset = b * channels * d * h * w + c * d * h * w + z * h * w + y * w + x;
        conv_out[offset] = expf(conv_out[offset] - max_val);
        sum += conv_out[offset];
    }
    
    for (int c = 0; c < channels; c++) {
        int offset = b * channels * d * h * w + c * d * h * w + z * h * w + y * w + x;
        conv_out[offset] /= sum;
    }
}

__global__ void maxpool3d_kernel(
    const float* input, float* output,
    int batch_size, int channels, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w, int pool_size) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.w;
    
    if (out_x >= out_w || out_y >= out_h || out_z >= out_d || b >= batch_size) return;
    
    for (int c = 0; c < channels; c++) {
        float max_val = -FLT_MAX;
        
        for (int pz = 0; pz < pool_size; pz++) {
            for (int py = 0; py < pool_size; py++) {
                for (int px = 0; px < pool_size; px++) {
                    int in_z = out_z * pool_size + pz;
                    int in_y = out_y * pool_size + py;
                    int in_x = out_x * pool_size + px;
                    
                    if (in_z < in_d && in_y < in_h && in_x < in_w) {
                        int in_idx = b * channels * in_d * in_h * in_w +
                                    c * in_d * in_h * in_w +
                                    in_z * in_h * in_w +
                                    in_y * in_w +
                                    in_x;
                        max_val = fmaxf(max_val, input[in_idx]);
                    }
                }
            }
        }
        
        int out_idx = b * channels * out_d * out_h * out_w +
                     c * out_d * out_h * out_w +
                     out_z * out_h * out_w +
                     out_y * out_w +
                     out_x;
        output[out_idx] = max_val;
    }
}

torch::Tensor fused_conv_softmax_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int pool_kernel_size) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);
    auto out_channels = weight.size(0);
    
    int kernel_radius = kernel_size / 2;
    int out_d = in_d - 2 * kernel_radius;
    int out_h = in_h - 2 * kernel_radius;
    int out_w = in_w - 2 * kernel_radius;
    
    auto conv_out = torch::zeros({batch_size, out_channels, out_d, out_h, out_w}, input.options());
    
    dim3 block_size(8, 8, 8);
    dim3 grid_size(
        (out_w + block_size.x - 1) / block_size.x,
        (out_h + block_size.y - 1) / block_size.y,
        (out_d + block_size.z - 1) / block_size.z,
        batch_size
    );
    
    conv3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        conv_out.data_ptr<float>(), batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w, kernel_size
    );
    
    int total_elements = batch_size * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    softmax_kernel<<<blocks, threads>>>(
        conv_out.data_ptr<float>(), batch_size, out_channels, out_d, out_h, out_w
    );
    
    int pool1_out_d = out_d / pool_kernel_size;
    int pool1_out_h = out_h / pool_kernel_size;
    int pool1_out_w = out_w / pool_kernel_size;
    auto pool1_out = torch::zeros({batch_size, out_channels, pool1_out_d, pool1_out_h, pool1_out_w}, input.options());
    
    dim3 pool1_block_size(8, 8, 8);
    dim3 pool1_grid_size(
        (pool1_out_w + pool1_block_size.x - 1) / pool1_block_size.x,
        (pool1_out_h + pool1_block_size.y - 1) / pool1_block_size.y,
        (pool1_out_d + pool1_block_size.z - 1) / pool1_block_size.z,
        batch_size
    );
    
    maxpool3d_kernel<<<pool1_grid_size, pool1_block_size>>>(
        conv_out.data_ptr<float>(), pool1_out.data_ptr<float>(),
        batch_size, out_channels, out_d, out_h, out_w,
        pool1_out_d, pool1_out_h, pool1_out_w, pool_kernel_size
    );
    
    int pool2_out_d = pool1_out_d / pool_kernel_size;
    int pool2_out_h = pool1_out_h / pool_kernel_size;
    int pool2_out_w = pool1_out_w / pool_kernel_size;
    auto output = torch::zeros({batch_size, out_channels, pool2_out_d, pool2_out_h, pool2_out_w}, input.options());
    
    dim3 pool2_block_size(8, 8, 8);
    dim3 pool2_grid_size(
        (pool2_out_w + pool2_block_size.x - 1) / pool2_block_size.x,
        (pool2_out_h + pool2_block_size.y - 1) / pool2_block_size.y,
        (pool2_out_d + pool2_block_size.z - 1) / pool2_block_size.z,
        batch_size
    );
    
    maxpool3d_kernel<<<pool2_grid_size, pool2_block_size>>>(
        pool1_out.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, out_channels, pool1_out_d, pool1_out_h, pool1_out_w,
        pool2_out_d, pool2_out_h, pool2_out_w, pool_kernel_size
    );
    
    return output;
}
"""

fused_conv_softmax_pool_cpp_source = (
    "torch::Tensor fused_conv_softmax_pool_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int kernel_size, int pool_kernel_size);"
)

fused_conv_softmax_pool = load_inline(
    name="fused_conv_softmax_pool",
    cpp_sources=fused_conv_softmax_pool_cpp_source,
    cuda_sources=fused_conv_softmax_pool_source,
    functions=["fused_conv_softmax_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.fused_op = fused_conv_softmax_pool

    def forward(self, x):
        return self.fused_op.fused_conv_softmax_pool_cuda(
            x, self.weight, self.bias, self.kernel_size, self.pool_kernel_size
        )