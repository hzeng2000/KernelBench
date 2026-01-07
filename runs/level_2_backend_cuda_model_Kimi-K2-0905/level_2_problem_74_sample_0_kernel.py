import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ConvTranspose3d + LeakyReLU + multiply + LeakyReLU + MaxPool3d
fused_3d_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

__global__ void conv_transpose_3d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_c, int out_c,
    int in_d, int in_h, int in_w, int out_d, int out_h, int out_w,
    int k_d, int k_h, int k_w, int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w, int out_pad_d, int out_pad_h, int out_pad_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_c * out_d * out_h * out_w;
    
    if (idx < total_size) {
        int tmp = idx;
        int b = tmp / (out_c * out_d * out_h * out_w);
        tmp %= (out_c * out_d * out_h * out_w);
        int oc = tmp / (out_d * out_h * out_w);
        tmp %= (out_d * out_h * out_w);
        int od = tmp / (out_h * out_w);
        tmp %= (out_h * out_w);
        int oh = tmp / out_w;
        int ow = tmp % out_w;
        
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_c; ic++) {
            for (int kd = 0; kd < k_d; kd++) {
                for (int kh = 0; kh < k_h; kh++) {
                    for (int kw = 0; kw < k_w; kw++) {
                        int in_od = od - kd + pad_d;
                        int in_oh = oh - kh + pad_h;
                        int in_ow = ow - kw + pad_w;
                        
                        if (in_od >= 0 && in_od % stride_d == 0 &&
                            in_oh >= 0 && in_oh % stride_h == 0 &&
                            in_ow >= 0 && in_ow % stride_w == 0) {
                            
                            int in_d_pos = in_od / stride_d;
                            int in_h_pos = in_oh / stride_h;
                            int in_w_pos = in_ow / stride_w;
                            
                            if (in_d_pos < in_d && in_h_pos < in_h && in_w_pos < in_w) {
                                int in_idx = ((b * in_c + ic) * in_d + in_d_pos) * in_h * in_w + in_h_pos * in_w + in_w_pos;
                                int weight_idx = ((oc * in_c + ic) * k_d + kd) * k_h * k_w + kh * k_w + kw;
                                sum += input[in_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        output[idx] = sum;
    }
}

__global__ void leaky_relu_and_multiply_kernel(
    float* data, const float* multiplier, int size, int channels, int d, int h, int w,
    float negative_slope) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int tmp = idx;
        int c = (tmp / (d * h * w)) % channels;
        
        float val = data[idx];
        val = val > 0 ? val : val * negative_slope;
        val *= multiplier[c];
        val = val > 0 ? val : val * negative_slope;
        data[idx] = val;
    }
}

__global__ void max_pool_3d_kernel(
    const float* input, float* output,
    int batch_size, int channels, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w, int k_d, int k_h, int k_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * out_d * out_h * out_w;
    
    if (idx < total_size) {
        int tmp = idx;
        int b = tmp / (channels * out_d * out_h * out_w);
        tmp %= (channels * out_d * out_h * out_w);
        int c = tmp / (out_d * out_h * out_w);
        tmp %= (out_d * out_h * out_w);
        int od = tmp / (out_h * out_w);
        tmp %= (out_h * out_w);
        int oh = tmp / out_w;
        int ow = tmp % out_w;
        
        float max_val = -1e20f;
        
        for (int kd = 0; kd < k_d; kd++) {
            for (int kh = 0; kh < k_h; kh++) {
                for (int kw = 0; kw < k_w; kw++) {
                    int id = od * k_d + kd;
                    int ih = oh * k_h + kh;
                    int iw = ow * k_w + kw;
                    
                    if (id < in_d && ih < in_h && iw < in_w) {
                        int in_idx = ((b * channels + c) * in_d + id) * in_h * in_w + ih * in_w + iw;
                        max_val = fmaxf(max_val, input[in_idx]);
                    }
                }
            }
        }
        
        output[idx] = max_val;
    }
}

torch::Tensor fused_3d_op_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor multiplier) {
    
    auto batch_size = input.size(0);
    auto in_c = input.size(1);
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);
    
    auto out_c = weight.size(0);
    auto k_d = weight.size(2);
    auto k_h = weight.size(3);
    auto k_w = weight.size(4);
    
    int stride_d = 2, stride_h = 2, stride_w = 2;
    int pad_d = 1, pad_h = 1, pad_w = 1;
    int out_pad_d = 1, out_pad_h = 1, out_pad_w = 1;
    
    int out_d = (in_d - 1) * stride_d - 2 * pad_d + k_d + out_pad_d;
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + k_h + out_pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + k_w + out_pad_w;
    
    auto conv_out = torch::zeros({batch_size, out_c, out_d, out_h, out_w}, input.options());
    
    int conv_size = batch_size * out_c * out_d * out_h * out_w;
    int block_size = 256;
    int num_blocks = (conv_size + block_size - 1) / block_size;
    
    conv_transpose_3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        conv_out.data_ptr<float>(), batch_size, in_c, out_c,
        in_d, in_h, in_w, out_d, out_h, out_w,
        k_d, k_h, k_w, stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w, out_pad_d, out_pad_h, out_pad_w);
    
    CUDA_CHECK(cudaGetLastError());
    
    leaky_relu_and_multiply_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(), multiplier.data_ptr<float>(),
        conv_size, out_c, out_d, out_h, out_w, 0.2f);
    
    CUDA_CHECK(cudaGetLastError());
    
    int pool_out_d = out_d / 2;
    int pool_out_h = out_h / 2;
    int pool_out_w = out_w / 2;
    
    auto final_out = torch::zeros({batch_size, out_c, pool_out_d, pool_out_h, pool_out_w}, input.options());
    
    int pool_size = batch_size * out_c * pool_out_d * pool_out_h * pool_out_w;
    int pool_blocks = (pool_size + block_size - 1) / block_size;
    
    max_pool_3d_kernel<<<pool_blocks, block_size>>>(
        conv_out.data_ptr<float>(), final_out.data_ptr<float>(),
        batch_size, out_c, out_d, out_h, out_w,
        pool_out_d, pool_out_h, pool_out_w, 2, 2, 2);
    
    CUDA_CHECK(cudaGetLastError());
    
    return final_out;
}
"""

fused_3d_op_cpp_source = (
    "torch::Tensor fused_3d_op_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor multiplier);"
)

fused_3d_op = load_inline(
    name="fused_3d_op",
    cpp_sources=fused_3d_op_cpp_source,
    cuda_sources=fused_3d_op_source,
    functions=["fused_3d_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.fused_3d_op = fused_3d_op

    def forward(self, x):
        return self.fused_3d_op.fused_3d_op_cuda(x, self.conv_transpose.weight, self.conv_transpose.bias, self.multiplier)