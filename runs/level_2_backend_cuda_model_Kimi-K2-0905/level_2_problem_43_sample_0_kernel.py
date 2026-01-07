import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused 3D convolution + max pooling + logsumexp + ReLU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void fused_conv_pool_logsumexp_relu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* workspace,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding) {
    
    const int pool_kernel = 2;
    const int pool_stride = 2;
    const int pooled_depth = out_depth / 2;
    const int pooled_height = out_height / 2;
    const int pooled_width = out_width / 2;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * pooled_depth * pooled_height * pooled_width;
    
    if (idx < total_elements) {
        int pw = idx % pooled_width;
        int ph = (idx / pooled_width) % pooled_height;
        int pd = (idx / (pooled_width * pooled_height)) % pooled_depth;
        int b = idx / (pooled_width * pooled_height * pooled_depth);
        
        float max_val = -FLT_MAX;
        
        // Compute conv for this spatial location across all output channels
        for (int od = 0; od < out_depth; od++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    if (od/2 == pd && oh/2 == ph && ow/2 == pw) {
                        float conv_result = 0.0f;
                        
                        for (int ic = 0; ic < in_channels; ic++) {
                            for (int kd = 0; kd < kernel_size; kd++) {
                                for (int kh = 0; kh < kernel_size; kh++) {
                                    for (int kw = 0; kw < kernel_size; kw++) {
                                        int in_d = od * stride - padding + kd;
                                        int in_h = oh * stride - padding + kh;
                                        int in_w = ow * stride - padding + kw;
                                        
                                        if (in_d >= 0 && in_d < in_depth && 
                                            in_h >= 0 && in_h < in_height && 
                                            in_w >= 0 && in_w < in_width) {
                                            
                                            int in_idx = ((b * in_channels + ic) * in_depth + in_d) * in_height * in_width + in_h * in_width + in_w;
                                            int weight_idx = (ic * out_channels * kernel_size * kernel_size * kernel_size) + 
                                                           (0 * kernel_size * kernel_size * kernel_size) + 
                                                           (kd * kernel_size * kernel_size) + (kh * kernel_size) + kw;
                                            conv_result += input[in_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Add bias (assuming bias is added after conv)
                        if (bias) {
                            conv_result += bias[0];
                        }
                        
                        // Max pooling (simplified for 2x2x2 with stride 2)
                        if (conv_result > max_val) {
                            max_val = conv_result;
                        }
                    }
                }
            }
        }
        
        // LogSumExp across channels (simplified for 1 channel output)
        float logsumexp_val = max_val + logf(1.0f + expf(0.0f - max_val));
        
        // ReLU
        float relu_val = fmaxf(0.0f, logsumexp_val);
        
        int out_idx = b * pooled_depth * pooled_height * pooled_width + pd * pooled_height * pooled_width + ph * pooled_width + pw;
        output[out_idx] = relu_val;
    }
}

torch::Tensor fused_conv_pool_logsumexp_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    auto out_channels = weight.size(0);
    
    int out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (out_width + 2 * padding - kernel_size) / stride + 1;
    
    int pooled_depth = out_depth / 2;
    int pooled_height = out_height / 2;
    int pooled_width = out_width / 2;
    
    auto output = torch::zeros({batch_size, 1, pooled_depth, pooled_height, pooled_width}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * pooled_depth * pooled_height * pooled_width + threads - 1) / threads;
    
    fused_conv_pool_logsumexp_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), nullptr,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_conv_pool_logsumexp_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_pool_logsumexp_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_ops = fused_ops
        
    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return self.fused_ops.fused_conv_pool_logsumexp_relu_cuda(x, weight, bias, self.conv.kernel_size[0], self.conv.stride[0], self.conv.padding[0])

def get_inputs():
    return [torch.rand(4, 32, 32, 128, 128).cuda()]

def get_init_inputs():
    return [32, 64, 3, 1, 1]