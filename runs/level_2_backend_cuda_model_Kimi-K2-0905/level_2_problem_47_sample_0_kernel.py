import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv3D + Mish + Tanh
conv3d_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_mish_tanh_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int D, int H, int W, int kernel_size, int stride, int padding,
    int out_D, int out_H, int out_W) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_D * out_H * out_W;
    
    if (idx < total_elements) {
        int w = idx % out_W;
        int h = (idx / out_W) % out_H;
        int d = (idx / (out_W * out_H)) % out_D;
        int c = (idx / (out_W * out_H * out_D)) % out_channels;
        int b = idx / (out_W * out_H * out_D * out_channels);
        
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_size; kd++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_d = d * stride - padding + kd;
                        int in_h = h * stride - padding + kh;
                        int in_w = w * stride - padding + kw;
                        
                        if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                            int in_idx = ((b * in_channels + ic) * D + in_d) * H * W + in_h * W + in_w;
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
        
        // Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        float softplus = logf(1.0f + expf(sum));
        float mish = sum * tanhf(softplus);
        
        // Tanh activation
        output[idx] = tanhf(mish);
    }
}

torch::Tensor conv3d_mish_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto out_channels = weight.size(0);
    
    int out_D = (D + 2 * padding - kernel_size) / stride + 1;
    int out_H = (H + 2 * padding - kernel_size) / stride + 1;
    int out_W = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_D, out_H, out_W}, input.options());
    
    int total_elements = batch_size * out_channels * out_D * out_H * out_W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv3d_mish_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        D, H, W, kernel_size, stride, padding, out_D, out_H, out_W);
    
    return output;
}
"""

conv3d_mish_tanh_cpp_source = (
    "torch::Tensor conv3d_mish_tanh_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code
conv3d_mish_tanh = load_inline(
    name="conv3d_mish_tanh",
    cpp_sources=conv3d_mish_tanh_cpp_source,
    cuda_sources=conv3d_mish_tanh_source,
    functions=["conv3d_mish_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution fused with Mish and Tanh activations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv3d_mish_tanh = conv3d_mish_tanh

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        weight = self.conv.weight
        bias = self.conv.bias
        return self.conv3d_mish_tanh.conv3d_mish_tanh_cuda(x, weight, bias, self.kernel_size, self.stride, self.padding)


batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]