import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused average pooling, bias addition, and scaling
fused_avg_pool_bias_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_avg_pool_bias_scale_kernel(const float* input, float* output, const float* bias, float scale2, int batch, int channels, int in_d, int in_h, int in_w, int out_d, int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_d * out_h * out_w;
    if (idx >= total) return;
    
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int od = (idx / (out_w * out_h)) % out_d;
    int c = (idx / (out_w * out_h * out_d)) % channels;
    int b = idx / (out_w * out_h * out_d * channels);
    
    float sum = 0.0f;
    for (int dd = 0; dd < 2; dd++) {
        for (int dh = 0; dh < 2; dh++) {
            for (int dw = 0; dw < 2; dw++) {
                int id = 2 * od + dd;
                int ih = 2 * oh + dh;
                int iw = 2 * ow + dw;
                if (id < in_d && ih < in_h && iw < in_w) {
                    int in_idx = ((b * channels + c) * in_d + id) * in_h * in_w + ih * in_w + iw;
                    sum += input[in_idx];
                }
            }
        }
    }
    float avg = sum / 8.0f;
    avg += bias[c];
    avg *= scale2;
    
    int out_idx = ((b * channels + c) * out_d + od) * out_h * out_w + oh * out_w + ow;
    output[out_idx] = avg;
}

torch::Tensor fused_avg_pool_bias_scale_cuda(torch::Tensor input, torch::Tensor bias, float scale2) {
    int batch = input.size(0);
    int channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);
    int out_d = (in_d - 2) / 2 + 1;
    int out_h = (in_h - 2) / 2 + 1;
    int out_w = (in_w - 2) / 2 + 1;
    auto output = torch::empty({batch, channels, out_d, out_h, out_w}, input.options());
    
    int total = batch * channels * out_d * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_avg_pool_bias_scale_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), bias.data_ptr<float>(), scale2, batch, channels, in_d, in_h, in_w, out_d, out_h, out_w);
    
    return output;
}
"""

fused_avg_pool_bias_scale_cpp_source = (
    "torch::Tensor fused_avg_pool_bias_scale_cuda(torch::Tensor input, torch::Tensor bias, float scale2);"
)

# Compile the inline CUDA code for fused average pooling, bias addition, and scaling
fused_avg_pool_bias_scale = load_inline(
    name="fused_avg_pool_bias_scale",
    cpp_sources=fused_avg_pool_bias_scale_cpp_source,
    cuda_sources=fused_avg_pool_bias_scale_source,
    functions=["fused_avg_pool_bias_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, scaling, fused average pooling with bias addition and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = scale2
        self.fused = fused_avg_pool_bias_scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale1
        x = self.fused.fused_avg_pool_bias_scale_cuda(x, self.bias, self.scale2)
        return x