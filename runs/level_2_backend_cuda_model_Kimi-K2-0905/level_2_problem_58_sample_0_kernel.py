import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_kernel(const float* input, const float* bias, float* output, 
                             int batch_size, int d, int h, int w, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * d * h * w;
    
    if (idx < total_size) {
        int tmp = idx;
        int w_idx = tmp % w;
        tmp /= w;
        int h_idx = tmp % h;
        tmp /= h;
        int d_idx = tmp % d;
        int b_idx = tmp / d;
        
        // Compute LogSumExp across channels
        float max_val = -FLT_MAX;
        for (int c = 0; c < channels; c++) {
            int input_idx = ((b_idx * channels + c) * d + d_idx) * h * w + h_idx * w + w_idx;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            int input_idx = ((b_idx * channels + c) * d + d_idx) * h * w + h_idx * w + w_idx;
            sum_exp += expf(input[input_idx] - max_val);
        }
        
        float logsumexp = logf(sum_exp) + max_val;
        
        // HardSwish activation
        float sigmoid = 1.0f / (1.0f + expf(-logsumexp - 3.0f));
        float hardswish = logsumexp * sigmoid / 6.0f;
        
        // Subtract bias and clamp
        float result = hardswish - bias[0];
        result = fmaxf(-1.0f, fminf(1.0f, result));
        
        int out_idx = ((b_idx * 1) * d + d_idx) * h * w + h_idx * w + w_idx;
        output[out_idx] = result;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto d = input.size(2);
    auto h = input.size(3);
    auto w = input.size(4);
    
    auto output = torch::zeros({batch_size, 1, d, h, w}, input.options());
    
    int total_size = batch_size * d * h * w;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, d, h, w, channels
    );
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias);"

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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias)
        return x