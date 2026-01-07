import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv3D + Min + Softmax
fused_conv3d_min_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 8
#define MAX_CHANNELS 32

__global__ void conv3d_min_softmax_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp,
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W, int kernel_size, int dim) {
    
    int b = blockIdx.z;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || oc >= out_channels || hw >= H * W) return;
    
    int h = hw / W;
    int w = hw % W;
    
    float min_val = FLT_MAX;
    
    // Compute convolution for each depth slice and find min
    for (int d = 0; d < D - kernel_size + 1; d++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_size; kd++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_idx = b * in_channels * D * H * W +
                                    ic * D * H * W +
                                    (d + kd) * H * W +
                                    (h + kh) * W +
                                    (w + kw);
                        int weight_idx = oc * in_channels * kernel_size * kernel_size * kernel_size +
                                        ic * kernel_size * kernel_size * kernel_size +
                                        kd * kernel_size * kernel_size +
                                        kh * kernel_size +
                                        kw;
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        min_val = fminf(min_val, sum);
    }
    
    // Store min value in temp for softmax
    int out_idx = b * out_channels * H * W + oc * H * W + hw;
    temp[out_idx] = min_val;
    
    // Synchronize to ensure all threads have computed their min values
    __syncthreads();
    
    // Compute softmax
    // First find max for numerical stability
    float max_val = -FLT_MAX;
    for (int c = 0; c < out_channels; c++) {
        int idx = b * out_channels * H * W + c * H * W + hw;
        max_val = fmaxf(max_val, temp[idx]);
    }
    
    // Compute exp and sum
    float exp_sum = 0.0f;
    for (int c = 0; c < out_channels; c++) {
        int idx = b * out_channels * H * W + c * H * W + hw;
        float exp_val = expf(temp[idx] - max_val);
        temp[idx] = exp_val;
        exp_sum += exp_val;
    }
    
    // Normalize
    output[out_idx] = temp[out_idx] / exp_sum;
}

torch::Tensor fused_conv3d_min_softmax_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int dim) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, H, W}, input.options());
    auto temp = torch::zeros({batch_size, out_channels, H, W}, input.options());
    
    dim3 block_size(16, 4);
    dim3 grid_size((H * W + 15) / 16, (out_channels + 3) / 4, batch_size);
    
    conv3d_min_softmax_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), temp.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        D, H, W, kernel_size, dim);
    
    return output;
}
"""

fused_conv3d_min_softmax_cpp_source = (
    "torch::Tensor fused_conv3d_min_softmax_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int kernel_size, int dim);"
)

# Compile the inline CUDA code
fused_conv3d_min_softmax = load_inline(
    name="fused_conv3d_min_softmax",
    cpp_sources=fused_conv3d_min_softmax_cpp_source,
    cuda_sources=fused_conv3d_min_softmax_source,
    functions=["fused_conv3d_min_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses 3D convolution, minimum operation, and softmax into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.fused_op = fused_conv3d_min_softmax
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        weight = self.conv.weight
        bias = self.conv.bias
        return self.fused_op.fused_conv3d_min_softmax_cuda(x, weight, bias, self.conv.kernel_size[0], self.dim)