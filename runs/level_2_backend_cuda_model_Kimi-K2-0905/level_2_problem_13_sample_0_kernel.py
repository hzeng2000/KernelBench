import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused operations: mean pooling + bias add + softmax + tanh + scaling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_ops_kernel(const float* input, const float* bias, float* output, 
                                int B, int C, int D, int H, int W, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * H * W;
    
    if (idx < total_elements) {
        int b = idx / (C * H * W);
        int c = (idx / (H * W)) % C;
        int h = (idx / W) % H;
        int w = idx % W;
        
        // Mean pooling over depth
        float sum = 0.0f;
        for (int d = 0; d < D; d++) {
            int input_idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
            sum += input[input_idx];
        }
        float mean_val = sum / D;
        
        // Add bias
        float biased_val = mean_val + bias[c];
        
        // Store in shared memory for softmax computation
        extern __shared__ float shared_data[];
        int tid = threadIdx.x;
        shared_data[tid] = biased_val;
        __syncthreads();
        
        // Compute max for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < blockDim.x; i++) {
            if (shared_data[i] > max_val) max_val = shared_data[i];
        }
        
        // Compute exp and sum
        float exp_val = expf(biased_val - max_val);
        shared_data[tid] = exp_val;
        __syncthreads();
        
        float sum_exp = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum_exp += shared_data[i];
        }
        
        // Softmax
        float softmax_val = exp_val / sum_exp;
        
        // Tanh and scaling
        output[idx] = tanhf(softmax_val) * scaling_factor;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor) {
    auto B = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    
    auto output = torch::zeros({B, C, 1, H, W}, input.options());
    
    int total_elements = B * C * H * W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        B, C, D, H, W, scaling_factor
    );
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for transposed 3D convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int B, int inC, int outC, int D, int H, int W, int kD, int kH, int kW,
    int strideD, int strideH, int strideW, int padD, int padH, int padW,
    int outD, int outH, int outW) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * outC * outD * outH * outW;
    
    if (idx < total_threads) {
        int tmp = idx;
        int b = tmp / (outC * outD * outH * outW);
        tmp %= (outC * outD * outH * outW);
        int c_out = tmp / (outD * outH * outW);
        tmp %= (outD * outH * outW);
        int d_out = tmp / (outH * outW);
        tmp %= (outH * outW);
        int h_out = tmp / outW;
        int w_out = tmp % outW;
        
        float sum = 0.0f;
        
        for (int c_in = 0; c_in < inC; c_in++) {
            for (int kd = 0; kd < kD; kd++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int d_in = d_out - kd + padD;
                        int h_in = h_out - kh + padH;
                        int w_in = w_out - kw + padW;
                        
                        if (d_in >= 0 && d_in % strideD == 0 && h_in >= 0 && h_in % strideH == 0 && w_in >= 0 && w_in % strideW == 0) {
                            d_in /= strideD;
                            h_in /= strideH;
                            w_in /= strideW;
                            
                            if (d_in < D && h_in < H && w_in < W) {
                                int input_idx = b * inC * D * H * W + c_in * D * H * W + d_in * H * W + h_in * W + w_in;
                                int weight_idx = c_out * inC * kD * kH * kW + c_in * kD * kH * kW + kd * kH * kW + kh * kW + kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        
        output[idx] = sum;
    }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int strideD, int strideH, int strideW, int padD, int padH, int padW) {
    auto B = input.size(0);
    auto inC = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto outC = weight.size(1);
    auto kD = weight.size(2);
    auto kH = weight.size(3);
    auto kW = weight.size(4);
    
    int outD = (D - 1) * strideD - 2 * padD + kD;
    int outH = (H - 1) * strideH - 2 * padH + kH;
    int outW = (W - 1) * strideW - 2 * padW + kW;
    
    auto output = torch::zeros({B, outC, outD, outH, outW}, input.options());
    
    int total_elements = B * outC * outD * outH * outW;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), B, inC, outC, D, H, W, kD, kH, kW,
        strideD, strideH, strideW, padD, padH, padW, outD, outH, outW
    );
    
    return output;
}
"""

conv_transpose3d_cpp_source = "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int strideD, int strideH, int strideW, int padD, int padH, int padW);"

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops
        self.conv_transpose3d = conv_transpose3d

    def forward(self, x):
        x = self.conv_transpose3d.conv_transpose3d_cuda(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            1, 1, 1, 1, 1, 1
        )
        x = self.fused_ops.fused_ops_cuda(x, self.bias.squeeze(), self.scaling_factor)
        return x