import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d + LayerNorm + GELU + scaling fusion
conv_transpose_ln_gelu_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void conv_transpose_ln_gelu_scale_kernel(
    const float* input, const float* weight, const float* bias,
    const float* ln_weight, const float* ln_bias,
    float* output, float* mean, float* var,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    float eps, float scale) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx < total_size) {
        // Compute output location
        int tmp = idx;
        int w = tmp % out_w; tmp /= out_w;
        int h = tmp % out_h; tmp /= out_h;
        int d = tmp % out_d; tmp /= out_d;
        int c = tmp % out_channels; tmp /= out_channels;
        int n = tmp;

        // Compute input region
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_d; kd++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int in_d_idx = (d * stride_d - pad_d + kd);
                        int in_h_idx = (h * stride_h - pad_h + kh);
                        int in_w_idx = (w * stride_w - pad_w + kw);
                        
                        if (in_d_idx >= 0 && in_d_idx < in_d &&
                            in_h_idx >= 0 && in_h_idx < in_h &&
                            in_w_idx >= 0 && in_w_idx < in_w) {
                            
                            int input_idx = ((n * in_channels + ic) * in_d + in_d_idx) * in_h * in_w + in_h_idx * in_w + in_w_idx;
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
        
        // Store intermediate result for layer norm
        int out_idx = ((n * out_channels + c) * out_d + d) * out_h * out_w + h * out_w + w;
        output[out_idx] = sum;
    }
}

__global__ void compute_mean_var_kernel(
    float* output, float* mean, float* var,
    int batch_size, int out_channels, int out_d, int out_h, int out_w,
    int spatial_size, float eps) {
    
    int n = blockIdx.x;
    int c = blockIdx.y;
    
    if (n < batch_size && c < out_channels) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int d = 0; d < out_d; d++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int idx = ((n * out_channels + c) * out_d + d) * out_h * out_w + h * out_w + w;
                    float val = output[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }
        }
        
        float m = sum / spatial_size;
        float v = sum_sq / spatial_size - m * m;
        
        mean[n * out_channels + c] = m;
        var[n * out_channels + c] = 1.0f / sqrtf(v + eps);
    }
}

__global__ void apply_ln_gelu_scale_kernel(
    float* output, const float* mean, const float* var,
    const float* ln_weight, const float* ln_bias,
    int batch_size, int out_channels, int out_d, int out_h, int out_w,
    int spatial_size, float scale) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx < total_size) {
        int tmp = idx;
        int w = tmp % out_w; tmp /= out_w;
        int h = tmp % out_h; tmp /= out_h;
        int d = tmp % out_d; tmp /= out_d;
        int c = tmp % out_channels; tmp /= out_channels;
        int n = tmp;
        
        float m = mean[n * out_channels + c];
        float inv_std = var[n * out_channels + c];
        
        float val = output[idx];
        val = (val - m) * inv_std;
        
        if (ln_weight != nullptr) {
            val *= ln_weight[c];
        }
        if (ln_bias != nullptr) {
            val += ln_bias[c];
        }
        
        // GELU activation
        float gelu_val = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        
        // Scaling
        output[idx] = gelu_val * scale;
    }
}

torch::Tensor conv_transpose_ln_gelu_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor ln_weight, torch::Tensor ln_bias,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    float eps, float scale) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);
    auto out_channels = weight.size(0);
    
    int out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d;
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_d, out_h, out_w}, input.options());
    auto mean = torch::zeros({batch_size, out_channels}, input.options());
    auto var = torch::zeros({batch_size, out_channels}, input.options());
    
    const int block_size = 256;
    int total_size = batch_size * out_channels * out_d * out_h * out_w;
    int num_blocks = (total_size + block_size - 1) / block_size;
    
    conv_transpose_ln_gelu_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        ln_weight.defined() ? ln_weight.data_ptr<float>() : nullptr,
        ln_bias.defined() ? ln_bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        eps, scale);
    
    int spatial_size = out_d * out_h * out_w;
    dim3 grid(batch_size, out_channels);
    compute_mean_var_kernel<<<grid, 1>>>(
        output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        batch_size, out_channels, out_d, out_h, out_w,
        spatial_size, eps);
    
    apply_ln_gelu_scale_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        ln_weight.defined() ? ln_weight.data_ptr<float>() : nullptr,
        ln_bias.defined() ? ln_bias.data_ptr<float>() : nullptr,
        batch_size, out_channels, out_d, out_h, out_w,
        spatial_size, scale);
    
    return output;
}
"""

conv_transpose_ln_gelu_scale_cpp_source = """
torch::Tensor conv_transpose_ln_gelu_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor ln_weight, torch::Tensor ln_bias,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    float eps, float scale);
"""

conv_transpose_ln_gelu_scale = load_inline(
    name="conv_transpose_ln_gelu_scale",
    cpp_sources=conv_transpose_ln_gelu_scale_cpp_source,
    cuda_sources=conv_transpose_ln_gelu_scale_source,
    functions=["conv_transpose_ln_gelu_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that fuses 3D transposed convolution, layer normalization, GELU activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        
        self.fused_op = conv_transpose_ln_gelu_scale

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.Tensor()
        ln_weight = self.layer_norm.weight
        ln_bias = self.layer_norm.bias
        
        kernel_d, kernel_h, kernel_w = self.conv_transpose.kernel_size
        stride_d, stride_h, stride_w = self.conv_transpose.stride
        pad_d, pad_h, pad_w = self.conv_transpose.padding
        
        return self.fused_op.conv_transpose_ln_gelu_scale_cuda(
            x, weight, bias, ln_weight, ln_bias,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            self.layer_norm.eps, self.scaling_factor
        )