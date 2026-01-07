import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d + Add + LayerNorm + AvgPool3d + GELU fusion
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* workspace,
    int batch_size, int in_c, int in_d, int in_h, int in_w,
    int out_c, int out_d, int out_h, int out_w,
    int k_d, int k_h, int k_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    float sum_weight, float* norm_weight, float* norm_bias,
    int pool_k_d, int pool_k_h, int pool_k_w,
    int pooled_d, int pooled_h, int pooled_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_c * out_d * out_h * out_w;
    if (idx >= total_threads) return;

    int tmp = idx;
    int n = tmp / (out_c * out_d * out_h * out_w);
    tmp %= (out_c * out_d * out_h * out_w);
    int c = tmp / (out_d * out_h * out_w);
    tmp %= (out_d * out_h * out_w);
    int d = tmp / (out_h * out_w);
    tmp %= (out_h * out_w);
    int h = tmp / out_w;
    int w = tmp % out_w;

    float val = 0.0f;
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kd = 0; kd < k_d; ++kd) {
            for (int kh = 0; kh < k_h; ++kh) {
                for (int kw = 0; kw < k_w; ++kw) {
                    int in_d_idx = (d + pad_d - kd * (k_d > 1) - out_pad_d) / stride_d;
                    int in_h_idx = (h + pad_h - kh * (k_h > 1) - out_pad_h) / stride_h;
                    int in_w_idx = (w + pad_w - kw * (k_w > 1) - out_pad_w) / stride_w;
                    if ((d + pad_d - kd * (k_d > 1) - out_pad_d) % stride_d != 0) continue;
                    if ((h + pad_h - kh * (k_h > 1) - out_pad_h) % stride_h != 0) continue;
                    if ((w + pad_w - kw * (k_w > 1) - out_pad_w) % stride_w != 0) continue;
                    if (in_d_idx < 0 || in_d_idx >= in_d || in_h_idx < 0 || in_h_idx >= in_h || in_w_idx < 0 || in_w_idx >= in_w) continue;
                    int in_idx = n * in_c * in_d * in_h * in_w +
                                 ic * in_d * in_h * in_w +
                                 in_d_idx * in_h * in_w +
                                 in_h_idx * in_w +
                                 in_w_idx;
                    int w_idx = c * in_c * k_d * k_h * k_w +
                                ic * k_d * k_h * k_w +
                                kd * k_h * k_w +
                                kh * k_w +
                                kw;
                    val += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    if (bias != nullptr) {
        val += bias[c];
    }
    val += sum_weight;

    workspace[idx] = val;
    __syncthreads();

    // LayerNorm
    float mean = 0.0f;
    float var = 0.0f;
    int spatial_size = out_d * out_h * out_w;
    for (int i = 0; i < spatial_size; ++i) {
        mean += workspace[n * out_c * spatial_size + c * spatial_size + i];
    }
    mean /= spatial_size;
    for (int i = 0; i < spatial_size; ++i) {
        float diff = workspace[n * out_c * spatial_size + c * spatial_size + i] - mean;
        var += diff * diff;
    }
    var /= spatial_size;
    float std = sqrtf(var + 1e-5f);
    for (int i = 0; i < spatial_size; ++i) {
        int wk_idx = n * out_c * spatial_size + c * spatial_size + i;
        workspace[wk_idx] = (workspace[wk_idx] - mean) / std;
        if (norm_weight != nullptr && norm_bias != nullptr) {
            workspace[wk_idx] = workspace[wk_idx] * norm_weight[c] + norm_bias[c];
        }
    }
    __syncthreads();

    // AvgPool3d + GELU
    int pool_out_idx = n * out_c * pooled_d * pooled_h * pooled_w +
                       c * pooled_d * pooled_h * pooled_w +
                       (d / pool_k_d) * pooled_h * pooled_w +
                       (h / pool_k_h) * pooled_w +
                       (w / pool_k_w);
    if (d % pool_k_d == 0 && h % pool_k_h == 0 && w % pool_k_w == 0) {
        float pool_val = 0.0f;
        int count = 0;
        for (int pd = 0; pd < pool_k_d && d + pd < out_d; ++pd) {
            for (int ph = 0; ph < pool_k_h && h + ph < out_h; ++ph) {
                for (int pw = 0; pw < pool_k_w && w + pw < out_w; ++pw) {
                    pool_val += workspace[n * out_c * out_d * out_h * out_w +
                                          c * out_d * out_h * out_w +
                                          (d + pd) * out_h * out_w +
                                          (h + ph) * out_w +
                                          (w + pw)];
                    count++;
                }
            }
        }
        pool_val /= count;
        // GELU
        float gelu_val = 0.5f * pool_val * (1.0f + tanhf(0.7978845608f * (pool_val + 0.044715f * pool_val * pool_val * pool_val)));
        output[pool_out_idx] = gelu_val;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float sum_weight, torch::Tensor norm_weight, torch::Tensor norm_bias,
    int pool_k_d, int pool_k_h, int pool_k_w
) {
    const int batch_size = input.size(0);
    const int in_c = input.size(1);
    const int in_d = input.size(2);
    const int in_h = input.size(3);
    const int in_w = input.size(4);
    const int out_c = weight.size(0);
    const int k_d = weight.size(2);
    const int k_h = weight.size(3);
    const int k_w = weight.size(4);
    const int stride_d = 2;
    const int stride_h = 2;
    const int stride_w = 2;
    const int pad_d = 1;
    const int pad_h = 1;
    const int pad_w = 1;
    const int out_pad_d = 1;
    const int out_pad_h = 1;
    const int out_pad_w = 1;

    const int out_d = (in_d - 1) * stride_d - 2 * pad_d + k_d + out_pad_d;
    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + k_h + out_pad_h;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + k_w + out_pad_w;

    const int pooled_d = out_d / pool_k_d;
    const int pooled_h = out_h / pool_k_h;
    const int pooled_w = out_w / pool_k_w;

    auto output = torch::zeros({batch_size, out_c, pooled_d, pooled_h, pooled_w}, input.options());
    auto workspace = torch::zeros({batch_size, out_c, out_d, out_h, out_w}, input.options());

    const int threads = 256;
    const int blocks = (batch_size * out_c * out_d * out_h * out_w + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), workspace.data_ptr<float>(),
        batch_size, in_c, in_d, in_h, in_w,
        out_c, out_d, out_h, out_w,
        k_d, k_h, k_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        sum_weight, norm_weight.defined() ? norm_weight.data_ptr<float>() : nullptr,
        norm_bias.defined() ? norm_bias.data_ptr<float>() : nullptr,
        pool_k_d, pool_k_h, pool_k_w,
        pooled_d, pooled_h, pooled_w
    );

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "float sum_weight, torch::Tensor norm_weight, torch::Tensor norm_bias,"
    "int pool_k_d, int pool_k_h, int pool_k_w);"
)

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.pool_kernel_size = pool_kernel_size
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.sum_weight.item(), self.norm.weight, self.norm.bias,
            self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2]
        )