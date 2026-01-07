import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused avg_pool3d + conv_transpose3d + clamp + softmax + scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int in_c, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int pool_kd, int pool_kh, int pool_kw)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_c * out_d * out_h * out_w;
    if (idx >= total) return;

    int tmp = idx;
    int w = tmp % out_w; tmp /= out_w;
    int h = tmp % out_h; tmp /= out_h;
    int d = tmp % out_d; tmp /= out_d;
    int c = tmp % in_c; tmp /= in_c;
    int b = tmp;

    int start_d = d * pool_kd;
    int start_h = h * pool_kh;
    int start_w = w * pool_kw;
    int end_d = min(start_d + pool_kd, in_d);
    int end_h = min(start_h + pool_kh, in_h);
    int end_w = min(start_w + pool_kw, in_w);

    float sum = 0.0f;
    int count = 0;
    for (int pd = start_d; pd < end_d; ++pd) {
        for (int ph = start_h; ph < end_h; ++ph) {
            for (int pw = start_w; pw < end_w; ++pw) {
                int in_idx = ((b * in_c + c) * in_d + pd) * in_h * in_w + ph * in_w + pw;
                sum += input[in_idx];
                count++;
            }
        }
    }
    output[idx] = (count > 0) ? sum / count : 0.0f;
}

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_c, int out_c,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int k_d, int k_h, int k_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_c * out_d * out_h * out_w;
    if (idx >= total) return;

    int tmp = idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int od = tmp % out_d; tmp /= out_d;
    int oc = tmp % out_c; tmp /= out_c;
    int b = tmp;

    float val = (bias) ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_c; ++ic) {
        for (int kd = 0; kd < k_d; ++kd) {
            for (int kh = 0; kh < k_h; ++kh) {
                for (int kw = 0; kw < k_w; ++kw) {
                    int in_d_idx = od + pad_d - kd * 1;
                    if (in_d_idx % stride_d != 0) continue;
                    in_d_idx /= stride_d;
                    if (in_d_idx < 0 || in_d_idx >= in_d) continue;

                    int in_h_idx = oh + pad_h - kh * 1;
                    if (in_h_idx % stride_h != 0) continue;
                    in_h_idx /= stride_h;
                    if (in_h_idx < 0 || in_h_idx >= in_h) continue;

                    int in_w_idx = ow + pad_w - kw * 1;
                    if (in_w_idx % stride_w != 0) continue;
                    in_w_idx /= stride_w;
                    if (in_w_idx < 0 || in_w_idx >= in_w) continue;

                    int w_idx = ((oc * in_c + ic) * k_d + kd) * k_h * k_w + kh * k_w + kw;
                    int in_idx = ((b * in_c + ic) * in_d + in_d_idx) * in_h * in_w + in_h_idx * in_w + in_w_idx;
                    val += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    output[idx] = val;
}

__global__ void clamp_kernel(float* x, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = x[idx];
        x[idx] = fminf(fmaxf(v, min_val), max_val);
    }
}

__global__ void spatial_softmax_kernel(float* x, int batch_size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels;
    if (idx >= total) return;

    int b = idx / channels;
    int c = idx % channels;
    float* row = x + (b * channels + c) * spatial_size;

    float max_val = -FLT_MAX;
    for (int i = 0; i < spatial_size; ++i) {
        max_val = fmaxf(max_val, row[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < spatial_size; ++i) {
        row[i] = expf(row[i] - max_val);
        sum += row[i];
    }

    for (int i = 0; i < spatial_size; ++i) {
        row[i] /= sum;
    }
}

__global__ void scale_kernel(float* x, const float* scale, int batch_size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * spatial_size;
    if (idx >= total) return;

    int tmp = idx;
    int s = tmp % spatial_size; tmp /= spatial_size;
    int c = tmp % channels; tmp /= channels;
    int b = tmp;

    x[idx] *= scale[c];
}

torch::Tensor fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    int pool_k, int k, int stride, int pad, int out_pad,
    float clamp_min, float clamp_max)
{
    const auto batch_size = x.size(0);
    const auto in_c = x.size(1);
    const auto in_d = x.size(2);
    const auto in_h = x.size(3);
    const auto in_w = x.size(4);

    const auto out_c = weight.size(0);

    const auto pooled_d = in_d / pool_k;
    const auto pooled_h = in_h / pool_k;
    const auto pooled_w = in_w / pool_k;

    auto pooled = torch::zeros({batch_size, in_c, pooled_d, pooled_h, pooled_w}, x.options());

    {
        int total = batch_size * in_c * pooled_d * pooled_h * pooled_w;
        const int block = 256;
        const int grid = (total + block - 1) / block;
        avg_pool3d_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            pooled.data_ptr<float>(),
            batch_size, in_c, in_d, in_h, in_w,
            pooled_d, pooled_h, pooled_w,
            pool_k, pool_k, pool_k);
    }

    const auto out_d = (pooled_d - 1) * stride - 2 * pad + k + out_pad;
    const auto out_h = (pooled_h - 1) * stride - 2 * pad + k + out_pad;
    const auto out_w = (pooled_w - 1) * stride - 2 * pad + k + out_pad;

    auto out = torch::zeros({batch_size, out_c, out_d, out_h, out_w}, x.options());

    {
        int total = batch_size * out_c * out_d * out_h * out_w;
        const int block = 256;
        const int grid = (total + block - 1) / block;
        conv_transpose3d_kernel<<<grid, block>>>(
            pooled.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            out.data_ptr<float>(),
            batch_size, in_c, out_c,
            pooled_d, pooled_h, pooled_w,
            out_d, out_h, out_w,
            k, k, k,
            stride, stride, stride,
            pad, pad, pad,
            out_pad, out_pad, out_pad);
    }

    {
        int size = out.numel();
        const int block = 256;
        const int grid = (size + block - 1) / block;
        clamp_kernel<<<grid, block>>>(out.data_ptr<float>(), size, clamp_min, clamp_max);
    }

    {
        int spatial_size = out_d * out_h * out_w;
        int total = batch_size * out_c;
        const int block = 256;
        const int grid = (total + block - 1) / block;
        spatial_softmax_kernel<<<grid, block>>>(out.data_ptr<float>(), batch_size, out_c, spatial_size);
    }

    {
        int spatial_size = out_d * out_h * out_w;
        int total = batch_size * out_c * spatial_size;
        const int block = 256;
        const int grid = (total + block - 1) / block;
        scale_kernel<<<grid, block>>>(out.data_ptr<float>(), scale.data_ptr<float>(), batch_size, out_c, spatial_size);
    }

    return out;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    int pool_k, int k, int stride, int pad, int out_pad,
    float clamp_min, float clamp_max);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

        self.fused_ops = fused_ops

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.fused_ops.fused_forward_cuda(
            x, self.weight, self.bias, self.scale,
            self.pool_kernel_size, self.kernel_size, self.stride, self.padding, self.output_padding,
            self.clamp_min, self.clamp_max
        )