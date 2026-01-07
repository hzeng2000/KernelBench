import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + multiply + LeakyReLU
fused_gemm_mul_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void leakyrelu_kernel(float* x, int size, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        x[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor fused_gemm_mul_leakyrelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float multiplier, float negative_slope) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::zeros({batch_size, out_features}, input.options());

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);

    if (bias.defined()) {
        auto bias_repeated = bias.unsqueeze(0).repeat({batch_size, 1});
        output = output + bias_repeated;
    }

    int num_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    // Multiply by multiplier
    output = output * multiplier;

    // Apply LeakyReLU
    leakyrelu_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), num_elements, negative_slope);

    cublasDestroy(handle);
    return output;
}
"""

fused_gemm_mul_leakyrelu_cpp_source = (
    "torch::Tensor fused_gemm_mul_leakyrelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float multiplier, float negative_slope);"
)

# Compile the inline CUDA code
fused_gemm_mul_leakyrelu = load_inline(
    name="fused_gemm_mul_leakyrelu",
    cpp_sources=fused_gemm_mul_leakyrelu_cpp_source,
    cuda_sources=fused_gemm_mul_leakyrelu_source,
    functions=["fused_gemm_mul_leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.fused_op = fused_gemm_mul_leakyrelu

    def forward(self, x):
        return self.fused_op.fused_gemm_mul_leakyrelu_cuda(x, self.weight, self.bias, self.multiplier, self.negative_slope)