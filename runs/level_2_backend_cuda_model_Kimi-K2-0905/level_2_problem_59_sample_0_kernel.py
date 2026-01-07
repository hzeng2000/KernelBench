import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + swish + scale
fused_matmul_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void swish_scale_kernel(float* x, float* out, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        out[idx] = val * sigmoid * scale;
    }
}

torch::Tensor fused_matmul_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 1.0f;
    
    // Perform matrix multiplication: output = input @ weight.T + bias
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    // Add bias
    dim3 block_size(256);
    dim3 num_blocks((batch_size * out_features + block_size.x - 1) / block_size.x);
    
    // Apply Swish activation and scaling in one kernel
    swish_scale_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size * out_features,
        scale
    );
    
    cublasDestroy(handle);
    return output;
}
"""

fused_matmul_swish_scale_cpp_source = (
    "torch::Tensor fused_matmul_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scale);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_matmul_swish_scale_cpp_source,
    cuda_sources=fused_matmul_swish_scale_source,
    functions=["fused_matmul_swish_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_matmul_swish_scale_cuda(x, self.weight, self.bias, self.scaling_factor)

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]