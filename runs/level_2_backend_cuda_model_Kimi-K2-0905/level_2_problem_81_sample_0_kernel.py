import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused gemm + swish + divide + clamp + tanh + clamp
fused_gemm_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void swish_div_clamp_tanh_clamp_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Swish: x * sigmoid(x)
        val = val * (1.0f / (1.0f + expf(-val)));
        // Divide by 2.0
        val = val * 0.5f;
        // Clamp between -1.0 and 1.0
        val = fmaxf(-1.0f, fminf(1.0f, val));
        // Tanh
        val = tanhf(val);
        // Clamp between -1.0 and 1.0 again
        val = fmaxf(-1.0f, fminf(1.0f, val));
        x[idx] = val;
    }
}

torch::Tensor fused_gemm_activation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Perform GEMM: output = input @ weight.T + bias
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
    
    // Add bias if it exists
    if (bias.defined()) {
        // Broadcast bias addition
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    out_features, batch_size, 1,
                    &alpha,
                    bias.data_ptr<float>(), out_features,
                    torch::ones({1, batch_size}, input.options()).data_ptr<float>(), 1,
                    &alpha,
                    output.data_ptr<float>(), out_features);
    }
    
    // Apply fused activation
    int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    swish_div_clamp_tanh_clamp_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), total_elements);
    
    cublasDestroy(handle);
    return output;
}
"""

fused_gemm_activation_cpp_source = (
    "torch::Tensor fused_gemm_activation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_gemm_activation = load_inline(
    name="fused_gemm_activation",
    cpp_sources=fused_gemm_activation_cpp_source,
    cuda_sources=fused_gemm_activation_source,
    functions=["fused_gemm_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.fused_op = fused_gemm_activation

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_op.fused_gemm_activation_cuda(x, self.weight, self.bias if self.bias is not None else torch.Tensor())