import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_kernel(const float* input, const float* weight, const float* bias, 
                             const float* add_val, float* out, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_features;
    
    if (idx < total_size) {
        int row = idx / out_features;
        int col = idx % out_features;
        
        // Matmul + bias + add_value
        float sum = 0.0f;
        for (int k = 0; k < 8192; k++) {
            sum += input[row * 8192 + k] * weight[col * 8192 + k];
        }
        sum += bias[col] + add_val[col];
        
        // Swish: x * sigmoid(x)
        float sigmoid = 1.0f / (1.0f + expf(-sum));
        float swish = sum * sigmoid;
        
        // Tanh
        float tanh_val = tanhf(swish);
        
        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float gelu_input = tanh_val;
        float pi = 3.14159265359f;
        float sqrt_2_over_pi = sqrtf(2.0f / pi);
        float cube = gelu_input * gelu_input * gelu_input;
        float inner = sqrt_2_over_pi * (gelu_input + 0.044715f * cube);
        float tanh_inner = tanhf(inner);
        float gelu = 0.5f * gelu_input * (1.0f + tanh_inner);
        
        // Hardtanh: clamp between -1 and 1
        out[idx] = fminf(fmaxf(gelu, -1.0f), 1.0f);
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor add_val) {
    auto batch_size = input.size(0);
    auto out_features = weight.size(0);
    auto out = torch::zeros({batch_size, out_features}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;
    
    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_val.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor add_val);"
)

# Compile the inline CUDA code
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
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(x, self.weight, self.bias, self.add_value)

batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]