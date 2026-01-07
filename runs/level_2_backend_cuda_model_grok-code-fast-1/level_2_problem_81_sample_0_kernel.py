import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activations: swish, /2, clamp(-1,1), tanh, clamp(-1,1)
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_activation_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = input[idx];
        // swish: y * sigmoid(y)
        float sig = 1.0f / (1.0f + expf(-y));
        y = y * sig;
        // /2
        y = y / 2.0f;
        // clamp -1,1
        y = fminf(fmaxf(y, -1.0f), 1.0f);
        // tanh
        y = tanhf(y);
        // clamp -1,1
        y = fminf(fmaxf(y, -1.0f), 1.0f);
        output[idx] = y;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

fused_activation_cpp_source = (
    "torch::Tensor fused_activation_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for fused activation
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a gemm followed by fused custom CUDA activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.fused_activation = fused_activation

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.fused_activation.fused_activation_cuda(x)
        return x