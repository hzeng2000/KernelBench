import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GEMM + GroupNorm + Min + Bias
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_gemm_groupnorm_min_bias_kernel(
    const float* input, const float* weight, const float* bias_gemm,
    float* output, const float* bias_final,
    int batch_size, int in_features, int out_features,
    int num_groups, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int batch_idx = idx / out_features;
        int out_idx = idx % out_features;
        
        // GEMM computation
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }
        sum += bias_gemm[out_idx];
        
        // GroupNorm computation
        int group_size = out_features / num_groups;
        int group_idx = out_idx / group_size;
        int group_start = group_idx * group_size;
        
        // Compute mean for this group
        float mean = 0.0f;
        for (int i = 0; i < group_size; i++) {
            int elem_idx = batch_idx * out_features + group_start + i;
            mean += output[elem_idx];
        }
        mean /= group_size;
        
        // Compute variance for this group
        float var = 0.0f;
        for (int i = 0; i < group_size; i++) {
            int elem_idx = batch_idx * out_features + group_start + i;
            float diff = output[elem_idx] - mean;
            var += diff * diff;
        }
        var /= group_size;
        
        // Normalize
        sum = (sum - mean) / sqrtf(var + eps);
        
        // Store intermediate result
        output[idx] = sum;
    }
    
    __syncthreads();
    
    // Min operation across dimension 1
    if (threadIdx.x == 0 && blockIdx.x < batch_size) {
        int batch_idx = blockIdx.x;
        float min_val = FLT_MAX;
        for (int i = 0; i < out_features; i++) {
            min_val = fminf(min_val, output[batch_idx * out_features + i]);
        }
        
        // Add bias and store result
        for (int i = 0; i < out_features; i++) {
            output[batch_idx * out_features + i] = min_val + bias_final[i];
        }
    }
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_gemm_groupnorm_min_bias_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias_gemm,
    torch::Tensor bias_final, int num_groups, float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_gemm_groupnorm_min_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.eps = 1e-5
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_gemm = nn.Parameter(torch.randn(out_features))
        self.bias_final = nn.Parameter(torch.randn(bias_shape).squeeze())
        
        self.fused_ops = fused_ops
        
    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        # Launch fused kernel
        threads = 256
        blocks = (batch_size * self.out_features + threads - 1) // threads
        
        self.fused_ops.fused_gemm_groupnorm_min_bias_cuda(
            x, self.weight.t(), self.bias_gemm,
            output, self.bias_final,
            batch_size, self.in_features, self.out_features,
            self.num_groups, self.eps
        )
        
        return output.unsqueeze(2).unsqueeze(3)