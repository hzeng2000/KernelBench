import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused avg_pool + sigmoid + sum
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_avgpool_sigmoid_sum_kernel(const float* x, float* out, int batch_size, int channels, int H, int W, int pool_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_pooled = (H - pool_k) / pool_k + 1;
    int W_pooled = (W - pool_k) / pool_k + 1;
    int total_positions = batch_size * channels * H_pooled * W_pooled;
    if (idx >= total_positions) return;
    
    int pos = idx;
    int total_pooled = H_pooled * W_pooled;
    int b = pos / (channels * total_pooled);
    pos %= (channels * total_pooled);
    int c = pos / total_pooled;
    pos %= total_pooled;
    int i_p = pos / W_pooled;
    int j_p = pos % W_pooled;
    
    // Compute average over the pooling window
    float sum_val = 0.0f;
    int count = 0;
    for (int di = 0; di < pool_k; di++) {
        int i = i_p * pool_k + di;
        if (i >= H) continue;
        for (int dj = 0; dj < pool_k; dj++) {
            int j = j_p * pool_k + dj;
            if (j >= W) continue;
            sum_val += x[((b * channels + c) * H + i) * W + j];
            count++;
        }
    }
    float avg = sum_val / count;
    float sig = 1.0f / (1.0f + expf(-avg));
    
    atomicAdd(&out[b], sig);
}

torch::Tensor fused_avgpool_sigmoid_sum_cuda(torch::Tensor x, int pool_k) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto out = torch::zeros({batch_size}, x.options());
    
    int H_pooled = (H - pool_k) / pool_k + 1;
    int W_pooled = (W - pool_k) / pool_k + 1;
    int total_positions = batch_size * channels * H_pooled * W_pooled;
    
    const int block_size = 256;
    const int num_blocks = (total_positions + block_size - 1) / block_size;
    
    fused_avgpool_sigmoid_sum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, channels, H, W, pool_k);
    
    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_avgpool_sigmoid_sum_cuda(torch::Tensor x, int pool_k);"
)

# Compile the inline CUDA code for fused operation
fused_op = load_inline(
    name="fused_avgpool_sigmoid_sum",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_avgpool_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    This model performs a convolution, then fused average pooling, sigmoid, and sum.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        return self.fused_op.fused_avgpool_sigmoid_sum_cuda(x, self.pool_kernel_size)