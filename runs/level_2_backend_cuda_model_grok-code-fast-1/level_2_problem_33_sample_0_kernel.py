import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + bias + scale
fused_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 16;

__global__ void fused_gemm_kernel(const float* A, const float* B, const float* bias, const float* scale, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B (which is W.t())
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Write result with bias and scale
    if (row < M && col < N) {
        C[row * N + col] = (sum + bias[col]) * scale[col];
    }
}

torch::Tensor fused_gemm_cuda(torch::Tensor A, torch::Tensor W, torch::Tensor bias, torch::Tensor scale) {
    auto Wt = W.t();
    int M = A.size(0);
    int K = A.size(1);
    int N = W.size(0);
    auto C = torch::zeros({M, N}, A.options());

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    fused_gemm_kernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), Wt.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

fused_gemm_cpp_source = (
    "torch::Tensor fused_gemm_cuda(torch::Tensor A, torch::Tensor W, torch::Tensor bias, torch::Tensor scale);"
)

# Compile the inline CUDA code for fused GEMM + bias + scale
fused_gemm = load_inline(
    name="fused_gemm",
    cpp_sources=fused_gemm_cpp_source,
    cuda_sources=fused_gemm_source,
    functions=["fused_gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM, bias addition, and scaling into a single custom CUDA kernel,
    then applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.fused_gemm_op = fused_gemm

    def forward(self, x):
        x = self.fused_gemm_op.fused_gemm_cuda(x, self.gemm.weight, self.gemm.bias, self.scale)
        x = self.bn(x)
        return x