import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_gemm_bn_scale_softmax_kernel(batch_size: int, in_features: int, out_features: int,
                                       block_M: int = 64, block_N: int = 64, block_K: int = 32,
                                       threads: int = 256, dtype: str = "float16"):
    @T.prim_func
    def gemm_bn_scale_softmax_kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        running_mean: T.Tensor((out_features,), "float32"),
        running_var: T.Tensor((out_features,), "float32"),
        gamma: T.Tensor((out_features,), "float32"),
        beta: T.Tensor((out_features,), "float32"),
        scale: T.Tensor((1,), dtype),
        Y: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_M), T.ceildiv(out_features, block_N), threads=threads) as (bx, by):
            start_m = bx * block_M
            start_n = by * block_N

            # Shared memory for tile of output
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            # Local accumulation
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            # Initialize local accumulator
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = 0.0

            # GEMM computation
            for k in T.serial(T.ceildiv(in_features, block_K)):
                # Load input tile
                X_shared = T.alloc_shared((block_M, block_K), dtype)
                W_shared = T.alloc_shared((block_K, block_N), dtype)

                # Load X tile
                for i, kk in T.Parallel(block_M, block_K):
                    if start_m + i < batch_size and k * block_K + kk < in_features:
                        X_shared[i, kk] = X[start_m + i, k * block_K + kk]
                    else:
                        X_shared[i, kk] = 0.0

                # Load W tile
                for kk, j in T.Parallel(block_K, block_N):
                    if k * block_K + kk < in_features and start_n + j < out_features:
                        W_shared[kk, j] = W[start_n + j, k * block_K + kk]
                    else:
                        W_shared[kk, j] = 0.0

                # Compute partial GEMM
                for i, j in T.Parallel(block_M, block_N):
                    for kk in T.serial(block_K):
                        C_local[i, j] += T.cast(X_shared[i, kk], "float32") * T.cast(W_shared[kk, j], "float32")

            # Add bias and convert back to dtype
            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < batch_size and start_n + j < out_features:
                    C_shared[i, j] = T.cast(C_local[i, j] + T.cast(B[start_n + j], "float32"), dtype)

            # BatchNorm + Scale + Softmax
            # First compute max for numerical stability
            max_val = T.alloc_fragment((block_M,), "float32")
            for i in T.Parallel(block_M):
                if start_m + i < batch_size:
                    max_val[i] = -1e9
                    for j in T.serial(block_N):
                        if start_n + j < out_features:
                            # BN: (x - mean) / sqrt(var + eps) * gamma + beta
                            bn_val = (T.cast(C_shared[i, j], "float32") - running_mean[start_n + j]) / T.sqrt(running_var[start_n + j] + 1e-5)
                            bn_val = bn_val * gamma[start_n + j] + beta[start_n + j]
                            # Scale
                            scaled_val = bn_val * T.cast(scale[0], "float32")
                            if scaled_val > max_val[i]:
                                max_val[i] = scaled_val

            # Compute exp and sum
            sum_exp = T.alloc_fragment((block_M,), "float32")
            for i in T.Parallel(block_M):
                if start_m + i < batch_size:
                    sum_exp[i] = 0.0
                    for j in T.serial(block_N):
                        if start_n + j < out_features:
                            bn_val = (T.cast(C_shared[i, j], "float32") - running_mean[start_n + j]) / T.sqrt(running_var[start_n + j] + 1e-5)
                            bn_val = bn_val * gamma[start_n + j] + beta[start_n + j]
                            scaled_val = bn_val * T.cast(scale[0], "float32")
                            exp_val = T.exp(scaled_val - max_val[i])
                            sum_exp[i] += exp_val

            # Compute softmax output
            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < batch_size and start_n + j < out_features:
                    bn_val = (T.cast(C_shared[i, j], "float32") - running_mean[start_n + j]) / T.sqrt(running_var[start_n + j] + 1e-5)
                    bn_val = bn_val * gamma[start_n + j] + beta[start_n + j]
                    scaled_val = bn_val * T.cast(scale[0], "float32")
                    exp_val = T.exp(scaled_val - max_val[i])
                    softmax_val = exp_val / sum_exp[i]
                    Y[start_m + i, start_n + j] = T.cast(softmax_val, dtype)

    return tilelang.compile(gemm_bn_scale_softmax_kernel, out_idx=[7], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)
        
        self._kernel_cache = {}
        self.in_features = in_features
        self.out_features = out_features

    def _get_kernel(self, batch_size: int, in_features: int, out_features: int):
        key = (batch_size, in_features, out_features)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_bn_scale_softmax_kernel(batch_size, in_features, out_features)
        return self._kernel_cache[key]

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Get kernel
        kernel = self._get_kernel(batch_size, self.in_features, self.out_features)
        
        # Get BN parameters
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        
        # Run fused kernel
        output = kernel(
            x.half(),
            self.gemm.weight.half(),
            self.gemm.bias.half(),
            running_mean.float(),
            running_var.float(),
            gamma.float(),
            beta.float(),
            self.scale.half()
        )
        
        return output