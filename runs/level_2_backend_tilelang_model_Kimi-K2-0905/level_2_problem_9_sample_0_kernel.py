import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_linear_sub_mul_relu_kernel(batch_size: int, in_features: int, out_features: int,
                                           block_M: int = 64, block_N: int = 64, block_K: int = 32,
                                           threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        subtract_value: T.float32,
        multiply_value: T.float32,
        Y: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(out_features, block_N), T.ceildiv(batch_size, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M

            # Allocate shared memory for tiles
            X_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_K, block_N), dtype)
            Y_local = T.alloc_fragment((block_M, block_N), dtype, accum=True)

            # Initialize accumulator
            for i, j in T.Parallel(block_M, block_N):
                Y_local[i, j] = T.cast(0.0, dtype)

            # Loop over K dimension
            for k in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                start_k = k * block_K

                # Load tile of X into shared memory
                for i, j in T.Parallel(block_M, block_K):
                    if start_m + i < batch_size and start_k + j < in_features:
                        X_shared[i, j] = X[start_m + i, start_k + j]
                    else:
                        X_shared[i, j] = T.cast(0.0, dtype)

                # Load tile of W into shared memory
                for i, j in T.Parallel(block_K, block_N):
                    if start_k + i < in_features and start_n + j < out_features:
                        W_shared[i, j] = W[start_n + j, start_k + i]  # Transposed access
                    else:
                        W_shared[i, j] = T.cast(0.0, dtype)

                # Compute partial results
                for i, j in T.Parallel(block_M, block_N):
                    for kk in T.serial(block_K):
                        Y_local[i, j] += X_shared[i, kk] * W_shared[kk, j]

            # Apply bias, subtract, multiply, and ReLU
            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < batch_size and start_n + j < out_features:
                    val = Y_local[i, j] + B[start_n + j]
                    val = val - T.cast(subtract_value, dtype)
                    val = val * T.cast(multiply_value, dtype)
                    val = T.max(val, T.cast(0.0, dtype))
                    Y[start_m + i, start_n + j] = val

    return tilelang.compile(fused_kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, in_features: int, out_features: int):
        key = (batch_size, in_features, out_features)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_linear_sub_mul_relu_kernel(
                batch_size, in_features, out_features
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        in_features = x.shape[1]
        out_features = self.linear.weight.shape[0]

        # Ensure weights and bias are in fp16
        weight = self.linear.weight.half()
        bias = self.linear.bias.half()

        # Get kernel
        kernel = self._get_kernel(batch_size, in_features, out_features)

        # Allocate output tensor
        output = torch.empty(batch_size, out_features, dtype=torch.float16, device=x.device)

        # Run kernel
        kernel(x.half(), weight, bias, self.subtract_value, self.multiply_value, output)

        return output