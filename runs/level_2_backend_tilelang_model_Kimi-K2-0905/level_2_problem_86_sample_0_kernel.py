import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_linear_div_gelu_kernel(batch_size: int, input_size: int, output_size: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_linear_div_gelu_kernel(
        X: T.Tensor((batch_size, input_size), dtype),
        W: T.Tensor((output_size, input_size), dtype),
        B: T.Tensor((output_size,), dtype),
        Y: T.Tensor((batch_size, output_size), dtype),
        divisor: T.float32,
    ):
        with T.Kernel(T.ceildiv(batch_size, block_M), T.ceildiv(output_size, block_N), threads=threads) as (bx, by):
            start_m = bx * block_M
            start_n = by * block_N

            # Allocate shared memory for tiles
            X_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_K, block_N), dtype)
            Y_local = T.alloc_fragment((block_M, block_N), dtype, accum=True)

            # Initialize accumulator
            for i, j in T.Parallel(block_M, block_N):
                Y_local[i, j] = T.cast(0.0, dtype)

            # Loop over K dimension
            for k in T.Pipelined(T.ceildiv(input_size, block_K), num_stages=2):
                # Load tile from X
                for i, j in T.Parallel(block_M, block_K):
                    global_i = start_m + i
                    global_j = k * block_K + j
                    if global_i < batch_size and global_j < input_size:
                        X_shared[i, j] = X[global_i, global_j]
                    else:
                        X_shared[i, j] = T.cast(0.0, dtype)

                # Load tile from W (transposed)
                for i, j in T.Parallel(block_K, block_N):
                    global_i = k * block_K + i
                    global_j = start_n + j
                    if global_i < input_size and global_j < output_size:
                        W_shared[i, j] = W[global_j, global_i]
                    else:
                        W_shared[i, j] = T.cast(0.0, dtype)

                # Compute matmul
                for i, j, k_inner in T.Parallel(block_M, block_N, block_K):
                    Y_local[i, j] += X_shared[i, k_inner] * W_shared[k_inner, j]

            # Apply bias, division and GELU
            for i, j in T.Parallel(block_M, block_N):
                global_i = start_m + i
                global_j = start_n + j
                
                if global_i < batch_size and global_j < output_size:
                    # Add bias
                    val = Y_local[i, j] + B[global_j]
                    
                    # Divide by divisor
                    val = val / T.cast(divisor, dtype)
                    
                    # GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    # Using approximate form for efficiency
                    val_fp32 = T.cast(val, "float32")
                    val_cubed = val_fp32 * val_fp32 * val_fp32
                    tanh_arg = T.cast(0.7978845608, "float32") * (val_fp32 + T.cast(0.044715, "float32") * val_cubed)
                    tanh_val = T.tanh(tanh_arg)
                    gelu_val = T.cast(0.5, "float32") * val_fp32 * (T.cast(1.0, "float32") + tanh_val)
                    
                    Y[global_i, global_j] = T.cast(gelu_val, dtype)

    return tilelang.compile(fused_linear_div_gelu_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, input_size: int, output_size: int):
        key = (batch_size, input_size, output_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_linear_div_gelu_kernel(batch_size, input_size, output_size)
        return self._kernel_cache[key]

    def forward(self, x):
        batch_size = x.shape[0]
        input_size = x.shape[1]
        output_size = self.linear.weight.shape[0]
        
        # Convert to fp16
        x_fp16 = x.half()
        weight_fp16 = self.linear.weight.half()
        bias_fp16 = self.linear.bias.half()
        
        # Get kernel
        kernel = self._get_kernel(batch_size, input_size, output_size)
        
        # Allocate output tensor
        output = torch.empty(batch_size, output_size, dtype=torch.float16, device=x.device)
        
        # Run fused kernel
        kernel(x_fp16, weight_fp16, bias_fp16, output, self.divisor)
        
        return output.float()