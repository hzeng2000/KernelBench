import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_linear_instancenorm_kernel(batch_size: int, in_features: int, out_features: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_linear_instancenorm_kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        Y: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(out_features, block_N), T.ceildiv(batch_size, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            # Shared memory for reduction
            shared_mean = T.alloc_shared((block_M,), "float32")
            shared_var = T.alloc_shared((block_M,), "float32")

            # Compute mean for each row
            for local_y in T.Parallel(block_M):
                y = start_y + local_y
                if y < batch_size:
                    sum_val = T.alloc_local((1,), "float32")
                    sum_val[0] = 0.0
                    
                    # Compute sum for this row
                    for local_x in T.serial(block_N):
                        x = start_x + local_x
                        if x < out_features:
                            # Compute linear output
                            linear_sum = T.alloc_local((1,), "float32")
                            linear_sum[0] = 0.0
                            
                            for k in T.serial(in_features):
                                linear_sum[0] += T.cast(X[y, k], "float32") * T.cast(W[x, k], "float32")
                            
                            linear_sum[0] += T.cast(B[x], "float32")
                            sum_val[0] += linear_sum[0]
                    
                    # Store mean in shared memory
                    shared_mean[local_y] = sum_val[0] / T.cast(out_features, "float32")

            # Compute variance for each row
            for local_y in T.Parallel(block_M):
                y = start_y + local_y
                if y < batch_size:
                    sum_sq = T.alloc_local((1,), "float32")
                    sum_sq[0] = 0.0
                    
                    # Compute sum of squared differences
                    for local_x in T.serial(block_N):
                        x = start_x + local_x
                        if x < out_features:
                            # Compute linear output
                            linear_sum = T.alloc_local((1,), "float32")
                            linear_sum[0] = 0.0
                            
                            for k in T.serial(in_features):
                                linear_sum[0] += T.cast(X[y, k], "float32") * T.cast(W[x, k], "float32")
                            
                            linear_sum[0] += T.cast(B[x], "float32")
                            diff = linear_sum[0] - shared_mean[local_y]
                            sum_sq[0] += diff * diff
                    
                    # Store variance in shared memory
                    shared_var[local_y] = sum_sq[0] / T.cast(out_features, "float32")

            # Apply instance normalization and write output
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                
                if y < batch_size and x < out_features:
                    # Compute linear output
                    linear_sum = T.alloc_local((1,), "float32")
                    linear_sum[0] = 0.0
                    
                    for k in T.serial(in_features):
                        linear_sum[0] += T.cast(X[y, k], "float32") * T.cast(W[x, k], "float32")
                    
                    linear_sum[0] += T.cast(B[x], "float32")
                    
                    # Apply instance normalization
                    mean = shared_mean[local_y]
                    var = shared_var[local_y]
                    eps = 1e-5
                    normalized = (linear_sum[0] - mean) / T.sqrt(var + eps)
                    
                    Y[y, x] = T.cast(normalized, dtype)

    return tilelang.compile(fused_linear_instancenorm_kernel, out_idx=[3], target="cuda")


def build_fused_add_mul_kernel(batch_size: int, out_features: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_add_mul_kernel(
        X: T.Tensor((batch_size, out_features), dtype),
        Y: T.Tensor((batch_size, out_features), dtype),
        Z: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(out_features, block_N), T.ceildiv(batch_size, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                
                if y < batch_size and x < out_features:
                    # X = X + Y
                    add_result = T.cast(X[y, x], "float32") + T.cast(Y[y, x], "float32")
                    # X = X * Y
                    mul_result = add_result * T.cast(Y[y, x], "float32")
                    Z[y, x] = T.cast(mul_result, dtype)

    return tilelang.compile(fused_add_mul_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self._kernel_cache1 = {}
        self._kernel_cache2 = {}

    def _get_kernel1(self, batch_size: int, in_features: int, out_features: int):
        key = (batch_size, in_features, out_features)
        if key not in self._kernel_cache1:
            self._kernel_cache1[key] = build_fused_linear_instancenorm_kernel(batch_size, in_features, out_features)
        return self._kernel_cache1[key]

    def _get_kernel2(self, batch_size: int, out_features: int):
        key = (batch_size, out_features)
        if key not in self._kernel_cache2:
            self._kernel_cache2[key] = build_fused_add_mul_kernel(batch_size, out_features)
        return self._kernel_cache2[key]

    def forward(self, x, y):
        batch_size = x.shape[0]
        
        # Get weight and bias from linear layer
        W = self.linear.weight.half()
        B = self.linear.bias.half()
        
        # Fused linear + instance norm
        kernel1 = self._get_kernel1(batch_size, self.in_features, self.out_features)
        x_normalized = kernel1(x.half(), W, B)
        
        # Fused add + mul
        kernel2 = self._get_kernel2(batch_size, self.out_features)
        output = kernel2(x_normalized, y.half())
        
        return output