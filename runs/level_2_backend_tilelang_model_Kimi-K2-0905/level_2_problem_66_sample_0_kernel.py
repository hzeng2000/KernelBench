import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_matmul_dropout_softmax_kernel(batch_size: int, in_features: int, out_features: int, dropout_p: float, block_M: int = 64, block_N: int = 128, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        Y: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_M), threads=threads) as (bx,):
            start_m = bx * block_M
            
            # Shared memory for tile of output
            shared_out = T.alloc_shared((block_M, block_N), dtype)
            # Shared memory for max values per row
            shared_max = T.alloc_shared((block_M,), "float32")
            # Shared memory for sum values per row
            shared_sum = T.alloc_shared((block_M,), "float32")
            
            for local_m in T.Parallel(block_M):
                m = start_m + local_m
                if m < batch_size:
                    # Compute matmul + bias for this row
                    max_val = T.min_value("float32")
                    
                    # First pass: compute matmul and find max for numerical stability
                    for n in range(out_features):
                        acc = T.cast(0.0, "float32")
                        for k in range(in_features):
                            acc += T.cast(X[m, k], "float32") * T.cast(W[n, k], "float32")
                        acc += T.cast(B[n], "float32")
                        
                        # Apply dropout during computation
                        # Simple deterministic dropout using index-based mask
                        dropout_keep = (m * out_features + n) % 100 >= int(dropout_p * 100)
                        if dropout_keep:
                            acc = acc / (1.0 - dropout_p)
                        else:
                            acc = T.cast(0.0, "float32")
                        
                        if acc > max_val:
                            max_val = acc
                    
                    shared_max[local_m] = max_val
                    
                    # Second pass: compute exp and sum
                    sum_val = T.cast(0.0, "float32")
                    for n in range(out_features):
                        acc = T.cast(0.0, "float32")
                        for k in range(in_features):
                            acc += T.cast(X[m, k], "float32") * T.cast(W[n, k], "float32")
                        acc += T.cast(B[n], "float32")
                        
                        # Apply dropout during computation
                        dropout_keep = (m * out_features + n) % 100 >= int(dropout_p * 100)
                        if dropout_keep:
                            acc = acc / (1.0 - dropout_p)
                        else:
                            acc = T.cast(0.0, "float32")
                        
                        exp_val = T.exp(acc - max_val)
                        sum_val += exp_val
                        shared_out[local_m, n] = T.cast(exp_val, dtype)
                    
                    shared_sum[local_m] = sum_val
                    
                    # Third pass: normalize to get softmax
                    for n in range(out_features):
                        softmax_val = T.cast(shared_out[local_m, n], "float32") / sum_val
                        Y[m, n] = T.cast(softmax_val, dtype)

    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, in_features: int, out_features: int, dropout_p: float):
        key = (batch_size, in_features, out_features, dropout_p)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_matmul_dropout_softmax_kernel(
                batch_size, in_features, out_features, dropout_p, dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch_size = x.shape[0]
        
        # Convert to fp16
        x_fp16 = x.half()
        weight_fp16 = self.weight.half()
        bias_fp16 = self.bias.half()
        
        # Get kernel
        kernel = self._get_kernel(batch_size, self.in_features, self.out_features, self.dropout_p)
        
        # Allocate output
        output = torch.empty(batch_size, self.out_features, dtype=torch.float16, device=x.device)
        
        # Run kernel
        kernel(x_fp16, weight_fp16, bias_fp16, output)
        
        return output.float()