import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def build_fused_linear_sigmoid_sum_kernel(batch_size: int, hidden_size: int, block_M: int = 32, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, hidden_size), dtype),
        W: T.Tensor((hidden_size, hidden_size), dtype),
        B: T.Tensor((hidden_size,), dtype),
        Out: T.Tensor((batch_size, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_M), threads=threads) as bx:
            start_m = bx * block_M
            sum_shared = T.alloc_shared((block_M,), dtype)
            
            for local_m in T.Parallel(block_M):
                m = start_m + local_m
                if m < batch_size:
                    sum_val = T.alloc_fragment((1,), dtype, 0)
                    
                    for k in T.serial(hidden_size):
                        x_val = X[m, k]
                        w_val = W[k, 0]
                        matmul_val = x_val * w_val
                        sigmoid_val = T.sigmoid(matmul_val + B[0])
                        sum_val[0] = sum_val[0] + sigmoid_val
                    
                    sum_shared[local_m] = sum_val[0]
                    
                    # Reduction within block
                    for stride in T.serial(T.ceildiv(block_M, 2), 1, -1):
                        if local_m < stride and local_m + stride < block_M:
                            sum_shared[local_m] = sum_shared[local_m] + sum_shared[local_m + stride]
                        T.tvm_storage_sync("shared")
                    
                    if local_m == 0:
                        Out[m // block_M, 0] = sum_shared[0]
    
    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self._kernel_cache = {}
        self.input_size = input_size
        self.hidden_size = hidden_size

    def _get_kernel(self, batch_size: int, hidden_size: int, tl_dtype: str):
        key = (batch_size, hidden_size, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_linear_sigmoid_sum_kernel(batch_size, hidden_size, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Convert to fp16 for TileLang kernel
        x_fp16 = x.half()
        weight_fp16 = self.linear.weight.t().half()
        bias_fp16 = self.linear.bias.half()
        
        # Get kernel for this batch size
        kernel = self._get_kernel(batch_size, self.hidden_size, "float16")
        
        # Allocate output tensor
        out = torch.zeros((batch_size, 1), dtype=torch.float16, device=x.device)
        
        # Run fused kernel
        kernel(x_fp16, weight_fp16, bias_fp16, out)
        
        return out.float()