import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(batch_size: int, out_features: int, block_size: int = 256, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((batch_size, out_features), dtype),
        B: T.Tensor((out_features, out_features), dtype),
        C: T.Tensor((batch_size, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_size), threads=threads) as bx:
            start_idx = bx * block_size
            local_batch = T.min(block_size, batch_size - start_idx)
            
            # Shared memory for reduction
            shared_sum = T.alloc_shared((block_size,), dtype)
            shared_max = T.alloc_shared((block_size,), dtype)
            
            # Initialize thread-local accumulators
            thread_sum = T.alloc_local((1,), dtype)
            thread_max = T.alloc_local((1,), dtype)
            thread_mean = T.alloc_local((1,), dtype)
            thread_lse1 = T.alloc_local((1,), dtype)
            thread_lse2 = T.alloc_local((1,), dtype)
            
            thread_sum[0] = T.cast(0.0, dtype)
            thread_max[0] = T.cast(-1e10, dtype)
            
            # First: Linear + Sum + Max fusion
            for k in T.Parallel(out_features):
                for i in range(local_batch):
                    idx = start_idx + i
                    if idx < batch_size:
                        # Compute matmul row
                        row_sum = T.alloc_local((1,), dtype)
                        row_sum[0] = T.cast(0.0, dtype)
                        
                        for j in range(out_features):
                            row_sum[0] += A[idx, j] * B[j, k]
                        
                        # Update sum and max
                        thread_sum[0] += row_sum[0]
                        thread_max[0] = T.max(thread_max[0], row_sum[0])
            
            # Store to shared memory for reduction
            shared_sum[bx] = thread_sum[0]
            shared_max[bx] = thread_max[0]
            T.tvm_storage_sync("shared")
            
            # Compute mean = sum / out_features
            thread_mean[0] = thread_sum[0] / T.cast(out_features, dtype)
            
            # First LogSumExp: log(mean + exp(max))
            exp_max = T.exp(thread_max[0])
            thread_lse1[0] = T.log(thread_mean[0] + exp_max)
            
            # Second LogSumExp: log(lse1 + exp(max))
            thread_lse2[0] = T.log(thread_lse1[0] + exp_max)
            
            # Store final result
            for i in range(local_batch):
                idx = start_idx + i
                if idx < batch_size:
                    C[idx, 0] = thread_lse2[0]

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._kernel_cache = {}
        self.in_features = in_features
        self.out_features = out_features

    def _get_kernel(self, batch_size: int, tl_dtype: str):
        key = (batch_size, self.out_features, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(batch_size, self.out_features, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get weight matrix from linear layer
        weight = self.linear.weight  # (out_features, in_features)
        
        # Convert to FP16
        x_fp16 = x.half()
        weight_fp16 = weight.half()
        
        batch_size = x_fp16.shape[0]
        kernel = self._get_kernel(batch_size, "float16")
        
        # Allocate output tensor
        output = torch.empty(batch_size, 1, dtype=torch.float16, device=x.device)
        
        # Run fused kernel
        kernel(x_fp16, weight_fp16, output)
        
        return output.float()