import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_linear_min_sub_kernel(
    batch_size: int, 
    in_features: int, 
    out_features: int, 
    block_M: int = 64, 
    block_N: int = 64, 
    block_K: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        C: T.Tensor((batch_size, out_features), dtype),
        constant: T.Tensor((1,), dtype),
    ):
        with T.Kernel(T.ceildiv(out_features, block_N), T.ceildiv(batch_size, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M
            
            # Allocate shared memory
            shared_x = T.alloc_shared((block_M, block_K), dtype)
            shared_w = T.alloc_shared((block_K, block_N), dtype)
            
            # Allocate local accumulator
            local_c = T.alloc_fragment((block_M, block_N), dtype, accum=True)
            
            # Initialize accumulator
            for i, j in T.Parallel(block_M, block_N):
                local_c[i, j] = T.cast(0, dtype)
            
            # Main computation loop
            for k in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                # Load input tile to shared memory
                for i, j in T.Parallel(block_M, block_K):
                    m_idx = start_m + i
                    k_idx = k * block_K + j
                    if m_idx < batch_size and k_idx < in_features:
                        shared_x[i, j] = X[m_idx, k_idx]
                    else:
                        shared_x[i, j] = T.cast(0, dtype)
                
                # Load weight tile to shared memory
                for i, j in T.Parallel(block_K, block_N):
                    k_idx = k * block_K + i
                    n_idx = start_n + j
                    if k_idx < in_features and n_idx < out_features:
                        shared_w[i, j] = W[n_idx, k_idx]
                    else:
                        shared_w[i, j] = T.cast(0, dtype)
                
                # Compute matrix multiplication
                for i, j, k_inner in T.Parallel(block_M, block_N, block_K):
                    local_c[i, j] += shared_x[i, k_inner] * shared_w[k_inner, j]
            
            # Add bias, apply min, subtract constant
            for i, j in T.Parallel(block_M, block_N):
                m_idx = start_m + i
                n_idx = start_n + j
                
                if m_idx < batch_size and n_idx < out_features:
                    # Add bias
                    val = local_c[i, j] + B[n_idx]
                    # Apply min with constant
                    val = T.min(val, constant[0])
                    # Subtract constant
                    C[m_idx, n_idx] = val - constant[0]
    
    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self._kernel_cache = {}
        
    def _get_kernel(self, batch_size: int, tl_dtype: str):
        key = (batch_size, self.in_features, self.out_features, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_linear_min_sub_kernel(
                batch_size, self.in_features, self.out_features, dtype=tl_dtype
            )
        return self._kernel_cache[key]
    
    def forward(self, x):
        # Ensure input is contiguous and in fp16
        x = x.contiguous().half()
        
        batch_size = x.shape[0]
        kernel = self._get_kernel(batch_size, "float16")
        
        # Get weight and bias in fp16
        weight = self.linear.weight.t().contiguous().half()
        bias = self.linear.bias.contiguous().half()
        constant_tensor = self.constant.data.half().cuda()
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_features, dtype=torch.float16, device=x.device)
        
        # Run fused kernel
        kernel(x, weight, bias, output, constant_tensor)
        
        return output