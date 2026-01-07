import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(M: int, N: int, K: int, block_M: int = 128, block_N: int = 128, block_K: int = 64, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),  # W is (out, in), so W.T for matmul
        b_matmul: T.Tensor((N,), dtype),
        running_mean: T.Tensor((N,), dtype),
        running_var: T.Tensor((N,), dtype),
        bn_weight: T.Tensor((N,), dtype),
        bn_bias: T.Tensor((N,), dtype),
        bias: T.Tensor((1,), dtype),
        divide_value: T.Tensor((1,), dtype),
        eps: T.Tensor((1,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(X[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(W[bx * block_N : (bx + 1) * block_N, k * block_K : (k + 1) * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            
            for i, j in T.Parallel(block_M, block_N):
                y = by * block_M + i
                x = bx * block_N + j
                if y < M and x < N:
                    # Add matmul bias
                    C_local[i, j] = C_local[i, j] + b_matmul[x]
                    # Batch norm
                    mean = running_mean[x]
                    var = running_var[x]
                    gamma = bn_weight[x]
                    beta = bn_bias[x]
                    C_local[i, j] = (C_local[i, j] - mean) / T.sqrt(var + eps[0]) * gamma + beta
                    # Add self.bias
                    C_local[i, j] = C_local[i, j] + bias[0]
                    # Divide
                    C_local[i, j] = C_local[i, j] / divide_value[0]
                    # Swish
                    C_local[i, j] = C_local[i, j] * T.sigmoid(C_local[i, j])
            
            T.copy(C_local, Y[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])
    
    return tilelang.compile(fused_kernel, out_idx=[10], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that fuses matrix multiplication, batch normalization, bias addition, division, and Swish activation into a single TileLang kernel.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x_c = x.contiguous()
        original_shape = x_c.shape
        x_c = x_c.view(-1, self.in_features)
        
        M, K = x_c.shape
        N = self.out_features
        kernel = self._get_kernel(M, N, K, "float16")
        
        # Convert to half
        x_half = x_c.half()
        W_half = self.matmul.weight.half()
        b_matmul_half = self.matmul.bias.half()
        running_mean_half = self.bn.running_mean.half()
        running_var_half = self.bn.running_var.half()
        bn_weight_half = self.bn.weight.half()
        bn_bias_half = self.bn.bias.half()
        bias_half = self.bias.half()
        divide_value_tensor = torch.tensor([self.divide_value], dtype=torch.float16)
        eps_tensor = torch.tensor([self.bn_eps], dtype=torch.float16)
        
        Y = kernel(x_half, W_half, b_matmul_half, running_mean_half, running_var_half, bn_weight_half, bn_bias_half, bias_half, divide_value_tensor, eps_tensor)
        
        return Y.view(original_shape).float()  # Convert back to float32 for compatibility