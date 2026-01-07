import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_gemm_activation_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    threads: int = 128,
    dtype: str = "float16"
):
    @T.prim_func
    def fused_gemm_activation_kernel(
        A: T.Tensor((batch_size, in_features), dtype),
        B: T.Tensor((in_features, out_features), dtype),  # B is W.T
        bias: T.Tensor((out_features,), dtype),
        C: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(
            T.ceildiv(batch_size, block_M),
            T.ceildiv(out_features, block_N),
            threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=3):
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=False)

            T.copy(C_local, C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

            # Apply bias and activations
            for i, j in T.Parallel(block_M, block_N):
                m = bx * block_M + i
                n = by * block_N + j
                if m < batch_size and n < out_features:
                    c = C[m, n] + bias[n]
                    # Swish: c * sigmoid(c)
                    sig = 1.0 / (1.0 + T.exp(-c))
                    c = c * sig
                    # Divide by 2
                    c = c / 2.0
                    # Clamp -1 to 1
                    c = T.min(T.max(c, T.float16(-1.0)), T.float16(1.0))
                    # Tanh
                    c = T.tanh(c)
                    # Clamp -1 to 1
                    c = T.min(T.max(c, T.float16(-1.0)), T.float16(1.0))
                    C[m, n] = c

    return tilelang.compile(fused_gemm_activation_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM and activations in a single TileLang kernel.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, in_features: int, out_features: int, tl_dtype: str):
        key = (batch_size, in_features, out_features, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_gemm_activation_kernel(
                batch_size, in_features, out_features, dtype=tl_dtype
            )
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        batch_size, in_features = x.shape
        out_features = self.gemm.out_features
        kernel = self._get_kernel(batch_size, in_features, out_features, "float16")
        W_t = self.gemm.weight.t().contiguous()
        bias = self.gemm.bias.contiguous() if self.gemm.bias is not None else torch.zeros(out_features, dtype=torch.float16, device=x.device)
        C = kernel(x.contiguous(), W_t, bias)
        return C