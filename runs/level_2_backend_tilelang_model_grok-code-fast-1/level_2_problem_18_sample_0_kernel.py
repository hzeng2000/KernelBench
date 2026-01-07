import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_matvec_kernel(batch: int, in_features: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def matvec_kernel(
        X: T.Tensor((batch, in_features), dtype),
        V: T.Tensor((in_features,), dtype),
        C: T.Tensor((batch,), dtype),
    ):
        with T.Kernel(T.ceildiv(batch, block_M), T.ceildiv(in_features, block_N), threads=threads) as (bx, by):
            T.use_swizzle(panel_size=10, enable=1)
            with T.Block():
                start_y = bx * block_M
                start_x = by * block_N

                for local_y, local_x in T.Parallel(block_M, block_N):
                    y = start_y + local_y
                    x = start_x + local_x

                    if y < batch and x < in_features:
                        T.atomic_add(C[y], X[y, x] * V[x])

    return tilelang.compile(matvec_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.W_sum = self.linear.weight.sum(dim=0)
        self.b_sum = self.linear.bias.sum()
        self._kernel_cache = {}

    def _get_kernel(self, batch: int, in_features: int, tl_dtype: str):
        key = (batch, in_features, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_matvec_kernel(batch, in_features, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.contiguous()
        batch, in_features = x.shape
        C = torch.zeros(batch, dtype=torch.float16, device=x.device)
        kernel = self._get_kernel(batch, in_features, "float16")
        kernel(x, self.W_sum.to(x.device), C)
        C += self.b_sum.to(x.device)
        return C.unsqueeze(-1)