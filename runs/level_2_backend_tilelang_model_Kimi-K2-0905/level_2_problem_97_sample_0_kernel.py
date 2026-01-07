import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(M: int, N: int, block_M: int = 64, block_N: int = 64, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((M, N), dtype),
        Weight: T.Tensor((N, N), dtype),
        Bias: T.Tensor((N,), dtype),
        RunningMean: T.Tensor((N,), "float32"),
        RunningVar: T.Tensor((N,), "float32"),
        Gamma: T.Tensor((N,), "float32"),
        Beta: T.Tensor((N,), "float32"),
        AddBias: T.Tensor((1,), dtype),
        DivideValue: T.Tensor((1,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < M and x < N:
                    # Matmul
                    acc = T.cast(0.0, "float32")
                    for k in range(N):
                        acc += T.cast(X[y, k], "float32") * T.cast(Weight[x, k], "float32")
                    acc += T.cast(Bias[x], "float32")

                    # BatchNorm
                    bn_out = (acc - RunningMean[x]) / T.sqrt(RunningVar[x] + T.cast(1e-5, "float32"))
                    bn_out = bn_out * Gamma[x] + Beta[x]

                    # Add bias, divide, swish
                    bn_out = bn_out + T.cast(AddBias[0], "float32")
                    bn_out = bn_out / T.cast(DivideValue[0], "float32")
                    sigmoid = T.cast(1.0, "float32") / (T.cast(1.0, "float32") + T.exp(-bn_out))
                    swish_out = bn_out * sigmoid

                    Out[y, x] = T.cast(swish_out, dtype)

    return tilelang.compile(fused_kernel, out_idx=[8], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.divide_value = divide_value
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_linear = nn.Parameter(torch.randn(out_features))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value_tensor = torch.tensor([divide_value])
        
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int):
        key = (M, N)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(M, N)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        M, N_in = x_c.shape
        N_out = self.out_features
        
        kernel = self._get_kernel(M, N_out)
        
        # Ensure all tensors are in correct dtype and device
        device = x_c.device
        dtype = torch.float16
        
        x_fp16 = x_c.to(dtype)
        weight_fp16 = self.weight.to(dtype)
        bias_linear_fp16 = self.bias_linear.to(dtype)
        running_mean_fp32 = self.running_mean.to(torch.float32)
        running_var_fp32 = self.running_var.to(torch.float32)
        bn_weight_fp32 = self.bn_weight.to(torch.float32)
        bn_bias_fp32 = self.bn_bias.to(torch.float32)
        bias_fp16 = self.bias.to(dtype)
        divide_fp16 = self.divide_value_tensor.to(dtype)
        
        out = kernel(
            x_fp16,
            weight_fp16,
            bias_linear_fp16,
            running_mean_fp32,
            running_var_fp32,
            bn_weight_fp32,
            bn_bias_fp32,
            bias_fp16,
            divide_fp16
        )
        
        return out