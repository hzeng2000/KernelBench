import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_features,
    out_features,
    combined_scale,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(out_features, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + (offs_m[:, None] * in_features + offs_k[None, :])
    weight_ptrs = weight_ptr + (offs_k[:, None] * out_features + offs_n[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        a = tl.load(input_ptrs, mask=(offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features), other=0.0)
        b = tl.load(weight_ptrs, mask=(offs_k[:, None] < in_features) & (offs_n[None, :] < out_features), other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_SIZE_K
        input_ptrs += BLOCK_SIZE_K
        weight_ptrs += BLOCK_SIZE_K * out_features

    bias_ptrs = bias_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=offs_n < out_features, other=0.0)
    accumulator += bias[None, :]

    accumulator *= combined_scale

    output_ptrs = output_ptr + offs_m[:, None] * out_features + offs_n[None, :]
    tl.store(output_ptrs, accumulator, mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features))


def triton_fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, combined_scale: float):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_features = x.shape
    out_features = weight.shape[0]

    output = torch.empty((batch_size, out_features), dtype=torch.float32, device=x.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = lambda meta: (
        triton.cdiv(batch_size, meta["BLOCK_SIZE_M"]) * triton.cdiv(out_features, meta["BLOCK_SIZE_N"]),
    )

    fused_linear_kernel[grid](
        x, weight, bias, output,
        batch_size, in_features, out_features, combined_scale,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M
    )
    return output


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, scaling, and residual addition.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        combined_scale = self.scaling_factor + 1.0
        return triton_fused_linear(x, self.matmul.weight, self.matmul.bias, combined_scale)