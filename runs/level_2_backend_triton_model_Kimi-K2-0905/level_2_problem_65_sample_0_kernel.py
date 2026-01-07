import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_sigmoid_pool_sum_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, pool_kernel_size,
    stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate output dimensions after conv and pooling
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    pooled_height = out_height // pool_kernel_size
    pooled_width = out_width // pool_kernel_size
    
    # Total elements in output tensor per sample
    elements_per_sample = out_channels * pooled_height * pooled_width
    
    # Determine which sample and output element this program handles
    sample_idx = pid // elements_per_sample
    elem_idx = pid % elements_per_sample
    
    if sample_idx >= batch_size:
        return
    
    # Decompose elem_idx into output channel, pooled row, pooled col
    out_c = elem_idx // (pooled_height * pooled_width)
    rem = elem_idx % (pooled_height * pooled_width)
    pooled_row = rem // pooled_width
    pooled_col = rem % pooled_width
    
    if out_c >= out_channels:
        return
    
    # Compute pooling window bounds
    pool_start_row = pooled_row * pool_kernel_size
    pool_end_row = pool_start_row + pool_kernel_size
    pool_start_col = pooled_col * pool_kernel_size
    pool_end_col = pool_start_col + pool_kernel_size
    
    # Initialize accumulator for this output element
    acc = 0.0
    
    # Iterate over pooling window
    for pool_row in range(pool_start_row, pool_end_row):
        for pool_col in range(pool_start_col, pool_end_col):
            # Compute convolution for this position
            conv_acc = 0.0
            
            # Iterate over input channels and kernel
            for in_c in range(in_channels):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        # Input position
                        in_row = pool_row * stride - padding + kh
                        in_col = pool_col * stride - padding + kw
                        
                        # Bounds check
                        if in_row >= 0 and in_row < height and in_col >= 0 and in_col < width:
                            # Load input value
                            input_idx = sample_idx * (in_channels * height * width) + in_c * (height * width) + in_row * width + in_col
                            input_val = tl.load(input_ptr + input_idx)
                            
                            # Load weight
                            weight_idx = out_c * (in_channels * kernel_size * kernel_size) + in_c * (kernel_size * kernel_size) + kh * kernel_size + kw
                            weight_val = tl.load(weight_ptr + weight_idx)
                            
                            conv_acc += input_val * weight_val
            
            # Add bias
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + out_c)
                conv_acc += bias_val
            
            # Apply sigmoid
            sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_acc))
            
            # Accumulate for pooling
            acc += sigmoid_val
    
    # Average pooling
    acc /= (pool_kernel_size * pool_kernel_size)
    
    # Store result
    output_idx = sample_idx * (out_channels * pooled_height * pooled_width) + out_c * (pooled_height * pooled_width) + pooled_row * pooled_width + pooled_col
    tl.store(output_ptr + output_idx, acc)


@triton.jit
def sum_reduce_kernel(
    input_ptr, output_ptr,
    batch_size, out_channels, pooled_height, pooled_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Sum over all spatial dimensions for this sample
    sum_val = 0.0
    elements_per_sample = out_channels * pooled_height * pooled_width
    start_idx = pid * elements_per_sample
    
    for i in range(elements_per_sample):
        val = tl.load(input_ptr + start_idx + i)
        sum_val += val
    
    tl.store(output_ptr + pid, sum_val)


def fused_conv_sigmoid_pool_sum(x, weight, bias, kernel_size, pool_kernel_size, stride=1, padding=0):
    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    pooled_height = out_height // pool_kernel_size
    pooled_width = out_width // pool_kernel_size
    
    # Allocate intermediate tensor
    intermediate = torch.empty(batch_size, out_channels, pooled_height, pooled_width, device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    elements_per_batch = out_channels * pooled_height * pooled_width
    total_elements = batch_size * elements_per_batch
    BLOCK_SIZE = 128
    grid = lambda meta: ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    conv_sigmoid_pool_sum_kernel[grid](
        x, weight, bias, intermediate,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, pool_kernel_size,
        stride, padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Allocate output tensor for final sum
    output = torch.empty(batch_size, device=x.device, dtype=x.dtype)
    
    # Launch reduction kernel
    grid = lambda meta: (batch_size,)
    sum_reduce_kernel[grid](
        intermediate, output,
        batch_size, out_channels, pooled_height, pooled_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Get conv parameters
        weight = self.conv.weight
        bias = self.conv.bias
        padding = self.conv.padding[0]
        
        # Use fused Triton kernel
        return fused_conv_sigmoid_pool_sum(x, weight, bias, self.kernel_size, self.pool_kernel_size, stride=1, padding=padding)