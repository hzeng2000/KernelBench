import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_mish_mish_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate output position
    out_hw = height * width
    out_area = out_channels * out_hw
    n = pid // out_area
    c = (pid // out_hw) % out_channels
    hw = pid % out_hw
    h = hw // width
    w = hw % width
    
    if n >= batch_size:
        return
    
    # Calculate input position
    in_h_start = h * stride - padding
    in_w_start = w * stride - padding
    
    acc = 0.0
    
    # Convolution
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                in_h = in_h_start + kh
                in_w = in_w_start + kw
                
                if in_h >= 0 and in_h < height and in_w >= 0 and in_w < width:
                    in_idx = n * in_channels * height * width + ic * height * width + in_h * width + in_w
                    weight_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw
                    
                    in_val = tl.load(input_ptr + in_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    acc += in_val * weight_val
    
    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c)
        acc += bias_val
    
    # First Mish
    x = acc
    softplus = tl.log(1.0 + tl.exp(x))
    tanh_sp = tl.tanh(softplus)
    mish1 = x * tanh_sp
    
    # Second Mish
    softplus2 = tl.log(1.0 + tl.exp(mish1))
    tanh_sp2 = tl.tanh(softplus2)
    mish2 = mish1 * tanh_sp2
    
    # Store output
    out_idx = n * out_channels * height * width + c * height * width + h * width + w
    tl.store(output_ptr + out_idx, mish2)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2

    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Ensure contiguous
        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous() if self.conv.bias is not None else None
        
        # Prepare output
        output = torch.empty(batch_size, self.out_channels, height, width, 
                           dtype=x.dtype, device=x.device)
        
        # Calculate grid
        total_elements = batch_size * self.out_channels * height * width
        BLOCK_SIZE = 128
        
        grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        
        # Launch kernel
        conv_mish_mish_kernel[grid](
            x, weight, bias, output,
            batch_size, self.in_channels, self.out_channels, height, width,
            self.kernel_size, self.stride, self.padding,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output


batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]