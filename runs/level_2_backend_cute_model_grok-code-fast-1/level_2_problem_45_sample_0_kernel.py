import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def bias_sigmoid_kernel(gA: cute.Tensor, gB: cute.Tensor, gBias: cute.Tensor): 
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bdim = cute.arch.block_dim()[0]
    thread_idx = bidx * bdim + tidx
    m, n = gA.shape
    total = m * n
    if thread_idx < total:
        ni = thread_idx % n  
        mi = thread_idx // n  
        val = gA[mi, ni] + gBias[ni]
        gB[mi, ni] = 1.0 / (1.0 + cute.exp(-val))

@cute.kernel
def bias_logsumexp_kernel(gA: cute.Tensor, gB: cute.Tensor, gBias: cute.Tensor): 
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    row = bidx
    col = tidx
    n = gA.shape[1]
    val = gA[row, col] + gBias[col]
    sdata = cute.shared_memory(float, 1024)
    sdata[tidx] = val
    cute.sync()
    for s in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if tidx % (2 * s) == 0 and tidx + s < n:
            sdata[tidx] = max(sdata[tidx], sdata[tidx + s])
        cute.sync()
    max_val = sdata[0]
    sdata[tidx] = cute.exp(val - max_val)
    cute.sync()
    for s in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if tidx % (2 * s) == 0 and tidx + s < n:
            sdata[tidx] += sdata[tidx + s]
        cute.sync()
    sum_val = sdata[0]
    if tidx == 0:
        gB[row] = cute.log(sum_val) + max_val

@cute.jit
def linear1_sigmoid_host(mX: cute.Tensor, mW1: cute.Tensor, mTemp: cute.Tensor, mB1: cute.Tensor, mOut: cute.Tensor):
    cute.ops.gemm(mX, mW1, mTemp)
    M, N = mTemp.shape
    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    bias_sigmoid_kernel(mTemp, mOut, mB1).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.jit
def linear2_logsumexp_host(mIn: cute.Tensor, mW2: cute.Tensor, mTemp: cute.Tensor, mB2: cute.Tensor, mOut: cute.Tensor):
    cute.ops.gemm(mIn, mW2, mTemp)
    batch = mTemp.shape[0]
    output_size = mTemp.shape[1]
    bias_logsumexp_kernel(mTemp, mOut, mB2).launch(grid=(batch, 1, 1), block=(output_size, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w1 = torch.empty(hidden_size, input_size, dtype=torch.float32, device='cuda')
        torch.nn.init.xavier_uniform_(self.w1)
        self.w1_t = self.w1.t().contiguous()
        self.b1 = torch.empty(hidden_size, dtype=torch.float32, device='cuda')
        torch.nn.init.zeros_(self.b1)
        self.w2 = torch.empty(output_size, hidden_size, dtype=torch.float32, device='cuda')
        torch.nn.init.xavier_uniform_(self.w2)
        self.w2_t = self.w2.t().contiguous()
        self.b2 = torch.empty(output_size, dtype=torch.float32, device='cuda')
        torch.nn.init.zeros_(self.b2)
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda()
        batch = x.shape[0]
        temp1 = torch.empty(batch, self.hidden_size, dtype=x.dtype, device=x.device)
        out1 = torch.empty(batch, self.hidden_size, dtype=x.dtype, device=x.device)
        temp2 = torch.empty(batch, self.output_size, dtype=x.dtype, device=x.device)
        out2 = torch.empty(batch, dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW1 = from_dlpack(self.w1_t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mTemp1 = from_dlpack(temp1, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB1 = from_dlpack(self.b1, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mOut1 = from_dlpack(out1, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key1 = (x.dtype,)
        compiled1 = self.compiled.get('linear1', {}).get(key1)
        if compiled1 is None:
            compiled1 = cute.compile(linear1_sigmoid_host, mX, mW1, mTemp1, mB1, mOut1)
            self.compiled.setdefault('linear1', {})[key1] = compiled1
        compiled1(mX, mW1, mTemp1, mB1, mOut1)

        mIn = from_dlpack(out1, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW2 = from_dlpack(self.w2_t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mTemp2 = from_dlpack(temp2, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB2 = from_dlpack(self.b2, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mOut2 = from_dlpack(out2, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key2 = (x.dtype,)
        compiled2 = self.compiled.get('linear2', {}).get(key2)
        if compiled2 is None:
            compiled2 = cute.compile(linear2_logsumexp_host, mIn, mW2, mTemp2, mB2, mOut2)
            self.compiled.setdefault('linear2', {})[key2] = compiled2
        compiled2(mIn, mW2, mTemp2, mB2, mOut2)

        return out2