import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_min_tanh_tanh_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gOut: cute.Tensor,
    N: int, C: int, H: int, W: int, K: int, R: int, S: int, P: int, Q: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    # Thread global index
    thread_id = bidz * (bidx * bdimx + tidy) + tidx

    # Output spatial dimensions
    out_h = (H - R + 2 * (R // 2)) + 1
    out_w = (W - S + 2 * (S // 2)) + 1

    # Total output elements per sample: K * out_h * out_w
    elems_per_sample = K * out_h * out_w
    total_out_elems = N * elems_per_sample

    if thread_id >= total_out_elems:
        return

    # Decompose thread_id
    sample = thread_id // elems_per_sample
    rem = thread_id % elems_per_sample
    out_k = rem // (out_h * out_w)
    rem = rem % (out_h * out_w)
    out_y = rem // out_w
    out_x = rem % out_w

    # Compute convolution for this output pixel
    acc = 0.0
    for c in range(C):
        for r in range(R):
            for s in range(S):
                in_y = out_y + r - (R // 2)
                in_x = out_x + s - (S // 2)
                if in_y >= 0 and in_y < H and in_x >= 0 and in_x < W:
                    x_val = gX[sample, c, in_y, in_x]
                    w_val = gW[out_k, c, r, s]
                    acc += x_val * w_val

    # Add bias
    if gB.shape[0] > 0:
        acc += gB[out_k]

    # Store intermediate in shared memory (simulated with global for now)
    # Then reduce min across K dimension
    # We'll do a two-pass approach: first compute all K outputs per spatial, then reduce

    # For now, compute min on the fly
    # We need to sync across K dimension, so we use a different grid strategy

    # Recompute indices for min reduction
    # Each thread handles one spatial location, reduces across K
    spatial_id = sample * (out_h * out_w) + out_y * out_w + out_x
    k_idx = out_k

    # Use atomicMin-style reduction (but for float)
    # Since we can't do atomic float min, we do a block reduction

    # Re-launch with one thread per spatial location
    # This kernel fuses conv + min + tanh + tanh

    # Compute conv value
    val = acc

    # Now reduce min across K dimension for this spatial location
    # We'll use a shared memory reduction per block
    smem = cute.shared.array([64], dtype=cute.float32)  # assumes blockDim.x <= 64
    tid = tidx

    # Each thread loads its conv result
    smem[tid] = val
    cute.arch.sync_threads()

    # Block reduction min
    s = 32
    while s > 0:
        if tid < s and tid + s < K:
            smem[tid] = cute.min(smem[tid], smem[tid + s])
        cute.arch.sync_threads()
        s //= 2

    min_val = smem[0]

    # Only first thread in block writes min result
    if tid == 0:
        # Apply tanh twice
        t1 = cute.tanh(min_val)
        t2 = cute.tanh(t1)
        # Write to output (shape: [N, 1, out_h, out_w])
        gOut[sample, 0, out_y, out_x] = t2

@cute.jit
def conv_min_tanh_tanh_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mOut: cute.Tensor,
    N: int, C: int, H: int, W: int, K: int, R: int, S: int
):
    P = (H - R + 2 * (R // 2)) + 1
    Q = (W - S + 2 * (S // 2)) + 1

    # Launch one block per spatial location, threads reduce across K
    total_spatial = N * P * Q
    threads_per_block = 64  # must match smem size
    grid_x = (total_spatial + threads_per_block - 1) // threads_per_block

    conv_min_tanh_tanh_kernel(mX, mW, mB, mOut, N, C, H, W, K, R, S, P, Q).launch(
        grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1)
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.compiled = {}

    def forward(self, x):
        # Get conv weights and bias
        w = self.conv.weight  # [K, C, R, S]
        b = self.conv.bias    # [K]

        N, C, H, W = x.shape
        K, _, R, S = w.shape
        P = (H - R + 2 * (R // 2)) + 1
        Q = (W - S + 2 * (S // 2)) + 1

        # Allocate output
        out = torch.empty((N, 1, P, Q), dtype=x.dtype, device=x.device)

        # Convert to CuTe tensors
        mX = from_dlpack(x.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(w.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(b.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype, N, C, H, W, K, R, S)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_min_tanh_tanh_host, mX, mW, mB, mOut, N, C, H, W, K, R, S)
            self.compiled[key] = compiled

        compiled(mX, mW, mB, mOut)
        return out