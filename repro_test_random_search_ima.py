from __future__ import annotations

import torch
import helion
import triton
import triton.language as tl
from helion.runtime import default_launcher as _default_launcher

helion.runtime.set_triton_allocator()

@triton.jit
def _helion_matmul(x, y, out, _NUM_SM: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(x, [512, 512], [512, 1], [_BLOCK_SIZE_0, _BLOCK_SIZE_2])
    y_desc = tl.make_tensor_descriptor(y, [512, 512], [512, 1], [_BLOCK_SIZE_2, _BLOCK_SIZE_1])
    out_desc = tl.make_tensor_descriptor(out, [512, 512], [512, 1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1])
    total_pids = tl.cdiv(512, _BLOCK_SIZE_0) * tl.cdiv(512, _BLOCK_SIZE_1)
    for virtual_pid in tl.range(tl.program_id(0), total_pids, _NUM_SM, loop_unroll_factor=3, warp_specialize=False, disallow_acc_multi_buffer=True):
        num_pid_m = tl.cdiv(512, _BLOCK_SIZE_0)
        num_pid_n = tl.cdiv(512, _BLOCK_SIZE_1)
        inner_2d_pid = virtual_pid
        num_pid_in_group = 8 * num_pid_n
        group_id = inner_2d_pid // num_pid_in_group
        first_pid_m = group_id * 8
        group_size_m = min(num_pid_m - first_pid_m, 8)
        pid_0 = first_pid_m + inner_2d_pid % num_pid_in_group % group_size_m
        pid_1 = inner_2d_pid % num_pid_in_group // group_size_m
        offset_0 = pid_0 * _BLOCK_SIZE_0
        offset_1 = pid_1 * _BLOCK_SIZE_1
        acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
        for offset_2 in tl.range(0, 512, _BLOCK_SIZE_2, warp_specialize=True, disallow_acc_multi_buffer=True, flatten=True):
            acc_copy = acc
            acc_copy_0 = acc_copy
            load = x_desc.load([offset_0, offset_2])
            load_1 = y_desc.load([offset_2, offset_1])
            acc = tl.dot(tl.cast(load, tl.float32), tl.cast(load_1, tl.float32), acc=acc_copy_0, input_precision='tf32', out_dtype=tl.float32)
        out_desc.store([offset_0, offset_1], acc)

def matmul(x: Tensor, y: Tensor, epilogue: Callable[[Tensor, tuple[Tensor, ...]], Tensor]=lambda acc, tile: acc, *, _launcher=_default_launcher):
    """
    Performs matrix multiplication of x and y with an optional epilogue function.
    Args:
        x (Tensor): Left matrix of shape [m, k].
        y (Tensor): Right matrix of shape [k, n].
        epilogue (Callable, optional): Function applied to the accumulator and tile indices
            after the matmul. Defaults to identity (no change).
    Returns:
        Tensor: Resulting matrix of shape [m, n].
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _NUM_SM = helion.runtime.get_num_sm(x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 64
    _launcher(_helion_matmul, (_NUM_SM,), x, y, out, _NUM_SM, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=1)
    return out

DEVICE = "cuda"
args = (
    torch.randn([512, 512], device=DEVICE),
    torch.randn([512, 512], device=DEVICE),
)
matmul(*args)
