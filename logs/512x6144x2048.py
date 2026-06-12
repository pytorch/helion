# Helion plain scaled_mm for M=512 K=6144 N=2048
# vLLM=10.94us  helion=15.09us  ratio=1.379x  MISS
# config: Config(block_sizes=[128, 64, 128], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], num_stages=4, num_warps=8, pid_type='persistent_blocked', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[False], split_k=1)

from __future__ import annotations

import torch
import helion
import triton
import triton.language as tl
from helion.runtime import default_launcher as _default_launcher
import examples.scaled_mm as _source_module

_BLOCK_SIZE_0 = tl.constexpr(128)
_BLOCK_SIZE_1 = tl.constexpr(64)
_BLOCK_SIZE_3 = tl.constexpr(128)

@triton.jit
def _helion_scaled_mm(x, y, scale_a, scale_b, out, split_k, _NUM_SM: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    # src[scaled_mm.py:69]: for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
    # src[scaled_mm.py:70]:     acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    # src[scaled_mm.py:71]:     for inner_k in hl.tile(outer_k.begin, outer_k.end):
    # src[scaled_mm.py:69-79]: ...
    total_pids = tl.cdiv(512, _BLOCK_SIZE_0) * tl.cdiv(2048, _BLOCK_SIZE_1) * tl.cdiv(6144, _BLOCK_SIZE_2)
    block_size = tl.cdiv(total_pids, _NUM_SM)
    start_pid = tl.program_id(0) * block_size
    end_pid = start_pid + block_size
    if end_pid > total_pids:
        end_pid = total_pids
    for virtual_pid in tl.range(start_pid, end_pid, warp_specialize=False):
        # src[scaled_mm.py:69]: for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
        num_blocks_0 = tl.cdiv(512, _BLOCK_SIZE_0)
        num_blocks_1 = tl.cdiv(2048, _BLOCK_SIZE_1)
        pid_0 = virtual_pid % num_blocks_0
        pid_1 = virtual_pid // num_blocks_0 % num_blocks_1
        pid_2 = virtual_pid // (num_blocks_0 * num_blocks_1)
        offset_0 = pid_0 * _BLOCK_SIZE_0
        indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
        offset_1 = pid_1 * _BLOCK_SIZE_1
        indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
        offset_2 = pid_2 * _BLOCK_SIZE_2
        # src[scaled_mm.py:70]: acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
        # src[scaled_mm.py:71]: for inner_k in hl.tile(outer_k.begin, outer_k.end):
        tile_end = tl.minimum(offset_2 + _BLOCK_SIZE_2, 6144)
        # src[scaled_mm.py:71]: for inner_k in hl.tile(outer_k.begin, outer_k.end):
        # src[scaled_mm.py:72]:     acc = hl.dot(x[tile_m, inner_k], y[inner_k, tile_n], acc=acc)
        for offset_3 in tl.range(tl.cast(offset_2, tl.int32), tl.cast(tile_end, tl.int32), _BLOCK_SIZE_3):
            indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
            mask_3 = indices_3 < tile_end
            acc_copy = acc
            acc_copy_0 = acc_copy
            # src[scaled_mm.py:72]: acc = hl.dot(x[tile_m, inner_k], y[inner_k, tile_n], acc=acc)
            load = tl.load(x + (indices_0[:, None] * 6144 + indices_3[None, :] * 1), mask_3[None, :], other=0.0)
            load_1 = tl.load(y + (indices_3[:, None] * 1 + indices_1[None, :] * 6144), mask_3[:, None], other=0.0)
            acc = tl.dot(tl.cast(load, tl.float8e4nv), tl.cast(load_1, tl.float8e4nv), acc=acc_copy_0, input_precision='tf32', out_dtype=tl.float32)
        # src[scaled_mm.py:75]: acc = acc * scale_a[tile_m, :] * scale_b[:, tile_n]
        load_2 = tl.load(scale_a + indices_0[:, None] * 1, None)
        v_0 = acc * load_2
        load_3 = tl.load(scale_b + indices_1[None, :] * 1, None)
        v_1 = v_0 * load_3
        # src[scaled_mm.py:76]: if split_k == 1:
        eq = split_k == 1
        # src[scaled_mm.py:76]: if split_k == 1:
        # src[scaled_mm.py:77]:     out[tile_m, tile_n] = acc.to(torch.bfloat16)
        # src[scaled_mm.py:78]: else:
        # src[scaled_mm.py:76-79]: ...
        if eq:
            v_1_copy = v_1
            v_1_copy_0 = v_1_copy
            # src[scaled_mm.py:77]: out[tile_m, tile_n] = acc.to(torch.bfloat16)
            v_2 = tl.cast(v_1_copy_0, tl.bfloat16)
            tl.store(out + (indices_0[:, None] * 2048 + indices_1[None, :] * 1), v_2, None)
        else:
            # src[scaled_mm.py:76]: if split_k == 1:
            # src[scaled_mm.py:77]:     out[tile_m, tile_n] = acc.to(torch.bfloat16)
            # src[scaled_mm.py:78]: else:
            # src[scaled_mm.py:76-79]: ...
            v_1_copy_1 = v_1
            v_1_copy_1_0 = v_1_copy_1
            # src[scaled_mm.py:79]: hl.atomic_add(out, [tile_m, tile_n], acc.to(torch.bfloat16))
            v_3 = tl.cast(v_1_copy_1_0, tl.bfloat16)
            tl.atomic_add(out + (indices_0[:, None] * 2048 + indices_1[None, :] * 1), v_3, mask=None, sem='relaxed')

def scaled_mm(x: torch.Tensor, y: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, *, _launcher=_default_launcher):
    """
    FP8 RowWise scaled matrix multiplication using split-K parallelism.

    Args:
        x (torch.Tensor): Left input matrix of shape [m, k] in FP8 (e4m3).
        y (torch.Tensor): Right input matrix of shape [k, n] in FP8 (e4m3).
        scale_a (torch.Tensor): Per-row scale of shape [m, 1].
        scale_b (torch.Tensor): Per-column scale of shape [1, n].

    Returns:
        torch.Tensor: Output matrix of shape [m, n] in BF16.
    """
    # src[scaled_mm.py:54]: m, k = x.size()
    m, k = x.size()
    # src[scaled_mm.py:55]: k2, n = y.size()
    k2, n = y.size()
    # src[scaled_mm.py:56]: assert k == k2, f"size mismatch {k} != {k2}"
    assert k == k2, f'size mismatch {k} != {k2}'
    # src[scaled_mm.py:57]: split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
    split_k = 1
    # src[scaled_mm.py:64]: if split_k == 1:
    # src[scaled_mm.py:65]:     out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    # src[scaled_mm.py:66]: else:
    # src[scaled_mm.py:64-67]: ...
    if split_k == 1:
        # src[scaled_mm.py:65]: out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
        out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    else:
        # src[scaled_mm.py:67]: out = torch.zeros([m, n], dtype=torch.bfloat16, device=x.device)
        out = torch.zeros([m, n], dtype=torch.bfloat16, device=x.device)
    # src[scaled_mm.py:68]: k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    # src[scaled_mm.py:69]: for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
    _NUM_SM = helion.runtime.get_num_sm(x.device)
    _BLOCK_SIZE_2 = k_block
    # src[scaled_mm.py:69]: for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
    # src[scaled_mm.py:70]:     acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    # src[scaled_mm.py:71]:     for inner_k in hl.tile(outer_k.begin, outer_k.end):
    # src[scaled_mm.py:69-79]: ...
    _launcher(_helion_scaled_mm, (_NUM_SM,), x, y, scale_a, scale_b, out, split_k, _NUM_SM, _BLOCK_SIZE_2, num_warps=8, num_stages=4)
    # src[scaled_mm.py:80]: return out
    return out
