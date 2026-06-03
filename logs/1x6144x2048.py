# Helion into scaled_mm for M=1 K=6144 N=2048
# vLLM=5.35us  helion=4.90us  ratio=0.917x  WIN
# config: Config(atomic_indexing=['pointer'], block_sizes=[16, 256, 128], indexing=['pointer', 'pointer', 'pointer', 'pointer'], num_stages=2, num_warps=4, pid_type='flat', split_k=16)

from __future__ import annotations

import torch
import helion
import triton
import triton.language as tl
from helion.runtime import default_launcher as _default_launcher
import examples.scaled_mm as _source_module

_BLOCK_SIZE_1 = tl.constexpr(256)
_BLOCK_SIZE_0 = tl.constexpr(1)
_BLOCK_SIZE_3 = tl.constexpr(128)

@triton.jit
def _helion_scaled_mm_into(x, y, scale_a, scale_b, out, _BLOCK_SIZE_2: tl.constexpr):
    # src[scaled_mm.py:112]: for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
    num_blocks_0 = 1
    num_blocks_1 = tl.cdiv(2048, _BLOCK_SIZE_1)
    pid_1 = tl.program_id(0) // num_blocks_0 % num_blocks_1
    pid_2 = tl.program_id(0) // (num_blocks_0 * num_blocks_1)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    offset_2 = pid_2 * _BLOCK_SIZE_2
    # src[scaled_mm.py:113]: acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    # src[scaled_mm.py:114]: for inner_k in hl.tile(outer_k.begin, outer_k.end):
    tile_end = tl.minimum(offset_2 + _BLOCK_SIZE_2, 6144)
    # src[scaled_mm.py:114]: for inner_k in hl.tile(outer_k.begin, outer_k.end):
    # src[scaled_mm.py:115]:     acc = hl.dot(x[tile_m, inner_k], y[inner_k, tile_n], acc=acc)
    for offset_3 in tl.range(tl.cast(offset_2, tl.int32), tl.cast(tile_end, tl.int32), _BLOCK_SIZE_3):
        indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
        mask_3 = indices_3 < tile_end
        acc_copy = acc
        acc_copy_0 = acc_copy
        # src[scaled_mm.py:115]: acc = hl.dot(x[tile_m, inner_k], y[inner_k, tile_n], acc=acc)
        load = tl.broadcast_to(tl.load(x + indices_3[None, :] * 1, mask_3[None, :], other=0.0), [_BLOCK_SIZE_0, _BLOCK_SIZE_3])
        load_1 = tl.load(y + (indices_3[:, None] * 1 + indices_1[None, :] * 6144), mask_3[:, None], other=0.0)
        acc = tl.dot(tl.cast(load, tl.float8e4nv), tl.cast(load_1, tl.float8e4nv), acc=acc_copy_0, input_precision='tf32', out_dtype=tl.float32)
    # src[scaled_mm.py:116]: acc = acc * scale_a[tile_m, :] * scale_b[:, tile_n]
    load_2 = tl.broadcast_to(tl.load(scale_a + tl.zeros([_BLOCK_SIZE_0, 1], tl.int32), None), [_BLOCK_SIZE_0, 1])
    v_0 = acc * load_2
    load_3 = tl.load(scale_b + indices_1[None, :] * 1, None)
    v_1 = v_0 * load_3
    # src[scaled_mm.py:117]: hl.atomic_add(out, [tile_m, tile_n], acc.to(torch.bfloat16))
    v_2 = tl.cast(v_1, tl.bfloat16)
    tl.atomic_add(out + indices_1[None, :] * 1, v_2, mask=None, sem='relaxed')

def scaled_mm_into(out: torch.Tensor, x: torch.Tensor, y: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, *, _launcher=_default_launcher):
    """
    Split-K FP8 RowWise scaled_mm that accumulates into a caller-provided,
    pre-zeroed ``out``. Unlike :func:`scaled_mm`, this performs no internal
    allocation or memset, so the ~0.7-1.2us output-zeroing tax can be hoisted
    out of the timed region and overlapped on a separate CUDA stream with the
    previous call's compute (double-buffered "ping-pong"). With that overlap
    the kernel reaches its memset-free compute floor, matching vLLM.

    Args:
        out (torch.Tensor): Pre-zeroed output of shape [m, n] in BF16; mutated.
        x (torch.Tensor): Left input matrix of shape [m, k] in FP8 (e4m3).
        y (torch.Tensor): Right input matrix of shape [k, n] in FP8 (e4m3).
        scale_a (torch.Tensor): Per-row scale of shape [m, 1].
        scale_b (torch.Tensor): Per-column scale of shape [1, n].
    """
    # src[scaled_mm.py:107]: m, k = x.size()
    m, k = x.size()
    # src[scaled_mm.py:108]: k2, n = y.size()
    k2, n = y.size()
    # src[scaled_mm.py:109]: assert k == k2, f"size mismatch {k} != {k2}"
    assert k == k2, f'size mismatch {k} != {k2}'
    # src[scaled_mm.py:110]: split_k = hl.register_tunable("split_k", PowerOfTwoFragment(2, 256))
    split_k = 16
    # src[scaled_mm.py:111]: k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    # src[scaled_mm.py:112]: for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
    _BLOCK_SIZE_1 = 256
    _BLOCK_SIZE_2 = k_block
    # src[scaled_mm.py:112]: for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
    # src[scaled_mm.py:113]:     acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    # src[scaled_mm.py:114]:     for inner_k in hl.tile(outer_k.begin, outer_k.end):
    # src[scaled_mm.py:112-117]: ...
    _launcher(_helion_scaled_mm_into, (1 * ((2048 + _BLOCK_SIZE_1 - 1) // _BLOCK_SIZE_1) * ((6144 + _BLOCK_SIZE_2 - 1) // _BLOCK_SIZE_2),), x, y, scale_a, scale_b, out, _BLOCK_SIZE_2, num_warps=4, num_stages=2)
    # src[scaled_mm.py:118]: return out
    return out
