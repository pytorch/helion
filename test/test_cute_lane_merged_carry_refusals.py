"""Lane-merged loop carries the CuTe backend refuses, checked at codegen.

``acc = hl.zeros([tile_m, bn])`` updated in ``for tile_n in hl.tile(n,
block_size=bn)`` is one scalar per thread on CuTe, which folds the thread's
lanes of ``bn`` (see test_cute_lane_merged_carries.py).  An update that a
thread would add once for all its lanes, or a sum after the loop that adds
the carry's elements in another dtype than the carry's, would see the folded
lanes, so the backend refuses them.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _plus_one(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for _ in hl.tile(n, block_size=bn):
            acc = acc + 1.0
        out[tile_m] = acc.sum(-1)
    return out


def _plus_row(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for _ in hl.tile(n, block_size=bn):
            acc = acc + y[tile_m][:, None]
        out[tile_m] = acc.sum(-1)
    return out


def _partly_broadcast(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    bk = hl.register_block_size(y.size(0))
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn, bk], dtype=x.dtype)
        for tile_n, _ in hl.tile([n, y.size(0)], block_size=[bn, bk]):
            acc = acc + torch.exp(x[tile_m, tile_n])[:, :, None]
        out[tile_m] = acc.sum(2).sum(1)
    return out


def _widened_masked_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            # A padded lane would add 1: the sum reads acc through a mask.
            acc = acc + (x[tile_m, tile_n] + 1)
        out[tile_m] = acc.sum(-1)
    return out


def _widened_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    # acc.sum(-1) of an int32 carry is int64.
    dtype = torch.int64 if x.dtype == torch.int32 else x.dtype
    out = torch.zeros([m], dtype=dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


def _narrowed_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=torch.float16, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1, dtype=torch.float16)
    return out


def _plus_one_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for _ in hl.tile(n, block_size=bn):
            acc = acc + 1.0
        out[tile_m] = acc.amax(-1)
    return out


def _exp_plus_row_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            acc = acc + torch.exp(v) + v.sum(-1, keepdim=True)
        out[tile_m] = acc.sum(-1)
    return out


def _exp_or_row_sum(x: torch.Tensor, beta: float) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            if beta > 0:
                acc = acc + torch.exp(v)
            else:
                acc = acc + v.sum(-1, keepdim=True)
        out[tile_m] = acc.sum(-1)
    return out


def _exp_or_one_then_one(x: torch.Tensor, beta: float) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            if beta > 0:
                acc = acc + torch.exp(x[tile_m, tile_n])
            else:
                acc = acc + 1.0
            acc = acc + 1.0
        out[tile_m] = acc.sum(-1)
    return out


def _code(fn: object, args: tuple[torch.Tensor, ...], config: helion.Config) -> str:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with mock_cuda_unavailable():
        return cpu_bind(kernel, args).to_code(config)


def _config(threads: int, vector: int = 1) -> helion.Config:
    """16-wide tiles of ``bn`` under ``threads`` threads."""
    return helion.Config(
        block_sizes=[16, 1], num_threads=[threads, 1], cute_vector_widths=[vector, 1]
    )


def test_an_update_the_same_in_every_lane_is_refused() -> None:
    """``acc + 1.0`` and a row's value reach no lane loop, so each thread
    would add them once for its four lanes (128 instead of 512 per row), with
    scalar and 4-wide vector lanes.  So would
    ``torch.exp(x[tile_m, tile_n])[:, :, None]`` along ``bk`` in a loop over
    ``tile_n`` and ``tile_k`` whose config folds only ``bk``: the update
    reads the lanes of ``bn`` but not of ``bk`` (8192 instead of 32768 per
    row)."""
    x = torch.empty(64, 512)
    for fn, args in ((_plus_one, (x,)), (_plus_row, (x, torch.empty(64)))):
        for config in (_config(4), _config(4, 4)):
            with pytest.raises(exc.BackendUnsupported, match="same in every lane"):
                _code(fn, args, config)
    partly = helion.Config(block_sizes=[4, 16, 1], num_threads=[4, 4, 1])
    with pytest.raises(exc.BackendUnsupported, match="same in every lane"):
        _code(_partly_broadcast, (x, torch.empty(64)), partly)


def test_a_sum_in_another_dtype_than_the_carry_is_refused() -> None:
    """A sum that widens the carry would have each thread add its columns in
    the carry's dtype first: an int32 carry's sum adds in int64 (four
    columns of 2**30 wrap to 0 in one Int32), also through the mask after
    the loop, and a float16 or bfloat16 carry's in float32; with scalar and
    4-wide vector lanes.  ``acc.sum(-1, dtype=torch.float16)`` of a float32
    carry casts each thread's folded total of four columns, not each column:
    totals of +-20,000 per column fit float16 and cancel in torch, but
    +-80,000 per thread overflow to +-inf.  With a thread per column nothing
    folds, and the narrowing sum compiles."""
    for dtype, added in (
        (torch.int32, "torch.int64"),
        (torch.float16, "torch.float32"),
        (torch.bfloat16, "torch.float32"),
    ):
        args = (torch.empty(64, 512, dtype=dtype),)
        for config in (_config(4), _config(4, 4)):
            with pytest.raises(exc.BackendUnsupported, match=f"adds it in {added}"):
                _code(_widened_sum, args, config)
    with pytest.raises(exc.BackendUnsupported, match="adds it in torch.int64"):
        _code(
            _widened_masked_sum, (torch.empty(64, 512, dtype=torch.int32),), _config(4)
        )
    x = torch.empty(64, 512)
    with pytest.raises(exc.BackendUnsupported, match="casts it to torch.float16"):
        _code(_narrowed_sum, (x,), _config(4))
    assert "cutlass.Float16" in _code(_narrowed_sum, (x,), _config(16))


def test_a_maximum_after_the_loop_is_what_the_refusal_names() -> None:
    """``acc + 1.0`` read by ``acc.amax(-1)`` through the mask after the
    loop: the refusal names the maximum, not a sum."""
    with pytest.raises(exc.BackendUnsupported, match="amax.* reads it after the loop"):
        _code(_plus_one_max, (torch.empty(64, 512),), _config(4))


def test_a_row_sum_in_the_update_is_refused_under_vector_lanes() -> None:
    """``acc + exp(v) + v.sum(-1, keepdim=True)`` at 4-wide vector lanes:
    the row sum would stay in the V-loop and add one V lane of every thread,
    4 of the tile's 16 elements (4 * S instead of 16 * S per row), also with
    the two terms in the branches of an ``if``.  The vector lane loop refuses
    it (hoist_warp_reduce)."""
    x = torch.empty(64, 64)
    for fn, args in ((_exp_plus_row_sum, (x,)), (_exp_or_row_sum, (x, 0.5))):
        with pytest.raises(exc.BackendUnsupported, match="vector lane loop"):
            _code(fn, args, _config(4, 4))


def test_an_update_the_masking_leaves_alone_is_refused() -> None:
    """``acc = acc + 1.0`` after ``if beta > 0: acc = acc + exp(x) else:
    acc = acc + 1.0`` changes the carry after the ``if``; a select of the
    whole update would drop the 1.0s in the padded lanes of the partial tile
    at 500 columns (1000 instead of 1024 per row), so ``mask_loop_carries``
    leaves the update unmasked, and its padded lanes would add their loads'
    fill values to the sum through the mask."""
    with pytest.raises(exc.BackendUnsupported, match="padded lanes of a partial"):
        _code(_exp_or_one_then_one, (torch.empty(64, 500), 0.5), _config(4))
