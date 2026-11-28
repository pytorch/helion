"""Masked fragment updates and carried values checked without a GPU."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target as _target
from helion._testing.cute_fragment_simulator import (
    simulate_independent_fragment as _simulate_independent_fragment,
)
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", autotune_effort="none")
def _routed_partial_carry(x):
    rows, columns = x.shape
    hl.specialize(columns)
    chunk = hl.register_block_size(columns)
    out = torch.empty((rows, 2), dtype=x.dtype, device=x.device)
    for row in hl.grid(rows):
        candidates, _ = torch.topk(x[row, :], 4)
        selected, _ = torch.topk(candidates, 2)
        bias = selected.sum(dtype=torch.int32)
        sums = hl.zeros([chunk], dtype=torch.int32)
        state = hl.full([chunk], 1, dtype=torch.int32)
        for tile in hl.tile(columns, block_size=chunk):
            value = x[row, tile]
            sums = sums + (value + 3)
            state = torch.where(value > 0, state * 2 + value, state - 1)
        out[row, 0] = sums.sum() + bias
        out[row, 1] = state.sum() + bias
    return out


def test_fragment_router_keeps_carried_values_on_partial_tile():
    # Each row has two full four-element tiles, then three elements. An
    # unmasked last update would add three to sums and decrement state[3].
    # The successive top-k operations read the complete scalar-indexed row.
    x = (torch.arange(22, dtype=torch.int32).reshape(2, 11) % 7) - 2
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_routed_partial_carry, (x,))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [4]
        source = bound.to_code(config)
    actual = torch.empty((2, 2), dtype=x.dtype)
    _simulate_independent_fragment(source, {"x": x}, {"out": actual}, 2)
    state = torch.ones((2, 4), dtype=x.dtype)
    for i, value in enumerate(x.T):
        lane = state[:, i % 4]
        lane.copy_(torch.where(value > 0, 2 * lane + value, lane - 1))
    expected = torch.stack(((x + 3).sum(1), state.sum(1)), dim=1)
    expected += x.topk(2).values.sum(1, keepdim=True)
    torch.testing.assert_close(actual, expected.to(x.dtype))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _partial_carry_row_max(x):
    rows, columns = x.shape
    chunk = hl.register_block_size(columns)
    rb = hl.register_block_size(rows)
    out = torch.empty(((rows + rb - 1) // rb,), dtype=x.dtype, device=x.device)
    for row in hl.tile(rows, block_size=rb):
        carry = hl.zeros([rb // 2, 2, chunk], dtype=torch.int32)
        for tile in hl.tile(columns, block_size=chunk):
            value = x[row, tile].reshape([rb // 2, 2, chunk])
            carry = ((carry + value + 1) * 2).amax(0, keepdim=True).expand_as(carry)
        out[row.id] = carry.amax(0).sum(0, dtype=torch.int32).sum(0, dtype=torch.int32)
    return out


def test_fragment_partial_rows_do_not_enter_carried_max():
    x = -torch.arange(1, 49, dtype=torch.int32).reshape(6, 8)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_partial_carry_row_max, (x,))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [4, 4]
        source = bound.to_code(config)
    expected = []
    for rows in x.split(4):
        carry = torch.zeros((1, 2, 4), dtype=x.dtype)
        for tile in rows.split(4, dim=1):
            carry = ((carry + tile.reshape(-1, 2, 4) + 1) * 2).amax(0, keepdim=True)
        expected.append(carry.sum().item())
    actual = torch.empty(2, dtype=x.dtype)
    _simulate_independent_fragment(source, {"x": x}, {"out": actual}, 2)
    assert actual.tolist() == expected


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_carried_row(x):
    rows, depth, columns = x.shape
    chunk = hl.register_block_size(columns)
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    for row in hl.grid(rows):
        carry = hl.zeros([depth, chunk], dtype=torch.int32)
        for tile in hl.tile(columns, block_size=chunk):
            carry = (carry.squeeze(0) + x[row, :, tile]) + 1
        out[row] = (
            hl.cumsum(carry, dim=1).sum(0, dtype=torch.int32).sum(0, dtype=torch.int32)
        )
    return out


def test_fragment_partial_update_broadcasts_its_rank_one_fallback():
    x = torch.arange(14, dtype=torch.int32).reshape(2, 1, 7)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_broadcast_carried_row, (x,))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [4]
        source = bound.to_code(config)
    actual = torch.empty(2, dtype=x.dtype)
    _simulate_independent_fragment(source, {"x": x}, {"out": actual}, 2)
    # The second tile has three columns; the fourth keeps the first tile's
    # nonzero value through the rank-one fallback of a rank-two update.
    carried = x[..., :4] + 1
    carried[..., :3] += x[..., 4:] + 1
    torch.testing.assert_close(
        actual, carried.cumsum(dim=-1).sum((1, 2), dtype=x.dtype)
    )
