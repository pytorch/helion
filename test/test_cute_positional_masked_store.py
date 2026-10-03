"""Positional-root torch factories and masked stores on repeated axes.

Every destination is filled with a sentinel and the payloads are nonzero and
position dependent, so a missing, extra, or transposed write is visible.
"""

from __future__ import annotations

import itertools
import re

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")

SENTINEL = -7.0


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def grouped_cartesian_store(
    out: torch.Tensor, group_offsets: torch.Tensor, mode: hl.constexpr
) -> torch.Tensor:
    n = out.size(1)
    for g in hl.grid(group_offsets.size(0) - 1):
        start = group_offsets[g]
        end = group_offsets[g + 1]
        # Rows and columns use independent arange values on one static axis.
        rows = start + hl.arange(8)
        cols = hl.arange(8)
        payload = torch.zeros(8, 8, device=out.device, dtype=torch.float32)
        # Group dependent: a write from the wrong group is visible.
        payload = payload + rows[:, None].float() * 100.0 + cols[None, :].float()
        payload = payload + start.to(torch.float32) * 1000.0
        if mode == "cartesian":
            # Stricter than the host extent: the last column stays untouched.
            mask = (rows < end)[:, None] & (cols < n - 1)[None, :]
        else:
            # Broadcast mask: columns are bounded by the host extent only.
            mask = (rows < end)[:, None]
        hl.store(out, [rows, cols], payload.to(out.dtype), extra_mask=mask)
    return out


def _grouped_cartesian_reference(
    out: torch.Tensor, group_offsets: torch.Tensor, mode: str
) -> torch.Tensor:
    expected = out.clone()
    columns = out.size(1) - 1 if mode == "cartesian" else out.size(1)
    for start, end in itertools.pairwise(group_offsets.tolist()):
        for row in range(start, min(end, start + 8)):
            for col in range(min(8, columns)):
                expected[row, col] = row * 100.0 + col + start * 1000.0
    return expected


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def flagged_block_store(out: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    for g in hl.grid(flags.size(0)):
        rows = g * 8 + hl.arange(8)
        cols = hl.arange(8)
        payload = torch.ones(8, 8, device=out.device, dtype=out.dtype)
        payload = payload * (g + 1) + cols[None, :].to(out.dtype)
        # A 0-d mask: one predicate for the whole block.
        hl.store(out, [rows, cols], payload, extra_mask=flags[g] > 0)
    return out


def _flagged_block_reference(out: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    expected = out.clone()
    cols = torch.arange(8, dtype=out.dtype, device=out.device)
    for g, flag in enumerate(flags.tolist()):
        if flag > 0:
            expected[g * 8 : g * 8 + 8, :8] = (g + 1) + cols[None, :]
    return expected


@helion.kernel(backend="cute", autotune_effort="none")
def grouped_cartesian_store_3d(
    group_offsets: torch.Tensor, total_m: int, n: int, p: int
) -> torch.Tensor:
    # The destination is sentinel-filled and its extents are host scalars.
    out = torch.full(
        (total_m, n, p), SENTINEL, device=group_offsets.device, dtype=torch.float16
    )
    for g in hl.grid(group_offsets.size(0) - 1):
        start = group_offsets[g]
        end = group_offsets[g + 1]
        rows = start + hl.arange(4)
        cols = hl.arange(5)
        depth = hl.arange(6)
        mask = (
            (rows < end)[:, None, None]
            & (cols < n)[None, :, None]
            & (depth < p)[None, None, :]
        )
        payload = torch.ones(4, 5, 6, device=out.device, dtype=out.dtype)
        payload = payload + cols[None, :, None].to(out.dtype) * 8 + depth[None, None, :]
        hl.store(out, [rows, cols, depth], payload, extra_mask=mask)
    return out


def _grouped_cartesian_3d_reference(
    group_offsets: torch.Tensor, total_m: int, n: int, p: int
) -> torch.Tensor:
    expected = torch.full(
        (total_m, n, p), SENTINEL, dtype=torch.float16, device=group_offsets.device
    )
    offsets = group_offsets.tolist()
    for start, end in itertools.pairwise(offsets):
        for row in range(start, min(end, start + 4)):
            for col in range(min(5, n)):
                for d in range(min(6, p)):
                    expected[row, col, d] = 1 + col * 8 + d
    return expected


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def uniform_tail_store(out: torch.Tensor) -> torch.Tensor:
    for g in hl.grid(out.size(0) // 8):
        rows = g * 8 + hl.arange(8)
        cols = hl.arange(5)
        # Uniform payload, no mask: only the exact destination domain keeps
        # the padded columns 5-7 of the index from being written.
        payload = torch.full((8, 5), 3.0, device=out.device, dtype=out.dtype)
        hl.store(out, [rows, cols], payload)
    return out


def _uniform_tail_reference(out: torch.Tensor) -> torch.Tensor:
    expected = out.clone()
    expected[:, :5] = 3.0
    return expected


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def masked_transpose_in_place(x: torch.Tensor) -> torch.Tensor:
    for g in hl.grid(x.size(0)):
        i = hl.arange(8)
        j = hl.arange(8)
        # Value and predicate both read the destination at transposed
        # positions, so they must observe it before any thread's store.
        swapped = x[g, :, :].transpose(-2, -1)
        hl.store(x, [g, i, j], swapped + 1.0, extra_mask=swapped > 0)
    return x


def _masked_transpose_reference(x: torch.Tensor) -> torch.Tensor:
    swapped = x.transpose(-2, -1)
    return torch.where(swapped > 0, swapped + 1.0, x)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def padded_factory_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for g in hl.grid(x.size(0)):
        # ones(8, 5) is traced with a padded 8: its logical length is unknown.
        weights = torch.ones(8, 5, device=x.device, dtype=x.dtype).sum(-1)
        out[g, :, :] = weights[:, None] + x[g, :, :]
    return out


def _guarded(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """A sentinel destination view inside a larger sentinel-filled storage."""
    storage = torch.full(
        (2 * 64 + torch.Size(shape).numel(),), SENTINEL, dtype=dtype, device=DEVICE
    )
    return storage, storage[64 : 64 + torch.Size(shape).numel()].view(shape)


def _cpu_source(kernel, args: tuple[object, ...]) -> str:
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        return bound.to_code(bound.config_spec.default_config())


def _store_predicate(code: str) -> str:
    (line,) = [line for line in code.splitlines() if ").store(" in line]
    index = code.splitlines().index(line)
    return code.splitlines()[index - 1]


@pytest.mark.parametrize("mode", ["cartesian", "rows"])
def test_masked_store_codegen_uses_positional_route(mode: str) -> None:
    out = torch.empty(24, 6)
    offsets = torch.tensor([0, 8, 20, 24], dtype=torch.int32)
    code = _cpu_source(grouped_cartesian_store, (out, offsets, mode))
    assert "ptp_thread" in code
    predicate = _store_predicate(code)
    # Host bounds, then the evaluated mask, guard the only global write.
    assert "< 24" in predicate and "< 6" in predicate
    assert predicate.rstrip(":").split(" and ")[-1].startswith("chain_value")


def test_dynamic_extent_store_bounds_use_host_scalars() -> None:
    offsets = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    code = _cpu_source(grouped_cartesian_store_3d, (offsets, 6, 7, 8))
    assert "ptp_thread" in code
    predicate = _store_predicate(code)
    for extent in ("total_m", "n", "p"):
        assert f"< {extent}" in predicate
    # Padded mask lanes beyond the logical cols/depth are never requested.
    assert "< 5)" in predicate and "< 6)" in predicate


def test_aliasing_mask_and_value_are_snapshotted() -> None:
    code = _cpu_source(masked_transpose_in_place, (torch.empty(3, 8, 8),))
    assert "ptp_thread" in code
    assert code.count("alloc_smem") == 1


def test_padded_factory_extent_fails_closed() -> None:
    with pytest.raises(exc.BackendUnsupported, match="unproven logical extent"):
        _cpu_source(padded_factory_sum, (torch.empty(2, 8, 8),))


@requires_cuda
@pytest.mark.parametrize("mode", ["cartesian", "rows"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_grouped_cartesian_masked_store(mode: str, dtype: torch.dtype) -> None:
    # Groups of 3, 10 and 5 rows: masked-off rows land in other groups or in
    # unowned rows 18-23, and the 10-row group keeps an unwritten tail.
    offsets = torch.tensor([0, 3, 13, 18], dtype=torch.int32, device=DEVICE)
    storage, out = _guarded((24, 6), dtype)
    expected = _grouped_cartesian_reference(out, offsets, mode)
    code, result = code_and_output(grouped_cartesian_store, (out, offsets, mode))
    assert "ptp_thread" in code
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(storage[:64], torch.full_like(storage[:64], SENTINEL))
    torch.testing.assert_close(storage[-64:], torch.full_like(storage[-64:], SENTINEL))


@requires_cuda
def test_zero_dim_mask_store() -> None:
    flags = torch.tensor([1, 0, 1], dtype=torch.int32, device=DEVICE)
    storage, out = _guarded((24, 10), torch.float32)
    expected = _flagged_block_reference(out, flags)
    code, result = code_and_output(flagged_block_store, (out, flags))
    assert "ptp_thread" in code
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(storage[:64], torch.full_like(storage[:64], SENTINEL))
    torch.testing.assert_close(storage[-64:], torch.full_like(storage[-64:], SENTINEL))


@requires_cuda
@pytest.mark.parametrize("n,p", [(4, 3), (7, 8)])
def test_grouped_cartesian_masked_store_3d(n: int, p: int) -> None:
    offsets = torch.tensor([0, 2, 5, 6], dtype=torch.int32, device=DEVICE)
    code, result = code_and_output(grouped_cartesian_store_3d, (offsets, 6, n, p))
    assert "ptp_thread" in code
    torch.testing.assert_close(
        result, _grouped_cartesian_3d_reference(offsets, 6, n, p)
    )


@requires_cuda
def test_uniform_payload_tail_store() -> None:
    storage, out = _guarded((16, 10), torch.float32)
    expected = _uniform_tail_reference(out)
    code, result = code_and_output(uniform_tail_store, (out,))
    assert "ptp_thread" in code
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(storage[:64], torch.full_like(storage[:64], SENTINEL))
    torch.testing.assert_close(storage[-64:], torch.full_like(storage[-64:], SENTINEL))


@requires_cuda
def test_masked_transpose_in_place() -> None:
    x = torch.randn(3, 8, 8, device=DEVICE)
    expected = _masked_transpose_reference(x)
    code, result = code_and_output(masked_transpose_in_place, (x,))
    assert "ptp_thread" in code
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


# Padded destination indexers. A loaded index reads 0 at its padded lanes and
# a loop tile continues past its end while host storage does, so host bounds
# alone would write column/row 0 (or past the loop end) with a uniform payload.
# The square transpose forces the positional root.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def loaded_index_uniform_store(
    out: torch.Tensor,
    idx: torch.Tensor,
    x: torch.Tensor,
    sq: torch.Tensor,
    masked: hl.constexpr,
) -> torch.Tensor:
    for g in hl.grid(out.size(0)):
        cols = idx[:]
        payload = torch.full((5,), 3.0, device=out.device, dtype=out.dtype)
        if masked:
            keep = torch.full((5,), True, device=out.device, dtype=torch.bool)
            hl.store(out, [g, cols], payload, extra_mask=keep)
        else:
            out[g, cols] = payload
        sq[g, :, :] = x[g, :, :].transpose(-2, -1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def loaded_cartesian_uniform_store(
    out: torch.Tensor,
    ridx: torch.Tensor,
    cidx: torch.Tensor,
    x: torch.Tensor,
    sq: torch.Tensor,
) -> torch.Tensor:
    for g in hl.grid(out.size(0)):
        rows = ridx[:]
        cols = cidx[:]
        payload = torch.full((5, 3), 3.0, device=out.device, dtype=out.dtype)
        hl.store(out, [g, rows, cols], payload)
        sq[g, :, :] = x[g, :, :].transpose(-2, -1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def loaded_broadcast_uniform_store(
    out: torch.Tensor,
    ridx: torch.Tensor,
    cidx: torch.Tensor,
    x: torch.Tensor,
    sq: torch.Tensor,
) -> torch.Tensor:
    for g in hl.grid(x.size(0)):
        rows = ridx[:]
        cols = cidx[:]
        payload = torch.full((5, 3), 3.0, device=out.device, dtype=out.dtype)
        hl.store(out, [rows[:, None], cols[None, :]], payload)
        sq[g, :, :] = x[g, :, :].transpose(-2, -1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def loop_tail_uniform_store(
    out: torch.Tensor, x: torch.Tensor, sq: torch.Tensor, n: int
) -> torch.Tensor:
    for g in hl.grid(out.size(0)):
        # The loop ends at n < out.size(1): host storage continues past it.
        for t in hl.tile(n, block_size=8):
            out[g, t] = torch.full((8,), 3.0, device=out.device, dtype=out.dtype)
        sq[g, :, :] = x[g, :, :].transpose(-2, -1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def uniform_indexer_store(
    out: torch.Tensor, idx: torch.Tensor, x: torch.Tensor, sq: torch.Tensor
) -> torch.Tensor:
    for g in hl.grid(out.size(0)):
        # A uniform indexer has no logical length of its own.
        cols = torch.full((5,), 2, device=out.device, dtype=torch.int64)
        out[g, cols] = idx[:].to(out.dtype) + 0.5
        sq[g, :, :] = x[g, :, :].transpose(-2, -1)
    return out


_ROWS, _COLS = [1, 3, 4, 6, 7], [2, 5, 6]


def _padded_index_case(
    name: str, device: torch.device | str
) -> tuple[object, tuple[object, ...], tuple[int, ...], object, torch.Tensor]:
    """(kernel, args after ``out``, out shape, written selector, x)."""
    batch = 1 if name == "broadcast" else 3
    x = torch.randn(batch, 8, 8, device=device)
    sq = torch.full_like(x, SENTINEL)
    idx = torch.tensor([1, 2, 3, 4, 5], device=device)
    rows = torch.tensor(_ROWS, device=device)
    cols = torch.tensor(_COLS, device=device)
    if name in ("loaded", "loaded_mask"):
        args = (idx, x, sq, name == "loaded_mask")
        return loaded_index_uniform_store, args, (3, 8), (slice(None), idx), x
    if name == "cartesian":
        selector = (slice(None), rows[:, None], cols[None, :])
        args = (rows, cols, x, sq)
        return loaded_cartesian_uniform_store, args, (3, 8, 8), selector, x
    if name == "broadcast":
        selector = (rows[:, None], cols[None, :])
        args = (rows, cols, x, sq)
        return loaded_broadcast_uniform_store, args, (8, 8), selector, x
    assert name == "loop_tail"
    selector = (slice(None), slice(0, 5))
    return loop_tail_uniform_store, (x, sq, 5), (3, 16), selector, x


_PADDED_INDEX_CASES = ["loaded", "loaded_mask", "cartesian", "broadcast", "loop_tail"]


@pytest.mark.parametrize(
    "name,bound",
    [
        ("loaded", r"\bptp_c0_2 < 5(?=[ )])"),
        ("cartesian", r"\bptp_c1_3 < 3(?=[ )])"),
        ("loop_tail", r"< ptp_end_1\)"),
    ],
)
def test_padded_destination_index_is_bounded(name: str, bound: str) -> None:
    kernel, args, shape, _, _ = _padded_index_case(name, "cpu")
    code = _cpu_source(kernel, (torch.empty(shape), *args))
    assert "ptp_thread" in code
    # The first store is the padded-index one; the transpose store follows.
    lines = code.splitlines()
    first = next(i for i, line in enumerate(lines) if ").store(" in line)
    # Boolean reassociation can remove redundant parentheses around a bound.
    assert re.search(bound, lines[first - 1])


def test_uniform_destination_indexer_fails_closed() -> None:
    args = (torch.empty(3, 8), torch.arange(5), torch.empty(3, 8, 8))
    with pytest.raises(exc.BackendUnsupported, match="unproven logical extent"):
        _cpu_source(uniform_indexer_store, (*args, torch.empty(3, 8, 8)))


@requires_cuda
@pytest.mark.parametrize("name", _PADDED_INDEX_CASES)
def test_padded_destination_index_uniform_store(name: str) -> None:
    kernel, args, shape, selector, x = _padded_index_case(name, DEVICE)
    sq = next(
        a for a in args if isinstance(a, torch.Tensor) and a.dim() == 3 and a is not x
    )
    storage, out = _guarded(shape, torch.float32)
    expected = out.clone()
    expected[selector] = 3.0
    code, result = code_and_output(kernel, (out, *args))
    assert "ptp_thread" in code
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(sq, x.transpose(-1, -2), rtol=0, atol=0)
    torch.testing.assert_close(storage[:64], torch.full_like(storage[:64], SENTINEL))
    torch.testing.assert_close(storage[-64:], torch.full_like(storage[-64:], SENTINEL))
