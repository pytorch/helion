"""GPU numerics for the CuTe ``hl.associative_scan`` lowering.

``test_cute_scan_lowering.py`` pins down (without a GPU) which configs take
the register / warp-shuffle paths and which keep the serial fallback.  This
file runs the same shapes on the GPU and checks every path against torch or a
straightforward Python fold of the combine, in particular the cases the
lowering review flagged: a scan sharing its lane loop with a row reduction, a
reverse scan with a non-commutative combine under both lowerings, reverse scans
over device-loop and vectorised lane loops, and a masked reverse cross-warp
scan.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
pytestmark = skipUnlessBackends(["cute"])

_SERIAL_RESCAN = "for scan_i in range("


def _segment_combine(
    left_values: torch.Tensor,
    left_indices: torch.Tensor,
    right_values: torch.Tensor,
    right_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segmented sum: not commutative, the right index always wins."""
    return (
        torch.where(
            left_indices == right_indices, left_values + right_values, right_values
        ),
        right_indices,
    )


@helion.kernel(backend="cute", static_shapes=True)
def _segment_scan(indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    num_elements, num_features = x.shape
    out = torch.empty_like(x)
    for tile_e, tile_f in hl.tile([num_elements, num_features]):
        vals = x[tile_e, tile_f]
        idxs = indices[tile_e].float().unsqueeze(1).expand_as(vals)
        out_vals, _ = hl.associative_scan(_segment_combine, (vals, idxs), dim=0)
        out[tile_e, tile_f] = out_vals
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _segment_scan_reverse(indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    num_elements, num_features = x.shape
    out = torch.empty_like(x)
    for tile_e, tile_f in hl.tile([num_elements, num_features]):
        vals = x[tile_e, tile_f]
        idxs = indices[tile_e].float().unsqueeze(1).expand_as(vals)
        out_vals, _ = hl.associative_scan(
            _segment_combine, (vals, idxs), dim=0, reverse=True
        )
        out[tile_e, tile_f] = out_vals
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _scan_then_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        row = x[i, :]
        prefix = hl.cumsum(row, dim=1)
        out[i, :] = prefix / prefix.sum(-1, keepdim=True)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _scan_and_row_total(x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        row = x[i, :]
        out[i, :] = hl.cumsum(row, dim=1)
        total[i] = row.sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _row_cumsum_reverse_and_total(x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        row = x[i, :]
        out[i, :] = hl.cumsum(row, dim=1, reverse=True)
        total[i] = row.sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _column_cumsum_reverse(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 128]):
        out[row, channel] = hl.cumsum(x[row, channel], dim=0, reverse=True)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _device_loop_row_cumsum_reverse(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = hl.cumsum(x[tile_m, tile_n], dim=1, reverse=True)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _flat_cumsum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.cumsum(x[tile], dim=0)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _flat_cumsum_reverse(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.cumsum(x[tile], dim=0, reverse=True)
    return out


def _run(
    kernel: helion.Kernel, args: tuple[object, ...], **overrides: Any
) -> tuple[str, Any]:
    """Run ``kernel`` under its default config with ``overrides`` applied.

    Every run compiles a fresh kernel object.  Compiling several configs of
    the same bound kernel interleaved with other kernels can make the lane
    reduction split's aliasing proof (``proven_disjoint_tensor_pairs``) lose
    the ``x`` / ``total`` disjointness and reject the config; that failure
    predates the scan lowering and is not what these tests measure.
    """
    fresh = helion.kernel(kernel.fn, backend="cute", static_shapes=True)
    bound = fresh.bind(args)
    config: dict[str, Any] = {**bound.config_spec.default_config().config, **overrides}
    return code_and_output(fresh, args, **config)


def _blockwise_cumsum(
    x: torch.Tensor, dim: int, block: int, *, reverse: bool = False
) -> torch.Tensor:
    """Inclusive cumsum restarted every ``block`` elements along ``dim``.

    ``hl.associative_scan`` scans one tile at a time, so a scan axis split
    into several tiles restarts at every tile boundary.
    """
    chunks = []
    for chunk in torch.split(x, block, dim=dim):
        if reverse:
            chunk = torch.flip(torch.cumsum(torch.flip(chunk, [dim]), dim), [dim])
        else:
            chunk = torch.cumsum(chunk, dim)
        chunks.append(chunk)
    return torch.cat(chunks, dim=dim)


def _flipped_cumsum(x: torch.Tensor, dim: int, block: int) -> torch.Tensor:
    return _blockwise_cumsum(x, dim, block, reverse=True)


def _segmented_reference(
    indices: torch.Tensor, x: torch.Tensor, *, reverse: bool, block: int
) -> torch.Tensor:
    """Fold ``combine(accumulated, current)`` in scan order like the ref impl,
    restarting at every ``block``-row tile like the kernel does."""
    out = torch.empty_like(x)
    for base in range(0, x.size(0), block):
        rows = range(base, min(base + block, x.size(0)))
        accumulated: tuple[torch.Tensor, torch.Tensor] | None = None
        for k in reversed(rows) if reverse else rows:
            current = (x[k], indices[k].float().expand_as(x[k]))
            if accumulated is None:
                accumulated = current
            else:
                accumulated = _segment_combine(*accumulated, *current)
            out[k] = accumulated[0]
    return out


def _segment_data(
    num_elements: int = 100, num_features: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    steps = torch.randint(0, 2, (num_elements,), generator=generator)
    # Rows 0..3 spell out the review's example: segments [0, 0], [1], [2, ...]
    # with values 1, 2, 3 in column 0, whose reverse segmented sum is
    # [3, 2, 3] (the ascending fold used to give [3, 3, 3]).
    steps[:4] = torch.tensor([0, 0, 1, 1])
    indices = steps.cumsum(0)
    values = torch.randint(-4, 5, (num_elements, num_features), generator=generator)
    values = values.float()
    values[:3, 0] = torch.tensor([1.0, 2.0, 3.0])
    return indices.to(DEVICE), values.to(DEVICE)


_SEGMENT_CONFIGS: dict[str, dict[str, Any]] = {
    # Blocked 4 x 8 rows per thread has no parallel lowering.
    "serial_fallback": {
        "block_sizes": [32, 128],
        "num_threads": [4, 32],
        "cute_lane_layouts": ["blocked", "blocked"],
    },
    # One thread walks the rows; 32 threads x 4-wide vectors cover the columns.
    "lane_loop": {
        "block_sizes": [32, 128],
        "num_threads": [1, 32],
        "cute_vector_widths": [1, 4],
    },
    # 32 threads own the 32 rows: one Kogge-Stone warp scan.
    "warp_scan": {"block_sizes": [32, 128], "num_threads": [32, 4]},
    # 4 threads x 8 strided chunks, carrying each chunk's total.
    "strided_chunks": {
        "block_sizes": [32, 128],
        "num_threads": [4, 32],
        "cute_lane_layouts": ["strided", "blocked"],
    },
}


@skipIfNotCUDA()
@pytest.mark.parametrize("name", sorted(_SEGMENT_CONFIGS))
@pytest.mark.parametrize(
    ("kernel", "reverse"), [(_segment_scan, False), (_segment_scan_reverse, True)]
)
def test_segmented_scan_matches_reference_under_every_lowering(
    kernel: helion.Kernel, reverse: bool, name: str
) -> None:
    indices, values = _segment_data()
    code, out = _run(kernel, (indices, values), **_SEGMENT_CONFIGS[name])
    if name == "serial_fallback":
        assert _SERIAL_RESCAN in code
    else:
        assert _SERIAL_RESCAN not in code
    expected = _segmented_reference(indices, values, reverse=reverse, block=32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if reverse:
        assert out[:3, 0].tolist() == [3.0, 2.0, 3.0]


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "overrides", [{}, {"block_sizes": [32], "num_threads": [0, 128]}]
)
def test_scan_sharing_a_lane_loop_with_a_row_reduction(
    overrides: dict[str, Any],
) -> None:
    x = torch.rand(64, 128, device=DEVICE) + 0.5
    prefix = torch.cumsum(x, 1)
    code, out = _run(_scan_then_sum, (x,), **overrides)
    assert _SERIAL_RESCAN in code
    torch.testing.assert_close(
        out, prefix / prefix.sum(1, keepdim=True), rtol=1e-4, atol=1e-5
    )
    total = torch.empty(64, device=DEVICE)
    code, out = _run(_scan_and_row_total, (x, total), **overrides)
    assert _SERIAL_RESCAN in code
    torch.testing.assert_close(out, prefix, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(total, x.sum(1), rtol=1e-4, atol=1e-4)
    total = torch.empty(64, device=DEVICE)
    code, out = _run(_row_cumsum_reverse_and_total, (x, total), **overrides)
    assert _SERIAL_RESCAN in code
    torch.testing.assert_close(out, _flipped_cumsum(x, 1, 128), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(total, x.sum(1), rtol=1e-4, atol=1e-4)


@skipIfNotCUDA()
def test_reverse_masked_cross_warp_scan() -> None:
    x = torch.randn(100, 3, device=DEVICE)
    out = torch.empty_like(x)
    code, result = _run(_column_cumsum_reverse, (x, out))
    assert "scan_smem_valid" in code
    assert "cute.arch.shuffle_sync_down(" in code
    torch.testing.assert_close(result, _flipped_cumsum(x, 0, 128), rtol=1e-4, atol=1e-4)


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "overrides",
    [
        {"block_sizes": [32, 64], "num_threads": [32, 1]},
        {"block_sizes": [8, 128], "num_threads": [8, 1], "cute_vector_widths": [1, 4]},
    ],
)
def test_device_loop_reverse_scan(overrides: dict[str, Any]) -> None:
    x = torch.randn(64, 256, device=DEVICE)
    code, out = _run(_device_loop_row_cumsum_reverse, (x,), **overrides)
    if "cute_vector_widths" in overrides:
        # A vectorised scan axis cannot run its vector loop backwards: the
        # serial fallback scans each column tile of the device loop.
        assert _SERIAL_RESCAN in code
    else:
        assert re.search(r"for lane_1 in range\(\d+, -1, -1\):", code)
    expected = _flipped_cumsum(x, 1, overrides["block_sizes"][1])
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@skipIfNotCUDA()
@pytest.mark.parametrize("numel", [1024, 1000])
def test_grid_vectorised_scan(numel: int) -> None:
    x = torch.randn(numel, device=DEVICE)
    config = {"block_sizes": [128], "num_threads": [1], "cute_vector_widths": [4]}
    # Forward: one thread per tile walks 32 vectors of 4 with a register carry.
    code, out = _run(_flat_cumsum, (x,), **config)
    assert _SERIAL_RESCAN not in code
    assert "scan_carry = scan_out" in code
    torch.testing.assert_close(out, _blockwise_cumsum(x, 0, 128), rtol=1e-4, atol=1e-4)
    # Reverse: the vector loop cannot run backwards, so the tile is rescanned
    # serially (block-local rows, including the masked partial last tile).
    code, out = _run(_flat_cumsum_reverse, (x,), **config)
    assert _SERIAL_RESCAN in code
    torch.testing.assert_close(out, _flipped_cumsum(x, 0, 128), rtol=1e-4, atol=1e-4)


@skipIfNotCUDA()
@pytest.mark.parametrize("numel", [1024, 1000])
def test_multi_tile_flat_serial_fallback_scans_block_local_rows(numel: int) -> None:
    # Blocked 32 rows per thread keeps the serial fallback; it has to rescan
    # the rows of *this* tile (the old last-dim rescan reread the first tile).
    x = torch.randn(numel, device=DEVICE)
    config = {"block_sizes": [128], "num_threads": [4]}
    code, out = _run(_flat_cumsum, (x,), **config)
    assert _SERIAL_RESCAN in code
    torch.testing.assert_close(out, _blockwise_cumsum(x, 0, 128), rtol=1e-4, atol=1e-4)
    code, out = _run(_flat_cumsum_reverse, (x,), **config)
    assert _SERIAL_RESCAN in code
    torch.testing.assert_close(out, _flipped_cumsum(x, 0, 128), rtol=1e-4, atol=1e-4)
