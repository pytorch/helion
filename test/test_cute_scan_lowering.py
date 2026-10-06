"""GPU-free codegen checks for the CuTe ``hl.associative_scan`` lowering.

Every test binds a kernel on CPU tensors and inspects ``bound.to_code`` for the
register / warp-shuffle scan shapes documented in ``cute/scan_ops.py``.
"""

from __future__ import annotations

import ast
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler import tile_strategy
from helion._compiler.cute import scan_ops
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])
CPU_DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
    ):
        yield


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Prefix-sum combine (kept a traced function, like the GPU tests)."""
    return left + right


def _segment_combine(
    left_values: torch.Tensor,
    left_indices: torch.Tensor,
    right_values: torch.Tensor,
    right_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.where(
            left_indices == right_indices, left_values + right_values, right_values
        ),
        right_indices,
    )


def _segment_scan(indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    num_elements, num_features = x.shape
    out = torch.empty_like(x)
    for tile_e, tile_f in hl.tile([num_elements, num_features]):
        vals = x[tile_e, tile_f]
        idxs = indices[tile_e].float().unsqueeze(1).expand_as(vals)
        out_vals, _ = hl.associative_scan(_segment_combine, (vals, idxs), dim=0)
        out[tile_e, tile_f] = out_vals
    return out


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


def _row_cumsum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        out[i, :] = hl.associative_scan(_add, x[i, :], dim=1)
    return out


def _row_cumsum_reverse(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        out[i, :] = hl.associative_scan(_add, x[i, :], dim=1, reverse=True)
    return out


def _column_cumsum(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 128]):
        out[row, channel] = hl.cumsum(x[row, channel], dim=0)
    return out


def _both_directions(x: torch.Tensor) -> torch.Tensor:
    num_elements, num_features = x.shape
    out = torch.empty_like(x)
    for tile_e, tile_f in hl.tile([num_elements, num_features]):
        vals = x[tile_e, tile_f]
        forward = hl.associative_scan(_add, vals, dim=0)
        backward = hl.associative_scan(_add, vals, dim=0, reverse=True)
        out[tile_e, tile_f] = forward + backward
    return out


def _column_cumsum_reverse(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 128]):
        out[row, channel] = hl.cumsum(x[row, channel], dim=0, reverse=True)
    return out


def _device_loop_row_cumsum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = hl.cumsum(x[tile_m, tile_n], dim=1)
    return out


def _device_loop_row_cumsum_reverse(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = hl.cumsum(x[tile_m, tile_n], dim=1, reverse=True)
    return out


def _flat_cumsum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.cumsum(x[tile], dim=0)
    return out


def _flat_cumsum_reverse(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.cumsum(x[tile], dim=0, reverse=True)
    return out


def _scan_then_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        row = x[i, :]
        prefix = hl.cumsum(row, dim=1)
        out[i, :] = prefix / prefix.sum(-1, keepdim=True)
    return out


def _scan_and_row_total(x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        row = x[i, :]
        out[i, :] = hl.cumsum(row, dim=1)
        total[i] = row.sum(-1)
    return out


def _scan_then_amax(x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        prefix = hl.cumsum(x[i, :], dim=1)
        out[i, :] = prefix
        total[i] = prefix.amax(-1)
    return out


def _row_cumsum_reverse_and_total(x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        row = x[i, :]
        out[i, :] = hl.cumsum(row, dim=1, reverse=True)
        total[i] = row.sum(-1)
    return out


def _segment_scan_with_feature_sum(
    indices: torch.Tensor, x: torch.Tensor, total: torch.Tensor
) -> torch.Tensor:
    num_elements, num_features = x.shape
    out = torch.empty_like(x)
    for tile_e, tile_f in hl.tile([num_elements, num_features]):
        vals = x[tile_e, tile_f]
        idxs = indices[tile_e].float().unsqueeze(1).expand_as(vals)
        out_vals, _ = hl.associative_scan(_segment_combine, (vals, idxs), dim=0)
        out[tile_e, tile_f] = out_vals
        total[tile_e] = vals.sum(-1)
    return out


def _segment_scan_with_row_sum(
    indices: torch.Tensor, x: torch.Tensor, total: torch.Tensor
) -> torch.Tensor:
    num_elements, num_features = x.shape
    out = torch.empty_like(x)
    for tile_e, tile_f in hl.tile([num_elements, num_features]):
        vals = x[tile_e, tile_f]
        idxs = indices[tile_e].float().unsqueeze(1).expand_as(vals)
        out_vals, _ = hl.associative_scan(_segment_combine, (vals, idxs), dim=0)
        out[tile_e, tile_f] = out_vals
        total[tile_f] = vals.sum(0)
    return out


def _kernel_body(code: str) -> str:
    """The device kernel only: no host wrapper, no ``# src[...]`` comments."""
    start = code.index("@cute.kernel")
    kernel_def = code.index("\ndef ", start)
    end = code.find("\ndef ", kernel_def + 1)
    if end < 0:
        end = len(code)
    return "\n".join(
        line for line in code[start:end].splitlines() if "# src[" not in line
    )


def _codegen(
    fn: Callable[..., torch.Tensor], args: tuple[object, ...], **overrides: Any
) -> str:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    config = bound.config_spec.default_config()
    if overrides:
        merged: dict[str, Any] = {**config.config, **overrides}
        config = helion.Config(**merged)
    return _kernel_body(bound.to_code(config))


def _segment_args(num_elements: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(num_elements, dtype=torch.int64, device=CPU_DEVICE),
        torch.zeros(num_elements, 128, device=CPU_DEVICE),
    )


_SERIAL_RESCAN = "for scan_i in range("


def test_lane_loop_scan_carries_prefix_without_rereads() -> None:
    # One thread walks the 32 scanned rows (``num_threads[0] == 1``) while 32
    # threads x 4-wide vectors cover the 128 columns: the study winner shape.
    body = _codegen(
        _segment_scan,
        _segment_args(),
        block_sizes=[32, 128],
        num_threads=[1, 32],
        cute_vector_widths=[1, 4],
    )
    assert _SERIAL_RESCAN not in body
    assert "shuffle_sync" not in body
    assert "sync_threads" not in body
    # Both streams keep a per-column prefix in a register fragment indexed by
    # the vector lane, seeded on the first scanned row.
    assert body.count("cute.make_rmem_tensor((4,), cutlass.Float32)") == 2
    assert "scan_slot = cutlass.Int32(vec_lane_1)" in body
    assert "scan_rest = scan_pos != cutlass.Int32(0)" in body
    assert "scan_carry[scan_slot] = scan_out" in body
    # The element loop reuses the values it already loaded: exactly one vector
    # load of ``x`` and one scalar load of ``indices``, no tensor re-indexing.
    assert body.count("cute.arch.load(") == 1
    assert body.count(".load()") == 1
    assert "x[" not in body
    assert "indices[" not in body
    # Validity flags are only needed for reverse scans (masked rows form a
    # suffix); this divisible extent has no mask at all.
    assert "scan_carry_valid" not in body
    assert "mask_0" not in body


def test_lane_loop_scan_inlines_tuple_combine() -> None:
    body = _codegen(
        _segment_scan,
        _segment_args(),
        block_sizes=[32, 128],
        num_threads=[1, 32],
        cute_vector_widths=[1, 4],
    )
    # ``where(left_idx == right_idx, left + right, right)`` on the carried
    # prefix (``scan_prev*``) and the incoming element (``scan_value*``).
    assert "scan_combine = scan_prev_1 == scan_value_1" in body
    assert "scan_combine_1 = scan_prev + scan_value" in body
    assert "scan_combine_2 = scan_combine_1 if scan_combine else scan_value" in body
    # The second stream (segment index) is passed through and carried too.
    assert "scan_carry_1[scan_slot] = scan_out_1" in body


def test_forward_partial_tile_lane_loop_needs_no_validity_flags() -> None:
    body = _codegen(
        _segment_scan,
        _segment_args(num_elements=1000),
        block_sizes=[32, 128],
        num_threads=[1, 32],
        cute_vector_widths=[1, 4],
    )
    assert "mask_0 = indices_0 < 1000" in body
    assert _SERIAL_RESCAN not in body
    assert "scan_carry_valid" not in body


def test_thread_split_scan_emits_warp_shuffle_scan() -> None:
    # Persistent row scan: 32 threads x 4 strided lane steps over 128 columns.
    body = _codegen(_row_cumsum, (torch.zeros(64, 128, device=CPU_DEVICE),))
    assert _SERIAL_RESCAN not in body
    assert "for synthetic_lane_1 in range(4):" in body
    assert body.count("cute.arch.shuffle_sync_up(") == 5
    assert "if scan_lane >= 16 else" in body
    # The chunk total (lane 31) is broadcast as the carry for the next chunk.
    assert re.search(r"cute\.arch\.shuffle_sync\(scan_out\w*, 31\)", body)
    assert "scan_rest = cutlass.Int32(synthetic_lane_1) != cutlass.Int32(0)" in body
    assert "sync_threads" not in body


def test_thread_split_single_chunk_has_no_carry() -> None:
    # 32 threads own the 32 scanned rows outright: a pure warp scan.
    body = _codegen(
        _segment_scan,
        _segment_args(),
        block_sizes=[32, 128],
        num_threads=[32, 4],
    )
    assert _SERIAL_RESCAN not in body
    assert body.count("cute.arch.shuffle_sync_up(") == 10  # 2 streams x log2(32)
    assert "scan_carry" not in body
    assert "sync_threads" not in body


def test_strided_thread_split_scans_chunks_with_carry() -> None:
    body = _codegen(
        _segment_scan,
        _segment_args(),
        block_sizes=[32, 128],
        num_threads=[4, 32],
        cute_lane_layouts=["strided", "blocked"],
    )
    assert _SERIAL_RESCAN not in body
    assert "scan_lane = cutlass.Int32(cute.arch.thread_idx()[0]) % 4" in body
    assert body.count("cute.arch.shuffle_sync_up(") == 4  # 2 streams x log2(4)
    # Per-column carry (4 inner lanes over the feature axis), refreshed from
    # the chunk's last thread.
    assert body.count("cute.make_rmem_tensor((4,), cutlass.Float32)") == 2
    assert "scan_source_lane = cutlass.Int32(cute.arch.lane_idx()) // 4 * 4 + 3" in body


def test_blocked_multi_element_thread_split_keeps_serial_fallback() -> None:
    # Each of the 4 threads owns 8 consecutive rows: needs a second pass, so
    # this stays on the serial rescan.
    body = _codegen(
        _segment_scan,
        _segment_args(),
        block_sizes=[32, 128],
        num_threads=[4, 32],
        cute_lane_layouts=["blocked", "blocked"],
    )
    assert _SERIAL_RESCAN in body
    assert "shuffle_sync" not in body


def test_cross_warp_scan_exchanges_warp_totals_through_smem() -> None:
    # The default config puts the 128 scanned rows on 128 threads (4 warps).
    body = _codegen(
        _column_cumsum,
        (
            torch.zeros(128, 3, device=CPU_DEVICE),
            torch.zeros(128, 3, device=CPU_DEVICE),
        ),
    )
    assert _SERIAL_RESCAN not in body
    assert body.count("cute.arch.shuffle_sync_up(") == 5
    assert "scan_smem_ptr = cute.arch.alloc_smem(cutlass.Float32, 32)" in body
    assert "scan_cta_warp = cutlass.Int32(cute.arch.warp_idx())" in body
    assert "if scan_lane == 31:" in body
    assert body.count("cute.arch.sync_threads()") == 2
    assert (
        "for scan_j in range(cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)):"
        in body
    )
    assert "scan_before = scan_step < scan_warp" in body


def test_reverse_thread_split_scan_mirrors_shuffles_and_lane_order() -> None:
    body = _codegen(_row_cumsum_reverse, (torch.zeros(64, 128, device=CPU_DEVICE),))
    assert _SERIAL_RESCAN not in body
    assert "for synthetic_lane_1 in range(3, -1, -1):" in body
    assert body.count("cute.arch.shuffle_sync_down(") == 5
    assert "if scan_lane < 16 else" in body
    assert re.search(r"cute\.arch\.shuffle_sync\(scan_out\w*, 0\)", body)
    assert "scan_rest = cutlass.Int32(synthetic_lane_1) != cutlass.Int32(3)" in body


def test_reverse_lane_loop_scan_excludes_masked_rows() -> None:
    body = _codegen(
        _segment_scan_reverse,
        _segment_args(num_elements=1000),
        block_sizes=[32, 128],
        num_threads=[1, 32],
        cute_vector_widths=[1, 4],
    )
    assert _SERIAL_RESCAN not in body
    # Lanes are visited in descending order and the prefix is seeded on the
    # last row of the tile.
    assert "for lane_0 in range(31, -1, -1):" in body
    assert "scan_rest = scan_pos != cutlass.Int32(31)" in body
    # Rows past the end of the tensor come first in this order, so their
    # validity is carried next to the values and gates every combine.
    assert "scan_carry_valid = cute.make_rmem_tensor((4,), cutlass.Int32)" in body
    assert "scan_use = scan_rest and scan_prev_valid" in body
    assert "scan_valid = mask_0 or scan_use" in body
    assert re.search(r"if scan_use and mask_0 else scan_prev if scan_use else", body)


def test_conflicting_scan_directions_fall_back_to_serial() -> None:
    body = _codegen(
        _both_directions,
        (torch.zeros(4096, 128, device=CPU_DEVICE),),
        block_sizes=[32, 128],
        num_threads=[1, 32],
    )
    # The forward scan claims the ascending lane loop; the reverse scan over
    # the same lanes cannot flip it and keeps the serial rescan.
    assert "for lane_0 in range(32):" in body
    assert "scan_rest = scan_pos != cutlass.Int32(0)" in body
    assert body.count(_SERIAL_RESCAN) == 1
    # The fallback walks the positions descending for the reverse scan.
    assert "scan_index = cutlass.Int32(31) - scan_i" in body
    assert "scan_include = scan_index >= scan_out_pos" in body


def test_unit_scan_axis_is_the_identity() -> None:
    # A size-1 scan axis has no tile block to resolve and needs no scan at
    # all: the input is stored as is, for scalar and tuple scans alike.
    body = _codegen(_row_cumsum, (torch.zeros(2, 1, device=CPU_DEVICE),))
    assert "scan_" not in body
    assert "shuffle_sync" not in body
    body = _codegen(
        _segment_scan,
        (
            torch.zeros(1, dtype=torch.int64, device=CPU_DEVICE),
            torch.zeros(1, 128, device=CPU_DEVICE),
        ),
    )
    assert "scan_" not in body
    assert "shuffle_sync" not in body


# ---------------------------------------------------------------------------
# Scans sharing a lane loop with a lane reduction keep the serial fallback
# ---------------------------------------------------------------------------


def _rows() -> tuple[torch.Tensor, ...]:
    return (torch.zeros(64, 128, device=CPU_DEVICE),)


def _rows_and_total() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(64, 128, device=CPU_DEVICE),
        torch.zeros(64, device=CPU_DEVICE),
    )


@pytest.mark.parametrize(
    "overrides", [{}, {"block_sizes": [32], "num_threads": [0, 128]}]
)
@pytest.mark.parametrize(
    ("fn", "make_args"),
    [
        (_scan_then_sum, _rows),
        (_scan_and_row_total, _rows_and_total),
        (_scan_then_amax, _rows_and_total),
    ],
)
def test_scan_sharing_a_lane_loop_with_a_reduction_keeps_serial_fallback(
    fn: Callable[..., torch.Tensor],
    make_args: Callable[[], tuple[torch.Tensor, ...]],
    overrides: dict[str, Any],
) -> None:
    # The row reduction splits the synthetic lane loop into accumulate /
    # consume passes; a lane-carried scan prefix cannot survive that split,
    # so the scan declines up front and the kernel still compiles.
    body = _codegen(fn, make_args(), **overrides)
    assert "for synthetic_lane_1 in range(4):" in body
    assert _SERIAL_RESCAN in body
    assert "scan_carry" not in body
    assert "shuffle_sync" not in body
    assert re.search(r"_lane_acc = ", body)
    assert re.search(r"cute\.arch\.warp_reduction_(sum|max)\(", body)


def test_scan_with_reduction_on_a_thread_split_axis_stays_parallel() -> None:
    # One row per block on 128 threads: no lane loop, so the reduction is a
    # plain cross-thread combine and the scan keeps the warp-shuffle path.
    body = _codegen(
        _scan_and_row_total, _rows_and_total(), block_sizes=[1], num_threads=[0, 128]
    )
    assert "synthetic_lane" not in body
    assert _SERIAL_RESCAN not in body
    assert body.count("cute.arch.shuffle_sync_up(") == 5


def test_lane_loop_scan_declines_only_for_reductions_over_its_lanes() -> None:
    # 32 features on 32 threads: the feature axis has no lane loop, so its
    # reduction is a plain warp combine and the row-lane scan keeps its carry.
    indices = torch.zeros(4096, dtype=torch.int64, device=CPU_DEVICE)
    x = torch.zeros(4096, 32, device=CPU_DEVICE)
    config = {"block_sizes": [32, 32], "num_threads": [1, 32]}
    body = _codegen(
        _segment_scan_with_feature_sum,
        (indices, x, torch.zeros(4096, device=CPU_DEVICE)),
        **config,
    )
    assert _SERIAL_RESCAN not in body
    assert "for lane_0 in range(32):" in body
    assert "scan_carry = scan_out" in body
    assert "cute.arch.warp_reduction_sum(" in body
    # A reduction over the lane-looped rows splits that lane loop: fallback.
    body = _codegen(
        _segment_scan_with_row_sum,
        (indices, x, torch.zeros(32, device=CPU_DEVICE)),
        **config,
    )
    assert _SERIAL_RESCAN in body
    assert "scan_carry" not in body
    assert re.search(r"_lane_acc = ", body)


# ---------------------------------------------------------------------------
# Reverse serial fallbacks walk the axis descending
# ---------------------------------------------------------------------------


def test_reverse_tuple_fallback_walks_positions_descending() -> None:
    # Blocked 4 x 8 rows per thread has no parallel lowering; the fallback
    # must fold ``combine(suffix, current)`` from the last row down so a
    # non-commutative combine sees the reference operand order.
    body = _codegen(
        _segment_scan_reverse,
        _segment_args(),
        block_sizes=[32, 128],
        num_threads=[4, 32],
        cute_lane_layouts=["blocked", "blocked"],
    )
    assert body.count(_SERIAL_RESCAN) == 1
    assert "scan_index = cutlass.Int32(31) - scan_i" in body
    assert "scan_include = scan_index >= scan_out_pos" in body
    assert re.search(
        r"scan_row = cutlass\.Int32\(.*\) \+ scan_index$", body, re.MULTILINE
    )
    # The forward fallback is unchanged: ascending positions, no remapping.
    body = _codegen(
        _segment_scan,
        _segment_args(),
        block_sizes=[32, 128],
        num_threads=[4, 32],
        cute_lane_layouts=["blocked", "blocked"],
    )
    assert "scan_index" not in body
    assert "scan_include = scan_i <= scan_out_pos" in body


def test_reverse_scalar_fallback_walks_positions_descending() -> None:
    # The same-row reduction forces the scalar last-dim fallback.
    body = _codegen(_row_cumsum_reverse_and_total, _rows_and_total())
    assert body.count(_SERIAL_RESCAN) == 1
    assert "scan_index = cutlass.Int32(127) - scan_i" in body
    assert "scan_include = scan_index >= scan_out_pos" in body
    assert re.search(
        r"scan_row = cutlass\.Int32\(.*\) \+ scan_index$", body, re.MULTILINE
    )
    assert re.search(r"scan_value = .*x\[indices_0, scan_row\]", body)


def test_multi_tile_flat_fallback_scans_block_local_rows() -> None:
    # Blocked 32 rows per thread has no parallel lowering.  The fallback must
    # rescan *this* tile's rows: the old last-dim rescan indexed the first
    # tile's rows by the global position and was wrong past the first tile.
    body = _codegen(
        _flat_cumsum,
        (torch.zeros(1024, device=CPU_DEVICE),),
        block_sizes=[128],
        num_threads=[4],
    )
    assert body.count(_SERIAL_RESCAN) == 1
    assert "for scan_i in range(cutlass.Int32(0), cutlass.Int32(128)" in body
    assert "scan_out_pos = cutlass.Int32(indices_0 - pid_flat * _BLOCK_SIZE_0)" in body
    assert "scan_row = cutlass.Int32(pid_flat * _BLOCK_SIZE_0) + scan_i" in body
    assert "x[scan_row]" in body
    assert "x[scan_i]" not in body


# ---------------------------------------------------------------------------
# Reverse lane loops: device loops, vector partitions, masked cross-warp
# ---------------------------------------------------------------------------


def test_device_loop_reverse_scan_reverses_its_lane_loop_in_place() -> None:
    body = _codegen(
        _device_loop_row_cumsum_reverse,
        (torch.zeros(64, 256, device=CPU_DEVICE),),
        block_sizes=[32, 64],
        num_threads=[32, 1],
    )
    assert _SERIAL_RESCAN not in body
    assert "for tile_offset_1 in range(" in body
    assert "for lane_1 in range(63, -1, -1):" in body
    assert "scan_rest = scan_pos != cutlass.Int32(63)" in body
    assert "scan_carry = scan_out" in body


def test_device_loop_vectorised_reverse_scan_keeps_serial_fallback() -> None:
    config = {
        "block_sizes": [8, 128],
        "num_threads": [8, 1],
        "cute_vector_widths": [1, 4],
    }
    x = (torch.zeros(64, 256, device=CPU_DEVICE),)
    # The constexpr vector loop feeds the vector store in iteration order, so
    # it cannot run backwards: a reverse scan over a vectorised axis declines.
    body = _codegen(_device_loop_row_cumsum_reverse, x, **config)
    assert _SERIAL_RESCAN in body
    assert "(3, -1, -1)" not in body
    assert "scan_carry" not in body
    # The forward scan over the same partition keeps the register carry.
    body = _codegen(_device_loop_row_cumsum, x, **config)
    assert _SERIAL_RESCAN not in body
    assert "for vec_lane_1 in cutlass.range_constexpr(4):" in body
    assert "scan_pos = cutlass.Int32(lane_1) * 4 + cutlass.Int32(vec_lane_1)" in body
    assert "scan_carry = scan_out" in body


def test_grid_vectorised_reverse_scan_keeps_serial_fallback() -> None:
    config = {"block_sizes": [128], "num_threads": [1], "cute_vector_widths": [4]}
    x = (torch.zeros(1024, device=CPU_DEVICE),)
    body = _codegen(_flat_cumsum_reverse, x, **config)
    assert _SERIAL_RESCAN in body
    assert "-1, -1)" not in body
    assert "scan_carry" not in body
    # The fallback rescans this tile's block-local rows, descending.
    assert "scan_out_pos = cutlass.Int32(indices_0 - pid_flat * _BLOCK_SIZE_0)" in body
    assert "scan_row = cutlass.Int32(pid_flat * _BLOCK_SIZE_0) + scan_index" in body
    # The forward scan keeps the pre-built vector partition (the vector-load
    # hoist then turns the plain ``range`` into a constexpr loop).
    body = _codegen(_flat_cumsum, x, **config)
    assert _SERIAL_RESCAN not in body
    assert re.search(r"for lane_0 in (cutlass\.range_constexpr|range)\(32\):", body)
    assert "for vec_lane_0 in cutlass.range_constexpr(4):" in body
    assert "scan_pos = cutlass.Int32(lane_0) * 4 + cutlass.Int32(vec_lane_0)" in body
    assert "scan_rest = scan_pos != cutlass.Int32(0)" in body


def test_reverse_masked_cross_warp_scan_tracks_validity_through_smem() -> None:
    # 100 rows on 128 threads (4 warps): the padded rows come first in scan
    # order, so validity rides next to the values through the shuffles, the
    # shared-memory warp totals and the exclusive warp prefix.
    body = _codegen(
        _column_cumsum_reverse,
        (
            torch.zeros(100, 3, device=CPU_DEVICE),
            torch.zeros(100, 3, device=CPU_DEVICE),
        ),
    )
    assert _SERIAL_RESCAN not in body
    assert re.search(r"mask_1 = .*indices_1 < 100", body)
    assert body.count("cute.arch.shuffle_sync_down(") == 10  # values + validity
    assert "cute.arch.shuffle_sync_down(cutlass.Int32(mask_1), 1)" in body
    assert "scan_smem_valid_ptr = cute.arch.alloc_smem(cutlass.Int32, 32)" in body
    assert "if scan_lane == 0:" in body
    assert "scan_step = cutlass.Int32(3) - scan_j" in body
    assert "scan_before = scan_step > scan_warp" in body
    assert body.count("cute.arch.sync_threads()") == 2


# ---------------------------------------------------------------------------
# Transactional lane-loop reversal
# ---------------------------------------------------------------------------


def _fake_state() -> Any:
    cute_state = SimpleNamespace(scan_lane_directions={})
    return SimpleNamespace(device_function=SimpleNamespace(cute_state=cute_state))


def _directions(state: Any) -> dict[str, bool]:
    return state.device_function.cute_state.scan_lane_directions


def _geometry(owner: Any, lane_var: str) -> Any:
    return scan_ops._CuteScanGeometry(
        block_id=1,
        extent=8,
        threads=1,
        lane_var=lane_var,
        lane_steps=8,
        vec_lane_var=None,
        vec_width=1,
        strided=False,
        inner_lanes=(),
        mask_expr=None,
        owner=owner,
    )


def _for(source: str) -> ast.For:
    return cast("ast.For", ast.parse(source).body[0])


def _grid(lane_var: str = "lane_0") -> tile_strategy.DeviceGridState:
    grid = tile_strategy.DeviceGridState(
        strategy=cast("Any", SimpleNamespace(block_ids=[0])), block_id_to_info={}
    )
    grid.lane_loops.append((lane_var, 8))
    return grid


def _vector_partition(grid: tile_strategy.DeviceGridState, lane_var: str) -> None:
    outer = tile_strategy._create_lane_loop(lane_var, 2, [_for("for i in x:\n pass")])
    vloop = _for("for vec_lane_0 in cutlass.range_constexpr(4):\n    pass")
    grid.vec_lane_wrappers[lane_var] = tile_strategy.VecLaneWrapper(
        outer_for=outer,
        vloop=vloop,
        vec_lane_var="vec_lane_0",
        base_index_var="lane_base_0",
    )


def test_device_loop_reversal_flips_every_lane_loop_and_records_the_direction() -> None:
    inner = _for("for i in x:\n    pass")
    lane_loop = tile_strategy._create_lane_loop("lane_1", 8, [inner])
    tail_loop = tile_strategy._create_lane_loop(
        "lane_1", 8, [_for("for j in y:\n pass")]
    )
    device_loop = _for("for tile_offset_1 in range(0, 256, 64):\n    pass")
    device_loop.body = [lane_loop, tail_loop]
    state = _fake_state()
    owner = SimpleNamespace(for_node=device_loop)
    assert scan_ops._cute_scan_prepare_lane_direction(
        state, _geometry(owner, "lane_1"), True
    )
    # Every loop over the scan lanes runs backwards; unrelated loops are kept.
    assert ast.unparse(lane_loop.iter) == "range(7, -1, -1)"
    assert ast.unparse(tail_loop.iter) == "range(7, -1, -1)"
    assert ast.unparse(inner.iter) == "x"
    assert ast.unparse(device_loop.iter) == "range(0, 256, 64)"
    assert _directions(state) == {"lane_1": True}
    # A second reverse scan over the same lanes agrees without touching them;
    # a forward scan disagrees and must fall back.
    assert scan_ops._cute_scan_prepare_lane_direction(
        state, _geometry(owner, "lane_1"), True
    )
    assert ast.unparse(lane_loop.iter) == "range(7, -1, -1)"
    assert not scan_ops._cute_scan_prepare_lane_direction(
        state, _geometry(owner, "lane_1"), False
    )


def test_failed_device_loop_reversal_mutates_nothing() -> None:
    lane_loop = tile_strategy._create_lane_loop(
        "lane_1", 8, [_for("for i in x:\n pass")]
    )
    stale = tile_strategy._create_lane_loop("lane_1", 8, [_for("for j in y:\n pass")])
    assert tile_strategy._reverse_lane_loop_iter(stale)
    device_loop = _for("for tile_offset_1 in range(0, 256, 64):\n    pass")
    device_loop.body = [lane_loop, stale]
    state = _fake_state()
    owner = SimpleNamespace(for_node=device_loop)
    # ``stale`` is not in the plain ascending form: the whole request fails
    # before ``lane_loop`` is rewritten and no direction is recorded.
    assert not scan_ops._cute_scan_prepare_lane_direction(
        state, _geometry(owner, "lane_1"), True
    )
    assert ast.unparse(lane_loop.iter) == "range(8)"
    assert ast.unparse(stale.iter) == "range(7, -1, -1)"
    assert _directions(state) == {}
    # No lane loop at all is a failure too, not a silent forward scan.
    device_loop.body = []
    assert not scan_ops._cute_scan_prepare_lane_direction(
        state, _geometry(owner, "lane_1"), True
    )
    assert _directions(state) == {}


def test_grid_reversal_is_recorded_and_applied_when_the_loop_is_built() -> None:
    grid = _grid()
    state = _fake_state()
    assert scan_ops._cute_scan_prepare_lane_direction(
        state, _geometry(grid, "lane_0"), True
    )
    assert grid.reversed_lane_vars == {"lane_0"}
    assert _directions(state) == {"lane_0": True}
    body = ast.parse("value = lane_0 + 1").body
    (loop,) = grid.wrap_body(cast("list[ast.AST]", list(body)))
    assert isinstance(loop, ast.For)
    assert ast.unparse(loop.iter) == "range(7, -1, -1)"
    assert ast.unparse(loop.body[0]) == "value = lane_0 + 1"


def test_grid_reversal_declines_a_vector_partition_without_recording() -> None:
    grid = _grid()
    _vector_partition(grid, "lane_0")
    state = _fake_state()
    assert not scan_ops._cute_scan_prepare_lane_direction(
        state, _geometry(grid, "lane_0"), True
    )
    assert grid.reversed_lane_vars == set()
    assert _directions(state) == {}
    wrapper = grid.vec_lane_wrappers["lane_0"]
    assert ast.unparse(wrapper.outer_for.iter) == "range(2)"
    assert ast.unparse(wrapper.vloop.iter) == "cutlass.range_constexpr(4)"


def test_wrap_body_refuses_to_reverse_a_vector_partition() -> None:
    grid = _grid()
    _vector_partition(grid, "lane_0")
    grid.reversed_lane_vars.add("lane_0")
    body = ast.parse("value = lane_base_0 + vec_lane_0").body
    with pytest.raises(helion.exc.BackendUnsupported, match="vectorised lane loop"):
        grid.wrap_body(cast("list[ast.AST]", list(body)))
