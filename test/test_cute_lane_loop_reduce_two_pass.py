"""A reduction over a lane-looped tile axis combines across threads ONCE.

``col_reduce_sum``-style kernels reduce a 2-D tile along its threaded row
axis while the row block is also traversed by a per-thread lane loop.  The
reduction used to fall through to the per-element strided thread reduction,
so every synthetic lane paid a full cross-thread combine (a warp shuffle
tree, or a CTA-wide two-stage shared-memory reduction when the row threads
sit above the column threads on the linear thread index).  The two-pass
lane-reduction marker now owns these reductions: each thread accumulates
across its lanes and the cross-thread combine runs once per row tile.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Callable

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True)
def _col_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros(n, dtype=x.dtype, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        col_acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            col_acc += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = col_acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _col_reduce_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        col_acc = hl.full([tile_n], float("-inf"), dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            col_acc = torch.maximum(
                col_acc, torch.amax(x[tile_m, tile_n].to(torch.float32), dim=0)
            )
        out[tile_n] = col_acc.to(out.dtype)
    return out


def _col_sum(t: torch.Tensor) -> torch.Tensor:
    return t.float().sum(0).to(t.dtype)


def _col_max(t: torch.Tensor) -> torch.Tensor:
    return t.float().amax(0).to(t.dtype)


def _lane_loop_body(code: str) -> str:
    """Return the source of the row lane loop (``for lane_0 in range(N)``)."""
    lines = code.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.lstrip().startswith("for lane_0 in")
    )
    indent = len(lines[start]) - len(lines[start].lstrip())
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return "\n".join(body)


@onlyBackends(["cute"])
class TestCuteLaneLoopReduceTwoPass(TestCase):
    def _check(
        self,
        kernel: object,
        ref: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        **config: object,
    ) -> str:
        code, out = code_and_output(kernel, (x,), **config)  # pyrefly: ignore
        torch.testing.assert_close(out, ref(x), atol=2e-2, rtol=2e-2)
        # A second, different input must produce a different (correct) output:
        # kernels allocate their own result, so a lowering that skipped the
        # combine could otherwise pass on a stale buffer.
        y = torch.randn_like(x) * 3
        _, out_y = code_and_output(kernel, (y,), **config)  # pyrefly: ignore
        torch.testing.assert_close(out_y, ref(y), atol=5e-2, rtol=2e-2)
        self.assertFalse(torch.allclose(out.float(), out_y.float()))
        return code

    def test_cross_warp_group_reduces_once_per_row_tile(self) -> None:
        """Row threads above the column threads: pre=8, span=128 (4 warps).

        The lane loop must only accumulate; the two-stage shared reduction
        runs once after it (per row tile), not once per element.
        """
        x = torch.randn(2048, 64, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            _col_reduce_sum,
            _col_sum,
            x,
            block_sizes=[2048, 16],
            num_threads=[16, 8],
            cute_vector_widths=[1, 2],
        )
        self.assertEqual(code.count("_cute_grouped_reduce_shared_two_stage("), 1)
        body = _lane_loop_body(code)
        self.assertNotIn("_cute_grouped_reduce", body)
        self.assertNotIn("warp_reduction", body)
        self.assertIn("_lane_acc", body)
        self.assertIn("pre=8, group_span=128, group_count=1", code)

    def test_single_warp_group_reduces_once_per_row_tile(self) -> None:
        """Row threads above the column threads inside one warp: pre=8,
        span=32 -> a single grouped warp shuffle after the lane loop."""
        x = torch.randn(1024, 64, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            _col_reduce_sum,
            _col_sum,
            x,
            block_sizes=[1024, 16],
            num_threads=[4, 8],
            cute_vector_widths=[1, 2],
        )
        self.assertEqual(code.count("_cute_grouped_reduce_warp("), 1)
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)
        self.assertNotIn("_cute_grouped_reduce", _lane_loop_body(code))
        self.assertIn("pre=8, group_span=32", code)

    def test_bottom_axis_warp_reduces_once_per_row_tile(self) -> None:
        """Row threads at the bottom of the thread index (pre=1): a plain
        32-lane warp reduction once after the lane loop, even without the
        resident reduction sequence."""
        x = torch.randn(2048, 64, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            _col_reduce_sum,
            _col_sum,
            x,
            block_sizes=[2048, 4],
            num_threads=[32, 4],
            cute_reduction_sequence="scalar",
            cute_lane_layouts=["strided", "blocked"],
        )
        self.assertEqual(code.count("warp_reduction_sum("), 1)
        self.assertNotIn("_cute_grouped_reduce", code)
        self.assertNotIn("warp_reduction", _lane_loop_body(code))

    def test_max_reduction_cross_warp_group(self) -> None:
        x = torch.randn(2048, 64, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            _col_reduce_max,
            _col_max,
            x,
            block_sizes=[2048, 16],
            num_threads=[16, 8],
            cute_vector_widths=[1, 2],
        )
        self.assertEqual(code.count("_cute_grouped_reduce_shared_two_stage("), 1)
        self.assertNotIn("_cute_grouped_reduce", _lane_loop_body(code))


@onlyBackends(["cute"])
def test_owned_marker_restore_when_two_pass_split_is_unsafe() -> None:
    """jagged_layer_norm at 256 threads x 2 lanes: the lane loop carries extra
    cross-lane sums, so the two-pass split is refused and the owned marker is
    restored with its own running accumulator (finalized every lane)."""
    from examples.jagged_layer_norm import jagged_layer_norm_kernel
    from examples.jagged_layer_norm import reference_jagged_layer_norm_pytorch

    torch.manual_seed(0)
    lengths = torch.randint(1, 65, (32,), device=DEVICE)
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=DEVICE), torch.cumsum(lengths, 0)]
    )
    x = torch.randn(int(x_offsets[-1]), 512, dtype=torch.float32, device=DEVICE)
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    code, out = code_and_output(
        kernel,
        (x, x_offsets, 1e-6),
        block_sizes=[1, 512, 32, 512, 32, 512, 32],
        cute_lane_layouts=["strided"] * 7,
        cute_vector_widths=[1] * 7,
        num_threads=[1, 256, 1, 256, 1, 256, 1],
    )
    assert "_lane_acc = " in code
    torch.testing.assert_close(
        out,
        reference_jagged_layer_norm_pytorch(x, x_offsets, 1e-6),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_two_lane_nested_row_seeds_match_their_static_twin_under_dynamic_shapes() -> (
    None
):
    """jagged_layer_norm's 512-column nested-row seeds with two column lanes.

    Under dynamic shapes the column mask turns ``row_sums.sum()`` into a
    masked two-pass lane reduction, whose accumulate pass must run the
    jagged loop that accumulates ``row_sums``; it used to fold the fresh
    zero in a pass ahead of that loop and return mean = variance = 0 (``x *
    rsqrt(eps)``).  The dynamic render now takes the static twin's per-lane
    schedule, so the two agree bitwise.
    """
    from examples.jagged_layer_norm import jagged_layer_norm_kernel
    from examples.jagged_layer_norm import reference_jagged_layer_norm_pytorch

    from helion._compiler.autotuner_heuristics.cute import CuteNestedRowHeuristic

    torch.manual_seed(0)
    lengths = torch.randint(0, 40, (17,))
    lengths[3] = 0
    x_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    x_offsets = x_offsets.to(DEVICE)
    x = torch.randn((int(lengths.sum()), 512), device=DEVICE)
    expected = reference_jagged_layer_norm_pytorch(x, x_offsets, 1e-6)
    outputs: dict[bool, list[torch.Tensor]] = {}
    for static_shapes in (True, False):
        kernel = helion.kernel(
            jagged_layer_norm_kernel.fn,
            backend="cute",
            static_shapes=static_shapes,
            autotune_effort="none",
        )
        bound = kernel.bind((x, x_offsets, 1e-6))
        host = bound.host_function
        assert host is not None
        seeds = [
            seed
            for seed in CuteNestedRowHeuristic.get_seed_configs(
                bound.env, host.device_ir
            )
            if max(seed.num_threads) == 256
        ]
        assert len(seeds) == 3
        for seed in seeds:
            code = bound.to_code(seed)
            assert "for lane_1 in range(2):" in code
            out = bound.compile_config(seed)(x, x_offsets, 1e-6)
            torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
            outputs.setdefault(static_shapes, []).append(out)
    for static, dynamic in zip(outputs[True], outputs[False], strict=True):
        assert torch.equal(static, dynamic)


def _accumulated_columns_reduced_and_stored(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column K accumulation, reduced across the columns into a carried row sum, and stored per column."""
    b, k, n = x.shape
    out = torch.empty([b, n], dtype=x.dtype, device=x.device)
    tot = torch.empty([b], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(b):
        row_acc = hl.zeros([tile_b], dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc = hl.zeros([tile_b, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = acc + x[tile_b, tile_k, tile_n].sum(dim=1)
            row_acc = row_acc + acc.sum(dim=1)
            out[tile_b, tile_n] = acc
        tot[tile_b] = row_acc
    return out, tot


def _reduced_columns_then_accumulated(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row's column reduction, after which a K loop keeps accumulating the row that is then stored."""
    b, n = x.shape
    k = y.shape[1]
    out = torch.empty([b, n], dtype=x.dtype, device=x.device)
    tot = torch.empty([b], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(b):
        for tile_n in hl.tile(n):
            acc = x[tile_b, tile_n]
            total = acc.sum(dim=1)
            for tile_k in hl.tile(k):
                acc = acc + y[tile_b, tile_k, tile_n].sum(dim=1)
            out[tile_b, tile_n] = acc
            tot[tile_b] = total
    return out, tot


def _column_lanes_config(lanes: int) -> helion.Config:
    """512 columns over ``512 // lanes`` threads: ``lanes`` strided column lanes."""
    return helion.Config.from_dict(
        {
            "block_sizes": [1, 512, 32],
            "num_threads": [1, 512 // lanes, 1],
            "cute_vector_widths": [1, 1, 1],
            "cute_lane_layouts": ["strided"] * 3,
        }
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
def test_a_stored_accumulator_restarts_in_every_lane_of_the_consume_pass(
    lanes: int, static_shapes: bool
) -> None:
    """The consume pass re-runs ``acc = 0`` per lane before the K loop.

    The K loop rewrites ``acc`` under its loop-output name; classified as
    spelled, the initializer was taken for lane-invariant and hoisted between
    the passes, so every lane after the first continued from the previous
    lane's final ``acc``: the stored columns of those lanes were wrong while
    the reduced row sum was right.
    """
    torch.manual_seed(0)
    x = torch.randn(3, 64, 512, device=DEVICE)
    kernel = helion.kernel(
        _accumulated_columns_reduced_and_stored,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = kernel.bind((x,))
    config = _column_lanes_config(lanes)
    code = bound.to_code(config)
    assert f"for lane_1 in range({lanes}):" in code
    assert "_lane_acc = " in code
    out, tot = bound.compile_config(config)(x)
    torch.testing.assert_close(out, x.sum(dim=1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tot, x.sum(dim=(1, 2)), rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
def test_a_reduction_folds_its_input_before_a_later_loop_accumulates_it(
    lanes: int, static_shapes: bool
) -> None:
    """``total = acc.sum()`` reads ``acc`` as loaded, not as the K loop left it.

    The K loop after the reduction rewrites ``acc`` under its loop-output
    name; a producer slice over the whole body took that loop into the
    accumulate pass, and the reduced row sum included the K accumulation.
    """
    torch.manual_seed(0)
    x = torch.randn(3, 512, device=DEVICE)
    y = torch.randn(3, 48, 512, device=DEVICE)
    kernel = helion.kernel(
        _reduced_columns_then_accumulated,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = kernel.bind((x, y))
    config = _column_lanes_config(lanes)
    code = bound.to_code(config)
    assert f"for lane_1 in range({lanes}):" in code
    assert "_lane_acc = " in code
    out, tot = bound.compile_config(config)(x, y)
    torch.testing.assert_close(out, x + y.sum(dim=1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tot, x.sum(dim=1), rtol=1e-4, atol=1e-3)


def _jagged_inputs(columns: int) -> tuple[torch.Tensor, torch.Tensor]:
    """17 sequences of 731 rows in all, one of them empty (jagged_layer_norm's test shape)."""
    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, 40, (17,), generator=generator)
    lengths[3] = 0
    lengths[-1] = 731 - int(lengths[:-1].sum())
    offsets = torch.zeros(18, dtype=torch.int64)
    offsets[1:] = lengths.cumsum(0)
    values = torch.randn((731, columns), generator=generator)
    return values.to(DEVICE), offsets.to(DEVICE)


def _jagged_row_sums_stored(
    x_values: torch.Tensor, x_offsets: torch.Tensor
) -> torch.Tensor:
    """Each sequence's row sums over a jagged device loop, their total stored."""
    total_rows, columns = x_values.shape
    batch = x_offsets.size(0) - 1
    tot = torch.empty([batch], dtype=x_values.dtype, device=x_values.device)
    x_flat = x_values.view(-1)
    for tile_b in hl.tile(batch):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        seq_lengths = ends - starts
        for tile_m in hl.tile(columns):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_values.dtype)
            for tile_k in hl.jagged_tile(seq_lengths):
                flat = (starts[:, None] + tile_k.index[None, :])[:, :, None] * columns
                flat = flat + tile_m.index[None, None, :]
                row_sums = row_sums + hl.load(x_flat, [flat]).sum(dim=1)
            tot[tile_b] = row_sums.sum(dim=1)
    return tot


def _jagged_row_sums_scaled(
    x_values: torch.Tensor, x_offsets: torch.Tensor
) -> torch.Tensor:
    """The row sums scaled by their total: the total is consumed per column."""
    total_rows, columns = x_values.shape
    batch = x_offsets.size(0) - 1
    out = torch.empty([batch, columns], dtype=x_values.dtype, device=x_values.device)
    x_flat = x_values.view(-1)
    for tile_b in hl.tile(batch):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        seq_lengths = ends - starts
        for tile_m in hl.tile(columns):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_values.dtype)
            for tile_k in hl.jagged_tile(seq_lengths):
                flat = (starts[:, None] + tile_k.index[None, :])[:, :, None] * columns
                flat = flat + tile_m.index[None, None, :]
                row_sums = row_sums + hl.load(x_flat, [flat]).sum(dim=1)
            total = row_sums.sum(dim=1)
            out[tile_b, tile_m] = row_sums * total[:, None]
    return out


def _jagged_lanes_config(lanes: int) -> helion.Config:
    """One row per CTA, the 512 columns on ``512 // lanes`` threads."""
    return helion.Config.from_dict(
        {
            "block_sizes": [1, 512, 32],
            "num_threads": [1, 512 // lanes, 1],
            "cute_vector_widths": [1, 1, 1],
            "cute_lane_layouts": ["strided"] * 3,
        }
    )


def _jagged_rows_config(lanes: int) -> helion.Config:
    """Two rows per CTA on the y axis: the max over the rows' lengths that
    bounds the jagged loop is a cross-thread collective in the lane body, so
    the two-pass split stays off it and the strided markers are restored or
    re-reduced.  (With one row per CTA that max is the row's own length and the
    body splits like any other.)"""
    return helion.Config.from_dict(
        {
            "block_sizes": [2, 512, 32],
            "num_threads": [2, 512 // lanes, 1],
            "cute_vector_widths": [1, 1, 1],
            "cute_lane_layouts": ["strided"] * 3,
        }
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
def test_a_stored_strided_reduction_totals_the_lanes_shares(
    lanes: int, static_shapes: bool
) -> None:
    """The jagged loop's collective trip count (the max over the two rows' lengths) keeps the two-pass split off this body; the restored per-lane shares used to be stored as they were (the last lane's half-row won), and are now totalled across the lanes before the store."""
    x, offsets = _jagged_inputs(512)
    kernel = helion.kernel(
        _jagged_row_sums_stored,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = kernel.bind((x, offsets))
    config = _jagged_rows_config(lanes)
    code = bound.to_code(config)
    assert code.count(f"for lane_1 in range({lanes}):") == 1
    assert "_lane_share = " in code and "_lane_total = " in code
    out = bound.compile_config(config)(x, offsets)
    expected = torch.stack(
        [x[int(offsets[b]) : int(offsets[b + 1])].sum() for b in range(17)]
    )
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
def test_a_strided_reduction_scaled_back_per_lane_is_declined(
    lanes: int, static_shapes: bool
) -> None:
    """Every lane's row sums are needed after the total, which neither the restore nor a lane-invariant tail provides; the two-row config, whose body cannot split, is declined instead of scaling by one lane's share."""
    x, offsets = _jagged_inputs(512)
    kernel = helion.kernel(
        _jagged_row_sums_scaled,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = kernel.bind((x, offsets))
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        bound.to_code(_jagged_rows_config(lanes))


def _jagged_row_sums_running_max(
    x_values: torch.Tensor, x_offsets: torch.Tensor
) -> torch.Tensor:
    """The row sums' max over each column tile feeds a running max across the tiles."""
    total_rows, columns = x_values.shape
    batch = x_offsets.size(0) - 1
    tot = torch.empty([batch], dtype=x_values.dtype, device=x_values.device)
    x_flat = x_values.view(-1)
    for tile_b in hl.tile(batch):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        seq_lengths = ends - starts
        m_acc = hl.full([tile_b], float("-inf"), dtype=x_values.dtype)
        for tile_m in hl.tile(columns):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_values.dtype)
            for tile_k in hl.jagged_tile(seq_lengths):
                flat = (starts[:, None] + tile_k.index[None, :])[:, :, None] * columns
                flat = flat + tile_m.index[None, None, :]
                row_sums = row_sums + hl.load(x_flat, [flat]).sum(dim=1)
            m_acc = torch.maximum(m_acc, row_sums.amax(dim=1))
        tot[tile_b] = m_acc
    return tot


def _jagged_row_sums_atomically_added(
    x_values: torch.Tensor, x_offsets: torch.Tensor
) -> torch.Tensor:
    """Each column tile's total is added atomically into the sequence's slot."""
    total_rows, columns = x_values.shape
    batch = x_offsets.size(0) - 1
    tot = torch.zeros([batch], dtype=x_values.dtype, device=x_values.device)
    x_flat = x_values.view(-1)
    for tile_b in hl.tile(batch):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        seq_lengths = ends - starts
        for tile_m in hl.tile(columns):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_values.dtype)
            for tile_k in hl.jagged_tile(seq_lengths):
                flat = (starts[:, None] + tile_k.index[None, :])[:, :, None] * columns
                flat = flat + tile_m.index[None, None, :]
                row_sums = row_sums + hl.load(x_flat, [flat]).sum(dim=1)
            hl.atomic_add(tot, [tile_b], row_sums.sum(dim=1))
    return tot


def _jagged_row_sums_sum_and_max_stored(
    x_values: torch.Tensor, x_offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two reductions of the row sums, both stored."""
    total_rows, columns = x_values.shape
    batch = x_offsets.size(0) - 1
    tot = torch.empty([batch], dtype=x_values.dtype, device=x_values.device)
    tot2 = torch.empty([batch], dtype=x_values.dtype, device=x_values.device)
    x_flat = x_values.view(-1)
    for tile_b in hl.tile(batch):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        seq_lengths = ends - starts
        for tile_m in hl.tile(columns):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_values.dtype)
            for tile_k in hl.jagged_tile(seq_lengths):
                flat = (starts[:, None] + tile_k.index[None, :])[:, :, None] * columns
                flat = flat + tile_m.index[None, None, :]
                row_sums = row_sums + hl.load(x_flat, [flat]).sum(dim=1)
            tot[tile_b] = row_sums.sum(dim=1)
            tot2[tile_b] = row_sums.amax(dim=1)
    return tot, tot2


def _jagged_row_means_stored(
    x_values: torch.Tensor, x_offsets: torch.Tensor
) -> torch.Tensor:
    """The mean of the row sums over the columns, stored."""
    total_rows, columns = x_values.shape
    batch = x_offsets.size(0) - 1
    tot = torch.empty([batch], dtype=x_values.dtype, device=x_values.device)
    x_flat = x_values.view(-1)
    for tile_b in hl.tile(batch):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        seq_lengths = ends - starts
        for tile_m in hl.tile(columns):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_values.dtype)
            for tile_k in hl.jagged_tile(seq_lengths):
                flat = (starts[:, None] + tile_k.index[None, :])[:, :, None] * columns
                flat = flat + tile_m.index[None, None, :]
                row_sums = row_sums + hl.load(x_flat, [flat]).sum(dim=1)
            tot[tile_b] = row_sums.mean(dim=1)
    return tot


def _jagged_row_sums(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Each sequence's column sums, zeros for the empty one."""
    return torch.stack(
        [x[int(offsets[b]) : int(offsets[b + 1])].sum(dim=0) for b in range(17)]
    )


def _jagged_column_block_config(lanes: int, block: int) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [1, block, 32],
            "num_threads": [1, block // lanes, 1],
            "cute_vector_widths": [1, 1, 1],
            "cute_lane_layouts": ["strided"] * 3,
        }
    )


def _jagged_kernel(fn: Callable[..., object], *, static_shapes: bool) -> helion.Kernel:
    return helion.kernel(
        fn, backend="cute", static_shapes=static_shapes, autotune_effort="none"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
def test_a_running_max_carry_keeps_the_per_lane_restore(
    lanes: int, static_shapes: bool
) -> None:
    """``m_acc = torch.maximum(m_acc, row_sums.amax(dim=1))`` renders as ``cute.math.max`` under casts: the max over the lanes of the lanes' maxes is exact, so the marker stays restored per lane."""
    x, offsets = _jagged_inputs(512)
    bound = _jagged_kernel(
        _jagged_row_sums_running_max, static_shapes=static_shapes
    ).bind((x, offsets))
    config = _jagged_rows_config(lanes)
    code = bound.to_code(config)
    assert code.count(f"for lane_1 in range({lanes}):") == 1
    assert "_lane_total" not in code
    assert "_reduced = _cute_grouped_reduce" in code
    out = bound.compile_config(config)(x, offsets)
    torch.testing.assert_close(
        out, _jagged_row_sums(x, offsets).amax(dim=1), rtol=1e-4, atol=1e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
def test_an_atomic_add_of_the_row_total_runs_once_on_the_lanes_total(
    lanes: int, static_shapes: bool
) -> None:
    """The emitted ``cute.arch.atomic_add(ptr, val=..., sem=...)`` is a lane-invariant consumer: the shares are totalled over the lanes and one guarded atomic adds the total."""
    x, offsets = _jagged_inputs(512)
    bound = _jagged_kernel(
        _jagged_row_sums_atomically_added, static_shapes=static_shapes
    ).bind((x, offsets))
    config = _jagged_rows_config(lanes)
    code = bound.to_code(config)
    assert "_lane_total" in code
    assert code.count("cute.arch.atomic_add(") == 1
    out = bound.compile_config(config)(x, offsets)
    torch.testing.assert_close(
        out, _jagged_row_sums(x, offsets).sum(dim=1), rtol=1e-4, atol=1e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lanes", [2, 4])
def test_two_stored_reductions_of_one_accumulator_take_one_schedule_in_both_shape_modes(
    lanes: int,
) -> None:
    """Under dynamic shapes the second reduction's masked input sits between the two markers; it joins the prefix, so the static and dynamic twins both total the shares and agree bitwise."""
    x, offsets = _jagged_inputs(512)
    config = _jagged_rows_config(lanes)
    outputs = []
    for static_shapes in (True, False):
        bound = _jagged_kernel(
            _jagged_row_sums_sum_and_max_stored, static_shapes=static_shapes
        ).bind((x, offsets))
        code = bound.to_code(config)
        assert code.count(f"for lane_1 in range({lanes}):") == 1
        assert (
            len(
                re.findall(
                    r"^\s*\w+_lane_total = cutlass\.Float32\((?:0|float\('-inf'\))\)$",
                    code,
                    re.MULTILINE,
                )
            )
            == 2
        )
        outputs.append(bound.compile_config(config)(x, offsets))
    (static_tot, static_max), (dynamic_tot, dynamic_max) = outputs
    row_sums = _jagged_row_sums(x, offsets)
    torch.testing.assert_close(static_tot, row_sums.sum(dim=1), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(static_max, row_sums.amax(dim=1), rtol=1e-4, atol=1e-3)
    assert torch.equal(static_tot, dynamic_tot)
    assert torch.equal(static_max, dynamic_max)


def _jagged_expectations(
    x: torch.Tensor, offsets: torch.Tensor
) -> dict[Callable[..., object], tuple[torch.Tensor, ...]]:
    row_sums = _jagged_row_sums(x, offsets)
    return {
        _jagged_row_sums_stored: (row_sums.sum(dim=1),),
        _jagged_row_sums_scaled: (row_sums * row_sums.sum(dim=1, keepdim=True),),
        _jagged_row_sums_running_max: (row_sums.amax(dim=1),),
        _jagged_row_sums_atomically_added: (row_sums.sum(dim=1),),
        _jagged_row_sums_sum_and_max_stored: (
            row_sums.sum(dim=1),
            row_sums.amax(dim=1),
        ),
        _jagged_row_means_stored: (row_sums.mean(dim=1),),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    "fn",
    [
        _jagged_row_sums_stored,
        _jagged_row_sums_scaled,
        _jagged_row_sums_running_max,
        _jagged_row_sums_atomically_added,
        _jagged_row_sums_sum_and_max_stored,
        _jagged_row_means_stored,
    ],
    ids=lambda fn: fn.__name__.removeprefix("_jagged_row_"),
)
def test_one_row_tiles_take_the_plain_two_pass_split(
    fn: Callable[..., object], static_shapes: bool
) -> None:
    """With one row per CTA the jagged loop's bound is the row's own length (a reduction over a block of one combined the column threads before, an unduplicatable collective that kept the split off); the body now splits into the accumulate and consume passes, including the per-column scaling the restore path declines."""
    x, offsets = _jagged_inputs(512)
    # Exact quarter-integer sums isolate the split from reduction-order
    # roundoff, which the per-column scaling can amplify near cancellation.
    x = (x * 4).round() * 0.25
    bound = _jagged_kernel(fn, static_shapes=static_shapes).bind((x, offsets))
    config = _jagged_lanes_config(2)
    code = bound.to_code(config)
    assert code.count("for lane_1 in range(2):") == 2
    assert "_lane_total" not in code and "_lane_share" not in code
    outputs = bound.compile_config(config)(x, offsets)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    for out, expected in zip(
        outputs, _jagged_expectations(x, offsets)[fn], strict=True
    ):
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lanes", [1, 2])
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
def test_a_mean_over_a_padded_column_block_divides_by_the_columns(
    lanes: int, static_shapes: bool
) -> None:
    """A 1024-column block over 512 columns masks half its elements out of the sum; the mean divides by the 512 the tile holds, not by the block (it gave half the mean)."""
    x, offsets = _jagged_inputs(512)
    bound = _jagged_kernel(_jagged_row_means_stored, static_shapes=static_shapes).bind(
        (x, offsets)
    )
    config = _jagged_column_block_config(lanes, 1024)
    code = bound.to_code(config)
    assert "1.0 / _BLOCK_SIZE_1" not in code and "/ _BLOCK_SIZE_1" not in code
    assert "- tile_offset_1" in code
    out = bound.compile_config(config)(x, offsets)
    torch.testing.assert_close(
        out, _jagged_row_sums(x, offsets).mean(dim=1), rtol=1e-4, atol=1e-4
    )


def _column_sums_added(x: torch.Tensor) -> torch.Tensor:
    """Every row tile's column sums added into ``out``: the full column sums."""
    b, m = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(b):
        for tile_m in hl.tile(m):
            hl.atomic_add(out, [tile_m], x[tile_b, tile_m].sum(dim=0))
    return out


def _column_means_added(x: torch.Tensor) -> torch.Tensor:
    b, m = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(b):
        for tile_m in hl.tile(m):
            hl.atomic_add(out, [tile_m], x[tile_b, tile_m].mean(dim=0))
    return out


def _grid_lane_config(bound: BoundKernel, lanes: int) -> helion.Config:
    """Eight rows per tile on ``8 // lanes`` row threads, 32 column threads."""
    config = dict(bound.config_spec.default_config().config)
    config.update(
        block_sizes=[8, 32],
        num_threads=[8 // lanes, 32],
        cute_vector_widths=[1, 1],
        cute_lane_layouts=["strided", "strided"],
    )
    return helion.Config.from_dict(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lanes", [8, 4])
@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
@pytest.mark.parametrize("reduction", ["sum", "mean"], ids=["sum", "mean"])
def test_an_atomic_consumer_of_an_interchanged_lane_reduction_runs_once(
    lanes: int, static_shapes: bool, reduction: str
) -> None:
    """The row lane loop around the column loop is interchanged for the column reduction; the first pass used to keep the atomic over its per-lane partials, so the total came out doubled."""
    x = torch.randn(37, 100, device=DEVICE)
    fn = _column_sums_added if reduction == "sum" else _column_means_added
    bound = _jagged_kernel(fn, static_shapes=static_shapes).bind((x,))
    config = _grid_lane_config(bound, lanes)
    code = bound.to_code(config)
    assert code.count("cute.arch.atomic_add(") == 1
    assert code.count(f"for lane_0 in range({lanes}):") == 1
    out = bound.compile_config(config)(x)
    if reduction == "sum":
        expected = x.sum(dim=0)
    else:
        expected = sum(x[r : r + 8].mean(dim=0) for r in range(0, 37, 8))
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)
