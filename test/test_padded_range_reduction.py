"""A grid kernel's reduction of a load over ``hl.arange`` of a static extent
that is not a power of two.

The range runs over ``next_power_of_2(n)`` lanes, its padding lanes masked,
and the load over it has that padded size, so the reduction dim is the padded
one: a loop over it would read the load and the range from outside, at their
full extent.  The dim stays persistent, keeping the reduction_loops slot older
versions offered, pinned; on CuTe the range runs on the reduction's lanes,
whose reductions mask the lanes past the range.
"""

from __future__ import annotations

import os
import re

import pytest
import torch

import helion
from helion import exc
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.padded_range_kernels import grid_offset_range_sum
from helion._testing.padded_range_kernels import grid_padded_range_sum
from helion._testing.padded_range_kernels import grid_padded_range_sum_and_max
from helion._testing.padded_range_kernels import grid_padded_range_sum_scales_a_row
from helion._testing.padded_range_kernels import grid_padded_range_sums_accumulated
from helion._testing.padded_range_kernels import grid_store_through_a_view_of_the_range
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _grid_sum_through_a_view_of_the_range(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        out[i] = hl.load(x, [i, hl.arange(n).view(n)]).sum(-1)
    return out


def _grid_padded_range_stats(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.shape
    mean = torch.empty([m], dtype=x.dtype, device=x.device)
    var = torch.empty([m, 1], dtype=x.dtype, device=x.device)
    var_mean = torch.empty([m, 1], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        row = hl.load(x, [i, hl.arange(n)])
        mean[i] = row.mean(-1)
        var[i, :], var_mean[i, :] = torch.var_mean(
            row, dim=-1, keepdim=True, correction=0
        )
    return mean, var, var_mean


def _grid_range_read_in_an_inner_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for i in hl.grid(x.size(0)):
        cols = hl.arange(3, 203)
        acc = hl.zeros([], dtype=torch.float32)
        for _ in hl.grid(2):
            acc = acc + hl.load(x, [i, cols]).sum(-1)
        out[i] = acc
    return out


def _grid_padded_range_rows_accumulate_then_reduce(x: torch.Tensor) -> torch.Tensor:
    m, k, n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for i in hl.grid(m):
        acc = hl.zeros([n], dtype=torch.float32)
        for j in hl.grid(k):
            acc = acc + hl.load(x, [i, j, hl.arange(n)])
        out[i] = acc.sum(-1)
    return out


def _grid_reduced_range_scales_a_row(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        scale = hl.arange(200).sum().to(x.dtype)
        out[i] = (x[i, :] * scale).sum(-1)
    return out


def _tile_reduced_range_scales_a_row(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        s = hl.load(w, [tile_m, hl.arange(3000)]).sum(-1)
        out[tile_m] = (x[tile_m, :] * s[:, None]).sum(-1)
    return out


def _mean_divisors(code: str) -> list[int]:
    """The constants the sums of the mean lowerings (``*_extra``) are divided
    by, followed through the casts of the divisor."""
    assigned = dict(re.findall(r"^\s*(v_\d+) = (.+)$", code, flags=re.MULTILINE))

    def constant(name: str) -> int:
        value = assigned[name]
        cast = re.fullmatch(
            r"(?:cutlass\.Float32\((v_\d+)\)|tl\.cast\((v_\d+), .*\))", value
        )
        if cast is not None:
            return constant(cast.group(1) or cast.group(2))
        return int(value)

    return [
        constant(divisor)
        for divisor in re.findall(
            r"= \w+_extra\w* / (v_\d+)$", code, flags=re.MULTILINE
        )
    ]


def _range_summed_directly_and_through_a_load(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for i in hl.grid(x.size(0)):
        cols = hl.arange(5)
        out[i] = hl.load(x, [i, cols]).sum(-1) + cols.sum()
    return out


def _bind(fn: object, *shapes: tuple[int, ...]) -> helion.BoundKernel:
    return cpu_bind(
        helion.kernel(fn, static_shapes=True), tuple(torch.empty(s) for s in shapes)
    )


@onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager(
    "inspects config specs and generated code; ref mode has neither"
)
class TestPaddedRangeReduction(TestCase):
    def test_padded_reduction_dim_stays_persistent(self) -> None:
        # 3000 pads to 4096, past the thread budget that rolls a slot's dim by
        # default.  Saved chunk sizes still load, as the persistent reduction
        # (a chunk never compiled on Triton and folded one chunk on CuTe).
        for fn, width, length, padded in (
            (grid_padded_range_sum, 200, 200, 256),
            (grid_padded_range_sum, 3000, 3000, 4096),
            (grid_offset_range_sum, 210, 200, 256),
            # A view of the range in its own shape is the range.
            (_grid_sum_through_a_view_of_the_range, 200, 200, 256),
        ):
            with self.subTest(kernel=fn.__name__, width=width), cpu_render_target():
                bound = _bind(fn, (16, width))
                spec = bound.config_spec
                rdims = [info.size for info in bound.env.block_sizes if info.reduction]
                self.assertEqual(rdims, [padded])
                self.assertEqual(spec.default_config().reduction_loops, [None])
                for saved in (None, 32, padded):
                    config = spec.normalized_config(
                        helion.Config(reduction_loops=[saved])
                    )
                    self.assertEqual(config.reduction_loops, [None])
                code = bound.to_code(
                    spec.normalized_config(helion.Config(reduction_loops=[32]))
                )
                self.assertNotIn("roffset", code)
                (slot,) = spec.reduction_loops
                self.assertTrue(slot.pinned)
                if _get_backend() == "cute":
                    # The range takes the reduction's index, whose lanes past
                    # the range's length the load masks, also when the row
                    # has columns there; no lane loop of its own.
                    self.assertNotIn("arange_lane", code)
                    (index,) = set(
                        re.findall(
                            rf"cutlass\.Int32\((indices_\d+)\) < {length}\b", code
                        )
                    )
                    self.assertIn(f"{index} = cutlass.Int32(", code)
                    # The masked load is zero past the range already, the
                    # sum's identity: its input is not masked again.
                    self.assertNotIn("padded_range_masked", code)

    def test_a_range_reduced_away_does_not_pin_the_row(self) -> None:
        # The range is summed to a scalar, or over its own 3000 columns,
        # before the row's reduction: it never reaches the row's lanes, whose
        # dim keeps its chunks (and on CuTe a rolled default).
        with cpu_render_target():
            grid = _bind(_grid_reduced_range_scales_a_row, (16, 256))
            tile = _bind(_tile_reduced_range_scales_a_row, (64, 4096), (64, 3000))
            grid_config = grid.config_spec.normalized_config(
                helion.Config(reduction_loops=[32])
            )
            tile_config = tile.config_spec.normalized_config(
                helion.Config(block_sizes=[1], reduction_loops=[32])
            )
            tile_default = tile.config_spec.default_config()
        for bound in (grid, tile):
            (slot,) = bound.config_spec.reduction_loops
            self.assertFalse(slot.pinned)
        self.assertEqual(grid_config.reduction_loops, [32])
        self.assertEqual(tile_config.reduction_loops, [32])
        if _get_backend() == "cute":
            self.assertIsNotNone(tile_default.reduction_loops[0])

    def test_a_pinned_dim_offers_no_reduction_fusion(self) -> None:
        # Its two reductions would fuse in a loop no config can roll.
        with cpu_render_target():
            bound = _bind(grid_padded_range_sum_and_max, (16, 200))
        self.assertTrue(bound.config_spec.reduction_loops[0].pinned)
        self.assertFalse(bound.config_spec.reduction_fusion_available)

    @onlyBackends(["cute"])
    def test_reductions_mask_the_lanes_past_the_range(self) -> None:
        # The reductions run over all 256 (or 8) lanes, and ``v * 2 + 1`` is
        # one past the range: each reduction's input takes its identity there.
        for width, lanes in ((200, 256), (5, 8)):
            with self.subTest(width=width), cpu_render_target():
                bound = _bind(grid_padded_range_sum_and_max, (16, width))
                code = bound.to_code(bound.config_spec.default_config())
            self.assertEqual(
                re.findall(
                    rf"padded_range_masked_\d+ = [^\n]* if cutlass\.Int32\(indices_\d+\)"
                    rf" < {width} else cutlass\.Float32\((0|float\('-inf'\))\)",
                    code,
                ),
                ["0", "float('-inf')"],
                code,
            )
            self.assertNotIn(f"< {lanes} else", code)

    def test_a_mean_and_var_mean_count_the_range(self) -> None:
        # The row has 200 elements, not the 256 of its padded dim, and on CuTe
        # the squared deviations from its mean are masked past them.
        with cpu_render_target():
            bound = _bind(_grid_padded_range_stats, (16, 200))
            code = bound.to_code(bound.config_spec.default_config())
        self.assertEqual(_mean_divisors(code), [200] * 3, code)
        if _get_backend() == "cute":
            self.assertRegex(
                code,
                r"(v_\d+) = (v_\d+) \* \2\n\s+padded_range_masked_\d+ = "
                r"cutlass\.Float32\(\1\) if cutlass\.Int32\(indices_\d+\) < 200 ",
            )

    @onlyBackends(["cute"])
    def test_a_reduced_range_may_reach_a_loop(self) -> None:
        # The sum of the range's lanes is a scalar: carried by an inner loop
        # or read in one, it needs no mask, and the range runs on the
        # reduction's lanes.  Treating it as the range escaping into the loop
        # ran the range on a lane loop of its own around the reduction, which
        # summed 256 copies of one element a pass (512, not 400, for an
        # all-ones accumulator over two 200-column rows).
        for fn, shapes in (
            (grid_padded_range_sums_accumulated, [(16, 2, 200)]),
            (grid_padded_range_sum_scales_a_row, [(16, 200), (16, 4)]),
        ):
            with self.subTest(kernel=fn.__name__), cpu_render_target():
                bound = _bind(fn, *shapes)
                code = bound.to_code(bound.config_spec.default_config())
            self.assertNotIn("arange_lane", code)
            (index,) = set(re.findall(r"cutlass\.Int32\((indices_\d+)\) < 200\b", code))
            self.assertIn(f"{index} = cutlass.Int32(cute.arch.thread_idx()[0])", code)
            self.assertIn("group_count=1", code)

    @onlyBackends(["cute"])
    def test_a_view_of_the_range_keeps_its_length_mask(self) -> None:
        # ``hl.arange(1, 6).reshape(5)`` beside the sum of an 8-column row
        # runs on the row's 8 lanes: the load and store through the view are
        # masked to the range's 5 lanes (lanes 5 and 6 wrote columns 6 and 7).
        with cpu_render_target():
            bound = _bind(grid_store_through_a_view_of_the_range, (16, 8))
            code = bound.to_code(bound.config_spec.default_config())
        (index,) = set(re.findall(r"iota = 1 \+ (indices_\d+)", code))
        guard = f"cutlass.Int32({index}) < 5"
        self.assertRegex(code, rf"load_\d+ = [^\n]*\.load\(\) if {re.escape(guard)}")
        self.assertIn(f"if {guard}", code)

    @onlyBackends(["cute"])
    def test_a_range_on_lanes_of_its_own_reduced_through_a_load_is_refused(
        self,
    ) -> None:
        # ``cols.sum()`` gives the range a reduction dim of its own length, 5,
        # while the load over it has the padded 8 and its sum runs on that
        # dim's lanes, which hold copies of one element: refused.
        with cpu_render_target():
            bound = _bind(_range_summed_directly_and_through_a_load, (16, 5))
            with self.assertRaisesRegex(
                exc.BackendUnsupported, "does not run on the reduction's lanes"
            ):
                bound.to_code(bound.config_spec.default_config())

    @onlyBackends(["cute"])
    def test_lanes_carried_out_of_a_loop_into_a_reduction_are_refused(self) -> None:
        # The accumulated row carries the range's lanes out of the inner loop
        # into the sum, past the placeholder its masks cannot follow: refused
        # rather than run on a lane loop of the range's own around the sum.
        with cpu_render_target():
            bound = _bind(_grid_padded_range_rows_accumulate_then_reduce, (16, 2, 200))
            with self.assertRaisesRegex(
                exc.BackendUnsupported, "padded lanes reach a loop or branch body"
            ):
                bound.to_code(bound.config_spec.default_config())

    @onlyBackends(["cute"])
    def test_a_range_read_in_an_inner_loop_is_refused(self) -> None:
        # Its loads in the loop body read it through a placeholder, past which
        # their mask cannot follow the range to its length (they summed
        # columns 3..209 of a 210-column row): refused, as before.
        with cpu_render_target():
            bound = _bind(_grid_range_read_in_an_inner_loop, (16, 210))
            with self.assertRaisesRegex(
                exc.BackendUnsupported, "padded lanes reach a loop or branch body"
            ):
                bound.to_code(bound.config_spec.default_config())


def _half_row_variance(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    var = torch.empty([m, 1], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        row = hl.load(x, [i, hl.arange(n)])
        var[i, :], _ = torch.var_mean(row, dim=-1, keepdim=True, correction=0)
    return var


def _sorted_range_sum(x: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        row = hl.load(x, [i, hl.arange(5)])
        values, _ = torch.sort(row, dim=-1)
        out[i] = values.sum(-1)
    return out


def _running_max(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.maximum(left, right)


def _reverse_scans_of_the_range(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = x.size(0)
    prods = torch.empty([m], dtype=x.dtype, device=x.device)
    maxes = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        row = hl.load(x, [i, hl.arange(5)])
        prods[i] = hl.cumprod(row, 0, reverse=True).sum(-1)
        maxes[i] = hl.associative_scan(_running_max, row, 0, reverse=True).sum(-1)
    return prods, maxes


def _full_row_sorted_beside_a_range_sum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m = x.size(0)
    sums = torch.empty([m], dtype=x.dtype, device=x.device)
    rows = torch.empty_like(y)
    for i in hl.grid(m):
        sums[i] = hl.load(x, [i, hl.arange(5)]).sum(-1)
        values, _ = torch.sort(y[i, :], dim=-1)
        rows[i, :] = values
    return sums, rows


@onlyBackends(["cute"])
def test_masked_half_precision_reductions_select_in_float32() -> None:
    # The lowering computes a float16 row's mean and squared deviations in
    # float32; a select back in float16 rounded them before the sums, and a
    # squared deviation of 400 overflowed.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_half_row_variance, static_shapes=True),
            (torch.empty(4, 5, dtype=torch.float16),),
        )
        code = bound.to_code(bound.config_spec.default_config())
    selects = [line for line in code.splitlines() if "padded_range_masked_" in line]
    assert selects and all("cutlass.Float16(" not in line for line in selects), code


@onlyBackends(["cute"])
def test_a_sort_over_the_range_ranks_its_lanes_only() -> None:
    # The sort walks the 8 lanes of the row's padded reduction dim; the 3 past
    # the range hold no elements and must neither take nor shift a rank.
    with cpu_render_target():
        bound = _bind(_sorted_range_sum, (4, 8))
        code = bound.to_code(bound.config_spec.default_config())
    (selected,) = [line for line in code.splitlines() if "sort_selected =" in line]
    assert "< 5" in selected, code


@onlyBackends(["cute"])
def test_reverse_scans_of_the_range_walk_its_lanes_only() -> None:
    # The range's 3 padding lanes on the row's 8 hold the load's zeros, which
    # the shuffle scan folded into each element (five ones' reverse cumprod
    # summed to 0); the serial walk stops at the range's 5.
    with cpu_render_target():
        bound = _bind(_reverse_scans_of_the_range, (4, 8))
        code = bound.to_code(bound.config_spec.default_config())
    assert "shuffle_sync" not in code, code
    assert "< 5" in code, code


@onlyBackends(["cute"])
def test_a_full_row_sorted_beside_the_range_ranks_all_its_lanes() -> None:
    # The range takes the lanes of the 8-wide reduction dim that the sorted
    # row's slice allocates, and only a walk over the range's own values ends
    # at its 5: this sort selected none of its last 3 outputs.
    with cpu_render_target():
        bound = _bind(_full_row_sorted_beside_a_range_sum, (4, 8), (4, 8))
        code = bound.to_code(bound.config_spec.default_config())
    (selected,) = [line for line in code.splitlines() if "sort_selected =" in line]
    assert "< 5" not in selected, code


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
def test_half_variance_and_sorted_sum_of_the_range_match_eager() -> None:
    device = helion._testing.DEVICE
    x = torch.zeros(16, 5, dtype=torch.float16, device=device)
    x[:, 0] = 500.0
    _, var = helion._testing.code_and_output(
        helion.kernel(_half_row_variance, static_shapes=True), (x,)
    )
    torch.testing.assert_close(
        var, x.float().var(-1, correction=0, keepdim=True).half()
    )
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, -10.0, -20.0, -30.0])
    _, out = helion._testing.code_and_output(
        helion.kernel(_sorted_range_sum, static_shapes=True),
        (values.repeat(16, 1).to(device),),
    )
    torch.testing.assert_close(out, torch.full([16], 15.0, device=device))


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
def test_a_full_row_sorted_beside_the_range_matches_eager() -> None:
    device = helion._testing.DEVICE
    x = torch.randn(16, 8, device=device)
    y = torch.randn(16, 8, device=device)
    _, (sums, rows) = helion._testing.code_and_output(
        helion.kernel(_full_row_sorted_beside_a_range_sum, static_shapes=True),
        (x, y),
    )
    torch.testing.assert_close(sums, x[:, :5].sum(-1))
    torch.testing.assert_close(rows, y.sort(dim=-1).values)


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
@pytest.mark.xfail(
    _get_backend() == "triton" and os.environ.get("HELION_INTERPRET") != "1",
    reason="Triton's reverse scans fold the range's padding lanes too, as on "
    "origin/main: their validity flag follows a tile's mask, and the range's "
    "dim has none",
    strict=True,
)
def test_reverse_scans_of_the_range_match_eager() -> None:
    x = -torch.rand(16, 8, device=helion._testing.DEVICE) - 0.5
    _, (prods, maxes) = helion._testing.code_and_output(
        helion.kernel(_reverse_scans_of_the_range, static_shapes=True), (x,)
    )
    row = x[:, :5].flip(-1)
    torch.testing.assert_close(prods, row.cumprod(-1).sum(-1))
    torch.testing.assert_close(maxes, row.cummax(-1).values.sum(-1))


def _sum_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return a + b


def _max_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


def _range_reduced_by_hl_reduce(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, _ = x.shape
    total = torch.empty([m], dtype=x.dtype, device=x.device)
    top = torch.empty([m], dtype=x.dtype, device=x.device)
    counted = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        row = hl.load(x, [i, hl.arange(200)])
        total[i] = hl.reduce(_sum_combine, row + 1.0, dim=0)
        top[i] = hl.reduce(_max_combine, row, dim=0, other=float("-inf"))
        counted[i] = hl.reduce(_sum_combine, row, dim=0, other=1.0)
    return total, top, counted


@onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_hl_reduce_takes_its_other_past_the_range() -> None:
    # The combine functions' inputs are masked on their tiles' masks, which
    # do not cover the 56 lanes past the range: those take each reduction's
    # ``other`` (0, -inf, 1.0), as masked tile elements do, also where it is
    # not the combine's identity; on Triton before ``tl.reduce``.
    with cpu_render_target():
        bound = _bind(_range_reduced_by_hl_reduce, (16, 200))
        code = bound.to_code(bound.config_spec.default_config())
    if _get_backend() == "cute":
        pattern = (
            r"padded_range_masked_\d+ = [^\n]* < 200 "
            r"else cutlass\.Float32\((0|float\('-inf'\)|1\.0)\)"
        )
    else:
        pattern = (
            r"tl\.reduce\(tl\.where\(tl\.arange\(0, 256\) < 200, \w+, "
            r"(0|float\('-inf'\)|1\.0)\)"
        )
    assert re.findall(pattern, code) == ["0", "float('-inf')", "1.0"], code


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager(
    "ref mode's hl.arange has no padding lanes for other to fill"
)
def test_hl_reduce_of_the_range_matches_eager() -> None:
    # A negative row, whose padding's 0 would win the max.
    x = -torch.rand(16, 200, device=helion._testing.DEVICE) - 1.0
    _, (total, top, counted) = helion._testing.code_and_output(
        helion.kernel(_range_reduced_by_hl_reduce, static_shapes=True), (x,)
    )
    torch.testing.assert_close(total, (x + 1.0).sum(-1), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(top, x.amax(-1))
    # Each of the 56 padding lanes counts its ``other``.
    torch.testing.assert_close(counted, x.sum(-1) + 56.0, rtol=1e-4, atol=1e-3)


def _tile_rows_reduced_by_hl_reduce(x: torch.Tensor) -> torch.Tensor:
    m, k, _ = x.shape
    out = torch.empty([m, k], dtype=x.dtype, device=x.device)
    for tile_m, tile_k in hl.tile([m, k]):
        row = hl.load(x, [tile_m, tile_k, hl.arange(5)])
        out[tile_m, tile_k] = hl.reduce(_sum_combine, row + 1.0, dim=2)
    return out


def _tile_range_reduced_by_hl_reduce(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = hl.reduce(
            _sum_combine, hl.load(x, [tile_m, hl.arange(200)]) + 1.0, dim=1
        )
    return out


_FLATTENED = helion.Config(block_sizes=[2, 2], flatten_loops=[True])


@onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_hl_reduce_masks_the_range_on_the_physical_dims() -> None:
    # A flattened tile keeps tile_m and tile_k in one dimension, so the mask
    # of the 3 lanes past the range spans the value's two physical dims, and
    # tl.reduce reduces the second; a three-dim mask broadcast the value.
    with cpu_render_target():
        bound = _bind(_tile_rows_reduced_by_hl_reduce, (4, 4, 8))
        code = bound.to_code(bound.config_spec.normalized_config(_FLATTENED))
    assert re.search(
        r"tl\.reduce\(tl\.where\(\(tl\.arange\(0, 8\) < 5\)\[None, :\], \w+, 0\), 1,",
        code,
    ), code


@onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_hl_reduce_of_a_range_with_its_own_reduction_dim_is_masked_once() -> None:
    # The range's reduction dim, 200 long, masks the lanes past it already
    # (``_mask_to``).
    with cpu_render_target():
        bound = _bind(_tile_range_reduced_by_hl_reduce, (16, 200))
        code = bound.to_code(bound.config_spec.default_config())
    assert re.search(r"tl\.reduce\(_mask_to, 1,", code), code
    assert "tl.arange(0, 256) < 200" not in code, code


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@onlyBackends(["triton"])
def test_hl_reduce_of_a_flattened_tile_matches_eager() -> None:
    x = torch.rand(4, 4, 8, device=helion._testing.DEVICE)
    _, out = helion._testing.code_and_output(
        helion.kernel(_tile_rows_reduced_by_hl_reduce, static_shapes=True),
        (x,),
        **_FLATTENED.config,
    )
    torch.testing.assert_close(out, (x[..., :5] + 1.0).sum(-1))


def _two_ranges_reduced_by_hl_reduce(x: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        row = hl.load(x, [i, hl.arange(5), hl.arange(3)])
        # The aten sums give both padded dims (8 and 4 lanes) reduction dims.
        out[i] = hl.reduce(_sum_combine, row, other=1.0) + row.sum(-1).sum(-1)
    return out


def _argmax_combine(
    left: tuple[torch.Tensor, torch.Tensor], right: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    take_right = right[0] > left[0]
    return (
        torch.where(take_right, right[0], left[0]),
        torch.where(take_right, right[1], left[1]),
    )


def _range_argmax_by_hl_reduce(
    x: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, _ = x.shape
    values = torch.empty([m], dtype=x.dtype, device=x.device)
    indices = torch.empty([m], dtype=idx.dtype, device=x.device)
    sums = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        row = hl.load(x, [i, hl.arange(5)])
        values[i], indices[i] = hl.reduce(
            _argmax_combine, (row, hl.load(idx, [i, hl.arange(5)])), other=(0.0, -1)
        )
        # The sum gives the range's 8 lanes a reduction dim.
        sums[i] = row.sum(-1)
    return values, indices, sums


@onlyBackends(["cute"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_hl_reduce_fills_the_lanes_past_every_range_before_reducing() -> None:
    # One select gives the lanes past either range ``other`` before the first
    # reduction: one per dim replaced each padded row's sum with one ``other``.
    # An arg-reduction's padding lanes take the ``other`` pair, not the
    # identities of the reductions that find the fold's element.
    with cpu_render_target():
        bound = _bind(_two_ranges_reduced_by_hl_reduce, (4, 5, 3))
        code = bound.to_code(bound.config_spec.default_config())
        fills = [line for line in code.splitlines() if "cutlass.Float32(1.0)" in line]
        assert len(fills) == 1 and "< 5 and " in fills[0] and "< 3" in fills[0], code
        bound = cpu_bind(
            helion.kernel(_range_argmax_by_hl_reduce, static_shapes=True),
            (torch.empty(4, 5), torch.empty(4, 5, dtype=torch.int64)),
        )
        code = bound.to_code(bound.config_spec.default_config())
    masks = re.findall(r"padded_range_masked_\d+ = [^\n]* < 5 else ([^\n]*)", code)
    assert masks == ["cutlass.Float32(0)", "cutlass.Int64(-1)"], code


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager(
    "ref mode's hl.arange has no padding lanes for other to fill"
)
def test_hl_reduce_takes_its_other_past_every_range_and_in_arg_reductions() -> None:
    # The 8 x 4 lanes of the load hold 15 elements: the 17 past either range
    # count ``other`` (1.0) once each.
    x = torch.randn(4, 5, 3, device=helion._testing.DEVICE)
    _, out = helion._testing.code_and_output(
        helion.kernel(_two_ranges_reduced_by_hl_reduce, static_shapes=True), (x,)
    )
    torch.testing.assert_close(out, 2 * x.sum((1, 2)) + 17.0, rtol=1e-4, atol=1e-4)
    # The padding lanes' (0.0, -1) beats the five negative values.
    x = -torch.rand(4, 5, device=helion._testing.DEVICE) - 1.0
    idx = torch.arange(5, device=helion._testing.DEVICE).repeat(4, 1)
    _, (values, indices, _) = helion._testing.code_and_output(
        helion.kernel(_range_argmax_by_hl_reduce, static_shapes=True), (x, idx)
    )
    assert values.tolist() == [0.0] * 4, values
    assert indices.tolist() == [-1] * 4, indices


def _range_outer_product_sums(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        out[i] = (hl.arange(200)[:, None] * x[i, :][None, :]).sum(-1).sum(-1)
    return out


@onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("inspects the config spec; ref mode has none")
def test_a_range_on_another_axis_does_not_pin_the_row() -> None:
    # The range pads the product's first axis, which has a reduction dim of
    # its own (200), not the row's 256 columns: their dim keeps its chunks.
    with cpu_render_target():
        bound = _bind(_range_outer_product_sums, (4, 256))
        config = bound.config_spec.normalized_config(
            helion.Config(reduction_loops=[32])
        )
    (slot,) = bound.config_spec.reduction_loops
    assert not slot.pinned
    assert config.reduction_loops == [32]


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
def test_a_range_on_another_axis_reduces_in_chunks_of_the_row() -> None:
    x = torch.randn(4, 256, device=helion._testing.DEVICE)
    _, out = helion._testing.code_and_output(
        helion.kernel(_range_outer_product_sums, static_shapes=True),
        (x,),
        reduction_loops=[32],
    )
    torch.testing.assert_close(out, 19900.0 * x.sum(-1), rtol=1e-4, atol=1e-2)


def _tile_range_rows_amax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 200], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        rows = hl.arange(200)
        out[tile, rows] = x[tile, rows, :].amax(-1)
    return out


def _grid_range_rows_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([200], dtype=x.dtype, device=x.device)
    for _ in hl.grid(1):
        rows = hl.arange(200)
        out[rows] = hl.load(x, [rows, slice(None)]).sum(-1)
    return out


_ROWS_KERNELS = [
    (_tile_range_rows_amax, (4, 200, 256), {"block_sizes": [2]}),
    (_grid_range_rows_sum, (200, 256), {}),
]


@onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
@pytest.mark.parametrize(("fn", "shape", "config"), _ROWS_KERNELS, ids=["tile", "grid"])
def test_a_range_indexing_another_dim_of_the_load_pins_nothing(
    fn: object, shape: tuple[int, ...], config: dict[str, object]
) -> None:
    # ``rows`` indexes the load's second dim, which has a reduction dim of its
    # own (200): the 256 columns' sum reads no padding lanes, so CuTe needs no
    # mask and refuses nothing, and their dim keeps its chunks.
    with cpu_render_target():
        bound = _bind(fn, shape)
        bound.to_code(bound.config_spec.default_config())
        normalized = bound.config_spec.normalized_config(
            helion.Config(**config, reduction_loops=[32])
        )
    (slot,) = bound.config_spec.reduction_loops
    assert not slot.pinned
    assert normalized.reduction_loops == [32]


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(("fn", "shape", "config"), _ROWS_KERNELS, ids=["tile", "grid"])
@pytest.mark.parametrize("loops", [None, 32])
def test_a_range_indexing_another_dim_of_the_load_matches_eager(
    fn: object, shape: tuple[int, ...], config: dict[str, object], loops: int | None
) -> None:
    x = torch.randn(*shape, device=helion._testing.DEVICE)
    _, out = helion._testing.code_and_output(
        helion.kernel(fn, static_shapes=True),
        (x,),
        **config,
        reduction_loops=[loops],
    )
    expected = x.amax(-1) if fn is _tile_range_rows_amax else x.sum(-1)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def _tuple_sum_combine(
    left: tuple[torch.Tensor, torch.Tensor], right: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    return left[0] + right[0], left[1] + right[1]


def _range_and_row_reduced_by_hl_reduce(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, _ = x.shape
    ranged = torch.empty([m], dtype=x.dtype, device=x.device)
    full = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        ranged[i], full[i] = hl.reduce(
            _tuple_sum_combine,
            (hl.load(x, [i, hl.arange(5)]), x[i, :]),
            dim=0,
            other=1.0,
        )
    return ranged, full


@onlyBackends(["cute"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_hl_reduce_fills_only_the_inputs_that_read_the_range() -> None:
    # The full row beside the range keeps its columns 5 to 7, as on Triton,
    # which masks each input of ``tl.reduce`` on its own.
    with cpu_render_target():
        bound = _bind(_range_and_row_reduced_by_hl_reduce, (4, 8))
        code = bound.to_code(bound.config_spec.default_config())
    masks = re.findall(r"padded_range_masked_\d+ = [^\n]*", code)
    assert len(masks) == 1 and masks[0].endswith("else cutlass.Float32(1.0)"), code


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager(
    "ref mode's hl.arange has no padding lanes for other to fill"
)
def test_hl_reduce_of_a_range_beside_a_full_row_matches_eager() -> None:
    x = torch.randn(4, 8, device=helion._testing.DEVICE)
    _, (ranged, full) = helion._testing.code_and_output(
        helion.kernel(_range_and_row_reduced_by_hl_reduce, static_shapes=True), (x,)
    )
    # The 3 lanes past the range count ``other``; the row has no padding.
    torch.testing.assert_close(ranged, x[:, :5].sum(-1) + 3.0, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(full, x.sum(-1), rtol=1e-4, atol=1e-4)
