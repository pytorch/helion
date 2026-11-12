"""Loads-first split of a V-loop's scalar loads under ``cute_lane_unroll``
(``helion/_compiler/cute/unroll_lane_loads.py``).

A thread that holds a vector of rows of a tile loaded along its columns (the
rows of a transposed store's packet) loads each row's element with a scalar
load inside the V-loop, behind the previous row's stores."""

from __future__ import annotations

import ast
import itertools
import re

import pytest
import torch

import helion
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.ast_read_writes import HELION_VEC_LANE_OF_ATTR
from helion._compiler.cute.unroll_lane_loads import _unroll_one
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import kernel_body
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pad_and_transpose(
    x: torch.Tensor, padded: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    out_t = torch.empty([padded, rows], dtype=x.dtype, device=x.device)
    out = torch.empty([rows, padded], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, padded]):
        in_range = tile_c.index < cols
        value = hl.load(x, [tile_r, tile_c], extra_mask=in_range[None, :])
        value = torch.where(in_range[None, :], value, 0.0)
        out_t[tile_c, tile_r] = value.permute(1, 0)
        out[tile_r, tile_c] = value
    return out_t, out


# Each thread holds 16 consecutive rows (one 32-byte packet of ``out_t``) of
# two strided columns; the rows' loads are scalar, the columns contiguous.
_ROW_PACKETS = {
    "block_sizes": [32, 64],
    "num_threads": [2, 32],
    "loop_orders": [[1, 0]],
    "cute_vector_widths": [16, 1],
    "cute_lane_layouts": ["strided", "strided"],
}
# One column per thread: the column lane loop has a single trip.
_ONE_COLUMN = {**_ROW_PACKETS, "block_sizes": [32, 32]}


def _render(
    kernel: helion.Kernel, args: tuple[object, ...], config: dict[str, object]
) -> str:
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
        return bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**config))
        )


def _code(config: dict[str, object], rows: int = 64, cols: int = 157) -> str:
    args = (torch.randn(rows, cols).to(torch.bfloat16), cols + 3)
    return _render(_pad_and_transpose, args, config)


def _assert_loads_first(code: str, copies: str) -> None:
    body = kernel_body(code)
    # Two copies of the V-loop: the loads, then the compute and stores, which
    # read the loads back from the trace-time list.
    assert body.count("for vec_lane_0 in cutlass.range_constexpr(16):") == 2
    assert "_lane_unroll_loads_0.append(value)" in body
    assert f"value = _lane_unroll_loads_0[{copies}]" in body
    first_store = min(body.index("(out.iterator +"), body.index("_cute_store_u16_vec("))
    assert body.rindex(".load()") < first_store


def test_the_vloop_loads_issue_before_the_stores() -> None:
    _assert_loads_first(
        _code({**_ROW_PACKETS, "cute_lane_unroll": 2}),
        "(lane_1 * 1 + lane_0) * 16 + vec_lane_0",
    )


def test_a_single_column_lane_still_splits_its_vloop() -> None:
    _assert_loads_first(
        _code({**_ONE_COLUMN, "cute_lane_unroll": 2}), "lane_0 * 16 + vec_lane_0"
    )


def test_a_split_at_factor_one_keeps_the_lane_loop_rolled() -> None:
    # Four row packets per column lane (64 loads a lane body holds) leave the
    # lane budget no second copy: the split takes a factor of one, and the
    # column lane loop stays rolled around both phases, each trip loading its
    # rows before it stores them.
    config = {**_ROW_PACKETS, "block_sizes": [128, 64], "cute_lane_unroll": 2}
    body = kernel_body(_code(config, rows=256))
    assert "for lane_1_blk in range(2):" in body
    assert body.count("for lane_1_u in cutlass.range_constexpr(1):") == 2
    assert (
        "value = _lane_unroll_loads_0[(lane_1_u * 4 + lane_0) * 16 + vec_lane_0]"
        in body
    )
    first_store = min(body.index("(out.iterator +"), body.index("_cute_store_u16_vec("))
    assert body.rindex(".load()") < first_store


def test_a_split_past_the_lane_budget_keeps_the_rolled_order() -> None:
    # Eight row packets of 16 per column lane: 128 loads a lane body would
    # hold, past the unroll's 64.
    config = {**_ROW_PACKETS, "block_sizes": [256, 64], "cute_lane_unroll": 2}
    body = kernel_body(_code(config, rows=512))
    assert "_lane_unroll_loads_" not in body


@pytest.mark.parametrize("config", [_ROW_PACKETS, _ONE_COLUMN])
def test_the_vloop_keeps_its_order_without_the_knob(
    config: dict[str, object],
) -> None:
    body = kernel_body(_code(config))
    assert "_lane_unroll_loads_" not in body
    assert body.count("for vec_lane_0 in cutlass.range_constexpr(16):") == 1


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _add_transposed(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    out_t = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    out = torch.empty([rows, cols], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = x[tile_r, tile_c] + y[tile_c, tile_r].permute(1, 0)
        out_t[tile_c, tile_r] = value.permute(1, 0)
        out[tile_r, tile_c] = value
    return out_t, out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scale_transposed(
    x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    out_t = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    out = torch.empty([rows, cols], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = x[tile_r, tile_c] * scale[tile_c][None, :]
        out_t[tile_c, tile_r] = value.permute(1, 0)
        out[tile_r, tile_c] = value
    return out_t, out


@pytest.mark.parametrize(
    ("kernel", "other", "load"),
    [
        (_add_transposed, (64, 512), "cute.arch.load("),
        (_scale_transposed, (64,), "(scale.iterator +"),
    ],
    ids=["packets", "scalar"],
)
def test_a_vloop_past_the_lane_budget_keeps_the_loads_before_it(
    kernel: helion.Kernel, other: tuple[int, ...], load: str
) -> None:
    """Eight row packets per column lane put the V-loop's copies past the
    unroll's budget: the V-loop stays in place, and the loads ahead of it
    (``y``'s packets, ``scale``'s scalars) still issue first, as before
    the V-loop became a level."""
    args = (
        torch.randn(512, 64).to(torch.bfloat16),
        torch.randn(*other).to(torch.bfloat16),
    )
    config = {**_ROW_PACKETS, "block_sizes": [256, 64], "cute_lane_unroll": 2}
    body = kernel_body(_render(kernel, args, config))
    assert "_lane_unroll_loads_0.append(" in body
    load_phase = body[: body.index("_lane_unroll_loads_0[")]
    assert load in load_phase
    assert "for vec_lane_0 in" not in load_phase


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transpose(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out_t = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        out_t[tile_c, tile_r] = x[tile_r, tile_c].permute(1, 0)
    return out_t


def test_a_vloop_that_writes_nothing_after_its_loads_does_not_split() -> None:
    """The V-loop only collects the packet; the store follows it, so its
    loads already issue back to back and the single-column lane loop keeps
    the knob-off code."""
    args = (torch.randn(64, 64).to(torch.bfloat16),)
    config = {**_ONE_COLUMN, "cute_lane_unroll": 2}
    assert _render(_transpose, args, config) == _render(
        _transpose, args, {**config, "cute_lane_unroll": 1}
    )


def _unrolled_vloop_nest(lanes: int, unroll: int, *, require_full: bool) -> str:
    """``_unroll_one`` on a lane loop around a tile V-loop, tagged as a
    ``VecLaneWrapper`` tags it."""
    loop = ast.parse(
        f"""
for lane_0 in range({lanes}):
    a = (x.iterator + lane_0).load()
    for vec_lane_0 in cutlass.range_constexpr(16):
        b = (y.iterator + lane_0 * 16 + vec_lane_0).load()
        (out.iterator + lane_0 * 16 + vec_lane_0).store(a + b)
"""
    ).body[0]
    assert isinstance(loop, ast.For)
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "lane_0")
    setattr(loop.body[1], HELION_VEC_LANE_OF_ATTR, "lane_0")
    names = itertools.count()
    result = _unroll_one(
        loop,
        unroll,
        lambda prefix: f"{prefix}{next(names)}",
        require_full=require_full,
    )
    assert result is not None
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


@pytest.mark.parametrize("lanes", [2, 8], ids=["within_budget", "past_budget"])
def test_a_forced_unroll_keeps_its_vloops_in_place(lanes: int) -> None:
    """A forced unroll keeps its V-loops in place (it runs without the
    knob), within the lane budget too: the load ahead of the V-loop still
    takes the loads-first form, not the in-place unroll.  The knob's unroll
    of the same nest takes the V-loop as a level."""
    code = _unrolled_vloop_nest(lanes, lanes, require_full=True)
    assert "a = _lane_unroll_loads_00[lane_0]" in code
    assert "b = (y.iterator + lane_0 * 16 + vec_lane_0).load()" in code
    assert code.count("for vec_lane_0 in cutlass.range_constexpr(16):") == 1
    code = _unrolled_vloop_nest(lanes, 2, require_full=False)
    assert "b = _lane_unroll_loads_" in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sum_of_transposed_products(x: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(x.size(1)):
            s = torch.sum(x[tile_m, tile_n] * wt[tile_n, tile_m].permute(1, 0), dim=-1)
            acc = acc + s * s
        out[tile_m] = acc
    return out


def test_the_cross_lane_fold_keeps_its_vloop_loads_in_place() -> None:
    """Without the knob, the forced unroll of a vectorized reduction's lane
    loop (its cross-lane fold) leaves the V-loop's scalar loads of ``wt``,
    read along its rows, in place; only the lane's packet of ``x`` loads
    first, as before tile V-loops became levels of the unroll."""
    args = (torch.randn(8, 8192), torch.randn(8192, 8))
    config = {
        "block_sizes": [1, 2048],
        "num_threads": [1, 128],
        "cute_vector_widths": [1, 8],
    }
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(_sum_of_transposed_products, args)
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**config))
        )
    hoisted = set(re.findall(r"_lane_unroll_loads_\d+\[([^\]]*)\]", code))
    assert hoisted == {"lane_1"}, code


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestCuteVloopLoadsFirst(TestCase):
    def test_loads_first_is_bitwise_the_rolled_order(self) -> None:
        # 157 columns padded to 160 and 72 rows: partial tiles along both.
        x = torch.randn(72, 157, device=DEVICE).to(torch.bfloat16)
        expected = torch.nn.functional.pad(x, (0, 3))
        for config in (_ROW_PACKETS, _ONE_COLUMN):
            code, (out_t, out) = code_and_output(
                _pad_and_transpose, (x, 160), **config, cute_lane_unroll=2
            )
            self.assertIn("_lane_unroll_loads_0", code)
            torch.testing.assert_close(out_t, expected.t(), rtol=0, atol=0)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
