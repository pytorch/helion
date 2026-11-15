"""CPU codegen tests: the trace-time unrolls of a lane loop are bounded.

``cute/split_lane_loads.py`` rewrites a vectorized grid lane loop, which walks
``U = block_size / (num_threads * V)`` packets per thread, into a load phase
and a compute phase that are both unrolled at trace time, so every packet load
issues before the first store.  The trace grows with the ``U * V`` elements a
thread unrolls and every packet stays live until its lane computes, so the
split takes at most ``_MAX_UNROLLED_LANES`` packets (the bound of the
``cute_lane_unroll`` unroll) and a bounded number of elements per thread
(``_MAX_SPLIT_ELEMENTS``).  A longer loop stays rolled: a 2**21 block over two
threads gave ``range_constexpr(131072)``, which never finished compiling.  So
does one whose copies would trace more than ``_MAX_UNROLLED_STATEMENTS``
statements (a long body costs far more per element than a short one), and a
lane body holding a rolled loop, which every trace-time copy (of the split or
of the ``cute_lane_unroll`` unroll) would repeat.  The forced full unroll of
a cross-lane fold's lane loop, the fallback when its loads-first form
declines, refuses a loop past the same lane and statement bounds.
"""

from __future__ import annotations

import ast
import functools
import re
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import exc
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute.split_lane_loads import _MAX_SPLIT_ELEMENTS
from helion._compiler.cute.split_lane_loads import _lane_width
from helion._compiler.cute.unroll_lane_loads import _MAX_UNROLLED_LANES
from helion._compiler.cute.unroll_lane_loads import _MAX_UNROLLED_STATEMENTS
from helion._compiler.cute.unroll_lane_loads import _trace_size
from helion._compiler.cute.unroll_lane_loads import unroll_lane_loads
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


def _scale(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.shape
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        out[tile] = (x[tile].to(torch.float32) * 2.0).to(out.dtype)
    return out


@functools.cache
def _bind(fn: object, *args: object) -> BoundKernel:
    """``fn`` bound on CPU as a CuTe kernel, each ``(shape, dtype)`` argument
    an empty tensor and any other argument passed as given."""
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    tensors = tuple(
        torch.empty(arg[0], dtype=arg[1]) if isinstance(arg, tuple) else arg
        for arg in args
    )
    with mock_cuda_unavailable():
        return cpu_bind(kernel, tensors)


def _bound() -> BoundKernel:
    return _bind(_scale, ((1 << 21,), torch.bfloat16))


@pytest.mark.parametrize(
    ("packets", "num_threads", "width", "split"),
    [
        (2, 128, 8, True),
        # 64 packets of 8 elements: at both bounds.
        (64, 128, 8, True),
        (32, 128, 16, True),
        # 128 packets of 2 elements: past the packet bound only.
        (128, 128, 2, False),
        # 64 packets of 16 elements: past the element bound only.
        (64, 128, 16, False),
        # The whole 2**21-element input in one block of two threads.
        (131072, 2, 8, False),
    ],
)
def test_lane_loop_splits_within_the_unroll_bounds(
    packets: int, num_threads: int, width: int, split: bool
) -> None:
    # The cases straddle the two bounds.
    assert split == (
        packets <= _MAX_UNROLLED_LANES and packets * width <= _MAX_SPLIT_ELEMENTS
    )
    config = helion.Config(
        block_sizes=[packets * num_threads * width],
        num_threads=[num_threads],
        cute_vector_widths=[width],
    )
    with mock_cuda_unavailable():
        code = _bound().to_code(config)
    lane_loops = re.findall(r"for lane_0 in (.+):", code)
    if split:
        assert lane_loops == [f"cutlass.range_constexpr({packets})"] * 2
        assert "_lane_loads_" in code
    else:
        assert lane_loops == [f"range({packets})"]
        assert "_lane_loads_" not in code


def _poly(x: torch.Tensor, terms: hl.constexpr) -> torch.Tensor:
    (n,) = x.shape
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        v = x[tile].to(torch.float32)
        acc = v * 0.5
        for _ in hl.static_range(terms):
            acc = acc * v + 0.25
        out[tile] = acc.to(out.dtype)
    return out


def _poly_bound(terms: int) -> BoundKernel:
    """``_poly`` bound with ``terms`` multiply-adds per element."""
    return _bind(_poly, ((1 << 21,), torch.bfloat16), terms)


@pytest.mark.parametrize(("packets", "split"), [(16, True), (64, False)])
def test_lane_loop_split_bounds_its_traced_statements(
    packets: int, split: bool
) -> None:
    # 32 multiply-adds per element trace about 300 statements per 8-element
    # lane: 16 lanes stay under ``_MAX_UNROLLED_STATEMENTS``, while 64 lanes,
    # within the packet and element bounds, trace more than twice that.
    config = helion.Config(
        block_sizes=[packets * 128 * 8],
        num_threads=[128],
        cute_vector_widths=[8],
    )
    with mock_cuda_unavailable():
        code = _poly_bound(32).to_code(config)
    lane_loops = re.findall(r"for lane_0 in (.+):", code)
    if split:
        assert lane_loops == [f"cutlass.range_constexpr({packets})"] * 2
    else:
        assert lane_loops == [f"range({packets})"]


def test_lane_unroll_bounds_its_traced_statements() -> None:
    # 80 multiply-adds per element trace about 700 statements per 8-element
    # lane: of the 16 copies asked of a 128-lane loop (too long to split),
    # which would trace a third more than ``_MAX_UNROLLED_STATEMENTS``, the
    # knob's unroll takes the 8 that trace a third less.
    config = helion.Config(
        block_sizes=[128 * 128 * 8],
        num_threads=[128],
        cute_vector_widths=[8],
        cute_lane_unroll=16,
    )
    with mock_cuda_unavailable():
        code = _poly_bound(80).to_code(config)
    assert re.findall(r"for (lane\w*) in (.+):", code) == [
        ("lane_0_blk", "range(16)"),
        ("lane_0_u", "cutlass.range_constexpr(8)"),
        ("lane_0_u", "cutlass.range_constexpr(8)"),
    ]


def _row_shift_sum(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        # Loaded before the looped sum: a per-row load both passes would
        # issue first if they copied the lane body.
        shift = b[tile_m]
        row = x[tile_m, :].to(torch.float32)
        total = torch.sum(row, dim=-1, keepdim=True)
        out[tile_m, :] = (row - shift[:, None] - total).to(out.dtype)
    return out


def _row_bound() -> BoundKernel:
    return _bind(
        _row_shift_sum, ((2048, 1024), torch.bfloat16), ((2048,), torch.float32)
    )


@pytest.mark.parametrize("lane_unroll", [1, 8])
def test_lane_loop_holding_a_rolled_loop_stays_rolled(lane_unroll: int) -> None:
    # 64 rows per thread in 8 packets of the [M] operand: within the bounds
    # of both unrolls, but every row runs the two rolled sweeps of its looped
    # reduction, which each copy would repeat (the split's 256 copies at 32
    # packets compiled for 13 s, 106 s with register-prefetched reloads).
    config = helion.Config(
        block_sizes=[256],
        num_threads=[4, 128],
        reduction_loops=[256],
        cute_vector_widths=[8, 8],
        cute_lane_unroll=lane_unroll,
    )
    with mock_cuda_unavailable():
        code = _row_bound().to_code(config)
    assert re.findall(r"for lane_0 in (.+):", code) == ["range(8)"]
    assert code.count("for roffset_1 in range(") == 2
    assert "_lane_loads_" not in code
    assert "_lane_unroll_loads_" not in code


def _row_cumsum_reverse_and_total(x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        out[tile_m, :] = hl.cumsum(row, dim=1, reverse=True)
        total[tile_m] = row.sum(-1)
    return out


@pytest.mark.parametrize("lane_unroll", [1, 8])
def test_lane_loop_holding_a_serial_scan_stays_rolled(lane_unroll: int) -> None:
    # The row total keeps the reverse scan on its serial fallback, a rolled
    # loop of 128 steps in each of a thread's 32 rows: the rows stay a rolled
    # lane loop under both passes.
    config = helion.Config(
        block_sizes=[128], num_threads=[4, 32], cute_lane_unroll=lane_unroll
    )
    bound = _bind(
        _row_cumsum_reverse_and_total,
        ((4096, 128), torch.float32),
        ((4096,), torch.float32),
    )
    with mock_cuda_unavailable():
        code = bound.to_code(config)
    assert re.findall(r"for lane_0 in (.+):", code) == ["range(32)"]
    assert code.count("for scan_i in range(") == 1
    assert "_lane_loads_" not in code
    assert "_lane_unroll_loads_" not in code


def _row_and_tile_outputs(
    x: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    out_rows = torch.empty([m], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        row = b[tile_m]
        out_rows[tile_m] = row * 2
        out[tile_m, tile_n] = x[tile_m, tile_n] + row[:, None]
    return out_rows, out


def test_lane_unroll_copies_a_rolled_lane_loop_of_another_axis() -> None:
    # 8 rows per thread around a 2-column lane loop that stays rolled (its
    # load follows the first store): unlike a reduction's sweep, a lane loop
    # counts as unrolled, so the knob still copies the rows 4 ways.
    config = helion.Config(
        block_sizes=[64, 32], num_threads=[8, 16], cute_lane_unroll=4
    )
    bound = _bind(
        _row_and_tile_outputs, ((256, 128), torch.float32), ((256,), torch.float32)
    )
    with mock_cuda_unavailable():
        code = bound.to_code(config)
    assert re.findall(r"for (lane\w*) in (.+):", code) == [
        ("lane_0_blk", "range(2)"),
        ("lane_0_u", "cutlass.range_constexpr(4)"),
        ("lane_0_u", "cutlass.range_constexpr(4)"),
        ("lane_1", "range(2)"),
    ]
    assert "_lane_unroll_loads_" in code


def _shift_tiles(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        shift = b[tile_m]
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + shift[:, None]
    return out


@pytest.mark.parametrize(("columns", "unrolled"), [(32, True), (64, False)])
def test_lane_unroll_copies_an_inner_tile_loop_of_one_trip(
    columns: int, unrolled: bool
) -> None:
    # 8 rows per thread around the inner ``hl.tile`` loop over the columns:
    # a 32-wide block covers 32 columns in one trip, which a copy traces as
    # straight-line code, so the knob copies the rows 4 ways with ``shift``
    # loaded first; over 64 columns the loop runs twice and keeps the rows
    # rolled.
    config = helion.Config(
        block_sizes=[64, 32], num_threads=[8, 32], cute_lane_unroll=4
    )
    bound = _bind(
        _shift_tiles, ((1024, columns), torch.float32), ((1024,), torch.float32)
    )
    with mock_cuda_unavailable():
        code = bound.to_code(config)
    lanes = re.findall(r"for (lane\w*) in (.+):", code)
    if unrolled:
        assert lanes == [
            ("lane_0_blk", "range(2)"),
            ("lane_0_u", "cutlass.range_constexpr(4)"),
            ("lane_0_u", "cutlass.range_constexpr(4)"),
        ]
        assert "_lane_unroll_loads_" in code
    else:
        assert lanes == [("lane_0", "range(8)")]
        assert "_lane_unroll_loads_" not in code


def test_trace_size_counts_a_rolled_loop_of_one_trip() -> None:
    """A rolled ``range`` loop whose static bounds give one trip (or none)
    traces like straight-line code; one of two trips, one whose bound is not
    known, and one over any other callee (``cutlass.range``) stay uncopyable."""
    body = "for t in {}:\n    x = y\n"
    one = "range(cutlass.Int32(0), cutlass.Int32(32), cutlass.Int32(BLOCK))"
    loop = ast.parse(body.format(one)).body
    assert _trace_size(loop, {"BLOCK": 32}) == 1 + 1
    assert _trace_size(loop, {"BLOCK": 64}) == 1 + 1
    assert _trace_size(loop, {"BLOCK": 16}) is None
    assert _trace_size(loop) is None
    assert _trace_size(ast.parse(body.format("range(0)")).body) == 1 + 1
    lane = ast.parse(body.format("cutlass.range(4)")).body
    setattr(lane[0], HELION_LANE_LOOP_VAR_ATTR, "t")
    assert _trace_size(lane) is None


def test_trace_size_counts_reversed_lane_loops() -> None:
    """A reverse scan's ``range(N - 1, -1, -1)`` lane loop counts its N trips,
    unrolled or rolled; any other rolled loop makes the body uncopyable."""
    body = "for lane_0 in {}(7, -1, -1):\n    x = y\n"
    unrolled = ast.parse(body.format("cutlass.range_constexpr")).body
    assert _trace_size(unrolled) == 1 + 8
    rolled = ast.parse(body.format("range")).body
    assert _trace_size(rolled) is None
    setattr(rolled[0], HELION_LANE_LOOP_VAR_ATTR, "lane_0")
    assert _trace_size(rolled) == 1 + 8


def test_trace_size_resolves_constexpr_bounds() -> None:
    """A ``range_constexpr`` and a lane loop over a bound the kernel's
    constexpr values give count its trips, as literal bounds do."""
    constexpr = ast.parse("for v in cutlass.range_constexpr(V):\n    x = y\n").body
    assert _trace_size(constexpr, {"V": 4}) == 1 + 4
    assert _trace_size(constexpr) == 1 + 1
    lane = ast.parse("for t in range(cutlass.Int32(BLOCK)):\n    x = y\n").body
    setattr(lane[0], HELION_LANE_LOOP_VAR_ATTR, "t")
    assert _trace_size(lane, {"BLOCK": 8}) == 1 + 8
    assert _trace_size(lane) is None


def test_trace_size_counts_literal_constexpr_ranges() -> None:
    """A ``cutlass.range_constexpr`` over literal bounds traces its body once
    per trip whatever its spelling; one whose bound is read at trace time
    (``cute.size(...)``) counts once."""
    body = "for v in cutlass.range_constexpr({}):\n    x = y\n"
    assert _trace_size(ast.parse(body.format("2, 10, 2")).body) == 1 + 4
    assert _trace_size(ast.parse(body.format("cute.size(t)")).body) == 1 + 1


@pytest.mark.parametrize(
    ("lanes", "width", "terms", "unrolled"),
    [
        # 64 lanes of a 16-element one-term fold: 1088 statements.
        (64, 16, 1, True),
        (32, 16, 8, True),
        # 64 lanes of 16 elements of eight terms: past the statement bound.
        (64, 16, 8, False),
    ],
)
def test_forced_fold_unroll_stays_within_the_unroll_bounds(
    lanes: int, width: int, terms: int, unrolled: bool
) -> None:
    # The cases straddle the statement bound, which the trace's time follows,
    # not the elements: the first fold has as many as the last.
    assert unrolled == (
        lanes <= _MAX_UNROLLED_LANES
        and lanes * (1 + width * terms) <= _MAX_UNROLLED_STATEMENTS
    )
    # A lane body without loads: the loads-first form declines, so the forced
    # unroll of the fold's lane loop falls back to unrolling it in place.
    source = "\n".join(
        [
            f"for lane_0 in range({lanes}):",
            f"    for v in cutlass.range_constexpr({width}):",
            *(f"        acc = acc + lane_0 * v * {term}" for term in range(terms)),
        ]
    )
    (loop,) = ast.parse(source).body
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "lane_0")
    if not unrolled:
        with pytest.raises(exc.BackendUnsupported, match="statements unrolled"):
            unroll_lane_loads([loop], 1, lambda name: name, forced_full={"lane_0"})
        return
    body, fired = unroll_lane_loads(
        [loop], 1, lambda name: name, forced_full={"lane_0"}
    )
    assert not fired
    assert f"for lane_0 in cutlass.range_constexpr({lanes}):" in ast.unparse(body)


@pytest.mark.parametrize("bounds", ["16", "0, 16", "15, -1, -1", "0, 32, 2", "V"])
def test_lane_width_counts_every_spelling_of_a_constexpr_range(bounds: str) -> None:
    """A lane's 16 elements, whatever the spelling of its V-loop's range: a
    start, a negative or larger step, a constexpr bound."""
    loop = ast.parse(f"for v in cutlass.range_constexpr({bounds}):\n    x = y\n").body
    assert _lane_width(loop, {"V": 16}) == 16


@pytest.mark.parametrize("statements", [400, 640])
def test_forced_loads_first_unroll_stays_within_the_statement_bound(
    statements: int,
) -> None:
    # 16 lanes, each loading one 16-wide packet ahead of the statements: the
    # loads-first form of the fold's forced unroll holds 16 packets, within
    # ``_MAX_LIVE_PACKETS``.  With 400 its copies stay within the statement
    # bound and it takes that form; with 640 they would trace past it, so it
    # declines, and the in-place fallback refuses the loop.
    packet = (
        "    packet = cute.arch.load(x.iterator + lane_0 * 16,"
        " ir.VectorType.get([16], cutlass.Float32.mlir_type))"
    )
    source = "\n".join(
        [
            "for lane_0 in range(16):",
            packet,
            *(f"    acc = acc + packet[{i % 16}] * {i}" for i in range(statements)),
        ]
    )
    (loop,) = ast.parse(source).body
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "lane_0")
    size = _trace_size(loop.body)
    assert size is not None
    if 16 * size <= _MAX_UNROLLED_STATEMENTS:
        assert statements == 400
        body, _fired = unroll_lane_loads(
            [loop], 1, lambda name: name, forced_full={"lane_0"}
        )
        assert "_lane_unroll_loads_0" in ast.unparse(ast.Module(body, []))
    else:
        assert statements == 640
        with pytest.raises(exc.BackendUnsupported, match="statements unrolled"):
            unroll_lane_loads([loop], 1, lambda name: name, forced_full={"lane_0"})
