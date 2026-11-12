"""A masked persistent row walks only the lanes that hold its elements.

A persistent row whose length is not a power of two pads to the next one
(768 to 1024) and walks the padding in scalar lanes of ``threads``
consecutive elements; from lane ``ceil(n / threads)`` on, every thread is
masked, so those lanes only fold the identities torch's reductions
substitute and store nothing.  ``_live_lane_extent`` drops them, except in
a row an ``hl.reduce`` reduces, whose fill is the user's ``other``.  The
rows render on CPU; the ``cuda`` tests run them against the padded lane
count, bit for bit, on rows of signed zeros, infinities and NaNs.
"""

from __future__ import annotations

import ast
import contextlib
import functools
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import reduction_emitter
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


def _dropout_row_stats_fn(
    x: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, n = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    peak = torch.empty([rows], dtype=torch.float32, device=x.device)
    where = torch.empty([rows], dtype=torch.int64, device=x.device)
    centered = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    for row in hl.tile(rows):
        # The draw keeps the row persistent; dropping a negative, infinite or
        # NaN element yields -0.0 or NaN.
        keep = hl.rand([row, n], seed=seed) > 0.25
        data = x[row, :].to(torch.float32) * keep.to(torch.float32)
        row_sum = torch.sum(data, dim=-1)
        total[row] = row_sum
        peak[row] = torch.amax(data, dim=-1)
        where[row] = torch.argmax(data, dim=-1)
        centered[row, :] = data - row_sum[:, None]
    return total, peak, where, centered


_row_stats = helion.kernel(
    _dropout_row_stats_fn,
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    cute_rng_stream="word0",
)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scaled_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        out[row, :] = x[row, :] * 2.0
    return out


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return left + right


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _padded_count(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for row in hl.tile(x.size(0)):
        out[row] = hl.reduce(_add, x[row, :].float(), dim=1, other=1.0)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sum_then_padded_count(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    count = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for row in hl.tile(x.size(0)):
        total[row] = torch.sum(x[row, :].float(), dim=-1)
    for row in hl.tile(y.size(0)):
        count[row] = hl.reduce(_add, y[row, :].float(), dim=1, other=1.0)
    return total, count


# 32 threads for each loop's reduction (blocks 1 and 3, in the spec's order
# 0, 2, 1, 3).
_TWO_LOOP_CONFIG = helion.Config(block_sizes=[1, 1], num_threads=[0, 0, 32, 32])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _suffix_sums(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        out[row, :] = hl.associative_scan(_add, x[row, :], dim=1, reverse=True)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        out[tile_m, tile_n] = torch.matmul(x[tile_m, :], y[:, tile_n])
    return out


# The 1500-long contraction is longer than the CTA: one thread per output
# folds it serially (``emit_cute_synthetic_lane_fold_mm``).
_MATMUL_CONFIG = helion.Config(block_sizes=[16, 16])


def _config() -> helion.Config:
    return helion.Config(block_sizes=[4], num_threads=[4, 32])


def _lane_loops(code: str) -> list[str]:
    return [
        ast.unparse(node.iter)
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and ast.unparse(node.target).startswith("synthetic_lane_")
    ]


def _render(width: int) -> str:
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_row_stats, (torch.empty((6, width)), 123))
        return bound.to_code(_config())


@pytest.mark.parametrize(
    ("width", "lanes"),
    [
        pytest.param(768, 24, id="768-pads-to-1024"),
        pytest.param(600, 19, id="partial-last-lane"),
        pytest.param(96, 3, id="96-pads-to-128"),
        pytest.param(1024, 32, id="power-of-two"),
    ],
)
def test_masked_rows_walk_only_their_live_lanes(width: int, lanes: int) -> None:
    code = _render(width)
    loops = _lane_loops(code)
    assert loops, code
    assert set(loops) == {f"range({lanes})"}, code
    # The mask stays: the lanes keep their padded form, bit for bit.
    assert (f"< {width}" in code) is (width != 1024), code


@pytest.mark.parametrize(
    ("width", "proven_bounds", "lanes"),
    [
        pytest.param(768, False, 24, id="768"),
        pytest.param(768, True, 32, id="768-proven-bounds"),
        pytest.param(600, True, 19, id="600-proven-bounds"),
    ],
)
def test_proven_bounds_keep_the_padded_lanes_of_an_exact_count(
    width: int, proven_bounds: bool, lanes: int
) -> None:
    # A full-slice axis walks the same masked lanes.  Under the proven-bounds
    # cleanup, lanes covering the row exactly (768 = 24 x 32) would let it
    # fold the mask and drop the masked loads' branches.  NVVM contracts a
    # multiply and an add only within one basic block, so products those
    # branches kept apart from their adds could fuse, and cute_proven_bounds
    # would change the results.  A partial last lane (600) keeps the mask
    # either way.
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_scaled_copy, (torch.empty((6, width)),))
        code = bound.to_code(
            helion.Config(
                block_sizes=[1], num_threads=[0, 32], cute_proven_bounds=proven_bounds
            )
        )
    assert set(_lane_loops(code)) == {f"range({lanes})"}, code
    assert f"< {width}" in code, code


def test_user_reduce_keeps_its_padded_lanes() -> None:
    # ``hl.reduce`` folds its ``other`` into every masked slot: 1.0 under a
    # sum counts the 256 padded slots of a 768-wide row, so a zero row sums
    # to 256 at 32 lanes and would sum to 0 at 24.
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_padded_count, (torch.empty((6, 768)),))
        code = bound.to_code(helion.Config(block_sizes=[1], num_threads=[0, 32]))
    assert set(_lane_loops(code)) == {"range(32)"}, code
    assert "cutlass.Float32(1.0)" in code, code


def test_a_user_reduce_keeps_only_the_padded_lanes_it_reduces() -> None:
    # The hl.reduce reduces the 96-wide rows: they keep their 4 lanes, one
    # past the 3 live ones folding other=1.0, while the 768-wide sum of the
    # other loop walks its 24 live lanes.
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(
            _sum_then_padded_count, (torch.empty((6, 768)), torch.empty((6, 96)))
        )
        code = bound.to_code(_TWO_LOOP_CONFIG)
    assert sorted(_lane_loops(code)) == ["range(24)", "range(4)"], code


def test_reverse_scan_starts_at_the_last_live_lane() -> None:
    # A suffix scan walks the lanes from the last: at width 600 it starts at
    # lane 18, whose threads past the row are masked.  The scan's carry only
    # takes a valid element, so the masked slots never enter a sum.
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_suffix_sums, (torch.empty((6, 600)),))
        code = bound.to_code(helion.Config(block_sizes=[1], num_threads=[0, 32]))
    assert set(_lane_loops(code)) == {"range(18, -1, -1)"}, code
    assert "< 600" in code, code


def test_a_folded_matmul_contraction_walks_its_live_lanes() -> None:
    # An fp32 matmul's contraction on one thread walks a lane per element:
    # 1500, not the padded 2048, still masked.
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_matmul, (torch.empty((64, 1500)), torch.empty((1500, 64))))
        code = bound.to_code(_MATMUL_CONFIG)
    assert set(_lane_loops(code)) == {"range(1500)"}, code
    assert "< 1500" in code, code


def _row_sums_fn(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for row in hl.tile(x.size(0)):
        out[row] = torch.sum(x[row, :].float(), dim=-1)
    return out


@pytest.mark.parametrize(
    ("static_shapes", "lanes"),
    [pytest.param(True, 24, id="static"), pytest.param(False, 32, id="symbolic")],
)
def test_a_symbolic_row_length_keeps_its_padded_lanes(
    static_shapes: bool, lanes: int
) -> None:
    # A symbolic length is only a hint: a kernel compiled for its bucket must
    # still cover a longer row, so the lane count stays the padded one.
    kernel = helion.kernel(
        _row_sums_fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(kernel, (torch.empty((6, 768)),))
        code = bound.to_code(helion.Config(block_sizes=[1], num_threads=[0, 32]))
    assert set(_lane_loops(code)) == {f"range({lanes})"}, code


def _special_rows(width: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((8, width), generator=generator)
    x[1] = -0.0
    x[2] = float("-inf")
    x[3, 5] = float("nan")
    x[4, width - 1] = float("inf")
    x[5] = 0.0
    x[5, width // 2] = -0.0
    return x.to(DEVICE)


def _lane_loops_and_results(
    bound: helion.BoundKernel,
    config: helion.Config,
    args: tuple[object, ...],
    live: bool,
) -> tuple[list[str], tuple[torch.Tensor, ...]]:
    """``bound``'s lane loops and results at ``config``: with its live lanes,
    or with ``_live_lane_extent`` the identity (whatever its keywords), all
    of the padded ones."""
    with contextlib.ExitStack() as stack:
        if not live:
            stack.enter_context(
                patch.object(
                    reduction_emitter,
                    "_live_lane_extent",
                    lambda lane_extent, numel, threads, **_: lane_extent,
                )
            )
        loops = _lane_loops(bound.to_code(config))
        return loops, bound.compile_config(config)(*args)


@functools.cache
def _run(width: int, live: bool) -> tuple[torch.Tensor, ...]:
    x = _special_rows(width)
    loops, results = _lane_loops_and_results(
        _row_stats._bind_isolated((x, 123)), _config(), (x, 123), live
    )
    assert set(loops) == {f"range({-(-width // 32) if live else 32})"}, loops
    return results


def _two_loop_run(live: bool) -> tuple[torch.Tensor, ...]:
    x, y = _special_rows(768), _special_rows(96)
    loops, results = _lane_loops_and_results(
        _sum_then_padded_count._bind_isolated((x, y)), _TWO_LOOP_CONFIG, (x, y), live
    )
    assert sorted(loops) == ["range(24)" if live else "range(32)", "range(4)"], loops
    return results


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_a_row_beside_a_user_reduce_matches_its_padded_lanes() -> None:
    for live, padded in zip(_two_loop_run(True), _two_loop_run(False), strict=True):
        assert torch.equal(live.view(torch.int32), padded.view(torch.int32))


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_reverse_scan_starts_at_the_last_live_lane() -> None:
    # At width 600 the suffix scan starts at lane 18, partly masked, whose
    # carry and warp-scan flags must take only its live elements.  Integer
    # values keep every partial sum exact in fp32, so the scan's order does
    # not matter.
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(-8, 8, (6, 600), generator=generator).to(torch.float32)
    x = x.to(DEVICE)
    bound = _suffix_sums._bind_isolated((x,))
    config = helion.Config(block_sizes=[1], num_threads=[0, 32])
    assert set(_lane_loops(bound.to_code(config))) == {"range(18, -1, -1)"}
    torch.testing.assert_close(
        bound.compile_config(config)(x), x.flip(1).cumsum(1).flip(1), rtol=0, atol=0
    )


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("width", [768, 600])
def test_cuda_live_lanes_match_the_padded_lanes(width: int) -> None:
    for live, padded in zip(_run(width, True), _run(width, False), strict=True):
        assert live.dtype == padded.dtype
        bits = torch.int64 if live.dtype == torch.int64 else torch.int32
        assert torch.equal(live.view(bits), padded.view(bits))


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_a_folded_matmul_contraction_matches_its_padded_lanes() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((64, 1500), generator=generator).to(DEVICE)
    y = torch.randn((1500, 64), generator=generator).to(DEVICE)
    outputs = []
    for live, lanes in ((True, 1500), (False, 2048)):
        loops, out = _lane_loops_and_results(
            _matmul._bind_isolated((x, y)), _MATMUL_CONFIG, (x, y), live
        )
        assert set(loops) == {f"range({lanes})"}, loops
        outputs.append(out)
    assert torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32))
    torch.testing.assert_close(outputs[0], x @ y, rtol=1e-4, atol=1e-3)
