"""Config normalization predicts a persistent row's lanes as the code does.

``cute_lane_unroll`` stays on only where some lane loop takes it
(``_cute_config_forms_lane_loop``).  For a persistent reduction row the
prediction follows ``CutePersistentReductionEmitter.plan_synthetic_lanes``
with the rules both apply: a thread count whose lanes do not tile the padded
row drops to a power of two (``tiled_row_threads``), and the row then runs on
one warp unless one vector per thread covers it.  A masked row walks only its
live lanes (``masked_row_lanes``), which the knob rounds up to its multiple;
the model counts its padded lanes, which form a lane loop where those do.
Each case renders on CPU with the knob forced through normalization, and the
knob must be kept exactly where that render unrolls a lane loop.
"""

from __future__ import annotations

import functools
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.autotune import search_surface
from helion._compiler.cute.reduction_lanes import masked_row_lanes
from helion._compiler.cute.reduction_lanes import requested_lane_unroll
from helion._compiler.cute.reduction_lanes import walks_live_lanes
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_center(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        values = x[row, :].to(torch.float32)
        out[row, :] = (values - torch.sum(values, dim=-1, keepdim=True)).to(out.dtype)
    return out


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return left + right


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _padded_count(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for row in hl.tile(x.size(0)):
        out[row] = hl.reduce(_add, x[row, :].float(), dim=1, other=1.0)
    return out


@functools.cache
def _bind(kernel: helion.Kernel, width: int) -> tuple[Any, torch.Tensor]:
    # The argument stays alive with the bound kernel: the vector alignment
    # proof reads its metadata.
    x = torch.empty((8, width), dtype=torch.bfloat16)
    with mock_cuda_unavailable(), forbid_native_compile():
        return cpu_bind(kernel, (x,)), x


def _config(threads: int, vec: int) -> helion.Config:
    # ``cute_vector_widths`` lists the reduction block first.
    return helion.Config(
        block_sizes=[1],
        num_threads=[1, threads],
        cute_vector_widths=[vec, 1],
        cute_lane_unroll=8,
    )


def _kept_and_unrolled(kernel: helion.Kernel, width: int, config: helion.Config):
    bound, _ = _bind(kernel, width)
    spec = bound.config_spec
    normalized = spec.normalized_config(config)
    kept = normalized.config["cute_lane_unroll"] != 1
    # ``to_code`` normalizes the config again, so the knob stays forced
    # through the render.
    with (
        patch.object(search_surface, "_cute_config_forms_lane_loop", return_value=True),
        mock_cuda_unavailable(),
        forbid_native_compile(),
    ):
        forced = spec.normalized_config(config)
        assert forced.config["cute_lane_unroll"] == 8
        code = bound.to_code(forced)
    return kept, "_lane_unroll_loads_" in code


@pytest.mark.parametrize(
    ("width", "threads", "vec", "kept"),
    [
        # One 4-wide vector per thread on eight warps: its lanes are the
        # vector's elements, and the row keeps its threads.
        pytest.param(1024, 256, 4, False, id="vector-row"),
        # 16 lanes on 64 threads, not one vector each: one warp walks 32.
        pytest.param(1024, 64, 1, True, id="one-warp"),
        # 96 threads leave 11 lanes, which do not tile the padded row: 64
        # threads take it, one 16-wide vector each.
        pytest.param(1024, 96, 16, False, id="rounded-to-a-vector-row"),
        pytest.param(1024, 96, 1, True, id="rounded-then-one-warp"),
        # 600 elements on one warp hold 19 live lanes, which no factor
        # divides: the knob walks them rounded up to 24.
        pytest.param(600, 32, 1, True, id="odd-live-lanes"),
        pytest.param(768, 32, 1, True, id="even-live-lanes"),
    ],
)
def test_the_knob_stays_where_a_row_lane_loop_unrolls(
    width: int, threads: int, vec: int, kept: bool
) -> None:
    assert _kept_and_unrolled(_row_center, width, _config(threads, vec)) == (
        kept,
        kept,
    )


def test_a_row_of_several_vectors_per_thread_takes_no_unroll() -> None:
    # 64 threads give a 1024-wide row 16 lanes, not one 8-wide vector each:
    # the row runs on one warp, whose 32 lanes are four vectors per thread,
    # walked by one constexpr loop.
    assert _kept_and_unrolled(_row_center, 1024, _config(64, 8)) == (False, False)


def test_an_hl_reduce_row_keeps_its_padded_lanes() -> None:
    # ``hl.reduce`` folds its ``other`` into the padded slots, so the code
    # keeps all 32 lanes of a 600-wide row on one warp, and the knob unrolls
    # them.  The model counts a masked row's padded lanes, which form a lane
    # loop as the lanes the code walks under the knob do.
    assert _kept_and_unrolled(_padded_count, 600, _config(32, 1)) == (True, True)
    bound, _ = _bind(_padded_count, 600)
    config = helion.Config(**dict(_config(32, 1).config, cute_lane_unroll=1))
    with mock_cuda_unavailable(), forbid_native_compile():
        code = bound.to_code(config)
    assert "for synthetic_lane_1 in range(32):" in code, code


@pytest.mark.parametrize(
    ("width", "proven_bounds", "unroll", "live"),
    [
        pytest.param(768, False, 1, True, id="768"),
        # Under cute_proven_bounds live lanes covering the row exactly (768 =
        # 24 x 32) keep the padded ones; 600 elements leave 19, trimmed.
        pytest.param(768, True, 1, False, id="768-proven-bounds"),
        pytest.param(600, True, 1, True, id="600-proven-bounds"),
        # Under the knob 24 live lanes take a factor; 19 take none and are
        # rounded up (``masked_row_lanes``).
        pytest.param(768, False, 8, True, id="768-unrolled"),
        pytest.param(600, False, 8, False, id="600-unrolled"),
        pytest.param(600, True, 8, False, id="600-proven-bounds-unrolled"),
    ],
)
def test_the_rows_that_walk_only_their_live_lanes(
    width: int, proven_bounds: bool, unroll: int, live: bool
) -> None:
    # The seed heuristic asks this rule; the code generator asks
    # ``masked_row_lanes``.  A CPU bind refuses the proven-bounds rendering,
    # so this asks the rule alone.
    assert (
        walks_live_lanes(width, 32, proven_bounds=proven_bounds, unroll=unroll) is live
    )


@pytest.mark.parametrize(
    ("width", "proven_bounds", "unroll", "lanes"),
    [
        # 19 live lanes of 32: walked as they are, or rounded up to the knob.
        (600, False, 1, 19),
        (600, False, 2, 20),
        (600, False, 8, 24),
        (600, True, 8, 24),
        # 24 live lanes take a factor; under proven bounds a row the live
        # lanes cover exactly keeps its padded count.
        (768, False, 8, 24),
        (768, True, 1, 32),
    ],
)
def test_the_lanes_a_masked_row_walks(
    width: int, proven_bounds: bool, unroll: int, lanes: int
) -> None:
    assert (
        masked_row_lanes(width, 32, 32, proven_bounds=proven_bounds, unroll=unroll)
        == lanes
    )


@pytest.mark.parametrize(
    ("config", "unroll"),
    [
        ({"cute_lane_unroll": 8}, 8),
        ({}, 1),
        # Under sinking an odd row keeps its live lanes.
        ({"cute_lane_unroll": 8, "cute_vloop_sink": True}, 1),
    ],
)
def test_the_unroll_the_lane_loops_take(config: dict[str, object], unroll: int) -> None:
    assert requested_lane_unroll(config) == unroll
