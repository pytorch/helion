"""Config normalization predicts a persistent row's lanes as the code does.

``cute_lane_unroll`` stays on only where some lane loop takes it
(``_cute_config_forms_lane_loop``).  For a persistent reduction row the
prediction follows ``CutePersistentReductionEmitter.plan_synthetic_lanes``
with the rules both apply: a thread count whose lanes do not tile the padded
row drops to a power of two (``tiled_row_threads``), the row then runs on one
warp unless one vector per thread covers it, and a masked row walks only its
live lanes (``trims_padded_lanes``).  Each case renders on CPU with the knob forced
through normalization, and the knob must be kept exactly where that render
unrolls a lane loop.
"""

from __future__ import annotations

import functools
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.autotune import search_surface
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
    with patch.object(
        search_surface, "_cute_config_forms_lane_loop", return_value=True
    ):
        forced = spec.normalized_config(config)
    assert forced.config["cute_lane_unroll"] == 8
    with mock_cuda_unavailable(), forbid_native_compile():
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
        # 600 elements on one warp walk 19 live lanes, which no factor
        # divides.
        pytest.param(600, 32, 1, False, id="odd-live-lanes"),
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


def test_an_hl_reduce_row_keeps_its_padded_lanes() -> None:
    # ``hl.reduce`` folds its ``other`` into the padded slots, so the code
    # keeps all 32 lanes of a 600-wide row on one warp, where the model,
    # which cannot see the ``hl.reduce``, counts the 19 live ones and resets
    # the knob.  The render takes no unroll of that loop either, so the
    # reset loses nothing.
    assert _kept_and_unrolled(_padded_count, 600, _config(32, 1)) == (False, False)
    bound, _ = _bind(_padded_count, 600)
    with mock_cuda_unavailable(), forbid_native_compile():
        code = bound.to_code(bound.config_spec.normalized_config(_config(32, 1)))
    assert "for synthetic_lane_1 in range(32):" in code, code


@pytest.mark.parametrize(("width", "kept"), [(768, True), (600, False)])
def test_proven_bounds_trim_the_lanes_the_code_trims(width: int, kept: bool) -> None:
    # Under cute_proven_bounds a masked row keeps its 32 padded lanes only
    # when its live lanes cover it exactly (768 = 24 x 32); a 600-wide row
    # walks its 19 live lanes, as without the flag (``trims_padded_lanes``,
    # which the code generator asks too).  A CPU bind refuses the flag's
    # rendering, so this asks the model alone.
    bound, _ = _bind(_row_center, width)
    config = dict(_config(32, 1).config, cute_proven_bounds=True)
    assert (
        search_surface._cute_config_forms_lane_loop(bound.config_spec, config) is kept
    )
