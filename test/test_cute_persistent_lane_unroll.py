"""``cute_lane_unroll`` on the synthetic lanes of a persistent reduction row.

A persistent row whose length is not a power of two is masked, so one warp
walks it in rolled scalar lanes, each lane loading right before its
arithmetic.  The knob unrolls those lanes in blocks whose loads issue first,
as it does for the lane loops of tile axes; config normalization used to
reset it to 1 for kernels without such a tile axis.  A seed heuristic offers
it for a masked persistent row no ``reduction_loops`` slot can roll.  The
rows render on CPU; the ``cuda`` test runs the knob against the knob-off
code: masks bit for bit, and floats within the rounding of the multiply-adds
LLVM contracts once the loads leave the arithmetic.
"""

from __future__ import annotations

import functools
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import unroll_lane_loads
from helion._compiler.cute.autotune.heuristics import CutePersistentLaneUnrollHeuristic
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_dropout_rows import dropout_residual_norm
from helion._testing.cute_dropout_rows import dropout_residual_norm_args

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


_dropout_residual_norm = helion.kernel(
    dropout_residual_norm,
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    cute_rng_stream="word0",
)


def _config(unroll: int | None) -> helion.Config:
    config = helion.Config(
        block_sizes=[8], num_threads=[8, 32], cute_reduction_reloads=["register"]
    )
    if unroll is not None:
        config.config["cute_lane_unroll"] = unroll
    return config


@functools.cache
def _bind(width: int) -> object:
    with mock_cuda_unavailable(), forbid_native_compile():
        return cpu_bind(_dropout_residual_norm, dropout_residual_norm_args(width))


def _render_once(width: int, unroll: int | None) -> str:
    bound = _bind(width)
    with mock_cuda_unavailable(), forbid_native_compile():
        return bound.to_code(  # pyrefly: ignore[missing-attribute]
            bound.config_spec.normalized_config(  # pyrefly: ignore[missing-attribute]
                _config(unroll)
            )
        )


# Each configuration renders once per module.
_render = functools.cache(_render_once)


@pytest.mark.parametrize("unroll", [4, 8])
def test_the_knob_unrolls_a_masked_row_loads_first(unroll: int) -> None:
    code = _render(768, unroll)
    # 768 columns over 32 threads walk 24 lanes: three or six blocks whose
    # loads issue before their arithmetic.
    assert f"range({24 // unroll}):" in code, code
    assert f"cutlass.range_constexpr({unroll})" in code, code
    assert "_lane_unroll_loads_" in code, code
    # The lanes keep their masks.
    assert "< 768" in code, code


def test_the_knob_off_code_is_the_code_without_the_knob() -> None:
    code = _render(768, 1)
    assert code == _render(768, None)
    assert "_lane_unroll_loads_" not in code
    assert "range(24):" in code, code


@pytest.mark.parametrize(("unroll", "lanes"), [(2, 20), (8, 24)])
def test_the_knob_unrolls_an_odd_row_over_its_live_lanes_rounded_up(
    unroll: int, lanes: int
) -> None:
    # 600 columns over 32 threads hold elements in 19 lanes, which no unroll
    # factor divides: the knob walks them rounded up to its multiple, masked
    # as the trailing lanes are.
    code = _render(600, unroll)
    assert f"range({lanes // unroll}):" in code, code
    assert f"cutlass.range_constexpr({unroll})" in code, code
    assert "< 600" in code, code


def test_the_knob_off_code_of_an_odd_row_walks_its_live_lanes() -> None:
    assert "range(19):" in _render(600, 1)


def test_a_knob_no_lane_loop_takes_keeps_the_live_lanes() -> None:
    # The knob-off code is generated again where the unroll takes no loop
    # (``LaneUnrollNotApplied``), and it walks the 19 live lanes.
    with patch.object(unroll_lane_loads, "_unroll_factor", return_value=0):
        assert _render_once(600, 8) == _render(600, 1)


@pytest.mark.parametrize(("width", "kept"), [(768, 8), (32, 1), (600, 8)])
def test_normalization_keeps_the_knob_where_a_lane_loop_takes_it(
    width: int, kept: int
) -> None:
    # A row of one warp's width walks no synthetic lanes: the knob resets.  A
    # 600-wide row's 19 live lanes take no unroll factor, so under the knob
    # it walks 24, which take one.
    bound = _bind(width)
    normalized = bound.config_spec.normalized_config(  # pyrefly: ignore
        _config(8)
    )
    assert normalized.config["cute_lane_unroll"] == kept


@pytest.mark.parametrize(
    ("num_threads", "kept"),
    [
        # A thread for every element of the padded row, by default or asked
        # for: no synthetic lanes.
        ([1, 0], 1),
        ([0, 0], 1),
        ([1, 1024], 1),
        # Fewer: the masked row walks its 24 lanes on one warp.
        ([1, 64], 8),
    ],
)
def test_normalization_keeps_the_knob_only_for_synthetic_lanes(
    num_threads: list[int], kept: int
) -> None:
    bound = _bind(768)
    spec = bound.config_spec  # pyrefly: ignore[missing-attribute]
    normalized = spec.normalized_config(
        helion.Config(block_sizes=[1], num_threads=num_threads, cute_lane_unroll=8)
    )
    assert normalized.config["cute_lane_unroll"] == kept
    # A normalized config normalizes to itself.
    assert spec.normalized_config(normalized) == normalized
    with mock_cuda_unavailable(), forbid_native_compile():
        code = bound.to_code(normalized)  # pyrefly: ignore[missing-attribute]
    assert ("synthetic_lane_1" in code) is (kept > 1), code


@pytest.mark.parametrize(("width", "other"), [(24, 48), (512, 2000), (600, 768)])
def test_the_search_space_is_the_same_at_every_width(width: int, other: int) -> None:
    # ``autotune_multi`` needs one search space for the kernel's shapes: the
    # knob opens on its structure, while the seed plants only for a masked
    # row whose warp walks an even number of live lanes (48 and 768, but not
    # 24, 512, 600 or 2000).
    spec = _bind(width).config_spec  # pyrefly: ignore[missing-attribute]
    other_spec = _bind(other).config_spec  # pyrefly: ignore[missing-attribute]
    assert spec.structural_fingerprint() == other_spec.structural_fingerprint()


# 600 columns leave a warp 19 live lanes, which no unroll factor divides.
@pytest.mark.parametrize(("width", "seeded"), [(768, True), (512, False), (600, False)])
def test_the_seed_and_the_search_choices(width: int, seeded: bool) -> None:
    spec = _bind(width).config_spec  # pyrefly: ignore[missing-attribute]
    seeds = [dict(seed) for seed in spec.compiler_seed_configs]
    seed = {
        "block_sizes": [8],
        "num_threads": [8, 32],
        "cute_reduction_reloads": ["register"],
        "cute_lane_unroll": CutePersistentLaneUnrollHeuristic.LANE_UNROLL,
    }
    assert (seed in seeds) is seeded, seeds
    # The knob is open to the search at every width.
    field = spec._flat_fields()["cute_lane_unroll"]
    assert set(field.search_choices) == {1, 2, 4, 8, 16}


@functools.cache
def _outputs(width: int, unroll: int) -> tuple[torch.Tensor, ...]:
    args = dropout_residual_norm_args(width, DEVICE)
    bound = _dropout_residual_norm._bind_isolated(args)
    config = bound.config_spec.normalized_config(_config(unroll))
    assert ("_lane_unroll_loads_" in bound.to_code(config)) is (unroll > 1)
    return bound.compile_config(config)(*args)


@pytest.mark.cuda
@skipIfNotCUDA()
# 600 columns: the knob unrolls the 19 live lanes rounded up to 24, the
# knob-off code walks the 19.
@pytest.mark.parametrize("width", [768, 600])
def test_cuda_loads_first_lanes_match_the_rolled_lanes(width: int) -> None:
    keep, centered, norm = _outputs(width, 8)
    rolled_keep, rolled_centered, rolled_norm = _outputs(width, 1)
    assert torch.equal(keep.view(torch.uint8), rolled_keep.view(torch.uint8))
    # The contraction of ``residual + keep * x * c`` may change a rounding.
    torch.testing.assert_close(centered, rolled_centered, rtol=0, atol=1e-5)
    torch.testing.assert_close(
        norm.float(), rolled_norm.float(), rtol=0.008, atol=0.008
    )
