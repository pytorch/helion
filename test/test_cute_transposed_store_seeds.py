"""The seeds of a transposed tile store
(``helion/_compiler/cute/autotune/heuristics/cute_transposed_store_seeds.py``,
``helion/_compiler/cute/transposed_stores.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

_NAME = "cute_transposed_store_seeds"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pad_and_transpose(
    x: torch.Tensor, padded: hl.constexpr, transpose_first: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` zero-padded to ``padded`` columns, stored transposed and as is:
    the shape of better-benchmark's ``pad_vocab_and_transpose``, which
    ``sum_6119d597d188`` stores transposed first and ``sum_81b4fd73f8d1``
    padded first (each store has the other's tile contiguous)."""
    rows, cols = x.shape
    out_t = torch.empty([padded, rows], dtype=x.dtype, device=x.device)
    out = torch.empty([rows, padded], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, padded]):
        in_range = tile_c.index < cols
        value = hl.load(x, [tile_r, tile_c], extra_mask=in_range[None, :])
        value = torch.where(in_range[None, :], value, 0.0)
        if transpose_first:
            out_t[tile_c, tile_r] = value.permute(1, 0)
        out[tile_r, tile_c] = value
        if not transpose_first:
            out_t[tile_c, tile_r] = value.permute(1, 0)
    return out_t, out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scaled_transpose(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        out[tile_c, tile_r] = (x[tile_r, tile_c] * 2).T
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transposed_input_copy(xt: torch.Tensor) -> torch.Tensor:
    """``_scaled_transpose`` with the load transposed against a plain store."""
    cols, rows = xt.shape
    out = torch.empty([rows, cols], dtype=xt.dtype, device=xt.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        out[tile_r, tile_c] = (xt[tile_c, tile_r] * 2).T
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _wide_column_transpose(x: torch.Tensor) -> torch.Tensor:
    """``_scaled_transpose`` with column tiles of 2048 or more."""
    rows, cols = x.shape
    out = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    block_c = hl.register_block_size(2048, 4096)
    for tile_r, tile_c in hl.tile([rows, cols], block_size=[None, block_c]):
        out[tile_c, tile_r] = (x[tile_r, tile_c] * 2).T
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_transpose(x: torch.Tensor) -> torch.Tensor:
    """``_scaled_transpose`` with the column tiles in a loop of their own."""
    rows, cols = x.shape
    out = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        for tile_c in hl.tile(cols):
            out[tile_c, tile_r] = (x[tile_r, tile_c] * 2).T
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transpose_beside_a_row_loop(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_scaled_transpose`` and a loop over ``y`` of a tile of its own."""
    rows, cols = x.shape
    out = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    out_y = torch.empty_like(y)
    for tile_r, tile_c in hl.tile([rows, cols]):
        out[tile_c, tile_r] = (x[tile_r, tile_c] * 2).T
    for tile_n in hl.tile(y.size(0)):
        out_y[tile_n] = y[tile_n] + 1
    return out, out_y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile(x.shape):
        out[tile_r, tile_c] = x[tile_r, tile_c] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scale_rows(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile(x.shape):
        out[tile_r, tile_c] = x[tile_r, tile_c] * scale[tile_r][:, None]
    return out


def _seeds(bound: BoundKernel, packet: int) -> list[dict[str, object]]:
    """The seeds with 64 threads along the columns, two along the rows and
    ``packet``-row vectors."""
    return [
        dict(seed.config)
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("num_threads") == [2, 64]
        and seed.config.get("cute_vector_widths") == [packet, 1]
    ]


@pytest.mark.parametrize("transpose_first", [True, False])
def test_bf16_seeds_give_the_threads_packets_of_rows(transpose_first: bool) -> None:
    """With either store first.  Stored first, the padded copy is transposed
    against ``out_t`` as well; the seeds follow ``out_t``, the store the load
    disagrees with, and do not run the threads along the rows, where every
    warp load would read rows apart."""
    args = (torch.empty(4096, 8189, dtype=torch.bfloat16), 8192, transpose_first)
    with cpu_render_target():
        bound = cpu_bind(_pad_and_transpose, args)
        assert _NAME in bound.config_spec.autotuner_heuristics
        seeds = _seeds(bound, 16)
        # 32-byte packets of the rows, two threads per column, 64 threads
        # along the columns; four and two column lanes per thread.
        assert [seed["block_sizes"] for seed in seeds] == [[32, 256], [64, 128]]
        assert [seed["num_threads"] for seed in seeds] == [[2, 64], [2, 64]]
        assert [seed["cute_vector_widths"] for seed in seeds] == [[16, 1], [16, 1]]
        assert [seed["cute_lane_unroll"] for seed in seeds] == [4, 2]
        # Thread axis 0 along the columns: their tile leads the loop order,
        # and both tiles' lanes are strided.
        assert [seed["loop_orders"] for seed in seeds] == [[[1, 0]], [[1, 0]]]
        assert [seed["cute_lane_layouts"] for seed in seeds] == [["strided"] * 2] * 2
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
    # Every lane's loads issue before the stores.
    assert "_lane_unroll_loads_0" in code


def test_a_transposed_load_runs_the_threads_along_the_store_rows() -> None:
    """The store's other dim is its rows here: 64 threads along them, which
    lead the loop order, and the packets along its columns."""
    with cpu_render_target():
        bound = cpu_bind(
            _transposed_input_copy, (torch.empty(4096, 2048, dtype=torch.bfloat16),)
        )
        seeds = [
            dict(seed.config)
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("num_threads") == [64, 2]
        ]
    assert [seed["block_sizes"] for seed in seeds] == [[256, 32], [128, 64]]
    assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 16], [1, 16]]
    assert [seed["loop_orders"] for seed in seeds] == [[[0, 1]], [[0, 1]]]
    assert [seed["cute_lane_layouts"] for seed in seeds] == [["strided"] * 2] * 2


def test_fp32_seeds_hold_eight_rows_per_packet() -> None:
    with cpu_render_target():
        bound = cpu_bind(_scaled_transpose, (torch.randn(2048, 4096),))
        seeds = _seeds(bound, 8)
    assert [seed["block_sizes"] for seed in seeds] == [[16, 256], [32, 128]]
    assert [seed["cute_vector_widths"] for seed in seeds] == [[8, 1], [8, 1]]


def test_one_byte_seeds_take_the_widest_vector_the_spec_admits() -> None:
    # 32 one-byte rows would fill the 32-byte packet, but no vector width
    # reaches 32 elements: the seeds take 16, which the autotuner's transfer
    # keeps (a width of 32 would be an invalid config, dropped there).
    x = torch.randint(-8, 8, (2048, 4096), dtype=torch.int8)
    with cpu_render_target():
        bound = cpu_bind(_scaled_transpose, (x,))
        seeds = _seeds(bound, 16)
        generation = bound.config_spec.create_config_generation()
        transferred = [
            dict(config.config) for _, config in generation.seed_flat_config_pairs()
        ]
    assert [seed["block_sizes"] for seed in seeds] == [[64, 256], [128, 128]]
    for seed in seeds:
        assert any(seed.items() <= config.items() for config in transferred)


def test_a_wide_column_tile_keeps_the_largest_lane_unroll() -> None:
    """Column tiles of 2048 over 64 threads hold 32 lanes per thread, past
    the largest ``cute_lane_unroll`` factor: the seeds take 16, which the
    autotuner's transfer keeps (it turned 32 into 1, losing the
    loads-first lanes)."""
    with cpu_render_target():
        bound = cpu_bind(_wide_column_transpose, (torch.empty(1024, 8192),))
        # The registered block comes first in the config's lists.
        seeds = [
            dict(seed.config)
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("num_threads") == [64, 2]
        ]
        generation = bound.config_spec.create_config_generation()
        transferred = [
            dict(config.config) for _, config in generation.seed_flat_config_pairs()
        ]
    assert [seed["block_sizes"] for seed in seeds] == [[2048, 16], [2048, 32]]
    assert [seed["cute_lane_unroll"] for seed in seeds] == [16, 16]
    for seed in seeds:
        assert any(seed.items() <= config.items() for config in transferred)


def test_another_loops_tile_keeps_its_default_size() -> None:
    # The seeds size the transposed pair; the other loop's tile takes its
    # default (a seed with that slot left out was dropped).
    with cpu_render_target():
        bound = cpu_bind(
            _transpose_beside_a_row_loop,
            (torch.empty(2048, 4096), torch.empty(1000)),
        )
        default = bound.config_spec.default_config().config["block_sizes"]
        seeds = [
            dict(seed.config)
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_lane_layouts") == ["strided", "strided", "blocked"]
        ]
    assert [seed["block_sizes"] for seed in seeds] == [
        [16, 256, default[2]],
        [32, 128, default[2]],
    ]


def test_one_column_lane_per_thread_unrolls_nothing() -> None:
    # 32 columns over 32 threads: one column lane each, nothing to unroll.
    with cpu_render_target():
        bound = cpu_bind(
            _scaled_transpose, (torch.empty(256, 32, dtype=torch.bfloat16),)
        )
        seeds = [
            dict(seed.config)
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_lane_layouts") == ["strided"] * 2
        ]
    assert [seed["block_sizes"] for seed in seeds] == [[32, 32], [64, 32]]
    assert [seed["cute_lane_unroll"] for seed in seeds] == [1, 1]


def test_no_seeds_for_tiles_of_nested_loops() -> None:
    # No loop order spans both tiles, so the seeds could neither lead with
    # the column dim nor unroll lanes the two loops share.
    with cpu_render_target():
        bound = cpu_bind(_nested_transpose, (torch.empty(256, 512),))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_no_seeds_without_a_transposed_store() -> None:
    with cpu_render_target():
        bound = cpu_bind(_copy, (torch.randn(256, 512),))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_no_seeds_beside_a_row_broadcast() -> None:
    # ``scale`` has the store's rows on its contiguous dim, but it indexes
    # only the rows; an expanded column (strides (1, 0)) has them contiguous
    # too, and its other dim does not move: nothing in either kernel is
    # transposed.
    with cpu_render_target():
        for bound in (
            cpu_bind(_scale_rows, (torch.empty(256, 512), torch.empty(256))),
            cpu_bind(_copy, (torch.empty(256, 1).expand(256, 512),)),
        ):
            assert _NAME not in bound.config_spec.autotuner_heuristics


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestCuteTransposedStoreSeeds(TestCase):
    # One seed per dtype: the seeds of a dtype differ in tile shape only.
    def test_seeds_are_bitwise_torch(self) -> None:
        # 1021 columns padded to 1100 and 200 rows: partial tiles along both
        # dims, with either store first.
        x = torch.randn(200, 1021, device=DEVICE).to(torch.bfloat16)
        expected = torch.nn.functional.pad(x, (0, 79))
        for transpose_first in (True, False):
            args = (x, 1100, transpose_first)
            seed = _seeds(_pad_and_transpose.bind(args), 16)[0]
            _, (out_t, out) = code_and_output(_pad_and_transpose, args, **seed)
            torch.testing.assert_close(out_t, expected.t(), rtol=0, atol=0)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_fp32_and_int8_seeds_are_bitwise_torch(self) -> None:
        # Partial tiles along both dims: 100 (and 96) rows, 520 columns.
        for x, packet in (
            (torch.randn(100, 520, device=DEVICE), 8),
            (torch.randint(-32, 32, (96, 520), device=DEVICE, dtype=torch.int8), 16),
        ):
            seed = _seeds(_scaled_transpose.bind((x,)), packet)[0]
            _, out = code_and_output(_scaled_transpose, (x,), **seed)
            torch.testing.assert_close(out, (x * 2).t(), rtol=0, atol=0)
