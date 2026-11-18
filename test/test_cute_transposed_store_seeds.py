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


def _to_channels_last(x: torch.Tensor) -> torch.Tensor:
    """``[N, C, S]`` copied into the channels-last ``[N, S, C]``: the layout
    copy of better-benchmark's ``pointwise_06377502b8f5``."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


_channels_last = helion.kernel(
    _to_channels_last, backend="cute", static_shapes=True, autotune_effort="none"
)
_channels_last_dynamic = helion.kernel(
    _to_channels_last, backend="cute", static_shapes=False, autotune_effort="none"
)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _from_channels_last(x: torch.Tensor) -> torch.Tensor:
    """The reverse copy: channels-last ``[N, S, C]`` into ``[N, C, S]``."""
    n, plane, channels = x.shape
    out = torch.empty([n, channels, plane], dtype=x.dtype, device=x.device)
    for tile_n, tile_s, tile_c in hl.tile([n, plane, channels]):
        out[tile_n, tile_c, tile_s] = x[tile_n, tile_s, tile_c].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` into a tensor of the caller."""
    n, channels, plane = x.shape
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_masked(x: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` with an extra mask on the store."""
    n, channels, plane = x.shape
    out = torch.zeros([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        value = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
        hl.store(out, [tile_n, tile_s, tile_c], value, extra_mask=value > 0)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _from_channels_last_masked_load(
    x: torch.Tensor, keep: hl.constexpr
) -> torch.Tensor:
    """``_from_channels_last`` with an extra mask on the load: the first
    ``keep`` channels, zeros for the others."""
    n, plane, channels = x.shape
    out = torch.empty([n, channels, plane], dtype=x.dtype, device=x.device)
    for tile_n, tile_s, tile_c in hl.tile([n, plane, channels]):
        kept = (tile_c.index < keep)[None, None, :]
        value = hl.load(x, [tile_n, tile_s, tile_c], extra_mask=kept)
        out[tile_n, tile_c, tile_s] = value.permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_per_channel_tile(x: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` with the batch and position tiles in a loop of
    their own under the channel tiles."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_c in hl.tile(channels):
        for tile_n, tile_s in hl.tile([n, plane]):
            out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_under_a_loop(x: torch.Tensor, rounds: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` with its three tiles in a loop under another
    tile's (one copy per tile of ``rounds``)."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    for _tile_r in hl.tile(rounds.size(0)):
        for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
            out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_window(
    x: torch.Tensor, out: torch.Tensor, begin: hl.constexpr, end: hl.constexpr
) -> torch.Tensor:
    """``_channels_last_into`` for the channels from ``begin`` to ``end``."""
    n, _, plane = x.shape
    for tile_n, tile_c, tile_s in hl.tile([0, begin, 0], [n, end, plane]):
        out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _from_channels_last_window(
    x: torch.Tensor, out: torch.Tensor, begin: hl.constexpr, end: hl.constexpr
) -> torch.Tensor:
    """The reverse copy for the channels from ``begin`` to ``end``."""
    n, plane, _ = x.shape
    for tile_n, tile_s, tile_c in hl.tile([0, 0, begin], [n, plane, end]):
        out[tile_n, tile_c, tile_s] = x[tile_n, tile_s, tile_c].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_shifted_store(
    x: torch.Tensor, out: torch.Tensor, lane: hl.constexpr
) -> torch.Tensor:
    """``_channels_last_into`` through a store subscript that adds a scalar
    to a tile's index: 16 channels into the rows of ``out`` (``lane``), or
    two rows down."""
    n, channels, plane = x.shape
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        value = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
        if lane:
            out[tile_n, tile_s, tile_c.index + 16] = value
        else:
            out[tile_n, tile_s.index + 2, tile_c] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _from_channels_last_shifted_load(
    x: torch.Tensor, out: torch.Tensor, lane: hl.constexpr
) -> torch.Tensor:
    """The reverse copy through a load subscript that adds a scalar to a
    tile's index: of the channels of ``x`` from 16 on (``lane``), or of its
    rows from two down."""
    n, channels, plane = out.shape
    for tile_n, tile_s, tile_c in hl.tile([n, plane, channels]):
        if lane:
            value = x[tile_n, tile_s, tile_c.index + 16]
        else:
            value = x[tile_n, tile_s.index + 2, tile_c]
        out[tile_n, tile_c, tile_s] = value.permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_wide_channel_tiles(
    x: torch.Tensor, smallest: hl.constexpr
) -> torch.Tensor:
    """``_to_channels_last`` with channel tiles of ``smallest`` or more."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    block_c = hl.register_block_size(smallest, 2048)
    for tile_n, tile_c, tile_s in hl.tile(
        [n, channels, plane], block_size=[None, block_c, None]
    ):
        out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """The same layout change over four tiles."""
    n, channels, height, width = x.shape
    out = torch.empty([n, height, width, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_h, tile_w in hl.tile([n, channels, height, width]):
        out[tile_n, tile_h, tile_w, tile_c] = x[tile_n, tile_c, tile_h, tile_w].permute(
            0, 2, 3, 1
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_n, tile_c, tile_s in hl.tile(x.shape):
        out[tile_n, tile_c, tile_s] = x[tile_n, tile_c, tile_s] * 2
    return out


def _batched_seeds(bound: BoundKernel) -> list[dict[str, object]]:
    """The seeds of a batched layout change: the only ones that lead the
    loop order with the second tile and pin the default store policy."""
    return [
        dict(seed.config)
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_store_policy") == "default"
        and "loop_orders" in seed.config
    ]


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


def test_a_batched_layout_change_stores_whole_sectors() -> None:
    """NCHW to channels-last: the store has the channels contiguous, the load
    the positions.  Every thread stores one packet of channels and loads it
    at one position; the channel threads lead the thread order.  (Rows of 50
    positions hold no packets of the load.)"""
    with cpu_render_target():
        bound = cpu_bind(
            _channels_last, (torch.empty(4, 96, 50, dtype=torch.bfloat16),)
        )
        spec = bound.config_spec
        assert _NAME in spec.autotuner_heuristics
        seeds = _batched_seeds(bound)
        # 32-byte packets on two threads beside 16 and 32 positions, 16-byte
        # packets on four threads beside 8; the batch tile at its smallest
        # block (a 4-image batch is searched from 1).
        assert [seed["block_sizes"] for seed in seeds] == [
            [1, 32, 16],
            [1, 32, 32],
            [1, 32, 8],
        ]
        assert [seed["num_threads"] for seed in seeds] == [
            [1, 2, 16],
            [1, 2, 32],
            [1, 4, 8],
        ]
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 16, 1],
            [1, 16, 1],
            [1, 8, 1],
        ]
        for seed in seeds:
            assert seed["loop_orders"] == [[1, 2, 0]]
            assert seed["cute_lane_layouts"] == ["blocked"] * 3
            # No loads-first unroll: one packet per thread has nothing to
            # unroll, and the knob stays closed where only these seeds fire.
            assert "cute_lane_unroll" not in seed
        generation = spec.create_config_generation()
        transferred = [
            dict(config.config) for _, config in generation.seed_flat_config_pairs()
        ]
        for seed in seeds:
            assert any(seed.items() <= config.items() for config in transferred)
        # Each seed's thread stores its packet with one flush: a 32-byte one
        # and a 16-byte one.
        for seed, lanes in ((seeds[0], 16), (seeds[2], 8)):
            code = bound.to_code(spec.normalized_config(helion.Config(**seed)))
            assert code.count("_cute_store_u16_vec(") == 1
            assert f"cutlass.range_constexpr({lanes})" in code
            assert "_lane_unroll_loads_" not in code


def test_a_store_that_takes_no_packets_gets_packets_of_the_load() -> None:
    """Channels-last to NCHW with 49 positions: the stored rows hold no
    packet, the loaded rows of 96 channels do.  Every thread loads one
    packet of channels and stores it at one position; a warp of position
    threads leads the thread order, so each scalar store of a warp writes 32
    consecutive positions."""
    with cpu_render_target():
        bound = cpu_bind(
            _from_channels_last, (torch.empty(4, 49, 96, dtype=torch.bfloat16),)
        )
        spec = bound.config_spec
        assert _NAME in spec.autotuner_heuristics
        seeds = _batched_seeds(bound)
        # 32 position threads beside one and two threads of 32-byte packets
        # and two of 16-byte ones.
        assert [seed["block_sizes"] for seed in seeds] == [
            [1, 32, 16],
            [1, 32, 32],
            [1, 32, 16],
        ]
        assert [seed["num_threads"] for seed in seeds] == [
            [1, 32, 1],
            [1, 32, 2],
            [1, 32, 2],
        ]
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 1, 16],
            [1, 1, 16],
            [1, 1, 8],
        ]
        # The store's contiguous tile, the positions, leads here too.
        assert [seed["loop_orders"] for seed in seeds] == [[[1, 2, 0]]] * 3
        generation = spec.create_config_generation()
        transferred = [
            dict(config.config) for _, config in generation.seed_flat_config_pairs()
        ]
        for seed in seeds:
            assert any(seed.items() <= config.items() for config in transferred)
        code = bound.to_code(spec.normalized_config(helion.Config(**seeds[0])))
        assert code.count("cute.arch.load(") == 1
        assert "cutlass.range_constexpr(16)" in code
        assert "_cute_store_u16_vec(" not in code

        # Before SM100 the 16-byte packets alone.
        bound = cpu_bind(
            _from_channels_last,
            (torch.empty(4, 49, 96, dtype=torch.bfloat16),),
            capability=(9, 0),
        )
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 1, 8]]

        # A load one element into its storage holds no aligned packet.
        offset = torch.empty(4 * 49 * 96 + 1, dtype=torch.bfloat16)[1:]
        bound = cpu_bind(_from_channels_last, (offset.view(4, 49, 96),))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_both_sides_packets_seed_the_stores_first() -> None:
    # 48 positions and 96 channels: either side's rows hold 32-byte packets.
    with cpu_render_target():
        bound = cpu_bind(
            _from_channels_last, (torch.empty(4, 48, 96, dtype=torch.bfloat16),)
        )
        seeds = _batched_seeds(bound)
        # A masked store keeps the load's seeds.
        masked = cpu_bind(
            _channels_last_masked, (torch.empty(4, 96, 48, dtype=torch.bfloat16),)
        )
        masked_seeds = _batched_seeds(masked)
    assert [seed["cute_vector_widths"] for seed in seeds] == [
        [1, 16, 1],
        [1, 16, 1],
        [1, 8, 1],
        [1, 1, 16],
        [1, 1, 16],
        [1, 1, 8],
    ]
    assert [seed["cute_vector_widths"] for seed in masked_seeds] == [
        [1, 1, 16],
        [1, 1, 16],
        [1, 1, 8],
    ]


def test_a_batched_fp32_seed_holds_eight_channels_per_packet() -> None:
    with cpu_render_target():
        bound = cpu_bind(_channels_last, (torch.empty(4, 64, 50),))
        seeds = _batched_seeds(bound)
        assert [seed["block_sizes"] for seed in seeds] == [
            [1, 16, 16],
            [1, 16, 32],
            [1, 16, 8],
        ]
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 8, 1],
            [1, 8, 1],
            [1, 4, 1],
        ]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
    assert code.count("_cute_store_u32_vec(") == 1


def test_four_tiles_keep_both_batch_tiles_at_their_smallest_block() -> None:
    with cpu_render_target():
        bound = cpu_bind(
            _nchw_to_nhwc, (torch.empty(4, 96, 7, 7, dtype=torch.bfloat16),)
        )
        seeds = _batched_seeds(bound)
    # The width tile is the load's contiguous one: 7 positions take 8
    # threads in every seed, so two of the three shapes coincide.
    assert [seed["block_sizes"] for seed in seeds] == [[1, 32, 1, 8], [1, 32, 1, 8]]
    assert [seed["num_threads"] for seed in seeds] == [[1, 2, 1, 8], [1, 4, 1, 8]]
    assert [seed["cute_vector_widths"] for seed in seeds] == [
        [1, 16, 1, 1],
        [1, 8, 1, 1],
    ]
    # Channels, width, then the batch tiles.
    assert [seed["loop_orders"] for seed in seeds] == [[[1, 3, 0, 2]]] * 2


def test_larger_smallest_blocks_take_more_threads() -> None:
    """A large grid starts the search at larger blocks
    (``raise_grid_block_minimums``): the batch tile takes a thread per
    element of its smallest block and the 16-byte seed's position tile its
    smallest block.  (Rows of 4100 positions hold no packets of the load.)"""
    with cpu_render_target():
        bound = cpu_bind(
            _channels_last, (torch.empty(256, 2048, 4100, dtype=torch.bfloat16),)
        )
        seeds = _batched_seeds(bound)
    assert [seed["block_sizes"] for seed in seeds] == [
        [8, 32, 16],
        [8, 32, 32],
        [8, 32, 16],
    ]
    assert [seed["num_threads"] for seed in seeds] == [
        [8, 2, 16],
        [8, 2, 32],
        [8, 4, 16],
    ]


def test_a_smallest_block_past_the_cta_gets_no_seed() -> None:
    """Every thread takes one packet of the channel tile's smallest block
    and one position beside it.  Channel tiles of 512 or more take 32 x 16
    and 64 x 8 threads, a seed's whole CTA, and 32 x 32 would be past it:
    that shape gets no seed and the other two keep theirs.  Channel tiles
    of 1024 or more leave no shape a CTA.  (Rows of 50 positions hold no
    packets of the load.)"""
    x = torch.empty(2, 2048, 50, dtype=torch.bfloat16)
    with cpu_render_target():
        bound = cpu_bind(_channels_last_wide_channel_tiles, (x, 512))
        seeds = _batched_seeds(bound)
        # The registered channel block comes first.
        assert [seed["block_sizes"] for seed in seeds] == [[512, 1, 16], [512, 1, 8]]
        assert [seed["num_threads"] for seed in seeds] == [[32, 1, 16], [64, 1, 8]]
        bound = cpu_bind(_channels_last_wide_channel_tiles, (x, 1024))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_a_16_byte_target_gets_the_16_byte_seed_alone() -> None:
    # Before SM100 a packet is 16 bytes at most.
    with cpu_render_target():
        bound = cpu_bind(
            _channels_last,
            (torch.empty(4, 96, 50, dtype=torch.bfloat16),),
            capability=(9, 0),
        )
        seeds = _batched_seeds(bound)
    assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 8, 1]]
    assert [seed["num_threads"] for seed in seeds] == [[1, 4, 8]]


def test_batched_seeds_need_packets_the_lowering_proves() -> None:
    """72-channel bf16 rows (144 bytes) hold 16-byte packets only; 60-channel
    rows (120 bytes) hold none, as do the 289-position rows the two
    BatchNorm kernels of ``pointwise_06377502b8f5`` store into, a base one
    element into its storage, and a view that ends inside a packet.  (Rows
    of 50 positions hold no packets of the load.)"""
    bf16 = torch.bfloat16
    with cpu_render_target():
        bound = cpu_bind(_channels_last, (torch.empty(4, 72, 50, dtype=bf16),))
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 8, 1]]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1

        bound = cpu_bind(_channels_last, (torch.empty(4, 60, 50, dtype=bf16),))
        assert _NAME not in bound.config_spec.autotuner_heuristics

        # ``out`` as an argument: aligned, it gets the seeds.
        x = torch.empty(4, 96, 50, dtype=bf16)
        bound = cpu_bind(_channels_last_into, (x, torch.empty(4, 50, 96, dtype=bf16)))
        assert len(_batched_seeds(bound)) == 3
        offset = torch.empty(4 * 50 * 96 + 1, dtype=bf16)[1:].view(4, 50, 96)
        bound = cpu_bind(_channels_last_into, (x, offset))
        assert _NAME not in bound.config_spec.autotuner_heuristics
        # 88 of 96 channels: the rows are 32-byte aligned, but the tile's 88
        # channels are whole 16-byte packets only.
        narrow = torch.empty(4, 50, 96, dtype=bf16)[:, :, :88]
        bound = cpu_bind(
            _channels_last_into, (torch.empty(4, 88, 50, dtype=bf16), narrow)
        )
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 8, 1]]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1


def test_batched_seeds_need_tiles_that_begin_on_a_packet() -> None:
    """A tile's packets start at its begin plus multiples of their width:
    channel tiles that begin at 16 keep every seed, those that begin at 8
    the 16-byte one, and those that begin at 3 none (the lowering stores
    their channels one by one).  The same for the packets of a load.  (Rows
    of 50 positions hold no packets of the other side.)"""
    bf16 = torch.bfloat16
    wide = torch.empty(3, 128, 50, dtype=bf16)
    last = torch.empty(3, 50, 128, dtype=bf16)
    with cpu_render_target():
        bound = cpu_bind(_channels_last_window, (wide, last, 16, 112))
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 16, 1],
            [1, 16, 1],
            [1, 8, 1],
        ]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1

        bound = cpu_bind(_channels_last_window, (wide, last, 8, 104))
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 8, 1]]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1

        bound = cpu_bind(_channels_last_window, (wide, last, 3, 99))
        assert _NAME not in bound.config_spec.autotuner_heuristics

        bound = cpu_bind(_from_channels_last_window, (last, wide, 8, 104))
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 1, 8]]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("cute.arch.load(") == 1

        bound = cpu_bind(_from_channels_last_window, (last, wide, 3, 99))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_a_masked_load_gets_no_packets_of_its_own() -> None:
    """A masked load keeps a packet only where its mask is uniform over the
    packet: the first 40 of 96 channels end inside a 32-byte packet, whose
    channels the lowering then loads one by one.  A seed does not prove a
    mask, so a masked load gets no load-packet seed (rows of 50 positions
    hold no packets of the store).  The packets of the store's tile do not
    depend on it: rows of 48 positions keep the three store seeds."""
    bf16 = torch.bfloat16
    with cpu_render_target():
        bound = cpu_bind(
            _from_channels_last_masked_load, (torch.empty(4, 50, 96, dtype=bf16), 40)
        )
        assert _NAME not in bound.config_spec.autotuner_heuristics

        bound = cpu_bind(
            _from_channels_last_masked_load, (torch.empty(4, 48, 96, dtype=bf16), 40)
        )
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 16, 1],
            [1, 16, 1],
            [1, 8, 1],
        ]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1


def test_a_shifted_subscript_gets_no_packets_of_its_own() -> None:
    """A subscript that adds a scalar to a tile's index reads as the tile in
    the access's fact, and the lowering of a tile's lane packets admits it
    for loads only.  A store with one keeps its lanes scalar, on its
    contiguous dim (``tile_c.index + 16``) and on another
    (``tile_s.index + 2``): it gets no store-packet seed, and the load keeps
    its three.  A load with one on its contiguous dim keeps the packets the
    scalar leaves aligned, which a seed does not prove: it gets no
    load-packet seed, and the store keeps its three.  One on another dim of
    the load moves no packet: all six."""
    bf16 = torch.bfloat16
    store_seeds = [[1, 16, 1], [1, 16, 1], [1, 8, 1]]
    load_seeds = [[1, 1, 16], [1, 1, 16], [1, 1, 8]]
    with cpu_render_target():
        x = torch.empty(4, 96, 48, dtype=bf16)
        for lane, shape in ((True, (4, 48, 112)), (False, (4, 50, 96))):
            bound = cpu_bind(
                _channels_last_shifted_store, (x, torch.empty(shape, dtype=bf16), lane)
            )
            seeds = _batched_seeds(bound)
            assert [seed["cute_vector_widths"] for seed in seeds] == load_seeds
            code = bound.to_code(
                bound.config_spec.normalized_config(helion.Config(**seeds[0]))
            )
            assert code.count("cute.arch.load(") == 1
            assert "_cute_store_u16_vec(" not in code

        out = torch.empty(4, 96, 48, dtype=bf16)
        bound = cpu_bind(
            _from_channels_last_shifted_load,
            (torch.empty(4, 48, 112, dtype=bf16), out, True),
        )
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == store_seeds
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1
        assert "cute.arch.load(" not in code

        bound = cpu_bind(
            _from_channels_last_shifted_load,
            (torch.empty(4, 50, 96, dtype=bf16), out, False),
        )
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            *store_seeds,
            *load_seeds,
        ]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[3]))
        )
        assert code.count("cute.arch.load(") == 1


@pytest.mark.parametrize("dtype", [torch.int8, torch.int32, torch.float64])
def test_no_batched_seeds_for_dtypes_stored_as_scalars(dtype: torch.dtype) -> None:
    # Tile stores are packets for bf16, fp16 and fp32 only, and the seeds
    # ask the same of the loads: rows of 48 positions and 96 channels would
    # hold the packets of either side.
    with cpu_render_target():
        bound = cpu_bind(_channels_last, (torch.empty(4, 96, 48, dtype=dtype),))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_no_batched_seeds_without_a_layout_change_the_seeds_serve() -> None:
    bf16 = torch.bfloat16
    with cpu_render_target():
        for kernel, args in (
            # The same tile contiguous on both sides.
            (_batched_scale, (torch.empty(4, 96, 48, dtype=bf16),)),
            (
                _channels_last,
                (torch.empty(4, 48, 96, dtype=bf16).permute(0, 2, 1),),
            ),
            # A masked store takes no packets (and rows of 50 positions hold
            # none of the load).
            (_channels_last_masked, (torch.empty(4, 96, 50, dtype=bf16),)),
            # No loop order spans the three tiles (the channel tiles, whose
            # packets the store would take, are the root loop's).
            (
                _channels_last_per_channel_tile,
                (torch.empty(4, 96, 48, dtype=bf16),),
            ),
            # The three tiles of a loop under a root loop: its threads and
            # packets are not a launch grid's.
            (
                _channels_last_under_a_loop,
                (torch.empty(4, 96, 48, dtype=bf16), torch.empty(4)),
            ),
            # Dynamic shapes: no static extent to prove the packets on.
            (_channels_last_dynamic, (torch.empty(4, 96, 48, dtype=bf16),)),
        ):
            bound = cpu_bind(kernel, args)
            assert _NAME not in bound.config_spec.autotuner_heuristics


def test_batched_seeds_leave_the_search_space_as_it_was() -> None:
    """The seeds open no knob: a shape they decline (60-channel rows of 50
    positions) has the search space of one they serve."""
    with cpu_render_target():
        served = cpu_bind(
            _channels_last, (torch.empty(4, 96, 48, dtype=torch.bfloat16),)
        )
        declined = cpu_bind(
            _channels_last, (torch.empty(4, 60, 50, dtype=torch.bfloat16),)
        )
        assert len(_batched_seeds(served)) == 6
        assert _NAME not in declined.config_spec.autotuner_heuristics
        assert (
            served.config_spec.structural_fingerprint()
            == declined.config_spec.structural_fingerprint()
        )


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

    def test_batched_seeds_are_bitwise_torch(self) -> None:
        # Partial tiles along both lane dims: 80 bf16 channels in tiles of
        # 32 (the last tile holds one of its two packets), 72 fp32 channels
        # in tiles of 16, 50 positions in tiles of 16 and 8.  The 32-byte
        # and the 16-byte seed for bf16, the 32-byte one for fp32.
        for x, picks in (
            (torch.randn(3, 80, 50, device=DEVICE).to(torch.bfloat16), (0, 2)),
            (torch.randn(3, 72, 50, device=DEVICE), (0,)),
        ):
            seeds = _batched_seeds(_channels_last.bind((x,)))
            assert len(seeds) == 3
            for pick in picks:
                code, out = code_and_output(_channels_last, (x,), **seeds[pick])
                assert "_vec(" in code
                torch.testing.assert_close(out, x.permute(0, 2, 1), rtol=0, atol=0)
        # The load's packets where the store takes none: 50 positions in
        # tiles of 32 threads, 80 channels in packets of 16 (one thread, and
        # two with the last tile holding one of its two packets).
        x = torch.randn(3, 50, 80, device=DEVICE).to(torch.bfloat16)
        seeds = _batched_seeds(_from_channels_last.bind((x,)))
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 1, 16],
            [1, 1, 16],
            [1, 1, 8],
        ]
        for seed in seeds[:2]:
            code, out = code_and_output(_from_channels_last, (x,), **seed)
            assert "cute.arch.load(" in code
            torch.testing.assert_close(out, x.permute(0, 2, 1), rtol=0, atol=0)

    def test_store_seeds_beside_a_masked_load_are_bitwise_torch(self) -> None:
        # The first 40 of 80 channels, zeros for the others, into rows of 48
        # positions in tiles of 32 and 16 (the last tile of 32 holds one of
        # its two packets): the 32-byte and the 16-byte store seed.
        x = torch.randn(3, 48, 80, device=DEVICE).to(torch.bfloat16)
        expected = x.permute(0, 2, 1).clone()
        expected[:, 40:] = 0
        args = (x, 40)
        seeds = _batched_seeds(_from_channels_last_masked_load.bind(args))
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 16, 1],
            [1, 16, 1],
            [1, 8, 1],
        ]
        for seed in (seeds[1], seeds[2]):
            code, out = code_and_output(_from_channels_last_masked_load, args, **seed)
            assert "_cute_store_u16_vec(" in code
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_seeds_beside_a_shifted_subscript_are_bitwise_torch(self) -> None:
        # The packets of the other side beside a subscript that adds a
        # scalar to a tile's index: the load's under a store 16 channels
        # into its rows and under one two rows down (48 positions in tiles
        # of 32: the last tile holds one of its two packets), and the
        # store's under a load of the channels from 16 on.
        bf16 = torch.bfloat16
        x = torch.randn(3, 80, 48, device=DEVICE).to(bf16)
        for lane, shape in ((True, (3, 48, 96)), (False, (3, 50, 80))):
            expected = torch.zeros(shape, device=DEVICE, dtype=bf16)
            if lane:
                expected[:, :, 16:] = x.permute(0, 2, 1)
            else:
                expected[:, 2:] = x.permute(0, 2, 1)
            args = (x, torch.zeros(shape, device=DEVICE, dtype=bf16), lane)
            seeds = _batched_seeds(_channels_last_shifted_store.bind(args))
            assert [seed["cute_vector_widths"] for seed in seeds] == [
                [1, 1, 16],
                [1, 1, 16],
                [1, 1, 8],
            ]
            code, out = code_and_output(_channels_last_shifted_store, args, **seeds[1])
            assert "cute.arch.load(" in code
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        x = torch.randn(3, 48, 96, device=DEVICE).to(bf16)
        args = (x, torch.zeros(3, 80, 48, device=DEVICE, dtype=bf16), True)
        seeds = _batched_seeds(_from_channels_last_shifted_load.bind(args))
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 16, 1],
            [1, 16, 1],
            [1, 8, 1],
        ]
        code, out = code_and_output(_from_channels_last_shifted_load, args, **seeds[1])
        assert "_cute_store_u16_vec(" in code
        expected = x[:, :, 16:].permute(0, 2, 1)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_a_batched_seed_is_bitwise_torch_on_other_layouts(self) -> None:
        """A seed's config on layouts its packets are not proven on (a
        pinned config meets any input): what the lowering cannot prove it
        stores as scalars."""
        bf16 = torch.bfloat16
        aligned = (
            torch.empty(3, 96, 50, device=DEVICE, dtype=bf16),
            torch.empty(3, 50, 96, device=DEVICE, dtype=bf16),
        )
        wide, narrow = (
            seed
            for seed in _batched_seeds(_channels_last_into.bind(aligned))
            if seed["num_threads"] in ([1, 2, 16], [1, 4, 8])
        )
        values = torch.randn(3, 96, 100, device=DEVICE).to(bf16)
        # The load contiguous along the channels as well (a permuted view),
        # the store one element into its storage: no packet is aligned.
        permuted = values[:, :, :50].permute(0, 2, 1).contiguous().permute(0, 2, 1)
        offset = torch.zeros(3 * 50 * 96 + 1, device=DEVICE, dtype=bf16)[1:]
        # A strided load, and a store into 88 of the 96 channels of a wider
        # buffer: the rows end inside a 32-byte packet (scalar stores) and on
        # a 16-byte one (packets; the last tile holds three of its four).
        strided = values[:, :88, ::2]
        buffer = torch.zeros(3, 50, 96, device=DEVICE, dtype=bf16)
        # The load and the store in one storage.
        shared = torch.zeros(2, 3 * 96 * 50, device=DEVICE, dtype=bf16)
        shared[0] = values[:, :, 50:].reshape(-1)
        for x, out, seed, packets in (
            (permuted, offset.view(3, 50, 96), wide, False),
            (strided, buffer[:, :, :88], wide, False),
            (strided, buffer[:, :, :88], narrow, True),
            (shared[0].view(3, 96, 50), shared[1].view(3, 50, 96), wide, True),
        ):
            expected = x.permute(0, 2, 1).clone()
            code, result = code_and_output(_channels_last_into, (x, out), **seed)
            assert ("_cute_store_u16_vec(" in code) == packets
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        # The channels past the view keep their zeros.
        assert not buffer[:, :, 88:].any()

    def test_batched_seeds_off_a_tile_begin_are_bitwise_torch(self) -> None:
        """Channel tiles that begin at 8 keep the 16-byte seed (88 channels
        in tiles of 32: the last tile holds three of its four packets), for
        the store's packets and for the load's.  Tiles that begin at 3 get
        no seed; a seed of tiles that begin at 0 pinned on them stores
        scalars."""
        bf16 = torch.bfloat16
        values = torch.randn(3, 128, 50, device=DEVICE).to(bf16)
        expected = torch.zeros(3, 50, 128, device=DEVICE, dtype=bf16)
        expected[:, :, 8:96] = values.permute(0, 2, 1)[:, :, 8:96]

        args = (values, torch.zeros(3, 50, 128, device=DEVICE, dtype=bf16), 8, 96)
        (seed,) = _batched_seeds(_channels_last_window.bind(args))
        assert seed["cute_vector_widths"] == [1, 8, 1]
        code, out = code_and_output(_channels_last_window, args, **seed)
        assert "_cute_store_u16_vec(" in code
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

        last = values.permute(0, 2, 1).contiguous()
        args = (last, torch.zeros(3, 128, 50, device=DEVICE, dtype=bf16), 8, 96)
        (seed,) = _batched_seeds(_from_channels_last_window.bind(args))
        assert seed["cute_vector_widths"] == [1, 1, 8]
        code, out = code_and_output(_from_channels_last_window, args, **seed)
        assert "cute.arch.load(" in code
        torch.testing.assert_close(out, expected.permute(0, 2, 1), rtol=0, atol=0)

        args = (values, torch.zeros(3, 50, 128, device=DEVICE, dtype=bf16), 0, 96)
        wide = _batched_seeds(_channels_last_window.bind(args))[0]
        assert wide["cute_vector_widths"] == [1, 16, 1]
        args = (values, torch.zeros(3, 50, 128, device=DEVICE, dtype=bf16), 3, 99)
        assert _batched_seeds(_channels_last_window.bind(args)) == []
        code, out = code_and_output(_channels_last_window, args, **wide)
        assert "_cute_store_u16_vec(" not in code
        expected = torch.zeros(3, 50, 128, device=DEVICE, dtype=bf16)
        expected[:, :, 3:99] = values.permute(0, 2, 1)[:, :, 3:99]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
