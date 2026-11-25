"""The autotuner seed for register-tile lane reductions (CPU codegen)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch

from test._cute_register_tile_kernels import col_scale_three
from test._cute_register_tile_kernels import col_softmax
from test._cute_register_tile_kernels import col_sum_dynamic
from test._cute_register_tile_kernels import col_sum_f32_out
from test._cute_register_tile_kernels import col_sum_guarded
from test._cute_register_tile_kernels import col_sum_nested
from test.test_cute_register_tile_reductions import COLUMN_REDUCE
from test.test_cute_register_tile_reductions import CONSTEXPR_OWNER
from test.test_cute_register_tile_reductions import ROLLED_OWNER
from test.test_cute_register_tile_reductions import _bind

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_specialization_facts
from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.cute.autotune.heuristics.cute import CuteColumnReductionHeuristic
from helion._compiler.cute.autotune.heuristics.cute import CuteRegisterTileHeuristic
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_vloop_sink_kernels import col_reduce_sum_static
import helion.language as hl

# ``test_cute_register_tile_reductions``'s CPU model (its ``_cpu_only`` fixture
# was imported here before it became this marker).
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(reduction_smem_budget_bytes=232448),
]

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel


def test_register_tile_heuristic_seeds_the_geometry() -> None:
    bound, _arguments = _bind()
    host = bound.host_function
    assert host is not None
    assert CuteRegisterTileHeuristic in get_heuristics("cute")
    assert CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)
    seed = CuteRegisterTileHeuristic.get_seed_config(bound.env, host.device_ir)
    assert seed is not None
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    (dv,) = spec.block_sizes.valid_block_ids()
    threads = dict(
        zip(spec.num_threads.valid_block_ids(), seed.num_threads, strict=True)
    )
    widths = dict(
        zip(
            spec.cute_vector_widths.valid_block_ids(),
            cast("list[int]", seed.config["cute_vector_widths"]),
            strict=True,
        )
    )
    # Eight vector threads x V=4 cover a 128-byte row segment; eight
    # reduction threads leave 16 lanes x 4 = 64 elements per thread.
    assert seed.block_sizes == [32]
    assert threads[dv] == 8 and threads[reduction] == 8
    assert widths[dv] == 4
    assert seed.reduction_loops == [None]
    normalized = spec.normalized_config(seed)
    assert normalized.reduction_loops == [None]
    code = bound.to_code(normalized)
    assert "_lane_stash" in code
    assert "block=(8, 8, 1)" in code
    assert code.count(COLUMN_REDUCE) == 1


@pytest.mark.parametrize(
    ("kernel", "make_arguments", "capability"),
    (
        pytest.param(
            col_sum_f32_out,
            lambda: (torch.empty((128, 1024), dtype=torch.bfloat16),),
            (9, 0),
            id="bf16-column-sum-fp32-out",
        ),
        pytest.param(
            col_scale_three,
            lambda: tuple(
                torch.empty((128, 1024), dtype=torch.float32) for _ in range(3)
            ),
            (10, 0),
            id="three-stashed-tiles",
        ),
    ),
)
def test_seed_compiles_bodies_the_register_tile_cannot_schedule(
    kernel: helion.Kernel,
    make_arguments: Callable[[], tuple[Any, ...]],
    capability: tuple[int, int],
) -> None:
    # The device IR admits these bodies, so the seed is offered; the split-time
    # lowering rejects them (a 32-byte fp32 store on SM90, a
    # live-value budget overrun) and the seed compiles with the rolled lane
    # nesting instead of raising.
    arguments = make_arguments()
    bound = cpu_bind(kernel, arguments)
    bound.config_spec.target_device_capability = capability
    host = bound.host_function
    assert host is not None
    assert CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)
    seed = CuteRegisterTileHeuristic.get_seed_config(bound.env, host.device_ir)
    assert seed is not None
    normalized = bound.config_spec.normalized_config(seed)
    assert normalized.reduction_loops == [None]
    code = bound.to_code(normalized)
    assert ROLLED_OWNER in code
    assert CONSTEXPR_OWNER not in code
    assert "_lane_stash" not in code


def test_seed_keeps_bf16_to_fp32_column_store_on_sm100() -> None:
    arguments = (torch.empty((128, 1024), dtype=torch.bfloat16),)
    bound = cpu_bind(col_sum_f32_out, arguments)
    bound.config_spec.target_device_capability = (10, 0)
    host = bound.host_function
    assert host is not None
    seed = CuteRegisterTileHeuristic.get_seed_config(bound.env, host.device_ir)
    assert seed is not None
    config = bound.config_spec.normalized_config(seed)
    assert config.reduction_loops == [None]
    code = bound.to_code(config)
    assert CONSTEXPR_OWNER in code
    assert ROLLED_OWNER not in code
    assert code.count(COLUMN_REDUCE) == 1
    assert code.count("_cute_store_u32_vec(out.iterator") == 1


@pytest.mark.parametrize(
    ("kernel", "make_arguments"),
    (
        pytest.param(
            col_sum_guarded,
            lambda: (torch.empty((1024, 1024), dtype=torch.float32), 1),
            id="guard-inside-the-element-loops",
        ),
        pytest.param(
            col_scale_three,
            lambda: tuple(
                torch.empty((1024, 1024), dtype=torch.float32) for _ in range(3)
            ),
            id="three-stashed-tiles",
        ),
    ),
)
def test_seed_retries_wide_extents_as_the_looped_reduction(
    kernel: helion.Kernel, make_arguments: Callable[[], tuple[Any, ...]]
) -> None:
    # Over 1024 rows the seed asks for 64 reduction threads (16 lanes of four
    # elements per thread), a thread count that covers the extent only as a
    # register tile.  The split rejects these bodies, and the retry compiles
    # the seed as the same geometry without the register tile: the looped
    # reduction (reduction_loops=[128]), not a 32-thread scalar lane.
    arguments = make_arguments()
    bound = cpu_bind(kernel, arguments)
    host = bound.host_function
    assert host is not None
    seed = CuteRegisterTileHeuristic.get_seed_config(bound.env, host.device_ir)
    assert seed is not None
    spec = bound.config_spec
    normalized = spec.normalized_config(seed)
    assert normalized.reduction_loops == [None]
    looped = spec.normalized_config(seed, _cute_register_tiles=False)
    assert looped.reduction_loops == [128]
    code = bound.to_code(normalized)
    assert code == bound.to_code(looped)
    assert "synthetic_lane_1" not in code
    assert "block=(64, 8, 1)" in code


def test_seed_skips_bodies_outside_the_register_tile_shape() -> None:
    # A rolled device loop or a dynamic reduction extent: the block is not
    # admitted and no seed is offered.
    for kernel, arguments in (
        (
            col_sum_nested,
            (
                torch.empty((128, 1024), dtype=torch.float32),
                torch.empty((4, 1024), dtype=torch.float32),
            ),
        ),
        (col_sum_dynamic, (torch.empty((1024, 1024), dtype=torch.float32),)),
    ):
        bound = cpu_bind(kernel, arguments)
        host = bound.host_function
        assert host is not None
        assert not CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _narrow_col_sum(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile_c in hl.tile(x.size(1)):
        out[tile_c] = torch.sum(x[:, tile_c].to(torch.float32), -2)


def test_seed_tiles_narrow_extents_by_a_power_of_two() -> None:
    # A column sum over 48 bf16 columns (fewer than the eight 8-element vector
    # threads the seed asks for): the tile is the widest power of two that
    # divides the extent, 16, not the extent itself, which is no block size.
    # The seed is the autotune-off default, so the default config must
    # normalize and render (columns // block CTAs); with block_sizes=[48] it
    # raised InvalidConfig. A power-of-two extent keeps the whole extent as
    # its tile.
    # fp32 columns take 4-element vectors: 20 columns are 5 tiles of 4.
    bf16, f32 = torch.bfloat16, torch.float32
    for columns, dtype, block in (
        (48, bf16, 16),
        (24, bf16, 8),
        (32, bf16, 32),
        (20, f32, 4),
    ):
        arguments = (
            torch.empty((128, columns), dtype=dtype),
            torch.empty((columns,), dtype=torch.float32),
        )
        bound = cpu_bind(_narrow_col_sum, arguments)
        host = bound.host_function
        assert host is not None
        seed = CuteRegisterTileHeuristic.get_seed_config(bound.env, host.device_ir)
        assert seed is not None
        assert seed.block_sizes == [block]
        default = bound.config_spec.default_config()
        assert default.block_sizes == [block]
        code = bound.to_code(default)
        assert f"_launcher(_helion__narrow_col_sum, ({columns // block},)" in code


def test_a_short_grid_seeds_a_narrower_tile() -> None:
    # A column sum over 768 bf16 columns: the primary's 64-column tiles (8
    # vector threads of 8 elements) make 12 CTAs, fewer than half of 148
    # SMs, so a fourth seed halves the tile and its vectors while the grid
    # stays that short, down to 2-element vectors: 48 CTAs of 16 columns,
    # with the same threads and the schedule knobs. 16384 columns (256 CTAs)
    # get no such seed.
    for columns, narrow in ((768, (16, 2)), (16384, None)):
        arguments = (
            torch.empty((128, columns), dtype=torch.bfloat16),
            torch.empty((columns,), dtype=torch.float32),
        )
        bound = cpu_bind(_narrow_col_sum, arguments)
        host = bound.host_function
        assert host is not None
        seeds = CuteRegisterTileHeuristic.get_seed_configs(bound.env, host.device_ir)
        assert seeds is not None
        spec = bound.config_spec
        (tile,) = spec.block_sizes.valid_block_ids()
        widths = spec.cute_vector_widths.valid_block_ids()
        tiles = [
            (seed.block_sizes[0], seed.config["cute_vector_widths"][widths.index(tile)])
            for seed in seeds
        ]
        if narrow is None:
            assert len(seeds) == 3, tiles
            continue
        assert tiles == [(64, 8), (64, 8), (64, 8), narrow]
        assert seeds[3].num_threads == seeds[1].num_threads
        assert seeds[3].config["cute_register_tile_warp_fold"] is True
        assert seeds[3].config["cute_register_tile_store_first"] is True
        code = bound.to_code(seeds[3])
        assert f"_launcher(_helion__narrow_col_sum, ({columns // 16},)" in code


def test_the_short_grid_seed_keys_its_binding_on_the_sm_count() -> None:
    # 4224 bf16 columns make 66 CTAs of 64 columns: not fewer than half of
    # 132 SMs, fewer than half of 148, where the narrower tile (132 CTAs of
    # 32 columns, 4-element vectors) joins the seeds. So a binding is keyed
    # on the device's SM count (CACHE_SPECIALIZATION_FACTS): one made on 132
    # SMs and reused on 148 would miss the narrower tile.
    assert compiler_seed_specialization_facts(
        "cute", ["cute_register_tile"]
    ) == frozenset({"device_num_sm"})
    arguments = (
        torch.empty((128, 4224), dtype=torch.bfloat16),
        torch.empty((4224,), dtype=torch.float32),
    )
    tiles = [
        _seed_tiles(cpu_bind(_narrow_col_sum, arguments, num_sm=num_sm))
        for num_sm in (132, 148)
    ]
    assert tiles[0] < tiles[1]
    assert tiles[1] - tiles[0] == {(32, 4)}


def _seed_tiles(bound: BoundKernel[object]) -> set[tuple[int, int]]:
    """The (tile block, vector width) of the binding's compiler seeds that
    set vector widths, for a kernel of one tiled block."""
    spec = bound.config_spec
    (tile,) = spec.block_sizes.valid_block_ids()
    widths = spec.cute_vector_widths.valid_block_ids()
    return {
        (
            seed.config["block_sizes"][0],
            seed.config["cute_vector_widths"][widths.index(tile)],
        )
        for seed in spec.compiler_seed_configs
        if "cute_vector_widths" in seed.config
    }


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_col_sum(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile_b, tile_c in hl.tile([x.size(0), x.size(2)], block_size=[1, None]):
        out[tile_b, tile_c] = torch.sum(x[tile_b, :, tile_c].to(torch.float32), -2)


def test_a_fixed_tile_counts_its_ctas_in_the_short_grid() -> None:
    # 32 images of a column sum over 768 bf16 columns, the images a tile of
    # fixed block 1 (no autotuner block size): the primary's 12 tiles of 64
    # columns per image make 384 CTAs, more than half of 148 SMs, so no
    # narrower tile joins the seeds; one image (12 CTAs) gets it.
    for images, narrow in ((32, False), (1, True)):
        arguments = (
            torch.empty((images, 128, 768), dtype=torch.bfloat16),
            torch.empty((images, 768), dtype=torch.float32),
        )
        bound = cpu_bind(_batched_col_sum, arguments)
        host = bound.host_function
        assert host is not None
        seeds = CuteRegisterTileHeuristic.get_seed_configs(bound.env, host.device_ir)
        assert seeds
        assert (len(seeds) == 4) == narrow, (images, len(seeds))


def test_chained_reductions_seed_half_the_lanes_per_level() -> None:
    # One reduction feeding another (two accumulate nests) stashes more of the
    # tile between the combines, so the seed halves the lanes per thread:
    # 128 rows over 16 reduction threads instead of 8, with the same 32-column
    # tile on 8 vector threads.  The chained body is also the autotune-off
    # default and renders the two-level register tile.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = cpu_bind(col_softmax, arguments)
    host = bound.host_function
    assert host is not None
    assert CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)
    seeds = CuteRegisterTileHeuristic.get_seed_configs(bound.env, host.device_ir)
    assert seeds is not None
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    thread_ids = spec.num_threads.valid_block_ids()
    # The fourth seed is the narrower tile of a short grid (32 CTAs of 32
    # columns on 148 SMs: 64 of 16, _short_grid_tile).
    assert [seed.block_sizes for seed in seeds] == [[32], [32], [32], [16]]
    assert [seed.num_threads[thread_ids.index(reduction)] for seed in seeds] == [
        16,
        16,
        32,
        16,
    ]
    default = spec.default_config()
    assert default.num_threads[thread_ids.index(reduction)] == 16
    code = bound.to_code(default)
    assert code.count(CONSTEXPR_OWNER) == 3
    assert "_lane_stash" in code
    assert "block=(8, 16, 1)" in code


def test_register_tile_and_column_reduction_seeds_are_disjoint() -> None:
    # Both heuristics seed column reductions, of different IR shapes:
    # ``CuteColumnReductionHeuristic`` a reduction over a serial device-loop
    # axis (a kernel without reduction loops, for ``cute_vloop_sink``), this
    # seed a reduction-loop block reduced by a persistent lane (the register
    # tile).  No kernel is eligible for both, so the seeds need no precedence.
    bound, _arguments = _bind()
    host = bound.host_function
    assert host is not None
    assert CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)
    assert not CuteColumnReductionHeuristic.is_eligible(bound.env, host.device_ir)
    arguments = (torch.empty((1024, 1024), dtype=torch.bfloat16),)
    bound = cpu_bind(col_reduce_sum_static, arguments)
    host = bound.host_function
    assert host is not None
    assert not bound.config_spec.reduction_loops
    assert CuteColumnReductionHeuristic.is_eligible(bound.env, host.device_ir)
    assert not CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _col_sum_accumulate(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(1)], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(x.size(1)):
        out[tile_n] += x[:, tile_n].sum(0)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _col_sum_reload(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty([x.size(1)], dtype=torch.float32, device=x.device)
    other = torch.empty([x.size(1)], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(x.size(1)):
        out[tile_n] = x[:, tile_n].sum(0)
        other[tile_n] = out[tile_n] * 2.0
    return out, other


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _col_sum_scale_in_place(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(1)], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(x.size(1)):
        t = x[:, tile_n]
        x[:, tile_n] = t * 2.0
        out[tile_n] = t.sum(0)
    return out


def test_seed_skips_bodies_whose_lane_split_would_order_memory() -> None:
    # The lane split cannot order two threads' accesses to one tensor: no
    # barrier fits next to the lane-reduce placeholder, and the RAW barrier
    # the IR marks cannot be predicated.  An accumulating store (every
    # reduction thread loads and stores the same ``out[tile]``) and a store
    # the body reads back get no seed, so the autotune-off default is the
    # base default, which compiles.  A read-modify-write whose accesses span
    # the reduction block keeps each element in one thread and keeps the seed.
    for kernel in (_col_sum_accumulate, _col_sum_reload):
        arguments = (torch.empty((128, 1024), dtype=torch.float32),)
        bound = cpu_bind(kernel, arguments)
        host = bound.host_function
        assert host is not None
        assert not CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)
        assert bound.config_spec.compiler_default_config is None
        default = bound.config_spec.default_config()
        assert "num_threads" not in default.config
        bound.to_code(default)
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = cpu_bind(_col_sum_scale_in_place, arguments)
    host = bound.host_function
    assert host is not None
    assert CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)
    default = bound.config_spec.default_config()
    assert default.num_threads == [8, 8]
    assert "_lane_stash" in bound.to_code(default)
