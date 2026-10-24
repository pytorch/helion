"""The autotuner seed for register-tile lane reductions (CPU codegen)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_register_tile_kernels import col_scale_three
from test._cute_register_tile_kernels import col_softmax
from test._cute_register_tile_kernels import col_sum_dynamic
from test._cute_register_tile_kernels import col_sum_f32_out
from test._cute_register_tile_kernels import col_sum_guarded
from test._cute_register_tile_kernels import col_sum_nested
from test._cute_vloop_sink_kernels import col_reduce_sum_static
from test.test_cute_register_tile_reductions import COLUMN_REDUCE
from test.test_cute_register_tile_reductions import CONSTEXPR_OWNER
from test.test_cute_register_tile_reductions import ROLLED_OWNER
from test.test_cute_register_tile_reductions import _bind
from test.test_cute_register_tile_reductions import _cpu_only  # noqa: F401

import helion
from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.autotuner_heuristics.cute import CuteColumnReductionHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteRegisterTileHeuristic
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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
    bound = _cpu_bind(kernel, arguments)
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


@skipUnlessBackends(["cute"])
def test_seed_keeps_bf16_to_fp32_column_store_on_sm100() -> None:
    arguments = (torch.empty((128, 1024), dtype=torch.bfloat16),)
    bound = _cpu_bind(col_sum_f32_out, arguments)
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


@skipUnlessBackends(["cute"])
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
    bound = _cpu_bind(kernel, arguments)
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


@skipUnlessBackends(["cute"])
def test_seed_skips_bodies_outside_the_register_tile_shape() -> None:
    # A rolled device loop, one reduction feeding another, or a dynamic
    # reduction extent: the block is not admitted and no seed is offered.
    for kernel, arguments in (
        (
            col_sum_nested,
            (
                torch.empty((128, 1024), dtype=torch.float32),
                torch.empty((4, 1024), dtype=torch.float32),
            ),
        ),
        (col_softmax, (torch.empty((128, 1024), dtype=torch.float32),)),
        (col_sum_dynamic, (torch.empty((1024, 1024), dtype=torch.float32),)),
    ):
        bound = _cpu_bind(kernel, arguments)
        host = bound.host_function
        assert host is not None
        assert not CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)


@skipUnlessBackends(["cute"])
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
    bound = _cpu_bind(col_reduce_sum_static, arguments)
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


@skipUnlessBackends(["cute"])
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
        bound = _cpu_bind(kernel, arguments)
        host = bound.host_function
        assert host is not None
        assert not CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)
        assert bound.config_spec.compiler_default_config is None
        default = bound.config_spec.default_config()
        assert "num_threads" not in default.config
        bound.to_code(default)
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = _cpu_bind(_col_sum_scale_in_place, arguments)
    host = bound.host_function
    assert host is not None
    assert CuteRegisterTileHeuristic.is_eligible(bound.env, host.device_ir)
    default = bound.config_spec.default_config()
    assert default.num_threads == [8, 8]
    assert "_lane_stash" in bound.to_code(default)
