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

from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.autotuner_heuristics.cute import CuteColumnReductionHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteRegisterTileHeuristic
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Callable

    import helion


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
    ("kernel", "make_arguments"),
    (
        pytest.param(
            col_sum_f32_out,
            lambda: (torch.empty((128, 1024), dtype=torch.bfloat16),),
            id="bf16-column-sum-fp32-out",
        ),
        pytest.param(
            col_scale_three,
            lambda: tuple(
                torch.empty((128, 1024), dtype=torch.float32) for _ in range(3)
            ),
            id="three-stashed-tiles",
        ),
    ),
)
def test_seed_compiles_bodies_the_register_tile_cannot_schedule(
    kernel: helion.Kernel, make_arguments: Callable[[], tuple[Any, ...]]
) -> None:
    # The device IR admits these bodies, so the seed is offered; the split-time
    # lowering rejects them (a per-element fp32 store of a bf16 tile, a
    # live-value budget overrun) and the seed compiles with the rolled lane
    # nesting instead of raising.
    arguments = make_arguments()
    bound = _cpu_bind(kernel, arguments)
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
