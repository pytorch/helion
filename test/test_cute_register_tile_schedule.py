"""Schedule knobs of the CuTe register tile (CPU codegen).

``cute_register_tile_warp_fold`` folds the multi-warp shared-memory column
reduce within each warp first (``_cute_grouped_reduce_shared_columns(...,
fold='warp')``); ``cute_register_tile_store_first`` emits the per-lane consume
nest (the state stores) before the reductions' cross-thread combines.  Both
are off by default and render the knob-free schedule byte for byte; a knob the
tile cannot honour is ignored (see ``RegisterTileOptions``).
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

from examples.linear.linear_attention_engine import recurrent_step_correction_fused
from examples.linear.linear_attention_engine import recurrent_step_fused
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test._cute_register_tile_kernels import col_scale
from test._cute_register_tile_kernels import col_sum
from test._cute_register_tile_kernels import column_config

import helion
from helion._compiler.autotuner_heuristics.cute import CuteRegisterTileHeuristic
from helion._compiler.cute.register_tile_admission import RegisterTileUnsupported
from helion._compiler.generate_ast import GenerateAST
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])

BH, D, DV = 8, 128, 128
COLUMN_REDUCE = "_cute_grouped_reduce_shared_columns("
WARP_FOLD = "fold='warp'"
STATE_STORE = "_cute_store_u32_vec(state.iterator"
KNOBS: dict[str, Any] = {
    "cute_register_tile_warp_fold": True,
    "cute_register_tile_store_first": True,
}


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bind() -> tuple[Any, tuple[torch.Tensor, ...]]:
    """Bind the fused recurrent step on CPU; the caller keeps ``arguments``
    alive while generating code (the alignment proof reads the live inputs)."""
    kernel = helion.kernel(
        recurrent_step_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    arguments = (
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, DV), dtype=torch.float32),
        torch.empty((BH, D, DV), dtype=torch.float32),
        torch.empty((BH,), dtype=torch.float32),
    )
    return _cpu_bind(kernel, arguments), arguments


def _config(
    bound: Any,
    *,
    dv_threads: int,
    reduction_threads: int,
    dv_block: int = 32,
    **extra: Any,
) -> helion.Config:
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    (dv,) = spec.block_sizes.valid_block_ids()
    threads = {dv: dv_threads, reduction: reduction_threads}
    return spec.normalized_config(
        helion.Config(
            block_sizes=[dv_block],
            num_threads=[
                threads.get(block_id, 0)
                for block_id in spec.num_threads.valid_block_ids()
            ],
            reduction_loops=[None],
            cute_vector_widths=[
                4 if block_id == dv else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            **extra,
        )
    )


def _kernel_statements(code: str) -> list[str]:
    """The kernel body's top-level statements, unparsed, in program order."""
    module = ast.parse(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    return [ast.unparse(node) for node in kernel.body]


def _first_statement(statements: list[str], needle: str) -> int:
    return next(i for i, source in enumerate(statements) if needle in source)


def _owner_loops(statements: list[str]) -> list[int]:
    return [
        i
        for i, source in enumerate(statements)
        if source.startswith("for synthetic_lane_")
    ]


def test_knobs_off_render_the_knob_free_schedule() -> None:
    bound, _arguments = _bind()
    knob_free = bound.to_code(_config(bound, dv_threads=8, reduction_threads=16))
    explicit_off = bound.to_code(
        _config(
            bound,
            dv_threads=8,
            reduction_threads=16,
            cute_register_tile_warp_fold=False,
            cute_register_tile_store_first=False,
        )
    )
    assert explicit_off == knob_free
    assert WARP_FOLD not in knob_free
    statements = _kernel_statements(knob_free)
    # Knob-free order: accumulate nest, combine, consume nest (state store).
    assert _first_statement(statements, COLUMN_REDUCE) < _first_statement(
        statements, STATE_STORE
    )


def test_warp_fold_requests_the_fold_of_the_multi_warp_column_reduce() -> None:
    bound, _arguments = _bind()
    code = bound.to_code(
        _config(
            bound,
            dv_threads=8,
            reduction_threads=16,
            cute_register_tile_warp_fold=True,
        )
    )
    assert code.count(COLUMN_REDUCE) == 1
    assert f"pre=8, group_span=128, group_count=1, {WARP_FOLD})" in code
    statements = _kernel_statements(code)
    # The fold alone keeps the combine ahead of the consume nest.
    assert _first_statement(statements, COLUMN_REDUCE) < _first_statement(
        statements, STATE_STORE
    )
    # Everything else is the knob-free render.
    knob_free = bound.to_code(_config(bound, dv_threads=8, reduction_threads=16))
    assert code.replace(f", {WARP_FOLD})", ")") == knob_free


def test_store_first_emits_the_consume_nest_before_the_combine() -> None:
    bound, _arguments = _bind()
    code = bound.to_code(
        _config(
            bound,
            dv_threads=8,
            reduction_threads=8,
            cute_register_tile_store_first=True,
        )
    )
    statements = _kernel_statements(code)
    accumulate, consume = _owner_loops(statements)
    combine = _first_statement(statements, COLUMN_REDUCE)
    out_store = _first_statement(statements, "_cute_store_u32_vec(out.iterator")
    assert accumulate < consume < combine < out_store
    assert STATE_STORE in statements[consume]
    assert "_lane_stash" in statements[accumulate]
    assert WARP_FOLD not in code
    # The same statements as the knob-free render, reordered.
    knob_free = _kernel_statements(
        bound.to_code(_config(bound, dv_threads=8, reduction_threads=8))
    )
    assert sorted(statements) == sorted(knob_free)
    assert statements != knob_free


def test_both_knobs_compose() -> None:
    bound, _arguments = _bind()
    code = bound.to_code(_config(bound, dv_threads=8, reduction_threads=16, **KNOBS))
    statements = _kernel_statements(code)
    _accumulate, consume = _owner_loops(statements)
    assert consume < _first_statement(statements, COLUMN_REDUCE)
    assert f"pre=8, group_span=128, group_count=1, {WARP_FOLD})" in code


def test_store_first_keeps_the_combine_first_when_a_store_needs_the_result() -> None:
    # ``y = x * (1 / (sum(x * x) + 1))``: the per-lane store reads the
    # reduction result, so the consume nest cannot move ahead of the combine
    # and the knob renders the knob-free schedule.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = _cpu_bind(col_scale, arguments)
    knob_free = bound.to_code(
        column_config(bound, block=32, tile_threads=8, reduction_threads=8)
    )
    code = bound.to_code(
        column_config(
            bound,
            block=32,
            tile_threads=8,
            reduction_threads=8,
            cute_register_tile_store_first=True,
        )
    )
    assert code == knob_free
    statements = _kernel_statements(code)
    assert _first_statement(statements, COLUMN_REDUCE) < _first_statement(
        statements, "_cute_store_u32_vec(y.iterator"
    )


def test_warp_fold_is_ignored_by_the_consecutive_lane_column_reduce() -> None:
    # One vector thread under 64 consecutive reduction threads (``pre == 1``):
    # that branch of the helper already folds each warp with a shuffle, so the
    # knob adds nothing and the render is the knob-free one.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = _cpu_bind(col_sum, arguments)
    knob_free = bound.to_code(
        column_config(bound, block=4, tile_threads=1, reduction_threads=64)
    )
    code = bound.to_code(
        column_config(
            bound,
            block=4,
            tile_threads=1,
            reduction_threads=64,
            cute_register_tile_warp_fold=True,
        )
    )
    assert code == knob_free
    assert "pre=1, group_span=64" in code


def test_knobs_are_dropped_without_a_register_tile() -> None:
    bound, _arguments = _bind()
    spec = bound.config_spec
    looped = spec.normalized_config(
        helion.Config(block_sizes=[32], reduction_loops=[64], **KNOBS)
    )
    assert looped.config["cute_register_tile_warp_fold"] is False
    assert looped.config["cute_register_tile_store_first"] is False
    # The rolled-lane retry re-normalizes with the register tile withheld.
    tile = _config(bound, dv_threads=8, reduction_threads=16, **KNOBS)
    assert tile.config["cute_register_tile_warp_fold"] is True
    retried = spec.normalized_config(tile, _cute_register_tiles=False)
    assert retried.config["cute_register_tile_warp_fold"] is False
    assert retried.config["cute_register_tile_store_first"] is False
    assert {
        key: value for key, value in retried.config.items() if key not in KNOBS
    } == {key: value for key, value in tile.config.items() if key not in KNOBS}


def test_seed_alternates_carry_the_knobs_and_the_finer_tile() -> None:
    bound, _arguments = _bind()
    spec = bound.config_spec
    env = bound.env
    device_ir = bound.host_function.device_ir
    primary = CuteRegisterTileHeuristic.get_seed_config(env, device_ir)
    seeds = CuteRegisterTileHeuristic.get_seed_configs(env, device_ir)
    assert primary is not None and seeds is not None
    assert seeds[0].config == primary.config
    assert "cute_register_tile_warp_fold" not in primary.config
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    thread_ids = spec.num_threads.valid_block_ids()
    reduction_threads = [
        seed.num_threads[thread_ids.index(reduction)] for seed in seeds
    ]
    assert reduction_threads == [8, 8, 16]
    for seed in seeds[1:]:
        assert seed.config["cute_register_tile_warp_fold"] is True
        assert seed.config["cute_register_tile_store_first"] is True
    # The loader plants the alternates, which makes both knobs searchable.
    planted = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_register_tile_warp_fold") is True
    ]
    assert len(planted) == 2
    fields = spec._flat_fields()
    for key in KNOBS:
        assert fields[key].search_choices == (False, True)
    # The autotune-off default is still the knob-free primary.
    default = spec.default_config()
    assert default.config.get("cute_register_tile_warp_fold") in (None, False)
    assert WARP_FOLD not in bound.to_code(default)


# ---- two reduction levels: the delta-rule correction step -------------------

CORRECTION_OWNER = "for synthetic_lane_2 in cutlass.range_constexpr("


def _bind_correction(*, diagonal: bool = True) -> tuple[Any, tuple[torch.Tensor, ...]]:
    """Bind the delta-rule step on CPU (``alpha`` a ``[BH, D]`` row by
    default, as the kda decode step calls it)."""
    kernel = helion.kernel(
        recurrent_step_correction_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    arguments = (
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, DV), dtype=torch.float32),
        torch.empty((BH, D, DV), dtype=torch.float32),
        torch.empty((BH, D) if diagonal else (BH,), dtype=torch.float32),
        torch.empty((BH,), dtype=torch.float32),
    )
    return _cpu_bind(kernel, arguments), arguments


def _combines(statements: list[str]) -> list[int]:
    return [i for i, source in enumerate(statements) if COLUMN_REDUCE in source]


def test_correction_step_knob_free_schedule_stores_behind_the_last_combine() -> None:
    # ``kts = sum(k * s)`` feeds the state that ``o = sum(q * s)`` reduces:
    # the first nest loads the tile once and accumulates ``kts``; behind its
    # combine the second nest applies the correction (stashing the corrected
    # tile) and accumulates ``o``; the state store nest reads that stash
    # behind the ``o`` combine; the output tail comes last.
    bound, _arguments = _bind_correction()
    code = bound.to_code(_config(bound, dv_threads=8, reduction_threads=16))
    assert code.count("cute.arch.load(state.iterator") == 1
    assert code.count(STATE_STORE) == 1
    assert WARP_FOLD not in code
    statements = _kernel_statements(code)
    first, second, consume = _owner_loops(statements)
    kts_combine, o_combine = _combines(statements)
    out_store = _first_statement(statements, "_cute_store_u32_vec(out.iterator")
    assert first < kts_combine < second < o_combine < consume < out_store
    assert "cute.arch.load(" in statements[first]
    assert "cute.arch.load(" not in statements[second]
    assert "kts_lane_results[" in statements[second]
    assert "_lane_stash.append(" in statements[second]
    assert STATE_STORE in statements[consume]
    assert "_lane_stash[" in statements[consume]
    assert "kts" not in statements[consume]
    assert "o_vec_lane_results[" in statements[out_store]


def test_correction_step_store_first_joins_the_second_accumulate_nest() -> None:
    # ``store_first``: the state store needs ``kts`` but not ``o``, so it
    # joins the level-1 nest ahead of the ``o`` combine and the corrected tile
    # is never stashed; the warp fold applies to both combines.
    bound, _arguments = _bind_correction()
    code = bound.to_code(
        _config(
            bound,
            dv_threads=8,
            reduction_threads=16,
            cute_register_tile_store_first=True,
        )
    )
    statements = _kernel_statements(code)
    first, second = _owner_loops(statements)
    kts_combine, o_combine = _combines(statements)
    assert first < kts_combine < second < o_combine
    assert STATE_STORE in statements[second]
    assert "kts_lane_results[" in statements[second]
    assert "_lane_stash.append(" not in statements[second]
    folded = bound.to_code(_config(bound, dv_threads=8, reduction_threads=16, **KNOBS))
    assert folded.count(WARP_FOLD) == 2
    assert folded.replace(f", {WARP_FOLD})", ")") == code


def test_correction_step_default_is_the_two_level_register_tile() -> None:
    # The chained body seeds half the lanes per thread (16 reduction threads
    # over 128 rows) and that seed is the autotune-off default.
    bound, _arguments = _bind_correction()
    code = bound.to_code(bound.config_spec.default_config())
    assert "block=(8, 16, 1)" in code
    assert code.count(CORRECTION_OWNER) == 3
    assert code.count(COLUMN_REDUCE) == 2


def test_register_budget_counts_distinct_values() -> None:
    # Sixteen lanes x 4 columns per thread: the decayed tile and ``beta * k
    # * v`` are held once per element (64 each), ``beta * k`` and ``q`` once
    # per lane (16 each; the unrolled duplicates merge in the compiler), so
    # the tile fits the budget that a per-lane-per-element count (256) would
    # have exceeded, and the eight-thread geometry forms the register tile.
    bound, _arguments = _bind_correction()
    code = bound.to_code(_config(bound, dv_threads=8, reduction_threads=8, **KNOBS))
    assert "_lane_stash" in code
    assert "block=(8, 8, 1)" in code
    assert code.count(CORRECTION_OWNER) == 2


def test_rolled_fallback_passes_through_a_fragment_capacity_decline() -> None:
    # Once the computed-fragment owner has declined a root for capacity, the
    # generic lowering's refusals report the overflow as InvalidConfig.  The
    # register tile's regeneration signal is not a refusal: it must still
    # reach ``generate_ast``'s rolled retry unchanged.
    bound, _arguments = _bind()
    codegen = cast(
        "GenerateAST",
        SimpleNamespace(
            device_function=SimpleNamespace(
                cute_state=SimpleNamespace(fragment_capacity_decline=(262144, 232448))
            )
        ),
    )
    failure = RegisterTileUnsupported("a load inside a post-combine value")
    refusal = helion.exc.BackendUnsupported("cute", "lane restores")
    with bound.env:
        assert GenerateAST.fragment_capacity_failure(codegen, failure) is failure
        converted = GenerateAST.fragment_capacity_failure(codegen, refusal)
    assert isinstance(converted, helion.exc.InvalidConfig)
    assert "262144" in str(converted) and "232448" in str(converted)
