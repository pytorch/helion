"""Generic coverage for CuTe persistent subwarp reductions."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import sympy
import torch

import helion
from helion._compiler import tile_strategy
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.autotuner_heuristics.cute import (
    CuteAsyncPersistentSubwarpRowsHeuristic,
)
from helion._compiler.autotuner_heuristics.cute import CuteAsyncStateLoadHeuristic
from helion._compiler.autotuner_heuristics.cute import (
    CutePersistentSubwarpRowsHeuristic,
)
from helion._compiler.backend import CuteBackend
from helion._compiler.cute.memory_ops import _persistent_vec_scope_safe
from helion._compiler.device_function import _exact_thread_block_dims
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import LoopOrderSpec
from helion.autotuner.config_spec import NumThreadsSpec
from helion.autotuner.config_spec import ReductionLoopSpec
import helion.language as hl


def test_lane_split_without_marker_skips_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = ast.parse("if flag:\n    value = source + 1").body
    monkeypatch.setattr(
        tile_strategy,
        "_update_scalar_definitions",
        lambda *_: pytest.fail("unexpected provenance scan"),
    )
    assert tile_strategy.split_lane_loop_reductions(body) is body


def test_chunk_hoist_without_running_sums_skips_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = ast.parse("value = source + 1").body
    monkeypatch.setattr(
        tile_strategy,
        "_collect_statement_provenance",
        lambda *_: pytest.fail("unexpected provenance scan"),
    )
    assert (
        tile_strategy.hoist_lane_invariant_chunk_recurrence(body, running_sums=set())
        is body
    )


@helion.kernel(backend="cute", static_shapes=True)
def _persistent_state_update(
    state: torch.Tensor,
    packed_key_query: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = state.shape
    out = torch.empty([rows], dtype=state.dtype, device=state.device)
    next_state = torch.empty_like(state)
    for tile_rows in hl.tile(rows):
        cols = hl.arange(width)
        key_tile = packed_key_query[cols].float()
        key_tile = key_tile * torch.rsqrt((key_tile * key_tile).sum() + 1e-6)
        state_tile = state[tile_rows, cols].float()
        residual = value[tile_rows].float() - (state_tile * key_tile[None, :]).sum(-1)
        updated = state_tile + residual[:, None] * key_tile[None, :]
        query_tile = packed_key_query[width + cols].float()
        out[tile_rows] = (updated * query_tile[None, :]).sum(-1).to(state.dtype)
        next_state[tile_rows, cols] = updated
    return out, next_state


@helion.kernel(backend="cute", static_shapes=False)
def _persistent_dynamic_grid_update(
    state: torch.Tensor,
    packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = state.size(0)
    heads = hl.specialize(state.size(1))
    rows = hl.specialize(state.size(2))
    width = hl.specialize(state.size(3))
    out = torch.empty([batch, heads, rows], dtype=state.dtype, device=state.device)
    next_state = torch.empty_like(state)
    for tile_batch, tile_head, tile_row in hl.tile(
        [batch, heads, rows], block_size=[1, 1, None]
    ):
        cols = hl.arange(width)
        batch_index = tile_batch.id
        head_index = tile_head.id
        packed_values = packed[
            batch_index,
            head_index * width + cols,
        ].float()
        state_values = state[batch_index, head_index, tile_row, cols].float()
        updated = state_values + packed_values
        out[batch_index, head_index, tile_row] = updated.sum(-1).to(state.dtype)
        next_state[batch_index, head_index, tile_row, cols] = updated
    return out, next_state


@helion.kernel(backend="cute", static_shapes=False)
def _persistent_3d_inplace_state_update(
    state_indices: torch.Tensor,
    state: torch.Tensor,
    packed: torch.Tensor,
) -> torch.Tensor:
    batches = hl.specialize(state_indices.size(0))
    heads = hl.specialize(state.size(1))
    rows = hl.specialize(state.size(2))
    width = hl.specialize(state.size(3))
    out = torch.empty([batches, heads, rows], dtype=state.dtype, device=state.device)
    for tile_batch, tile_head, tile_row in hl.tile(
        [batches, heads, rows], block_size=[1, 1, None]
    ):
        batch_index = tile_batch.id
        head_index = tile_head.id
        state_index = state_indices[batch_index]
        cols = hl.arange(width)
        state_values = state[state_index, head_index, tile_row, cols].float()
        packed_values = packed[batch_index, head_index, cols].float()
        updated = state_values + packed_values[None, :]
        out[batch_index, head_index, tile_row] = updated.sum(-1).to(state.dtype)
        state[state_index, head_index, tile_row, cols] = updated
    return out


def _bind_composed_seed_case() -> tuple[Any, Any]:
    args = (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.empty(3, 3, 128, 128, dtype=torch.bfloat16),
        torch.empty(2, 3, 128, dtype=torch.bfloat16),
    )
    bound = _persistent_3d_inplace_state_update.bind(args)
    host_function = bound.host_function
    assert host_function is not None
    bound.env.config_spec.target_device_capability = (8, 0)
    facts = CuteAsyncStateLoadHeuristic.register_facts(
        bound.env, host_function.device_ir
    )
    bound.env.compiler_fact_specialization_facts |= facts
    return bound, host_function


@helion.kernel(backend="cute", static_shapes=False)
def _persistent_indexed_state_update(
    state_indices: torch.Tensor,
    state: torch.Tensor,
    packed: torch.Tensor,
) -> torch.Tensor:
    batch = state_indices.size(0)
    rows = hl.specialize(state.size(1))
    width = hl.specialize(state.size(2))
    out = torch.empty([batch, rows], dtype=state.dtype, device=state.device)
    for tile_batch, tile_row in hl.tile([batch, rows], block_size=[1, None]):
        batch_index = tile_batch.id
        state_index = state_indices[batch_index]
        if state_index < 0:
            out[batch_index, tile_row] = 0
        else:
            cols = hl.arange(width)
            packed_values = packed[batch_index, cols].float()
            packed_values = packed_values * torch.rsqrt(
                (packed_values * packed_values).sum() + 1e-6
            )
            state_values = state[state_index, tile_row, cols].float()
            updated = state_values + packed_values[None, :]
            out[batch_index, tile_row] = (
                (updated * packed_values[None, :]).sum(-1).to(state.dtype)
            )
    return out


@helion.kernel(backend="cute", static_shapes=False)
def _persistent_indexed_inplace_state_update(
    state_indices: torch.Tensor,
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    batch = state_indices.size(0)
    rows = hl.specialize(state.size(1))
    width = hl.specialize(state.size(2))
    for tile_batch, tile_row in hl.tile([batch, rows], block_size=[1, None]):
        batch_index = tile_batch.id
        state_index = state_indices[batch_index]
        if state_index < 0:
            out[batch_index, tile_row] = 0
        else:
            cols = hl.arange(width)
            key_values = key[batch_index, cols].float()
            key_values = key_values * torch.rsqrt(
                (key_values * key_values).sum() + 1e-6
            )
            state_values = state[state_index, tile_row, cols].float()
            residual = value[batch_index, tile_row].float() - (
                state_values * key_values[None, :]
            ).sum(-1)
            updated = state_values + residual[:, None] * key_values[None, :]
            out[batch_index, tile_row] = (
                (updated * key_values[None, :]).sum(-1).to(state.dtype)
            )
            state[state_index, tile_row, cols] = updated
    return out


@helion.kernel(backend="cute", static_shapes=False)
def _persistent_shifted_state_update(
    state_indices: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    batch = state_indices.size(0)
    rows = hl.specialize(state.size(1))
    width = hl.specialize(state.size(2))
    out = torch.empty([batch, rows], dtype=state.dtype, device=state.device)
    for tile_batch, tile_row in hl.tile([batch, rows], block_size=[1, None]):
        batch_index = tile_batch.id
        state_index = state_indices[batch_index]
        if state_index < 0:
            out[batch_index, tile_row] = 0
        else:
            cols = hl.arange(width)
            shifted_cols = cols + 1
            state_values = state[state_index, tile_row, shifted_cols].float()
            out[batch_index, tile_row] = state_values.sum(-1).to(state.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _persistent_shifted_direct_update(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = state.shape
    out = torch.empty([rows], dtype=state.dtype, device=state.device)
    next_state = torch.empty_like(state)
    for tile_rows in hl.tile(rows):
        cols = hl.arange(width)
        shifted_cols = cols + 1
        values = state[tile_rows, shifted_cols].float()
        out[tile_rows] = values.sum(-1).to(state.dtype)
        next_state[tile_rows, shifted_cols] = values + 1
    return out, next_state


@helion.kernel(backend="cute", static_shapes=True)
def _persistent_store_only(destination: torch.Tensor) -> torch.Tensor:
    rows, width = destination.shape
    out = torch.empty([rows], dtype=torch.float32, device=destination.device)
    for tile_rows in hl.tile(rows):
        cols = hl.arange(width)
        values = cols.to(torch.float32)
        out[tile_rows] = values.sum(-1)
        destination[tile_rows, cols] = values[None, :].to(destination.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _alias_sensitive_persistent_sweeps(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    rows, width = x.shape
    out = torch.empty([rows], dtype=torch.float32, device=x.device)
    for tile_rows in hl.tile(rows):
        cols = hl.arange(width)
        first = x[tile_rows, cols].float().sum(-1)
        y[tile_rows, cols] = first[:, None]
        second = (x[tile_rows, cols].float() + first[:, None]).sum(-1)
        out[tile_rows] = first + second
    return out


@onlyBackends(["cute"])
def test_seed_uses_block_ids_with_permuted_specs() -> None:
    args = (
        torch.randn(16, 128, dtype=torch.bfloat16),
        torch.randn(256, dtype=torch.bfloat16),
        torch.randn(16, dtype=torch.bfloat16),
    )
    bound = _persistent_state_update.bind(args)
    env = bound.env
    reduction_block = next(
        block_id for block_id, block in enumerate(env.block_sizes) if block.reduction
    )
    row_block = env.config_spec.block_sizes.valid_block_ids()[0]
    assert reduction_block not in env.config_spec.reduction_loops.valid_block_ids()
    assert reduction_block in env.config_spec.reduction_block_ids
    for sequence in (
        env.config_spec.num_threads,
        env.config_spec.cute_vector_widths,
        env.config_spec.cute_lane_layouts,
        env.config_spec.cute_reduction_reloads,
    ):
        assert reduction_block in sequence.valid_block_ids()

    # Exercise the helper against deliberately different sequence orders;
    # positional list literals would silently assign one block's setting to
    # the other.
    env.config_spec.num_threads.reverse()
    env.config_spec.cute_vector_widths.reverse()
    env.config_spec.cute_lane_layouts.reverse()
    host_function = bound.host_function
    assert host_function is not None
    seed = CutePersistentSubwarpRowsHeuristic.get_seed_config(
        env, host_function.device_ir
    )
    assert seed is not None

    def by_block(spec: Any, raw_values: object) -> dict[int, object]:
        assert isinstance(raw_values, list)
        return dict(
            zip(
                spec.valid_block_ids(),
                raw_values,
                strict=True,
            )
        )

    assert (
        by_block(env.config_spec.block_sizes, seed.config["block_sizes"])[row_block]
        == 16
    )
    threads = by_block(env.config_spec.num_threads, seed.config["num_threads"])
    assert threads[row_block] == 8
    assert threads[reduction_block] == 16
    assert (
        by_block(
            env.config_spec.cute_vector_widths,
            seed.config["cute_vector_widths"],
        )[reduction_block]
        == 8
    )
    assert (
        by_block(
            env.config_spec.cute_reduction_reloads,
            seed.config["cute_reduction_reloads"],
        )[reduction_block]
        == "register"
    )
    assert "reduction_loops" not in seed.config
    assert not CutePersistentSubwarpRowsHeuristic.promote_seed_to_default

    # The layout is semantically inactive for a persistent one-fragment
    # reduction.  Normalize both choices to one autotuner identity.
    raw_layouts = seed.config["cute_lane_layouts"]
    assert isinstance(raw_layouts, list)
    strided_values = cast("list[str]", list(raw_layouts))
    strided_values[
        env.config_spec.cute_lane_layouts.block_id_to_index(reduction_block)
    ] = "strided"
    strided = helion.Config.from_dict(
        {**seed.config, "cute_lane_layouts": strided_values}
    )
    assert env.config_spec.normalized_config(
        strided
    ) == env.config_spec.normalized_config(seed)
    config_generation = env.config_spec.create_config_generation()
    blocked_flat, _ = config_generation.canonicalize_flat(
        config_generation.flatten(seed)
    )
    strided_flat, _ = config_generation.canonicalize_flat(
        config_generation.flatten(strided)
    )
    assert strided_flat == blocked_flat

    code = bound.to_code(seed)
    assert "block=(16, 8, 1)" in code
    assert "threads_in_group=16" in code
    assert code.count("ir.VectorType.get([8], cutlass.Uint16.mlir_type)") >= 3
    assert any(
        " + 128" in line and "cute.arch.load" in line for line in code.splitlines()
    )
    assert "_cute_store_u16_vec" in code
    assert "_cute_grouped_reduce_shared_two_stage" not in code


@onlyBackends(["cute"])
def test_persistent_subwarp_seed_includes_full_row_alternate() -> None:
    args = (
        torch.randn(1, 12, 128, 128, dtype=torch.bfloat16),
        torch.randn(1, 12 * 128, dtype=torch.bfloat16),
    )
    bound = _persistent_dynamic_grid_update.bind(args)
    host_function = bound.host_function
    assert host_function is not None
    seeds = CutePersistentSubwarpRowsHeuristic.get_seed_configs(
        bound.env, host_function.device_ir
    )
    assert seeds is not None

    row_block = bound.env.config_spec.block_sizes.valid_block_ids()[0]
    row_index = bound.env.config_spec.block_sizes.block_id_to_index(row_block)
    assert [seed.config["block_sizes"][row_index] for seed in seeds] == [
        16,
        16,
        16,
        32,
        128,
    ]
    row_thread_index = bound.env.config_spec.num_threads.block_id_to_index(row_block)
    recurrent_style = seeds[1]
    assert recurrent_style.config["num_threads"][row_thread_index] == 1
    assert recurrent_style.config["num_warps"] == 1
    row_one = seeds[2]
    assert row_one.config["num_threads"][row_thread_index] == 16
    assert row_one.config["num_warps"] == 8
    one_warp = seeds[3]
    assert one_warp.config["num_threads"][row_thread_index] == 2
    assert one_warp.config["num_warps"] == 1
    # Rank zero remains the established schedule, so no-autotune behavior is
    # unchanged until the full-row alternate wins an actual benchmark.
    assert seeds[0] == CutePersistentSubwarpRowsHeuristic.get_seed_config(
        bound.env, host_function.device_ir
    )


@onlyBackends(["cute"])
def test_persistent_subwarp_seed_includes_recurrent_style_alternate() -> None:
    args = (
        torch.randn(1, 12, 128, 128, dtype=torch.bfloat16),
        torch.randn(1, 12 * 128, dtype=torch.bfloat16),
    )
    bound = _persistent_dynamic_grid_update.bind(args)
    host_function = bound.host_function
    assert host_function is not None
    seeds = CutePersistentSubwarpRowsHeuristic.get_seed_configs(
        bound.env, host_function.device_ir
    )
    assert seeds is not None

    row_block = bound.env.config_spec.block_sizes.valid_block_ids()[0]
    reduction_block = next(
        block_id
        for block_id, block in enumerate(bound.env.block_sizes)
        if block.reduction
    )

    def by_block(spec: Any, raw_values: object) -> dict[int, object]:
        assert isinstance(raw_values, list)
        return dict(
            zip(
                spec.valid_block_ids(),
                raw_values,
                strict=True,
            )
        )

    recurrent = [
        seed
        for seed in seeds
        if by_block(bound.env.config_spec.block_sizes, seed.config["block_sizes"])[
            row_block
        ]
        == 16
        and by_block(bound.env.config_spec.num_threads, seed.config["num_threads"])[
            row_block
        ]
        == 1
        and by_block(bound.env.config_spec.num_threads, seed.config["num_threads"])[
            reduction_block
        ]
        == 32
    ]
    assert len(recurrent) == 1
    seed = recurrent[0]
    assert (
        by_block(
            bound.env.config_spec.cute_vector_widths,
            seed.config["cute_vector_widths"],
        )[reduction_block]
        == 4
    )
    assert seed.config["num_warps"] == 1
    assert seed.config["num_stages"] == 1
    assert seed.config["pid_type"] == "flat"
    assert (
        by_block(
            bound.env.config_spec.cute_reduction_reloads,
            seed.config["cute_reduction_reloads"],
        )[reduction_block]
        == "register"
    )
    assert seed != CutePersistentSubwarpRowsHeuristic.get_seed_config(
        bound.env, host_function.device_ir
    )


@onlyBackends(["cute"])
def test_persistent_subwarp_recurrent_style_seed_requires_k128() -> None:
    args = (
        torch.randn(1, 12, 128, 256, dtype=torch.bfloat16),
        torch.randn(1, 12 * 256, dtype=torch.bfloat16),
    )
    bound = _persistent_dynamic_grid_update.bind(args)
    host_function = bound.host_function
    assert host_function is not None
    seeds = CutePersistentSubwarpRowsHeuristic.get_seed_configs(
        bound.env, host_function.device_ir
    )
    assert seeds is not None

    row_block = bound.env.config_spec.block_sizes.valid_block_ids()[0]
    reduction_block = next(
        block_id
        for block_id, block in enumerate(bound.env.block_sizes)
        if block.reduction
    )

    def by_block(spec: Any, raw_values: object) -> dict[int, object]:
        assert isinstance(raw_values, list)
        return dict(
            zip(
                spec.valid_block_ids(),
                raw_values,
                strict=True,
            )
        )

    assert not any(
        by_block(bound.env.config_spec.block_sizes, seed.config["block_sizes"])[
            row_block
        ]
        == 16
        and by_block(bound.env.config_spec.num_threads, seed.config["num_threads"])[
            row_block
        ]
        == 1
        and by_block(bound.env.config_spec.num_threads, seed.config["num_threads"])[
            reduction_block
        ]
        == 32
        and by_block(
            bound.env.config_spec.cute_vector_widths,
            seed.config["cute_vector_widths"],
        )[reduction_block]
        == 4
        for seed in seeds
    )


@onlyBackends(["cute"])
def test_composed_async_full_row_seed_survives_flatten() -> None:
    bound, host_function = _bind_composed_seed_case()
    spec = bound.env.config_spec

    # The early grid-size check cannot look up the two fixed-size grid axes,
    # so xyz starts conservatively disabled for this otherwise-safe 3-D grid.
    assert "xyz" not in spec.allowed_pid_types
    seeds = compiler_seed_configs(bound.env, host_function.device_ir)
    spec.compiler_seed_configs = seeds
    composed = [
        seed
        for seed in seeds
        if seed.config.get("pid_type") == "xyz"
        and seed.config.get("cute_async_load_stages") == 5
    ]
    assert len(composed) == 1
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts

    seed = composed[0]
    assert seed.config["block_sizes"] == [128]
    assert seed.config["num_threads"] == [8, 16]
    assert seed.config["loop_orders"] == [[1, 0, 2]]
    assert seed.config["cute_vector_widths"] == [8, 1, 1, 1]
    assert seed.config["cute_lane_layouts"] == ["blocked"] * 4
    assert seed.config["cute_reduction_reloads"] == ["register"]
    assert seed.config["num_warps"] == 4
    assert seed.config["num_stages"] == 1
    assert seed.config["cute_async_load_lookahead"] == 4
    assert seed.config["cute_async_load_group_rows"] == 2
    assert seed.config["cute_async_load_cache"] == "cg"
    assert seed.config["cute_async_store_policy"] == "default"
    assert seed.config["cute_bf16x2_recurrence"] is True
    assert seed.config["cute_proven_bounds"] is False

    config_generation = spec.create_config_generation()
    normalized = [
        config
        for _flat, config in config_generation.seed_flat_config_pairs()
        if config.config.get("pid_type") == "xyz"
    ]
    assert len(normalized) == 1
    normalized_seed = normalized[0]
    assert normalized_seed.config["block_sizes"] == [128]
    assert normalized_seed.config["num_threads"] == [8, 16]
    assert normalized_seed.config["loop_orders"] == [[1, 0, 2]]
    assert normalized_seed.config["cute_vector_widths"] == [8, 1, 1, 1]
    assert normalized_seed.config["cute_lane_layouts"] == ["blocked"] * 4
    assert normalized_seed.config["cute_reduction_reloads"] == ["register"]
    assert normalized_seed.config["cute_async_load_stages"] == 5
    assert normalized_seed.config["cute_async_load_lookahead"] == 4
    assert normalized_seed.config["cute_async_load_group_rows"] == 2
    assert normalized_seed.config["cute_async_load_cache"] == "cg"
    assert normalized_seed.config["cute_async_store_policy"] == "default"
    assert normalized_seed.config["cute_bf16x2_recurrence"] is True
    assert normalized_seed.config["cute_proven_bounds"] is False
    # CuTe derives the actual block dimensions from num_threads. num_warps is
    # intentionally not another search coordinate, and its implicit value is
    # the same four warps carried by the source seed.
    assert normalized_seed.num_warps == 4
    assert normalized_seed.num_stages == 1

    pid_index = config_generation._key_to_flat_indices["pid_type"][0][0]
    pid_fragment = config_generation.flat_spec[pid_index]
    assert tuple(cast("Any", pid_fragment).choices) == ("flat", "xyz")
    assert tuple(cast("Any", pid_fragment).search_choices) == ("flat",)
    assert "xyz" not in spec.allowed_pid_types


@onlyBackends(["cute"])
def test_composed_async_seed_uses_l2_store_only_when_supported() -> None:
    bound, host_function = _bind_composed_seed_case()

    with patch(
        "helion._compiler.cute.cutedsl_compat.fixed_l2_evict_last_store_policy_supported",
        return_value=True,
    ):
        seed = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )

    assert seed is not None
    assert seed.config["cute_async_store_policy"] == "l2_evict_last"


@onlyBackends(["cute"])
def test_async_pipeline_skips_rewrite_without_exact_thread_dims() -> None:
    args = (
        torch.tensor([0], dtype=torch.int32),
        torch.randn(2, 128, 128, dtype=torch.bfloat16),
        torch.randn(1, 128, dtype=torch.bfloat16),
        torch.randn(1, 128, dtype=torch.bfloat16),
        torch.empty(1, 128, dtype=torch.bfloat16),
    )
    bound = _persistent_indexed_inplace_state_update._bind_isolated(args)
    host_function = bound.host_function
    assert host_function is not None
    bound.env.config_spec.target_device_capability = (8, 0)
    facts = CuteAsyncStateLoadHeuristic.register_facts(
        bound.env, host_function.device_ir
    )
    bound.env.compiler_fact_specialization_facts |= facts
    config = CuteAsyncStateLoadHeuristic.get_seed_config(
        bound.env, host_function.device_ir
    )
    assert config is not None
    assert config.config["cute_async_load_stages"] > 0
    non_static_strategy = cast(
        "Any",
        SimpleNamespace(thread_block_dims=lambda: (sympy.Symbol("threads"), 1, 1)),
    )
    assert _exact_thread_block_dims(non_static_strategy) is None

    with (
        patch(
            "helion._compiler.device_function._exact_thread_block_dims",
            side_effect=lambda _strategy: _exact_thread_block_dims(non_static_strategy),
        ),
        patch(
            "helion._compiler.cute.pipeline_state_loads.pipeline_state_loads"
        ) as pipeline,
    ):
        code = bound.to_code(config)

    pipeline.assert_not_called()
    assert "cute.arch.cp_async_shared_global(" not in code
    assert "_async_state_" not in code


@onlyBackends(["cute"])
def test_composed_seed_keeps_flat_pid_without_exact_grid_extent() -> None:
    bound, host_function = _bind_composed_seed_case()
    spec = bound.env.config_spec
    root = spec.kernel_grid_fact
    assert root is not None
    leading_block_id = root.roots[0].block_ids[0]
    original_size = bound.env.block_sizes[leading_block_id].size
    bound.env.block_sizes[leading_block_id].size = None
    try:
        seed = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )
    finally:
        bound.env.block_sizes[leading_block_id].size = original_size

    assert seed is not None
    assert seed.config["pid_type"] == "flat"
    assert seed.config["block_sizes"] == [128]
    assert seed.config["loop_orders"] == [[1, 0, 2]]
    assert seed.config["cute_async_load_stages"] == 5


@pytest.mark.parametrize("axis_index", (0, 1, 2))
@onlyBackends(["cute"])
def test_composed_seed_keeps_flat_pid_for_oversized_grid_axis(
    axis_index: int,
) -> None:
    bound, host_function = _bind_composed_seed_case()
    spec = bound.env.config_spec
    grid_fact = spec.kernel_grid_fact
    assert grid_fact is not None
    block_id = grid_fact.roots[0].block_ids[axis_index]
    original_size = bound.env.block_sizes[block_id].size
    bound.env.block_sizes[block_id].size = 65_536
    try:
        seed = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )
    finally:
        bound.env.block_sizes[block_id].size = original_size

    if axis_index == 2:
        # The innermost row axis also ceases to be a full-row schedule.
        assert seed is None
    else:
        assert seed is not None
        assert seed.config["pid_type"] == "flat"


@onlyBackends(["cute"])
def test_composed_seed_keeps_flat_pid_without_metadata_specialization() -> None:
    bound, host_function = _bind_composed_seed_case()
    original_facts = bound.env.compiler_fact_specialization_facts
    bound.env.compiler_fact_specialization_facts = frozenset()
    try:
        seed = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )
    finally:
        bound.env.compiler_fact_specialization_facts = original_facts

    assert seed is not None
    assert seed.config["pid_type"] == "flat"


@onlyBackends(["cute"])
def test_composed_seed_does_not_override_an_unrelated_xyz_restriction() -> None:
    bound, host_function = _bind_composed_seed_case()
    spec = bound.env.config_spec
    original_reason = spec.disallowed_pid_type_reasons["xyz"]
    spec.disallowed_pid_type_reasons["xyz"] = "synthetic unrelated safety rule"
    try:
        seed = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )
    finally:
        spec.disallowed_pid_type_reasons["xyz"] = original_reason

    assert seed is not None
    assert seed.config["pid_type"] == "flat"


@onlyBackends(["cute"])
def test_composed_seed_rejects_non_3d_grid() -> None:
    bound, host_function = _bind_composed_seed_case()
    spec = bound.env.config_spec
    grid_fact = spec.kernel_grid_fact
    assert grid_fact is not None
    root = grid_fact.roots[0]
    spec.kernel_grid_fact = grid_fact._replace(
        roots=(root._replace(block_ids=root.block_ids[:2]),)
    )
    try:
        seed = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )
    finally:
        spec.kernel_grid_fact = grid_fact

    assert seed is None


@onlyBackends(["cute"])
def test_composed_seed_requires_full_innermost_row_tile() -> None:
    bound, host_function = _bind_composed_seed_case()
    spec = bound.env.config_spec
    grid_fact = spec.kernel_grid_fact
    assert grid_fact is not None
    root = grid_fact.roots[0]
    row_spec = spec.block_sizes[0]

    spec.kernel_grid_fact = grid_fact._replace(
        roots=(
            root._replace(
                block_ids=(root.block_ids[-1], *root.block_ids[:-1]),
            ),
        )
    )
    try:
        not_innermost = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )
    finally:
        spec.kernel_grid_fact = grid_fact
    assert not_innermost is None

    original_max_size = row_spec.max_size
    row_spec.max_size = 64
    try:
        not_full = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )
    finally:
        row_spec.max_size = original_max_size
    assert not_full is None

    row_block_id = row_spec.block_id
    original_row_size = bound.env.block_sizes[row_block_id].size
    bound.env.block_sizes[row_block_id].size = 256
    try:
        not_full_extent = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )
    finally:
        bound.env.block_sizes[row_block_id].size = original_row_size
    assert not_full_extent is None


@onlyBackends(["cute"])
def test_composed_seed_requires_async_pipeline() -> None:
    bound, host_function = _bind_composed_seed_case()
    spec = bound.env.config_spec
    original_enabled = spec.cute_async_load_pipeline_enabled
    spec.cute_async_load_pipeline_enabled = False
    try:
        seed = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )
    finally:
        spec.cute_async_load_pipeline_enabled = original_enabled

    assert seed is None


@onlyBackends(["cute"])
def test_composed_seed_addresses_loop_order_by_block_id() -> None:
    bound, host_function = _bind_composed_seed_case()
    spec = bound.env.config_spec
    grid_fact = spec.kernel_grid_fact
    assert grid_fact is not None
    root = grid_fact.roots[0]
    reduction_block_id = next(
        block.block_id for block in bound.env.block_sizes if block.reduction
    )
    spec.loop_orders.insert(0, LoopOrderSpec([reduction_block_id]))
    try:
        order_block_ids = [item.block_id for item in spec.loop_orders]
        seed = CuteAsyncPersistentSubwarpRowsHeuristic.get_seed_config(
            bound.env, host_function.device_ir
        )
    finally:
        del spec.loop_orders[0]

    assert seed is not None
    orders = cast("list[list[int]]", seed.config["loop_orders"])
    by_block_id = dict(
        zip(
            order_block_ids,
            orders,
            strict=True,
        )
    )
    assert by_block_id[reduction_block_id] == [0]
    assert by_block_id[root.block_ids[0]] == [1, 0, 2]


@onlyBackends(["cute"])
def test_full_row_seed_hoists_invariant_norm_before_state_rows() -> None:
    args = (
        torch.tensor([0], dtype=torch.int32),
        torch.randn(2, 128, 128, dtype=torch.bfloat16),
        torch.randn(1, 128, dtype=torch.bfloat16),
        torch.randn(1, 128, dtype=torch.bfloat16),
        torch.empty(1, 128, dtype=torch.bfloat16),
    )
    bound = _persistent_indexed_inplace_state_update.bind(args)
    host_function = bound.host_function
    assert host_function is not None
    seeds = CutePersistentSubwarpRowsHeuristic.get_seed_configs(
        bound.env, host_function.device_ir
    )
    assert seeds is not None and len(seeds) == 5

    code = bound.to_code(seeds[-1])
    lines = code.splitlines()
    active_row_loop = max(
        index
        for index, line in enumerate(lines)
        if "for lane_" in line and "range(16)" in line
    )
    norm_line = next(index for index, line in enumerate(lines) if "sum_1 =" in line)
    state_load_line = next(
        index
        for index, line in enumerate(lines)
        if "cute.arch.load(state.iterator" in line
    )
    state_store_line = next(
        index
        for index, line in enumerate(lines)
        if "_cute_store_u16_vec(state.iterator" in line
    )
    assert norm_line < active_row_loop < state_load_line < state_store_line


@onlyBackends(["cute"])
def test_tail_vector_load_uses_a_safe_outer_index() -> None:
    args = (
        torch.randn(17, 128, dtype=torch.bfloat16),
        torch.randn(256, dtype=torch.bfloat16),
        torch.randn(17, dtype=torch.bfloat16),
    )
    bound = _persistent_state_update.bind(args)
    host_function = bound.host_function
    assert host_function is not None
    seed = CutePersistentSubwarpRowsHeuristic.get_seed_config(
        bound.env, host_function.device_ir
    )
    assert seed is not None

    code = bound.to_code(seed)
    state_vector_load = next(
        line
        for line in code.splitlines()
        if "_persistent_branch_vec_" in line and "cute.arch.load(state.iterator" in line
    )
    assert "if mask_0" in state_vector_load
    assert "else cutlass.Int32(0)" in state_vector_load
    assert "_cute_store_u16_vec" in code


@onlyBackends(["cute"])
def test_dynamic_grid_vector_hoist_preserves_device_scope_aliases() -> None:
    args = (
        torch.randn(1, 12, 16, 128, dtype=torch.bfloat16),
        torch.randn(1, 12 * 128, dtype=torch.bfloat16),
    )
    bound = _persistent_dynamic_grid_update.bind(args)
    env = bound.env
    reduction_block = next(
        block_id for block_id, block in enumerate(env.block_sizes) if block.reduction
    )
    row_block = env.config_spec.block_sizes.valid_block_ids()[0]

    def values_for(spec: Any, values: dict[int, object], default: object):
        return [values.get(block_id, default) for block_id in spec.valid_block_ids()]

    config = helion.Config.from_dict(
        {
            "block_sizes": values_for(env.config_spec.block_sizes, {row_block: 16}, 16),
            "num_threads": values_for(
                env.config_spec.num_threads,
                {row_block: 8, reduction_block: 16},
                0,
            ),
            "cute_vector_widths": values_for(
                env.config_spec.cute_vector_widths,
                {reduction_block: 8},
                1,
            ),
            "cute_lane_layouts": values_for(
                env.config_spec.cute_lane_layouts,
                {reduction_block: "blocked"},
                "blocked",
            ),
            "cute_reduction_reloads": values_for(
                env.config_spec.cute_reduction_reloads,
                {reduction_block: "register"},
                "auto",
            ),
            "num_warps": 4,
            "num_stages": 1,
            "pid_type": "flat",
        }
    )
    code = bound.to_code(config)
    vector_loads = [
        line
        for line in code.splitlines()
        if "_persistent_branch_vec_" in line and "cute.arch.load" in line
    ]

    assert len(vector_loads) >= 2
    assert all(".size(" not in line for line in vector_loads)
    assert any("tile_offset_" in line or "indices_" in line for line in vector_loads)


@onlyBackends(["cute"])
def test_dynamic_state_index_uses_branch_local_exact_fragments() -> None:
    args = (
        torch.tensor([1], dtype=torch.int64),
        torch.randn(4, 16, 128, dtype=torch.bfloat16),
        torch.randn(1, 128, dtype=torch.bfloat16),
    )
    bound = _persistent_indexed_state_update.bind(args)
    env = bound.env
    reduction_block = next(
        block_id for block_id, block in enumerate(env.block_sizes) if block.reduction
    )
    row_block = env.config_spec.block_sizes.valid_block_ids()[0]

    def values_for(spec: Any, values: dict[int, object], default: object):
        return [values.get(block_id, default) for block_id in spec.valid_block_ids()]

    config = helion.Config.from_dict(
        {
            "block_sizes": values_for(env.config_spec.block_sizes, {row_block: 16}, 16),
            "num_threads": values_for(
                env.config_spec.num_threads,
                {row_block: 8, reduction_block: 16},
                0,
            ),
            "cute_vector_widths": values_for(
                env.config_spec.cute_vector_widths,
                {reduction_block: 8},
                1,
            ),
            "cute_lane_layouts": values_for(
                env.config_spec.cute_lane_layouts,
                {reduction_block: "blocked"},
                "blocked",
            ),
            "cute_reduction_reloads": values_for(
                env.config_spec.cute_reduction_reloads,
                {reduction_block: "register"},
                "auto",
            ),
            "num_warps": 4,
            "num_stages": 1,
            "pid_type": "flat",
        }
    )
    code = bound.to_code(config)

    state_index_load = code.index("state_indices.iterator")
    branch_vec_load = code.index("cute.arch.load(state.iterator")
    assert state_index_load < branch_vec_load
    assert code.count("cute.arch.load(state.iterator") == 1
    assert ".size(" not in code[branch_vec_load : code.index("\n", branch_vec_load)]
    assert not any(
        "state.iterator" in line and ".load()" in line for line in code.splitlines()
    )
    assert "_helion_persistent_branch_vec_" not in code


@onlyBackends(["cute"])
def test_dynamic_indexed_inplace_state_update_meta_compiles() -> None:
    """A decode-shaped exact state RMW is lane-local, despite sharing storage."""
    args = (
        torch.tensor([1, 2], dtype=torch.int64),
        torch.randn(4, 16, 128, dtype=torch.bfloat16),
        torch.randn(2, 128, dtype=torch.bfloat16),
        torch.randn(2, 16, dtype=torch.bfloat16),
        torch.empty(2, 16, dtype=torch.bfloat16),
    )
    bound = _persistent_indexed_inplace_state_update.bind(args)
    host_function = bound.host_function
    assert host_function is not None
    seed = CutePersistentSubwarpRowsHeuristic.get_seed_config(
        bound.env, host_function.device_ir
    )
    assert seed is not None

    code = bound.to_code(seed)

    assert code.count("cute.arch.load(state.iterator") >= 1
    assert "_cute_store_u16_vec(state.iterator" in code
    assert "_helion_persistent_branch_vec_" not in code


@onlyBackends(["cute"])
def test_dynamic_state_vector_alignment_is_cache_specialized() -> None:
    _persistent_indexed_state_update.reset()
    state_indices = torch.tensor([1], dtype=torch.int64)
    packed = torch.randn(1, 128, dtype=torch.bfloat16)
    aligned_state = torch.randn(4, 16, 128, dtype=torch.bfloat16)
    aligned = _persistent_indexed_state_update.bind(
        (state_indices, aligned_state, packed)
    )
    env = aligned.env
    reduction_block = next(
        block_id for block_id, block in enumerate(env.block_sizes) if block.reduction
    )
    row_block = env.config_spec.block_sizes.valid_block_ids()[0]

    def values_for(spec: Any, values: dict[int, object], default: object):
        return [values.get(block_id, default) for block_id in spec.valid_block_ids()]

    config = helion.Config.from_dict(
        {
            "block_sizes": values_for(env.config_spec.block_sizes, {row_block: 16}, 16),
            "num_threads": values_for(
                env.config_spec.num_threads,
                {row_block: 8, reduction_block: 16},
                0,
            ),
            "cute_vector_widths": values_for(
                env.config_spec.cute_vector_widths,
                {reduction_block: 8},
                1,
            ),
            "cute_lane_layouts": values_for(
                env.config_spec.cute_lane_layouts,
                {reduction_block: "blocked"},
                "blocked",
            ),
            "cute_reduction_reloads": values_for(
                env.config_spec.cute_reduction_reloads,
                {reduction_block: "register"},
                "auto",
            ),
            "num_warps": 4,
            "num_stages": 1,
            "pid_type": "flat",
        }
    )
    assert "cute.arch.load(state.iterator" in aligned.to_code(config)

    # Keep the same shape and logical strides but offset the input pointer by
    # one bf16.  This must receive a different bound-kernel specialization and
    # cannot use the 16-byte vector transaction emitted for ``aligned_state``.
    storage = torch.randn(4 * 16 * 128 + 1, dtype=torch.bfloat16)
    unaligned_state = torch.as_strided(
        storage,
        (4, 16, 128),
        (16 * 128, 128, 1),
        storage_offset=1,
    )
    unaligned = _persistent_indexed_state_update.bind(
        (state_indices, unaligned_state, packed)
    )
    assert unaligned is not aligned
    unaligned_code = unaligned.to_code(config)
    assert "cute.arch.load(state.iterator" not in unaligned_code


@onlyBackends(["cute"])
def test_shifted_state_fragment_stays_scalar_at_partial_boundary() -> None:
    args = (
        torch.tensor([1], dtype=torch.int64),
        torch.randn(4, 16, 128, dtype=torch.bfloat16),
    )
    bound = _persistent_shifted_state_update.bind(args)
    env = bound.env
    reduction_block = next(
        block_id for block_id, block in enumerate(env.block_sizes) if block.reduction
    )
    row_block = env.config_spec.block_sizes.valid_block_ids()[0]

    def values_for(spec: Any, values: dict[int, object], default: object):
        return [values.get(block_id, default) for block_id in spec.valid_block_ids()]

    config = helion.Config.from_dict(
        {
            "block_sizes": values_for(env.config_spec.block_sizes, {row_block: 16}, 16),
            "num_threads": values_for(
                env.config_spec.num_threads,
                {row_block: 8, reduction_block: 16},
                0,
            ),
            "cute_vector_widths": values_for(
                env.config_spec.cute_vector_widths,
                {reduction_block: 8},
                1,
            ),
            "cute_lane_layouts": values_for(
                env.config_spec.cute_lane_layouts,
                {reduction_block: "blocked"},
                "blocked",
            ),
            "cute_reduction_reloads": values_for(
                env.config_spec.cute_reduction_reloads,
                {reduction_block: "register"},
                "auto",
            ),
            "num_warps": 4,
            "num_stages": 1,
            "pid_type": "flat",
        }
    )
    code = bound.to_code(config)

    # The last logical fragment addresses columns 121..128, so a vector load
    # would read column 128 OOB.
    assert "cute.arch.load(state.iterator" not in code
    assert any(
        "state.iterator" in line and ".load()" in line for line in code.splitlines()
    )
    assert "_helion_persistent_branch_vec_" not in code


@onlyBackends(["cute"])
def test_normal_persistent_vectors_require_exact_aligned_fragments() -> None:
    shifted_args = (torch.randn(16, 128, dtype=torch.bfloat16),)
    shifted = _persistent_shifted_direct_update.bind(shifted_args)
    shifted_env = shifted.env
    shifted_reduction_block = next(
        block_id
        for block_id, block in enumerate(shifted_env.block_sizes)
        if block.reduction
    )
    shifted_row_block = shifted_env.config_spec.block_sizes.valid_block_ids()[0]

    def values_for(spec: Any, values: dict[int, object], default: object):
        return [values.get(block_id, default) for block_id in spec.valid_block_ids()]

    shifted_config = helion.Config.from_dict(
        {
            "block_sizes": values_for(
                shifted_env.config_spec.block_sizes,
                {shifted_row_block: 16},
                16,
            ),
            "num_threads": values_for(
                shifted_env.config_spec.num_threads,
                {shifted_row_block: 8, shifted_reduction_block: 16},
                0,
            ),
            "cute_vector_widths": values_for(
                shifted_env.config_spec.cute_vector_widths,
                {shifted_reduction_block: 8},
                1,
            ),
            "cute_lane_layouts": values_for(
                shifted_env.config_spec.cute_lane_layouts,
                {shifted_reduction_block: "blocked"},
                "blocked",
            ),
            "cute_reduction_reloads": values_for(
                shifted_env.config_spec.cute_reduction_reloads,
                {shifted_reduction_block: "register"},
                "auto",
            ),
            "num_warps": 4,
            "num_stages": 1,
            "pid_type": "flat",
        }
    )
    shifted_code = shifted.to_code(shifted_config)
    assert "cute.arch.load(state.iterator" not in shifted_code
    assert "_cute_store_u16_vec(next_state.iterator" not in shifted_code

    _persistent_state_update.reset()
    aligned_state = torch.randn(16, 128, dtype=torch.bfloat16)
    packed = torch.randn(256, dtype=torch.bfloat16)
    value = torch.randn(16, dtype=torch.bfloat16)
    aligned = _persistent_state_update.bind((aligned_state, packed, value))
    aligned_host = aligned.host_function
    assert aligned_host is not None
    seed = CutePersistentSubwarpRowsHeuristic.get_seed_config(
        aligned.env, aligned_host.device_ir
    )
    assert seed is not None
    aligned_code = aligned.to_code(seed)
    assert "cute.arch.load(state.iterator" in aligned_code
    assert "_cute_store_u16_vec(next_state.iterator" in aligned_code

    storage = torch.randn(16 * 128 + 1, dtype=torch.bfloat16)
    unaligned_state = torch.as_strided(
        storage,
        (16, 128),
        (128, 1),
        storage_offset=1,
    )
    unaligned = _persistent_state_update.bind((unaligned_state, packed, value))
    assert unaligned is not aligned
    unaligned_code = unaligned.to_code(seed)
    assert "cute.arch.load(state.iterator" not in unaligned_code


@onlyBackends(["cute"])
def test_normal_persistent_store_requires_runtime_alignment() -> None:
    aligned_destination = torch.empty(16, 128, dtype=torch.bfloat16)
    aligned = _persistent_store_only.bind((aligned_destination,))
    env = aligned.env
    reduction_block = next(
        block_id for block_id, block in enumerate(env.block_sizes) if block.reduction
    )
    row_block = env.config_spec.block_sizes.valid_block_ids()[0]

    def values_for(spec: Any, values: dict[int, object], default: object):
        return [values.get(block_id, default) for block_id in spec.valid_block_ids()]

    config = helion.Config.from_dict(
        {
            "block_sizes": values_for(env.config_spec.block_sizes, {row_block: 16}, 16),
            "num_threads": values_for(
                env.config_spec.num_threads,
                {row_block: 8, reduction_block: 16},
                0,
            ),
            "cute_vector_widths": values_for(
                env.config_spec.cute_vector_widths,
                {reduction_block: 8},
                1,
            ),
            "cute_lane_layouts": values_for(
                env.config_spec.cute_lane_layouts,
                {reduction_block: "blocked"},
                "blocked",
            ),
            "cute_reduction_reloads": values_for(
                env.config_spec.cute_reduction_reloads,
                {reduction_block: "register"},
                "auto",
            ),
            "num_warps": 4,
            "num_stages": 1,
            "pid_type": "flat",
        }
    )
    assert "_cute_store_u16_vec(destination.iterator" in aligned.to_code(config)

    storage = torch.empty(16 * 128 + 1, dtype=torch.bfloat16)
    unaligned_destination = torch.as_strided(
        storage,
        (16, 128),
        (128, 1),
        storage_offset=1,
    )
    unaligned = _persistent_store_only.bind((unaligned_destination,))
    assert unaligned is not aligned
    unaligned_code = unaligned.to_code(config)
    assert "_cute_store_u16_vec(destination.iterator" not in unaligned_code
    assert any(
        "destination.iterator" in line and ".store(" in line
        for line in unaligned_code.splitlines()
    )


@pytest.mark.parametrize("storage_relation", ("distinct", "overlap", "same"))
@onlyBackends(["cute"])
def test_external_tensor_runtime_alias_proof_is_cache_safe(
    monkeypatch: Any,
    storage_relation: str,
) -> None:
    from helion._compiler.device_function import DeviceFunction

    _alias_sensitive_persistent_sweeps.reset()
    if storage_relation == "overlap":
        storage = torch.randn(16 * 128 + 1, dtype=torch.float32)
        x = torch.as_strided(storage, (16, 128), (128, 1))
        y = torch.as_strided(storage, (16, 128), (128, 1), storage_offset=1)
    else:
        x = torch.randn(16, 128, dtype=torch.float32)
        y = x if storage_relation == "same" else torch.randn_like(x)
    captured_pairs: list[set[frozenset[str]]] = []
    original = DeviceFunction.proven_disjoint_tensor_pairs

    def capture(device_function: DeviceFunction) -> set[frozenset[str]]:
        result = original(device_function)
        captured_pairs.append(result)
        return result

    monkeypatch.setattr(DeviceFunction, "proven_disjoint_tensor_pairs", capture)
    bound = _alias_sensitive_persistent_sweeps.bind((x, y))
    host_function = bound.host_function
    assert host_function is not None
    seed = CutePersistentSubwarpRowsHeuristic.get_seed_config(
        bound.env, host_function.device_ir
    )
    assert seed is not None
    if storage_relation == "distinct":
        bound.to_code(seed)
    else:
        with pytest.raises(
            helion.exc.BackendUnsupported, match="potentially aliasing write"
        ):
            bound.to_code(seed)

    assert captured_pairs
    assert (frozenset(("x", "y")) in captured_pairs[-1]) is (
        storage_relation == "distinct"
    )
    assert frozenset(("x", "out")) in captured_pairs[-1]
    assert frozenset(("y", "out")) in captured_pairs[-1]


@onlyBackends(["cute"])
def test_runtime_alias_specialization_separates_overlapping_launches() -> None:
    _alias_sensitive_persistent_sweeps.reset()
    x = torch.randn(16, 128, dtype=torch.float32)
    distinct = _alias_sensitive_persistent_sweeps.bind((x, torch.randn_like(x)))
    same = _alias_sensitive_persistent_sweeps.bind((x, x))
    storage = torch.randn(16 * 128 + 1, dtype=torch.float32)
    overlapping = _alias_sensitive_persistent_sweeps.bind(
        (
            torch.as_strided(storage, (16, 128), (128, 1)),
            torch.as_strided(storage, (16, 128), (128, 1), storage_offset=1),
        )
    )

    assert distinct is not same
    assert distinct is not overlapping


def test_persistent_hoist_scope_rejects_inner_loop_targets() -> None:
    hoist_parent = ast.parse("dominating_value = thread_value").body
    grid = SimpleNamespace(
        lane_loops=[
            ("outer_lane", 2),
            ("synthetic_lane_1", 8),
            ("inner_lane", 2),
        ],
        vec_lane_wrappers={},
        lane_setup_statements=ast.parse(
            """
outer_alias = outer_lane + 3
indirect_outer_alias = outer_alias + 1
inner_alias = inner_lane + 7
indirect_inner_alias = inner_alias + 1
"""
        ).body,
        hoist_parent_statements=hoist_parent,
    )
    codegen = SimpleNamespace(
        current_grid_state=grid,
        statements_stack=[[], hoist_parent, []],
        active_device_loops={},
    )
    state = SimpleNamespace(codegen=codegen)
    strategy = SimpleNamespace(
        _synthetic_cute_lane_var="synthetic_lane_1",
        _cute_lane_base_index_var="reduction_lane_base_1",
    )
    typed_state = cast("Any", state)
    typed_strategy = cast("Any", strategy)

    assert _persistent_vec_scope_safe(typed_state, typed_strategy, "ptr + outer_lane")
    assert _persistent_vec_scope_safe(typed_state, typed_strategy, "ptr + outer_alias")
    assert _persistent_vec_scope_safe(
        typed_state, typed_strategy, "ptr + indirect_outer_alias"
    )
    assert _persistent_vec_scope_safe(
        typed_state, typed_strategy, "ptr + dominating_value"
    )
    assert not _persistent_vec_scope_safe(
        typed_state, typed_strategy, "ptr + inner_lane"
    )
    assert not _persistent_vec_scope_safe(
        typed_state, typed_strategy, "ptr + inner_alias"
    )
    assert not _persistent_vec_scope_safe(
        typed_state, typed_strategy, "ptr + indirect_inner_alias"
    )

    nested_parent = ast.parse("parent_value = source.load()").body
    codegen.statements_stack = [[], hoist_parent, nested_parent, []]
    assert not _persistent_vec_scope_safe(
        typed_state, typed_strategy, "ptr + parent_value"
    )

    for_node = ast.parse("for serial_index in range(4):\n    pass").body[0]
    codegen.active_device_loops = {
        0: [SimpleNamespace(for_node=for_node)],
    }
    assert not _persistent_vec_scope_safe(
        typed_state, typed_strategy, "ptr + serial_index"
    )


def test_persistent_reduction_threads_do_not_consume_tile_budget() -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
    # Deliberately permute num_threads relative to block_sizes.  Block 1 is a
    # static persistent reduction and block 2 is a rollable reduction.
    spec.num_threads.append(NumThreadsSpec(block_id=1, size_hint=128))
    spec.num_threads.append(NumThreadsSpec(block_id=0, size_hint=64))
    spec.num_threads.append(NumThreadsSpec(block_id=2, size_hint=512))
    spec.reduction_loops.append(ReductionLoopSpec(block_id=2, size_hint=512))
    spec.reduction_block_ids.update((1, 2))
    config = helion.Config(
        block_sizes=[64],
        num_threads=[16, 8, 32],
        reduction_loops=[None],
    )

    spec.normalize(config)

    # Only the eight row threads consume the competing CTA budget.  Counting
    # the persistent reduction's 16 threads as another tile axis would shrink
    # this loop chunk incorrectly to 8 instead of 128.
    assert config.config["reduction_loops"] == [128]


def test_persistent_loop_rewrites_preserve_constexpr_iterator() -> None:
    from helion._compiler.tile_strategy import _ChunkRecurrence
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import _rewrite_chunk_recurrence
    from helion._compiler.tile_strategy import (
        interchange_lane_outside_serial_reductions,
    )

    lane = _create_lane_loop("lane", 8, [])
    lane.iter = ast.parse("cutlass.range_constexpr(8)", mode="eval").body
    serial_loop = ast.parse(
        """
for mb in range(2):
    reduced = _helion_lane_reduce(
        lane_value, 'sum', cutlass.Float32(0), 32, 1, 0, '', 1
    )
    sink.store(reduced)
"""
    ).body[0]
    lane.body = [*ast.parse("lane_value = lane + 1").body, serial_loop]
    interchanged = interchange_lane_outside_serial_reductions([lane])
    assert (
        ast.unparse(
            ast.Module(body=cast("list[ast.stmt]", interchanged), type_ignores=[])
        ).count("for lane in cutlass.range_constexpr(8)")
        == 2
    )

    recurrence_lane = _create_lane_loop(
        "lane",
        8,
        cast(
            "list[ast.AST]",
            ast.parse(
                """
partial = lane + acc
base = acc
dot_acc = dot_acc + partial
acc = base + dot_acc
"""
            ).body,
        ),
    )
    recurrence_lane.iter = ast.parse("cutlass.range_constexpr(8)", mode="eval").body
    chunk_loop = ast.parse("for chunk in range(2):\n    pass").body[0]
    assert isinstance(chunk_loop, ast.For)
    chunk_loop.body = [recurrence_lane]
    rewritten = _rewrite_chunk_recurrence(
        chunk_loop,
        _ChunkRecurrence(
            dot_acc_var="dot_acc",
            lane_idx=0,
            lane_loop=recurrence_lane,
            lane_var="lane",
            sum_idx=2,
            base_indices=(1,),
            finalize_indices=(),
            final_idx=3,
            state_name="acc",
        ),
        ast.parse("dot_acc = 0").body[0],
        {},
        {},
        {},
    )
    assert "for lane in cutlass.range_constexpr(8)" in ast.unparse(rewritten)


@pytest.mark.parametrize("dependent", (False, True))
@pytest.mark.parametrize("write_kind", ("store", "atomic"))
def test_lane_split_rejects_interleaved_aliasing_write(
    monkeypatch: pytest.MonkeyPatch,
    dependent: bool,
    write_kind: str,
) -> None:
    from helion import exc
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    second_value = "second + reduced_0" if dependent else "second"
    write = (
        "(x.iterator + synthetic_lane_7).store(replacement)"
        if write_kind == "store"
        else "cute.arch.atomic_add(x.iterator + synthetic_lane_7, replacement)"
    )
    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            f"""
first = (x.iterator + synthetic_lane_7).load()
reduced_0 = _helion_lane_reduce(first, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
{write}
second = (x.iterator + synthetic_lane_7).load()
second_input = {second_value}
reduced_1 = _helion_lane_reduce(second_input, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
sink = reduced_0 + reduced_1
"""
        ).body,
    )
    with pytest.raises(exc.BackendUnsupported, match="potentially aliasing write"):
        split_lane_loop_reductions([loop])


def test_dependent_lane_split_preserves_invariant_carried_update() -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "lane",
        8,
        ast.parse(
            """
carry_copy = carry
first = lane + 1
reduced_0 = _helion_lane_reduce(first, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
second = first + reduced_0
reduced_1 = _helion_lane_reduce(second, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
next_carry = carry_copy + reduced_1
"""
        ).body,
    )

    split = split_lane_loop_reductions([loop])
    code = ast.unparse(ast.Module(body=cast("list[ast.stmt]", split), type_ignores=[]))

    assert code.count("for lane in range(8)") == 2
    assert "reduced_1 = cute.arch.warp_reduction_sum(" in code
    assert "next_carry = carry_copy + reduced_1" in code
    assert code.index("reduced_1 =") < code.index("next_carry =")


def test_lane_split_keeps_serially_carried_reduction_input_in_order() -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "lane",
        8,
        ast.parse(
            """
acc = cutlass.Float32(0)
for offset in range(4):
    acc_copy = acc
    acc_next = acc_copy + lane + offset
reduced = _helion_lane_reduce(acc, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
sink.store(reduced)
"""
        ).body,
    )

    split = split_lane_loop_reductions([loop])
    code = ast.unparse(ast.Module(body=cast("list[ast.stmt]", split), type_ignores=[]))

    assert code.count("for lane in range(8)") == 1
    assert "for offset in range(4)" in code
    assert "reduced = acc" in code
    assert "_lane_acc" not in code


@pytest.mark.parametrize("reload_mode", ("register", "gmem"))
@onlyBackends(["cute"])
def test_same_tensor_launch_reloads_after_interleaved_store(
    reload_mode: str,
) -> None:
    _alias_sensitive_persistent_sweeps.reset()
    shared = torch.randn(16, 128, dtype=torch.bfloat16)
    bound = _alias_sensitive_persistent_sweeps.bind((shared, shared))
    host_function = bound.host_function
    assert host_function is not None
    seed = CutePersistentSubwarpRowsHeuristic.get_seed_config(
        bound.env, host_function.device_ir
    )
    assert seed is not None
    reduction_block = next(
        block_id
        for block_id, block in enumerate(bound.env.block_sizes)
        if block.reduction
    )
    reloads = list(seed.config["cute_reduction_reloads"])
    reloads[
        bound.env.config_spec.cute_reduction_reloads.block_id_to_index(reduction_block)
    ] = reload_mode
    config = helion.Config.from_dict({**seed.config, "cute_reduction_reloads": reloads})

    with pytest.raises(
        helion.exc.BackendUnsupported, match="potentially aliasing write"
    ):
        bound.to_code(config)


def test_lane_split_allows_proven_disjoint_interleaved_write() -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            """
first = (x.iterator + synthetic_lane_7).load()
reduced_0 = _helion_lane_reduce(first, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
(y.iterator + synthetic_lane_7).store(replacement)
second = (x.iterator + synthetic_lane_7).load()
reduced_1 = _helion_lane_reduce(second, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
sink = reduced_0 + reduced_1
"""
        ).body,
    )
    split = split_lane_loop_reductions(
        [loop],
        proven_disjoint_tensor_pairs={frozenset(("x", "y"))},
    )
    code = ast.unparse(ast.Module(body=cast("list[ast.stmt]", split), type_ignores=[]))

    assert code.count("for synthetic_lane_7 in range(8)") >= 2
    assert "_lane_acc" in code


@pytest.mark.parametrize("store_shift", (0, 1))
def test_lane_split_exact_same_lane_state_update(
    store_shift: int,
) -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    store_index = "lane_index" if store_shift == 0 else "lane_index + 1"
    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            f"""
lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
loaded = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', state.iterator + cutlass.Int32(slot) * cutlass.Int32(state.layout.stride[0]) + cutlass.Int32(lane_index) * cutlass.Int32(state.layout.stride[1]), (state.iterator + cutlass.Int32(slot) * cutlass.Int32(state.layout.stride[0]) + cutlass.Int32(lane_index) * cutlass.Int32(state.layout.stride[1])).load())
partial = cutlass.Float32(loaded) * cutlass.Float32(loaded)
reduced = _helion_lane_reduce(partial, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
updated = cutlass.Float32(loaded) + reduced
_helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', state.iterator + cutlass.Int32(slot) * cutlass.Int32(state.layout.stride[0]) + cutlass.Int32({store_index}) * cutlass.Int32(state.layout.stride[1]), cutlass.BFloat16(updated), None)
"""
        ).body,
    )
    loop.iter = ast.parse("cutlass.range_constexpr(8)", mode="eval").body

    if store_shift:
        with pytest.raises(
            helion.exc.BackendUnsupported, match="potentially aliasing write"
        ):
            split_lane_loop_reductions([loop])
        return

    split = split_lane_loop_reductions([loop])
    code = ast.unparse(ast.Module(body=cast("list[ast.stmt]", split), type_ignores=[]))
    assert code.count("for synthetic_lane_7 in cutlass.range_constexpr(8)") >= 2
    assert "reduced_lane_acc" in code


@pytest.mark.parametrize(
    ("pointer_expr", "stride_values"),
    (
        ("state.iterator + cutlass.Int32(lane_index)", {}),
        (
            "state.iterator + cutlass.Int32(lane_index) * cutlass.Int32(state.layout.stride[1])",
            {("state", 1): 128},
        ),
    ),
    ids=("literal-unit-step", "specialized-tensor-stride"),
)
def test_lane_split_plain_scalar_exact_same_lane_state_update(
    pointer_expr: str,
    stride_values: dict[tuple[str, int], int],
) -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            f"""
lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
pointer = {pointer_expr}
loaded = pointer.load()
partial = cutlass.Float32(loaded) * cutlass.Float32(loaded)
reduced = _helion_lane_reduce(partial, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
updated = cutlass.Float32(loaded) + reduced
pointer.store(cutlass.BFloat16(updated))
"""
        ).body,
    )

    split = split_lane_loop_reductions(
        [loop],
        proven_tensor_stride_values=stride_values,
    )
    code = ast.unparse(ast.Module(body=cast("list[ast.stmt]", split), type_ignores=[]))

    assert code.count("for synthetic_lane_7 in range(8)") >= 2
    assert "reduced_lane_acc" in code


@pytest.mark.parametrize("marked", (False, True), ids=("scalar", "marked"))
def test_lane_split_allows_row_aligned_dynamic_state_slots(marked: bool) -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    load_pointer = (
        "state.iterator + cutlass.Int32(load_slot) * "
        "cutlass.Int32(state.layout.stride[0]) + "
        "cutlass.Int32(row) * cutlass.Int32(state.layout.stride[2]) + "
        "cutlass.Int32(lane_index) * cutlass.Int32(state.layout.stride[3])"
    )
    store_pointer = load_pointer.replace("load_slot", "store_slot")
    load = f"({load_pointer}).load()"
    store = f"({store_pointer}).store(cutlass.BFloat16(updated))"
    if marked:
        load = (
            "_helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', "
            f"{load_pointer}, {load})"
        )
        store = (
            "_helion_persistent_branch_vec_store(7, 8, "
            f"'cutlass.BFloat16', {store_pointer}, "
            "cutlass.BFloat16(updated), None)"
        )
    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            f"""
lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
loaded = {load}
partial = cutlass.Float32(loaded) * cutlass.Float32(loaded)
reduced = _helion_lane_reduce(partial, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
updated = cutlass.Float32(loaded) + reduced
{store}
"""
        ).body,
    )
    loop.iter = ast.parse("cutlass.range_constexpr(8)", mode="eval").body

    split = split_lane_loop_reductions(
        [loop],
        proven_tensor_stride_values={
            ("state", 0): 524288,
            ("state", 2): 128,
            ("state", 3): 1,
        },
    )
    code = ast.unparse(ast.Module(body=cast("list[ast.stmt]", split), type_ignores=[]))

    assert code.count("for synthetic_lane_7 in cutlass.range_constexpr(8)") >= 2
    assert "reduced_lane_acc" in code


def test_lane_split_rejects_dynamic_rows_with_insufficient_stride() -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            """
lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
loaded = (state.iterator + load_row * state.layout.stride[0] + lane_index).load()
partial = cutlass.Float32(loaded) * cutlass.Float32(loaded)
reduced = _helion_lane_reduce(partial, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
updated = cutlass.Float32(loaded) + reduced
(state.iterator + store_row * state.layout.stride[0] + lane_index).store(updated)
"""
        ).body,
    )

    with pytest.raises(
        helion.exc.BackendUnsupported, match="potentially aliasing write"
    ):
        split_lane_loop_reductions(
            [loop],
            proven_tensor_stride_values={("state", 0): 4},
        )


@pytest.mark.parametrize(
    ("second_pointer", "second_write", "row_stride", "vector_store_count"),
    (
        (
            "state.iterator + second_slot * state.layout.stride[0] + synthetic_lane_7",
            "_helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', {pointer}, second_value, None)",
            128,
            2,
        ),
        (
            "state.iterator + second_slot * state.layout.stride[0] + synthetic_lane_7 + 1",
            "_helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', {pointer}, second_value, None)",
            128,
            0,
        ),
        (
            "state.iterator + second_slot * state.layout.stride[0] + synthetic_lane_7",
            "_helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', {pointer}, second_value, None)",
            4,
            0,
        ),
        (
            "state.iterator + second_slot * state.layout.stride[0] + synthetic_lane_7",
            "cute.arch.atomic_add({pointer}, second_value)",
            128,
            0,
        ),
    ),
    ids=("row-disjoint", "shifted", "insufficient-stride", "atomic"),
)
def test_late_vectorizer_store_pairs_fail_closed(
    second_pointer: str,
    second_write: str,
    row_stride: int,
    vector_store_count: int,
) -> None:
    from helion._compiler.cute.persistent_branch_vec import (
        vectorize_branch_local_persistent_fragments,
    )

    first_pointer = (
        "state.iterator + first_slot * state.layout.stride[0] + synthetic_lane_7"
    )
    loop = ast.parse(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    first_value = cutlass.BFloat16(synthetic_lane_7)
    _helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', {first_pointer}, first_value, None)
    second_value = cutlass.BFloat16(synthetic_lane_7 + 1)
    {second_write.format(pointer=second_pointer)}
"""
    ).body
    transformed = vectorize_branch_local_persistent_fragments(
        loop,
        proven_tensor_stride_values={("state", 0): row_stride},
    )
    code = ast.unparse(
        ast.Module(body=cast("list[ast.stmt]", transformed), type_ignores=[])
    )

    assert code.count("_cute_store_u16_vec") == vector_store_count
    if vector_store_count == 0:
        assert "_persistent_branch_store_values" not in code


@pytest.mark.parametrize(
    ("load_pointer", "store_pointer", "write", "stride_values"),
    (
        (
            "state.iterator + lane_index * state.layout.stride[1]",
            "state.iterator + lane_index * state.layout.stride[1] + 1",
            "({pointer}).store(updated)",
            {("state", 1): 128},
        ),
        (
            "state.iterator + lane_index * 0",
            "state.iterator + lane_index * 0",
            "({pointer}).store(updated)",
            {},
        ),
        (
            "state.iterator + lane_index * state.layout.stride[1]",
            "state.iterator + lane_index * state.layout.stride[1]",
            "({pointer}).store(updated)",
            {("state", 1): 0},
        ),
        (
            "state.iterator + cutlass.Int32(lane_index * 1073741824)",
            "state.iterator + cutlass.Int32(lane_index * 1073741824)",
            "({pointer}).store(updated)",
            {},
        ),
        (
            "state.iterator + lane_index // 2",
            "state.iterator + lane_index // 2",
            "({pointer}).store(updated)",
            {},
        ),
        (
            "state.iterator + lane_index * state.layout.stride[1]",
            "state.iterator + lane_index * state.layout.stride[1]",
            "({pointer}).store(updated)",
            {},
        ),
        (
            "base_pointer + lane_index",
            "base_pointer + lane_index",
            "({pointer}).store(updated)",
            {},
        ),
        (
            "state.iterator + lane_index",
            "state.iterator + lane_index",
            "cute.arch.atomic_add({pointer}, updated)",
            {},
        ),
        (
            "state.iterator + lane_index + carried_offset",
            "state.iterator + lane_index + carried_offset",
            "({pointer}).store(updated)",
            {},
        ),
    ),
    ids=(
        "shifted",
        "literal-zero-stride",
        "specialized-zero-stride",
        "int32-wrapping-stride",
        "non-injective",
        "unproven-runtime-stride",
        "unknown-pointer",
        "atomic",
        "loop-carried-base",
    ),
)
def test_lane_split_plain_scalar_same_lane_proof_fails_closed(
    load_pointer: str,
    store_pointer: str,
    write: str,
    stride_values: dict[tuple[str, int], int],
) -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            f"""
lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
{"carried_offset = carried_offset - 1" if "carried_offset" in load_pointer else ""}
loaded = ({load_pointer}).load()
partial = cutlass.Float32(loaded) * cutlass.Float32(loaded)
reduced = _helion_lane_reduce(partial, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
updated = cutlass.Float32(loaded) + reduced
{write.format(pointer=store_pointer)}
"""
        ).body,
    )

    with pytest.raises(
        helion.exc.BackendUnsupported, match="potentially aliasing write"
    ):
        split_lane_loop_reductions(
            [loop],
            proven_tensor_stride_values=stride_values,
        )


def test_lane_split_plain_scalar_rejects_mutated_address_alias() -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            """
lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
offset = outer_offset
loaded = (state.iterator + lane_index + offset).load()
partial = cutlass.Float32(loaded) * cutlass.Float32(loaded)
reduced = _helion_lane_reduce(partial, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
offset += 1
updated = cutlass.Float32(loaded) + reduced
(state.iterator + lane_index + offset).store(updated)
"""
        ).body,
    )

    with pytest.raises(
        helion.exc.BackendUnsupported, match="potentially aliasing write"
    ):
        split_lane_loop_reductions([loop])


def test_lane_split_plain_scalar_rejects_mutated_lane_alias() -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            """
lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
pointer = state.iterator + lane_index
loaded = pointer.load()
partial = cutlass.Float32(loaded) * cutlass.Float32(loaded)
reduced = _helion_lane_reduce(partial, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
pointer += 1
updated = cutlass.Float32(loaded) + reduced
pointer.store(updated)
"""
        ).body,
    )

    with pytest.raises(
        helion.exc.BackendUnsupported, match="potentially aliasing write"
    ):
        split_lane_loop_reductions([loop])


@pytest.mark.parametrize(
    "mutation",
    (
        "lane_index *= 0",
        "lane_index: int = 0",
        "lane_index, other = 0, 0",
        "if predicate:\n    lane_index = 0",
    ),
    ids=("augassign", "annotated", "tuple", "conditional"),
)
def test_lane_split_plain_scalar_rejects_stale_preload_definition(
    mutation: str,
) -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    source = f"""
lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
{mutation}
loaded = (state.iterator + lane_index).load()
partial = cutlass.Float32(loaded) * cutlass.Float32(loaded)
reduced = _helion_lane_reduce(partial, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
updated = cutlass.Float32(loaded) + reduced
(state.iterator + lane_index).store(updated)
"""
    loop = _create_lane_loop("synthetic_lane_7", 8, ast.parse(source).body)

    with pytest.raises(
        helion.exc.BackendUnsupported, match="potentially aliasing write"
    ):
        split_lane_loop_reductions([loop])


def test_lane_split_plain_scalar_rejects_captured_shifted_aliases() -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            """
offset = 0
load_index = synthetic_lane_7 + offset
offset = 1
store_index = synthetic_lane_7 + offset
offset = 2
loaded = (state.iterator + load_index).load()
partial = cutlass.Float32(loaded) * cutlass.Float32(loaded)
reduced = _helion_lane_reduce(partial, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
updated = cutlass.Float32(loaded) + reduced
(state.iterator + store_index).store(updated)
"""
        ).body,
    )

    with pytest.raises(
        helion.exc.BackendUnsupported, match="potentially aliasing write"
    ):
        split_lane_loop_reductions([loop])


@pytest.mark.parametrize(
    ("initial", "mutation", "load_offset", "store_offset"),
    (
        (
            "offset = 0",
            "for offset in range(1, 2):\n    pass",
            "offset",
            "offset",
        ),
        (
            "holder.value = 0",
            "holder.value = 1",
            "holder.value",
            "holder.value",
        ),
        (
            "holder.value = 0",
            "alias = holder\nalias.value = 1",
            "holder.value",
            "holder.value",
        ),
    ),
    ids=("for-target", "attribute-store", "attribute-store-through-alias"),
)
def test_lane_split_plain_scalar_rejects_captured_structured_writes(
    initial: str,
    mutation: str,
    load_offset: str,
    store_offset: str,
) -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            f"""
{initial}
load_index = synthetic_lane_7 + {load_offset}
{mutation}
store_index = synthetic_lane_7 + {store_offset}
loaded = (state.iterator + load_index).load()
partial = cutlass.Float32(loaded) * cutlass.Float32(loaded)
reduced = _helion_lane_reduce(partial, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
updated = cutlass.Float32(loaded) + reduced
(state.iterator + store_index).store(updated)
"""
        ).body,
    )

    with pytest.raises(
        helion.exc.BackendUnsupported, match="potentially aliasing write"
    ):
        split_lane_loop_reductions([loop])


def test_lane_split_rejects_exact_store_before_same_lane_load() -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            """
lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
_helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', state.iterator + lane_index, replacement, None)
loaded = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', state.iterator + lane_index, (state.iterator + lane_index).load())
reduced = _helion_lane_reduce(loaded, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
sink = reduced + loaded
"""
        ).body,
    )
    loop.iter = ast.parse("cutlass.range_constexpr(8)", mode="eval").body

    with pytest.raises(
        helion.exc.BackendUnsupported, match="potentially aliasing write"
    ):
        split_lane_loop_reductions([loop])


@pytest.mark.parametrize("dependent", (False, True))
@pytest.mark.parametrize("write_kind", ("store", "atomic"))
@pytest.mark.parametrize("prove_x_y_disjoint", (False, True))
def test_lane_split_treats_later_write_as_a_backedge_barrier(
    dependent: bool,
    write_kind: str,
    prove_x_y_disjoint: bool,
) -> None:
    from helion._compiler.tile_strategy import _create_lane_loop
    from helion._compiler.tile_strategy import split_lane_loop_reductions

    second_value = "second + reduced_0" if dependent else "second"
    write = (
        "(y.iterator + synthetic_lane_7 + 1).store(replacement)"
        if write_kind == "store"
        else "cute.arch.atomic_add(y.iterator + synthetic_lane_7 + 1, replacement)"
    )
    loop = _create_lane_loop(
        "synthetic_lane_7",
        8,
        ast.parse(
            f"""
first = (x.iterator + synthetic_lane_7).load()
reduced_0 = _helion_lane_reduce(first, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
second = (x.iterator + synthetic_lane_7).load()
second_input = {second_value}
reduced_1 = _helion_lane_reduce(second_input, 'sum', cutlass.Float32(0), 16, 1, 0, '', 1, 1)
{write}
sink = reduced_0 + reduced_1
"""
        ).body,
    )
    if not prove_x_y_disjoint:
        with pytest.raises(
            helion.exc.BackendUnsupported, match="potentially aliasing write"
        ):
            split_lane_loop_reductions([loop])
        return

    split = split_lane_loop_reductions(
        [loop],
        proven_disjoint_tensor_pairs={frozenset(("x", "y"))},
    )
    code = ast.unparse(ast.Module(body=cast("list[ast.stmt]", split), type_ignores=[]))

    assert code.count("for synthetic_lane_7 in range(8)") >= 2
    assert "_lane_acc" in code


def test_fallback_setup_dependency_stays_in_innermost_scope() -> None:
    from helion._compiler.tile_strategy import DeviceGridState

    grid = DeviceGridState(
        cast("Any", SimpleNamespace(block_ids=[0, 1])),
        {},
        lane_loops=[("outer_lane", 2), ("inner_lane", 2)],
        lane_setup_statements=cast(
            "list[ast.AST]",
            ast.parse(
                """
fallback = external_value
dependent = outer_lane + fallback
"""
            ).body,
        ),
    )
    wrapped = grid.wrap_body(
        cast("list[ast.AST]", ast.parse("sink = dependent + inner_lane").body)
    )
    code = ast.unparse(
        ast.Module(body=cast("list[ast.stmt]", wrapped), type_ignores=[])
    )

    inner = code.index("for inner_lane in range(2)")
    fallback = code.index("fallback = external_value")
    dependent = code.index("dependent = outer_lane + fallback")
    assert inner < fallback < dependent


@onlyBackends(["cute"])
class TestCutePersistentReduction(TestCase):
    def test_persistent_subwarp_vector_topology(self) -> None:
        # Exercise a real tail tile: the second CTA has only one live row, so
        # an unguarded state-vector hoist would read beyond row 16.
        rows, width = 17, 128
        state = torch.randn(rows, width, device=DEVICE, dtype=torch.bfloat16)
        packed_key_query = torch.randn(2 * width, device=DEVICE, dtype=torch.bfloat16)
        value = torch.randn(rows, device=DEVICE, dtype=torch.bfloat16)
        state_ref = state.clone().float()

        bound = _persistent_state_update.bind((state, packed_key_query, value))
        env = bound.env
        reduction_blocks = [
            block_id
            for block_id, block in enumerate(env.block_sizes)
            if block.reduction
        ]
        self.assertEqual(len(reduction_blocks), 1)
        reduction_block = reduction_blocks[0]
        tile_blocks = env.config_spec.block_sizes.valid_block_ids()
        self.assertEqual(len(tile_blocks), 1)
        tile_block = tile_blocks[0]

        def values_for(spec: object, values: dict[int, object], default: object):
            return [
                values.get(block_id, default)
                for block_id in spec.valid_block_ids()  # type: ignore[attr-defined]
            ]

        code, (actual, actual_state) = code_and_output(
            _persistent_state_update,
            (state, packed_key_query, value),
            block_sizes=values_for(env.config_spec.block_sizes, {tile_block: 16}, 16),
            num_threads=values_for(
                env.config_spec.num_threads,
                {tile_block: 8, reduction_block: 16},
                0,
            ),
            cute_vector_widths=values_for(
                env.config_spec.cute_vector_widths,
                {reduction_block: 8},
                1,
            ),
            cute_lane_layouts=values_for(
                env.config_spec.cute_lane_layouts,
                {reduction_block: "blocked"},
                "blocked",
            ),
            cute_reduction_reloads=values_for(
                env.config_spec.cute_reduction_reloads,
                {reduction_block: "register"},
                "auto",
            ),
            num_warps=4,
            num_stages=1,
            pid_type="flat",
        )

        key_ref = packed_key_query[:width].float()
        key_ref *= torch.rsqrt((key_ref * key_ref).sum() + 1e-6)
        residual = value.float() - (state_ref * key_ref[None, :]).sum(-1)
        state_ref += residual[:, None] * key_ref[None, :]
        expected = (
            (state_ref * packed_key_query[width:].float()[None, :])
            .sum(-1)
            .to(state.dtype)
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            actual_state, state_ref.to(state.dtype), rtol=2e-2, atol=2e-2
        )

        self.assertIn("block=(16, 8, 1)", code)
        self.assertIn("threads_in_group=16", code)
        self.assertGreaterEqual(
            code.count("ir.VectorType.get([8], cutlass.Uint16.mlir_type)"), 3
        )
        self.assertIn("_cute_store_u16_vec", code)
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)
