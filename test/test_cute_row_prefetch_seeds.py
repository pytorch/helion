from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_fuser_occurrences import _args
from .test_cute_fuser_occurrences import _state_reduction
from helion._compiler.autotuner_heuristics import cute_row_prefetch
from helion._compiler.autotuner_heuristics.cute import (
    CutePersistentSubwarpRowsHeuristic,
)


@pytest.mark.parametrize(
    "dtype,state_dtype,extent,prefetch,threads",
    [
        (torch.bfloat16, torch.bfloat16, 128, 8, [4, 16]),
        (torch.float16, torch.float16, 128, 8, [4, 16]),
        (torch.bfloat16, torch.bfloat16, 256, 8, [2, 32]),
        (torch.bfloat16, torch.float32, 128, 4, [4, 16]),
        (torch.float32, torch.float32, 128, 8, [2, 32]),
    ],
)
def test_actual_structural_pool_preserves_default_and_prefix(
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    extent: int,
    prefetch: int,
    threads: list[int],
) -> None:
    args = _args(dtype, 128, extent)
    args = (args[0].to(state_dtype), *args[1:])
    with _cpu_codegen():
        with patch.object(
            cute_row_prefetch,
            "append_row_prefetch_seeds",
            side_effect=lambda env, ir, seeds: seeds,
        ):
            old = _state_reduction._bind_isolated(args)
        append_prefetch = cute_row_prefetch.append_row_prefetch_seeds
        stages = []

        def observe(env, ir, seeds):
            before = deepcopy(seeds)
            result = append_prefetch(env, ir, seeds)
            assert seeds == before
            stages.append((before, deepcopy(result)))
            return result

        with patch.object(cute_row_prefetch, "append_row_prefetch_seeds", observe):
            new = _state_reduction._bind_isolated(args)
        assert len(stages) == 1
        before, after = stages[0]
        assert after[:-3] == before
        assert old.config_spec.compiler_seed_configs[: len(before)] == before
        assert new.config_spec.compiler_seed_configs[: len(after)] == after
        assert new.config_spec.default_config() == old.config_spec.default_config()
        assert after[-3]["num_threads"] == threads
        assert after[-3]["block_sizes"] == [threads[0] * prefetch]
        assert after[-3]["cute_lane_unroll"] == prefetch
        assert after[-1]["block_sizes"] == [4]
        assert after[-1]["num_threads"] == [4, 32]
        for seed in after[-3:]:
            assert seed["cute_proven_bounds"] is True
            assert seed["cute_reduction_reloads"] == ["register"]
            assert seed.config.get("cute_bf16x2_recurrence", False) is False
            generation = new.config_spec.create_config_generation()
            normalized = generation.unflatten(generation.flatten(seed))
            _, strict = generation.strict_config_pair(normalized)
            assert normalized == strict
            for key in (
                "block_sizes",
                "num_threads",
                "cute_vector_widths",
                "cute_reduction_reloads",
                "cute_proven_bounds",
                "cute_lane_unroll",
            ):
                assert strict[key] == seed[key]


@pytest.mark.parametrize(
    "dtype,extent",
    [
        (torch.bfloat16, 64),
        (torch.bfloat16, 117),
        (torch.bfloat16, 512),
        (torch.float32, 256),
        (torch.float64, 128),
    ],
)
def test_unsupported_reduction_keeps_pool(dtype: torch.dtype, extent: int) -> None:
    with _cpu_codegen():
        bound = _state_reduction._bind_isolated(_args(dtype, 128, extent))
        assert bound.host_function is not None
        before = bound.config_spec.compiler_seed_configs
        assert (
            cute_row_prefetch.append_row_prefetch_seeds(
                bound.env, bound.host_function.device_ir, before
            )
            is before
        )


@pytest.mark.parametrize("boundary", ["bounds", "metadata", "reload", "unroll", "pid"])
def test_missing_facts_and_explicit_overrides_decline(
    boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _cpu_codegen():
        bound = _state_reduction._bind_isolated(_args(torch.bfloat16, 128, 128))
        assert bound.host_function is not None
        env = bound.env
        if boundary == "bounds":
            env.config_spec.cute_proven_bounds_enabled = False
        elif boundary == "metadata":
            env.compiler_fact_specialization_facts = frozenset()
        else:
            monkeypatch.setattr(
                env.settings,
                "autotune_config_overrides",
                {
                    "reload": {"cute_reduction_reloads": ["gmem"]},
                    "unroll": {"cute_lane_unroll": 2},
                    "pid": {"pid_type": "persistent_interleaved"},
                }[boundary],
            )
        seeds = []
        assert (
            cute_row_prefetch.append_row_prefetch_seeds(
                env, bound.host_function.device_ir, seeds
            )
            is seeds
        )


def test_axes_not_sequence_positions_and_no_duplicate_append() -> None:
    with _cpu_codegen():
        bound = _state_reduction._bind_isolated(_args(torch.bfloat16, 128, 128))
        assert bound.host_function is not None
        env = bound.env
        spec = env.config_spec
        ir = bound.host_function.device_ir
        plan = CutePersistentSubwarpRowsHeuristic._plan(env, ir)
        assert plan is not None
        row, reduction, _ = plan
        for seq in (spec.num_threads, spec.cute_vector_widths, spec.cute_lane_layouts):
            seq.reverse()
        seeds = cute_row_prefetch.append_row_prefetch_seeds(env, ir, [])
        assert len(seeds) == 3
        first = seeds[0]
        assert spec.num_threads.config_get(first["num_threads"], row, 0) == 4
        assert spec.num_threads.config_get(first["num_threads"], reduction, 0) == 16
        assert (
            spec.cute_vector_widths.config_get(
                first["cute_vector_widths"], reduction, 0
            )
            == 8
        )
        assert cute_row_prefetch.append_row_prefetch_seeds(env, ir, seeds) is seeds
        assert spec.autotuner_heuristics.count("cute_resident_row_prefetch_v2") == 1


def test_row_domain_excludes_small_sibling_without_repair() -> None:
    with _cpu_codegen():
        bound = _state_reduction._bind_isolated(_args(torch.bfloat16, 128, 128))
        assert bound.host_function is not None
        bound.config_spec.block_sizes[0].min_size = 16
        seeds = cute_row_prefetch.append_row_prefetch_seeds(
            bound.env, bound.host_function.device_ir, []
        )
        assert len(seeds) == 1
        assert seeds[0]["block_sizes"] == [32]


@pytest.mark.parametrize("change", ["dtype", "stride", "missing_value", "readonly"])
def test_packet_budget_requires_current_read_write_tensor_facts(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _cpu_codegen():
        bound = _state_reduction._bind_isolated(_args(torch.bfloat16, 128, 128))
        assert bound.host_function is not None
        ir = bound.host_function.device_ir
        plan = CutePersistentSubwarpRowsHeuristic._plan(bound.env, ir)
        assert plan is not None
        row, _, _ = plan
        assert (
            cute_row_prefetch._read_write_packet_bytes(bound.env, ir, row, 128, 8) == 16
        )
        nodes = list(ir.graphs[0].graph.nodes)
        state_load = next(
            node
            for node in nodes
            if node.target is cute_row_prefetch.load
            and node.args[0].meta["val"].ndim == 4
        )
        host = state_load.args[0]
        value = host.meta["val"]
        if change == "dtype":
            monkeypatch.setitem(host.meta, "val", value.to(torch.float64))
        elif change == "stride":
            strides = (*value.stride()[:-1], 2)
            monkeypatch.setitem(
                host.meta,
                "val",
                torch.empty_strided(
                    value.shape, strides, dtype=value.dtype, device=value.device
                ),
            )
        elif change == "missing_value":
            monkeypatch.setitem(state_load.meta, "val", None)
        else:
            for node in nodes:
                if node.target is cute_row_prefetch.store and node.args[0] is host:
                    monkeypatch.setattr(node, "target", "unrecognized_store")
        assert (
            cute_row_prefetch._read_write_packet_bytes(bound.env, ir, row, 128, 8)
            is None
        )
