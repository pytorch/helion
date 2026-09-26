from __future__ import annotations

import ast
from typing import Literal
from typing import cast
from unittest.mock import patch

from examples.aot_example import row_softmax
from examples.welford import welford
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute_resident_sequence import (
    CuteResidentSequenceHeuristic,
)
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = skipUnlessBackends(["cute"])


def _arguments(kind: str, columns: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    matrix = torch.empty((32, columns), dtype=dtype)
    if kind == "welford":
        return (
            torch.empty(columns, dtype=dtype),
            torch.empty(columns, dtype=dtype),
            matrix,
        )
    return (matrix,)


@pytest.mark.parametrize(
    "kind,columns,dtype",
    [
        ("welford", 512, torch.float32),
        ("welford", 1024, torch.float32),
        ("welford", 1536, torch.float32),
        ("welford", 2048, torch.float32),
        ("softmax", 256, torch.bfloat16),
        ("softmax", 1024, torch.bfloat16),
        ("softmax", 4096, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("online_rewrite", [False, True])
def test_original_sequence_seeds_roundtrip_and_emit(
    kind: str,
    columns: int,
    dtype: torch.dtype,
    static: bool,
    online_rewrite: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "0" if online_rewrite else "1")
    kernel = helion.kernel(
        welford.fn if kind == "welford" else row_softmax.fn,
        backend="cute",
        static_shapes=static,
        autotune_effort="none",
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, _arguments(kind, columns, dtype))
        host = bound.host_function
        assert host is not None
        with bound.env:
            seeds = CuteResidentSequenceHeuristic.get_seed_configs(
                bound.env, host.device_ir
            )
            assert seeds
            generation = ConfigGeneration(bound.config_spec)
            surviving = [
                config for _flat, config in generation.seed_flat_config_pairs()
            ]
        for seed in seeds:
            assert (
                seed.block_sizes[1]
                > seed.num_threads[1]
                * cast("list[int]", seed.config["cute_vector_widths"])[1]
            )
            normalized = bound.config_spec.normalized_config(seed)
            flat, roundtrip = generation.canonicalize_flat(generation.flatten(seed))
            assert roundtrip in surviving
            assert generation.unflatten(flat) == roundtrip
            for key in (
                "block_sizes",
                "num_threads",
                "cute_vector_widths",
                "cute_lane_layouts",
                "cute_reduction_sequence",
                "cute_cluster_n",
            ):
                assert roundtrip.config[key] == normalized.config[key]
            source = bound.to_code(roundtrip)
            assert "sequence_acc" in source
            assert "_helion_sequence_reduce" not in source
            if kind == "softmax" and online_rewrite:
                # Independent max/first-tile-max/sum sweeps do not require
                # the dependent-phase register cache. Keep the original
                # resident assertion below for every rewrite-disabled seed,
                # and check the full active-rewrite reduction schedule here.
                tree = ast.parse(source)
                parents = {
                    child: parent
                    for parent in ast.walk(tree)
                    for child in ast.iter_child_nodes(parent)
                }
                reductions = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and (
                        ast.unparse(node.func)
                        in {"cute.arch.warp_reduction", "cute.arch.warp_reduction_sum"}
                        or ast.unparse(node.func).startswith("_cute_grouped_reduce_")
                    )
                ]
                assert len(reductions) == 3
                for reduction in reductions:
                    current: ast.AST = reduction
                    while current in parents:
                        current = parents[current]
                        if isinstance(current, ast.For):
                            assert isinstance(current.target, ast.Name)
                            assert not current.target.id.startswith(
                                ("lane_", "vec_lane_")
                            )
            elif seed.config["cute_reduction_sequence"] == "resident":
                assert "sequence_values" in source
            else:
                assert "sequence_values" not in source


@pytest.mark.parametrize("value", ["other", 0, True, ["resident"]])
def test_sequence_config_rejects_unknown_schedule(value: object) -> None:
    kernel = helion.kernel(welford.fn, backend="cute", autotune_effort="none")
    bound = _cpu_bind(kernel, _arguments("welford", 2048, torch.float32))
    with pytest.raises(helion.exc.InvalidConfig):
        bound.config_spec.normalized_config(
            helion.Config(cute_reduction_sequence=value)
        )


def test_sequence_uses_realized_shared_sibling_thread_extent() -> None:
    kernel = helion.kernel(
        welford.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = _cpu_bind(kernel, _arguments("welford", 2048, torch.float32))
    config = helion.Config(
        block_sizes=[1, 2048, 2048],
        num_threads=[1, 32, 128],
        cute_vector_widths=[1, 4, 4],
        cute_lane_layouts=["blocked", "strided", "strided"],
        cute_reduction_sequence="resident",
        cute_cluster_n=1,
    )
    source = bound.to_code(config)
    # Existing sibling planning shares the physical 128-thread extent with
    # both loops. The sequence must use that extent, not the requested 32.
    assert "block=(128, 1, 1)" in source
    assert "cutlass.Int32(lane_1) * 128" in source
    assert "cute.arch.alloc_smem(cutlass.Int64, 4)" in source
    assert "sequence_tid // 128" in source


@pytest.mark.parametrize("policy", ["streaming", "first", "last", "l2_last"])
def test_sequence_vector_cache_policies_remain_eligible(
    policy: Literal["streaming", "first", "last", "l2_last"],
) -> None:
    kernel = helion.kernel(
        welford.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = _cpu_bind(kernel, _arguments("welford", 2048, torch.float32))
    config = helion.Config(
        block_sizes=[4, 2048, 2048],
        num_threads=[4, 32, 32],
        cute_vector_widths=[1, 4, 4],
        cute_lane_layouts=["blocked", "strided", "strided"],
        cute_reduction_sequence="resident",
        cute_cluster_n=1,
    )
    config = helion.Config.from_dict(
        config.config | {"load_eviction_policies": [policy, "", "", ""]}
    )
    source = bound.to_code(config)
    assert "sequence_acc" in source
    assert "if sequence_initialized" in source
