from __future__ import annotations

import ast
from copy import deepcopy
import math
import random
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test.test_cute_resident_reductions import _FEATURES
from test.test_cute_resident_reductions import _TENSORS
from test.test_cute_resident_reductions import _namespace
from test.test_cute_resident_reductions import _rewrite
from test.test_cute_resident_reductions import _source
from test.test_cute_resident_reductions import _Tensor
from test.test_cute_resident_serial_row_seeds import _bind
from test.test_cute_resident_serial_row_seeds import _cpu_only  # noqa: F401
from test.test_cute_resident_serial_row_seeds import _search

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.autotuner_heuristics.cute_resident_reductions import (
    CuteResidentReductionHeuristic,
)
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.metrics import AutotuneMetrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion.autotuner.base_search import PopulationMember
    from helion.runtime.kernel import BoundKernel


KEY = "cute_reduction_pipeline_depth"


def _family(bound: BoundKernel) -> list[helion.Config]:
    assert bound.host_function is not None
    with bound.env:
        return CuteResidentReductionHeuristic.pipeline_depth_seed_configs(
            bound.env, bound.host_function.device_ir
        )


def _run_pipeline(source: str, rows: int, depth: int, group_rows: int) -> None:
    rng = np.random.default_rng(9061)
    x = rng.normal(size=(rows, _FEATURES)).astype(np.float32)
    dy = rng.normal(size=x.shape).astype(np.float32)
    weight = rng.normal(size=_FEATURES).astype(np.float32)
    mean = rng.normal(size=rows).astype(np.float32)
    scale = rng.uniform(0.5, 2, size=rows).astype(np.float32)
    out = np.full(x.shape, 17, dtype=np.float32)
    partial = np.full(_FEATURES, 19, dtype=np.float32)
    values = (x, dy, weight, mean, scale, out, partial)
    tensors = dict(zip(_TENSORS, map(_Tensor, values), strict=True))
    namespace = _namespace() | tensors | {"initial": -0.0, "limit": 11}
    arch = namespace["cute"].arch
    # No implicit zero initialization may hide a read before its async group
    # completes. Copies publish only when the emitted wait permits them.
    arch.alloc_smem = lambda dtype, size, alignment: (
        _Tensor(np.full(size, np.nan, dtype=dtype)).iterator
    )
    commit, wait = arch.cp_async_commit_group, arch.cp_async_wait_group
    ages: list[int] = []
    waits: list[int] = []

    def committed() -> None:
        commit()
        ages.append(len(namespace["pending_copies"]))

    def waited(remaining: int) -> None:
        waits.append(remaining)
        wait(remaining)

    arch.cp_async_commit_group = committed
    arch.cp_async_wait_group = waited
    exec(compile(source, "<resident ring>", "exec"), namespace)
    normalized = (x - mean[:, None]) * scale[:, None]
    weighted = dy * weight
    expected = (
        weighted
        - (
            normalized * (normalized * weighted).sum(-1, keepdims=True) / _FEATURES
            + weighted.sum(-1, keepdims=True) / _FEATURES
        )
    ) * scale[:, None]
    expected[:, 11:] = np.float32(17)
    expected_partial = np.full(_FEATURES, -0.0, dtype=np.float32)
    for row in range(rows):
        expected_partial = expected_partial + dy[row] * normalized[row]
    np.testing.assert_allclose(out, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(partial, expected_partial)
    np.testing.assert_array_equal(np.signbit(partial), np.signbit(expected_partial))
    assert tensors["x"].reads == tensors["dy"].reads == rows * _FEATURES
    assert tensors["mean"].reads == tensors["scale"].reads == rows
    assert tensors["weight"].reads == _FEATURES
    assert len(ages) == len(waits) == math.ceil(rows / group_rows)
    assert max(ages, default=0) <= depth
    assert set(waits).issubset({depth - 1, 0})
    assert not namespace["pending_copies"]
    assert not namespace["uncommitted_copies"]


@pytest.mark.parametrize("depth", (2, 4))
@pytest.mark.parametrize("rows", (0, 1, 2, 3, 4, 5, 7, 13, 17))
@pytest.mark.parametrize("group_rows", (1, 2, 4))
@pytest.mark.parametrize("width", (1, 4))
def test_pipeline_depth_preserves_carries_masks_and_copy_lifetimes(
    depth: int, rows: int, group_rows: int, width: int
) -> None:
    source = _rewrite(
        _source(rows=rows, width=width),
        width=width,
        group_rows=group_rows,
        pipelined=True,
        pipeline_depth=depth,
    )
    assert "resident_state" in source
    _run_pipeline(source, rows, depth, group_rows)


def test_async_model_detects_an_incomplete_current_group() -> None:
    source = _rewrite(_source(rows=7), pipelined=True, pipeline_depth=4)
    wrong_wait = source.replace("cp_async_wait_group(3)", "cp_async_wait_group(4)")
    assert wrong_wait != source
    with pytest.raises(AssertionError):
        _run_pipeline(wrong_wait, 7, 4, 1)


@pytest.mark.parametrize("group_rows", (1, 2, 4))
@pytest.mark.parametrize("begin", (-(2**31), 2**31 - 8))
def test_pipeline_depth_keeps_original_int32_endpoints(
    group_rows: int, begin: int
) -> None:
    end = begin + 7
    source = _source(rows=7).replace(
        "range(cutlass.Int32(0), cutlass.Int32(7))",
        f"range(cutlass.Int32({begin}), cutlass.Int32({end}))",
    )
    source = source.replace(
        "cutlass.Int32(row)", f"(cutlass.Int32(row) - cutlass.Int32({begin}))"
    )
    rewritten = _rewrite(
        source, group_rows=group_rows, pipelined=True, pipeline_depth=4
    )
    assert "resident_state" in rewritten
    with np.errstate(over="raise", invalid="raise"):
        _run_pipeline(rewritten, 7, 4, group_rows)


@pytest.mark.parametrize("group_rows", (1, 2, 4))
def test_pipeline_budget_accounts_for_every_slot_and_reduction_scratch(
    group_rows: int,
) -> None:
    # Two feature loads, two broadcast loads, four aligned allocations, and
    # two FP32 sums for this one-thread semantic model.
    per_slot = (2 * _FEATURES + 2) * 4 * group_rows
    scratch = 16 * 4 + group_rows * 2 * 4
    for depth in (2, 4):
        required = depth * per_slot + scratch
        for budget, accepted in ((required - 1, False), (required, True)):
            source = _rewrite(
                _source(rows=7),
                group_rows=group_rows,
                pipelined=True,
                pipeline_depth=depth,
                shared_memory_budget=budget,
            )
            assert ("resident_state" in source) is accepted
    source = _rewrite(
        _source(rows=7),
        group_rows=group_rows,
        pipelined=True,
        pipeline_depth=4,
        shared_memory_budget=2 * per_slot + scratch,
    )
    assert "resident_state" not in source


def test_depth_does_not_bypass_effect_alignment_or_alias_proofs() -> None:
    for kwargs in ({"disjoint": False}, {"alignment": 4}):
        source = _rewrite(_source(), pipelined=True, pipeline_depth=4, **kwargs)
        assert "resident_state" not in source
    original = _source().replace(
        "old_carry = carry", "old_carry = carry\n        opaque(x)"
    )
    source = _rewrite(original, pipelined=True, pipeline_depth=4)
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(original))
    for depth in (False, True, 0, 1, 3, 8, "4", None):
        source = _rewrite(_source(), pipelined=True, pipeline_depth=cast("int", depth))
        assert "resident_state" not in source


@skipUnlessBackends(["cute"])
def test_depth_config_default_validation_and_inactive_repair() -> None:
    assert helion.Config().cute_reduction_pipeline_depth == 2
    assert KEY not in helion.Config()
    assert helion.Config(cute_reduction_pipeline_depth=4)[KEY] == 4
    bound, _args = _bind(64, 4096, kind="layer", dtype=torch.bfloat16)
    seed = _family(bound)[0]
    spec = bound.config_spec
    for depth in (False, True, 0, 1, 3, 8, "4", None):
        with pytest.raises(helion.exc.InvalidConfig, match="pipeline depth"):
            spec.normalize(deepcopy(seed.config) | {KEY: depth})
    # Public optional arguments use None to mean omitted; raw dictionaries
    # passed directly to strict normalization must contain actual integers.
    assert KEY not in spec.normalized_config(seed.config | {KEY: None})
    for schedule in ("scalar", "resident"):
        config = helion.Config.from_dict(
            seed.config | {"cute_reduction_schedule": schedule}
        )
        with pytest.raises(helion.exc.InvalidConfig, match="pipelined schedule"):
            spec.normalize(config)
        spec.normalize(config, _fix_invalid=True)
        assert config[KEY] == 2
    legacy = helion.Config.from_dict({k: v for k, v in seed.items() if k != KEY})
    assert KEY not in spec.normalized_config(legacy)
    with (
        patch.object(spec, "cute_resident_reduction_blocks", set()),
        pytest.raises(helion.exc.InvalidConfig, match="static serial-row sums"),
    ):
        spec.normalized_config(helion.Config.from_dict({KEY: 4}))


@skipUnlessBackends(["cute"])
def test_old_seed_prefix_default_and_owned_lists_are_preserved() -> None:
    bound, _args = _bind(64, 4096, kind="layer", dtype=torch.bfloat16)
    assert bound.host_function is not None
    spec = bound.config_spec
    family = _family(bound)
    assert [seed["cute_host_paired_sum"] for seed in family] == [
        "off",
        "mapped",
        "narrow",
    ]
    current = deepcopy(spec.compiler_seed_configs)
    promoted = deepcopy(spec.compiler_default_config)
    with bound.env:
        default = spec.default_config()
        with patch.object(
            CuteResidentReductionHeuristic,
            "pipeline_depth_seed_configs",
            return_value=[],
        ):
            old = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        assert spec.default_config() == default
        assert spec.compiler_default_config == promoted
    # The later local-tree overlay preserves the complete depth-family prefix.
    current = [
        seed
        for seed in current
        if not seed.config.get("cute_reduction_local_tree")
        and not seed.config.get("cute_reduction_row_schedule")
    ]
    old = [
        seed
        for seed in old
        if not seed.config.get("cute_reduction_local_tree")
        and not seed.config.get("cute_reduction_row_schedule")
    ]
    assert current == [*old, *family]
    assert all(KEY not in seed for seed in old)
    first = deepcopy(family[0])
    family[1].config["num_threads"][0] = 7
    assert family[0] == first and family[2].config["num_threads"][0] != 7


@skipUnlessBackends(["cute"])
def test_depth_survives_flat_random_biased_and_neighbor_routes() -> None:
    bound, _args = _bind(64, 4096)
    spec = bound.config_spec
    seed = _family(bound)[0]
    with bound.env:
        generation = ConfigGeneration(spec)
        flat, normalized = generation.strict_config_pair(seed)
        assert generation.unflatten(flat.copy()) == normalized
        schedule_index = generation._key_to_flat_indices["cute_reduction_schedule"][0][
            0
        ]
        depth_index = generation._key_to_flat_indices[KEY][0][0]
        for schedule in ("scalar", "resident", "pipelined"):
            trial = flat.copy()
            trial[schedule_index], trial[depth_index] = schedule, 4
            canonical, config = generation.canonicalize_flat(trial)
            assert config[KEY] == (4 if schedule == "pipelined" else 2)
            assert generation.unflatten(canonical.copy()) == config
        for seed_value in range(8):
            random.seed(seed_value)
            for trial in (
                generation.random_config(),
                generation.unflatten(generation.biased_random_flat()),
            ):
                flat, config = generation.canonicalize_flat(generation.flatten(trial))
                assert generation.unflatten(flat.copy()) == config
                assert config[KEY] in (2, 4)
                assert (
                    config[KEY] == 2 or config["cute_reduction_schedule"] == "pipelined"
                )
        neighbors = generation.coordinate_neighbor_projections(
            generation.flatten(normalized)
        )
        assert any(
            row.key == KEY and row.to_value == 2 and row.config is not None
            for row in neighbors
        )
        for depth in (2, 4):
            explicit = ConfigGeneration(
                spec, overrides={KEY: depth, "cute_reduction_schedule": "pipelined"}
            )
            flat, config = explicit.strict_config_pair(seed)
            assert config[KEY] == depth
            assert explicit.unflatten(flat.copy()) == config


@skipUnlessBackends(["cute"])
def test_full_lfbo_initial_population_receives_deep_pipeline_seeds() -> None:
    bound, args = _bind(64, 4096, kind="layer", dtype=torch.bfloat16)
    family = _family(bound)
    search = _search(bound, args)
    search._autotune_metrics = AutotuneMetrics()
    delivered = []

    class HeldBenchmark(Exception):
        pass

    def hold(members: Sequence[PopulationMember], *, desc: str) -> None:
        assert desc == "Initial population"
        delivered.extend(deepcopy(member.config) for member in members)
        raise HeldBenchmark

    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
        patch.object(search, "benchmark_population", side_effect=hold),
        pytest.raises(HeldBenchmark),
    ):
        random.seed(2026092811)
        search._autotune()
    assert len(delivered) == 100
    for seed in family:
        _flat, normalized = search.config_gen.strict_config_pair(seed)
        assert normalized in delivered and normalized[KEY] == 4


@pytest.mark.parametrize(
    "kind,dtype,static,rows,columns",
    [
        ("layer", torch.bfloat16, True, 4096, 4096),
        ("layer", torch.float16, False, 17, 512),
        ("rms", torch.float16, True, 17, 2048),
        ("rms", torch.float32, False, 17, 1024),
    ],
)
@skipUnlessBackends(["cute"])
def test_ordinary_codegen_preserves_host_abi_and_compute(
    kind: str, dtype: torch.dtype, static: bool, rows: int, columns: int
) -> None:
    bound, _args = _bind(rows, columns, kind=kind, static=static, dtype=dtype)
    family = _family(bound)
    assert family
    seed = family[-1]
    codes = [
        bound.to_code(helion.Config.from_dict(seed.config | {KEY: depth}))
        for depth in (2, 4)
    ]
    trees = [ast.parse(code) for code in codes]
    hosts = [
        [
            ast.dump(node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.decorator_list
        ]
        for tree in trees
    ]
    assert hosts[0] and hosts[0] == hosts[1]
    kernels = [
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
        )
        for tree in trees
    ]
    assert ast.dump(kernels[0].args) == ast.dump(kernels[1].args)
    loops = [
        next(
            node
            for node in kernel.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "resident_row_group"
        )
        for kernel in kernels
    ]
    assert ast.dump(ast.Module(body=loops[0].body[1:-1], type_ignores=[])) == ast.dump(
        ast.Module(body=loops[1].body[1:-1], type_ignores=[])
    )
    assert "cp_async_wait_group(1)" in codes[0]
    assert "cp_async_wait_group(3)" in codes[1]
    for code in codes:
        assert "cp_async_wait_group(0)" in code
        assert "block=(128, 1, 1)" in code
        assert "_cute_resident_copy_async" in code
    allocation_sizes = []
    for kernel in kernels:
        allocation_sizes.append(
            [
                ast.literal_eval(node.args[1])
                for node in ast.walk(kernel)
                if isinstance(node, ast.Call)
                and ast.unparse(node.func) == "cute.arch.alloc_smem"
            ]
        )
    assert allocation_sizes[1] == [2 * size for size in allocation_sizes[0]]


@pytest.mark.parametrize("columns", (32, 256, 513, 1025))
@skipUnlessBackends(["cute"])
def test_unmatched_feature_lattice_has_no_deeper_seed(columns: int) -> None:
    bound, _args = _bind(17, columns)
    assert not _family(bound)


@skipUnlessBackends(["cute"])
def test_disabling_heuristics_keeps_explicit_depth_support() -> None:
    bound, _args = _bind(17, 1024, disable_autotuner_heuristics=True)
    assert not bound.config_spec.compiler_seed_configs
    assert "cp_async_wait_group(3)" in bound.to_code(_family(bound)[0])
