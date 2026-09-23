from __future__ import annotations

import ast
from copy import deepcopy
from typing import Any
from typing import cast

import pytest
import torch

from test.cute_population_contracts import checked_initial_population
from test.test_cute_split_k_cluster import _bind
from test.test_cute_split_k_cluster import _carrier
from test.test_cute_split_k_cluster import cpu_only as cpu_only
from test.test_cute_split_k_workspace import _embedded_sources

import helion
from helion._compiler.cute.split_k_cluster_config import CLUSTER8_K4
from helion._compiler.cute.split_k_cluster_config import LEGACY
from helion._compiler.cute.split_k_cluster_config import SCHEDULE_KEY
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch

FINALIZER_KEY = "cute_split_k_finalizer_warps"
FINALIZER_MECHANISM = "cute.split_k_finalizer"


def _finalizer_config(bound, warps: int = 4) -> helion.Config:
    config = deepcopy(_carrier(bound))
    assert config is not None
    config.config[FINALIZER_KEY] = warps
    return config


def _full_search(bound, args) -> PatternSearch:
    profile = get_effort_profile("full").pattern_search
    assert profile is not None
    return PatternSearch(
        bound,
        args,
        initial_population=profile.initial_population,
        initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
    )


@pytest.mark.parametrize("cluster", (False, True))
@pytest.mark.parametrize("bias", (False, True))
def test_one_warp_default_is_omitted_and_preserves_complete_source(cluster, bias):
    bound, args = _bind(bias=bias)
    config = (
        _carrier(bound)
        if cluster
        else helion.Config(
            block_sizes=[32, 32, 128], num_threads=[4, 32, 1], split_k=32
        )
    )
    assert config is not None and FINALIZER_KEY not in config
    explicit = deepcopy(config)
    explicit.config[FINALIZER_KEY] = 1
    assert bound.to_code(explicit) == bound.to_code(config)
    with bound.env:
        bound.config_spec.normalize(explicit)
        assert FINALIZER_KEY not in bound.config_spec._base_default_config()
        assert bound.config_spec.supports_config_key(FINALIZER_KEY)
        fragment = bound.config_spec._flat_fields()[FINALIZER_KEY]
        assert isinstance(fragment, EnumFragment)
        assert fragment.choices == (1, 4)
    assert FINALIZER_KEY not in explicit


@pytest.mark.parametrize("value", (None, False, True, 0, 2, 8, 1.0, 4.0, "4"))
def test_finalizer_mode_requires_a_typed_supported_integer(value):
    bound, args = _bind()
    config = _finalizer_config(bound)
    config.config[FINALIZER_KEY] = value
    with bound.env, pytest.raises(helion.exc.InvalidConfig):
        bound.config_spec.normalize(config)


@pytest.mark.parametrize("schedule", (None, LEGACY))
def test_four_warps_requires_explicit_cluster_schedule(schedule):
    bound, args = _bind()
    config = _finalizer_config(bound)
    if schedule is None:
        config.config.pop(SCHEDULE_KEY)
    else:
        config.config[SCHEDULE_KEY] = schedule
    with bound.env, pytest.raises(helion.exc.InvalidConfig):
        bound.config_spec.normalize(config)


@pytest.mark.parametrize(
    "change",
    (
        {"block_sizes": [32, 16, 128]},
        {"num_threads": [4, 8, 1]},
        {"split_k": 32},
        {"cute_split_k_workspace": True},
        {"cute_collective_mma": True},
    ),
)
def test_invalid_cluster_repair_removes_dependent_finalizer(change):
    bound, args = _bind()
    config = _finalizer_config(bound)
    config.config.update(change)
    with bound.env:
        with pytest.raises(helion.exc.InvalidConfig):
            bound.config_spec.normalize(deepcopy(config))
        bound.config_spec.normalize(config, _fix_invalid=True)
        assert SCHEDULE_KEY not in config
        assert FINALIZER_KEY not in config
        bound.config_spec.normalize(config)


@pytest.mark.parametrize("shape", ((31, 4096, 64), (32, 2048, 64)))
def test_finalizer_does_not_widen_cluster_proof_domain(shape):
    bound, args = _bind(shape)
    assert _carrier(bound) is None
    config = helion.Config(block_sizes=[16, 16, 128], num_threads=[4, 8, 4], split_k=8)
    config.config.update({SCHEDULE_KEY: CLUSTER8_K4, FINALIZER_KEY: 4})
    with bound.env:
        assert FINALIZER_KEY not in bound.config_spec._flat_fields()
        with pytest.raises(helion.exc.InvalidConfig):
            bound.config_spec.normalize(config)


@pytest.mark.parametrize("bias", (False, True))
def test_full_population_keeps_cluster_and_dependent_finalizer_witnesses(bias):
    bound, args = _bind(bias=bias)
    seeds = bound.config_spec.compiler_seed_configs
    carrier = _carrier(bound)
    assert carrier is not None
    assert seeds[-1].config == carrier.config | {FINALIZER_KEY: 4}
    assert all(FINALIZER_KEY not in seed for seed in seeds[:-1])
    assert carrier in seeds[:-1]
    groups = bound.config_spec.compiler_coverage_groups
    group = next(group for group in groups if group.key == FINALIZER_KEY)
    assert group.mechanism == FINALIZER_MECHANISM
    assert group.domain == (1, 4) and group.legacy == 1
    assert [
        (dependency.mechanism, dependency.key, dependency.value)
        for dependency in group.dependencies
    ] == [("cute.split_k_cluster", SCHEDULE_KEY, CLUSTER8_K4)]
    assert group.witnesses[0].carrier == carrier
    with bound.env:
        search = _full_search(bound, args)
        rows = checked_initial_population(search)
        configs = [search.config_gen.unflatten(row) for row in rows]
    for mechanism, warps in (("cute.split_k_cluster", 1), (FINALIZER_MECHANISM, 4)):
        outcome = next(
            outcome
            for outcome in search.compiler_coverage_outcomes
            if outcome.mechanism == mechanism
        )
        assert outcome.outcome in ("added", "already_present")
        assert outcome.effective is not None
        assert outcome.effective[SCHEDULE_KEY] == CLUSTER8_K4
        assert outcome.effective.get(FINALIZER_KEY, 1) == warps
        assert outcome.effective in configs
        assert outcome.effective in search._pinned_finalist_configs


@pytest.mark.parametrize("overrides", ({FINALIZER_KEY: 1}, {SCHEDULE_KEY: LEGACY}))
def test_full_population_honors_finalizer_and_schedule_overrides(overrides):
    bound, args = _bind(autotune_config_overrides=overrides)
    with bound.env:
        search = _full_search(bound, args)
        rows = checked_initial_population(search)
        configs = [search.config_gen.unflatten(row) for row in rows]
    assert all(config.get(FINALIZER_KEY, 1) == 1 for config in configs)
    cluster_configs = [
        config for config in configs if config.get(SCHEDULE_KEY) == CLUSTER8_K4
    ]
    assert bool(cluster_configs) is (SCHEDULE_KEY not in overrides)
    assert not any(
        outcome.mechanism == FINALIZER_MECHANISM
        and outcome.outcome in ("added", "already_present")
        for outcome in search.compiler_coverage_outcomes
    )


def test_explicit_four_warp_override_survives_disabled_heuristics():
    bound, args = _bind(
        disable_heuristics=True,
        autotune_config_overrides={SCHEDULE_KEY: CLUSTER8_K4, FINALIZER_KEY: 4},
    )
    assert not bound.config_spec.compiler_seed_configs
    assert any(
        group.key == FINALIZER_KEY
        for group in bound.config_spec.compiler_coverage_groups
    )
    requested = _carrier(bound)
    assert requested is not None and FINALIZER_KEY not in requested
    with bound.env:
        search = _full_search(bound, args)
        assert not search.config_gen.compiler_coverage_enabled
        flat, effective = search.config_gen.strict_config_pair(requested)
    assert effective[SCHEDULE_KEY] == CLUSTER8_K4
    assert effective[FINALIZER_KEY] == 4
    assert effective.block_sizes == [16, 16, 128]
    assert effective["split_k"] == 8
    assert "warp * 4" in _embedded_sources(bound.to_code(effective))[0]


@pytest.mark.parametrize("shape", ((32, 4096, 64), (48, 8192, 80)))
@pytest.mark.parametrize("bias", (False, True))
def test_four_warp_allocating_host_and_ordered_finalizers(shape, bias):
    bound, args = _bind(shape, bias=bias)
    config = _finalizer_config(bound)
    bound.set_config(config)
    assert bound._run is not None
    host = cast("Any", bound._run)
    metadata = host._helion_cute_split_k_schedule
    assert metadata["finalizer_warps"] == 4
    assert metadata["tile"] == (16, 8, 128)
    assert metadata["block"] == (32, 4, 1)
    assert metadata["cluster"] == (8, 1, 1)
    assert metadata["waves"] == shape[1] // 4096
    assert metadata["bias_partition"] == (0 if bias else None)
    calls = []

    def capture(kernel, grid, *values, **kwargs):
        calls.append((kernel, grid, values, kwargs))

    outputs = [host(*args, _launcher=capture) for _ in range(2)]
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert len(calls) == 2
    for index, (kernel, grid, values, kwargs) in enumerate(calls):
        assert kernel._helion_cute_cluster_shape == (8, 1, 1)
        assert grid == ((shape[0] // 16) * (shape[2] // 8) * 8,)
        assert kwargs == {"block": (32, 4, 1)}
        assert values[0] is args[0] and values[1] is args[1]
        assert values[2] is outputs[index]
        assert len(values) == (4 if bias else 3)
        assert outputs[index].shape == (shape[0], shape[2])
        assert outputs[index].dtype is torch.float16
    source = _embedded_sources(bound.to_code(config))[0]
    assert "atomic" not in source and "torch.zeros" not in source
    assert "m = lane // 8 + warp * 4" in source
    assert "n = lane % 8" in source
    assert "warp == 0" not in source
    assert source.count("cluster_wait()") == 2
    assert "cluster_arrive_relaxed()" in source
    tree = ast.parse(source)
    kernel_ast = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    local = next(
        node
        for node in kernel_ast.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "local_peer"
    )
    assert ast.unparse(local.iter) == "cutlass.range_constexpr(1, 4)"
    assert ast.unparse(local.body[0]) == (
        "value = value + cutlass.Float32(c[m, n, local_peer])"
    )
    final_guard = next(
        node
        for node in kernel_ast.body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.For)
            and isinstance(child.target, ast.Name)
            and child.target.id == "peer"
            for child in node.body
        )
    )
    assert ast.unparse(final_guard.test) == "rank == 0"
    remote = next(node for node in final_guard.body if isinstance(node, ast.For))
    assert ast.unparse(remote.iter) == "cutlass.range_constexpr(1, 8)"
    assert ast.unparse(remote.body[0]) == (
        "value = value + load_shared_remote_f32(cp + element, cutlass.Int32(peer))"
    )
    if bias:
        assert source.index("value + cutlass.Float32((bias.iterator") < source.index(
            "cute.arch.cluster_arrive()"
        )
