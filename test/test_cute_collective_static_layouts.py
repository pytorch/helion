from __future__ import annotations

import ast
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from examples.matmul_split_k import matmul_split_k
import pytest
import torch

from test._cute_binding import _cpu_bind
from test.cute_population_contracts import checked_initial_population

import helion
from helion._compiler.generate_ast import GenerateAST
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.exc import InvalidConfig

KEY = "cute_collective_static_layouts"


def _body(source):
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node):
            node.body.pop(0)
    return ast.dump(module)


def _bound(m=32, k=4096, n=64, **settings):
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        **settings,
    )
    args = (
        torch.empty((m, k), dtype=torch.float16),
        torch.empty((k, n), dtype=torch.float16),
    )
    return _cpu_bind(kernel, args)


def _config():
    return helion.Config(
        block_sizes=[32, 32, 128],
        num_threads=[4, 32, 1],
        loop_orders=[[2, 0, 1]],
        cute_collective_mma=True,
        cute_collective_copy="async_cached",
        cute_collective_stages=2,
        split_k=32,
    )


@pytest.mark.parametrize("shape", ((32, 4096, 64), (37, 4123, 69)))
@skipUnlessBackends(["cute"])
def test_only_layout_guard_changes_on_complete_warp_lowering(shape):
    bound = _bound(*shape)
    config = _config()
    before = bound.to_code(config)
    assert "cute.gemm(" in before
    assert "return out.to(torch.float16)" in before
    config.config[KEY] = True
    after = bound.to_code(config)
    guard = (
        "    _helion_matmul_split_k._helion_cute_disable_bake_tensor_shapes = True\n"
    )
    assert before.count(guard) == 1
    # Removing the first host statement also restores its original docstring.
    assert _body(after) == _body(before.replace(guard, ""))
    config.config[KEY] = False
    assert bound.to_code(config) == before


def test_incomplete_or_mixed_lowerings_keep_runtime_layouts():
    codegen = GenerateAST.__new__(GenerateAST)
    codegen._cute_uses_matmul = False
    codegen._cute_matmul_declaration_count = 0
    codegen.cute_wrapper_plans = []
    state = SimpleNamespace(
        collective_mma_static_layouts=True, collective_mma_sites=[object()]
    )
    codegen.device_function = SimpleNamespace(config={KEY: True}, cute_state=state)
    codegen.cute_uses_matmul = True
    assert codegen._cute_has_complete_static_collectives()
    # A second matmul declaration must not inherit the first site's proof.
    codegen.cute_uses_matmul = True
    assert not codegen._cute_has_complete_static_collectives()
    state.collective_mma_sites.append(object())
    assert codegen._cute_has_complete_static_collectives()
    state.collective_mma_static_layouts = False
    assert not codegen._cute_has_complete_static_collectives()
    state.collective_mma_static_layouts = True
    codegen.cute_wrapper_plans = [{"kind": "unrelated"}]
    assert not codegen._cute_has_complete_static_collectives()


@skipUnlessBackends(["cute"])
def test_option_validation_and_full_search_seeds():
    bound = _bound()
    config = _config()
    for value in (1, 0, None, "true"):
        bad = deepcopy(config)
        bad.config[KEY] = value
        with pytest.raises(InvalidConfig, match=KEY):
            bound.config_spec.normalize(bad)
    bad = deepcopy(config)
    bad.config.update({KEY: True, "cute_collective_compute": "tcgen05"})
    with pytest.raises(InvalidConfig, match=KEY):
        bound.config_spec.normalize(bad)
    bound.config_spec.normalize(bad, _fix_invalid=True)
    assert KEY not in bad.config
    seeds = bound.config_spec.compiler_seed_configs
    normal = [seed for seed in seeds if not seed.config.get(KEY)]
    static = [seed for seed in seeds if seed.config.get(KEY)]
    assert normal and static
    assert seeds == normal + static
    for seed in static:
        control = deepcopy(seed)
        control.config.pop(KEY)
        assert control in normal
        assert seeds.index(control) < seeds.index(seed)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        pairs = generation.seed_flat_config_pairs()
        for flat, seed in pairs:
            if seed.config.get(KEY):
                assert generation.unflatten(flat) == seed


@skipUnlessBackends(["cute"])
def test_full_search_retains_old_population_and_transfers_static_witness():
    bound = _bound()
    args = (
        torch.empty((32, 4096), dtype=torch.float16),
        torch.empty((4096, 64), dtype=torch.float16),
    )
    normal = [
        seed for seed in bound.config_spec.compiler_seed_configs if not seed.get(KEY)
    ]
    with bound.env:
        search = PatternSearch(
            bound,
            args,
            initial_population=100,
            initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
        )
        with patch.object(bound.config_spec, "compiler_seed_configs", normal):
            old_base = search._generate_initial_population_base(
                generation=search.config_gen.initial_population_view()
            )
        population = checked_initial_population(search)
        assert population[:100] == old_base
        static = [
            search.config_gen.unflatten(flat)
            for flat in population
            if search.config_gen.unflatten(flat).get(KEY)
        ]
        assert static and all(
            seed in search._pinned_finalist_configs for seed in static
        )
        outcomes = [
            row
            for row in search.compiler_coverage_outcomes
            if row.mechanism == "cute.collective_static_layouts"
        ]
        assert len(outcomes) == 1 and outcomes[0].outcome == "added"
        delivered = []

        class HeldBenchmark(Exception):
            pass

        def hold(members, *, desc):
            assert desc == "Initial population"
            delivered.extend(member.config for member in members)
            raise HeldBenchmark

        with (
            patch.object(search, "_find_similar_cached_configs", return_value=[]),
            patch.object(search, "benchmark_population", side_effect=hold),
            pytest.raises(HeldBenchmark),
        ):
            search._autotune()
        assert all(seed in delivered for seed in static)


@pytest.mark.parametrize("disabled", (False, True))
@pytest.mark.parametrize("requested", (False, True))
@skipUnlessBackends(["cute"])
def test_explicit_override_and_disabled_seed_use(disabled, requested):
    bound = _bound(disable_autotuner_heuristics=disabled)
    args = (
        torch.empty((32, 4096), dtype=torch.float16),
        torch.empty((4096, 64), dtype=torch.float16),
    )
    groups = [
        group
        for group in bound.config_spec.compiler_coverage_groups
        if group.mechanism == "cute.collective_static_layouts"
    ]
    assert len(groups) == 1
    (witness,) = groups[0].witnesses
    assert witness.value is True
    assert witness.carrier.get("cute_collective_mma") is True
    assert witness.carrier.get("cute_collective_compute", "warp") == "warp"
    if disabled:
        assert bound.config_spec.compiler_seed_configs == []
        assert bound.config_spec.compiler_default_config is None
    with (
        bound.env,
        patch.object(
            bound.settings,
            "autotune_config_overrides",
            {
                KEY: requested,
                "cute_collective_mma": True,
                "cute_collective_compute": "warp",
            },
        ),
    ):
        search = PatternSearch(bound, args, initial_population=100)
        population = checked_initial_population(search)
        assert all(
            search.config_gen.unflatten(flat).get(KEY, False) is requested
            for flat in population
        )
        if disabled:
            assert len(population) == 100
            assert "compiler_coverage_outcomes" not in vars(search)
        else:
            admitted = any(
                row.mechanism == "cute.collective_static_layouts"
                and row.outcome in ("added", "already_present")
                for row in search.compiler_coverage_outcomes
            )
            assert admitted is requested


@skipUnlessBackends(["cute"])
def test_nonwarp_carriers_do_not_declare_static_coverage():
    nonwarp = _config()
    nonwarp.config["cute_collective_compute"] = "tcgen05"
    with patch(
        "helion._compiler.autotuner_heuristics.CuteCollectiveMatmulHeuristic.get_seed_configs",
        return_value=[nonwarp],
    ):
        bound = _bound(disable_autotuner_heuristics=True)
    assert not any(
        group.mechanism == "cute.collective_static_layouts"
        for group in bound.config_spec.compiler_coverage_groups
    )
