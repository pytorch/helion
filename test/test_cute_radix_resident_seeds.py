from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import cpu_codegen_target
from test.test_cute_radix_selection import _payload_selection

import helion
from helion._compiler.autotuner_heuristics.cute_radix_selection import (
    CuteRadixSelectionHeuristic,
)
from helion.autotuner import LFBOTreeSearch
from helion.autotuner import LLMGuidedSearch
from helion.autotuner import LLMSeededLFBOTreeSearch
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.llm.prompting import build_compiler_analysis_section
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch


@pytest.fixture
def bound(monkeypatch):
    monkeypatch.delenv("HELION_AUTOTUNE_EFFORT", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_CONFIG_OVERRIDES", raising=False)
    assert (
        Path(helion.__file__).resolve().parents[1]
        == Path(__file__).resolve().parents[1]
    )
    with cpu_codegen_target((10, 3)), _forbid_cuda_work():
        args = (
            torch.empty((4, 4096), dtype=torch.bfloat16),
            torch.full((4,), 4096, dtype=torch.int32),
            512,
        )
        with patch.object(
            CuteRadixSelectionHeuristic, "get_seed_configs", return_value=[]
        ):
            original = _payload_selection._bind_isolated(args)
        current = _payload_selection._bind_isolated(args)
        yield current, original, args


def _seeds(kernel):
    generation = ConfigGeneration(kernel.config_spec)
    return [
        generation.canonicalize_flat(generation.flatten(config))[1]
        for config in CuteRadixSelectionHeuristic.get_seed_configs(
            kernel.env, kernel.host_function.device_ir
        )
    ]


def _population(search):
    return [
        search.config_gen.unflatten(flat)
        for flat in search._generate_initial_population_flat()
    ]


def test_binding_adds_bounded_radix_seeds_and_preserves_reference(bound) -> None:
    kernel, original, args = bound
    spec, old = kernel.config_spec, original.config_spec
    assert CuteRadixSelectionHeuristic.name in spec.autotuner_heuristics
    assert spec.compiler_default_config == old.compiler_default_config
    assert spec.autotune_reference_config() == old.autotune_reference_config()
    assert all(
        config in spec.compiler_seed_configs for config in old.compiler_seed_configs
    )
    with kernel.env:
        seeds = _seeds(kernel)
        normalized = {
            config for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
        }
    assert len(seeds) == 6
    assert all(config in normalized for config in seeds)
    assert {config.get("cute_radix_packet_bits") for config in seeds} == {64, 128, 256}
    assert not CuteRadixSelectionHeuristic.should_promote(kernel.env)
    assert "input_tensor_metadata" in kernel.env.compiler_fact_specialization_facts


@pytest.mark.parametrize("search_cls", [PatternSearch, LFBOTreeSearch])
def test_standard_population_and_finalists_include_resident_radix_seeds(
    bound, search_cls
) -> None:
    kernel, original, args = bound
    sizes = []
    for item in (original, kernel):
        with item.env:
            search = search_cls(
                item,
                args,
                initial_population=100,
                initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
            )
            assert len(search.config_gen.random_population_flat(100)) == 100
            population = _population(search)
            sizes.append(len(population))
            if item is kernel:
                assert all(
                    config in population and config in search._pinned_finalist_configs
                    for config in _seeds(kernel)
                )
    assert sizes[0] == sizes[1]


def test_llm_starts_with_reference_and_hybrid_uses_ordinary_radix_seeds(bound) -> None:
    kernel, original, args = bound
    before = list(kernel.config_spec.compiler_seed_configs)
    with kernel.env:
        guided = LLMGuidedSearch(kernel, args, initial_random_configs=0)
        assert guided._build_seed_configs() == [
            kernel.config_spec.autotune_reference_config()
        ]
        assert CuteRadixSelectionHeuristic.name in build_compiler_analysis_section(
            kernel.config_spec
        )
        hybrid = LLMSeededLFBOTreeSearch(kernel, args, llm_initial_random_configs=0)
        assert (
            hybrid._make_llm_search()._build_seed_configs()
            == guided._build_seed_configs()
        )
        second_stage = hybrid._make_second_stage_search(seeded=True)
        hybrid._inject_seed_into_second_stage(
            second_stage, kernel.config_spec.autotune_reference_config()
        )
        population = _population(second_stage)
        assert all(
            config in population and config in second_stage._pinned_finalist_configs
            for config in _seeds(kernel)
        )
    assert kernel.config_spec.compiler_seed_configs == before


def test_radix_seeds_emit_all_six_legal_resident_packet_and_output_plans(bound) -> None:
    kernel, original, args = bound
    with kernel.env:
        seeds = _seeds(kernel)
    sources = [kernel.to_code(config) for config in seeds]
    assert len(set(sources)) == 6
    assert all("resident_keys = cute.make_rmem_tensor" in source for source in sources)
    for config in seeds:
        assert config.get("cute_selection_algorithm") == "radix"
        assert all(
            config.get(key) is True
            for key in (
                "cute_radix_resident_input",
                "cute_radix_boundary_early_exit",
                "cute_radix_defer_output",
                "cute_radix_valid_domain",
                "cute_radix_ticket_stores",
            )
        )


def test_radix_seeds_respect_global_disable_and_require_proved_graph(bound) -> None:
    kernel, original, args = bound
    disabled = helion.kernel(
        _payload_selection.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )._bind_isolated(args)
    assert disabled.config_spec.compiler_seed_configs == []
    assert disabled.config_spec.autotuner_heuristics == []
    # Storage/metadata proof still runs when optional seeds are disabled.
    assert disabled.config_spec.cute_radix_selection_graph is not None
    with patch.object(kernel.config_spec, "cute_radix_selection_graph", None):
        assert not CuteRadixSelectionHeuristic.is_eligible(
            kernel.env, kernel.host_function.device_ir
        )
        assert (
            CuteRadixSelectionHeuristic.get_seed_configs(
                kernel.env, kernel.host_function.device_ir
            )
            == []
        )
