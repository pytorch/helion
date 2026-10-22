from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import cpu_codegen_target
from test._cute_dense_tcgen05 import scaled_mm_column_major

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_specialization_facts
from helion._compiler.autotuner_heuristics.cute_tcgen05_fp8_acc1 import (
    CuteTcgen05Fp8Acc1Heuristic,
)
from helion.autotuner import LFBOTreeSearch
from helion.autotuner import LLMGuidedSearch
from helion.autotuner import LLMSeededLFBOTreeSearch
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.llm.prompting import build_compiler_analysis_section
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch


def _args(dtype: torch.dtype = torch.float8_e4m3fn) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((2048, 4096), dtype=dtype),
        torch.empty((256, 4096), dtype=dtype).T,
        torch.empty((), dtype=torch.float32),
        torch.empty(2048, dtype=torch.float32),
    )


@pytest.fixture
def bound(monkeypatch):
    monkeypatch.delenv("HELION_AUTOTUNE_EFFORT", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_CONFIG_OVERRIDES", raising=False)
    assert (
        Path(helion.__file__).resolve().parents[1]
        == Path(__file__).resolve().parents[1]
    )
    with cpu_codegen_target((10, 3)), _forbid_cuda_work():
        args = _args()
        with patch.object(
            CuteTcgen05Fp8Acc1Heuristic, "get_seed_configs", return_value=[]
        ):
            original = scaled_mm_column_major._bind_isolated(args)
        current = scaled_mm_column_major._bind_isolated(args)
        yield current, original, args


def _seeds(kernel):
    generation = ConfigGeneration(kernel.config_spec)
    return [
        generation.canonicalize_flat(generation.flatten(config))[1]
        for config in CuteTcgen05Fp8Acc1Heuristic.get_seed_configs(
            kernel.env, kernel.host_function.device_ir
        )
    ]


def _population(search):
    return [
        search.config_gen.unflatten(flat)
        for flat in search._generate_initial_population_flat()
    ]


def test_binding_adds_bounded_acc1_seeds_without_changing_default(bound) -> None:
    kernel, original, args = bound
    spec, old = kernel.config_spec, original.config_spec
    assert CuteTcgen05Fp8Acc1Heuristic.name in spec.autotuner_heuristics
    assert spec.compiler_default_config == old.compiler_default_config
    assert spec.autotune_reference_config() == old.autotune_reference_config()
    assert [
        config
        for config in spec.compiler_seed_configs
        if config in old.compiler_seed_configs
    ] == old.compiler_seed_configs
    with kernel.env:
        seeds = _seeds(kernel)
        normalized = {
            config for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
        }
        assert 0 < len(seeds) <= 12
        assert all(config in normalized for config in seeds)
        assert all(
            config.get("tcgen05_cta_group") == "two"
            and config.get("tcgen05_acc_stages") == 1
            for config in seeds
        )
    assert "device_num_sm" in compiler_seed_specialization_facts(
        "cute", spec.autotuner_heuristics
    )
    assert not CuteTcgen05Fp8Acc1Heuristic.should_promote(kernel.env)


@pytest.mark.parametrize("search_cls", [PatternSearch, LFBOTreeSearch])
def test_standard_population_and_finalists_include_acc1_seeds(
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
    # Existing compiler coverage is additive to the nominal population. Adding
    # ordinary ACC1 seeds does not enlarge that coverage or the requested budget.
    assert sizes[0] == sizes[1]


def test_llm_still_starts_with_reference_and_hybrid_consumes_native_acc1_seeds(
    bound,
) -> None:
    kernel, original, args = bound
    before = list(kernel.config_spec.compiler_seed_configs)
    with kernel.env:
        guided = LLMGuidedSearch(kernel, args, initial_random_configs=0)
        assert guided._build_seed_configs() == [
            kernel.config_spec.autotune_reference_config()
        ]
        assert CuteTcgen05Fp8Acc1Heuristic.name in build_compiler_analysis_section(
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


def test_acc1_seed_uses_deepest_legal_paired_ring_in_generated_source(bound) -> None:
    kernel, original, args = bound
    with kernel.env:
        seed = next(
            config for config in _seeds(kernel) if config.block_sizes == [128, 64, 256]
        )
        assert seed.get("tcgen05_ab_stages") == 8
        assert (
            kernel.config_spec._cute_tcgen05_config._paired_pipeline_stage_limit(
                seed.config
            )
            == 8
        )
    source = kernel.to_code(seed)
    assert "CtaGroup.TWO" in source and "num_stages=8" in source


def test_acc1_heuristic_respects_global_disable_and_declines_bf16(bound) -> None:
    kernel, original, args = bound
    disabled = helion.kernel(
        scaled_mm_column_major.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )._bind_isolated(args)
    assert disabled.config_spec.compiler_seed_configs == []
    assert disabled.config_spec.autotuner_heuristics == []
    bf16 = scaled_mm_column_major._bind_isolated(_args(torch.bfloat16))
    assert CuteTcgen05Fp8Acc1Heuristic.name not in bf16.config_spec.autotuner_heuristics
    assert not CuteTcgen05Fp8Acc1Heuristic.is_eligible(
        bf16.env, bf16.host_function.device_ir
    )
