from __future__ import annotations

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import cpu_codegen_target
from test._cute_dense_tcgen05 import scaled_mm_column_major
from test.test_cute_radix_selection import _payload_selection

from helion.autotuner import LFBOTreeSearch
from helion.autotuner import LLMGuidedSearch
from helion.autotuner import LLMSeededLFBOTreeSearch
from helion.autotuner import LLMSeededSearch
from helion.autotuner import cute_witness_seeds
from helion.autotuner.cute_witnesses import register_cute_witnesses
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.runtime.config import Config


@pytest.fixture(params=["acc1", "radix"])
def bound(request, monkeypatch):
    monkeypatch.delenv("HELION_AUTOTUNE_EFFORT", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_CONFIG_OVERRIDES", raising=False)
    # The native option must work without the experiment adapter's env flag.
    monkeypatch.delenv("HELION_GRAPH_AUTOTUNE_ACC1_SEEDS", raising=False)
    with cpu_codegen_target((10, 3)), _forbid_cuda_work():
        if request.param == "acc1":
            args = (
                torch.empty((2048, 4096), dtype=torch.float8_e4m3fn),
                torch.empty((256, 4096), dtype=torch.float8_e4m3fn).T,
                torch.empty((), dtype=torch.float32),
                torch.empty(2048, dtype=torch.float32),
            )
            kernel = scaled_mm_column_major._bind_isolated(args)
        else:
            args = (
                torch.empty((4, 4096), dtype=torch.bfloat16),
                torch.full((4,), 4096, dtype=torch.int32),
                512,
            )
            kernel = _payload_selection._bind_isolated(args)
        yield kernel, args, request.param


def _population(search):
    return [
        search.config_gen.unflatten(flat)
        for flat in search._generate_initial_population_flat()
    ]


def test_native_lfbo_context_registers_bounded_witnesses_in_population_and_finalists(
    bound,
):
    kernel, args, family = bound
    spec = kernel.config_spec
    original, default = list(spec.compiler_seed_configs), spec.compiler_default_config
    with kernel.env:
        baseline = PatternSearch(
            kernel,
            args,
            initial_population=100,
            initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
        )
        baseline_count = len(_population(baseline))
        before = {
            config for _flat, config in baseline.config_gen.seed_flat_config_pairs()
        }
        with cute_witness_seeds() as records:
            search = LFBOTreeSearch(
                kernel,
                args,
                initial_population=100,
                initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
            )
        population = _population(search)
        normalized = {
            config for _flat, config in search.config_gen.seed_flat_config_pairs()
        }
    assert len(records) == 1
    record = records[0]
    added = record[f"added_{family}_witnesses"]
    assert 0 < added <= (12 if family == "acc1" else 6)
    assert record["total_compiler_seeds"] <= 83
    assert record["minimum_random_slots_for_population_100"] >= 16
    assert len(population) == baseline_count
    assert spec.compiler_default_config == default
    assert all(config in spec.compiler_seed_configs for config in original)
    assert before <= normalized
    witnesses = [Config.from_dict(value) for value in record["witnesses"]]
    assert all(
        config in population and config in search._pinned_finalist_configs
        for config in witnesses
    )
    if family == "acc1":
        witness = next(
            config for config in witnesses if config.block_sizes == [128, 64, 256]
        )
        assert witness.get("tcgen05_cta_group") == "two"
        assert witness.get("tcgen05_acc_stages") == 1
        assert witness.get("tcgen05_ab_stages") == 8
        assert (
            spec._cute_tcgen05_config._paired_pipeline_stage_limit(witness.config) == 8
        )
        source = kernel.to_code(witness)
        assert "CtaGroup.TWO" in source and "num_stages=8" in source
    else:
        assert added == 6
        sources = [kernel.to_code(config) for config in witnesses]
        assert len(set(sources)) == 6
        assert all(
            "resident_keys = cute.make_rmem_tensor" in source for source in sources
        )
        for config in witnesses:
            assert config.get("cute_radix_packet_bits") in (64, 128, 256)
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


def test_native_hybrid_context_preserves_single_llm_default_and_handoff_inclusion(
    bound,
):
    kernel, args, family = bound
    with kernel.env, cute_witness_seeds() as records:
        hybrid = LLMSeededLFBOTreeSearch(kernel, args)
        guided = hybrid._make_llm_search()
        guided.initial_random_configs = 0
        assert guided._build_seed_configs() == [
            kernel.config_spec.autotune_reference_config()
        ]
        second_stage = hybrid._make_second_stage_search(seeded=True)
        hybrid._inject_seed_into_second_stage(
            second_stage, kernel.config_spec.autotune_reference_config()
        )
        population = _population(second_stage)
    assert records[0]["method"] == "llm_lfbo"
    assert records[0][f"added_{family}_witnesses"] > 0
    witnesses = [Config.from_dict(value) for value in records[0]["witnesses"]]
    assert all(
        config in population and config in second_stage._pinned_finalist_configs
        for config in witnesses
    )
    assert records[-1][f"added_{family}_witnesses"] == 0


def test_pure_llm_and_non_lfbo_hybrid_do_not_inject_witnesses(bound):
    kernel, args, family = bound
    original = list(kernel.config_spec.compiler_seed_configs)
    with kernel.env, cute_witness_seeds() as records:
        guided = LLMGuidedSearch(kernel, args, initial_random_configs=0)
        assert guided._build_seed_configs() == [
            kernel.config_spec.autotune_reference_config()
        ]
        LLMSeededSearch(kernel, args, second_stage_algorithm="PatternSearch")
    assert records == []
    assert kernel.config_spec.compiler_seed_configs == original
    record = register_cute_witnesses(kernel, method="llm")
    assert record[f"added_{family}_witnesses"] == 0
    assert kernel.config_spec.compiler_seed_configs == original


def test_registration_is_disabled_without_context_and_keeps_exhausted_seed_budget(
    bound,
):
    kernel, args, family = bound
    original = list(kernel.config_spec.compiler_seed_configs)
    with kernel.env:
        LFBOTreeSearch(kernel, args)
    assert kernel.config_spec.compiler_seed_configs == original
    assert (
        register_cute_witnesses(kernel, method="lfbo", enabled=False)[
            f"added_{family}_witnesses"
        ]
        == 0
    )
    # Every original entry remains available even when it already exceeds 83.
    seed = original[0] if original else kernel.config_spec.autotune_reference_config()
    kernel.config_spec.compiler_seed_configs = [seed] * 84
    record = register_cute_witnesses(kernel, method="lfbo")
    assert record[f"added_{family}_witnesses"] == 0
    assert kernel.config_spec.compiler_seed_configs == [seed] * 84
