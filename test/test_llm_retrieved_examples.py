from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from helion.autotuner.candidate_budget import AttemptBudget
from helion.autotuner.llm.prompting import build_retrieved_examples_section
from helion.autotuner.llm_search import LLMGuidedSearch
from helion.runtime.config import Config


def test_empty_examples_is_noop():
    assert build_retrieved_examples_section([]) == ""


def test_section_includes_names_and_minified_config():
    examples = [
        {
            "kernel_name": "add",
            "input_shapes": "[(1024, 1024)]",
            "dtypes": "['torch.float32']",
            "config": {"num_warps": 8},
        }
    ]
    section = build_retrieved_examples_section(examples)
    assert "Retrieved Neighbor Examples" in section
    assert "add" in section
    assert '{"num_warps":8}' in section  # minified, embedded in the prompt


def test_missing_config_is_marked_unavailable():
    section = build_retrieved_examples_section([{"kernel_name": "k"}])
    assert "(config unavailable)" in section


def test_guided_search_passes_retrieved_examples_to_initial_prompt():
    examples = [{"kernel_name": "add", "config": {"num_warps": 8}}]
    search = LLMGuidedSearch.__new__(LLMGuidedSearch)
    search.kernel = object()
    search.args = ()
    search.config_spec = object()
    search.configs_per_round = 4
    search.settings = SimpleNamespace(autotune_compile_timeout=30)

    search.set_retrieved_examples(examples)
    with patch(
        "helion.autotuner.llm_search.build_initial_prompt", return_value="prompt"
    ) as build:
        assert search._build_initial_prompt() == "prompt"

    assert build.call_args.kwargs["retrieved_examples"] == examples


def test_best_available_configs_become_llm_seed_and_prompt_evidence():
    class ConfigGeneration:
        def flatten(self, config: Config) -> list[int]:
            return [int(config.config["value"])]

        def unflatten(self, flat: list[int]) -> Config:
            return Config(value=flat[0])

        def random_flat(self) -> list[int]:
            return [99]

    search = LLMGuidedSearch.__new__(LLMGuidedSearch)
    search.settings = SimpleNamespace(
        autotune_best_available_read=True,
        autotune_best_available_max_configs=4,
    )
    search.config_gen = ConfigGeneration()
    search.config_spec = SimpleNamespace(default_config=lambda: Config(value=0))
    search.initial_random_configs = 2
    search._fixed_initial_population_flat = None
    search._candidate_attempt_budget = AttemptBudget(4)
    search._candidate_attempt_configs = set()
    search._candidate_attempt_categories = {}
    search._candidate_sources = {}
    search._candidate_initial_population_open = True
    search._precounted_initial_population_configs = set()
    search._retrieved_examples = [{"kernel_name": "qwen", "config": {"value": 1}}]
    search._find_similar_cached_configs = lambda limit: [
        SimpleNamespace(to_mutable_flat_config=lambda: [2])
    ]

    search._prepare_best_available_initial_evidence()

    assert search._build_seed_configs() == [
        Config(value=2),
        Config(value=0),
        Config(value=99),
    ]
    assert search._retrieved_examples == [
        {"kernel_name": "qwen", "config": {"value": 1}},
        {"kernel_name": "best_available", "config": {"value": 2}},
    ]
    assert search._candidate_sources[Config(value=2)] == "best_available"
