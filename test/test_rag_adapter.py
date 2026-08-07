from __future__ import annotations

import hashlib
import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from helion import exc
from helion._utils import counters
from helion.autotuner.base_cache import AutotuneCacheBase
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.candidate_budget import AttemptBudget
from helion.autotuner.candidate_budget import AttemptCategory
from helion.autotuner.llm.transport import ProviderMetadata
from helion.autotuner.llm.transport import TokenUsage
from helion.autotuner.llm_search import LLMGuidedSearch
from helion.autotuner.rag.adapter import _LiveDecisionRunner
from helion.autotuner.rag.adapter import apply_retrieval_control
from helion.autotuner.rag.adapter import autotune_with_rag
from helion.autotuner.rag.adapter import build_workload_descriptor
from helion.autotuner.rag.adapter import convert_lookup_result
from helion.autotuner.rag.adapter import emit_attempt_event
from helion.autotuner.rag.adapter import resolve_driver_identity
from helion.autotuner.rag.adapter import resolve_tuner_mode
from helion.autotuner.rag.adapter import retrieve_evidence
from helion.autotuner.rag.instrumentation import InstrumentationCollector
from helion.autotuner.rag.policy_config import FrozenRetrievalPolicy
from helion.autotuner.rag.seeding import PopulationUnderfilled
from helion.autotuner.rag.types import BaselineSearch
from helion.autotuner.rag.types import ContextualSearch
from helion.autotuner.rag.types import ExecutionOutcome
from helion.autotuner.rag.types import LookupTier
from helion.autotuner.rag.types import Phase
from helion.autotuner.rag.types import PhaseTimingEvent
from helion.autotuner.rag.types import RetrievalEvidence
from helion.autotuner.rag.types import RetrievalSeededSearch
from helion.autotuner.rag.types import RetrievedNeighbor
from helion.autotuner.rag.types import TunerMode
from helion.autotuner.rag.types import WorkloadDescriptor
from helion.runtime.config import Config

# These exercise the adapter's seam onto the standalone helion_rag bundle
# (reranking and the campaign event schema). CI installs helion only, so the
# bundle is absent there and these skip rather than fail.
requires_helion_rag = pytest.mark.skipif(
    importlib.util.find_spec("helion_rag") is None,
    reason="requires the helion_rag bundle under scripts/helion_rag",
)


class _KernelDefinition:
    name = "add"

    def kernel_source(self) -> str:
        return "def add(x, y):\n    return x + y\n"

    def _base_specialization_key(self, args):
        return ("specialized", args[2])


class _ConfigSpec:
    def structural_fingerprint_hash(self, *, advanced_controls_files=None) -> str:
        assert advanced_controls_files == ["controls.txt"]
        return "fingerprint"


class _IdentityConfigGeneration:
    def flatten(self, config: Config) -> list[dict[str, object]]:
        return [dict(config.config)]

    def unflatten(self, flat: list[dict[str, object]]) -> Config:
        return Config.from_dict(flat[0])


def _search() -> SimpleNamespace:
    return SimpleNamespace(
        kernel=SimpleNamespace(
            kernel=_KernelDefinition(),
            env=SimpleNamespace(backend=SimpleNamespace(name="triton")),
        ),
        args=(torch.empty(2, 3), torch.empty(2, 3, dtype=torch.float16), 7),
        config_spec=_ConfigSpec(),
        settings=SimpleNamespace(autotune_search_acf=["controls.txt"]),
    )


def test_build_workload_descriptor_uses_live_kernel_identity() -> None:
    descriptor = build_workload_descriptor(_search())

    assert descriptor.kernel_name == "add"
    assert descriptor.kernel_source.startswith("def add")
    assert descriptor.input_shapes == "[(2, 3), (2, 3)]"
    assert descriptor.dtypes == "['torch.float32', 'torch.float16']"
    assert descriptor.hardware == "cpu"
    assert descriptor.backend == "triton"
    assert descriptor.structural_fingerprint_hash == "fingerprint"
    assert descriptor.specialization_key == "('specialized', 7)"


def test_driver_identity_uses_the_resolved_cuda_device() -> None:
    completed = SimpleNamespace(stdout="580.126.09\n")

    with patch(
        "helion.autotuner.rag.adapter.subprocess.run", return_value=completed
    ) as run:
        assert resolve_driver_identity(torch.device("cuda:2")) == "580.126.09"

    assert "--id=2" in run.call_args.args[0]


@requires_helion_rag
def test_convert_lookup_reranks_pool_and_uses_fastest_compatible_config() -> None:
    descriptor = build_workload_descriptor(_search())
    neighbors = []
    for i, dim in enumerate((1024, 3, 4, 8, 16, 32, 64, 128)):
        neighbors.append(
            {
                "kernel_name": f"k{i}",
                "input_shapes": f"[(2, {dim}), (2, {dim})]",
                "dtypes": "['torch.float32', 'torch.float16']",
                "score": 1.0 - i * 0.01,
                "ref": {"run_id": f"run-{i}"},
                "top_n": [
                    {"config_id": f"bad-{i}", "config": {"invalid": True}},
                    {
                        "config_id": f"good-{i}",
                        "config": {"block_sizes": [dim]},
                    },
                ],
            }
        )

    def normalize(raw):
        if raw.get("invalid"):
            raise ValueError("not compatible")
        return Config.from_dict(raw)

    evidence = convert_lookup_result(
        {
            "tier": 1,
            "family": "h100",
            "neighbors": neighbors,
            "artifact_identity": {
                "index_id": "000001",
                "manifest_id": "manifest-hash",
                "corpus_id": "corpus-hash",
            },
        },
        descriptor,
        normalize_config=normalize,
    )

    assert evidence.tier is LookupTier.SEMANTIC
    assert evidence.artifact_identity == {
        "index_id": "000001",
        "manifest_id": "manifest-hash",
        "corpus_id": "corpus-hash",
    }
    assert len(evidence.neighbors) == 5
    assert evidence.neighbors[0].kernel_name == "k1"
    assert evidence.neighbors[0].config == {"block_sizes": [3]}
    assert evidence.neighbors[0].ref is not None
    assert evidence.neighbors[0].ref["config_id"] == "good-1"
    assert evidence.neighbors[0].ref["rank_before_rerank"] == 1
    assert evidence.neighbors[0].ref["rank_after_rerank"] == 0


def test_convert_lookup_denies_historical_tier0_reuse() -> None:
    evidence = convert_lookup_result(
        {
            "tier": 0,
            "family": "h100",
            "best_config": {"num_warps": 8},
            "tier0_identity_combo": "S3-historical",
            "tier0_collision_count": 2,
        },
        build_workload_descriptor(_search()),
        normalize_config=lambda raw: Config.from_dict(raw),
    )

    assert evidence.tier is LookupTier.EXACT
    assert evidence.exact_eligible is False
    assert evidence.tier0_identity_combo == "S3-historical"
    assert evidence.tier0_collision_count == 2


def _compatible_neighbors() -> list[dict[str, object]]:
    neighbors: list[dict[str, object]] = []
    for i, dim in enumerate((3, 4, 8, 16, 32, 64, 128, 256)):
        neighbors.append(
            {
                "kernel_name": f"k{i}",
                "input_shapes": f"[(2, {dim}), (2, {dim})]",
                "dtypes": "['torch.float32', 'torch.float16']",
                "ref": {"run_id": f"run-{i}"},
                "score": 1.0 - i * 0.01,
                "top_n": [{"config_id": f"good-{i}", "config": {"block_sizes": [dim]}}],
            }
        )
    return neighbors


@requires_helion_rag
def test_convert_lookup_honors_non_default_final_neighbors() -> None:
    descriptor = build_workload_descriptor(_search())
    policy = FrozenRetrievalPolicy(semantic_pool_size=8, final_neighbors=3)

    evidence = convert_lookup_result(
        {"tier": 1, "family": "h100", "neighbors": _compatible_neighbors()},
        descriptor,
        normalize_config=lambda raw: Config.from_dict(raw),
        policy=policy,
    )

    assert evidence.tier is LookupTier.SEMANTIC
    assert len(evidence.neighbors) == 3


@requires_helion_rag
def test_convert_lookup_default_policy_retains_five() -> None:
    descriptor = build_workload_descriptor(_search())

    evidence = convert_lookup_result(
        {"tier": 1, "family": "h100", "neighbors": _compatible_neighbors()},
        descriptor,
        normalize_config=lambda raw: Config.from_dict(raw),
    )

    assert len(evidence.neighbors) == 5


@requires_helion_rag
def test_live_adapter_requests_a_pinned_generation_lookup() -> None:
    descriptor = build_workload_descriptor(_search())
    autotuner = SimpleNamespace(
        settings=SimpleNamespace(to_dict=dict, autotune_random_seed=123),
        config_gen=_IdentityConfigGeneration(),
    )

    with patch("helion_rag.lookup.lookup", return_value={"tier": 2}) as lookup:
        evidence = retrieve_evidence(descriptor, autotuner)

    assert evidence.tier is LookupTier.MISS
    assert lookup.call_args.kwargs["require_generation_pin"] is True


def test_cache_delegates_to_adapter_only_when_rag_is_enabled() -> None:
    class Cache(AutotuneCacheBase):
        def get(self):
            raise AssertionError("adapter owns the enabled path")

        def put(self, config):
            raise AssertionError("adapter owns the enabled path")

        def _get_cache_key(self):
            raise AssertionError("adapter owns the enabled path")

        def _list_cache_entries(self):
            raise AssertionError("adapter owns the enabled path")

    cache = Cache.__new__(Cache)
    cache.autotuner = SimpleNamespace(
        settings=SimpleNamespace(autotune_rag_enabled=True)
    )
    selected = Config(block_sizes=[32])

    with patch(
        "helion.autotuner.rag.adapter.autotune_with_rag", return_value=selected
    ) as run:
        assert AutotuneCacheBase.autotune(cache, skip_cache=True) is selected

    run.assert_called_once_with(cache, skip_cache=True)


def test_only_standalone_lfbo_and_llm_modes_are_supported() -> None:
    from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch

    assert resolve_tuner_mode(LFBOTreeSearch.__new__(LFBOTreeSearch)) is TunerMode.LFBO
    assert resolve_tuner_mode(LLMGuidedSearch.__new__(LLMGuidedSearch)) is TunerMode.LLM
    with pytest.raises(ValueError, match="unsupported RAG autotuner"):
        resolve_tuner_mode(SimpleNamespace())


def test_fixed_initial_population_bypasses_cache_and_random_generation() -> None:
    class ConfigGeneration:
        def flatten(self, config: Config) -> list[object]:
            return [config.config["value"]]

        def unflatten(self, flat: list[object]) -> Config:
            return Config(value=flat[0])

    search = PopulationBasedSearch.__new__(PopulationBasedSearch)
    search.config_gen = ConfigGeneration()
    search._fixed_initial_population_flat = None
    configs = [Config(value=1), Config(value=2), Config(value=3)]

    search.set_fixed_initial_population_configs(configs)

    assert search.fixed_initial_population_flat() == [[1], [2], [3]]


class _LiveCache(AutotuneCacheBase):
    def get(self):
        self.get_calls += 1
        return self.cached

    def put(self, config):
        self.puts.append(config)

    def _get_cache_key(self):
        return SimpleNamespace(stable_hash=lambda: "workload")

    def _list_cache_entries(self):
        return []

    def _should_report_cache_hit(self) -> bool:
        return False


def _live_cache(
    *, exact_read: bool, write: bool = False, best_of_k: int = 1
) -> _LiveCache:
    from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch

    search = LFBOTreeSearch.__new__(LFBOTreeSearch)
    search.settings = SimpleNamespace(
        autotune_rag_enabled=True,
        autotune_exact_read=exact_read,
        autotune_best_available_read=True,
        autotune_cache_write=write,
        autotune_best_of_k=best_of_k,
        autotune_random_seed=123,
        autotune_search_acf=[],
    )
    search.kernel = SimpleNamespace(
        kernel=SimpleNamespace(name="add"),
    )
    search.best_perf_so_far = float("inf")
    search.config_gen = _IdentityConfigGeneration()
    search._candidate_attempt_budget = AttemptBudget(None)
    search._candidate_attempt_configs = set()
    search._candidate_attempt_categories = {}
    search._candidate_sources = {}
    search._candidate_initial_population_open = True
    search._precounted_initial_population_configs = set()
    search._attempt_instrumentation = None
    search.log = lambda message: None
    cache = _LiveCache.__new__(_LiveCache)
    cache.autotuner = search
    cache.kernel = search.kernel
    cache.args = ()
    cache.cached = None
    cache.get_calls = 0
    cache.puts = []
    cache.baseline_calls = 0
    cache._run_autotune_trials = lambda: _run_baseline(cache)
    return cache


def _run_baseline(cache: _LiveCache) -> Config:
    cache.baseline_calls += 1
    return Config(block_sizes=[64])


def _miss_descriptor() -> object:
    return SimpleNamespace(kernel_name="add")


def test_exact_read_off_never_touches_cache_and_tier2_runs_baseline_once() -> None:
    cache = _live_cache(exact_read=False)
    before_misses = counters["autotune"]["cache_miss"]
    with (
        patch(
            "helion.autotuner.rag.adapter.build_workload_descriptor",
            return_value=_miss_descriptor(),
        ),
        patch(
            "helion.autotuner.rag.adapter.retrieve_evidence",
            return_value=RetrievalEvidence(tier=LookupTier.MISS),
        ),
        patch("helion.autotuner.rag.adapter.emit_attempt_event"),
    ):
        selected = autotune_with_rag(cache)

    assert selected == Config(block_sizes=[64])
    assert cache.get_calls == 0
    assert cache.baseline_calls == 1
    assert cache.puts == []
    assert counters["autotune"]["cache_miss"] == before_misses + 1


def test_tier2_llm_runs_the_unchanged_llm_search_not_lfbo_fallback() -> None:
    cache = _live_cache(exact_read=False)
    llm = LLMGuidedSearch.__new__(LLMGuidedSearch)
    llm.__dict__.update(cache.autotuner.__dict__)
    cache.autotuner = llm

    with (
        patch(
            "helion.autotuner.rag.adapter.build_workload_descriptor",
            return_value=_miss_descriptor(),
        ),
        patch(
            "helion.autotuner.rag.adapter.retrieve_evidence",
            return_value=RetrievalEvidence(tier=LookupTier.MISS),
        ),
        patch("helion.autotuner.rag.adapter.emit_attempt_event"),
        patch(
            "helion.autotuner.surrogate_pattern_search.LFBOTreeSearch.autotune"
        ) as lfbo_fallback,
    ):
        selected = autotune_with_rag(cache)

    assert selected == Config(block_sizes=[64])
    assert cache.baseline_calls == 1
    lfbo_fallback.assert_not_called()


def test_exact_hit_preserves_cache_accounting_and_skips_tuning_and_write() -> None:
    cache = _live_cache(exact_read=True, write=True)
    cache.cached = Config(block_sizes=[128])
    before_hits = counters["autotune"]["cache_hit"]
    with patch("helion.autotuner.rag.adapter.emit_attempt_event"):
        selected = autotune_with_rag(cache)

    assert selected == cache.cached
    assert cache.get_calls == 1
    assert cache.baseline_calls == 0
    assert cache.puts == []
    assert counters["autotune"]["cache_hit"] == before_hits + 1


def test_exact_hit_skips_descriptor_and_retrieval() -> None:
    cache = _live_cache(exact_read=True)
    cache.cached = Config(block_sizes=[128])

    with (
        patch("helion.autotuner.rag.adapter.build_workload_descriptor") as describe,
        patch("helion.autotuner.rag.adapter.retrieve_evidence") as retrieve,
        patch("helion.autotuner.rag.adapter.emit_attempt_event"),
    ):
        selected = autotune_with_rag(cache)

    assert selected == cache.cached
    describe.assert_not_called()
    retrieve.assert_not_called()


@requires_helion_rag
def test_exact_hit_event_records_explicit_probe_outcome(tmp_path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("HELION_RAG_EVENT_LOG", str(path))
    cache = _live_cache(exact_read=True)
    cache.cached = Config(block_sizes=[128])

    autotune_with_rag(cache)

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["retrieval"]["exact_cache_probe"] == "hit"


@requires_helion_rag
def test_propagated_error_emits_one_terminal_event_before_reraising(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("HELION_RAG_EVENT_LOG", str(path))
    cache = _live_cache(exact_read=False)
    descriptor = WorkloadDescriptor(
        kernel_name="add",
        kernel_source="source",
        input_shapes="[(1,)]",
        dtypes="['torch.float32']",
        hardware="h100",
        backend="triton",
        structural_fingerprint_hash="fp",
    )

    with (
        patch(
            "helion.autotuner.rag.adapter.build_workload_descriptor",
            return_value=descriptor,
        ),
        patch(
            "helion.autotuner.rag.adapter.retrieve_evidence",
            side_effect=exc.ProviderAuthError("bad credentials"),
        ),
        pytest.raises(exc.ProviderAuthError, match="bad credentials"),
    ):
        autotune_with_rag(cache)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["outcome"]["decision_name"] == "BaselineSearch"
    assert event["outcome"]["terminal_error"] == "ProviderAuthError"
    assert event["outcome"]["accumulated_seconds"] >= 0.0
    assert event["timings"]["total"] == event["outcome"]["accumulated_seconds"]
    assert event["outcome"]["fallback_reason"] is None


def test_rag_rejects_best_of_k_before_cache_or_retrieval() -> None:
    cache = _live_cache(exact_read=False, best_of_k=2)

    with pytest.raises(ValueError, match="best-of-K"):
        autotune_with_rag(cache)

    assert cache.get_calls == 0
    assert cache.baseline_calls == 0


def test_lfbo_seeding_and_generation_share_one_candidate_budget() -> None:
    from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch

    class ConfigGeneration:
        def __init__(self) -> None:
            self.random_values = iter([2])

        def flatten(self, config: Config) -> list[list[int]]:
            return [[int(config.config["value"])]]

        def unflatten(self, flat: list[list[int]]) -> Config:
            return Config(value=flat[0][0])

        def random_flat(self) -> list[list[int]]:
            return [[next(self.random_values)]]

        def encode_config(self, flat: list[list[int]]) -> list[float]:
            return [float(flat[0][0])]

    budget = AttemptBudget(3)
    search = LFBOTreeSearch.__new__(LFBOTreeSearch)
    search.config_gen = ConfigGeneration()
    search.initial_population = 2
    search.settings = SimpleNamespace(autotune_best_available_read=True)
    search._candidate_attempt_budget = budget
    search._candidate_attempt_configs = set()
    search._candidate_attempt_categories = {}
    search._candidate_sources = {}
    search._candidate_initial_population_open = True
    search._precounted_initial_population_configs = set()
    search._fixed_initial_population_flat = None
    collector = InstrumentationCollector()

    def run_search() -> Config:
        members = [
            search.make_unbenchmarked(flat)
            for flat in search.fixed_initial_population_flat() or []
        ]
        assert all(member is not None for member in members)
        assert budget.spent() == 2
        search._finish_initial_candidate_attempts()
        assert search.make_unbenchmarked([[3]]) is not None
        return Config(value=3)

    runner = _LiveDecisionRunner(search, TunerMode.LFBO, run_search, collector)
    result = runner.run(
        RetrievalSeededSearch(
            (
                RetrievedNeighbor(
                    kernel_name=None,
                    input_shapes=None,
                    dtypes=None,
                    config={"value": 1},
                    score=0.9,
                ),
            )
        )
    )

    assert result == Config(value=3)
    assert search.candidate_attempt_budget is budget
    assert budget.spent() == 3
    assert budget.spent_by(AttemptCategory.INITIAL_POPULATION) == 2
    assert budget.spent_by(AttemptCategory.GENERATION) == 1
    assert search._candidate_sources[Config(value=1)] == "retrieval"
    assert search._candidate_sources[Config(value=2)] == "random_replacement"


def test_combined_lfbo_population_contains_qwen_and_best_available_seeds() -> None:
    from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch

    class ConfigGeneration:
        def flatten(self, config: Config) -> list[int]:
            return [int(config.config["value"])]

        def unflatten(self, flat: list[int]) -> Config:
            return Config(value=flat[0])

        def random_flat(self) -> list[int]:
            return [99]

        def encode_config(self, flat: list[int]) -> list[float]:
            return [float(flat[0])]

    search = LFBOTreeSearch.__new__(LFBOTreeSearch)
    search.config_gen = ConfigGeneration()
    search.initial_population = 3
    search.settings = SimpleNamespace(
        autotune_best_available_read=True,
        autotune_best_available_max_configs=4,
    )
    search._find_similar_cached_configs = lambda limit: [
        SimpleNamespace(to_mutable_flat_config=lambda: [2])
    ]
    search._candidate_attempt_budget = AttemptBudget(4)
    search._candidate_attempt_configs = set()
    search._candidate_attempt_categories = {}
    search._candidate_sources = {}
    search._candidate_initial_population_open = True
    search._precounted_initial_population_configs = set()
    search._fixed_initial_population_flat = None

    def run_search() -> Config:
        assert search.fixed_initial_population_flat() == [[1], [2], [99]]
        return Config(value=1)

    runner = _LiveDecisionRunner(
        search,
        TunerMode.LFBO,
        run_search,
        InstrumentationCollector(),
        best_available_read=True,
    )
    result = runner.run(
        RetrievalSeededSearch(
            (
                RetrievedNeighbor(
                    kernel_name="qwen",
                    input_shapes="[(1,)]",
                    dtypes="['torch.float32']",
                    config={"value": 1},
                    score=0.9,
                ),
            )
        )
    )

    assert result == Config(value=1)
    assert search._candidate_sources[Config(value=1)] == "retrieval"
    assert search._candidate_sources[Config(value=2)] == "best_available"


def test_qwen_only_lfbo_does_not_read_best_available() -> None:
    from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch

    class ConfigGeneration:
        def flatten(self, config: Config) -> list[int]:
            return [int(config.config["value"])]

        def unflatten(self, flat: list[int]) -> Config:
            return Config(value=flat[0])

        def random_flat(self) -> list[int]:
            return [2]

        def encode_config(self, flat: list[int]) -> list[float]:
            return [float(flat[0])]

    search = LFBOTreeSearch.__new__(LFBOTreeSearch)
    search.config_gen = ConfigGeneration()
    search.initial_population = 2
    search.settings = SimpleNamespace(
        autotune_best_available_read=False,
        autotune_best_available_max_configs=4,
    )
    search._find_similar_cached_configs = lambda limit: (_ for _ in ()).throw(
        AssertionError("Qwen-only must not inspect best-available")
    )
    search._candidate_attempt_budget = AttemptBudget(3)
    search._candidate_attempt_configs = set()
    search._candidate_attempt_categories = {}
    search._candidate_sources = {}
    search._candidate_initial_population_open = True
    search._precounted_initial_population_configs = set()
    search._fixed_initial_population_flat = None
    runner = _LiveDecisionRunner(
        search,
        TunerMode.LFBO,
        lambda: Config(value=1),
        InstrumentationCollector(),
        best_available_read=False,
    )

    runner.run(
        RetrievalSeededSearch(
            (
                RetrievedNeighbor(
                    kernel_name=None,
                    input_shapes=None,
                    dtypes=None,
                    config={"value": 1},
                    score=0.9,
                ),
            )
        )
    )

    assert search.fixed_initial_population_flat() == [[1], [2]]


def test_lfbo_drops_unencodable_seed_and_replaces_with_random() -> None:
    # A retrieved seed that round-trips flatten/unflatten but cannot be encoded by
    # the LFBO surrogate must be dropped (it would otherwise crash training) and
    # replaced by a random draw.
    from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch

    class ConfigGeneration:
        def __init__(self) -> None:
            self.random_values = iter([2, 3])

        def flatten(self, config: Config) -> list[int]:
            return [int(config.config["value"])]

        def unflatten(self, flat: list[int]) -> Config:
            return Config(value=flat[0])

        def random_flat(self) -> list[int]:
            return [next(self.random_values)]

        def encode_config(self, flat: list[int]) -> list[float]:
            if flat == [1]:
                raise AssertionError("retrieved seed is not surrogate-encodable")
            return [float(flat[0])]

    search = LFBOTreeSearch.__new__(LFBOTreeSearch)
    search.config_gen = ConfigGeneration()
    search.initial_population = 2
    search.settings = SimpleNamespace(
        autotune_best_available_read=False,
        autotune_best_available_max_configs=4,
    )
    search._candidate_attempt_budget = AttemptBudget(8)
    search._candidate_attempt_configs = set()
    search._candidate_attempt_categories = {}
    search._candidate_sources = {}
    search._candidate_initial_population_open = True
    search._precounted_initial_population_configs = set()
    search._fixed_initial_population_flat = None
    runner = _LiveDecisionRunner(
        search,
        TunerMode.LFBO,
        lambda: Config(value=2),
        InstrumentationCollector(),
        best_available_read=False,
    )

    runner.run(
        RetrievalSeededSearch(
            (
                RetrievedNeighbor(
                    kernel_name=None,
                    input_shapes=None,
                    dtypes=None,
                    config={"value": 1},
                    score=0.9,
                ),
            )
        )
    )

    # The unencodable seed [1] is absent; the population is filled with random draws.
    assert search.fixed_initial_population_flat() == [[2], [3]]
    assert Config(value=1) not in search._candidate_sources


def test_retrieval_controls_are_deterministic_distinct_and_keep_k() -> None:
    descriptor = WorkloadDescriptor(
        kernel_name="add",
        kernel_source="source",
        input_shapes="[(1,)]",
        dtypes="['torch.float32']",
        hardware="h100",
        backend="triton",
        structural_fingerprint_hash="fp",
    )
    neighbors = tuple(
        RetrievedNeighbor(
            kernel_name=f"k{i}",
            input_shapes=f"[({i + 1},)]",
            dtypes="['torch.float32']",
            score=1.0 - i / 10,
            config={"value": i},
        )
        for i in range(5)
    )
    evidence = RetrievalEvidence(tier=LookupTier.SEMANTIC, neighbors=neighbors)

    frozen = apply_retrieval_control(evidence, descriptor, "frozen", seed=17)
    random_a = apply_retrieval_control(evidence, descriptor, "random", seed=17)
    random_b = apply_retrieval_control(evidence, descriptor, "random", seed=17)
    shuffled = apply_retrieval_control(evidence, descriptor, "shuffled", seed=17)

    assert frozen.neighbors == neighbors
    assert random_a == random_b
    assert len(random_a.neighbors) == len(shuffled.neighbors) == len(neighbors)
    assert random_a.neighbors != frozen.neighbors
    assert shuffled.neighbors != frozen.neighbors
    assert shuffled.neighbors != random_a.neighbors


@requires_helion_rag
def test_event_records_retrieval_control_candidate_source(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("HELION_RAG_EVENT_LOG", str(path))
    cache = _live_cache(exact_read=False)
    evidence = RetrievalEvidence(
        tier=LookupTier.SEMANTIC,
        neighbors=(
            RetrievedNeighbor(
                kernel_name="k",
                input_shapes="[(1,)]",
                dtypes="['torch.float32']",
                score=0.9,
                config={"value": 1},
            ),
        ),
    )
    outcome = ExecutionOutcome(
        decision=RetrievalSeededSearch(evidence.neighbors),
        result=Config(value=1),
        validated=True,
        accumulated_seconds=1.0,
    )
    descriptor = WorkloadDescriptor(
        kernel_name="add",
        kernel_source="source",
        input_shapes="[(1,)]",
        dtypes="['torch.float32']",
        hardware="h100",
        backend="triton",
        structural_fingerprint_hash="fp",
    )

    emit_attempt_event(
        cache,
        tuner_mode=TunerMode.LFBO,
        descriptor=descriptor,
        evidence=evidence,
        outcome=outcome,
        instrumentation=InstrumentationCollector(),
        attempt_budget=AttemptBudget(None),
        retrieval_control="random",
    )

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["retrieval"]["candidate_source"] == "random_neighbors"


@requires_helion_rag
def test_terminal_event_contains_all_neighbors_evaluations_and_accounting(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("HELION_RAG_EVENT_LOG", str(path))
    monkeypatch.setenv("HELION_RAG_WORKLOAD_ID", "frozen-workload-id")
    cache = _live_cache(exact_read=False)
    budget = AttemptBudget(8)
    assert budget.record(AttemptCategory.INITIAL_POPULATION, 2)
    assert budget.record(AttemptCategory.GENERATION)
    collector = InstrumentationCollector()
    collector.record_evaluation(
        config_id=hashlib.sha256(repr(Config(value=1)).encode()).hexdigest(),
        config_repr="Config(value=1)",
        candidate_source="retrieval",
        candidate_category="initial_population",
        compatibility_status="compatible",
        compilation_status="ok",
        compilation_seconds=0.1,
        correctness_status="ok",
        benchmark_status="ok",
        timeout_status="not_timed_out",
        performance=0.8,
    )
    collector.record_evaluation(
        config_id=hashlib.sha256(repr(Config(value=2)).encode()).hexdigest(),
        config_repr="Config(value=2)",
        candidate_source="generation",
        candidate_category="generation",
        compatibility_status="compatible",
        compilation_status="ok",
        compilation_seconds=0.1,
        correctness_status="ok",
        benchmark_status="ok",
        timeout_status="not_timed_out",
        performance=0.5,
    )
    collector.record_phase_transition(Phase.BENCHMARKING, phase_seconds=0.2)
    collector.record_phase_transition(Phase.PROVIDER, phase_seconds=0.25)
    neighbors = tuple(
        RetrievedNeighbor(
            kernel_name=f"k{i}",
            input_shapes="[(1,)]",
            dtypes="['torch.float32']",
            score=0.9 - i * 0.1,
            config={"value": i + 1},
            ref={
                "run_id": f"run-{i}",
                "config_id": f"cfg-{i}",
                "raw_semantic_score": 0.9 - i * 0.1,
                "raw_shape_score": float(i),
                "combined_score": float(i) / 10,
                "rank_before_rerank": i,
                "rank_after_rerank": i,
            },
        )
        for i in range(2)
    )
    evidence = RetrievalEvidence(
        tier=LookupTier.SEMANTIC,
        neighbors=neighbors,
        tier0_identity_combo="S4-extended",
        tier0_collision_count=0,
        artifact_identity={
            "model_id": "Qwen/Qwen3-Embedding-4B@resolved",
            "tokenizer_id": "Qwen/Qwen3-Embedding-4B@resolved",
        },
    )
    outcome = ExecutionOutcome(
        decision=RetrievalSeededSearch(neighbors),
        result=Config(value=1),
        validated=True,
        accumulated_seconds=1.0,
        timings=(PhaseTimingEvent(Phase.TOTAL, 1.0),),
    )
    descriptor = WorkloadDescriptor(
        kernel_name="add",
        kernel_source="source",
        input_shapes="[(1,)]",
        dtypes="['torch.float32']",
        hardware="h100",
        backend="triton",
        structural_fingerprint_hash="fp",
    )

    emit_attempt_event(
        cache,
        tuner_mode=TunerMode.LFBO,
        descriptor=descriptor,
        evidence=evidence,
        outcome=outcome,
        instrumentation=collector,
        attempt_budget=budget,
    )

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["run"]["workload_id"] == "frozen-workload-id"
    assert [item["config_id"] for item in event["retrieval"]["neighbors"]] == [
        "cfg-0",
        "cfg-1",
    ]
    assert event["evaluations"][0]["candidate_source"] == "retrieval"
    assert event["outcome"]["incumbent_best_perf"] == 0.5
    assert event["outcome"]["selected_performance"] == 0.8
    assert event["artifacts"]["model_id"] == "Qwen/Qwen3-Embedding-4B@resolved"
    assert event["artifacts"]["tokenizer_id"] == "Qwen/Qwen3-Embedding-4B@resolved"
    assert "backend=triton" in event["artifacts"]["runtime_id"]
    assert event["retrieval"]["tier0_identity_combo"] == "S4-extended"
    assert event["retrieval"]["tier0_collision_count"] == 0
    assert event["timings"]["provider"] == 0.25
    assert event["timings"]["readiness_provider_inclusive"] == 1.0
    assert event["timings"]["readiness_provider_exclusive"] == 0.75
    assert event["accounting"] == {
        "attempted": 3,
        "benchmarked": 2,
        "compiled": 2,
        "duplicate": 0,
        "frozen_limit": 8,
        "generation": 1,
        "initial_population": 2,
        "invalid": 0,
        "llm_proposed": 0,
        "validated": 2,
    }


@requires_helion_rag
def test_terminal_event_contains_aggregate_provider_identity_and_tokens(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("HELION_RAG_EVENT_LOG", str(path))
    cache = _live_cache(exact_read=False)
    search = LLMGuidedSearch.__new__(LLMGuidedSearch)
    search.settings = cache.autotuner.settings
    search.kernel = cache.autotuner.kernel
    search._provider_requests = 2
    search._token_usage = [
        TokenUsage(input_tokens=10, cached_input_tokens=5, output_tokens=2),
        TokenUsage(input_tokens=None, cached_input_tokens=0, output_tokens=3),
    ]
    search._provider_metadata = [
        ProviderMetadata("request-1", "response-1", "hit"),
        ProviderMetadata("request-2", "response-2", "miss"),
    ]
    search._provider_replay_identities = [
        ("sha256:request-1", "sha256:response-1"),
        ("sha256:request-2", "sha256:response-2"),
    ]
    cache.autotuner = search
    budget = AttemptBudget(None)
    outcome = ExecutionOutcome(
        decision=ContextualSearch(()),
        result=object(),
        validated=True,
        accumulated_seconds=1.0,
    )
    descriptor = WorkloadDescriptor(
        kernel_name="add",
        kernel_source="source",
        input_shapes="[(1,)]",
        dtypes="['torch.float32']",
        hardware="h100",
        backend="triton",
        structural_fingerprint_hash="fp",
    )

    emit_attempt_event(
        cache,
        tuner_mode=TunerMode.LLM,
        descriptor=descriptor,
        evidence=None,
        outcome=outcome,
        instrumentation=InstrumentationCollector(),
        attempt_budget=budget,
    )

    provider = json.loads(path.read_text(encoding="utf-8"))["provider"]
    assert provider["request_id"].startswith("sha256:")
    assert provider["response_id"].startswith("sha256:")
    assert provider["cache_state"] == "miss"
    assert provider["requests"] == 2
    assert provider["input_tokens"] is None
    assert provider["cached_input_tokens"] == 5
    assert provider["output_tokens"] == 5


@requires_helion_rag
def test_event_without_descriptor_uses_signed_static_artifact_identities(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("HELION_RAG_EVENT_LOG", str(path))
    expected = {
        "HELION_RAG_EXPECTED_CODE_ID": "signed-code",
        "HELION_RAG_EXPECTED_CORPUS_ID": "signed-corpus",
        "HELION_RAG_EXPECTED_MODEL_ID": "signed-model",
        "HELION_RAG_EXPECTED_TOKENIZER_ID": "signed-tokenizer",
        "HELION_RAG_EXPECTED_INDEX_ID": "000042",
        "HELION_RAG_EXPECTED_MANIFEST_ID": "signed-manifest",
        "HELION_RAG_EXPECTED_RUNTIME_ID": (
            "backend=triton;torch=2.9.0;triton=3.4.0;cuda=12.8"
        ),
        "HELION_RAG_EXPECTED_TRITON_ID": "3.4.0",
        "HELION_RAG_EXPECTED_DRIVER_ID": "580.1",
        "HELION_RAG_EXPECTED_HARDWARE_ID": "NVIDIA H100 80GB HBM3",
        "HELION_RAG_EXPECTED_DEVICE_UUID": "GPU-physical-1",
        "HELION_RAG_WORKLOAD_ID": "signed-kernel-environment",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    cache = _live_cache(exact_read=True)
    selected = Config(block_sizes=[128])
    outcome = ExecutionOutcome(
        decision=BaselineSearch("existing_exact_hit"),
        result=selected,
        validated=True,
        accumulated_seconds=0.1,
        timings=(PhaseTimingEvent(Phase.TOTAL, 0.1),),
    )

    with patch("helion_rag.config._config", side_effect=AssertionError("no lookup")):
        emit_attempt_event(
            cache,
            tuner_mode=TunerMode.LFBO,
            descriptor=None,
            evidence=None,
            outcome=outcome,
            instrumentation=InstrumentationCollector(),
            attempt_budget=AttemptBudget(None),
            exact_cache_probe="hit",
        )

    event = json.loads(path.read_text(encoding="utf-8"))
    artifacts = event["artifacts"]
    assert artifacts == {
        "code_id": "signed-code",
        "corpus_id": "signed-corpus",
        "device_uuid": "GPU-physical-1",
        "driver_id": "580.1",
        "hardware_id": "NVIDIA H100 80GB HBM3",
        "index_id": "000042",
        "kernel_environment_id": "signed-kernel-environment",
        "manifest_id": "signed-manifest",
        "model_id": "signed-model",
        "runtime_id": "backend=triton;torch=2.9.0;triton=3.4.0;cuda=12.8",
        "tokenizer_id": "signed-tokenizer",
        "triton_id": "3.4.0",
    }
    # An LFBO attempt makes no provider calls, so provider identity is empty.
    provider = event["provider"]
    assert provider["requests"] == 0
    assert provider["replay_identities"] == []


def test_lfbo_underfilled_population_fails_the_arm_without_baseline_fallback() -> None:
    from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch

    class ConfigGeneration:
        def flatten(self, config: Config) -> list[int]:
            return [int(config.config["value"])]

        def unflatten(self, flat: list[int]) -> Config:
            return Config(value=flat[0])

        def random_flat(self) -> list[int]:
            raise AssertionError("exhausted budget must stop replacement draws")

        def encode_config(self, flat: list[int]) -> list[float]:
            return [float(flat[0])]

    search = LFBOTreeSearch.__new__(LFBOTreeSearch)
    search.config_gen = ConfigGeneration()
    search.initial_population = 2
    search.settings = SimpleNamespace(autotune_best_available_read=True)
    search._candidate_attempt_budget = AttemptBudget(1)
    search._candidate_attempt_configs = set()
    search._candidate_attempt_categories = {}
    search._candidate_sources = {}
    search._candidate_initial_population_open = True
    search._precounted_initial_population_configs = set()
    search._fixed_initial_population_flat = None
    runner = _LiveDecisionRunner(
        search,
        TunerMode.LFBO,
        lambda: (_ for _ in ()).throw(AssertionError("baseline must not run")),
        InstrumentationCollector(),
    )

    with pytest.raises(PopulationUnderfilled, match="1/2"):
        runner.run(
            RetrievalSeededSearch(
                (
                    RetrievedNeighbor(
                        kernel_name=None,
                        input_shapes=None,
                        dtypes=None,
                        config={"value": 1},
                        score=0.9,
                    ),
                )
            )
        )
