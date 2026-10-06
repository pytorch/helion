from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from helion.autotuner import pipeline
from helion.autotuner.pipeline_benchmark import PipelineMeasurement
from helion.runtime.config import Config
from helion.runtime.pipeline import PipelineConfig
from helion.runtime.pipeline import PipelineStage
from helion.runtime.settings import Settings


def _stage(key, config):
    return PipelineStage(
        key=key,
        kernel=SimpleNamespace(),
        bound=SimpleNamespace(settings=Settings()),
        args=(),
        config=config,
        source_hash="test",
        argument_metadata=[],
    )


def test_new_topology_is_refined_before_a_slow_default_discards_it(monkeypatch):
    calls = []

    class Evaluator:
        def __init__(self, *args, **kwargs):
            self.operation = args[0]
            self.aggregation = kwargs["aggregation"]

        def evaluate(self, bundle):
            configs = dict(bundle.configs)
            configs.setdefault("upstream", Config(block_sizes=[64]))
            changed = configs["upstream"].block_sizes == [128]
            trace = ("upstream", "new") if changed else ("upstream", "old")
            configs.setdefault(trace[1], Config(block_sizes=[64]))
            latency = (
                (4.0 if configs["new"].block_sizes == [128] else 20.0)
                if changed
                else 10.0
            )
            calls.append((trace, latency))
            return PipelineMeasurement(
                bundle=PipelineConfig(configs),
                timings_ms=(latency, latency),
                aggregate_ms=latency,
                traces=(trace, trace),
                stages={key: _stage(key, configs[key]) for key in trace},
            )

    monkeypatch.setattr(pipeline, "PipelineEvaluator", Evaluator)
    monkeypatch.setattr(pipeline, "_fingerprint", lambda *args, **kwargs: {})
    result = pipeline.autotune_pipeline(
        lambda x: x,
        [(1,), (2,)],
        reference=lambda x: x,
        check=lambda actual, expected: None,
        algorithm="finite",
        config_candidates=lambda stage: [Config(block_sizes=[128])],
        max_evaluations=10,
        coordinate_evaluations=2,
        rounds=1,
    )
    assert (("upstream", "new"), 20.0) in calls
    assert (("upstream", "new"), 4.0) in calls
    assert result.aggregate_ms == 4.0
    assert result.traces == (("upstream", "new"),) * 2
    assert set(result.config.configs) == {"upstream", "new"}
    assert result.metadata["confirmation_evaluations"] >= 2


def test_final_confirmation_can_reject_an_optimistic_search_winner(monkeypatch):
    seen = {}

    class Evaluator:
        def __init__(self, *args, **kwargs):
            self.operation = args[0]
            self.aggregation = kwargs["aggregation"]

        def evaluate(self, bundle):
            config = bundle.configs.get("stage", Config(num_warps=4))
            count = seen.get(config.num_warps, 0)
            seen[config.num_warps] = count + 1
            latency = 10.0 if config.num_warps == 4 else (1.0 if count == 0 else 30.0)
            return PipelineMeasurement(
                bundle=PipelineConfig({"stage": config}),
                timings_ms=(latency,),
                aggregate_ms=latency,
                traces=(("stage",),),
                stages={"stage": _stage("stage", config)},
            )

    def propose(search):
        candidate = search.pool[0].bundle.with_config("stage", Config(num_warps=8))
        search.keep(search.measure(candidate, phase="coordinate"))

    monkeypatch.setattr(pipeline, "PipelineEvaluator", Evaluator)
    monkeypatch.setattr(pipeline, "_fingerprint", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline._PipelineSearch, "run_coordinates", propose)
    result = pipeline.autotune_pipeline(
        lambda x: x,
        [(1,)],
        reference=lambda x: x,
        check=lambda actual, expected: None,
        max_evaluations=2,
    )
    assert result.config.configs["stage"].num_warps == 4
    assert result.metadata["incumbent_preserved"]
    assert result.aggregate_ms == result.incumbent_aggregate_ms == 10.0
    assert result.metadata["search_evaluations"] == 2


def test_budget_does_not_skip_incumbent_and_final_confirmation(monkeypatch):
    class Evaluator:
        def __init__(self, *args, **kwargs):
            self.operation = args[0]
            self.aggregation = kwargs["aggregation"]

        def evaluate(self, bundle):
            config = Config(num_warps=4)
            return PipelineMeasurement(
                bundle=PipelineConfig({"stage": config}),
                timings_ms=(2.0, 8.0),
                aggregate_ms=4.0,
                traces=(("stage",), ("stage",)),
                stages={"stage": _stage("stage", config)},
            )

    monkeypatch.setattr(pipeline, "PipelineEvaluator", Evaluator)
    monkeypatch.setattr(pipeline, "_fingerprint", lambda *args, **kwargs: {})
    result = pipeline.autotune_pipeline(
        lambda x: x,
        [(1,), (2,)],
        reference=lambda x: x,
        check=lambda actual, expected: None,
        max_evaluations=1,
        max_seconds=1e-30,
        algorithm="finite",
    )
    assert result.metadata["search_evaluations"] == 1
    assert result.metadata["confirmation_evaluations"] == 1
    assert result.per_input_speedups == (1.0, 1.0)


def _deterministic_evaluator(monkeypatch, timings):
    class Evaluator:
        def __init__(self, *args, **kwargs):
            self.operation = args[0]
            self.aggregation = kwargs["aggregation"]

        def evaluate(self, bundle):
            config = bundle.configs.get("stage", Config(num_warps=4))
            times = timings[config.num_warps]
            return PipelineMeasurement(
                bundle=PipelineConfig({"stage": config}),
                timings_ms=times,
                aggregate_ms=math.prod(times) ** (1 / len(times)),
                traces=(("stage",),) * len(times),
                stages={"stage": _stage("stage", config)},
            )

    monkeypatch.setattr(pipeline, "PipelineEvaluator", Evaluator)
    monkeypatch.setattr(
        pipeline,
        "_fingerprint",
        lambda *args, **kwargs: {
            "workload_tag": kwargs["workload_tag"],
            "benchmark_tag": kwargs["benchmark_tag"],
        },
    )


def test_aggregate_acceptance_reports_individual_input_regression(monkeypatch):
    _deterministic_evaluator(monkeypatch, {4: (10.0, 10.0), 8: (2.0, 20.0)})
    result = pipeline.autotune_pipeline(
        lambda x: x,
        [(1,), (2,)],
        reference=lambda x: x,
        check=lambda actual, expected: None,
        algorithm="finite",
        config_candidates=lambda stage: [Config(num_warps=8)],
        max_evaluations=3,
        rounds=1,
    )
    assert result.config.configs["stage"].num_warps == 8
    assert result.aggregate_ms == pytest.approx(math.sqrt(40))
    assert result.per_input_speedups == (5.0, 0.5)
    assert result.regressed_input_indices == (1,)
    assert not result.metadata["incumbent_preserved"]


def test_pipeline_cache_is_only_a_validated_seed_and_tags_invalidate_it(
    monkeypatch, tmp_path
):
    _deterministic_evaluator(monkeypatch, {4: (10.0,), 8: (4.0,)})
    path = tmp_path / "pipeline.json"
    kwargs = {
        "reference": lambda x: x,
        "check": lambda actual, expected: None,
        "algorithm": "finite",
        "config_candidates": lambda stage: [Config(num_warps=8)],
        "max_evaluations": 3,
        "rounds": 1,
        "cache_path": path,
        "workload_tag": "values-v1",
        "benchmark_tag": "timer-v1",
    }
    first = pipeline.autotune_pipeline(lambda x: x, [(1,)], **kwargs)
    assert first.aggregate_ms == 4.0
    # Deliberately stale performance is never accepted as current evidence.
    cached = json.loads(path.read_text())
    cached["result"]["aggregate_ms"] = 1e-9
    path.write_text(json.dumps(cached))
    monkeypatch.setattr(pipeline._PipelineSearch, "run_coordinates", lambda self: None)
    second = pipeline.autotune_pipeline(lambda x: x, [(1,)], **kwargs)
    assert second.metadata["cache_used_as_seed"]
    assert second.aggregate_ms == 4.0
    assert [item["phase"] for item in second.history] == [
        "incumbent",
        "cached_seed",
        "confirmation",
        "confirmation",
    ]
    kwargs["workload_tag"] = "values-v2"
    third = pipeline.autotune_pipeline(lambda x: x, [(1,)], **kwargs)
    assert not third.metadata["cache_used_as_seed"]
    assert third.aggregate_ms == 10.0


@pytest.mark.parametrize("keyword,value", [("algorithm", "typo"), ("effort", "typo")])
def test_invalid_search_option_fails_before_initial_measurement(keyword, value):
    with pytest.raises(ValueError, match="Pipeline"):
        pipeline.autotune_pipeline(
            lambda x: x,
            [(1,)],
            reference=lambda x: x,
            check=lambda actual, expected: None,
            max_evaluations=1,
            **{keyword: value},
        )


def test_beam_refines_later_coordinates_of_a_temporarily_slower_topology(monkeypatch):
    class Evaluator:
        def __init__(self, operation, arg_sets, **kwargs):
            self.operation = operation
            self.aggregation = kwargs["aggregation"]

        def evaluate(self, bundle):
            configs = dict(bundle.configs)
            configs.setdefault("upstream", Config(num_warps=4))
            if configs["upstream"].num_warps == 4:
                trace = ("upstream", "old")
                configs.setdefault("old", Config(num_warps=4))
                latency = 10.0
            else:
                trace = ("upstream", "new_a", "new_b")
                for key in trace[1:]:
                    configs.setdefault(key, Config(num_warps=4))
                first_improved = configs["new_a"].num_warps == 8
                second_improved = configs["new_b"].num_warps == 8
                latency = (
                    5.0
                    if first_improved and second_improved
                    else (15.0 if first_improved else 20.0)
                )
            return PipelineMeasurement(
                bundle=PipelineConfig(configs),
                timings_ms=(latency,),
                aggregate_ms=latency,
                traces=(trace,),
                stages={key: _stage(key, configs[key]) for key in trace},
            )

    monkeypatch.setattr(pipeline, "PipelineEvaluator", Evaluator)
    monkeypatch.setattr(pipeline, "_fingerprint", lambda *args, **kwargs: {})
    result = pipeline.autotune_pipeline(
        lambda x: x,
        [(1,)],
        reference=lambda x: x,
        check=lambda actual, expected: None,
        algorithm="finite",
        config_candidates=lambda stage: [Config(num_warps=8)],
        max_evaluations=40,
        coordinate_evaluations=2,
        topology_refinements=0,
        beam_width=2,
        rounds=2,
    )
    assert result.aggregate_ms == 5.0
    assert result.traces == (("upstream", "new_a", "new_b"),)
    assert any(row["aggregate_ms"] == 15.0 for row in result.history)
