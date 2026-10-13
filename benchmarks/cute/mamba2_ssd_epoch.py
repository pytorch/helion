"""One natural full search with a fixed complete-pipeline objective.

The benchmark caller owns the source, partners, raw inputs and stream.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import secrets
import statistics
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion.autotuner.benchmark_objective import BenchmarkObjective
    from helion.autotuner.benchmark_objective import BenchmarkRequest
    from helion.autotuner.objective_timing import ObjectiveTiming
    from helion.runtime.kernel import BoundKernel


def write_new(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_epoch(
    bound: BoundKernel,
    args: Sequence[object],
    *,
    objective: BenchmarkObjective,
    timing: ObjectiveTiming,
    output: Path,
) -> BenchmarkRequest:
    from helion.autotuner.benchmark_objective import BenchmarkRequest
    from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
    from helion.autotuner.local_cache import LocalAutotuneCache
    from helion.autotuner.local_cache import _search_policy_json_value
    from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
    from helion.runtime.settings import default_autotuner_fn

    settings = bound.settings
    assert settings.backend == "cute"
    assert settings.autotune_effort == "full"
    assert settings.autotune_budget_seconds is None
    assert not settings.autotune_benchmark_subprocess
    assert not settings.autotune_config_overrides
    assert not settings.autotune_seed_configs
    assert not settings.autotune_search_acf
    assert settings.autotune_best_of_k == 1
    assert not bound.kernel.configs
    output.mkdir(exist_ok=False)
    settings.autotune_random_seed = secrets.randbits(32)
    outer = default_autotuner_fn(
        bound, args, benchmark_objective=objective, objective_timing=timing
    )
    assert isinstance(outer, LocalAutotuneCache)
    search = outer.autotuner
    assert isinstance(search, LFBOTreeSearch)
    benchmark = search.objective_benchmark
    assert benchmark is not None
    epoch = benchmark.epoch
    assert not benchmark.observations
    assert not outer._get_local_cache_path().exists()
    started = time.monotonic()
    write_new(
        output / "START.json",
        {
            "kind": "natural-full-conditional-pipeline-search",
            "eligible_for_goal": False,
            "seed": settings.autotune_random_seed,
            "skip_cache": True,
            "objective_epoch": epoch,
            "objective": dict(objective.cache_record()),
            "timing": timing.cache_record(),
            "search_policy": _search_policy_json_value(search.cache_policy()),
            "cache_key": repr(outer._get_cache_key()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    try:
        winner = outer.autotune(skip_cache=True)
        write_new(
            output / "SEARCH_RETURNED.json",
            {
                "status": "full search returned; source export pending",
                "eligible_for_goal": False,
                "config": winner.config,
                "objective_epoch": epoch,
                "metrics": search._autotune_metrics.to_dict(),
            },
        )
    finally:
        # Preserve all observed successes and full-oracle rejections, including
        # on a failed search. This does not change or rescore search members.
        with (output / "observations.jsonl").open("x") as stream:
            for phase, observation in benchmark.observations:
                row = {
                    "phase": phase,
                    "config": json.loads(observation.request.config_json),
                    "source_hash": observation.request.source_hash,
                    "request_digest": observation.request.digest,
                    "accuracy_error": observation.accuracy_error,
                    "measurement": (
                        None
                        if observation.measurement is None
                        else dataclasses.asdict(observation.measurement)
                    ),
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    assert outer.autotuner is search
    assert search._skip_cache is True
    assert benchmark.epoch == epoch
    benchmark.validate()
    provider = search.benchmark_provider
    assert isinstance(provider, LocalBenchmarkProvider)
    provider.check_objective_selection(winner)
    compiled = bound.compile_config(winner)
    source_hash = bound.config_spec.backend.generated_source_hash(compiled)
    assert source_hash is not None
    request = BenchmarkRequest.create(winner, compiled, source_hash)
    measurement = provider.objective_measurement_for(winner, compiled)
    assert measurement is not None
    measurement.validate(epoch, request)
    executed = Path(compiled.__code__.co_filename).read_bytes()
    assert hashlib.sha256(executed).hexdigest() == source_hash
    (output / "selected-executed.py").write_bytes(executed)
    write_new(
        output / "RESULT.json",
        {
            "status": "full search completed",
            "eligible_for_goal": False,
            "elapsed_seconds": time.monotonic() - started,
            "objective_epoch": epoch,
            "config": winner.config,
            "config_envelope": bound.config_envelope(winner).to_dict(),
            "source_hash": source_hash,
            "request_digest": request.digest,
            "measurement": dataclasses.asdict(measurement),
            "search_perf_ms": statistics.median(measurement.samples_ms),
            "metrics": search._autotune_metrics.to_dict(),
            "effective_search_policy": _search_policy_json_value(search.cache_policy()),
            "observations_sha256": hashlib.sha256(
                (output / "observations.jsonl").read_bytes()
            ).hexdigest(),
        },
    )
    return request
