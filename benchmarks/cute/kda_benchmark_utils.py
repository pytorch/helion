"""Shared measurement helpers for KDA benchmark drivers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Protocol
from typing import cast

import torch

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_COOLDOWN_C = 46.0


@dataclass(frozen=True)
class PairedGraphArm:
    """One provider with private mutable buffers for paired graph timing."""

    name: str
    launch: Callable[[], object]
    reset: Callable[[], None]


@dataclass(frozen=True)
class _CapturedGraphArm:
    arm: PairedGraphArm
    graph: torch.cuda.CUDAGraph


class _GeneratedSourceBackend(Protocol):
    def generated_source_hash(self, compiled_fn: object) -> str | None: ...


class _GeneratedSourceEnvironment(Protocol):
    backend: _GeneratedSourceBackend


class _GeneratedWrapperBound(Protocol):
    _run: object | None
    env: _GeneratedSourceEnvironment

    def to_triton_code(self, config: object) -> str: ...


def verify_helion_generated_wrapper(
    bound: object,
    config: object,
    *,
    expected_plan_kind: str,
    expected_source_markers: tuple[str, ...],
    expected_source_patterns: tuple[str, ...] = (),
    compiled_fn: object | None = None,
) -> dict[str, Any]:
    """Verify that the compiled Helion wrapper used the expected CuTe plan."""

    if not expected_plan_kind or not expected_source_markers:
        raise ValueError("Helion wrapper verification requires a plan and markers")
    typed_bound = cast("_GeneratedWrapperBound", bound)
    source = typed_bound.to_triton_code(config)
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    compiled = typed_bound._run if compiled_fn is None else compiled_fn
    if compiled is None:
        raise RuntimeError("Helion bound kernel has no compiled wrapper")
    compiled_source_hash = typed_bound.env.backend.generated_source_hash(compiled)
    if compiled_source_hash != source_hash:
        raise RuntimeError(
            "Helion compiled-wrapper source identity mismatch: "
            f"compiled={compiled_source_hash}, regenerated={source_hash}"
        )

    stripped_lines = [line.strip() for line in source.splitlines()]
    marker_counts = {
        marker: stripped_lines.count(marker) for marker in expected_source_markers
    }
    invalid_markers = {
        marker: count for marker, count in marker_counts.items() if count != 1
    }
    pattern_counts = {
        pattern: sum(re.fullmatch(pattern, line) is not None for line in stripped_lines)
        for pattern in expected_source_patterns
    }
    invalid_patterns = {
        pattern: count for pattern, count in pattern_counts.items() if count != 1
    }
    if invalid_markers or invalid_patterns:
        raise RuntimeError(
            f"Helion expected {expected_plan_kind} generated-wrapper markers "
            "exactly once, got "
            f"literal={invalid_markers}, regex={invalid_patterns}"
        )
    return {
        "plan_kind": expected_plan_kind,
        "source_sha256": source_hash,
        "source_bytes": len(source.encode("utf-8")),
        "source_lines": len(source.splitlines()),
        "expected_source_markers": list(expected_source_markers),
        "source_marker_counts": marker_counts,
        "expected_source_patterns": list(expected_source_patterns),
        "source_pattern_counts": pattern_counts,
        "compiled_source_identity_verified": True,
    }


def file_sha256(path: Path) -> str:
    """Hash one source artifact without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_json_object(value: str, option: str) -> dict[str, Any]:
    """Parse a finite JSON object while rejecting duplicate keys."""

    def reject_nonfinite(constant: str) -> None:
        raise ValueError(f"{option} contains non-finite JSON value {constant}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{option} contains duplicate key {key!r}")
            result[key] = item
        return result

    parsed = json.loads(
        value,
        parse_constant=reject_nonfinite,
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"{option} must contain a JSON object")
    return parsed


def canonical_config_json(value: str, option: str) -> str:
    return json.dumps(
        parse_json_object(value, option),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _git_output(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def _git_untracked_python_sha256(root: Path, pathspecs: tuple[str, ...]) -> str:
    scoped_pathspecs = pathspecs or (":(glob)**/*.py",)
    output = _git_output(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *scoped_pathspecs,
        text=False,
    )
    assert isinstance(output, bytes)
    paths = sorted(
        path for path in output.split(b"\0") if path and path.endswith(b".py")
    )
    digest = hashlib.sha256()
    for encoded_path in paths:
        digest.update(encoded_path)
        digest.update(b"\0")
        path = root / os.fsdecode(encoded_path)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def source_provenance(
    root: Path,
    source_files: dict[str, Path],
    *,
    expected_commit: str | None = None,
    pathspecs: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Fingerprint a checkout, including tracked and untracked Python edits."""

    root = root.resolve()
    status_command = ["status", "--porcelain", "--untracked-files=all"]
    if pathspecs:
        status_command.extend(("--", *pathspecs))
    status_output = _git_output(root, *status_command)
    assert isinstance(status_output, str)
    status = status_output.splitlines()

    diff_command = ["diff", "--binary", "HEAD", "--", *pathspecs]
    diff = _git_output(root, *diff_command, text=False)
    assert isinstance(diff, bytes)
    commit = _git_output(root, "rev-parse", "HEAD")
    remote = subprocess.run(
        ("git", "remote", "get-url", "origin"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert isinstance(commit, str)
    commit = commit.strip()
    return {
        "root": str(root),
        "commit": commit,
        "dirty": bool(status),
        "status": status,
        "status_includes_untracked": True,
        "pathspecs": list(pathspecs),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_python_sha256": _git_untracked_python_sha256(root, pathspecs),
        "remote": remote or None,
        "expected_commit": expected_commit,
        "commit_matches_pin": (
            None if expected_commit is None else commit == expected_commit
        ),
        "pinned_clean": bool(expected_commit == commit and not status),
        "sources": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path.resolve())}
            for name, path in sorted(source_files.items())
        },
    }


def validate_pinned_source(name: str, provenance: dict[str, Any]) -> None:
    expected_commit = provenance["expected_commit"]
    if expected_commit is None:
        raise RuntimeError(f"{name} source requires an explicit commit pin")
    if not provenance["commit_matches_pin"]:
        raise RuntimeError(
            f"expected {name} SHA {expected_commit}, got {provenance['commit']}"
        )
    if provenance["dirty"]:
        raise RuntimeError(
            f"{name} source must be clean; git status was {provenance['status']}"
        )


def assert_stable(label: str, before: object, after: object) -> None:
    if before != after:
        raise RuntimeError(f"{label} changed: before={before}, after={after}")


def _physical_gpu_index() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = visible.split(",", 1)[0].strip()
    return first or "0"


def _gpu_field(field: str) -> float | None:
    try:
        output = subprocess.run(
            (
                "nvidia-smi",
                "-i",
                _physical_gpu_index(),
                f"--query-gpu={field}",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        return float(output.splitlines()[0])
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        return None


def gpu_info() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "multiprocessor_count": properties.multi_processor_count,
        "l2_cache_bytes": l2_cache_bytes(properties),
        "power_limit_w": _gpu_field("power.limit"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def l2_cache_bytes(properties: object) -> int:
    value = getattr(properties, "L2_cache_size", None)
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            "PyTorch CUDA device properties did not provide a positive "
            "L2_cache_size; refusing to claim cold-L2 timing"
        )
    return value


def wait_for_cooldown(target_c: float, timeout_s: float) -> dict[str, Any]:
    torch.cuda.synchronize()
    start = _gpu_field("temperature.gpu")
    current = start
    started = time.monotonic()
    while current is not None and current > target_c:
        if time.monotonic() - started >= timeout_s:
            break
        time.sleep(5)
        current = _gpu_field("temperature.gpu")
    if current is None:
        raise RuntimeError("GPU temperature is unavailable; benchmark is invalid")
    if current > target_c:
        raise TimeoutError(
            f"GPU remained at {current} C above the {target_c} C limit after "
            f"{time.monotonic() - started:.1f} s"
        )
    return {
        "start_temp_c": start,
        "end_temp_c": current,
        "waited_s": round(time.monotonic() - started, 1),
        "target_c": target_c,
        "reached_target": True,
    }


def paired_graph_order(
    cycle_index: int, baseline: str, candidate: str
) -> tuple[str, str, str, str]:
    """Return a balanced ABBA/BAAB order for one measurement cycle."""

    if cycle_index % 2 == 0:
        return (baseline, candidate, candidate, baseline)
    return (candidate, baseline, baseline, candidate)


def capture_paired_cuda_graphs(
    arms: tuple[PairedGraphArm, PairedGraphArm],
    stream: torch.cuda.Stream,
) -> dict[str, _CapturedGraphArm]:
    """Capture exactly one provider invocation in each private CUDA graph."""

    captured: dict[str, _CapturedGraphArm] = {}
    for arm in arms:
        # CUDA graph capture must be warmed on a non-default stream. Both the
        # warmup and reset are outside every subsequently measured interval.
        with torch.cuda.stream(stream):
            arm.reset()
            arm.launch()
        stream.synchronize()
        with torch.cuda.stream(stream):
            arm.reset()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            arm.launch()
        stream.synchronize()
        captured[arm.name] = _CapturedGraphArm(arm=arm, graph=graph)
    return captured


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sample")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    median = statistics.median(values)
    return {
        "count": len(values),
        "min": min(values),
        "p10": _percentile(values, 0.10),
        "p25": _percentile(values, 0.25),
        "median": median,
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "max": max(values),
        "mad": statistics.median(abs(value - median) for value in values),
    }


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def measure_paired_cuda_graphs(
    captured: dict[str, _CapturedGraphArm],
    *,
    baseline: str,
    candidate: str,
    measurement_cycles: int,
    primer_cycles: int,
    cooldown_temp_c: float,
    cooldown_timeout_s: float,
    stream: torch.cuda.Stream,
) -> dict[str, Any]:
    """Measure one graph replay at a time with a balanced cold-L2 schedule."""

    if set(captured) != {baseline, candidate}:
        raise ValueError("paired timing requires exactly the named two graph arms")
    if measurement_cycles < 2 or measurement_cycles % 2:
        raise ValueError("paired graph measurement cycles must be an even integer >= 2")
    if primer_cycles < 1:
        raise ValueError("paired graph primer cycles must be positive")

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    cache_bytes = l2_cache_bytes(properties)
    flush = torch.zeros(2 * cache_bytes, device="cuda", dtype=torch.uint8)

    def prepare_and_replay(name: str) -> None:
        captured[name].arm.reset()
        flush.add_(1)
        captured[name].graph.replay()

    for cycle_index in range(primer_cycles):
        with torch.cuda.stream(stream):
            for name in paired_graph_order(cycle_index, baseline, candidate):
                prepare_and_replay(name)
    stream.synchronize()
    cooldown = wait_for_cooldown(cooldown_temp_c, cooldown_timeout_s)

    blocks: list[dict[str, Any]] = []
    samples_by_name = {baseline: [], candidate: []}
    all_pairs: list[dict[str, Any]] = []
    for cycle_index in range(measurement_cycles):
        order = paired_graph_order(cycle_index, baseline, candidate)
        events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in order
        ]
        with torch.cuda.stream(stream):
            for name, (start, end) in zip(order, events, strict=True):
                # State/output restoration and a 2x-L2 sweep precede the start
                # event on the same stream and are therefore outside timing.
                captured[name].arm.reset()
                flush.add_(1)
                start.record(stream)
                captured[name].graph.replay()
                end.record(stream)
        events[-1][1].synchronize()

        observations: list[dict[str, Any]] = []
        for position, (name, (start, end)) in enumerate(
            zip(order, events, strict=True)
        ):
            latency_ms = float(start.elapsed_time(end))
            if not math.isfinite(latency_ms) or latency_ms <= 0.0:
                raise RuntimeError(
                    f"invalid paired CUDA-graph timing sample: {latency_ms}"
                )
            samples_by_name[name].append(latency_ms)
            observations.append(
                {
                    "implementation": name,
                    "position": position,
                    "latency_ms": latency_ms,
                }
            )

        pairs: list[dict[str, Any]] = []
        for offset in (0, 2):
            first, second = observations[offset : offset + 2]
            by_name = {
                first["implementation"]: first["latency_ms"],
                second["implementation"]: second["latency_ms"],
            }
            baseline_ms = float(by_name[baseline])
            candidate_ms = float(by_name[candidate])
            pair = {
                "cycle_index": cycle_index,
                "pair_in_cycle": offset // 2,
                "order": (
                    "baseline-first"
                    if first["implementation"] == baseline
                    else "candidate-first"
                ),
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "candidate_speedup_over_baseline": baseline_ms / candidate_ms,
                "candidate_over_baseline_latency_ratio": candidate_ms / baseline_ms,
            }
            pairs.append(pair)
            all_pairs.append(pair)
        blocks.append(
            {
                "cycle_index": cycle_index,
                "order": list(order),
                "observations": observations,
                "pairs": pairs,
            }
        )

    speedups = [float(pair["candidate_speedup_over_baseline"]) for pair in all_pairs]
    latency_ratios = [
        float(pair["candidate_over_baseline_latency_ratio"]) for pair in all_pairs
    ]
    return {
        "timer": "torch.cuda.Event around one CUDA-graph replay",
        "timing_mode": "cuda-graph-one-invocation",
        "ordering_mode": "same-process balanced ABBA/BAAB crossover",
        "measurement_cycles": measurement_cycles,
        "primer_cycles": primer_cycles,
        "samples_per_implementation": 2 * measurement_cycles,
        "l2_cache_bytes": cache_bytes,
        "l2_flush_bytes": flush.numel(),
        "cache_mode": "cold",
        "state_reset": "pristine-copy-and-output-zero-before-each-replay",
        "l2_flush_placement": "2x-L2-sweep-before-start-event-for-every-replay",
        "cooldown": cooldown,
        "implementations": {
            name: {
                "median_ms": statistics.median(samples),
                "best_ms": min(samples),
                "samples_ms": samples,
                "distribution_ms": _sample_summary(samples),
            }
            for name, samples in samples_by_name.items()
        },
        "paired_statistics": {
            "baseline": baseline,
            "candidate": candidate,
            "pairs": len(all_pairs),
            "candidate_faster_pairs": sum(value > 1.0 for value in speedups),
            "candidate_speedup_over_baseline_geomean": _geometric_mean(speedups),
            "candidate_over_baseline_latency_ratio_geomean": _geometric_mean(
                latency_ratios
            ),
            "speedup_distribution": _sample_summary(speedups),
            "by_order": {
                order: _sample_summary(
                    [
                        float(pair["candidate_speedup_over_baseline"])
                        for pair in all_pairs
                        if pair["order"] == order
                    ]
                )
                for order in ("baseline-first", "candidate-first")
            },
        },
        "blocks": blocks,
    }
