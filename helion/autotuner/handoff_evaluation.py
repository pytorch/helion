"""Evaluate editable standalone kernels against a frozen handoff workload."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast
import uuid

import torch
from torch.utils._pytree import tree_flatten

from .accuracy import assert_close
from .accuracy import is_fp8_dtype
from .benchmark_provider import _aggregate_multi_shape_timings
from .benchmark_provider import _clone_args
from .benchmark_worker import BenchmarkTimeout
from .benchmark_worker import BenchmarkWorker
from .benchmark_worker import BenchmarkWorkerUnkillable
from .benchmarking import do_bench
from .benchmarking import do_bench_cuda_graph
from .benchmarking import do_bench_generic
from .benchmarking import synchronize_device
from .logger import capture_output
from .precompile_future import SerializedCompiledFunction
from .precompile_future import _load_compiled_fn
from .precompile_future import _unload_compiled_fn

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

    from .benchmark_provider import MultiShapeAggregation


@dataclasses.dataclass(frozen=True)
class HandoffCaseResult:
    """Correctness and latency for one immutable source snapshot.

    ``samples``, their median ``perf``, and median absolute deviation ``noise``
    are in milliseconds, even when the aggregate objective is a ratio.
    """

    index: int
    status: str
    source_hash: str | None
    source_path: str | None
    timing: str
    samples: tuple[float, ...] = ()
    perf: float | None = None
    noise: float | None = None
    error: str | None = None
    phase: str | None = None
    progress_path: str | None = None


@dataclasses.dataclass(frozen=True)
class HandoffEvaluation:
    """A recorded evaluation; ``perf`` exists only when every case succeeds."""

    ok: bool
    perf: float | None
    unit: str
    cases: tuple[HandoffCaseResult, ...]
    directory: str
    created_at: str
    completed_at: str | None = None


@dataclasses.dataclass(frozen=True)
class _CaseOutcome:
    status: str
    samples: tuple[float, ...] = ()
    error: str | None = None
    phase: str | None = None


@dataclasses.dataclass
class _EvaluateCase:
    source_path: str
    source_hash: str
    entrypoint: str
    inputs_path: str
    reference_path: str
    timing: str
    atol: float
    rtol: float
    scale_atol: bool
    repetitions: int
    alias_signature: list[int] | None = None
    progress_path: str | None = None

    def __call__(self) -> _CaseOutcome:
        # Each measured call receives its original inputs; implicit graph
        # warmups/replays would add unaccounted calls on mutated inputs.
        os.environ["HELION_BENCHMARK_CUDAGRAPH"] = "0"
        source = Path(self.source_path).read_text()
        status = "load_error"
        samples: list[float] = []
        error = None
        fn = None
        phase_name = "load"

        def phase(name: str) -> None:
            nonlocal phase_name
            phase_name = name
            if self.progress_path is not None:
                with Path(self.progress_path).open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "phase": name,
                                "timestamp": time.time(),
                                "samples": samples,
                            }
                        )
                        + "\n"
                    )

        with capture_output() as captured:
            try:
                phase("load")
                _check_standalone(source)
                # These are trusted artifacts made by build_handoff. Avoid
                # cached loaders: one candidate must never inherit mutations.
                inputs = cast(
                    "Sequence[object]", torch.load(self.inputs_path, weights_only=False)
                )
                reference, post_args = torch.load(
                    self.reference_path, weights_only=False
                )
                aliases = self.alias_signature
                if aliases is None:
                    aliases = _alias_signature((reference, post_args))

                def check(output: object, args: Sequence[object]) -> None:
                    phase("validation")
                    _check_result(
                        output,
                        args,
                        reference,
                        post_args,
                        inputs,
                        aliases,
                        self.atol,
                        self.rtol,
                        self.scale_atol,
                    )

                with _device_context(inputs), torch.no_grad():
                    fn = _load_compiled_fn(
                        SerializedCompiledFunction(
                            self.entrypoint,
                            source,
                            self.source_path,
                            None,
                            self.source_hash,
                        )
                    )
                    status = "error"
                    args = _clone_args(inputs, None, preserve_storage=True)
                    phase("compile_and_first_launch")
                    output = fn(*args)
                    synchronize_device()
                    check(output, args)
                    if self.timing == "cuda_graph":
                        reset = _input_reset(inputs, args)

                        def run() -> object:
                            nonlocal output
                            output = fn(*args)
                            return output

                        for _ in range(self.repetitions):
                            value = cast(
                                "float",
                                do_bench_cuda_graph(
                                    run,
                                    return_mode="median",
                                    pre_warmed=True,
                                    reset=reset,
                                    phase=phase,
                                ),
                            )
                            check(output, args)
                            samples.append(value)
                            phase("validated_sample")
                    else:
                        self._measure_isolated(fn, inputs, samples, phase, check)
                phase("complete")
                status = "ok"
            except Exception:
                # String diagnostics survive worker cleanup without retaining tensors.
                error = traceback.format_exc()
                if phase_name == "validation":
                    status = "accuracy_error"
            finally:
                if fn is not None:
                    _unload_compiled_fn(fn)
        if error and captured[0]:
            error += "\n" + captured[0]
        return _CaseOutcome(status, tuple(samples), error, phase_name)

    def _measure_isolated(
        self,
        fn: Callable[..., object],
        inputs: Sequence[object],
        samples: list[float],
        phase: Callable[[str], None],
        check: Callable[[object, Sequence[object]], None],
    ) -> None:
        """Replay legacy event/wall-clock policies with one fresh call per sample."""
        bench = do_bench if self.timing == "cuda_event" else do_bench_generic
        for _ in range(self.repetitions):
            args = _clone_args(inputs, None, preserve_storage=True)
            output: object = None

            def run(args: Sequence[object] = args) -> object:
                nonlocal output
                output = fn(*args)
                return output

            phase("measure")
            value = cast(
                "float",
                bench(
                    run,
                    fixed_repetitions=1,
                    pre_warmed=True,
                    return_mode="median",
                ),
            )
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Native kernel timing must be finite and positive")
            samples.append(value)
            check(output, args)


def _input_reset(
    inputs: Sequence[object], args: Sequence[object]
) -> Callable[[], None]:
    """Restore complete GPU storages, preserving offsets and overlapping views."""
    pairs = []
    seen = set()
    for original, actual in zip(
        tree_flatten(inputs)[0], tree_flatten(args)[0], strict=True
    ):
        if not isinstance(original, torch.Tensor):
            continue
        assert isinstance(actual, torch.Tensor)
        if original.device.type != "cuda":
            _assert_exact(actual, original)
            continue
        key = actual.untyped_storage()._cdata
        if key in seen:
            continue
        seen.add(key)
        shape = (original.untyped_storage().nbytes(),)
        source = torch.empty(0, dtype=torch.uint8, device=original.device).set_(
            original.untyped_storage(), 0, shape, (1,)
        )
        target = torch.empty(0, dtype=torch.uint8, device=actual.device).set_(
            actual.untyped_storage(), 0, shape, (1,)
        )
        pairs.append((target, source))

    def reset() -> None:
        for target, source in pairs:
            target.copy_(source)

    return reset


def _device_context(inputs: Sequence[object]) -> AbstractContextManager[object]:
    """Use the workload's device for launches, synchronization and timing."""
    leaves, _ = tree_flatten(inputs)
    devices = {
        value.device
        for value in leaves
        if isinstance(value, torch.Tensor) and value.device.type != "cpu"
    }
    if len(devices) > 1:
        raise ValueError(
            "Handoff evaluation requires a single accelerator device per case"
        )
    if devices:
        device = devices.pop()
        if device.type == "cuda":
            return torch.cuda.device(device)
        if device.type == "xpu":
            return torch.xpu.device(device)
    return contextlib.nullcontext()


def _alias_signature(values: object) -> list[int]:
    """Describe tensor storage aliases without retaining addresses or tensors."""
    leaves, _ = tree_flatten(values)
    groups: dict[int, int] = {}
    signature = []
    for value in leaves:
        if isinstance(value, torch.Tensor):
            storage_id = value.untyped_storage()._cdata
            signature.append(groups.setdefault(storage_id, len(groups)))
    return signature


def _assert_exact(actual: object, expected: torch.Tensor) -> None:
    if is_fp8_dtype(expected.dtype):
        expected = expected.view(torch.uint8)
        if isinstance(actual, torch.Tensor):
            actual = actual.view(torch.uint8)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


def _check_result(
    output: object,
    args: Sequence[object],
    reference: object,
    post_args: Sequence[object],
    inputs: Sequence[object],
    aliases: list[int],
    atol: float,
    rtol: float,
    scale_atol: bool,
) -> None:
    assert_close(output, reference, atol, rtol, scale_atol_by_expected_rms=scale_atol)
    actual_flat, actual_spec = tree_flatten(args)
    expected_flat, expected_spec = tree_flatten(post_args)
    pristine_flat, pristine_spec = tree_flatten(inputs)
    if actual_spec != expected_spec:
        raise AssertionError(
            "Post-call argument structure does not match the handoff reference"
        )
    for index, (actual, expected) in enumerate(
        zip(actual_flat, expected_flat, strict=True)
    ):
        if isinstance(expected, torch.Tensor) and expected_spec == pristine_spec:
            try:
                _assert_exact(pristine_flat[index], expected)
            except AssertionError:
                pass  # The reference intentionally changes this input.
            else:
                _assert_exact(actual, expected)
                continue
        assert_close(
            actual, expected, atol, rtol, scale_atol_by_expected_rms=scale_atol
        )
    if _alias_signature((output, args)) != aliases:
        raise AssertionError(
            "Output/input tensor aliases do not match the handoff kernel"
        )


def _check_standalone(source: str) -> None:
    """Reject direct Helion imports without rejecting the exported local shim."""
    for node in ast.walk(ast.parse(source)):
        names = (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
        if any(name == "helion" or name.startswith("helion.") for name in names):
            raise ValueError("Handoff source must run without importing Helion")


def _artifact_path(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(directory):
        raise ValueError("Handoff artifact paths must remain inside the bundle")
    return path


def evaluate_handoff(
    directory: str | Path,
    *,
    repetitions: int = 5,
    timeout: float = 120,
    deadline: float | None = None,
) -> HandoffEvaluation:
    """Snapshot and evaluate every native source in a handoff bundle.

    Compilation, correctness and timing run in a fresh killable worker. Every
    invocation uses fresh storage-preserving inputs; restore work is excluded
    from timing. Results and source snapshots remain in ``evaluations/`` even
    when an edited kernel fails. This evaluates one proposal without selecting
    winners, starting an agent, or changing the editable source files.
    ``deadline`` is an optional absolute ``time.monotonic()`` cutoff; no new
    case starts after it, and each worker timeout is capped by remaining time.
    """
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("repetitions must be a positive integer")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("deadline must be finite")
    directory = Path(directory).resolve()
    manifest_text = (directory / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    if manifest["schema_version"] != 1 or not manifest["cases"]:
        raise ValueError("Expected a version 1 handoff bundle with at least one case")
    objective = manifest["objective"]
    aggregation = objective["aggregation"]
    unit = objective["unit"]
    references = objective["reference_latencies"]
    if aggregation not in ("geomean", "max") or unit not in ("ms", "ratio"):
        raise ValueError("Unsupported handoff objective")
    if (unit == "ratio") != (references is not None):
        raise ValueError(
            "Ratio objectives require reference latencies; ms objectives do not"
        )
    if references is not None and (
        len(references) != len(manifest["cases"])
        or any(not math.isfinite(value) or value <= 0 for value in references)
    ):
        raise ValueError("Reference latencies must be positive and cover every case")
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output_directory = directory / "evaluations" / uuid.uuid4().hex
    output_directory.mkdir(parents=True)
    (output_directory / "manifest.json").write_text(manifest_text)
    snapshots: list[tuple[str, str] | Exception] = []
    # Freeze every source before starting the first worker case, so subsequent
    # workspace edits cannot change what this evaluation measures.
    for index, case in enumerate(manifest["cases"]):
        try:
            source = _artifact_path(directory, case["source"]).read_bytes()
            path = output_directory / f"case_{index}.py"
            path.write_bytes(source)
            snapshots.append((str(path), hashlib.sha256(source).hexdigest()))
        except (OSError, ValueError) as exception:
            snapshots.append(exception)
    results: list[HandoffCaseResult] = []
    worker = BenchmarkWorker(device=None)
    fatal_error = None
    try:
        for index, (case, snapshot) in enumerate(
            zip(manifest["cases"], snapshots, strict=True)
        ):
            timing = case["timing"]
            if isinstance(snapshot, Exception):
                results.append(
                    HandoffCaseResult(
                        index, "load_error", None, None, timing, error=str(snapshot)
                    )
                )
                continue
            path, source_hash = snapshot
            outcome = _CaseOutcome("error", error=fatal_error)
            progress_path = output_directory / f"case_{index}.progress.jsonl"
            try:
                if fatal_error is None:
                    if timing not in ("cuda_graph", "cuda_event", "wall_clock"):
                        raise ValueError(f"Unsupported handoff timing mode: {timing}")
                    remaining = (
                        timeout
                        if deadline is None
                        else min(timeout, deadline - time.monotonic())
                    )
                    if remaining <= 0:
                        raise BenchmarkTimeout(
                            "handoff deadline exceeded before case started"
                        )
                    outcome = worker.run(
                        _EvaluateCase(
                            path,
                            source_hash,
                            case["entrypoint"],
                            str(_artifact_path(directory, case["inputs"])),
                            str(_artifact_path(directory, case["reference"])),
                            timing,
                            case["atol"],
                            case["rtol"],
                            case["scale_atol"],
                            repetitions,
                            case.get("alias_signature"),
                            str(progress_path),
                        ),
                        timeout=remaining,
                    )
                    if outcome.status != "ok":
                        # A failed launch may poison its CUDA context even when
                        # Python recovered enough to return a diagnostic.
                        worker.shutdown()
            except BenchmarkWorkerUnkillable as exception:
                fatal_error = str(exception)
                outcome = _CaseOutcome("error", error=fatal_error)
            except BenchmarkTimeout as exception:
                outcome = _CaseOutcome("timeout", error=str(exception))
            except Exception as exception:
                outcome = _CaseOutcome(
                    "error", error=f"{type(exception).__qualname__}: {exception}"
                )
            last_phase = outcome.phase
            if last_phase is None and progress_path.exists():
                # A killed worker can leave a partial final line; retain its
                # most recent complete phase and partial timing samples.
                lines = progress_path.read_text().splitlines(keepends=True)
                for line in reversed(lines):
                    if line.endswith("\n"):
                        progress = json.loads(line)
                        last_phase = progress["phase"]
                        outcome = dataclasses.replace(
                            outcome, samples=tuple(progress.get("samples", ()))
                        )
                        break
            perf = (
                statistics.median(outcome.samples) if outcome.status == "ok" else None
            )
            noise = (
                statistics.median(abs(sample - perf) for sample in outcome.samples)
                if perf is not None
                else None
            )
            results.append(
                HandoffCaseResult(
                    index,
                    outcome.status,
                    source_hash,
                    path,
                    timing,
                    outcome.samples,
                    perf,
                    noise,
                    outcome.error,
                    last_phase,
                    str(progress_path),
                )
            )
    finally:
        if fatal_error is None:
            worker.shutdown()
    ok = all(result.status == "ok" for result in results)
    perf = (
        _aggregate_multi_shape_timings(
            [cast("float", result.perf) for result in results],
            aggregation=cast("MultiShapeAggregation", aggregation),
            references=references,
        )
        if ok
        else None
    )
    evaluation = HandoffEvaluation(
        ok,
        perf,
        unit,
        tuple(results),
        str(output_directory),
        created_at,
        datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    (output_directory / "evaluation.json").write_text(
        json.dumps(dataclasses.asdict(evaluation), indent=2, allow_nan=False) + "\n"
    )
    return evaluation
