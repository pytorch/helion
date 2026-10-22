"""Opt-in, verified CUDA graph scoring for standalone autotuning searches."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import math
import os
import pickle
import statistics
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch.utils._pytree import tree_flatten

from . import benchmarking
from .accuracy import assert_close
from .benchmark_job import BenchmarkJob
from .benchmark_job import _load_compiled_fn_for_worker
from .benchmark_job import _load_trusted_baseline_output
from .benchmark_worker import BenchmarkWorker
from .kernel_args import _clone_tensor_leaves
from .kernel_args import load_private_kernel_args
from .kernel_args import load_trusted_kernel_args
from .logger import capture_output
from .precompile_future import _serialize_compiled_fn
from .precompile_future import _unload_compiled_fn

if TYPE_CHECKING:
    from ..runtime.kernel import CompiledConfig
    from .benchmark_provider import LocalBenchmarkProvider
    from .precompile_future import SerializedCompiledFunction

GRAPH_SCORING_POLICY_VERSION = 1
TIMER = "cuda_graph_external_events"
OutputCheck = Callable[[Sequence[object], object, object], None]
InputCheck = Callable[[Sequence[object], Sequence[object]], None]
OutputPoisoner = Callable[[object], int]


class GraphScoringUnavailable(RuntimeError):
    """The requested graph scoring path cannot run without a fallback."""


def _has_cuda_tensors(arguments: Sequence[object]) -> bool:
    return any(
        isinstance(leaf, torch.Tensor) and leaf.is_cuda
        for leaf in tree_flatten(arguments)[0]
    )


def poison_output(output: object) -> int:
    """Poison floating outputs after capture and before the first replay."""
    tensors = [
        leaf
        for leaf in tree_flatten(output)[0]
        if isinstance(leaf, torch.Tensor) and leaf.is_floating_point()
    ]
    if not tensors:
        raise GraphScoringUnavailable("Graph scoring requires floating outputs")
    cuda = any(tensor.is_cuda for tensor in tensors)
    if cuda:
        torch.cuda.synchronize()
    for tensor in tensors:
        tensor.fill_(float("nan"))
    if cuda:
        torch.cuda.synchronize()
    return len(tensors)


def poison_indexed_output(output: object) -> int:
    """Poison values and Int32 indices, including an entirely padded result."""
    if not (
        isinstance(output, tuple)
        and len(output) == 2
        and all(isinstance(leaf, torch.Tensor) for leaf in output)
        and output[0].is_floating_point()
        and output[1].dtype == torch.int32
        and output[0].shape == output[1].shape
    ):
        raise GraphScoringUnavailable("Expected floating values and Int32 indices")
    values, indices = cast("tuple[torch.Tensor, torch.Tensor]", output)
    cuda = values.is_cuda or indices.is_cuda
    if cuda:
        torch.cuda.synchronize()
    values.fill_(float("nan"))
    # -1 can be valid padding. INT32_MIN must be rejected by the output oracle.
    indices.fill_(torch.iinfo(torch.int32).min)
    if cuda:
        torch.cuda.synchronize()
    return 2


def assert_preserved_input_bits(
    pristine: Sequence[object], arguments: Sequence[object]
) -> None:
    """Require identical tensor bits, strides, offsets and pytree structure."""
    before, before_spec = tree_flatten(pristine)
    after, after_spec = tree_flatten(arguments)
    if before_spec != after_spec:
        raise AssertionError("Graph candidate changed the input structure")
    for left, right in zip(before, after, strict=True):
        if not isinstance(left, torch.Tensor):
            continue
        if not (
            isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and left.stride() == right.stride()
            and left.storage_offset() == right.storage_offset()
            and torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8),
                right.contiguous().reshape(-1).view(torch.uint8),
            )
        ):
            raise AssertionError("Graph candidate changed a preserved input tensor")


@dataclasses.dataclass(frozen=True)
class GraphScoringPolicy:
    check_output: OutputCheck | None = None
    poison_output: OutputPoisoner = poison_output
    check_inputs: InputCheck = assert_preserved_input_bits


@dataclasses.dataclass
class CudaGraphScoring:
    policy: GraphScoringPolicy
    evidence: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def require_provider(self, provider: LocalBenchmarkProvider) -> None:
        """Reject unsupported paths before they could launch a candidate."""
        if (
            provider.config_spec.backend.name not in ("cute", "triton")
            or provider.kernel.env.process_group_name is not None
            or not provider.kernel.supports_subprocess_benchmark()
            or provider.mutated_arg_indices
            or provider._args_unpicklable
            or provider._subprocess_wrapper_unloadable
            or provider._subprocess_benchmark_uses_wall_clock()
            or provider.settings.autotune_benchmark_fn is not None
            or provider.settings.autotune_handoff
        ):
            raise GraphScoringUnavailable(
                "Graph scoring requires standalone read-only CUDA kernels "
                "with serializable arguments and the default benchmark function"
            )
        if (
            provider.settings.autotune_baseline_accuracy_check_fn is not None
            and self.policy.check_output is None
        ):
            raise GraphScoringUnavailable(
                "Supply check_output to run a custom accuracy oracle in the worker"
            )
        timeout = float(provider.settings.autotune_benchmark_timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise GraphScoringUnavailable("Graph worker timeout must be positive")
        if not _has_cuda_tensors(provider.args):
            raise GraphScoringUnavailable("Graph scoring requires CUDA arguments")

    def record(self, result: dict[str, Any]) -> None:
        checked_evidence(result)
        self.evidence.append(
            {key: value for key, value in result.items() if key != "trace"}
        )


_ACTIVE_SCORING: contextvars.ContextVar[CudaGraphScoring | None] = (
    contextvars.ContextVar("helion_cuda_graph_scoring", default=None)
)


def current_cuda_graph_scoring() -> CudaGraphScoring | None:
    return _ACTIVE_SCORING.get()


@contextlib.contextmanager
def cuda_graph_scoring(
    *,
    check_output: OutputCheck | None = None,
    poison_output: OutputPoisoner = poison_output,
    check_inputs: InputCheck = assert_preserved_input_bits,
) -> Iterator[CudaGraphScoring]:
    """Score new searches with verified device graph timings in killable workers.

    Callbacks must be importable module-level functions. ``check_output`` receives
    pristine arguments, the replayed output and the stock accuracy baseline.
    Without it, the ordinary baseline tolerances apply. Baseline construction
    remains the stock reference/mutation check; it does not rank candidates.
    Keep this context active throughout tuning. Multi-shape and handoff searches
    are unsupported. Existing configured kernels and cache hits do not retune.
    """
    for callback in (check_output, poison_output, check_inputs):
        if callback is None:
            continue
        if not inspect.isfunction(callback) or callback.__closure__:
            raise GraphScoringUnavailable(
                "Graph callbacks must be module-level functions"
            )
        try:
            pickle.dumps(callback)
        except (AttributeError, TypeError, pickle.PicklingError) as error:
            raise GraphScoringUnavailable(
                "Graph callback cannot be imported by the worker"
            ) from error
    scoring = CudaGraphScoring(
        GraphScoringPolicy(check_output, poison_output, check_inputs)
    )
    token = _ACTIVE_SCORING.set(scoring)
    try:
        yield scoring
    finally:
        _ACTIVE_SCORING.reset(token)


def timing_result(samples: Sequence[float], *, captures: int) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if (
        captures < 1
        or not values
        or any(not math.isfinite(value) or value <= 0 for value in values)
    ):
        raise ValueError("Graph scoring requires a capture and positive device samples")
    return {
        "timer": TIMER,
        "graph_scoring_policy_version": GRAPH_SCORING_POLICY_VERSION,
        "latency_ms": statistics.median(values),
        "sample_count": len(values),
        "captures": captures,
        "cold_l2_cache": True,
        "eager_fallback": False,
    }


def checked_evidence(result: dict[str, Any]) -> None:
    if any(
        (
            result.get("timer") != TIMER,
            result.get("graph_scoring_policy_version") != GRAPH_SCORING_POLICY_VERSION,
            result.get("eager_fallback") is not False,
            result.get("cold_l2_cache") is not True,
            result.get("graph_output_accuracy") != "ok",
            result.get("preserved_inputs") != "ok",
            not (type(result.get("captures")) is int and result["captures"] > 0),
            not (
                type(result.get("sample_count")) is int and result["sample_count"] > 0
            ),
            not (
                type(result.get("poisoned_outputs")) is int
                and result["poisoned_outputs"] >= result.get("captures", 1)
            ),
        )
    ):
        raise GraphScoringUnavailable("Missing verified graph scoring evidence")


def checked_result(result: dict[str, Any]) -> float:
    checked_evidence(result)
    value = float(result["latency_ms"])
    if not math.isfinite(value) or value <= 0:
        raise GraphScoringUnavailable("Invalid graph device latency")
    return value


def run_graph_job(
    provider: LocalBenchmarkProvider,
    fns: Sequence[CompiledConfig],
    *,
    warmup: int = 1,
    rep: int = 50,
    fixed_repetitions: int | None = None,
    args_slot: int | None = None,
    timeout: float | None = None,
    repeat: int | None = None,
) -> dict[str, Any]:
    """Submit a verified job using the provider's ordinary killable worker."""
    scoring = provider._graph_scoring
    assert scoring is not None
    scoring.require_provider(provider)
    if (
        not fns
        or provider._precompile_args_path is None
        or provider._precompile_baseline_path is None
    ):
        raise GraphScoringUnavailable(
            "Graph worker functions/arguments/baseline are unavailable"
        )
    specs = tuple(_serialize_compiled_fn(fn) for fn in fns)
    job = GraphBenchmarkJob(
        benchmark=BenchmarkJob(
            fn_spec=specs[0],
            args_path=provider._precompile_args_path,
            warmup=warmup,
            rep=rep,
            fixed_repetitions=fixed_repetitions,
            args_slot=args_slot,
        ),
        policy=scoring.policy,
        baseline_path=provider._precompile_baseline_path,
        atol=provider._effective_atol,
        rtol=provider._effective_rtol,
        scale_atol=provider._scale_atol,
        other_fn_specs=specs[1:],
        repeat=repeat,
    )
    if provider._benchmark_worker is None:
        provider._benchmark_worker = BenchmarkWorker(device=None)
    result = cast(
        "dict[str, Any]",
        provider._benchmark_worker.run(
            job,
            timeout=float(provider.settings.autotune_benchmark_timeout)
            if timeout is None
            else timeout,
        ),
    )
    if repeat is None:
        checked_result(result)
    else:
        trace = result.get("trace")
        if (
            not isinstance(trace, benchmarking.MirroredBenchmarkTrace)
            or len(trace.medians_ms) != len(fns)
            or result.get("captures") != len(fns)
        ):
            raise GraphScoringUnavailable("Incomplete mirrored graph cohort")
        for value in trace.medians_ms:
            checked_result({**result, "latency_ms": value})
    scoring.record(result)
    return result


def strict_benchmark(
    fn: Callable[[], object],
    *,
    warmup: int,
    rep: int,
    fixed_repetitions: int | None = None,
    poison: OutputPoisoner = poison_output,
) -> tuple[dict[str, Any], object]:
    captured_outputs: list[object] = []
    phases: list[str] = []
    poisoned_outputs = 0

    def phase(name: str) -> None:
        nonlocal poisoned_outputs
        phases.append(name)
        if name == "warmup":
            if len(captured_outputs) != 1:
                raise GraphScoringUnavailable(
                    "Graph timer did not retain one captured output"
                )
            poisoned_outputs = poison(captured_outputs[0])

    def retain_output() -> object:
        output = fn()
        if torch.cuda.is_current_stream_capturing():
            captured_outputs.append(output)
        return output

    samples = benchmarking.do_bench_cuda_graph(
        retain_output,
        return_mode="all",
        warmup=warmup,
        rep=rep,
        fixed_repetitions=fixed_repetitions,
        phase=phase,
    )
    if len(captured_outputs) != 1 or phases != ["capture", "warmup", "measure"]:
        raise GraphScoringUnavailable(
            "Graph timer did not complete one verified capture"
        )
    result = timing_result(cast("Sequence[float]", samples), captures=1)
    result.update(phases=phases, poisoned_outputs=poisoned_outputs)
    return result, captured_outputs[0]


class GraphReplay:
    """One graph, retained output, stream and external event pair."""

    def __init__(
        self, fn: Callable[[], object], poison: OutputPoisoner = poison_output
    ) -> None:
        from ..runtime import cute_cuda_graph

        reason = benchmarking._cudagraph_unavailable_reason()
        if reason is not None:
            raise GraphScoringUnavailable(
                f"Strict graph scoring is unavailable: {reason}"
            )
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: object = None
        self.sample_count = 0
        # Captured clearing nodes hold a raw buffer pointer. Keep the callable
        # (and its tensor owner) alive until the graph has been reset.
        self._clear_l2: Callable[[], None] | None = (
            benchmarking._make_l2_cache_clearer()
        )
        with torch.cuda.stream(self.stream):
            fn()
            self.start.record()
            self.end.record()
        self.stream.synchronize()
        try:
            with cute_cuda_graph(stream=self.stream) as graph:
                self.graph = graph
                self._clear_l2()
                self.start.record()
                if not torch.cuda.is_current_stream_capturing():
                    raise GraphScoringUnavailable("Graph context did not start capture")
                self.output = fn()
                self.end.record()
            self.poisoned_outputs = poison(self.output)
        except BaseException:
            with contextlib.suppress(Exception):
                self.close()
            raise

    def sample(self) -> float:
        assert self.graph is not None
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        self.stream.synchronize()
        value = float(self.start.elapsed_time(self.end))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid graph device sample: {value}")
        self.sample_count += 1
        return value

    def close(self) -> None:
        try:
            self.stream.synchronize()
        finally:
            if self.graph is not None:
                self.graph.reset()
            self.graph = None
            self.output = None
            self._clear_l2 = None


def mirrored_samples(
    samplers: Sequence[GraphReplay],
    *,
    repeat: int,
    after_call: Callable[[int], None] | None = None,
) -> benchmarking.MirroredBenchmarkTrace:
    if not samplers:
        return benchmarking.MirroredBenchmarkTrace([], [], [])
    sweeps, calls_per_sample, total_calls = benchmarking._mirrored_bench_call_layout(
        repeat
    )
    per_member: list[list[float]] = [[] for _ in samplers]
    orders: list[list[int]] = []
    elapsed_ms: list[list[float]] = []
    indices = list(range(len(samplers)))
    for sweep in range(sweeps):
        offset = (sweep // 2) % len(indices)
        rotated = indices[offset:] + indices[:offset]
        order = rotated if sweep % 2 == 0 else list(reversed(rotated))
        sweep_samples = []
        for index in order:
            values = []
            for _ in range(calls_per_sample):
                try:
                    value = float(samplers[index].sample())
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("Invalid mirrored graph sample")
                    values.append(value)
                finally:
                    if after_call is not None:
                        after_call(index)
            value = statistics.mean(values)
            per_member[index].append(value)
            sweep_samples.append(value)
        orders.append(order)
        elapsed_ms.append(sweep_samples)
    return benchmarking.MirroredBenchmarkTrace(
        orders,
        elapsed_ms,
        [statistics.median(values) for values in per_member],
        sweep_count=sweeps,
        calls_per_sample=calls_per_sample,
        total_calls=total_calls,
    )


@dataclasses.dataclass
class GraphBenchmarkJob:
    """Use the ordinary worker reconstruction/arguments with replay checks."""

    benchmark: BenchmarkJob
    policy: GraphScoringPolicy = dataclasses.field(default_factory=GraphScoringPolicy)
    baseline_path: str | None = None
    atol: float = 0.0
    rtol: float = 0.0
    scale_atol: bool = False
    other_fn_specs: tuple[SerializedCompiledFunction, ...] = ()
    repeat: int | None = None

    def _check(
        self, args: Sequence[object], pristine: Sequence[object], output: object
    ) -> None:
        assert self.baseline_path is not None
        baseline = _load_trusted_baseline_output(self.baseline_path)
        benchmarking.synchronize_device()
        if self.policy.check_output is not None:
            self.policy.check_output(pristine, output, baseline)
        else:
            assert_close(
                output,
                baseline,
                atol=self.atol,
                rtol=self.rtol,
                scale_atol_by_expected_rms=self.scale_atol,
            )
        assert_preserved_input_bits(pristine, args)
        if self.policy.check_inputs is not assert_preserved_input_bits:
            self.policy.check_inputs(pristine, args)

    def __call__(self) -> dict[str, Any]:
        job = self.benchmark
        if job.use_wall_clock or self.baseline_path is None:
            raise GraphScoringUnavailable(
                "Graph jobs require a baseline and device timing"
            )
        with capture_output():
            args = (
                load_private_kernel_args(job.args_path, job.args_slot)
                if job.args_slot is not None
                else _clone_tensor_leaves(load_trusted_kernel_args(job.args_path))
            )
            pristine = _clone_tensor_leaves(args)
            if not _has_cuda_tensors(args):
                raise GraphScoringUnavailable(
                    "Graph jobs require actual CUDA arguments"
                )
            functions: list[CompiledConfig] = []
            samplers: list[GraphReplay] = []
            try:
                for spec in (job.fn_spec, *self.other_fn_specs):
                    functions.append(_load_compiled_fn_for_worker(spec))
                calls = [functools.partial(fn, *args) for fn in functions]
                if self.repeat is None:
                    if len(calls) != 1:
                        raise GraphScoringUnavailable(
                            "Single graph scoring requires one function"
                        )
                    result, output = strict_benchmark(
                        calls[0],
                        warmup=job.warmup,
                        rep=job.rep,
                        fixed_repetitions=job.fixed_repetitions,
                        poison=self.policy.poison_output,
                    )
                    self._check(args, pristine, output)
                else:
                    for call in calls:
                        sampler = GraphReplay(call, self.policy.poison_output)
                        samplers.append(sampler)
                        sampler.sample()
                        self._check(args, pristine, sampler.output)
                    trace = mirrored_samples(
                        samplers,
                        repeat=self.repeat,
                        after_call=lambda index: (
                            benchmarking.clear_jit_fast_path_caches(functions[index])
                        ),
                    )
                    for sampler in samplers:
                        self._check(args, pristine, sampler.output)
                    result = {
                        "timer": TIMER,
                        "graph_scoring_policy_version": GRAPH_SCORING_POLICY_VERSION,
                        "eager_fallback": False,
                        "cold_l2_cache": True,
                        "captures": len(samplers),
                        "sample_count": sum(
                            sampler.sample_count for sampler in samplers
                        ),
                        "poisoned_outputs": sum(
                            sampler.poisoned_outputs for sampler in samplers
                        ),
                        "trace": trace,
                    }
                result.update(
                    graph_output_accuracy="ok",
                    preserved_inputs="ok",
                    worker_pid=os.getpid(),
                )
                checked_evidence(result)
                return result
            finally:
                # Graph nodes must die before their generated modules/owners.
                primary = sys.exc_info()[1]
                failures = []
                for sampler in samplers:
                    try:
                        sampler.close()
                    except Exception as error:
                        failures.append(error)
                for fn in functions:
                    try:
                        _unload_compiled_fn(fn)
                    except Exception as error:
                        failures.append(error)
                if failures and primary is None:
                    raise failures[0]
