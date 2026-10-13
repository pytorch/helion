"""Explicit search timing domain and isolated-worker measurement evidence.

The default search does not import the optional profiler collector. These records
are used only by the opt-in job/provider path; they do not replace backend timers.
"""

from __future__ import annotations

import dataclasses
import hashlib
from importlib.metadata import distribution
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NoReturn

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..runtime.settings import Settings
    from .profiler_timing import CallableIdentity
    from .profiler_timing import ProfilerTimingObservation
    from .profiler_timing import ProfilerTrace


def timing_error(message: str) -> NoReturn:
    from .profiler_timing import ProfilerTimingCapabilityError

    raise ProfilerTimingCapabilityError(message)


def raise_timing_error(
    error: BaseException, report: Callable[[BaseException], None] | None = None
) -> None:
    """Preserve the typed error across existing candidate-failure handlers."""
    from .profiler_timing import ProfilerTimingCapabilityError

    if isinstance(error, ProfilerTimingCapabilityError):
        if report is not None:
            report(error)
        raise error


@dataclasses.dataclass(frozen=True)
class SearchTimingPolicy:
    graph_enabled: bool
    torch_version: str
    cuda_version: str
    collector_digest: str
    triton_version: str
    clear_digest: str
    chunk_size: int = 128
    version: int = 1
    clear_bytes: int = 256 * 1024 * 1024

    @classmethod
    def create(cls) -> SearchTimingPolicy:
        from ..runtime.settings import _env_get_bool

        if torch.version.cuda is None or torch.version.hip is not None:
            timing_error("Profiler search timing requires NVIDIA CUDA")
        assert torch.version.cuda is not None
        installed_triton = distribution("triton")
        return cls(
            _env_get_bool("HELION_BENCHMARK_CUDAGRAPH", False),
            str(torch.__version__),
            torch.version.cuda,
            hashlib.sha256(
                Path(__file__).with_name("profiler_timing.py").read_bytes()
            ).hexdigest(),
            installed_triton.version,
            hashlib.sha256(
                Path(
                    str(
                        installed_triton.locate_file("triton/backends/nvidia/driver.py")
                    )
                ).read_bytes()
            ).hexdigest(),
        )

    def validate(self) -> None:
        if self != self.create():
            timing_error("Search timing policy or installed runtime changed")

    def validate_settings(self, settings: Settings) -> None:
        self.validate()
        if (
            settings.autotune_timing_method != "torch_profiler"
            or not settings.autotune_benchmark_subprocess
            or not settings.autotune_accuracy_check
            or settings.autotune_benchmark_fn is not None
            or settings.autotune_baseline_accuracy_check_fn is not None
        ):
            timing_error("Unsupported or changed explicit search timing settings")

    def cache_record(self) -> dict[str, object]:
        return {
            "method": "torch_profiler",
            "statistic": "mean_kernel_work_ms",
            "isolation": "spawned_worker",
            **dataclasses.asdict(self),
        }


def validate_timing_policy(
    policy: SearchTimingPolicy | None,
    resolved: SearchTimingPolicy | None,
    settings: Settings,
) -> None:
    """A resolved opt-in domain cannot be removed to regain a default path."""
    if policy != resolved or (
        policy is None and settings.autotune_timing_method != "default"
    ):
        timing_error("Resolved search timing policy was replaced or removed")
    if policy is not None:
        policy.validate_settings(settings)


@dataclasses.dataclass(frozen=True)
class BenchmarkMeasurement:
    policy: SearchTimingPolicy
    observation: ProfilerTimingObservation
    warmup_calls: int
    fixed_repetitions: int | None

    @property
    def perf(self) -> float:
        return self.observation.mean_ms

    def validate(
        self,
        policy: SearchTimingPolicy,
        identity: CallableIdentity | None = None,
        fixed_repetitions: int | None = None,
    ) -> None:
        from .profiler_timing import ProfilerChunk
        from .profiler_timing import ProfilerTimingObservation
        from .profiler_timing import ProfilerTimingPolicy

        if (
            self.policy != policy
            or not isinstance(self.observation, ProfilerTimingObservation)
            or self.observation.policy != ProfilerTimingPolicy(policy.chunk_size)
            or (identity is not None and self.observation.identity != identity)
            or not self.observation.chunks
            or not isinstance(self.observation.chunks, tuple)
            or any(
                not isinstance(chunk, ProfilerChunk)
                or type(chunk.call_count) is not int
                or chunk.call_count < 1
                or type(chunk.total_ns) is not int
                or chunk.total_ns < 1
                or type(chunk.kernel_count) is not int
                or chunk.kernel_count < chunk.call_count
                or type(chunk.launch_count) is not int
                or chunk.launch_count < chunk.call_count
                or (chunk.device_index, chunk.stream_id, chunk.work_signature)
                != (
                    self.observation.chunks[0].device_index,
                    self.observation.chunks[0].stream_id,
                    self.observation.chunks[0].work_signature,
                )
                for chunk in self.observation.chunks
            )
            or (
                self.fixed_repetitions is not None
                and (
                    type(self.fixed_repetitions) is not int
                    or self.fixed_repetitions < 1
                    or self.fixed_repetitions != self.observation.call_count
                )
            )
            or (
                fixed_repetitions is not None
                and self.observation.call_count != fixed_repetitions
            )
            or not math.isfinite(self.perf)
            or self.perf <= 0
            or type(self.warmup_calls) is not int
            or self.warmup_calls < 0
        ):
            timing_error("Invalid or foreign profiler measurement")

    def record(self) -> dict[str, object]:
        return {
            "domain": self.policy.cache_record(),
            "observation": dataclasses.asdict(self.observation),
            "warmup_calls": self.warmup_calls,
            "fixed_repetitions": self.fixed_repetitions,
            "perf_ms": self.perf,
        }

    def validate_callable(
        self, policy: SearchTimingPolicy, fn: Callable[..., object]
    ) -> None:
        from .precompile_future import _serialize_compiled_fn
        from .profiler_timing import CallableIdentity

        try:
            serialized = _serialize_compiled_fn(fn)
        except RuntimeError as error:
            timing_error(
                f"External measured callable has no serialized identity: {error}"
            )
        self.validate(
            policy,
            CallableIdentity(
                serialized.function_name,
                hashlib.sha256(serialized.source_code.encode()).hexdigest(),
            ),
        )


def collect_search_measurement(
    fn: Callable[[], object],
    *,
    policy: SearchTimingPolicy,
    identity: CallableIdentity,
    sample_count: int,
    clear: Callable[[], object],
    synchronize: Callable[[], object],
    warmup_calls: int,
    fixed_repetitions: int | None,
) -> BenchmarkMeasurement:
    from .profiler_timing import ProfilerTimingCapabilityError
    from .profiler_timing import ProfilerTimingPolicy
    from .profiler_timing import collect_profiler_timing

    # A callable failure remains a candidate failure. Errors in the profiler,
    # clear or attribution protocol never become candidate pruning decisions.
    callable_error: list[BaseException] = []
    last_trace: list[ProfilerTrace] = []

    def measured() -> object:
        try:
            return fn()
        except Exception as error:
            callable_error.append(error)
            raise

    def retain_trace(trace: ProfilerTrace) -> None:
        last_trace[:] = [trace]

    try:
        observation = collect_profiler_timing(
            measured,
            identity=identity,
            sample_count=sample_count,
            clear=clear,
            synchronize=synchronize,
            policy=ProfilerTimingPolicy(policy.chunk_size),
            trace_sink=retain_trace,
        )
    except Exception as error:
        if callable_error and error is callable_error[-1]:
            raise
        if not isinstance(error, ProfilerTimingCapabilityError):
            error = ProfilerTimingCapabilityError(str(error))
        # Exception args survive the existing spawned-worker transport. Keep the
        # final bounded raw chunk, including attribution failures, in the error.
        error.args = (
            *error.args,
            json.dumps(
                {
                    "identity": dataclasses.asdict(identity),
                    "domain": policy.cache_record(),
                    "trace": dataclasses.asdict(last_trace[-1]) if last_trace else None,
                },
                sort_keys=True,
            ),
        )
        raise error
    result = BenchmarkMeasurement(policy, observation, warmup_calls, fixed_repetitions)
    result.validate(policy, identity, fixed_repetitions)
    return result


@dataclasses.dataclass(frozen=True)
class ProfilerSweepTrace:
    orders: list[list[int]]
    observations: list[list[BenchmarkMeasurement]]
    means_ms: list[float]
    target_ms: float
    repeat_reference_perf_ms: float
    sweep_count: int
    calls_per_sample: int
    total_calls: int
