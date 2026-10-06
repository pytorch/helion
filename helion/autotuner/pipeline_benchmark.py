"""Measure complete, explicitly configured pipelines under GPU graph replay."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field
import functools
import math
import statistics
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map_only

from ..runtime.pipeline import PipelineConfig
from ..runtime.pipeline import PipelineInitialConfig
from ..runtime.pipeline import PipelineKeyFunction
from ..runtime.pipeline import PipelineScope
from ..runtime.pipeline import PipelineStage
from ..runtime.pipeline import argument_metadata
from .benchmark_provider import _aggregate_values
from .logger import match_unrecoverable_runtime_error

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


@dataclass
class PipelineMeasurement:
    """A whole-operation measurement; tensors are omitted from serialized records."""

    bundle: PipelineConfig
    timings_ms: tuple[float, ...] = ()
    aggregate_ms: float = math.inf
    traces: tuple[tuple[str, ...], ...] = ()
    stages: dict[str, PipelineStage] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle.to_dict(),
            "timings_ms": list(self.timings_ms),
            "aggregate_ms": self.aggregate_ms
            if math.isfinite(self.aggregate_ms)
            else None,
            "traces": [list(trace) for trace in self.traces],
            "stages": {
                key: {
                    "source_hash": stage.source_hash,
                    "argument_metadata": stage.argument_metadata,
                    "config": dict(stage.config),
                    "config_source": stage.config_source,
                    "config_space_identity": list(stage.config_space_identity),
                }
                for key, stage in self.stages.items()
            },
            "status": self.status,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "measurement": "complete GPU graph replay",
        }


def benchmark_graph_replay(
    replay: Callable[[], Any], *, warmup: int = 5, repeat: int = 40
) -> float:
    """Return median milliseconds for already captured CUDA or HIP graph replay.

    This function never captures another graph and never falls back to an eager
    operation. Callers can supply a profiler-based timer to PipelineEvaluator
    when matching another benchmark's measurement protocol.
    """
    if warmup < 0 or repeat < 1:
        raise ValueError("Graph timing requires warmup >= 0 and repeat >= 1")
    for _ in range(warmup):
        replay()
    torch.cuda.synchronize()
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeat)
    ]
    for start, end in pairs:
        start.record()
        replay()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in pairs)


def _require_gpu_inputs(arg_sets: Sequence[Sequence[object]]) -> None:
    tensors = [
        value
        for values in arg_sets
        for value in tree_flatten(values)[0]
        if isinstance(value, torch.Tensor)
    ]
    if not tensors or any(value.device.type != "cuda" for value in tensors):
        raise ValueError("Pipeline graph tuning requires CUDA or ROCm tensor inputs")
    device = tensors[0].device
    if any(value.device != device for value in tensors):
        raise ValueError("All pipeline inputs must use the same GPU")
    if device.index != torch.cuda.current_device():
        raise ValueError("Pipeline inputs must use the current GPU")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("Pipeline tuning cannot run inside graph capture")


def _tensor_bytes(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


def _clone_inputs(args: Sequence[object]) -> tuple[object, ...]:
    """Retain offsets, strides and repeated/cross-view aliases in private inputs.

    An ordinary contiguous clone resets a view's storage offset. Pipeline keys
    include that metadata, and alignment can affect performance, so copy each
    detached storage once, including a lone contiguous view.
    """
    detached: dict[int, torch.Tensor] = {}

    def detach(value: torch.Tensor) -> torch.Tensor:
        if id(value) not in detached:
            detached[id(value)] = value.detach()
        return detached[id(value)]

    return copy.deepcopy(tree_map_only(torch.Tensor, detach, tuple(args)))


@dataclass
class _InputSnapshot:
    value: torch.Tensor
    contents: torch.Tensor
    metadata: object
    storage_address: int


def _snapshots(args: Sequence[object]) -> list[_InputSnapshot]:
    return [
        _InputSnapshot(
            value,
            _tensor_bytes(value).clone(),
            argument_metadata(value),
            value.untyped_storage().data_ptr(),
        )
        for value in tree_flatten(args)[0]
        if isinstance(value, torch.Tensor)
    ]


def _check_inputs(snapshots: Sequence[_InputSnapshot]) -> None:
    for snapshot in snapshots:
        value = snapshot.value
        if (
            argument_metadata(value) != snapshot.metadata
            or value.untyped_storage().data_ptr() != snapshot.storage_address
            or not torch.equal(_tensor_bytes(value), snapshot.contents)
        ):
            raise ValueError(
                "Pipeline modified an external input; only immutable, replay-safe "
                "inputs are supported. General input-reset protocols are unsupported."
            )


def _poison_output(
    output: object,
    inputs: Sequence[_InputSnapshot],
    iteration: int,
) -> None:
    input_storage = {snapshot.storage_address for snapshot in inputs}
    for value in tree_flatten(output)[0]:
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            continue
        # Returning an unchanged input view is valid. Poisoning that view would
        # itself violate the immutable-input contract.
        if value.untyped_storage().data_ptr() in input_storage:
            continue
        if value.is_floating_point() or value.is_complex():
            value.fill_(float("nan"))
        elif value.dtype == torch.bool:
            value.fill_(bool(iteration % 2))
        else:
            value.fill_(torch.iinfo(value.dtype).min if iteration % 2 else 0)


def _capture_graph(call: Callable[[], object]) -> tuple[Any, object]:
    """Capture only the fully prepared operation; capture errors stay errors."""
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    return graph, output


class PipelineEvaluator:
    """Compile, validate and measure entire candidate pipelines on every input.

    The callable may allocate and mutate internal scratch. External tensor
    arguments are cloned with their alias relationships intact and must remain
    unchanged, including during warmup and graph replay. Compilation, input
    cloning, reference computation, graph capture and output poisoning are all
    outside the benchmark callback, which receives only ``graph.replay``.
    """

    def __init__(
        self,
        operation: Callable[..., object],
        arg_sets: Sequence[Sequence[object]],
        *,
        reference: Callable[..., object],
        check: Callable[[object, object], None],
        aggregation: Literal["geomean", "max"] = "geomean",
        key_fn: PipelineKeyFunction | None = None,
        initial_config: PipelineInitialConfig | None = None,
        benchmark: Callable[[Callable[[], Any]], float] | None = None,
        warmup: int = 3,
        reset: Callable[..., object] | None = None,
    ) -> None:
        if not arg_sets:
            raise ValueError("Pipeline tuning requires at least one argument set")
        if torch.distributed.is_initialized():
            raise ValueError("Distributed pipeline tuning is not supported")
        if aggregation not in ("geomean", "max"):
            raise ValueError("Pipeline aggregation must be 'geomean' or 'max'")
        if warmup < 1:
            raise ValueError("Pipeline preparation requires at least one warmup")
        if reset is not None:
            raise ValueError(
                "Input-reset callbacks are not supported by pipeline tuning; "
                "reset work must not be silently included in replay timing"
            )
        self.operation = operation
        self.arg_sets = tuple(tuple(values) for values in arg_sets)
        self.reference = reference
        self.check = check
        self.aggregation = aggregation
        self.key_fn = key_fn
        self.initial_config = initial_config
        self.benchmark = benchmark or benchmark_graph_replay
        self.warmup = warmup

    @torch.no_grad()
    def evaluate(self, bundle: PipelineConfig) -> PipelineMeasurement:
        """Realize new topology, freeze it, then validate and time every graph."""
        started = time.perf_counter()
        scope = PipelineScope(
            bundle.copy(), key_fn=self.key_fn, initial_config=self.initial_config
        )
        timings: list[float] = []
        traces: list[tuple[str, ...]] = []
        status, error_text = "ok", None
        try:
            _require_gpu_inputs(self.arg_sets)
            prepared = []
            # Reference inputs are independent of candidate mutation. Retain a
            # separate working set for each representative throughout capture.
            for source in self.arg_sets:
                args = _clone_inputs(source)
                expected = self.reference(*_clone_inputs(source))
                prepared.append((args, expected, _snapshots(args)))
            with scope.activate():
                # Discover the union of stage keys and bindings before freezing,
                # including bindings that occur only in later representative inputs.
                for args, expected, snapshots in prepared:
                    trace = None
                    for _ in range(self.warmup):
                        begin = len(scope.trace)
                        actual = self.operation(*args)
                        current_trace = tuple(scope.trace[begin:])
                        if trace is not None and current_trace != trace:
                            raise ValueError("Pipeline topology changed during warmup")
                        trace = current_trace
                        self.check(actual, expected)
                        _check_inputs(snapshots)
                    assert trace is not None
                    traces.append(trace)
                if not scope.stages:
                    raise ValueError(
                        "Pipeline tuning requires at least one Helion stage"
                    )
                scope.freeze()
                for index, (args, expected, snapshots) in enumerate(prepared):
                    begin = len(scope.trace)
                    graph, captured = _capture_graph(
                        functools.partial(self.operation, *args)
                    )
                    if tuple(scope.trace[begin:]) != traces[index]:
                        raise ValueError(
                            "Captured pipeline topology differs from warmup"
                        )
                    # Captured kernels need not execute during CUDA/HIP capture.
                    # Replay before reading outputs, then test two poisoned replays.
                    graph.replay()
                    torch.cuda.synchronize()
                    self.check(captured, expected)
                    _check_inputs(snapshots)
                    for iteration in range(2):
                        _poison_output(captured, snapshots, iteration)
                        graph.replay()
                        torch.cuda.synchronize()
                        self.check(captured, expected)
                        _check_inputs(snapshots)
                    latency = float(self.benchmark(graph.replay))
                    if not math.isfinite(latency) or latency <= 0:
                        raise ValueError(
                            "Pipeline benchmark must return positive finite milliseconds"
                        )
                    self.check(captured, expected)
                    _check_inputs(snapshots)
                    timings.append(latency)
        except Exception as error:
            if match_unrecoverable_runtime_error(error):
                raise
            status = "error"
            error_text = f"{type(error).__name__}: {error}"
        aggregate = (
            _aggregate_values(timings, self.aggregation)
            if status == "ok" and len(timings) == len(self.arg_sets)
            else math.inf
        )
        return PipelineMeasurement(
            bundle=scope.bundle.copy(),
            timings_ms=tuple(timings),
            aggregate_ms=aggregate,
            traces=tuple(traces),
            stages=dict(scope.stages),
            status=status,
            error=error_text,
            elapsed_seconds=time.perf_counter() - started,
        )
