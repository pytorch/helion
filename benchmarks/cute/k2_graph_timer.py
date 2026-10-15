"""Cold-L2 CUDA-event timing of complete, already captured K2 graphs.

Each timed batch contains ``batch_size`` copies of [clear L2, child graph].
Subtract a separately timed batch of the same L2 clears and divide by the
batch size.  CUDA events surround whole batches, amortizing their scheduling
cost.  Child graphs retain all captured kernels and copies; providers are
never called again or recaptured by this timer.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import dataclass
import math
import statistics
import time
from typing import TYPE_CHECKING
from typing import Any

from cuda.bindings import runtime
import torch
import triton

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence


def _cuda(result: tuple[Any, ...]) -> Any:  # noqa: ANN401 - heterogeneous CUDA C API tuples
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA graph timer: {error}")
    if len(values) == 1:
        return values[0]
    return tuple(values)


def _destroy_exec(executable: runtime.cudaGraphExec_t) -> None:
    _cuda(runtime.cudaGraphExecDestroy(executable))


def _destroy_graph(graph: runtime.cudaGraph_t) -> None:
    _cuda(runtime.cudaGraphDestroy(graph))


def _make_batch(
    stack: ExitStack,
    clear_graph: torch.cuda.CUDAGraph,
    child_graph: torch.cuda.CUDAGraph | None,
    batch_size: int,
) -> runtime.cudaGraphExec_t:
    graph = _cuda(runtime.cudaGraphCreate(0))
    stack.callback(_destroy_graph, graph)
    dependencies: list[Any] = []
    for _ in range(batch_size):
        node = _cuda(
            runtime.cudaGraphAddChildGraphNode(
                graph,
                dependencies,
                len(dependencies),
                clear_graph.raw_cuda_graph(),
            )
        )
        dependencies = [node]
        if child_graph is not None:
            node = _cuda(
                runtime.cudaGraphAddChildGraphNode(
                    graph, dependencies, 1, child_graph.raw_cuda_graph()
                )
            )
            dependencies = [node]
    executable = _cuda(runtime.cudaGraphInstantiate(graph, 0))
    stack.callback(_destroy_exec, executable)
    return executable


@dataclass(frozen=True)
class PreparedGraphBatches:
    """Launch handles valid only inside ``prepared_graph_batches``.

    Strong references to the child graphs keep CuTe descriptors and captured
    copy sources alive.  The preparation replay records the measurement stream
    with graph-specific resource owners before native graph launches bypass
    their Python replay wrappers.
    """

    graphs: tuple[torch.cuda.CUDAGraph, ...]
    cache: torch.Tensor
    clear_graph: torch.cuda.CUDAGraph
    stream: torch.cuda.Stream
    total_executables: tuple[Any, ...]
    clear_executable: Any
    batch_size: int
    child_node_types: tuple[dict[str, int], ...]

    def _launch(self, executable: runtime.cudaGraphExec_t) -> None:
        with torch.cuda.device(self.stream.device), torch.cuda.stream(self.stream):
            _cuda(runtime.cudaGraphLaunch(executable, self.stream.cuda_stream))

    def replay(self, graph_index: int) -> None:
        self._launch(self.total_executables[graph_index])

    def replay_clear(self) -> None:
        self._launch(self.clear_executable)


@contextmanager
def prepared_graph_batches(
    graphs: Sequence[torch.cuda.CUDAGraph], *, batch_size: int = 256
) -> Generator[PreparedGraphBatches, None, None]:
    """Build native child batches; synchronize and destroy every owned handle.

    Callers retain their ordinary input/output tensors for this context's
    lifetime, as for normal CUDA graph replay.  Child graphs must retain their
    raw templates (``keep_graph=True`` or ``enable_debug_mode()``).
    """
    if not graphs or batch_size <= 0:
        raise ValueError("graphs must be nonempty and batch_size must be positive")
    children = tuple(graphs)
    stream = torch.cuda.current_stream()
    with torch.cuda.device(stream.device), torch.cuda.stream(stream):
        # Also invokes CuTe's record_replay_streams resource ownership hook.
        for child in children:
            child.replay()
        stream.synchronize()
        node_types = []
        for child in children:
            raw = child.raw_cuda_graph()
            nodes, count = _cuda(runtime.cudaGraphGetNodes(raw))
            nodes, count = _cuda(runtime.cudaGraphGetNodes(raw, count))
            node_types.append(
                dict(
                    Counter(
                        _cuda(runtime.cudaGraphNodeGetType(node)).name
                        for node in nodes[:count]
                    )
                )
            )
        # DriverBase omits this backend-specific benchmarking API.
        cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()  # pyrefly: ignore [missing-attribute]
        clear_graph = torch.cuda.CUDAGraph(keep_graph=True)
        capture_stream = torch.cuda.Stream(device=stream.device)
        capture_stream.wait_stream(stream)
        with torch.cuda.graph(clear_graph, stream=capture_stream):
            cache.zero_()
        stream.wait_stream(capture_stream)
        stream.synchronize()
        with ExitStack() as stack:
            clear_executable = _make_batch(stack, clear_graph, None, batch_size)
            total_executables = tuple(
                _make_batch(stack, clear_graph, child, batch_size) for child in children
            )
            # Registered last: finish all native replays before destroying execs,
            # templates, or the Python objects owning their captured addresses.
            stack.callback(stream.synchronize)
            yield PreparedGraphBatches(
                children,
                cache,
                clear_graph,
                stream,
                total_executables,
                clear_executable,
                batch_size,
                tuple(node_types),
            )


class InvalidGraphTiming(RuntimeError):
    """Keep the unmodified raw evidence attached to an invalid timing pass."""

    def __init__(self, reason: str, measurement: dict[str, Any]) -> None:
        super().__init__(reason)
        self.measurement = measurement


def summarize_samples(raw_samples: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Use every finite signed sample, without trimming, clamping, or retries."""
    medians: list[float | None] = []
    deviations: list[float | None] = []
    negative_counts: list[int] = []
    errors = []
    for index, rows in enumerate(raw_samples):
        values = [row["difference_ms"] for row in rows]
        negative_counts.append(sum(value < 0 for value in values))
        if not values or any(
            not math.isfinite(row[field])
            for row in rows
            for field in ("total_ms", "clear_ms", "difference_ms")
        ):
            errors.append(f"graph_index={index}: empty or nonfinite timing samples")
            medians.append(None)
            deviations.append(None)
            continue
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        medians.append(median)
        deviations.append(mad)
        if not math.isfinite(median) or median <= 0:
            errors.append(
                f"graph_index={index}: nonpositive or nonfinite median {median}"
            )
    measurement = {
        "median_ms": medians,
        "mad_ms": deviations,
        "negative_sample_counts": negative_counts,
        "raw_samples": raw_samples,
        "estimator": "untrimmed median of all signed paired differences",
    }
    if errors:
        raise InvalidGraphTiming("; ".join(errors), measurement)
    return measurement


def benchmark_graphs(
    graphs: Sequence[torch.cuda.CUDAGraph],
    *,
    batch_size: int = 256,
    samples: int = 10,
    warmup_ms: int = 100,
) -> dict[str, Any]:
    """Return graph-aligned medians and every paired raw CUDA-event reading.

    Provider order rotates and reverses, and total/clear order alternates.
    ``total_ms`` and ``clear_ms`` are whole-batch durations;
    ``difference_ms`` is their difference divided by ``batch_size``.
    """
    if samples <= 0 or warmup_ms < 0:
        raise ValueError("samples must be positive and warmup_ms nonnegative")
    with prepared_graph_batches(graphs, batch_size=batch_size) as batches:
        warmup_start = time.monotonic()
        while (time.monotonic() - warmup_start) * 1000 < warmup_ms:
            # Warm the clear-only executable too; its first launch would
            # otherwise land inside the first timed sample.
            batches.replay_clear()
            for index in range(len(graphs)):
                batches.replay(index)
            batches.stream.synchronize()
        actual_warmup_ms = (time.monotonic() - warmup_start) * 1000
        raw_samples: list[list[dict[str, Any]]] = [[] for _ in graphs]
        sample_orders = []
        for sample in range(samples):
            indices = list(range(len(graphs)))
            rotation = sample % len(graphs)
            order = indices[rotation:] + indices[:rotation]
            if (sample // len(graphs)) % 2:
                order.reverse()
            sample_orders.append(order)
            for index in order:
                pair_order = ["total", "clear"]
                if (sample + index) % 2:
                    pair_order.reverse()
                row: dict[str, Any] = {"sample": sample, "pair_order": pair_order}
                for label in pair_order:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    # Queue work before the start event so Python launch
                    # submission does not leave the GPU waiting inside it.
                    batches.cache.zero_()
                    start.record(batches.stream)
                    if label == "total":
                        batches.replay(index)
                    else:
                        batches.replay_clear()
                    end.record(batches.stream)
                    end.synchronize()
                    row[f"{label}_ms"] = start.elapsed_time(end)
                difference = (row["total_ms"] - row["clear_ms"]) / batch_size
                row["difference_ms"] = difference
                raw_samples[index].append(row)
        return {
            **summarize_samples(raw_samples),
            "sample_orders": sample_orders,
            "protocol": {
                "timer": "CUDA events around native child-graph batches",
                "cache_policy": "clear before every child; paired clear-only subtraction",
                "batch_size": batch_size,
                "samples": samples,
                "requested_warmup_ms": warmup_ms,
                "actual_warmup_ms": actual_warmup_ms,
                "cache_bytes": batches.cache.numel() * batches.cache.element_size(),
                "child_node_types": batches.child_node_types,
                "stream": batches.stream.cuda_stream,
                "device": str(batches.stream.device),
            },
        }
