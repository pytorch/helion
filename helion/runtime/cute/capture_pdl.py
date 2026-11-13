"""Proof state for capture-time programmatic dependency overlap.

This module is deliberately independent of CUDA and tensors.  The adapter must
start every captured launch with its ordinary dependency wait, inspect the
*actual* kernel node and incoming edges, and only then replace that node's
runtime wait-mode argument.  Selecting a mode before enqueue is insufficient:
another submitting thread could insert a producer after the dependency query.

The captured graph's topology and parameter addresses must remain immutable
after this proof.  Replaying it with changed tensor contents is supported;
external graph surgery or parameter-address replacement is outside the proof.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum


class WaitMode(IntEnum):
    EARLY = 0
    BEFORE_WRITES = 1


@dataclass(frozen=True, order=True)
class StorageRange:
    """A conservative allocation-byte interval on one CUDA device/context."""

    device: int
    begin: int
    end: int

    def __post_init__(self) -> None:
        if self.device < 0 or self.begin < 0 or self.end < self.begin:
            raise ValueError("invalid storage interval")

    def overlaps(self, other: StorageRange) -> bool:
        return (
            self.device == other.device
            and self.begin < other.end
            and other.begin < self.end
            and self.begin != self.end
            and other.begin != other.end
        )


def storage_union(ranges: tuple[StorageRange, ...]) -> tuple[StorageRange, ...]:
    result: list[StorageRange] = []
    for interval in sorted(ranges):
        if interval.begin == interval.end:
            continue
        if (
            result
            and result[-1].device == interval.device
            and result[-1].end >= interval.begin
        ):
            previous = result.pop()
            interval = StorageRange(
                previous.device, previous.begin, max(previous.end, interval.end)
            )
        result.append(interval)
    return tuple(result)


@dataclass(frozen=True)
class LaunchEffects:
    """Complete effects of one audited kernel, including every hidden buffer.

    Reads include scalar preludes and descriptor-backed accesses. Writes include
    all stores and atomics, even conditionally executed or zero-trip epilogues.
    An unknown effect or a wrapper that performs another enqueue must not create
    this record. The kernel contract waits before *every* global write in late
    mode, and its all-CTA trigger follows the dependency wait in early mode.
    """

    reads: tuple[StorageRange, ...]
    writes: tuple[StorageRange, ...]


@dataclass(frozen=True)
class CaptureKey:
    """CUDA context, submitting thread, stream, capture ID and graph handle."""

    context: int
    thread: int
    stream: int
    capture_id: int
    graph: int


@dataclass(frozen=True)
class Dependency:
    node: int
    from_port: int = 0
    to_port: int = 0
    kind: int = 0
    reserved: bytes = b"\0" * 5

    def supported(self) -> bool:
        # CUDA's launch-order port is intentionally absent. It does not provide
        # the all-CTA trigger/termination aggregation required by this proof.
        return self.reserved == b"\0" * 5 and (
            (self.from_port, self.to_port, self.kind) in ((0, 0, 0), (1, 0, 1))
        )


@dataclass(frozen=True)
class Decision:
    mode: WaitMode
    pending_writes: tuple[StorageRange, ...]
    trackable: bool


class CaptureGuard:
    """Bounded, CPU-only ancestry proof; one instance per submitting thread.

    ``decide`` consumes verified incoming edges of an already captured kernel.
    ``record`` is called only after its mode argument was installed successfully
    (or left at EARLY). Eviction loses opportunities, never dependency safety.
    An early anchor discards older writes only because every CTA signals *after*
    waiting. A late node retains the entire pending ancestor-write union.
    """

    def __init__(
        self,
        *,
        max_captures: int = 8,
        max_nodes: int = 1024,
        max_ranges: int = 128,
    ) -> None:
        if min(max_captures, max_nodes, max_ranges) <= 0:
            raise ValueError("capture guard limits must be positive")
        self.max_captures = max_captures
        self.max_nodes = max_nodes
        self.max_ranges = max_ranges
        self.captures: OrderedDict[
            CaptureKey, OrderedDict[int, tuple[StorageRange, ...]]
        ] = OrderedDict()

    def decide(
        self,
        key: CaptureKey | None,
        node: int,
        dependencies: tuple[Dependency, ...],
        effects: LaunchEffects | None,
    ) -> Decision:
        if key is None or node <= 0 or effects is None:
            return Decision(WaitMode.EARLY, (), False)
        writes = storage_union(effects.writes)
        if len(writes) > self.max_ranges or any(
            not dependency.supported() for dependency in dependencies
        ):
            return Decision(WaitMode.EARLY, (), False)
        # A zero/multiple/unknown predecessor set cannot prove safe early reads.
        # This audited early kernel can nevertheless anchor the next launch.
        early = Decision(WaitMode.EARLY, writes, True)
        nodes = self.captures.get(key)
        if nodes is not None and node in nodes:
            return Decision(WaitMode.EARLY, (), False)
        if nodes is None or len(dependencies) != 1:
            return early
        pending = nodes.get(dependencies[0].node)
        if pending is None or any(
            write.overlaps(read) for write in pending for read in effects.reads
        ):
            return early
        combined = storage_union((*pending, *writes))
        if len(combined) > self.max_ranges:
            return early
        return Decision(WaitMode.BEFORE_WRITES, combined, True)

    def record(self, key: CaptureKey, node: int, decision: Decision) -> None:
        if not decision.trackable:
            return
        nodes = self.captures.setdefault(key, OrderedDict())
        nodes[node] = decision.pending_writes
        nodes.move_to_end(node)
        self.captures.move_to_end(key)
        while len(nodes) > self.max_nodes:
            nodes.popitem(last=False)
        while len(self.captures) > self.max_captures:
            self.captures.popitem(last=False)

    def release(self, key: CaptureKey) -> None:
        self.captures.pop(key, None)
