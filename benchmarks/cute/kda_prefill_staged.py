from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from itertools import pairwise
import operator
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeVar
from typing import cast

from benchmarks.cute.kda_prefill_kernels import BT
from benchmarks.cute.kda_prefill_kernels import DK
from benchmarks.cute.kda_prefill_kernels import kda_chunk_prepare
from benchmarks.cute.kda_prefill_kernels import kda_chunk_recurrence
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


_REGION_ALIGNMENT = 256
_R = TypeVar("_R")
KdaStagedTopology = Literal["origin_aux", "serial"]


@dataclass(frozen=True)
class StagedLongestPartition:
    """Stable longest-segment partition of cumulative sequence offsets."""

    offsets: tuple[int, ...]
    longest_sequence: int
    rest_ranges: tuple[tuple[int, int], ...]

    @property
    def longest_range(self) -> tuple[int, int]:
        return self.longest_sequence, self.longest_sequence + 1


def staged_longest_partition(offsets: Sequence[int]) -> StagedLongestPartition:
    """Split cumulative offsets into the longest segment and contiguous rest.

    Ties are resolved by sequence order. The returned rest ranges preserve
    contiguity, so callers can build zero-copy views or local metadata for the
    ranges before CUDA graph capture.
    """

    try:
        normalized = tuple(operator.index(value) for value in offsets)
    except TypeError as error:
        raise TypeError("offsets must contain integers") from error
    if len(normalized) < 2:
        raise ValueError("offsets must describe at least one segment")
    if any(end < begin for begin, end in pairwise(normalized)):
        raise ValueError("offsets must be nondecreasing")
    if any(end == begin for begin, end in pairwise(normalized)):
        raise ValueError("offsets must describe nonempty segments")

    longest_sequence = max(
        range(len(normalized) - 1),
        key=lambda index: normalized[index + 1] - normalized[index],
    )
    rest_ranges = tuple(
        sequence_range
        for sequence_range in (
            (0, longest_sequence),
            (longest_sequence + 1, len(normalized) - 1),
        )
        if sequence_range[0] < sequence_range[1]
    )
    return StagedLongestPartition(normalized, longest_sequence, rest_ranges)


@dataclass
class CuteOriginAuxDag:
    """Resources for a reusable prefix -> origin/auxiliary CUDA launch DAG.

    Create one instance per independently captured graph slot and keep it alive
    for the graph's lifetime. ``launch`` enqueues the prefix on the current
    stream, gives the origin branch launch priority, runs the auxiliary branch
    on the owned stream, and joins that branch back to the origin stream.
    """

    auxiliary_stream: torch.cuda.Stream
    fork_event: torch.cuda.Event
    done_event: torch.cuda.Event

    @classmethod
    def create(cls, device: torch.device | int | None = None) -> CuteOriginAuxDag:
        return cls(
            auxiliary_stream=torch.cuda.Stream(device=device),
            fork_event=torch.cuda.Event(enable_timing=False),
            done_event=torch.cuda.Event(enable_timing=False),
        )

    def launch(
        self,
        prefix: Callable[[], object],
        origin_branch: Callable[[], _R],
        auxiliary_branch: Callable[[], object],
    ) -> _R:
        with torch.cuda.device(self.auxiliary_stream.device):
            origin_stream = torch.cuda.current_stream()
            prefix()
            self.fork_event.record(origin_stream)
            self.auxiliary_stream.wait_event(self.fork_event)

            result = origin_branch()
            with torch.cuda.stream(self.auxiliary_stream):
                auxiliary_branch()
                self.done_event.record()
            origin_stream.wait_event(self.done_event)
            return result


@dataclass(frozen=True)
class KdaGroupHostMetadata:
    """Host metadata for one contiguous range of packed sequences."""

    sequence_range: tuple[int, int]
    cu_seqlens: tuple[int, ...]
    cu_chunks: tuple[int, ...]
    chunk_to_seq: tuple[int, ...]

    @property
    def total_chunks(self) -> int:
        return self.cu_chunks[-1]


@dataclass(frozen=True)
class KdaGroupMetadata:
    """Device metadata and its immutable graph-cache identity."""

    host: KdaGroupHostMetadata
    cu_seqlens: torch.Tensor
    cu_chunks: torch.Tensor
    chunk_to_seq: torch.Tensor


@dataclass(frozen=True)
class KdaFactorWorkspace:
    """Five factor views over one aligned packed allocation."""

    storage: torch.Tensor
    kd: torch.Tensor
    qd: torch.Tensor
    ak: torch.Tensor
    aq: torch.Tensor
    g_total: torch.Tensor


@dataclass(frozen=True)
class KdaStagedGroup:
    metadata: KdaGroupMetadata
    workspace: KdaFactorWorkspace


@dataclass
class KdaStagedResources:
    """Per-graph-slot resources for staged packed KDA prefill."""

    critical: KdaStagedGroup
    auxiliary: tuple[KdaStagedGroup, ...]
    dag: CuteOriginAuxDag


def kda_group_host_metadata(
    offsets: Sequence[int], sequence_range: tuple[int, int]
) -> KdaGroupHostMetadata:
    """Build local chunk ordinals while retaining global token offsets."""

    partition = staged_longest_partition(offsets)
    sequence_begin, sequence_end = sequence_range
    sequence_count = len(partition.offsets) - 1
    if not 0 <= sequence_begin < sequence_end <= sequence_count:
        raise ValueError(
            f"invalid sequence range {sequence_range!r} for {sequence_count} sequences"
        )

    cu_seqlens = partition.offsets[sequence_begin : sequence_end + 1]
    chunk_counts = tuple(
        (end - begin + BT - 1) // BT for begin, end in pairwise(cu_seqlens)
    )
    cu_chunks = (0, *accumulate(chunk_counts))
    chunk_to_seq = tuple(
        local_sequence
        for local_sequence, chunk_count in enumerate(chunk_counts)
        for _ in range(chunk_count)
    )
    return KdaGroupHostMetadata(
        sequence_range=sequence_range,
        cu_seqlens=cu_seqlens,
        cu_chunks=cu_chunks,
        chunk_to_seq=chunk_to_seq,
    )


def materialize_kda_group_metadata(
    host: KdaGroupHostMetadata, device: torch.device | str
) -> KdaGroupMetadata:
    """Materialize immutable group metadata before graph capture."""

    return KdaGroupMetadata(
        host=host,
        cu_seqlens=torch.tensor(host.cu_seqlens, dtype=torch.int32, device=device),
        cu_chunks=torch.tensor(host.cu_chunks, dtype=torch.int32, device=device),
        chunk_to_seq=torch.tensor(host.chunk_to_seq, dtype=torch.int32, device=device),
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def allocate_kda_factor_workspace(
    heads: int, total_chunks: int, device: torch.device | str
) -> KdaFactorWorkspace:
    """Allocate the exact packed v2 workspace layout used by both kernels."""

    rows = total_chunks * BT
    vector_bytes = heads * rows * DK * 2
    aq_bytes = heads * total_chunks * BT * BT * 2
    g_total_bytes = heads * total_chunks * DK * 4
    sizes = (vector_bytes, vector_bytes, vector_bytes, aq_bytes, g_total_bytes)
    offsets: list[int] = []
    cursor = 0
    for size in sizes:
        cursor = _align_up(cursor, _REGION_ALIGNMENT)
        offsets.append(cursor)
        cursor += size
    size = _align_up(cursor, _REGION_ALIGNMENT)
    raw = torch.empty(size + _REGION_ALIGNMENT, dtype=torch.uint8, device=device)
    pad = (-raw.data_ptr()) % _REGION_ALIGNMENT
    storage = raw[pad : pad + size]

    def view(index: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        offset = offsets[index]
        return storage[offset : offset + sizes[index]].view(dtype).view(shape)

    return KdaFactorWorkspace(
        storage=storage,
        kd=view(0, torch.bfloat16, (1, heads, rows, DK)),
        qd=view(1, torch.bfloat16, (1, heads, rows, DK)),
        ak=view(2, torch.bfloat16, (1, heads, rows, DK)),
        aq=view(3, torch.bfloat16, (1, heads, total_chunks, BT * BT)),
        g_total=view(4, torch.float32, (1, heads, total_chunks, DK)),
    )


def create_staged_kda_resources(
    offsets: Sequence[int],
    *,
    heads: int,
    device: torch.device | str,
) -> KdaStagedResources:
    """Create metadata, private workspaces, and DAG state for one graph slot."""

    partition = staged_longest_partition(offsets)

    def create_group(sequence_range: tuple[int, int]) -> KdaStagedGroup:
        metadata = materialize_kda_group_metadata(
            kda_group_host_metadata(partition.offsets, sequence_range), device
        )
        workspace = allocate_kda_factor_workspace(
            heads, metadata.host.total_chunks, device
        )
        return KdaStagedGroup(metadata, workspace)

    return KdaStagedResources(
        critical=create_group(partition.longest_range),
        auxiliary=tuple(create_group(item) for item in partition.rest_ranges),
        dag=CuteOriginAuxDag.create(torch.device(device)),
    )


def launch_staged_kda_prefill(
    resources: KdaStagedResources,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    values: torch.Tensor,
    output: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    gate_scale_log2: float,
    topology: KdaStagedTopology = "origin_aux",
    prepare_kernels: Sequence[Callable[..., object]] | None = None,
    recurrence_kernels: Sequence[Callable[..., object]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the staged longest-sequence KDA pipeline into caller storage.

    ``state`` must be in packed sequence order. Inputs and output keep their
    global token indexing; only factor workspaces and chunk ordinals are local
    to each contiguous sequence group.
    """

    groups = (resources.critical, *resources.auxiliary)
    if prepare_kernels is None:
        prepare_kernels = (kda_chunk_prepare,) * len(groups)
    if recurrence_kernels is None:
        recurrence_kernels = (kda_chunk_recurrence,) * len(groups)
    if len(prepare_kernels) != len(groups) or len(recurrence_kernels) != len(groups):
        raise ValueError("one prepare and recurrence kernel are required per group")

    def prepare(group: KdaStagedGroup, kernel: Callable[..., object]) -> None:
        metadata = group.metadata
        workspace = group.workspace
        kernel(
            q,
            k,
            gate,
            beta_logits,
            a_log,
            dt_bias,
            metadata.cu_seqlens,
            metadata.cu_chunks,
            metadata.chunk_to_seq,
            workspace.kd,
            workspace.qd,
            workspace.ak,
            workspace.aq,
            workspace.g_total,
            None,
            gate_scale_log2,
        )

    def recurrence(
        group: KdaStagedGroup, kernel: Callable[..., object]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = group.metadata
        workspace = group.workspace
        sequence_begin, sequence_end = metadata.host.sequence_range
        result = kernel(
            workspace.kd,
            workspace.qd,
            workspace.ak,
            workspace.aq,
            workspace.g_total,
            values,
            output,
            state[sequence_begin:sequence_end],
            metadata.cu_seqlens,
            metadata.cu_chunks,
            scale,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("recurrence kernel must return (output, state)")
        return cast("tuple[torch.Tensor, torch.Tensor]", result)

    if topology not in ("origin_aux", "serial"):
        raise ValueError(f"unsupported staged KDA topology: {topology!r}")
    if topology == "origin_aux" and len(resources.auxiliary) > 1:
        raise ValueError(
            "origin_aux topology supports at most one contiguous auxiliary range; "
            "use serial when the longest sequence is interior"
        )

    if topology == "serial" or not resources.auxiliary:
        for group, prepare_kernel, recurrence_kernel in zip(
            groups, prepare_kernels, recurrence_kernels, strict=True
        ):
            prepare(group, prepare_kernel)
            recurrence(group, recurrence_kernel)
        return output, state

    critical = resources.critical

    def auxiliary_branch() -> None:
        for group, prepare_kernel, recurrence_kernel in zip(
            resources.auxiliary,
            prepare_kernels[1:],
            recurrence_kernels[1:],
            strict=True,
        ):
            prepare(group, prepare_kernel)
            recurrence(group, recurrence_kernel)

    resources.dag.launch(
        lambda: prepare(critical, prepare_kernels[0]),
        lambda: recurrence(critical, recurrence_kernels[0]),
        auxiliary_branch,
    )
    return output, state


__all__ = [
    "CuteOriginAuxDag",
    "KdaFactorWorkspace",
    "KdaGroupHostMetadata",
    "KdaGroupMetadata",
    "KdaStagedGroup",
    "KdaStagedResources",
    "KdaStagedTopology",
    "StagedLongestPartition",
    "allocate_kda_factor_workspace",
    "create_staged_kda_resources",
    "kda_group_host_metadata",
    "launch_staged_kda_prefill",
    "materialize_kda_group_metadata",
    "staged_longest_partition",
]
