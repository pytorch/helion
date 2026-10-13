"""Optional native boundary transport for already-admitted vector producers.

Graph identity, publication, alias layout and lifetime belong to the supplied
input proofs. This helper only substitutes the storage read in the original
typed boundary expression. It neither admits a producer nor changes its masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_native_reads import plan_native_vector_read

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_native_read_inputs import NativeReadInput
    from .chained_vector_ownership import VectorOwnership


@dataclass
class NativeReadInputs:
    """Per-attempt activation; a failed producer must not set this flag."""

    inputs: tuple[NativeReadInput, ...]
    activated: bool = False
    emitted: tuple[NativeReadInput, ...] = ()


@dataclass(frozen=True)
class NativeInputEmission:
    setup: tuple[str, ...]
    loads: tuple[str, ...]
    replacements: dict[tuple[Node, tuple[str, ...]], str]
    sources: tuple[NativeReadInput, ...] = ()


def emit_native_inputs(
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    probes: Sequence[chain._Expression],
    inputs: NativeReadInputs,
    ownership: VectorOwnership,
    tag: str,
    row: str,
    base: str,
    element: str,
    execution: ChainedExecution,
) -> NativeInputEmission | None:
    """Read exact canonical boundary cells, after original producer admission.

    Setup goes before the existing output setup; copies go inside the original
    row guard before its element loop. No barrier, allocation in shared memory,
    publication, or activation is performed here. Unsupported requests leave
    the caller's ordinary scalar boundary reads intact.
    """
    if not ownership.matches(ownership.shape, execution.threads):
        return None
    coordinates = (row, f"({base} + {element})")
    setup: list[str] = []
    loads: list[str] = []
    replacements: dict[tuple[Node, tuple[str, ...]], str] = {}
    selected_indices: set[int] = set()
    selected_nodes: set[Node] = set()
    sources: list[NativeReadInput] = []
    for source in inputs.inputs:
        reads = [
            (probe, key)
            for probe in probes
            for key in probe.memo
            if key[0] is source.node
            and key[0] in probe.boundaries
            and key[0] not in probe.fragments
        ]
        if not reads:
            continue
        if (
            source.index in selected_indices
            or source.node in selected_nodes
            or source.shape != ownership.shape
            or not source.matches(plan, boundaries)
            or any(
                key[1] != coordinates or probe.boundaries[source.node] != source.tensor
                for probe, key in reads
            )
        ):
            return None
        geometry = plan_native_vector_read(
            source.full_shape,
            source.shape,
            source.row_offset,
            source.dtype,
            ownership,
        )
        if geometry is None:
            return None
        dtype = CompileEnvironment.current().backend.dtype_str(source.dtype)
        shape = chain._shape(source.node)
        original = chain._materialized_value(source.tensor, shape, coordinates, dtype)
        if any(probe.memo[key] != original for probe, key in reads):
            return None
        emission = geometry.emit(
            source.tensor,
            f"{tag}_input_{source.index}",
            execution.thread,
            f"{tag}_step",
        )
        setup.extend(emission.setup)
        loads.append(emission.copy)
        replacements[source.node, coordinates] = chain._materialized_value(
            source.tensor,
            shape,
            coordinates,
            dtype,
            storage_value=f"{emission.values}[{element}]",
        )
        selected_indices.add(source.index)
        selected_nodes.add(source.node)
        sources.append(source)
    if not replacements:
        return None
    return NativeInputEmission(tuple(setup), tuple(loads), replacements, tuple(sources))
