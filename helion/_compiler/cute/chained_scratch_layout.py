"""Shared logical-coordinate layouts for FP32 contraction-region scratch.

An XOR layout permutes the low column/bank bits by low row bits. A power-of-two
row width keeps the row bits above the column bits, so the two swizzle fields
cannot overlap. Every row stays inside its original allocation; no padding,
arena offset or lifetime changes are needed. Single-row buffers are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from .chained_matmul import ChainedMatmulPlan
    from .chained_vector_leaf import DenseVectorSink


def xor_swizzle(shape: tuple[int, ...]) -> tuple[int, int, int] | None:
    """Return CuTe's (bits, base, shift) for a nonidentity bounded permutation."""
    if len(shape) != 2:
        return None
    rows, columns = shape
    if rows < 2 or columns < 2 or columns & (columns - 1):
        return None
    shift = columns.bit_length() - 1
    # FP32 has one element per 4-byte bank: at most five low bits select a bank.
    return min(5, shift), 0, shift


@dataclass
class ScratchLayouts:
    """Allocation tracker local to one codegen attempt, never stored on a plan."""

    mode: str = "row_major"
    xor_buffers: set[str] = field(default_factory=set)
    read_buffers: frozenset[str] | None = None
    vector_sinks: dict[str, DenseVectorSink] = field(default_factory=dict)

    @classmethod
    def for_plan(cls, mode: str, plan: ChainedMatmulPlan) -> ScratchLayouts:
        """Only consumed allocations can make this attempt's choice effective."""
        from .chained_collectives import collective_bindings

        assert plan.region is not None
        boundaries = {
            **{node: f"chain_{stage}_c" for stage, node in enumerate(plan.dots)},
            **collective_bindings(plan),
        }
        if plan.loop is not None:
            boundaries.update(plan.loop.boundaries())
        pending = [
            operand
            for node in (*plan.dots, *plan.region.scans, *plan.region.reductions)
            for operand in node.all_input_nodes
        ]
        pending.extend(plan.region.stores)
        pending.append(plan.store)
        pending.extend(export.store for export in plan.scan_exports)
        pending.extend(plan.region.live_outs)
        pending.extend(carry.output for carry in plan.region.carries)
        visited, reads = set(), set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in boundaries:
                reads.add(boundaries[node])
            else:
                pending.extend(node.all_input_nodes)
        return cls(mode=mode, read_buffers=frozenset(reads))

    def layout(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        *,
        row_stride: int | None = None,
    ) -> str:
        if len(shape) != 2 or dtype != torch.float32:
            return f"cute.make_layout({shape!r})"
        rows, columns = shape
        stride = columns if row_stride is None else row_stride
        assert stride >= columns
        swizzle = xor_swizzle(shape) if self.mode == "xor" else None
        if swizzle is None:
            return f"cute.make_layout({shape!r}, stride=({stride}, 1))"
        # Legacy warp results may have extra row padding. The XOR variant uses
        # a compact view inside that same (larger) allocation, keeping accounting
        # and any resident arena ownership unchanged.
        if self.read_buffers is None or name in self.read_buffers:
            self.xor_buffers.add(name)
        bits, base, shift = swizzle
        return (
            f"cute.make_composed_layout(cute.make_swizzle({bits}, {base}, {shift}), "
            f"0, cute.make_layout({shape!r}, stride=({columns}, 1)))"
        )

    def validate(self) -> None:
        if self.mode == "xor" and not self.xor_buffers:
            raise exc.BackendUnsupported(
                "cute", "XOR scratch layout requires a materialized eligible FP32 tile"
            )


def has_scratch_candidate(
    graphs: Sequence[GraphInfo], *, include_dots: bool = True
) -> bool:
    """Check graph/shape facts before advertising the layout search dimension.

    Concrete codegen separately validates activation: register/TMEM bridges may
    remove all eligible allocations for a particular schedule.
    """
    from ...language import scan_ops
    from ...language.matmul_ops import dot
    from ..compile_environment import CompileEnvironment
    from ..compile_environment import FixedBlockSizeSource
    from ..device_ir import ForLoopGraphInfo

    env = CompileEnvironment.current()
    valid = set(env.config_spec.block_sizes.valid_block_ids())

    def possible(size: int | torch.SymInt, *, power_of_two: bool) -> bool:
        if isinstance(size, int):
            return size >= 2 and (not power_of_two or size & (size - 1) == 0)
        block_id = env.get_block_id(size)
        if block_id is None:
            return False
        block_id = env.canonical_block_id(block_id)
        source = env.block_sizes[block_id].block_size_source
        if isinstance(source, FixedBlockSizeSource) and type(source.value) is int:
            return possible(source.value, power_of_two=power_of_two)
        if block_id not in valid:
            return False
        block = env.config_spec.block_sizes.block_id_lookup(block_id)
        minimum = max(2, block.min_size)
        candidate = 1 << (minimum - 1).bit_length() if power_of_two else minimum
        return candidate <= block.max_size

    for graph in graphs:
        carries = set()
        if isinstance(graph, ForLoopGraphInfo) and graph.loop_interface is not None:
            placeholders = tuple(graph.graph.find_nodes(op="placeholder"))
            carries = {
                placeholders[carry.input_index]
                for carry in graph.loop_interface.carries
            }
        for node in graph.graph.nodes:
            if node.target == dot and not include_dots:
                continue
            if node not in carries and node.target not in (
                dot,
                scan_ops._associative_scan,
                torch.ops.aten.sum.dim_IntList,
            ):
                continue
            value = node.meta.get("val")
            if (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.float32
                and value.ndim == 2
                and possible(value.shape[0], power_of_two=False)
                and possible(value.shape[1], power_of_two=True)
            ):
                return True
    return False
