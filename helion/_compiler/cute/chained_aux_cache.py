"""Typed raw-vector caches with explicit early/late storage lifetimes."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


def _leaves(roots: tuple[Node, ...], plan: ChainedMatmulPlan) -> list[Node]:
    """Collect pointwise-region leaves, never crossing a contraction or scan."""
    stops = {*plan.dots, *plan.scans}
    pending = list(reversed(roots))
    visited: set[Node] = set()
    result = []
    while pending:
        node = pending.pop()
        if node in visited or node in stops:
            continue
        visited.add(node)
        if (
            node.target is chain.memory_ops.load
            and cast("Node", node.args[0]).target is chain._tracing_ops._host_tensor
            and len(chain._shape(node)) == 1
            and node.meta["val"].dtype in (torch.bfloat16, torch.float16, torch.float32)
            # Indirect indices can depend on an earlier live dot or scan.
            # Raw-cache fills have no such boundary: retain the original load.
            and not (chain._ancestors(node) & stops)
        ):
            result.append(node)
        pending.extend(reversed(node.all_input_nodes))
    return result


def _covered(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    candidate: chain._ScanInput,
    existing: chain._ScanInput,
) -> bool:
    """Deduplicate only complete equal address maps with matching domains."""
    if candidate.source is not existing.source or candidate.extent > existing.extent:
        return False
    coordinate = sympy.Symbol("chain_cache_coordinate", integer=True)
    fixed = {
        sympy.Symbol(f"chain_origin_{axis}", integer=True): 0
        for axis, extent, block in plan.axes
        if extent == block
    }
    if any(
        sympy.simplify(
            sympy.Add(
                left.subs(candidate.coordinate, coordinate).subs(fixed),
                sympy.Mul(-1, right.subs(existing.coordinate, coordinate).subs(fixed)),
            )
        )
        != 0
        for left, right in zip(candidate.indices, existing.indices, strict=True)
    ):
        return False
    coords = (str(coordinate),)
    return chain._operand_domain(
        cg, candidate.operand, coords, plan
    ) == chain._operand_domain(cg, existing.operand, coords, plan)


@dataclasses.dataclass(frozen=True)
class EarlyAuxiliaryInput(chain._ScanInput):
    """A separately allocated, published raw vector, not a borrowed MMA arena."""


def uses_early_cache(expression: chain._Expression, value: str) -> bool:
    """Do not replace a proven shared-cache read with a global vector load."""
    names = expression.dependencies(value)
    return any(
        isinstance(record, EarlyAuxiliaryInput) and record.shared in names
        for record in expression.scan_inputs
    )


def _unit_stride(candidate: chain._ScanInput) -> bool:
    coordinate = candidate.coordinate
    slopes = tuple(sympy.diff(value, coordinate) for value in candidate.indices)
    if any(not isinstance(slope, sympy.Integer) for slope in slopes):
        return False
    if any(
        sympy.expand(
            sympy.Add(
                value,
                sympy.Mul(-1, value.subs(coordinate, 0)),
                sympy.Mul(-1, slope, coordinate),
            )
        )
        != 0
        for value, slope in zip(candidate.indices, slopes, strict=True)
    ):
        return False
    return (
        sum(
            slope * stride
            for slope, stride in zip(
                slopes, candidate.source.meta["val"].stride(), strict=True
            )
        )
        == 1
    )


def _make_auxiliary_cache(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    existing: list[chain._ScanInput],
    roots: tuple[Node, ...],
    budget_bytes: int,
    *,
    arena: str | None = None,
) -> tuple[list[str], list[chain._ScanInput]]:
    """Share typed admission, deduplication, masked filling, and publication.

    Early caches allocate fresh storage and require unit-stride leaves. Late
    caches borrow the caller's proven-dead operand arena; they may be strided.
    Both policies account for each 128-byte-aligned allocation independently.
    """
    lines: list[str] = []
    records: list[chain._ScanInput] = []
    offset = 0
    for operand in _leaves(roots, plan):
        extent = chain._shape(operand)[0]
        size = extent * operand.meta["val"].element_size()
        aligned_size = (size + 127) // 128 * 128
        if extent <= 0 or offset + aligned_size > budget_bytes:
            continue
        phase = "early" if arena is None else "late"
        prefix = f"chain_{phase}_aux_{len(records)}"
        index = f"{prefix}_index"
        expression = chain._Expression(cg, plan, {})
        value = expression.value(operand, (index,))
        indices = next(
            indices
            for node, coords, indices, loaded in expression.loaded_inputs
            if node is operand and coords == (index,)
        )
        candidate = chain._scan_input(
            expression, operand, (index,), indices, prefix, extent, index
        )
        if candidate is None or (arena is None and not _unit_stride(candidate)):
            continue
        if any(
            _covered(cg, plan, candidate, record) for record in (*existing, *records)
        ):
            continue
        dtype = CompileEnvironment.current().backend.dtype_str(
            operand.meta["val"].dtype
        )
        domain = chain._operand_domain(cg, operand, (index,), plan)
        if arena is None:
            allocation = f"cute.arch.alloc_smem({dtype}, {extent}, alignment=128)"
            candidate = EarlyAuxiliaryInput(
                candidate.source,
                candidate.operand,
                candidate.shared,
                candidate.extent,
                candidate.indices,
                candidate.coordinate,
                candidate.inverse_axis,
            )
        else:
            # Operand arenas have two-byte elements; offsets are aligned bytes.
            allocation = f"cute.recast_ptr({arena} + {offset // 2}, dtype={dtype})"
        lines.extend(
            (
                f"{prefix} = cute.make_tensor({allocation}, cute.make_layout({extent}))",
                f"for {prefix}_step in cutlass.range_constexpr({(extent + plan.threads - 1) // plan.threads}):",
                f"    {index} = chain_thread + {prefix}_step * {plan.threads}",
                chain._indent(expression.lines),
                f"    if {index} < {extent}:",
                f"        {prefix}[{index}] = {chain._masked_operand(value, dtype, domain)}",
            )
        )
        records.append(candidate)
        offset += aligned_size
    if records:
        lines.append("cute.arch.sync_threads()")
    return lines, records


def make_early_auxiliary_cache(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    existing: list[chain._ScanInput],
    budget_bytes: int,
) -> tuple[list[str], list[chain._ScanInput]]:
    """Cache readonly independent operands without borrowing live storage."""
    roots = tuple(
        cast("Node", operand)
        for dot in plan.dots
        for operand in dot.args[:2]
        if not chain._direct_operand(cast("Node", operand))
        and not (chain._ancestors(cast("Node", operand)) & {*plan.dots, *plan.scans})
    )
    return _make_auxiliary_cache(cg, plan, existing, roots, budget_bytes)


def make_late_auxiliary_cache(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    existing: list[chain._ScanInput],
    *,
    arena: str,
    arena_bytes: int,
) -> tuple[list[str], list[chain._ScanInput]]:
    """Borrow a dead operand workspace after its last MMA use has retired.

    The caller proves this is the final TMEM-A stage and output does not alias
    the arena. Descriptors are published only after the emitted fill barrier.
    """
    roots = (cast("Node", plan.dots[-1].args[0]), cast("Node", plan.store.args[2]))
    return _make_auxiliary_cache(cg, plan, existing, roots, arena_bytes, arena=arena)
