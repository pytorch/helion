"""Typed raw-vector caches in a proven-dead contraction operand workspace."""

from __future__ import annotations

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


def make_late_auxiliary_cache(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    existing: list[chain._ScanInput],
    *,
    arena: str,
    arena_bytes: int,
) -> tuple[list[str], list[chain._ScanInput]]:
    """Fill a borrowed, dead BF16/FP16 workspace without new shared storage.

    The caller proves this is the final TMEM-A stage, all previous SMEM-A
    consumers have completed, and output storage does not alias the arena.
    Returned descriptors must become visible only after the emitted barrier.
    """
    roots = (cast("Node", plan.dots[-1].args[0]), cast("Node", plan.store.args[2]))
    lines: list[str] = []
    records: list[chain._ScanInput] = []
    offset = 0
    for operand in _leaves(roots, plan):
        extent = chain._shape(operand)[0]
        size = extent * operand.meta["val"].element_size()
        aligned_size = (size + 127) // 128 * 128
        if extent <= 0 or offset + aligned_size > arena_bytes:
            continue
        prefix = f"chain_late_aux_{len(records)}"
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
        if candidate is None or any(
            _covered(cg, plan, candidate, record) for record in (*existing, *records)
        ):
            continue
        dtype = CompileEnvironment.current().backend.dtype_str(
            operand.meta["val"].dtype
        )
        domain = chain._operand_domain(cg, operand, (index,), plan)
        lines.extend(
            (
                # The chain operand arena has a two-byte scalar element type.
                # Every slice starts at a128-byte aligned byte offset.
                f"{prefix} = cute.make_tensor(cute.recast_ptr({arena} + {offset // 2}, dtype={dtype}), cute.make_layout({extent}))",
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
