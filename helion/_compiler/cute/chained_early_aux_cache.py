"""Readonly coefficient vectors for independent computed MMA operands."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import sympy

from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_aux_cache import _covered
from .chained_aux_cache import _leaves

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


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


def make_early_auxiliary_cache(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    existing: list[chain._ScanInput],
    budget_bytes: int,
) -> tuple[list[str], list[chain._ScanInput]]:
    """Cache unit-stride typed leaves without borrowing any live operand.

    The admitted chain has one store to a genuinely fresh host allocation and
    no other writes. Its host input leaves are therefore readonly, including
    mutually aliasing input views. Source identity and complete logical maps,
    not storage overlap, control reuse. Original masked loads fill the cache;
    original domain checks and scalar fallbacks still guard every read.
    """
    roots = tuple(
        cast("Node", operand)
        for dot in plan.dots
        for operand in dot.args[:2]
        if not chain._direct_operand(cast("Node", operand))
        and not (chain._ancestors(cast("Node", operand)) & {*plan.dots, *plan.scans})
    )
    lines: list[str] = []
    records: list[chain._ScanInput] = []
    for operand in _leaves(roots, plan):
        extent = chain._shape(operand)[0]
        size = extent * operand.meta["val"].element_size()
        aligned_size = (size + 127) // 128 * 128
        if extent <= 0 or aligned_size > budget_bytes:
            continue
        prefix = f"chain_early_aux_{len(records)}"
        index = f"{prefix}_index"
        expression = chain._Expression(cg, plan, {})
        value = expression.value(operand, (index,))
        indices = next(
            indices
            for node, coords, indices, _ in expression.loaded_inputs
            if node is operand and coords == (index,)
        )
        candidate = chain._scan_input(
            expression, operand, (index,), indices, prefix, extent, index
        )
        if candidate is None:
            continue
        coordinate = candidate.coordinate
        slopes = tuple(sympy.diff(value, coordinate) for value in candidate.indices)
        if any(not isinstance(slope, sympy.Integer) for slope in slopes):
            continue
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
            continue
        strides = candidate.source.meta["val"].stride()
        if (
            sum(slope * stride for slope, stride in zip(slopes, strides, strict=True))
            != 1
        ):
            continue
        if any(
            _covered(cg, plan, candidate, record) for record in (*existing, *records)
        ):
            continue
        dtype = CompileEnvironment.current().backend.dtype_str(
            operand.meta["val"].dtype
        )
        domain = chain._operand_domain(cg, operand, (index,), plan)
        lines.extend(
            (
                f"{prefix} = cute.make_tensor(cute.arch.alloc_smem({dtype}, {extent}, alignment=128), cute.make_layout({extent}))",
                f"for {prefix}_step in cutlass.range_constexpr({(extent + plan.threads - 1) // plan.threads}):",
                f"    {index} = chain_thread + {prefix}_step * {plan.threads}",
                chain._indent(expression.lines),
                f"    if {index} < {extent}:",
                f"        {prefix}[{index}] = {chain._masked_operand(value, dtype, domain)}",
            )
        )
        records.append(
            EarlyAuxiliaryInput(
                candidate.source,
                candidate.operand,
                candidate.shared,
                candidate.extent,
                candidate.indices,
                candidate.coordinate,
                candidate.inverse_axis,
            )
        )
        budget_bytes -= aligned_size
    if records:
        lines.append("cute.arch.sync_threads()")
    return lines, records
