"""One-axis FP32 coefficients in a proven dead, separate output arena."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_index
from ...language import tile_ops
from . import chained_matmul as chain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


# This is a purity/precision whitelist, not an alternate expression evaluator.
# Actual operations, casts and masks always come from _Expression's lowerings.
_MATH = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.neg.default,
    torch.ops.aten.exp.default,
    torch.ops.aten.exp2.default,
    torch.ops.aten.clamp.default,
    torch.ops.aten.clamp_min.default,
    torch.ops.aten.clamp_max.default,
    torch.ops.aten.minimum.default,
    torch.ops.aten.maximum.default,
}
_CASTS = {
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
    torch.ops.prims.convert_element_type.default,
}
_LEAF_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _pure_computed(node: Node, published: set[Node]) -> bool:
    pending, seen = [node], set()
    computed = False
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in published:
            continue
        target = current.target
        value = current.meta.get("val")
        if target in _MATH or target in _CASTS:
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                return False
            computed |= target in _MATH
        elif target in (memory_ops.load, _tracing_ops._host_tensor):
            if not isinstance(value, torch.Tensor) or value.dtype not in _LEAF_DTYPES:
                return False
        elif target not in {
            *chain._VIEWS,
            *chain._SCALAR_BINARY,
            _tracing_ops._get_symnode,
            tile_ops.tile_begin,
            tile_index,
            torch.ops.prims.iota.default,
            torch.ops.aten.scalar_tensor.default,
        }:
            return False
        pending.extend(current.all_input_nodes)
    return computed


def _candidate(node: Node, published: set[Node]) -> bool:
    value = node.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and value.ndim == 1
        and value.dtype == torch.float32
        and _pure_computed(node, published)
    )


def has_coefficient_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Discovery superset; codegen proves projection, availability and arena."""
    root = chain._root_graph(graphs)
    if root is None:
        return False
    nodes = tuple(root.graph.nodes)
    dots = [node for node in nodes if node.target is chain.dot]
    if len(dots) != 1:
        return False
    primary = [
        node
        for node in nodes
        if node.target is memory_ops.store
        and isinstance(node.args[2], torch.fx.Node)
        and dots[0] in chain._ancestors(node.args[2])
    ]
    if len(primary) != 1 or primary[0].args[0].meta["val"].dtype != torch.float32:
        return False
    scans = {node for node in nodes if node.target is scan_ops._associative_scan}
    return any(
        _candidate(node, scans)
        for operand in dots[0].args[:2]
        for node in chain._ancestors(cast("Node", operand))
    )


def separate_output_capacity(plan: ChainedMatmulPlan) -> int:
    """FP32 cells, only for the independent allocation in this exact emitter.

    A single contraction cannot prefetch a final RHS or have a bridge. Neither
    A nor B is borrowed: chain_output_ptr is its own 128-byte-aligned allocation.
    Its first data use is the final epilogue, after the existing pre-MMA CTA
    barrier has completed all coefficient reads in both operand branches.
    """
    if (
        plan.strategy != "tcgen05_tmem"
        or len(plan.dots) != 1
        or plan.threads != 128
        or plan.direct_output
        or plan.late_rhs_reuse is not None
        or plan.initialized_accumulator is not None
        or cast("Node", plan.store.args[0]).meta["val"].dtype != torch.float32
        or any(extent % block for _, extent, block in plan.axes)
    ):
        return 0
    return math.prod(plan.shapes[0][:2])


def _axis_projection(
    expression: chain._Expression,
    coordinate: str,
    names: tuple[str, str],
    shape: tuple[int, ...],
    extent: int,
) -> bool:
    if len(shape) != 2:
        return False
    symbols = {
        name: sympy.Symbol(name, integer=True)
        for name in (*names, *expression.origins.values())
    }
    try:
        query = sympy.expand(
            chain._copy_index(coordinate, expression.definitions, symbols)
        )
    except chain._UnsupportedChain:
        return False
    # No offset, wrap, fixed-width-cast erasure, gather, flattened pair or tail.
    return any(
        query == symbols[name] and extent == size
        for name, size in zip(names, shape, strict=True)
    )


def _constant_true_domain(bounds: list[str]) -> bool:
    """Accept only literal tautologies (for example a scan's terminal index).

    Dynamic/tail bounds are not discharged here. The coefficient consumer's
    exact full-axis projection is a separate proof, not a mask simplifier.
    """
    for bound in bounds:
        tree = ast.parse(bound, mode="eval").body
        if not isinstance(tree, ast.Compare):
            return False
        terms = [tree.left, *tree.comparators]
        if any(
            not isinstance(term, ast.Constant) or type(term.value) is not int
            for term in terms
        ):
            return False
        values = [cast("int", cast("ast.Constant", term).value) for term in terms]
        if any(
            not (
                (isinstance(op, ast.Lt) and left < right)
                or (isinstance(op, ast.LtE) and left <= right)
            )
            for op, left, right in zip(tree.ops, values[:-1], values[1:], strict=True)
        ):
            return False
    return True


def make_coefficient_cache(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    published: set[Node],
) -> tuple[list[str], dict[Node, str]]:
    capacity = separate_output_capacity(plan)
    if capacity == 0:
        raise chain._UnsupportedChain(
            "coefficient cache requires a separate FP32 output arena"
        )
    eligible: dict[Node, bool] = {}
    for operand in plan.dots[0].args[:2]:
        operand = cast("Node", operand)
        names = ("chain_coefficient_probe_row", "chain_coefficient_probe_col")
        expression = chain._Expression(cg, plan, boundaries)
        expression.scan_inputs = scans
        expression.coordinate_names.update(names)
        expression.value(operand, names)
        for node, coordinates in expression.memo:
            if not _candidate(node, published):
                continue
            (extent,) = chain._shape(node)
            valid = len(coordinates) == 1 and _axis_projection(
                expression, coordinates[0], names, chain._shape(operand), extent
            )
            eligible[node] = eligible.get(node, True) and valid
    chosen = [node for node, valid in eligible.items() if valid]
    # Retain maximal complete expressions, not their nested arithmetic pieces.
    chosen = [
        node
        for node in chosen
        if not any(
            node is not other and node in chain._ancestors(other) for other in chosen
        )
    ]
    if not chosen or len(chosen) > 4:
        raise chain._UnsupportedChain(
            "coefficient cache requires pure one-axis FP32 expressions"
        )
    sizes = [chain._shape(node)[0] for node in chosen]
    if min(sizes) <= 0 or sum(sizes) > capacity:
        raise chain._UnsupportedChain("coefficient cache exceeds dead output capacity")
    lines, cached = [], {}
    offset = 0
    outputs = [
        cast("Node", plan.store.args[0]),
        *(cast("Node", export.store.args[0]) for export in plan.scan_exports),
    ]
    for number, (node, extent) in enumerate(zip(chosen, sizes, strict=True)):
        tag = f"chain_coefficient_{number}"
        index = f"{tag}_index"
        expression = chain._Expression(cg, plan, boundaries)
        expression.scan_inputs = scans
        expression.coordinate_names.add(index)
        value = expression.value(node, (index,))
        if not _constant_true_domain(chain._operand_domain(cg, node, (index,), plan)):
            raise chain._UnsupportedChain("coefficient cache requires an unpadded axis")
        if any(
            source.meta["val"].untyped_storage() is target.meta["val"].untyped_storage()
            for source, _ in expression.global_accesses
            for target in outputs
        ):
            raise chain._UnsupportedChain("coefficient cache source aliases an output")
        lines.extend(
            (
                f"{tag} = cute.make_tensor(chain_output_ptr + {offset}, cute.make_layout({extent}))",
                f"for {tag}_step in cutlass.range_constexpr({(extent + plan.threads - 1) // plan.threads}):",
                f"    {index} = chain_thread + {tag}_step * {plan.threads}",
                f"    if {index} < {extent}:",
                chain._indent(expression.lines, 8),
                f"        {tag}[{index}] = cutlass.Float32({value})",
            )
        )
        cached[node] = tag
        offset += extent
    lines.append("cute.arch.sync_threads()")
    return lines, cached
