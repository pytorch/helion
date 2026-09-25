"""Opt-in FP32 seed reuse for two independent TCgen05 contractions.

This deliberately reassociates the final FP32 addition into the second MMA.
Discovery is structural only; exact coordinates, availability and layout are
validated by the ordinary expression emitter before any source is committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import view_ops
from ...language.matmul_ops import dot
from ..inductor_lowering import PointwiseLowering

if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _ScanInput


@dataclass(frozen=True)
class InitializedAccumulator:
    first: Node
    second: Node
    seed: Node
    join: Node
    exclusive: frozenset[Node]
    scans: frozenset[Node]


class _NotCandidate(Exception):
    pass


_ARITHMETIC = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
}


def _value(node: Node) -> torch.Tensor:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise _NotCandidate
    return value


def _shape(node: Node) -> tuple[int | sympy.Basic, ...]:
    return tuple(
        size.node.expr if isinstance(size, torch.SymInt) else size
        for size in _value(node).shape
    )


def _fp32(node: Node) -> None:
    if _value(node).dtype != torch.float32:
        raise _NotCandidate


def _classify(nodes: Sequence[Node]) -> InitializedAccumulator:
    from .chained_matmul import _ancestors

    dots = [node for node in nodes if node.target is dot]
    stores = [node for node in nodes if node.target is memory_ops.store]
    if len(dots) != 2 or len(stores) != 1:
        raise _NotCandidate
    first, second = dots
    for contraction in dots:
        _fp32(contraction)
        if (
            len(_shape(contraction)) != 2
            or len(contraction.args) != 4
            or contraction.args[2:] != (None, None)
            or contraction.kwargs
        ):
            raise _NotCandidate
    if _shape(first) != _shape(second) or first in _ancestors(second):
        raise _NotCandidate
    if len(second.users) != 1:
        raise _NotCandidate
    join = next(iter(second.users))
    if (
        join.target is not torch.ops.aten.add.Tensor
        or len(join.args) != 2
        or join.args[1] is not second
        or join.kwargs.get("alpha", 1) != 1
        or set(join.kwargs) - {"alpha"}
        or not isinstance(join.meta.get("lowering"), PointwiseLowering)
    ):
        raise _NotCandidate
    seed = join.args[0]
    if not isinstance(seed, Node) or first not in _ancestors(seed):
        raise _NotCandidate
    _fp32(join)
    output = stores[0].args[2]
    if (
        not isinstance(output, Node)
        or _shape(join) != _shape(first)
        or _shape(output) != _shape(first)
        or join not in _ancestors(output)
    ):
        raise _NotCandidate
    exclusive: set[Node] = set()
    scans: set[Node] = set()

    def coefficient(node: Node) -> None:
        _fp32(node)
        if node.target is scan_ops._associative_scan:
            scans.add(node)
            return
        if node.target is memory_ops.load:
            source = node.args[0]
            if (
                not isinstance(source, Node)
                or source.target is not _tracing_ops._host_tensor
            ):
                raise _NotCandidate
            return
        if node.target is view_ops.subscript:
            coefficient(cast("Node", node.args[0]))
            return
        if node.target not in _ARITHMETIC | {
            torch.ops.aten.exp.default
        } or not isinstance(node.meta.get("lowering"), PointwiseLowering):
            raise _NotCandidate
        for item in node.all_input_nodes:
            coefficient(item)

    def visit(node: Node) -> None:
        if node in exclusive:
            return
        _fp32(node)
        if _shape(node) != _shape(first):
            raise _NotCandidate
        exclusive.add(node)
        if node is first:
            return
        if (
            node.target not in _ARITHMETIC
            or not isinstance(node.meta.get("lowering"), PointwiseLowering)
            or node.kwargs.get("alpha", 1) != 1
            or set(node.kwargs) - {"alpha"}
        ):
            raise _NotCandidate
        for item in node.all_input_nodes:
            if first in _ancestors(item):
                visit(item)
            else:
                coefficient(item)

    visit(seed)
    if any(set(node.users) - exclusive - {join} for node in exclusive):
        raise _NotCandidate
    return InitializedAccumulator(
        first, second, seed, join, frozenset(exclusive), frozenset(scans)
    )


def classify_initialized_accumulator(
    nodes: Sequence[Node],
) -> InitializedAccumulator | None:
    try:
        return _classify(nodes)
    except _NotCandidate:
        return None


def has_initialized_candidate(graphs: Sequence[GraphInfo]) -> bool:
    from .chained_matmul import _root_graph

    root = _root_graph(graphs)
    return (
        root is not None
        and classify_initialized_accumulator(tuple(root.graph.nodes)) is not None
    )


def _seed_geometry_supported(
    plan: ChainedMatmulPlan, inner_axes: dict[tuple[int, str], int]
) -> bool:
    """The shared M128 FP32 C map is independent of SMEM operand major mode.

    Ld/St32x32b repetition32 gives thread t the N slots (t, 0..N-1),
    for either K/MN mode of either operand in each contraction. This does
    not extend the sparse M64 map or replace semantic/SMEM admission.
    """
    return (
        plan.strategy == "tcgen05_tmem"
        and plan.threads == 128
        and plan.dtype in (torch.bfloat16, torch.float16)
        and len(plan.dots) == len(plan.shapes) == 2
        and plan.shapes[0][:2] == plan.shapes[1][:2]
        and all(
            m == 128 and 32 <= n <= 256 and n % 32 == 0 and k > 0 and k % 16 == 0
            for m, n, k in plan.shapes
        )
        and all(_value(node).dtype == torch.float32 for node in plan.dots)
        and bool(plan.axes)
        and all(
            size > 0 and block > 0 and size % block == 0 for _, size, block in plan.axes
        )
        and set(inner_axes)
        == {(stage, role) for stage in (0, 1) for role in ("a", "b")}
        and all(axis in (0, 1) for axis in inner_axes.values())
    )


def codegen_seed(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[_ScanInput],
    inner_axes: dict[tuple[int, str], int],
) -> list[str]:
    """Prove actual expression availability/coordinates, then render per-slot.

    With the proven M128 C partition and Ld/St32x32b repetition32,
    each lane owns (row=thread, col=slot), for every admitted N multiple32.
    The register and collective store partitions therefore have identical
    slot order; four warps own disjoint rows. No row-hoisting assumption is
    needed here. Unknown operand-major forms are intentionally rejected.
    """
    from . import chained_matmul as chain

    selected = plan.initialized_accumulator
    assert selected is not None
    if (
        plan.scan_exports
        or not _seed_geometry_supported(plan, inner_axes)
        or not selected.scans.issubset(boundaries)
    ):
        raise chain._UnsupportedChain("initialized accumulator layout or boundary")
    coords = ("chain_seed_row", "chain_seed_col")
    index = "chain_seed_index"
    expression = chain._Expression(cg, plan, boundaries)
    expression.scan_inputs = scans
    expression.coordinate_names.update(coords)
    expression.fragments[selected.first] = (coords, f"chain_0_values[{index}]")
    value = expression.value(selected.seed, coords)
    # Prove final consumer uses the join at exactly the same coordinates.
    final = chain._Expression(cg, plan, boundaries)
    final.scan_inputs = scans
    final.coordinate_names.update(coords)
    final.fragments[selected.join] = (coords, f"chain_1_values[{index}]")
    final.value(cast("Node", plan.store.args[2]), coords)
    return [
        f"for {index} in cutlass.range_constexpr(cute.size(chain_0_values)):",
        f"    {coords[0]}, {coords[1]} = chain_0_coords[{index}]",
        chain._indent(expression.lines),
        f"    chain_0_values[{index}] = cutlass.Float32({value})",
        "chain_seed_copy = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32)), cutlass.Float32), chain_0_acc)",
        "chain_seed_target = chain_seed_copy.get_slice(chain_thread).partition_D(chain_0_acc)",
        "cute.copy(chain_seed_copy, chain_0_values, chain_seed_target)",
        "cute.arch.fence_view_async_tmem_store()",
    ]
