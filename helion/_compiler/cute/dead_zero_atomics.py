"""Prove private zero-update returns dead at every observable consumer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import scan_ops
from ..device_ir import RootGraphInfo
from ..inductor_lowering import ReductionLowering
from .local_atomic import atomic_target_origins
from .local_atomic_registers import local_atomic_register_chain
from .register_loads import host_load_is_readonly

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo

# Bound proof work, not tensor geometry. Unknown predicates remain independent.
_MAX_BOOLEAN_ATOMS = 12
_INTEGER_DTYPES = (torch.int32, torch.int64, torch.bool)
_TOTAL_CONSUMERS = frozenset(
    (
        _tracing_ops._new_var,
        torch.ops.aten.alias.default,
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.where.self,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.bitwise_xor.Tensor,
        torch.ops.aten.bitwise_not.default,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.ne.Tensor,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.bitwise_and.Scalar,
        torch.ops.aten.bitwise_or.Scalar,
        torch.ops.aten.bitwise_xor.Scalar,
    )
)


def _boolean_update(value: object) -> Node | None:
    """Only Boolean 0/1 updates: converted Int32 zero is exactly false."""
    while isinstance(value, Node):
        fake = value.meta.get("val")
        if not isinstance(fake, torch.Tensor):
            return None
        if fake.dtype == torch.bool:
            return value
        if value.target in (_tracing_ops._new_var, torch.ops.aten.alias.default) or (
            value.target is torch.ops.prims.convert_element_type.default
            and fake.dtype in (torch.int32, torch.int64)
        ):
            value = value.args[0]
        else:
            return None
    return None


def _masks_imply_update(update: Node, masks: list[Node]) -> bool:
    """Prove each mask implies update for all Boolean atom assignments.

    Comparisons involving the unknown returned value are unconstrained atoms;
    no numeric assumption about a zero-add return enters this implication.
    """
    atoms: dict[Node, int] = {}
    operators = {
        torch.ops.aten.bitwise_and.Tensor: "and",
        torch.ops.aten.bitwise_or.Tensor: "or",
        torch.ops.aten.bitwise_xor.Tensor: "xor",
        torch.ops.aten.bitwise_not.default: "not",
        torch.ops.aten.where.self: "where",
    }

    def formula(node: Node) -> tuple[object, ...]:
        if node.target in (_tracing_ops._new_var, torch.ops.aten.alias.default):
            return formula(cast("Node", node.args[0]))
        if node.target in operators and all(
            isinstance(arg, Node)
            and isinstance(arg.meta.get("val"), torch.Tensor)
            and arg.meta["val"].dtype == torch.bool
            for arg in node.args
        ):
            return (
                operators[node.target],
                *(formula(cast("Node", arg)) for arg in node.args),
            )
        return ("atom", atoms.setdefault(node, len(atoms)))

    expressions = [formula(node) for node in (update, *masks)]
    if len(atoms) > _MAX_BOOLEAN_ATOMS:
        return False

    def evaluate(expr: tuple[object, ...], values: int) -> bool:
        op, *args = expr
        if op == "atom":
            return bool(values & (1 << cast("int", args[0])))
        operands = [evaluate(cast("tuple[object, ...]", arg), values) for arg in args]
        if op == "and":
            return operands[0] and operands[1]
        if op == "or":
            return operands[0] or operands[1]
        if op == "xor":
            return operands[0] != operands[1]
        if op == "not":
            return not operands[0]
        assert op == "where"
        return operands[1] if operands[0] else operands[2]

    return all(
        evaluate(expressions[0], values)
        or not any(evaluate(expr, values) for expr in expressions[1:])
        for values in range(1 << len(atoms))
    )


def _immutable_sources(
    sources: list[Node],
    node: Node,
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    *,
    allow_unbound: bool,
) -> bool:
    seen: set[Node] = set()

    def immutable(source: Node) -> bool:
        if source in seen or source is node:
            return True
        seen.add(source)
        if source.graph is not node.graph:
            return False
        target = source.target
        if target is memory_ops.load:
            if not host_load_is_readonly(
                source, env, graphs, allow_unbound=allow_unbound
            ):
                return False
            return all(immutable(arg) for arg in source.all_input_nodes[1:])
        if target not in (
            _tracing_ops._new_var,
            _tracing_ops._get_symnode,
            _tracing_ops._mask_to,
            torch.ops.aten.alias.default,
            creation_ops.full,
            scan_ops._associative_scan,
        ) and not (
            isinstance(target, torch._ops.OpOverload)
            and not target._schema.is_mutable
            and torch.Tag.nondeterministic_seeded not in target.tags
            and (
                torch.Tag.pointwise in target.tags
                or isinstance(source.meta.get("lowering"), ReductionLowering)
                or target
                in (
                    torch.ops.prims.iota.default,
                    torch.ops.aten.view.dtype,
                    torch.ops.aten.scalar_tensor.default,
                )
            )
        ):
            return False
        return all(immutable(arg) for arg in source.all_input_nodes)

    return all(immutable(source) for source in sources)


def dead_zero_atomic_results(
    graphs: list[GraphInfo], env: CompileEnvironment, *, allow_unbound: bool = False
) -> frozenset[Node]:
    """Share root admission between coverage and emission.

    Local allocation ownership/epochs are separately required by complete
    fragment admission. Same-coordinate return consumers must be total integer
    pointwise expressions ending only at masked host stores. The mask must
    imply a Boolean update, including every independent ticket comparison.
    Loads, remapping, reductions, other atomic effects, captures and output
    escapes of a returned value are excluded. Its existing immutable snapshot,
    initialization, allocation and publication remain in the emitter.
    """
    targets = atomic_target_origins(graphs)
    accepted: set[Node] = set()
    for info in graphs:
        if not isinstance(info, RootGraphInfo):
            continue
        for node in info.graph.nodes:
            if node not in targets:
                continue
            update = _boolean_update(node.args[2])
            if update is None:
                continue
            chain = local_atomic_register_chain(node, env, targets, dead_outputs=False)
            if not chain:
                continue
            masks = []
            for user in chain - {node}:
                if user.target is memory_ops.store:
                    mask = user.args[3]
                    if (
                        not isinstance(mask, Node)
                        or mask.meta["val"].dtype != torch.bool
                    ):
                        break
                    masks.append(mask)
                elif (
                    user.target not in _TOTAL_CONSUMERS
                    or user.meta["val"].dtype not in _INTEGER_DTYPES
                    or any(
                        isinstance(arg.meta.get("val"), torch.Tensor)
                        and arg.meta["val"].dtype not in _INTEGER_DTYPES
                        for arg in user.all_input_nodes
                    )
                ):
                    break
            else:
                if not masks or not _masks_imply_update(update, masks):
                    continue
                if _immutable_sources(
                    [update, *masks], node, env, graphs, allow_unbound=allow_unbound
                ):
                    accepted.add(node)
    return frozenset(accepted)
