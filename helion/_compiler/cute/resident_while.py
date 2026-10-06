"""Current-call structure and logical domains for resident while lowering.

The plan identifies current call edges and initialized backedges. It does not
prove physical uniformity, gather bounds, storage ownership or emission support.
Those obligations remain explicit even for a pure Boolean scalar condition.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import inline_asm_ops
from ...language import memory_ops
from ..device_ir import ForLoopGraphInfo
from ..device_ir import GraphInfo
from ..device_ir import RootGraphInfo
from ..device_ir import WhileConditionGraphInfo
from ..device_ir import WhileLoopGraphInfo
from ..inductor_lowering import SympyExprLowering
from .bounded_gather import inline_asm_shape

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from .gather_domains import GatherDomainFacts


@dataclass(frozen=True)
class ResidentCallContext:
    """One actual call, without consulting potentially stale node_args."""

    call: Node
    graph: GraphInfo
    captures: tuple[Node, ...]
    placeholders: tuple[Node, ...]
    outputs: tuple[Node, ...]
    carry_map: tuple[tuple[int, int], ...]
    parent_call: Node | None


@dataclass(frozen=True)
class ResidentWhilePlan:
    root: RootGraphInfo
    condition: ResidentCallContext
    body: ResidentCallContext
    nested_fors: tuple[ResidentCallContext, ...]
    invariant_slots: tuple[int, ...]
    requirements: tuple[str, ...] = (
        "whole_cta_entry_and_shared_predicate_publication",
        "predicate_reader_barrier_before_overwrite",
        "current_call_logical_domains_and_lowering_support",
        "readonly_capture_ownership_without_mutable_range_inheritance",
        "simultaneous_carry_snapshots_and_nested_storage_lifetimes",
        "configured_gather_capability_and_resource_budget",
    )


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise exc.InvalidConfig(f"resident while plan: {reason}")


def _signature(value: Node) -> tuple[torch.dtype, tuple[int, ...]]:
    fake = value.meta.get("val")
    _require(isinstance(fake, torch.Tensor), "tensor metadata required")
    fake = cast("torch.Tensor", fake)
    _require(
        all(type(size) is int and size > 0 for size in fake.shape),
        "positive static physical shape required",
    )
    return fake.dtype, cast("tuple[int, ...]", tuple(fake.shape))


def _outputs(info: GraphInfo) -> tuple[Node, ...]:
    nodes = list(info.graph.find_nodes(op="output"))
    _require(len(nodes) == 1, "one graph output required")
    values = nodes[0].args[0]
    _require(
        isinstance(values, (list, tuple))
        and all(isinstance(value, Node) for value in values),
        "tensor output list required",
    )
    return tuple(cast("Sequence[Node]", values))


def _carry_map(
    call: Node, captures: tuple[Node, ...], outputs: tuple[Node, ...]
) -> tuple[tuple[int, int], ...]:
    slots: dict[int, int] = {}
    order = {node: index for index, node in enumerate(call.graph.nodes)}
    _require(call in order, "loop call is absent from caller graph")
    for item in call.users:
        _require(
            item.op == "call_function"
            and item.graph is call.graph
            and item in order
            and order[item] > order[call]
            and item.target is operator.getitem
            and len(item.args) == 2
            and not item.kwargs
            and item.args[0] is call
            and type(item.args[1]) is int
            and 0 <= item.args[1] < len(outputs),
            "invalid loop output projection",
        )
        index = cast("int", item.args[1])
        for phi in item.users:
            _require(
                phi.op == "call_function"
                and phi.graph is call.graph
                and phi in order
                and order[phi] > order[item]
                and phi.target is _tracing_ops._phi
                and len(phi.args) == 2
                and not phi.kwargs
                and phi.args[1] is item,
                "output must have an explicit initialized phi",
            )
            matches = [i for i, value in enumerate(captures) if value is phi.args[0]]
            _require(len(matches) == 1, "ambiguous phi entry capture")
            slot = matches[0]
            _require(index not in slots or slots[index] == slot, "conflicting phi")
            _require(
                _signature(outputs[index]) == _signature(captures[slot]),
                "carry physical shape or dtype changed",
            )
            _require(
                _signature(item) == _signature(outputs[index]),
                "projection physical shape or dtype changed",
            )
            _require(
                _signature(phi) == _signature(captures[slot]),
                "phi physical shape or dtype changed",
            )
            slots[index] = slot
    _require(set(slots) == set(range(len(outputs))), "uninitialized loop output")
    _require(len(set(slots.values())) == len(slots), "duplicate carry destination")
    return tuple(sorted(slots.items()))


def resident_while_plan(call: Node, graphs: Sequence[GraphInfo]) -> ResidentWhilePlan:
    """Inspect one immediate root while and its static-for child call contexts.

    Every returned requirement must be discharged by a future emitter/admission
    integration. This plan alone does not authorize lowering.
    """
    roots = [info for info in graphs if isinstance(info, RootGraphInfo)]
    _require(
        len(roots) == 1 and call.graph is roots[0].graph,
        "one immediate root required",
    )
    _require(
        call.op == "call_function"
        and call.target is _tracing_ops._while_loop
        and len(call.args) == 4
        and call.args[3] is None
        and not call.kwargs,
        "ordinary while call required",
    )
    by_id = {info.graph_id: info for info in graphs}
    _require(len(by_id) == len(graphs), "duplicate graph ID")
    _require(
        all(info.graph_id == index for index, info in enumerate(graphs)),
        "graph list/index identity mismatch",
    )
    _require(
        len({id(info.graph) for info in graphs}) == len(graphs),
        "duplicate underlying graph object",
    )
    # Future domain/emitter caches are indexed by graph ID, not call site.
    # Count every actual control edge, including a second otherwise unused call.
    callers: dict[int, list[Node]] = {}
    control_slots = {
        _tracing_ops._for_loop: (0,),
        _tracing_ops._for_loop_step: (0,),
        _tracing_ops._while_loop: (0, 1, 3),
        _tracing_ops._if: (1, 2),
    }
    for info in graphs:
        for node in info.graph.nodes:
            _require(node.graph is info.graph, "node/graph ownership mismatch")
            if node.target not in control_slots:
                continue
            _require(node.op == "call_function", "control call op mismatch")
            _require(
                node.graph is roots[0].graph
                or node.target not in (_tracing_ops._while_loop, _tracing_ops._if),
                "nested while/if is unsupported",
            )
            for slot in control_slots[node.target]:
                _require(slot < len(node.args), "control call graph argument missing")
                graph_id = node.args[slot]
                if (
                    graph_id is None
                    and node.target is _tracing_ops._while_loop
                    and slot == 3
                ):
                    continue
                _require(type(graph_id) is int, "control graph ID must be an integer")
                callers.setdefault(cast("int", graph_id), []).append(node)

    def context(
        node: Node,
        graph_id: object,
        raw_captures: object,
        kind: type[GraphInfo],
        parent: Node | None,
        *,
        carries: bool,
    ) -> ResidentCallContext:
        _require(type(graph_id) is int and graph_id in by_id, "missing graph ID")
        info = by_id[cast("int", graph_id)]
        _require(isinstance(info, kind), "wrong graph kind")
        _require(
            callers.get(info.graph_id) == [node], "callee must have one actual call"
        )
        _require(
            isinstance(raw_captures, (list, tuple))
            and all(
                isinstance(value, Node) and value.graph is node.graph
                for value in raw_captures
            ),
            "captures must belong to the current caller",
        )
        captures = tuple(cast("Sequence[Node]", raw_captures))
        preceding: set[Node] = set()
        for entry in node.graph.nodes:
            if entry is node:
                break
            preceding.add(entry)
        _require(set(captures) <= preceding, "capture does not precede current call")
        _require(len(set(captures)) == len(captures), "duplicate capture identity")
        placeholders = tuple(info.graph.find_nodes(op="placeholder"))
        _require(len(placeholders) == len(captures), "capture/placeholder arity")
        for entry, placeholder in zip(captures, placeholders, strict=True):
            _require(
                _signature(entry) == _signature(placeholder),
                "capture/placeholder physical shape or dtype",
            )
        outputs = _outputs(info)
        return ResidentCallContext(
            node,
            info,
            captures,
            placeholders,
            outputs,
            _carry_map(node, captures, outputs) if carries else (),
            parent,
        )

    condition = context(
        call, call.args[0], call.args[2], WhileConditionGraphInfo, None, carries=False
    )
    body = context(
        call, call.args[1], call.args[2], WhileLoopGraphInfo, None, carries=True
    )
    _require(
        cast("WhileLoopGraphInfo", body.graph).cond_graph_id
        == condition.graph.graph_id,
        "condition/body graph identity mismatch",
    )
    _require(
        len(condition.outputs) == 1
        and _signature(condition.outputs[0]) == (torch.bool, ()),
        "one Boolean scalar condition required (not a uniformity proof)",
    )

    nested: list[ResidentCallContext] = []
    seen_graphs = {condition.graph.graph_id, body.graph.graph_id}
    for_targets = (_tracing_ops._for_loop, _tracing_ops._for_loop_step)

    def visit(current: ResidentCallContext, *, allow_for: bool) -> None:
        seen: set[Node] = set()
        for node in current.graph.graph.nodes:
            _require(
                all(value in seen for value in node.all_input_nodes),
                "graph input is not a preceding node in this context",
            )
            seen.add(node)
            if node.op in ("placeholder", "output"):
                continue
            _require(node.op == "call_function", "only functional graph nodes")
            if node.target in for_targets:
                _require(allow_for, "condition cannot contain loops")
                expected = 5 if node.target is _tracing_ops._for_loop_step else 4
                _require(len(node.args) == expected and not node.kwargs, "for call ABI")
                child = context(
                    node,
                    node.args[0],
                    node.args[3],
                    ForLoopGraphInfo,
                    current.call,
                    carries=True,
                )
                _require(
                    child.graph.graph_id not in seen_graphs,
                    "reused/recursive child graph",
                )
                seen_graphs.add(child.graph.graph_id)
                axis_count = len(cast("ForLoopGraphInfo", child.graph).block_ids)
                _require(axis_count > 0, "for requires a static axis")
                for bound in node.args[1:3]:
                    _require(
                        isinstance(bound, (list, tuple))
                        and len(bound) == axis_count
                        and all(type(value) is int for value in bound),
                        "nested for bounds must be static integers",
                    )
                if expected == 5:
                    steps = node.args[4]
                    _require(
                        isinstance(steps, (list, tuple))
                        and len(steps) == axis_count
                        and all(
                            value is None or (type(value) is int and value != 0)
                            for value in steps
                        ),
                        "nested for steps must be static nonzero integers",
                    )
                nested.append(child)
                visit(child, allow_for=True)
                continue
            _require(
                node.target not in (_tracing_ops._while_loop, _tracing_ops._if),
                "nested while/if is unsupported",
            )
            if node.target is inline_asm_ops.inline_asm_elementwise:
                # This establishes only the normalized pure scalar ABI and
                # physical broadcast. Logical domains are proved separately.
                _require(
                    not node.kwargs
                    and inline_asm_shape(
                        node,
                        lambda value: tuple(
                            sympy.Integer(size) for size in _signature(value)[1]
                        ),
                        sympy.sympify,
                    )
                    is not None,
                    "unsupported scalar assembly ABI",
                )
                continue
            if node.target is memory_ops.load:
                # Readonly ownership is a whole-region obligation, discharged
                # below after all current call edges have been checked.
                _require(
                    len(node.args) == 4
                    and not node.kwargs
                    and isinstance(node.args[0], Node)
                    and node.args[0].target is _tracing_ops._host_tensor,
                    "resident while load requires a direct host tensor",
                )
                continue
            if node.target in (
                operator.add,
                operator.sub,
                operator.mul,
                operator.floordiv,
                operator.mod,
            ):
                _require(
                    isinstance(node.meta.get("lowering"), SympyExprLowering)
                    and not any(
                        isinstance(value.meta.get("val"), torch.Tensor)
                        for value in node.all_input_nodes
                    ),
                    "symbolic arithmetic requires the existing scalar lowering",
                )
                continue
            if isinstance(node.target, torch._ops.OpOverload):
                _require(
                    not node.target._schema.is_mutable
                    and torch.Tag.nondeterministic_seeded not in node.target.tags
                    and torch.Tag.nondeterministic_bitwise not in node.target.tags,
                    "mutable/nondeterministic operator",
                )
                _require(
                    torch.Tag.pointwise in node.target.tags
                    or node.target
                    in (
                        torch.ops.prims.iota.default,
                        torch.ops.aten.full.default,
                        torch.ops.aten.scalar_tensor.default,
                        torch.ops.prims.sum.default,
                        torch.ops.aten.sum.default,
                        torch.ops.aten.sum.dim_IntList,
                        torch.ops.aten.amax.default,
                        torch.ops.aten.amin.default,
                        torch.ops.aten.max.default,
                        torch.ops.aten.min.default,
                        torch.ops.aten.all.default,
                        torch.ops.aten.any.default,
                        torch.ops.aten.gather.default,
                    ),
                    "operator purity is not established",
                )
                continue
            _require(
                node.target
                in (
                    operator.getitem,
                    _tracing_ops._phi,
                    _tracing_ops._new_var,
                    _tracing_ops._get_symnode,
                    _tracing_ops._constant_tensor,
                    _tracing_ops._host_tensor,
                    _tracing_ops._mask_to,
                    creation_ops.full,
                ),
                "opaque, memory-effect or unsupported operation",
            )

    visit(condition, allow_for=False)
    visit(body, allow_for=True)
    destinations = {slot for _output, slot in body.carry_map}
    return ResidentWhilePlan(
        roots[0],
        condition,
        body,
        tuple(nested),
        tuple(i for i in range(len(body.captures)) if i not in destinations),
    )


def resident_while_domain_facts(
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    incoming: GatherDomainFacts,
    *,
    allow_unbound: bool = False,
) -> GatherDomainFacts:
    """Publish a complete initialized recurrence after every child agrees.

    Mutable entry intervals say nothing about later iterations. Only immutable
    captures with a closed index recipe may retain their entry interval.
    """
    from .bounded_gather import integer_bounds
    from .computed_fragment import _selection_logical_shape
    from .gather_domains import GatherDomainFacts
    from .gather_domains import _readonly_index_recipe
    from .register_loads import host_load_is_readonly

    calls = [
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is _tracing_ops._while_loop
    ]
    if not calls:
        return incoming
    if len(calls) != 1:
        return incoming
    try:
        plan = resident_while_plan(calls[0], graphs)
    except exc.InvalidConfig:
        return incoming
    for context in (plan.condition, plan.body, *plan.nested_fors):
        for node in context.graph.graph.nodes:
            if node.target is memory_ops.load and not host_load_is_readonly(
                node, env, graphs, allow_unbound=allow_unbound
            ):
                return incoming
    pending = list(plan.body.captures)
    seen: set[Node] = set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        if value.target is memory_ops.load and not host_load_is_readonly(
            value, env, graphs, allow_unbound=allow_unbound
        ):
            return incoming
        pending.extend(value.all_input_nodes)
    trial = GatherDomainFacts(
        dict(incoming.shapes),
        dict(incoming.readonly_ranges),
        set(incoming.resident_whiles),
    )

    def shape(value: Node) -> tuple[sympy.Expr, ...] | None:
        return _selection_logical_shape(
            env,
            value,
            scalar_indexed_loads=True,
            tensor_indexed_loads=True,
            pure_inline_asm=True,
            all_axis_reductions=True,
            proven_domains=trial.shapes,
        )

    children = {context.call: context for context in plan.nested_fors}

    def capture_range(
        context: ResidentCallContext, placeholder: Node, entry: Node
    ) -> None:
        # The plan has checked every body instruction for writes and validated
        # the current call edges. A nested call may capture the emitter's typed
        # identity wrapper of an invariant from its parent, but never a mutable
        # carry's initial range. Root wrappers lack this whole-region proof.
        identities: set[Node] = set()
        if context.parent_call is not None:
            preceding: set[Node] = set()
            for node in context.call.graph.nodes:
                if node is context.call:
                    break
                if (
                    node.target is _tracing_ops._new_var
                    and len(node.args) == 1
                    and not node.kwargs
                    and isinstance(node.args[0], Node)
                    and node.args[0] in preceding
                    and _signature(node) == _signature(node.args[0])
                ):
                    identities.add(node)
                preceding.add(node)
        if _readonly_index_recipe(
            entry,
            set(trial.readonly_ranges),
            identity_captures=frozenset(identities),
        ):
            interval = integer_bounds(entry, captured_bounds=trial.readonly_ranges)
            if interval is not None:
                trial.readonly_ranges[placeholder] = interval

    def prove(context: ResidentCallContext) -> bool:
        entry_shapes = []
        destinations = {slot for _output, slot in context.carry_map}
        for slot, (placeholder, entry) in enumerate(
            zip(context.placeholders, context.captures, strict=True)
        ):
            logical = shape(entry)
            if logical is None:
                return False
            entry_shapes.append(logical)
            trial.shapes[placeholder] = logical
            trial.readonly_ranges.pop(placeholder, None)
            if slot not in destinations:
                capture_range(context, placeholder, entry)
        for node in context.graph.graph.nodes:
            if node in children and not prove(children[node]):
                return False
        for index, slot in context.carry_map:
            logical = shape(context.outputs[index])
            expected = entry_shapes[slot]
            if (
                logical is None
                or len(logical) != len(expected)
                or any(
                    sympy.expand(a) != sympy.expand(b)
                    for a, b in zip(logical, expected, strict=True)
                )
            ):
                return False
        slots = dict(context.carry_map)
        for item in context.call.users:
            logical = entry_shapes[slots[cast("int", item.args[1])]]
            trial.shapes[item] = logical
            for phi in item.users:
                trial.shapes[phi] = logical
        return True

    if not prove(plan.body):
        return incoming
    for slot, (placeholder, entry) in enumerate(
        zip(plan.condition.placeholders, plan.condition.captures, strict=True)
    ):
        logical = shape(entry)
        if logical is None:
            return incoming
        trial.shapes[placeholder] = logical
        trial.readonly_ranges.pop(placeholder, None)
        if slot in plan.invariant_slots:
            capture_range(plan.condition, placeholder, entry)
    if shape(plan.condition.outputs[0]) != ():
        return incoming
    trial.resident_whiles.add(plan.body.call)
    return trial
