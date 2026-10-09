"""Call-site scoped logical domains for bounded gathers inside scalar loops."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from itertools import starmap
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import inline_asm_ops
from ..device_ir import ForLoopGraphInfo
from ..device_ir import RootGraphInfo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo


@dataclass
class GatherDomainFacts:
    shapes: dict[Node, tuple[sympy.Expr, ...]] = field(default_factory=dict)
    readonly_ranges: dict[Node, tuple[int, int]] = field(default_factory=dict)
    resident_whiles: set[Node] = field(default_factory=set)


def _carry_map(
    call: Node, captures: list[Node], outputs: list[Node]
) -> dict[int, int] | None:
    """Match every backedge through its actual getitem/phi entry edge."""
    from .computed_fragment import _loop_carry_slots

    slots: dict[int, int] = {}
    for item in call.users:
        if (
            item.target is not operator.getitem
            or len(item.args) != 2
            or item.args[0] is not call
            or type(item.args[1]) is not int
            or not 0 <= item.args[1] < len(outputs)
        ):
            return None
        index = cast("int", item.args[1])
        for phi in item.users:
            if phi.target is not _tracing_ops._phi:
                continue
            if len(phi.args) != 2 or phi.args[1] is not item:
                return None
            matches = [i for i, entry in enumerate(captures) if entry is phi.args[0]]
            if len(matches) != 1:
                return None
            slot = matches[0]
            if index in slots and slots[index] != slot:
                return None
            slots[index] = slot
    if set(slots) != set(range(len(outputs))) or len(set(slots.values())) != len(slots):
        return None
    if slots != _loop_carry_slots(call):
        return None
    return slots


def _readonly_index_recipe(
    node: Node, known: set[Node], *, identity_captures: frozenset[Node] = frozenset()
) -> bool:
    """Only closed/iota-rooted pure values can inherit a captured interval.

    In particular, full() can denote mutable CTA storage and host/local loads
    can observe another epoch. Their entry interval is never forwarded.
    Explicit clamps/remainders at the actual use retain their usual proof.
    """
    pending = [node]
    seen = set()
    while pending:
        value = pending.pop()
        if value in seen or value in known:
            continue
        seen.add(value)
        if value.target in (
            torch.ops.prims.iota.default,
            torch.ops.aten.scalar_tensor.default,
        ):
            continue
        # A new nested tree has separately rejected all writes/aliases across
        # its entry and body. The emitter's identity capture wrapper may then
        # forward an already proved readonly recipe, never a mutable carry's
        # initial range. Preserve the old flat-loop rule otherwise.
        if value in identity_captures:
            pending.append(cast("Node", value.args[0]))
            continue
        if not (
            isinstance(value.target, torch._ops.OpOverload)
            and not value.target._schema.is_mutable
            and (
                torch.Tag.pointwise in value.target.tags
                or value.target
                in (
                    torch.ops.aten.alias.default,
                    torch.ops.aten.clone.default,
                    torch.ops.aten.unsqueeze.default,
                    torch.ops.aten.squeeze.dim,
                    torch.ops.aten.expand.default,
                    torch.ops.aten.permute.default,
                    torch.ops.aten.view.default,
                    torch.ops.aten.reshape.default,
                )
            )
        ):
            return False
        pending.extend(value.all_input_nodes)
    return True


def loop_domain_facts(
    env: CompileEnvironment, graphs: Sequence[GraphInfo], *, allow_unbound: bool = False
) -> GatherDomainFacts:
    """Publish a loop's facts only after all sibling backedges agree.

    Current call identity, argument order, logical axes and dtype are checked
    together. Nested for trees are transactional: provisional child facts never
    escape a failed enclosing recurrence. Facts affect admission only; the
    existing loop snapshots, storage dependencies and barriers are retained.
    """
    from .bounded_gather import integer_bounds
    from .computed_fragment import _fragment_logical_shape

    facts = GatherDomainFacts()

    def shape(
        value: Node, overrides: dict[Node, tuple[sympy.Expr, ...]]
    ) -> tuple[sympy.Expr, ...] | None:
        fake = value.meta.get("val")
        if not isinstance(fake, torch.Tensor):
            return None
        if value.target is _tracing_ops._host_tensor:
            return tuple(
                env.specialize_expr(
                    cast(
                        "sympy.Expr",
                        size._sympy_()
                        if isinstance(size, torch.SymInt)
                        else sympy.Integer(size),
                    )
                )
                for size in fake.shape
            )
        return _fragment_logical_shape(
            env,
            value,
            scalar_indexed_loads=True,
            tensor_indexed_loads=True,
            pure_inline_asm=True,
            all_axis_reductions=True,
            proven_domains=overrides,
        )

    def compatible(a: Node, b: Node) -> bool:
        left, right = a.meta.get("val"), b.meta.get("val")
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.ndim == right.ndim
            and left.dtype == right.dtype
            # The unchanged carry copier requires the same physical shape as
            # well as the independently proved logical domain.
            and all(starmap(env.known_equal, zip(left.shape, right.shape, strict=True)))
        )

    loops = (_tracing_ops._for_loop, _tracing_ops._for_loop_step)
    calls = [
        node for info in graphs for node in info.graph.nodes if node.target in loops
    ]
    dynamic_control = (_tracing_ops._if, _tracing_ops._while_loop)

    def body_for(call: Node) -> ForLoopGraphInfo | None:
        if call.target not in loops or len(call.args) < 4:
            return None
        graph_id = call.args[0]
        if type(graph_id) is not int or not 0 <= graph_id < len(graphs):
            return None
        body = graphs[graph_id]
        if not isinstance(body, ForLoopGraphInfo) or body.graph_id != graph_id:
            return None
        if sum(info.graph_id == graph_id for info in graphs) != 1:
            return None
        if sum(info.graph is body.graph for info in graphs) != 1:
            return None
        if sum(bool(other.args) and other.args[0] == graph_id for other in calls) != 1:
            return None
        # Even a differently typed control edge cannot reuse these Node keys
        # as another call context. The selected graph list is authoritative.
        for info in graphs:
            for node in info.graph.nodes:
                if node.target is _tracing_ops._if and graph_id in node.args[1:3]:
                    return None
                if (
                    node.target is _tracing_ops._while_loop
                    and graph_id in node.args[:2]
                ):
                    return None
        return body

    def effect_free(nodes: Sequence[Node], active: frozenset[int]) -> bool:
        defined: set[Node] = set()
        for node in nodes:
            if any(
                arg.graph is not node.graph or arg not in defined
                for arg in node.all_input_nodes
            ):
                return False
            defined.add(node)
            if node.target in dynamic_control:
                return False
            if node.target in loops:
                child = body_for(node)
                if child is None or child.graph_id in active:
                    return False
                if not effect_free(list(child.graph.nodes), active | {child.graph_id}):
                    return False
            elif node.target is inline_asm_ops.inline_asm_elementwise:
                # Normalized assembly has six positional arguments. Match the
                # existing single-output pack=1 shape rule, not source kwargs.
                # Logical broadcast domains still require their separate proof.
                output = node.meta.get("val")
                if (
                    node.kwargs
                    or len(node.args) != 6
                    or node.args[4] is not True
                    or type(node.args[5]) is not int
                    or node.args[5] != 1
                    or not isinstance(node.args[3], torch.dtype)
                    or not isinstance(output, torch.Tensor)
                    or output.dtype != node.args[3]
                    or not isinstance(node.args[2], (list, tuple))
                    or not node.args[2]
                    or any(
                        not isinstance(operand, Node)
                        or not isinstance(operand.meta.get("val"), torch.Tensor)
                        for operand in node.args[2]
                    )
                ):
                    return False
            elif node.op not in ("placeholder", "output") and node.target not in (
                _tracing_ops._phi,
                _tracing_ops._new_var,
                _tracing_ops._host_tensor,
            ):
                if node.is_impure():
                    return False
        return True

    def prove(
        call: Node,
        incoming: GatherDomainFacts,
        active: frozenset[int] = frozenset(),
        nested: bool = False,
    ) -> GatherDomainFacts | None:
        body = body_for(call)
        if body is None or body.graph_id in active:
            return None
        nodes = list(body.graph.nodes)
        children = [node for node in nodes if node.target in loops]
        if any(node.target in dynamic_control for node in nodes):
            return None
        new_tree = nested or bool(children)
        captures = call.args[3]
        if not isinstance(captures, (list, tuple)) or not all(
            isinstance(n, Node) for n in captures
        ):
            return None
        captures = list(cast("Sequence[Node]", captures))
        identity_captures: frozenset[Node] = frozenset()
        if new_tree:
            parent_nodes = list(call.graph.nodes)
            preceding = parent_nodes[: parent_nodes.index(call)]
            # Actual entry definitions must dominate the call. Exclude writes
            # even through distinct aliases: this new scope makes no storage
            # mutation proof, and cannot inherit a pre-mutation index interval.
            if any(
                entry.graph is not call.graph or entry not in preceding
                for entry in captures
            ):
                return None
            if any(
                item.graph is not call.graph
                or any(
                    phi.graph is not call.graph
                    for phi in item.users
                    if phi.target is _tracing_ops._phi
                )
                for item in call.users
            ):
                return None
            if not effect_free(preceding, active) or not effect_free(
                nodes, active | {body.graph_id}
            ):
                return None
            positions = {node: i for i, node in enumerate(preceding)}
            identity_captures = frozenset(
                node
                for node in preceding
                if node.target is _tracing_ops._new_var
                and len(node.args) == 1
                and not node.kwargs
                and isinstance(node.args[0], Node)
                and node.args[0] in positions
                and positions[node.args[0]] < positions[node]
                and compatible(node, node.args[0])
            )
        placeholders = list(body.graph.find_nodes(op="placeholder"))
        # GraphInfo.copy retains original node_args. Only current call edges
        # bind current placeholders; metadata can check arity, not substitute
        # stale outer values for the current entry.
        if len(placeholders) != len(captures) or len(body.node_args) != len(captures):
            return None
        output_nodes = list(body.graph.find_nodes(op="output"))
        if len(output_nodes) != 1:
            return None
        outputs = output_nodes[0].args[0]
        if not isinstance(outputs, (list, tuple)) or not all(
            isinstance(n, Node) for n in outputs
        ):
            return None
        outputs = list(cast("Sequence[Node]", outputs))
        slots = _carry_map(call, captures, outputs)
        if slots is None:
            return None
        trial = GatherDomainFacts(dict(incoming.shapes), dict(incoming.readonly_ranges))
        local: dict[Node, tuple[sympy.Expr, ...]] = {}
        for placeholder, entry in zip(placeholders, captures, strict=True):
            entry_shape = shape(entry, incoming.shapes)
            if entry_shape is not None and compatible(placeholder, entry):
                trial.shapes[placeholder] = local[placeholder] = entry_shape
            elif new_tree and isinstance(entry.meta.get("val"), torch.Tensor):
                # Unknown readonly captures cannot make a nested context look
                # complete merely because its mutable outputs happen to match.
                return None
        for slot, (placeholder, entry) in enumerate(
            zip(placeholders, captures, strict=True)
        ):
            if slot in slots.values() or placeholder not in local:
                continue
            if not _readonly_index_recipe(
                entry,
                set(incoming.readonly_ranges),
                identity_captures=identity_captures,
            ):
                continue
            interval = integer_bounds(entry, captured_bounds=incoming.readonly_ranges)
            if interval is not None:
                trial.readonly_ranges[placeholder] = interval
        # Descendants may assume this iteration's entry domains, but nothing
        # escapes the trial until every enclosing output proves the recurrence.
        for child in children:
            child_trial = prove(child, trial, active | {body.graph_id}, True)
            if child_trial is None:
                return None
            trial = child_trial
        for index, output in enumerate(outputs):
            slot = slots[index]
            expected = local.get(placeholders[slot])
            actual = shape(output, trial.shapes)
            if (
                expected is None
                or actual is None
                or not compatible(output, captures[slot])
                or len(actual) != len(expected)
                or any(
                    sympy.expand(a) != sympy.expand(b)
                    for a, b in zip(actual, expected, strict=True)
                )
            ):
                return None
        for item in call.users:
            index = cast("int", item.args[1])
            entry = captures[slots[index]]
            if not compatible(item, entry):
                return None
            trial.shapes[item] = local[placeholders[slots[index]]]
            for phi in item.users:
                if phi.target is _tracing_ops._phi:
                    if not compatible(phi, entry):
                        return None
                    trial.shapes[phi] = trial.shapes[item]
        return trial

    for root in graphs:
        if isinstance(root, RootGraphInfo):
            for call in root.graph.nodes:
                if call.target in loops:
                    result = prove(call, facts)
                    if result is not None:
                        facts = result
    from .resident_while import resident_while_domain_facts

    facts = resident_while_domain_facts(env, graphs, facts, allow_unbound=allow_unbound)
    calls = {
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is _tracing_ops._while_loop
    }
    if calls and calls <= facts.resident_whiles:
        return facts
    from .uniform_region_tree import uniform_region_domains

    return uniform_region_domains(env, graphs, facts, allow_unbound=allow_unbound)
