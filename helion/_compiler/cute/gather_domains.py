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


def _readonly_index_recipe(node: Node, known: set[Node]) -> bool:
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
    env: CompileEnvironment, graphs: Sequence[GraphInfo]
) -> GatherDomainFacts:
    """Publish a loop's facts only after all sibling backedges agree.

    Current call identity, argument order, logical axes and dtype are checked
    together. Nested/ambiguous graph calls decline. Facts affect admission only;
    the existing loop snapshots, storage dependencies and barriers are retained.
    """
    from .bounded_gather import integer_bounds
    from .computed_fragment import _selection_logical_shape

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
        return _selection_logical_shape(
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
    for root in graphs:
        if not isinstance(root, RootGraphInfo):
            continue
        for call in root.graph.nodes:
            if call.target not in loops or len(call.args) < 4:
                continue
            matches = [info for info in graphs if info.graph_id == call.args[0]]
            if len(matches) != 1 or not isinstance(matches[0], ForLoopGraphInfo):
                continue
            body = matches[0]
            graph_id = call.args[0]
            if (
                type(graph_id) is not int
                or not 0 <= graph_id < len(graphs)
                or graphs[graph_id] is not body
            ):
                continue
            if sum(other.args[0] == body.graph_id for other in calls) != 1:
                continue
            if any(
                node.target in loops or node.target is _tracing_ops._if
                for node in body.graph.nodes
            ):
                continue
            captures = call.args[3]
            if not isinstance(captures, (list, tuple)) or not all(
                isinstance(n, Node) for n in captures
            ):
                continue
            captures = list(cast("Sequence[Node]", captures))
            placeholders = list(body.graph.find_nodes(op="placeholder"))
            # GraphInfo.copy retains original node_args. The emitter binds the
            # current call's args[3] to current placeholders by strict position;
            # stale outer objects must never supply facts for this call.
            if len(placeholders) != len(captures) or len(body.node_args) != len(
                captures
            ):
                continue
            output_nodes = list(body.graph.find_nodes(op="output"))
            if len(output_nodes) != 1:
                continue
            outputs = output_nodes[0].args[0]
            if not isinstance(outputs, (list, tuple)) or not all(
                isinstance(n, Node) for n in outputs
            ):
                continue
            outputs = list(cast("Sequence[Node]", outputs))
            slots = _carry_map(call, captures, outputs)
            if slots is None:
                continue
            provisional = dict(facts.shapes)
            local: dict[Node, tuple[sympy.Expr, ...]] = {}
            for placeholder, entry in zip(placeholders, captures, strict=True):
                entry_shape = shape(entry, facts.shapes)
                if entry_shape is not None and compatible(placeholder, entry):
                    provisional[placeholder] = local[placeholder] = entry_shape
            valid = True
            for index, output in enumerate(outputs):
                slot = slots[index]
                expected = local.get(placeholders[slot])
                actual = shape(output, provisional)
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
                    valid = False
                    break
            if not valid:
                continue
            outgoing: dict[Node, tuple[sympy.Expr, ...]] = {}
            for item in call.users:
                index = cast("int", item.args[1])
                entry = captures[slots[index]]
                if not compatible(item, entry):
                    valid = False
                    break
                outgoing[item] = local[placeholders[slots[index]]]
                for phi in item.users:
                    if phi.target is _tracing_ops._phi:
                        if not compatible(phi, entry):
                            valid = False
                            break
                        outgoing[phi] = outgoing[item]
            if not valid:
                continue
            # Commit all domains together. Mutable carries never inherit ranges.
            facts.shapes.update(local)
            facts.shapes.update(outgoing)
            for slot, (placeholder, entry) in enumerate(
                zip(placeholders, captures, strict=True)
            ):
                if slot in slots.values() or placeholder not in local:
                    continue
                if not _readonly_index_recipe(entry, set(facts.readonly_ranges)):
                    continue
                interval = integer_bounds(entry, captured_bounds=facts.readonly_ranges)
                if interval is not None:
                    facts.readonly_ranges[placeholder] = interval
    return facts
