"""Original typed expression candidates invariant within vector ownership.

These records prove expression/coordinate equivalence, not shared-memory
lifetime or publication. A caller must bind the returned complete read set to
its actual stable owner interval before inserting any returned statements.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

import torch

from ...language import _tracing_ops
from ...language import creation_ops
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseLowering
from . import chained_matmul as chain
from .chained_register_islands import _freeze

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_vector_ownership import VectorOwnership


ExpressionKey = tuple["Node", tuple[str, ...]]


@dataclass(frozen=True)
class BroadcastRead:
    node: Node
    coordinates: tuple[str, ...]
    tensor: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class BroadcastValue:
    key: ExpressionKey
    shape: tuple[int, ...]
    dtype: torch.dtype
    placement: Literal["before_steps", "per_step"]
    value_ir: str
    reads: tuple[BroadcastRead, ...]
    ancestors: tuple[Node, ...]


def _lowering_facts(node: Node) -> object:
    lowering = node.meta.get("lowering")
    if not isinstance(lowering, PointwiseLowering):
        return lowering
    data = lowering.buffer.data
    return (
        lowering,
        tuple(lowering.input_names),
        lowering.buffer,
        data,
        data.dtype,
        data.device,
        tuple(data.ranges),
        data.inner_fn,
    )


def _facts(nodes: tuple[Node, ...]) -> tuple[object, ...]:
    return tuple(
        (
            node,
            node.op,
            node.target,
            _freeze(node.args),
            _freeze(node.kwargs),
            _lowering_facts(node),
            chain._shape(node),
            (value.dtype, _freeze(value.stride()))
            if isinstance(value := node.meta.get("val"), torch.Tensor)
            else _freeze(value),
            tuple(node.users),
        )
        for node in nodes
    )


@dataclass(frozen=True)
class BroadcastExpressionPlan:
    plan: ChainedMatmulPlan
    ownership: VectorOwnership
    row: str
    base: str
    element: str
    values: tuple[BroadcastValue, ...]
    boundaries: tuple[tuple[Node, str], ...]
    nodes: tuple[Node, ...]
    facts: tuple[object, ...]
    fast_math: bool
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            self.plan,
            tuple(vars(self.ownership).items()),
            self.row,
            self.base,
            self.element,
            tuple(
                (
                    value.key,
                    value.shape,
                    value.dtype,
                    value.placement,
                    value.value_ir,
                    tuple(
                        (
                            read.node,
                            read.coordinates,
                            read.tensor,
                            read.shape,
                            read.dtype,
                        )
                        for read in value.reads
                    ),
                    value.ancestors,
                )
                for value in self.values
            ),
            self.boundaries,
            self.nodes,
            self.facts,
            self.fast_math,
        )

    def matches(self, plan: ChainedMatmulPlan, boundaries: Mapping[Node, str]) -> bool:
        return (
            self._selection == self._fields()
            and self.plan is plan
            and self.ownership.matches(self.ownership.shape, self.ownership.threads)
            and all(boundaries.get(node) == tensor for node, tensor in self.boundaries)
            and self.facts == _facts(self.nodes)
            and self.fast_math is CompileEnvironment.current().settings.fast_math
        )


@dataclass(frozen=True)
class BroadcastExpressionEmission:
    """Non-authorizing statements and reads for an independently bound scope."""

    candidate: BroadcastExpressionPlan
    before_steps: tuple[str, ...]
    per_step: tuple[str, ...]
    replacements: tuple[tuple[ExpressionKey, str], ...]
    reads: tuple[BroadcastRead, ...]


def _value_ir(
    expression: chain._Expression, value: str
) -> tuple[str, frozenset[str], frozenset[str]]:
    """Expand structured scalar lowering definitions, never a generated kernel."""
    active: set[str] = set()

    class Expand(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id not in expression.definitions:
                return node
            if node.id in active:
                raise chain._UnsupportedChain("cyclic scalar expression")
            definition = ast.parse(expression.definitions[node.id], mode="eval").body
            if chain._names(definition) != expression.definition_inputs[node.id]:
                raise chain._UnsupportedChain("changed scalar dependency facts")
            active.add(node.id)
            result = self.visit(definition)
            active.remove(node.id)
            return result

    tree = Expand().visit(ast.parse(value, mode="eval").body)
    return (
        ast.dump(tree),
        chain._names(tree),
        frozenset(ast.dump(part) for part in ast.walk(tree)),
    )


def _pure_ancestry(
    node: Node, expression: chain._Expression
) -> tuple[tuple[Node, ...], frozenset[Node]] | None:
    pending, visited, reads = [node], set(), set()
    arithmetic = False
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in expression.fragments:
            return None
        if current in expression.boundaries:
            reads.add(current)
            continue
        if current.op != "call_function":
            return None
        if current.target in (
            *chain._VIEWS,
            _tracing_ops._mask_to,
            torch.ops.aten.scalar_tensor.default,
            creation_ops.full,
        ):
            pending.extend(current.all_input_nodes)
            continue
        if (
            not isinstance(current.target, torch._ops.OpOverload)
            or torch.Tag.pointwise not in current.target.tags
            or current.target._schema.is_mutable
            or torch.Tag.nondeterministic_seeded in current.target.tags
            or torch.Tag.nondeterministic_bitwise in current.target.tags
            or chain._pointwise_inputs(current) is None
        ):
            return None
        arithmetic = True
        pending.extend(current.all_input_nodes)
    if not reads or not arithmetic:
        return None
    # Original graph order supplies deterministic ties without spelling tests.
    return tuple(item for item in node.graph.nodes if item in visited), frozenset(reads)


def plan_broadcast_expressions(
    probes: Sequence[chain._Expression],
    ownership: VectorOwnership,
    *,
    row: str,
    base: str,
    element: str,
) -> BroadcastExpressionPlan | None:
    """Find maximal original pure subtrees independent of the element loop.

    Probes must be fresh, single-output ordinary expression traces. Admission
    of the enclosing producer and physical owner lifetime remain external.
    Initial coverage is a complete physical rectangle with no partial row tile.
    A varying memo coordinate alone is not a rejection: original view lowering
    can drop it, which is checked through the actual value dependency IR.
    """
    if (
        not probes
        or len({row, base, element}) != 3
        or not ownership.matches(ownership.shape, ownership.threads)
        or ownership.shape[0] % ownership.thread_rows
    ):
        return None
    plan = probes[0].plan
    candidates: dict[ExpressionKey, BroadcastValue] = {}
    bindings: dict[Node, str] = {}
    all_nodes: set[Node] = set()
    try:
        for probe in probes:
            if probe.plan is not plan or not probe.memo:
                return None
            output, output_coordinates = next(reversed(probe.memo))
            if any(node.graph is not plan.dots[0].graph for node, _ in probe.memo):
                return None
            # The complete original probe also owns the enclosing evaluation
            # domain. A mutation outside a retained subtree can remove its
            # original evaluation, so snapshot all traced nodes, not just reads.
            all_nodes.update(node for node, _ in probe.memo)
            expected_axes = {
                ast.dump(ast.parse(row, mode="eval").body): ownership.shape[0],
                ast.dump(
                    ast.parse(f"({base} + {element})", mode="eval").body
                ): ownership.shape[1],
            }
            if (
                len(output_coordinates) != 2
                or len(chain._shape(output)) != 2
                or {
                    ast.dump(ast.parse(coordinate, mode="eval").body)
                    for coordinate in output_coordinates
                }
                != set(expected_axes)
                or any(
                    expected_axes.get(ast.dump(ast.parse(coordinate, mode="eval").body))
                    != extent
                    for coordinate, extent in zip(
                        output_coordinates, chain._shape(output), strict=True
                    )
                )
            ):
                return None
            for key, value in probe.memo.items():
                node, coordinates = key
                if (
                    node in probe.boundaries
                    or node in probe.fragments
                    or node.meta["val"].dtype
                    not in (torch.bfloat16, torch.float16, torch.float32)
                ):
                    continue
                ancestry = _pure_ancestry(node, probe)
                if ancestry is None:
                    continue
                ancestors, inputs = ancestry
                ir, names, subtrees = _value_ir(probe, value)
                if element in names:
                    continue
                reads = tuple(
                    BroadcastRead(
                        source,
                        coords,
                        probe.boundaries[source],
                        chain._shape(source),
                        source.meta["val"].dtype,
                    )
                    for source, coords in probe.memo
                    if source in inputs
                    and probe.boundaries[source] in names
                    and _value_ir(probe, probe.memo[source, coords])[0] in subtrees
                )
                allowed = {row, base, "cutlass", "cute", "operator"}
                allowed.update(item.tensor for item in reads)
                if not reads or names - allowed:
                    continue
                placement = (
                    "per_step"
                    if base in names or row in names and ownership.row_tiles != 1
                    else "before_steps"
                )
                entry = BroadcastValue(
                    key,
                    chain._shape(node),
                    node.meta["val"].dtype,
                    placement,
                    ir,
                    reads,
                    ancestors,
                )
                if key in candidates and candidates[key] != entry:
                    return None
                candidates[key] = entry
                for item in reads:
                    if item.node in bindings and bindings[item.node] != item.tensor:
                        return None
                    bindings[item.node] = item.tensor
                all_nodes.update(ancestors)
        selected = tuple(
            entry
            for key, entry in candidates.items()
            if not any(
                key != other_key
                and key[0] is not other_key[0]
                and key[0] in other.ancestors
                and entry.placement == other.placement
                for other_key, other in candidates.items()
            )
        )
        if not selected:
            return None
        nodes = tuple(node for node in plan.dots[0].graph.nodes if node in all_nodes)
        facts = _facts(nodes)
    except chain._UnsupportedChain:
        return None
    fields = (
        plan,
        ownership,
        row,
        base,
        element,
        selected,
        tuple(bindings.items()),
        nodes,
        facts,
        CompileEnvironment.current().settings.fast_math,
    )
    result = BroadcastExpressionPlan(*fields, ())
    object.__setattr__(result, "_selection", result._fields())
    return result


def _project(text: str, substitutions: dict[str, str]) -> str:
    class Project(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            return (
                ast.parse(substitutions[node.id], mode="eval").body
                if node.id in substitutions
                else node
            )

    return ast.unparse(Project().visit(ast.parse(text, mode="eval").body))


def emit_broadcast_expressions(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    candidate: BroadcastExpressionPlan,
    *,
    tag: str,
    execution: ChainedExecution,
) -> BroadcastExpressionEmission | None:
    """Lower original nodes once; caller still owes a complete stable-read proof."""
    if (
        not candidate.matches(plan, boundaries)
        or execution.threads != candidate.ownership.threads
    ):
        return None
    before, per_step, replacements, reads = [], [], [], []
    stable_row = f"{tag}_row"
    try:
        for entry in candidate.values:
            substitutions = {candidate.element: "0"}
            if entry.placement == "before_steps":
                substitutions[candidate.row] = stable_row
                substitutions[candidate.base] = "0"
            coords = tuple(_project(coord, substitutions) for coord in entry.key[1])
            expression = chain._Expression(cg, plan, boundaries)
            expression.coordinate_names.update(
                (stable_row, candidate.row, candidate.base)
            )
            value = expression.value(entry.key[0], coords)
            # Compare original structured lowering again after reversing only
            # the proven invariant-row rename. No arithmetic normalization.
            renamed = _project(value, {stable_row: candidate.row})
            expression.definitions = {
                name: _project(code, {stable_row: candidate.row})
                for name, code in expression.definitions.items()
            }
            expression.definition_inputs = {
                name: chain._names(ast.parse(code, mode="eval"))
                for name, code in expression.definitions.items()
            }
            if _value_ir(expression, renamed)[0] != entry.value_ir:
                return None
            # Keep original statements only in the live value dependency slice.
            live = set(chain._names(ast.parse(value, mode="eval")))
            lines = []
            for statement in reversed(expression.statements):
                if statement.target is None:
                    return None
                if statement.target in live:
                    lines.append(statement.code)
                    live.update(statement.inputs)
            lines.reverse()
            name = cg.device_function.new_var("chain_broadcast")
            lines.append(f"{name} = {value}")
            (before if entry.placement == "before_steps" else per_step).extend(lines)
            replacements.append((entry.key, name))
            reads.extend(item for item in entry.reads if item not in reads)
    except chain._UnsupportedChain:
        return None
    if before:
        before.insert(
            0,
            f"{stable_row} = {candidate.ownership.row_expression(execution.thread, '0')}",
        )
    return BroadcastExpressionEmission(
        candidate, tuple(before), tuple(per_step), tuple(replacements), tuple(reads)
    )
