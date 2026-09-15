# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture and coordinate primitives for transactional affine replay."""

from __future__ import annotations

import ast
import dataclasses
import enum
import keyword
from typing import TYPE_CHECKING
from typing import AbstractSet
from typing import Protocol
from typing import TypeVar
from typing import cast

import torch  # noqa: TC002

from ..ast_extension import ExtendedAST
from .direct_affine_candidate import DirectAffineCandidate
from .direct_affine_candidate import revalidate_direct_affine_candidate
from .direct_affine_plan import STATE_VECTOR_BYTES
from .direct_affine_plan import DirectAffinePlan

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..device_ir import GraphInfo


_A = TypeVar("_A", bound=ast.AST)
_INTEGER_CASTS = frozenset(
    {
        "cutlass.Int32",
        "cutlass.Int64",
        "cutlass.Uint32",
        "cutlass.Uint64",
    }
)


class _ReplayCodegen(Protocol):
    _statements_by_owner_node_id: dict[int, list[tuple[list[ast.AST], ast.AST]]]
    referenced_thread_block_dims: list[int]

    def statements_owned_by_node(
        self, node: torch.fx.Node
    ) -> tuple[tuple[list[ast.AST], ast.AST], ...]: ...

    def codegen_result_for_node(self, node: torch.fx.Node) -> tuple[bool, object]: ...

    def replace_owned_statement_span(
        self,
        body: list[ast.AST],
        nodes: tuple[torch.fx.Node, ...],
        replacement: tuple[ast.AST, ...],
    ) -> bool: ...


def _clone_ast(node: _A) -> _A:
    """Clone generated AST while retaining ``ExtendedAST`` metadata."""

    def clone(value: object) -> object:
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, ast.AST):
            fields = {field: clone(getattr(value, field)) for field in value._fields}
            result = (
                value.copy(**fields)
                if isinstance(value, ExtendedAST)
                else ast.copy_location(type(value)(**fields), value)
            )
            for name, metadata in vars(value).items():
                if name not in value._fields:
                    setattr(result, name, metadata)
            return result
        return value

    result = clone(node)
    assert isinstance(result, ast.AST)
    return cast("_A", result)


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _exact_dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=True)


def _ast_identity_set(trees: Iterable[ast.AST]) -> set[int]:
    singleton_types = (
        ast.boolop,
        ast.cmpop,
        ast.expr_context,
        ast.operator,
        ast.unaryop,
    )
    return {
        id(node)
        for tree in trees
        for node in ast.walk(tree)
        if not isinstance(node, singleton_types)
    }


def _qualified_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _ordered_unique_nodes(
    nodes: Iterable[torch.fx.Node],
    source_interval: Sequence[torch.fx.Node],
) -> tuple[torch.fx.Node, ...] | None:
    requested_tuple = tuple(nodes)
    requested = set(requested_tuple)
    if len(requested) != len(requested_tuple):
        return None
    ordered = tuple(node for node in source_interval if node in requested)
    return ordered if len(ordered) == len(requested) else None


@dataclasses.dataclass(frozen=True)
class DirectAffineNodeReplay:
    """Ordinary-lowering output for one node in the source interval."""

    node: torch.fx.Node = dataclasses.field(repr=False)
    statements: tuple[ast.stmt, ...] = dataclasses.field(repr=False)
    result: object = dataclasses.field(repr=False)
    result_fingerprint: object = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass(frozen=True)
class DirectAffineReplay:
    """Immutable witness for one exactly owned, contiguous root span."""

    candidate: DirectAffineCandidate = dataclasses.field(repr=False)
    nodes: tuple[DirectAffineNodeReplay, ...] = dataclasses.field(repr=False)
    statement_nodes: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)
    source_statements: tuple[ast.stmt, ...] = dataclasses.field(repr=False)
    source_span: tuple[int, int]
    fence_nodes: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)
    root_statement_ids: tuple[int, ...] = dataclasses.field(repr=False)
    root_statement_dumps: tuple[str, ...] = dataclasses.field(repr=False)
    reserved_names: frozenset[str] = dataclasses.field(
        default_factory=frozenset, repr=False
    )

    def node_replay(self, node: torch.fx.Node) -> DirectAffineNodeReplay | None:
        return next((item for item in self.nodes if item.node is node), None)


def _owner_index_is_exact(
    codegen: _ReplayCodegen,
    root_body: list[ast.AST],
    claimed_ids: set[int],
) -> bool:
    """Reject a claimed statement indexed under any second FX owner."""

    owner_index = codegen._statements_by_owner_node_id
    root_counts: dict[int, int] = {}
    for entries in owner_index.values():
        if not isinstance(entries, (list, tuple)):
            return False
        for entry in entries:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], list)
                or not isinstance(entry[1], ast.AST)
            ):
                return False
            statement_id = id(entry[1])
            if statement_id in claimed_ids:
                if entry[0] is not root_body:
                    return False
                root_counts[statement_id] = root_counts.get(statement_id, 0) + 1
    return all(root_counts.get(statement_id) == 1 for statement_id in claimed_ids)


def _result_fingerprint(value: object) -> object:
    """Take an immutable structural snapshot of a recorded codegen result."""

    if isinstance(value, ast.AST):
        return ("ast", _exact_dump(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_result_fingerprint(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_result_fingerprint(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        repr(key),
                        _result_fingerprint(item),
                    )
                    for key, item in value.items()
                )
            ),
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value).__qualname__, value)
    return (type(value).__module__, type(value).__qualname__, id(value), repr(value))


def _snapshot_result(value: object) -> object:
    if isinstance(value, ast.AST):
        return _clone_ast(value)
    if isinstance(value, tuple):
        return tuple(_snapshot_result(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_result(item) for item in value]
    if isinstance(value, dict):
        return {key: _snapshot_result(item) for key, item in value.items()}
    return value


def _result_snapshots_are_intact(replay: DirectAffineReplay) -> bool:
    return all(
        item.result_fingerprint is None
        or _result_fingerprint(item.result) == item.result_fingerprint
        for item in replay.nodes
    )


def _visible_program_names(
    codegen: _ReplayCodegen, root_body: list[ast.AST]
) -> set[str]:
    trees: list[ast.AST] = list(root_body)
    module_statements = getattr(codegen, "module_statements", ())
    if isinstance(module_statements, (list, tuple)):
        trees.extend(
            statement
            for statement in module_statements
            if isinstance(statement, ast.AST)
        )
    grid = getattr(codegen, "current_grid_state", None)
    if grid is not None:
        for attribute in (
            "outer_prefix",
            "lane_setup_statements",
            "hoist_parent_statements",
        ):
            statements = getattr(grid, attribute, ())
            if isinstance(statements, (list, tuple)):
                trees.extend(
                    statement
                    for statement in statements
                    if isinstance(statement, ast.AST)
                )
    names = {
        node.id
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    names.update(
        node.arg
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.arg)
    )
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.alias):
                names.add(node.asname or node.name.partition(".")[0])
    device_function = getattr(codegen, "device_function", None)
    arguments = getattr(device_function, "arguments", ())
    names.update(
        name
        for argument in arguments
        if isinstance((name := getattr(argument, "name", None)), str)
    )
    return names


def _resolve_direct_affine_source_replay(
    candidate: DirectAffineCandidate,
    codegen: _ReplayCodegen,
    root_body: list[ast.AST],
) -> DirectAffineReplay | None:
    """Resolve codegen artifacts after the candidate itself was revalidated."""

    region = candidate.region
    positions = {id(statement): index for index, statement in enumerate(root_body)}
    if len(positions) != len(root_body):
        return None

    replacement_nodes = _ordered_unique_nodes(
        (*region.producer_nodes, *region.owned_nodes),
        region.source_interval,
    )
    if replacement_nodes is None:
        return None

    node_replays: list[DirectAffineNodeReplay] = []
    statement_nodes: list[torch.fx.Node] = []
    statement_owner: dict[int, torch.fx.Node] = {}
    claimed: list[ast.stmt] = []
    replacement_set = set(replacement_nodes)
    for node in region.source_interval:
        if node.op != "call_function":
            continue
        found, result = codegen.codegen_result_for_node(node)
        if not found:
            return None
        entries = codegen.statements_owned_by_node(node)
        statements: list[ast.stmt] = []
        for owner_body, statement in entries:
            if owner_body is not root_body or not isinstance(statement, ast.stmt):
                return None
            if id(statement) not in positions:
                return None
            previous_owner = statement_owner.setdefault(id(statement), node)
            if previous_owner is not node:
                return None
            statements.append(statement)
        if len({id(statement) for statement in statements}) != len(statements):
            return None
        node_replays.append(
            DirectAffineNodeReplay(
                node=node,
                statements=tuple(statements),
                result=_snapshot_result(result),
                result_fingerprint=_result_fingerprint(result),
            )
        )
        if node in replacement_set and statements:
            statement_nodes.append(node)
            claimed.extend(statements)

    if not claimed or len({id(statement) for statement in claimed}) != len(claimed):
        return None
    claimed_ids = {id(statement) for statement in claimed}
    if not _owner_index_is_exact(codegen, root_body, claimed_ids):
        return None
    first = min(positions[statement_id] for statement_id in claimed_ids)
    last = max(positions[statement_id] for statement_id in claimed_ids)
    source_statements = tuple(root_body[first : last + 1])
    if (
        not all(isinstance(statement, ast.stmt) for statement in source_statements)
        or {id(statement) for statement in source_statements} != claimed_ids
    ):
        return None

    fence_nodes: list[torch.fx.Node] = []
    for fence in region.access_fences:
        for access, before in ((fence.preceding, True), (fence.following, False)):
            if access is None:
                continue
            entries = codegen.statements_owned_by_node(access.node)
            if not entries:
                return None
            fence_positions: list[int] = []
            for owner_body, statement in entries:
                if owner_body is not root_body or id(statement) not in positions:
                    return None
                fence_positions.append(positions[id(statement)])
            if (before and max(fence_positions) >= first) or (
                not before and min(fence_positions) <= last
            ):
                return None
            fence_nodes.append(access.node)

    return DirectAffineReplay(
        candidate=candidate,
        nodes=tuple(node_replays),
        statement_nodes=tuple(statement_nodes),
        source_statements=cast("tuple[ast.stmt, ...]", source_statements),
        source_span=(first, last),
        fence_nodes=tuple(dict.fromkeys(fence_nodes)),
        root_statement_ids=tuple(id(statement) for statement in root_body),
        root_statement_dumps=tuple(_exact_dump(statement) for statement in root_body),
        reserved_names=frozenset(_visible_program_names(codegen, root_body)),
    )


def resolve_direct_affine_replay(
    candidate: DirectAffineCandidate,
    graph_info: GraphInfo,
    codegen: _ReplayCodegen,
    root_body: list[ast.AST],
) -> DirectAffineReplay | None:
    """Revalidate a candidate and snapshot its exact ordinary root lowering.

    This function is read-only.  In particular, a failed ownership, result, or
    fence check leaves both the AST and GenerateAST owner index untouched.
    """

    if revalidate_direct_affine_candidate(candidate, graph_info) is None:
        return None
    return _resolve_direct_affine_source_replay(candidate, codegen, root_body)


def _same_replay(left: DirectAffineReplay, right: DirectAffineReplay) -> bool:
    return (
        _result_snapshots_are_intact(left)
        and _result_snapshots_are_intact(right)
        and left.candidate == right.candidate
        and tuple(item.node for item in left.nodes)
        == tuple(item.node for item in right.nodes)
        and tuple(
            tuple(id(statement) for statement in item.statements) for item in left.nodes
        )
        == tuple(
            tuple(id(statement) for statement in item.statements)
            for item in right.nodes
        )
        and tuple(item.result_fingerprint for item in left.nodes)
        == tuple(item.result_fingerprint for item in right.nodes)
        and left.statement_nodes == right.statement_nodes
        and left.source_span == right.source_span
        and tuple(id(statement) for statement in left.source_statements)
        == tuple(id(statement) for statement in right.source_statements)
        and left.fence_nodes == right.fence_nodes
        and left.root_statement_ids == right.root_statement_ids
        and left.root_statement_dumps == right.root_statement_dumps
        and right.reserved_names.issubset(left.reserved_names)
    )


def replace_direct_affine_replay(
    replay: DirectAffineReplay,
    graph_info: GraphInfo,
    codegen: _ReplayCodegen,
    root_body: list[ast.AST],
    replacement: tuple[ast.stmt, ...],
) -> bool:
    """Atomically install a detached replacement after full late revalidation."""

    if (
        not replacement
        or not all(isinstance(statement, ast.stmt) for statement in replacement)
        or _ast_identity_set(replacement).intersection(_ast_identity_set(root_body))
    ):
        return False
    current = resolve_direct_affine_replay(
        replay.candidate,
        graph_info,
        codegen,
        root_body,
    )
    if current is None or not _same_replay(replay, current):
        return False
    body_snapshot = tuple(root_body)
    owner_snapshot = {
        owner: list(entries)
        for owner, entries in codegen._statements_by_owner_node_id.items()
    }
    thread_dims_snapshot = tuple(codegen.referenced_thread_block_dims)
    try:
        replaced = codegen.replace_owned_statement_span(
            root_body,
            replay.statement_nodes,
            cast("tuple[ast.AST, ...]", replacement),
        )
    except Exception:
        replaced = False
    if replaced:
        return True
    root_body[:] = body_snapshot
    codegen._statements_by_owner_node_id.clear()
    codegen._statements_by_owner_node_id.update(owner_snapshot)
    codegen.referenced_thread_block_dims[:] = thread_dims_snapshot
    return False


@dataclasses.dataclass(frozen=True)
class DirectAffineOrdinaryAxis:
    """Witness for one ordinary lowering's complete local coordinate."""

    source: ast.expr = dataclasses.field(repr=False)
    tile_offset: ast.expr = dataclasses.field(repr=False)
    local_expression: ast.expr = dataclasses.field(repr=False)
    extent: int
    thread_axis: int
    thread_extent: int
    lane_name: str | None = None
    lane_extent: int = 1


@dataclasses.dataclass(frozen=True)
class DirectAffineCoordinates:
    """Validated ordinary row/feature coordinates used as replay keys."""

    row: DirectAffineOrdinaryAxis
    feature: DirectAffineOrdinaryAxis


def _thread_axis(node: ast.AST) -> int | None:
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and _qualified_name(node.value.func) == "cute.arch.thread_idx"
        and not node.value.args
        and not node.value.keywords
        and isinstance(node.slice, ast.Constant)
        and type(node.slice.value) is int
    ):
        return None
    return node.slice.value


def _evaluate_coordinate(
    expression: ast.AST,
    *,
    thread_axis: int,
    thread: int,
    lane_name: str | None,
    lane: int,
) -> int | None:
    if isinstance(expression, ast.Constant) and type(expression.value) is int:
        return expression.value
    if isinstance(expression, ast.Name) and isinstance(expression.ctx, ast.Load):
        return lane if expression.id == lane_name else None
    axis = _thread_axis(expression)
    if axis is not None:
        return thread if axis == thread_axis else None
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.USub):
        value = _evaluate_coordinate(
            expression.operand,
            thread_axis=thread_axis,
            thread=thread,
            lane_name=lane_name,
            lane=lane,
        )
        return -value if value is not None else None
    if isinstance(expression, ast.Call):
        if (
            _qualified_name(expression.func) not in _INTEGER_CASTS
            or len(expression.args) != 1
            or expression.keywords
        ):
            return None
        return _evaluate_coordinate(
            expression.args[0],
            thread_axis=thread_axis,
            thread=thread,
            lane_name=lane_name,
            lane=lane,
        )
    if not isinstance(expression, ast.BinOp):
        return None
    left = _evaluate_coordinate(
        expression.left,
        thread_axis=thread_axis,
        thread=thread,
        lane_name=lane_name,
        lane=lane,
    )
    right = _evaluate_coordinate(
        expression.right,
        thread_axis=thread_axis,
        thread=thread,
        lane_name=lane_name,
        lane=lane,
    )
    if left is None or right is None:
        return None
    try:
        if isinstance(expression.op, ast.Add):
            return left + right
        if isinstance(expression.op, ast.Sub):
            return left - right
        if isinstance(expression.op, ast.Mult):
            return left * right
        if isinstance(expression.op, ast.FloorDiv):
            return left // right
        if isinstance(expression.op, ast.Mod):
            return left % right
    except (ArithmeticError, ValueError):
        return None
    return None


def _axis_is_exact(axis: DirectAffineOrdinaryAxis) -> bool:
    if (
        type(axis.extent) is not int
        or type(axis.thread_axis) is not int
        or type(axis.thread_extent) is not int
        or type(axis.lane_extent) is not int
        or axis.extent <= 0
        or not 0 <= axis.thread_axis < 3
        or axis.thread_extent <= 0
        or axis.lane_extent <= 0
        or axis.thread_extent * axis.lane_extent != axis.extent
        or (axis.lane_extent > 1 and axis.lane_name is None)
        or (axis.lane_name is not None and not axis.lane_name.isidentifier())
    ):
        return False
    values = tuple(
        _evaluate_coordinate(
            axis.local_expression,
            thread_axis=axis.thread_axis,
            thread=thread,
            lane_name=axis.lane_name,
            lane=lane,
        )
        for thread in range(axis.thread_extent)
        for lane in range(axis.lane_extent)
    )
    return None not in values and sorted(cast("tuple[int, ...]", values)) == list(
        range(axis.extent)
    )


def resolve_direct_affine_coordinates(
    plan: DirectAffinePlan,
    *,
    row: DirectAffineOrdinaryAxis,
    feature: DirectAffineOrdinaryAxis,
) -> DirectAffineCoordinates | None:
    """Validate complete ordinary axes before changing their thread mapping."""

    try:
        plan.validate()
    except ValueError:
        return None
    if (
        plan.cta_shape[0] != 32
        or plan.cta_shape[1] != max(plan.step_count, plan.row_warps)
        or plan.cta_shape[2] != 1
        or row.extent != plan.row_extent
        or feature.extent != plan.feature_extent
        or row.thread_axis != 1
        or row.thread_extent != plan.row_warps
        or feature.thread_axis != 0
        or feature.thread_extent != plan.cta_shape[0]
        or row.thread_axis == feature.thread_axis
        or _dump(row.source) == _dump(feature.source)
        or not _axis_is_exact(row)
        or not _axis_is_exact(feature)
    ):
        return None
    return DirectAffineCoordinates(row=row, feature=feature)


class DirectAffineCoordinateRole(enum.Enum):
    ROW = "row"
    FEATURE = "feature"
    ROW_LOCAL = "row_local"
    FEATURE_LOCAL = "feature_local"


@dataclasses.dataclass(frozen=True)
class DirectAffineReplayBindings:
    """Structural coordinate keys present in one opaque source template."""

    row: ast.expr | None = dataclasses.field(default=None, repr=False)
    feature: ast.expr | None = dataclasses.field(default=None, repr=False)
    row_local: ast.expr | None = dataclasses.field(default=None, repr=False)
    feature_local: ast.expr | None = dataclasses.field(default=None, repr=False)
    row_alternates: tuple[ast.expr, ...] = dataclasses.field(default=(), repr=False)
    feature_alternates: tuple[ast.expr, ...] = dataclasses.field(default=(), repr=False)
    row_global_alternates: tuple[ast.expr, ...] = dataclasses.field(
        default=(), repr=False
    )
    feature_global_alternates: tuple[ast.expr, ...] = dataclasses.field(
        default=(), repr=False
    )

    def items(self) -> tuple[tuple[DirectAffineCoordinateRole, ast.expr], ...]:
        return tuple(
            (role, value)
            for role, value in (
                (DirectAffineCoordinateRole.ROW, self.row),
                (DirectAffineCoordinateRole.FEATURE, self.feature),
                (DirectAffineCoordinateRole.ROW_LOCAL, self.row_local),
                (DirectAffineCoordinateRole.FEATURE_LOCAL, self.feature_local),
                *(
                    (DirectAffineCoordinateRole.ROW_LOCAL, value)
                    for value in self.row_alternates
                ),
                *(
                    (DirectAffineCoordinateRole.FEATURE_LOCAL, value)
                    for value in self.feature_alternates
                ),
                *(
                    (DirectAffineCoordinateRole.ROW, value)
                    for value in self.row_global_alternates
                ),
                *(
                    (DirectAffineCoordinateRole.FEATURE, value)
                    for value in self.feature_global_alternates
                ),
            )
            if value is not None
        )

    def values(self) -> tuple[ast.expr, ...]:
        return tuple(
            value
            for value in (
                self.row,
                self.feature,
                self.row_local,
                self.feature_local,
                *self.row_alternates,
                *self.feature_alternates,
                *self.row_global_alternates,
                *self.feature_global_alternates,
            )
            if value is not None
        )


@dataclasses.dataclass(frozen=True)
class DirectAffineValueReplay:
    """Straight-line/structured producer slice with one scalar result."""

    statements: tuple[ast.stmt, ...] = dataclasses.field(repr=False)
    value: ast.expr = dataclasses.field(repr=False)
    bindings: DirectAffineReplayBindings = DirectAffineReplayBindings()


@dataclasses.dataclass(frozen=True)
class DirectAffineEffectReplay:
    """Output effect closure whose logical value is replaced after consume."""

    statements: tuple[ast.stmt, ...] = dataclasses.field(repr=False)
    logical_value: ast.expr = dataclasses.field(repr=False)
    bindings: DirectAffineReplayBindings = DirectAffineReplayBindings()


@dataclasses.dataclass(frozen=True)
class DirectAffineStateStoreReplay:
    """Opaque state-address closure for one packed eight-feature store."""

    statements: tuple[ast.stmt, ...] = dataclasses.field(repr=False)
    pointer: ast.expr = dataclasses.field(repr=False)
    slot: ast.expr = dataclasses.field(repr=False)
    slot_extent: ast.expr = dataclasses.field(repr=False)
    valid: ast.expr = dataclasses.field(
        default_factory=lambda: ast.Constant(value=True), repr=False
    )
    bindings: DirectAffineReplayBindings = DirectAffineReplayBindings()


@dataclasses.dataclass(frozen=True)
class DirectAffineAsyncEntryReplay:
    """Exact source pointer, row stride, and guard for async state ingress."""

    statements: tuple[ast.stmt, ...] = dataclasses.field(repr=False)
    pointer: ast.expr = dataclasses.field(repr=False)
    row_stride: ast.expr = dataclasses.field(repr=False)
    valid: ast.expr = dataclasses.field(repr=False)
    bindings: DirectAffineReplayBindings = DirectAffineReplayBindings()


@dataclasses.dataclass(frozen=True)
class DirectAffineStepReplay:
    diagonal: DirectAffineValueReplay
    prediction_vector: DirectAffineValueReplay
    row_input: DirectAffineValueReplay
    update_scale: DirectAffineValueReplay
    update_vector: DirectAffineValueReplay
    observation_vector: DirectAffineValueReplay
    output_effect: DirectAffineEffectReplay
    state_effect: DirectAffineStateStoreReplay
    output_before_state: bool = True


@dataclasses.dataclass(frozen=True)
class DirectAffineResolvedTemplates:
    """Names-agnostic templates derived entirely from a replayed candidate."""

    entry_state: DirectAffineValueReplay
    steps: tuple[DirectAffineStepReplay, ...]
    async_entry_state: DirectAffineAsyncEntryReplay | None = None


@dataclasses.dataclass(frozen=True)
class DirectAffineReplayProof:
    """Late facts required by the scalar-replay/vector-checkpoint composer."""

    state_output_alias_free: bool
    state_feature_stride_one: bool
    state_vector_aligned: bool
    masks_preserved: bool
    row_tail_free: bool
    feature_tail_free: bool
    runtime_state_shape: tuple[int, int]
    runtime_output_row_extents: tuple[int, ...]
    source_replay: DirectAffineReplay | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    state_feature_extent_expression: ast.expr | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    state_row_extent_expression: ast.expr | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    output_store_pre_wrap_proven: bool = False
    state_argument_name: str | None = None
    state_rank: int = 0
    output_argument_names: tuple[str, ...] = ()
    output_ranks: tuple[int, ...] = ()

    def supports(self, plan: DirectAffinePlan) -> bool:
        if len(self.runtime_state_shape) != 2:
            return False
        state_rows, state_features = self.runtime_state_shape
        return (
            self.state_output_alias_free is True
            and self.state_feature_stride_one is True
            and self.state_vector_aligned is True
            and self.masks_preserved is True
            and self.row_tail_free is True
            and self.feature_tail_free is True
            and state_rows >= plan.row_extent
            and state_rows % plan.row_extent == 0
            and state_features == plan.feature_extent
            and len(self.runtime_output_row_extents) == plan.step_count
            and all(
                extent >= plan.row_extent and extent % plan.row_extent == 0
                for extent in self.runtime_output_row_extents
            )
            and plan.state_access_proof.source_vector_alignment_bytes
            % STATE_VECTOR_BYTES
            == 0
            and plan.state_access_proof.destination_vector_alignment_bytes
            % STATE_VECTOR_BYTES
            == 0
        )


@dataclasses.dataclass(frozen=True)
class DirectAffineSharedBuffer:
    role: str
    name: str
    dtype_name: str
    element_count: int
    byte_offset: int


@dataclasses.dataclass(frozen=True)
class DirectAffineEmission:
    """Complete detached root replacement and its integration metadata."""

    module_statements: tuple[ast.stmt, ...]
    replacement_statements: tuple[ast.stmt, ...]


@dataclasses.dataclass(frozen=True)
class _AsyncStateEmission:
    begin: tuple[ast.stmt, ...]
    finish: tuple[ast.stmt, ...]


_BANNED_REPLAY_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.Break,
    ast.ClassDef,
    ast.DictComp,
    ast.Continue,
    ast.FunctionDef,
    ast.GeneratorExp,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.ListComp,
    ast.Match,
    ast.NamedExpr,
    ast.Nonlocal,
    ast.Return,
    ast.SetComp,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


def _template_is_supported(trees: Sequence[ast.AST]) -> bool:
    return all(
        not isinstance(node, _BANNED_REPLAY_NODES)
        for tree in trees
        for node in ast.walk(tree)
    )


class _StructuralSubstitution(ast.NodeTransformer):
    def __init__(self, substitutions: Mapping[str, ast.expr]) -> None:
        self.substitutions = substitutions
        self.counts = dict.fromkeys(substitutions, 0)

    def visit(self, node: ast.AST) -> ast.AST:  # type: ignore[override]
        if isinstance(node, ast.expr) and isinstance(
            getattr(node, "ctx", ast.Load()), ast.Load
        ):
            key = _dump(node)
            replacement = self.substitutions.get(key)
            if replacement is not None:
                for child in ast.walk(node):
                    if isinstance(child, ast.expr):
                        child_key = _dump(child)
                        if child_key in self.counts:
                            self.counts[child_key] += 1
                return _clone_ast(replacement)
        return super().visit(node)


class _RenameLocals(ast.NodeTransformer):
    def __init__(self, names: Mapping[str, str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self.names.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)


def _bound_names(statements: Sequence[ast.stmt]) -> set[str]:
    return {
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _instantiate_nodes(
    statements: Sequence[ast.stmt],
    outputs: Sequence[ast.expr],
    substitutions: Sequence[tuple[ast.expr, ast.expr]],
    *,
    suffix: str,
    reserved_names: AbstractSet[str] = frozenset(),
) -> tuple[tuple[ast.stmt, ...], tuple[ast.expr, ...]] | None:
    if (
        not suffix.isidentifier()
        or keyword.iskeyword(suffix)
        or not _template_is_supported((*statements, *outputs))
    ):
        return None
    substitution_map: dict[str, ast.expr] = {}
    for source, replacement in substitutions:
        key = _dump(source)
        if key in substitution_map and _dump(substitution_map[key]) != _dump(
            replacement
        ):
            return None
        substitution_map[key] = replacement

    cloned_statements = tuple(_clone_ast(statement) for statement in statements)
    cloned_outputs = tuple(_clone_ast(output) for output in outputs)
    transformer = _StructuralSubstitution(substitution_map)
    transformed_statements = tuple(
        transformer.visit(statement) for statement in cloned_statements
    )
    transformed_outputs = tuple(transformer.visit(output) for output in cloned_outputs)
    if any(count == 0 for count in transformer.counts.values()) or not all(
        isinstance(statement, ast.stmt) for statement in transformed_statements
    ):
        return None

    local_names = _bound_names(cast("tuple[ast.stmt, ...]", transformed_statements))
    if any(
        isinstance(source, ast.Name) and source.id in local_names
        for source, _ in substitutions
    ):
        return None
    occupied = set(reserved_names)
    occupied.update(
        node.id
        for tree in (*transformed_statements, *transformed_outputs)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    )
    renames: dict[str, str] = {}
    for name in sorted(local_names):
        base = f"{name}_{suffix}"
        replacement = base
        counter = 0
        while replacement in occupied or keyword.iskeyword(replacement):
            counter += 1
            replacement = f"{base}_{counter}"
        if not replacement.isidentifier():
            return None
        renames[name] = replacement
        occupied.add(replacement)
    renamer = _RenameLocals(renames)
    renamed_statements = tuple(
        renamer.visit(statement) for statement in transformed_statements
    )
    renamed_outputs = tuple(renamer.visit(output) for output in transformed_outputs)
    if not all(
        isinstance(statement, ast.stmt) for statement in renamed_statements
    ) or not all(isinstance(output, ast.expr) for output in renamed_outputs):
        return None
    return (
        cast("tuple[ast.stmt, ...]", renamed_statements),
        cast("tuple[ast.expr, ...]", renamed_outputs),
    )


def instantiate_direct_affine_value(
    replay: DirectAffineValueReplay,
    *,
    row: ast.expr | None = None,
    feature: ast.expr | None = None,
    row_local: ast.expr | None = None,
    feature_local: ast.expr | None = None,
    suffix: str,
    reserved_names: AbstractSet[str] = frozenset(),
) -> tuple[tuple[ast.stmt, ...], ast.expr] | None:
    """Clone one opaque producer under explicit semantic substitutions."""

    result = _instantiate_bound_nodes(
        replay.bindings,
        replay.statements,
        (replay.value,),
        row=row,
        feature=feature,
        row_local=row_local,
        feature_local=feature_local,
        suffix=suffix,
        reserved_names=reserved_names,
    )
    if result is None:
        return None
    statements, outputs = result
    return statements, outputs[0]


def _contains_expression(trees: Sequence[ast.AST], expression: ast.expr) -> bool:
    expected = _dump(expression)
    return any(
        isinstance(node, ast.expr) and _dump(node) == expected
        for tree in trees
        for node in ast.walk(tree)
    )


def _inferred_bindings(
    statements: Sequence[ast.stmt],
    outputs: Sequence[ast.expr],
    coordinates: DirectAffineCoordinates,
) -> DirectAffineReplayBindings:
    trees: tuple[ast.AST, ...] = (*statements, *outputs)

    def equivalent_aliases(axis: DirectAffineOrdinaryAxis) -> tuple[ast.expr, ...]:
        expected = tuple(
            _evaluate_coordinate(
                axis.local_expression,
                thread_axis=axis.thread_axis,
                thread=thread,
                lane_name=axis.lane_name,
                lane=lane,
            )
            for thread in range(axis.thread_extent)
            for lane in range(axis.lane_extent)
        )
        aliases: dict[str, ast.expr] = {}

        def visit(node: ast.AST) -> None:
            if isinstance(node, ast.expr):
                values = tuple(
                    _evaluate_coordinate(
                        node,
                        thread_axis=axis.thread_axis,
                        thread=thread,
                        lane_name=axis.lane_name,
                        lane=lane,
                    )
                    for thread in range(axis.thread_extent)
                    for lane in range(axis.lane_extent)
                )
                if values == expected:
                    aliases.setdefault(_dump(node), _clone_ast(node))
                    return
            for child in ast.iter_child_nodes(node):
                visit(child)

        for tree in trees:
            visit(tree)
        aliases.pop(_dump(axis.source), None)
        return tuple(aliases.values())

    def global_aliases(
        axis: DirectAffineOrdinaryAxis,
    ) -> tuple[ast.expr, ...]:
        if isinstance(axis.tile_offset, ast.Constant) and axis.tile_offset.value == 0:
            return ()
        expected = tuple(
            _evaluate_coordinate(
                axis.local_expression,
                thread_axis=axis.thread_axis,
                thread=thread,
                lane_name=axis.lane_name,
                lane=lane,
            )
            for thread in range(axis.thread_extent)
            for lane in range(axis.lane_extent)
        )
        result: dict[str, ast.expr] = {}

        def visit(node: ast.AST) -> None:
            if isinstance(node, ast.expr):
                transformer = _StructuralSubstitution(
                    {_dump(axis.tile_offset): ast.Constant(value=0)}
                )
                without_offset = transformer.visit(_clone_ast(node))
                if transformer.counts[_dump(axis.tile_offset)] == 1:
                    values = tuple(
                        _evaluate_coordinate(
                            without_offset,
                            thread_axis=axis.thread_axis,
                            thread=thread,
                            lane_name=axis.lane_name,
                            lane=lane,
                        )
                        for thread in range(axis.thread_extent)
                        for lane in range(axis.lane_extent)
                    )
                    if values == expected:
                        result.setdefault(_dump(node), _clone_ast(node))
                        return
            for child in ast.iter_child_nodes(node):
                visit(child)

        for tree in trees:
            visit(tree)
        return tuple(result.values())

    row_aliases = equivalent_aliases(coordinates.row)
    feature_aliases = equivalent_aliases(coordinates.feature)
    row_globals = global_aliases(coordinates.row)
    feature_globals = global_aliases(coordinates.feature)
    return DirectAffineReplayBindings(
        row=(
            _clone_ast(coordinates.row.source)
            if _contains_expression(trees, coordinates.row.source)
            else None
        ),
        feature=(
            _clone_ast(coordinates.feature.source)
            if _contains_expression(trees, coordinates.feature.source)
            else None
        ),
        row_local=row_aliases[0] if row_aliases else None,
        feature_local=feature_aliases[0] if feature_aliases else None,
        row_alternates=row_aliases[1:],
        feature_alternates=feature_aliases[1:],
        row_global_alternates=row_globals,
        feature_global_alternates=feature_globals,
    )


def _instantiate_bound_nodes(
    bindings: DirectAffineReplayBindings,
    statements: Sequence[ast.stmt],
    outputs: Sequence[ast.expr],
    *,
    row: ast.expr | None,
    feature: ast.expr | None,
    row_local: ast.expr | None,
    feature_local: ast.expr | None,
    extra_substitutions: Sequence[tuple[ast.expr, ast.expr]] = (),
    suffix: str,
    reserved_names: AbstractSet[str],
) -> tuple[tuple[ast.stmt, ...], tuple[ast.expr, ...]] | None:
    replacements = {
        DirectAffineCoordinateRole.ROW: row,
        DirectAffineCoordinateRole.FEATURE: feature,
        DirectAffineCoordinateRole.ROW_LOCAL: row_local,
        DirectAffineCoordinateRole.FEATURE_LOCAL: feature_local,
    }
    substitutions: list[tuple[ast.expr, ast.expr]] = []
    for role, source in bindings.items():
        replacement = replacements[role]
        if replacement is None:
            return None
        substitutions.append((source, replacement))
    return _instantiate_nodes(
        statements,
        outputs,
        (*substitutions, *extra_substitutions),
        suffix=suffix,
        reserved_names=reserved_names,
    )
