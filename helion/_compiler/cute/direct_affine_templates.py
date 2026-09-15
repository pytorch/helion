# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve and validate opaque AST templates for direct affine replay."""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from .direct_affine_plan import DirectAffinePlan
from .direct_affine_plan import DirectAffineStateIngress
from .direct_affine_replay_core import DirectAffineAsyncEntryReplay
from .direct_affine_replay_core import DirectAffineCoordinateRole
from .direct_affine_replay_core import DirectAffineCoordinates
from .direct_affine_replay_core import DirectAffineEffectReplay
from .direct_affine_replay_core import DirectAffineNodeReplay
from .direct_affine_replay_core import DirectAffineOrdinaryAxis
from .direct_affine_replay_core import DirectAffineReplay
from .direct_affine_replay_core import DirectAffineReplayBindings
from .direct_affine_replay_core import DirectAffineReplayProof
from .direct_affine_replay_core import DirectAffineResolvedTemplates
from .direct_affine_replay_core import DirectAffineStateStoreReplay
from .direct_affine_replay_core import DirectAffineStepReplay
from .direct_affine_replay_core import DirectAffineValueReplay
from .direct_affine_replay_core import _bound_names
from .direct_affine_replay_core import _clone_ast
from .direct_affine_replay_core import _contains_expression
from .direct_affine_replay_core import _dump
from .direct_affine_replay_core import _inferred_bindings
from .direct_affine_replay_core import _qualified_name
from .direct_affine_replay_core import _result_snapshots_are_intact

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from .short_affine_scan import ShortAffineScanAccess


_INTEGER_CASTS = frozenset(
    {
        "cutlass.Int32",
        "cutlass.Int64",
        "cutlass.Uint32",
        "cutlass.Uint64",
    }
)


def _replay_by_node(
    replay: DirectAffineReplay,
) -> dict[torch.fx.Node, DirectAffineNodeReplay]:
    return {item.node: item for item in replay.nodes}


def _statements_for_nodes(
    replay: DirectAffineReplay,
    nodes: Iterable[torch.fx.Node],
) -> tuple[ast.stmt, ...] | None:
    by_node = _replay_by_node(replay)
    requested = set(nodes)
    if not requested:
        return ()
    source_positions = {
        id(statement): index for index, statement in enumerate(replay.source_statements)
    }
    collected: dict[int, ast.stmt] = {}
    for node in requested:
        item = by_node.get(node)
        if item is None:
            return None
        for statement in item.statements:
            if id(statement) not in source_positions:
                return None
            collected[id(statement)] = statement
    return tuple(
        collected[statement_id]
        for statement_id in sorted(collected, key=source_positions.__getitem__)
    )


def _result_expression(replay: DirectAffineReplay, value: object) -> ast.expr | None:
    if isinstance(value, Node):
        item = replay.node_replay(value)
        return (
            _clone_ast(item.result)
            if item is not None and isinstance(item.result, ast.expr)
            else None
        )
    if value is None or isinstance(value, (bool, int, float)):
        return ast.Constant(value=value)
    return None


def _dependency_value_replay(
    replay: DirectAffineReplay,
    value: object,
    coordinates: DirectAffineCoordinates,
) -> DirectAffineValueReplay | None:
    result = _result_expression(replay, value)
    if result is None:
        return None
    if not isinstance(value, Node):
        return DirectAffineValueReplay(statements=(), value=result)
    slices = tuple(
        dependency
        for dependency in replay.candidate.region.dependency_slices
        if dependency.value is value
    )
    if len(slices) != 1:
        return None
    statements = _statements_for_nodes(replay, slices[0].nodes)
    if statements is None:
        return None
    from .layout_propagation import _is_reduction_target

    reductions = tuple(
        node
        for node in slices[0].nodes
        if node.op == "call_function" and _is_reduction_target(node.target)
    )
    reduction_set = frozenset(reductions)
    if (
        any(
            sum(
                _is_lane_reduce_call(child)
                for statement in replay_item.statements
                for child in ast.walk(statement)
            )
            != (1 if node in reduction_set else 0)
            for node in slices[0].nodes
            if (replay_item := replay.node_replay(node)) is not None
        )
        or reductions
        and not _feature_reduction_markers_are_supported(
            statements,
            coordinates,
        )
    ):
        return None
    return DirectAffineValueReplay(
        statements=statements,
        value=result,
        bindings=_inferred_bindings(statements, (result,), coordinates),
    )


def _is_lane_reduce_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_helion_lane_reduce"
    )


def _has_lane_reduce_marker(trees: Iterable[ast.AST]) -> bool:
    return any(_is_lane_reduce_call(node) for tree in trees for node in ast.walk(tree))


def _feature_reduction_markers_are_supported(
    statements: Sequence[ast.stmt],
    coordinates: DirectAffineCoordinates,
) -> bool:
    """Require each reduction marker to be a direct feature-lane reduction."""

    from ..tile_strategy import _is_lane_reduce_marker_assign

    marker_statements = tuple(
        (index, statement)
        for index, statement in enumerate(statements)
        if any(_is_lane_reduce_call(node) for node in ast.walk(statement))
    )
    if not marker_statements:
        return True
    if not _flat_unique_assignments(statements):
        return False
    if sum(
        _is_lane_reduce_call(node)
        for statement in statements
        for node in ast.walk(statement)
    ) != len(marker_statements):
        return False
    for index, statement in marker_statements:
        try:
            marker = _is_lane_reduce_marker_assign(statement)
        except (AssertionError, TypeError, ValueError):
            return False
        if (
            marker is None
            or marker.reduction_type != "sum"
            or marker.threads_in_group != coordinates.feature.thread_extent
            or marker.group_pre != 1
            or marker.group_span != 0
            or marker.group_count != 1
            or marker.group_cluster_n != 1
        ):
            return False
        try:
            input_expression = ast.parse(marker.input_name, mode="eval").body
        except SyntaxError:
            return False
        if not isinstance(input_expression, ast.expr):
            return False
        support = _supporting_statements(statements, (input_expression,), index)
        bindings = _inferred_bindings(support, (input_expression,), coordinates)
        binding_items = bindings.items()
        roles = {item[0] for item in binding_items}
        if not roles.intersection(
            {
                DirectAffineCoordinateRole.FEATURE,
                DirectAffineCoordinateRole.FEATURE_LOCAL,
            }
        ):
            return False
    return True


def _fx_closure(
    roots: Iterable[torch.fx.Node],
    *,
    stop: frozenset[torch.fx.Node],
    allowed: frozenset[torch.fx.Node],
) -> tuple[torch.fx.Node, ...]:
    selected: set[torch.fx.Node] = set()

    def visit(node: torch.fx.Node) -> None:
        if node in stop or node in selected or node not in allowed:
            return
        selected.add(node)
        for source in node.all_input_nodes:
            visit(source)

    for root in roots:
        visit(root)
    return tuple(selected)


def _effect_replay(
    replay: DirectAffineReplay,
    *,
    effect: torch.fx.Node,
    logical_value: torch.fx.Node,
    allowed: frozenset[torch.fx.Node],
    coordinates: DirectAffineCoordinates,
) -> DirectAffineEffectReplay | None:
    nodes = _fx_closure(
        (effect,),
        stop=frozenset((logical_value,)),
        allowed=allowed,
    )
    statements = _statements_for_nodes(replay, nodes)
    logical_expression = _result_expression(replay, logical_value)
    if statements is None or logical_expression is None or not statements:
        return None
    if not _contains_expression(statements, logical_expression):
        return None
    return DirectAffineEffectReplay(
        statements=statements,
        logical_value=logical_expression,
        bindings=_inferred_bindings(
            statements,
            (logical_expression,),
            coordinates,
        ),
    )


def _read_names(trees: Sequence[ast.AST]) -> set[str]:
    return {
        node.id
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _supporting_statements(
    statements: Sequence[ast.stmt],
    outputs: Sequence[ast.expr],
    stop_index: int,
) -> tuple[ast.stmt, ...]:
    needed = _read_names(outputs)
    selected: list[ast.stmt] = []
    for statement in reversed(statements[:stop_index]):
        writes = _bound_names((statement,))
        if not writes.intersection(needed):
            continue
        selected.append(statement)
        needed.difference_update(writes)
        needed.update(_read_names((statement,)))
    selected.reverse()
    return tuple(selected)


def _flat_unique_assignments(statements: Sequence[ast.stmt]) -> bool:
    names: set[str] = set()
    for statement in statements:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id not in names
        ):
            return False
        names.add(statement.targets[0].id)
    return True


def _assignments_are_pure(statements: Sequence[ast.stmt], *, allow_load: bool) -> bool:
    from ..tile_strategy import _is_proven_relocatable_assignment

    return all(
        _is_proven_relocatable_assignment(statement, allow_load=allow_load)
        for statement in statements
    )


def _dropped_assignments_are_pure(
    statements: Sequence[ast.stmt],
    kept: Sequence[ast.stmt],
    effect_index: int,
) -> bool:
    """Prove assignments discarded while extracting one memory op are inert."""

    from ..tile_strategy import _is_proven_relocatable_assignment

    kept_ids = {id(statement) for statement in kept}
    return all(
        index == effect_index
        or id(statement) in kept_ids
        or _is_proven_relocatable_assignment(statement, allow_load=True)
        for index, statement in enumerate(statements)
    )


def _load_details(
    statements: Sequence[ast.stmt],
) -> tuple[int, ast.expr, ast.expr] | None:
    def load(expression: ast.expr) -> ast.Call | None:
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "load"
            and not expression.args
            and not expression.keywords
        ):
            return expression
        return None

    def zero(expression: ast.expr) -> bool:
        while (
            isinstance(expression, ast.Call)
            and (_qualified_name(expression.func) or "").startswith("cutlass.")
            and len(expression.args) == 1
            and not expression.keywords
            and isinstance(expression.args[0], ast.expr)
        ):
            expression = expression.args[0]
        return (
            isinstance(expression, ast.Constant)
            and isinstance(expression.value, (int, float))
            and expression.value == 0
        )

    candidates: list[tuple[int, ast.Call, ast.expr]] = []
    for statement_index, statement in enumerate(statements):
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            return None
        value = statement.value
        direct = load(value)
        if direct is not None:
            candidates.append((statement_index, direct, ast.Constant(value=True)))
            continue
        if isinstance(value, ast.IfExp):
            guarded = load(value.body)
            if guarded is not None and zero(value.orelse):
                candidates.append((statement_index, guarded, _clone_ast(value.test)))
                continue
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            for node in ast.walk(value)
        ):
            return None
    if len(candidates) != 1:
        return None
    index, load, guard = candidates[0]
    assert isinstance(load.func, ast.Attribute)
    if isinstance(guard, ast.BoolOp) and len(guard.values) == 2:
        first, second = guard.values
        if isinstance(first, ast.Constant) and first.value is True:
            guard = second
    return index, _clone_ast(load.func.value), guard


def _store_details(
    statements: Sequence[ast.stmt],
) -> tuple[int, ast.expr, ast.expr, ast.expr] | None:
    candidates: list[tuple[int, ast.Call, ast.expr]] = []
    unsupported = False

    def visit(statement: ast.stmt, statement_index: int, guard: ast.expr) -> None:
        nonlocal unsupported
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "store"
            and len(statement.value.args) == 1
            and not statement.value.keywords
        ):
            candidates.append((statement_index, statement.value, guard))
            return
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "store"
                    for node in ast.walk(statement)
                )
            ):
                unsupported = True
            return
        if isinstance(statement, ast.If):
            if (
                statement.orelse
                or len(statement.body) != 1
                or not isinstance(statement.body[0], (ast.Expr, ast.If))
            ):
                unsupported = True
                return
            body_guard = ast.BoolOp(
                op=ast.And(),
                values=[_clone_ast(guard), _clone_ast(statement.test)],
            )
            for child in statement.body:
                visit(child, statement_index, body_guard)
            return
        unsupported = True

    for index, statement in enumerate(statements):
        visit(statement, index, ast.Constant(value=True))
    if unsupported or len(candidates) != 1:
        return None
    index, store, guard = candidates[0]
    assert isinstance(store.func, ast.Attribute)
    if isinstance(guard, ast.BoolOp) and len(guard.values) == 2:
        first, second = guard.values
        if isinstance(first, ast.Constant) and first.value is True:
            guard = second
    return (
        index,
        _clone_ast(store.func.value),
        _clone_ast(store.args[0]),
        guard,
    )


def _strip_integer_cast(expression: ast.expr) -> ast.expr:
    while (
        isinstance(expression, ast.Call)
        and _qualified_name(expression.func) in _INTEGER_CASTS
        and len(expression.args) == 1
        and not expression.keywords
        and isinstance(expression.args[0], ast.expr)
    ):
        expression = expression.args[0]
    return expression


def _flatten_and(expression: ast.expr) -> tuple[ast.expr, ...]:
    if isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.And):
        return tuple(
            child for value in expression.values for child in _flatten_and(value)
        )
    return (expression,)


def _slot_guard(guard: ast.expr, slot: ast.expr) -> tuple[ast.expr, ast.expr] | None:
    expected = _dump(_strip_integer_cast(slot))
    terms = tuple(
        term
        for term in _flatten_and(guard)
        if not (isinstance(term, ast.Constant) and term.value is True)
    )
    matches = tuple(
        (index, term.comparators[0])
        for index, term in enumerate(terms)
        if isinstance(term, ast.Compare)
        and len(term.ops) == 1
        and isinstance(term.ops[0], ast.Lt)
        and len(term.comparators) == 1
        and isinstance(term.left, ast.expr)
        and isinstance(term.comparators[0], ast.expr)
        and _dump(_strip_integer_cast(term.left)) == expected
    )
    if len(matches) != 1:
        return None
    match_index, extent = matches[0]
    residual = tuple(term for index, term in enumerate(terms) if index != match_index)
    if not residual:
        valid: ast.expr = ast.Constant(value=True)
    elif len(residual) == 1:
        valid = _clone_ast(residual[0])
    else:
        valid = ast.BoolOp(op=ast.And(), values=[_clone_ast(term) for term in residual])
    return _clone_ast(extent), valid


def _store_value_is_logical_cast(
    value: ast.expr,
    logical_value: ast.expr,
    statements: Sequence[ast.stmt],
    stop_index: int,
    storage_dtype: torch.dtype,
) -> bool:
    cast_name = {
        torch.bfloat16: "cutlass.BFloat16",
        torch.float16: "cutlass.Float16",
    }.get(storage_dtype)
    if cast_name is None:
        return False
    assignments: dict[str, list[ast.expr]] = {}
    for statement in statements[: stop_index + 1]:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            assignments.setdefault(statement.targets[0].id, []).append(statement.value)
    current = value
    seen_names: set[str] = set()
    saw_cast = False
    for _ in range(len(assignments) + 2):
        if _dump(current) == _dump(logical_value):
            return saw_cast
        if isinstance(current, ast.Name):
            if current.id in seen_names or len(assignments.get(current.id, ())) != 1:
                return False
            seen_names.add(current.id)
            current = assignments[current.id][0]
            continue
        if (
            isinstance(current, ast.Call)
            and _qualified_name(current.func) == cast_name
            and len(current.args) == 1
            and not current.keywords
            and isinstance(current.args[0], ast.expr)
        ):
            saw_cast = True
            current = current.args[0]
            continue
        return False
    return False


def _output_store_value_is_logical(
    value: ast.expr,
    logical_value: ast.expr,
    statements: Sequence[ast.stmt],
    stop_index: int,
) -> bool:
    """Prove a pure, single-use dataflow from the logical output to the store."""

    from ..tile_strategy import _is_proven_relocatable_call

    assignments: dict[str, list[ast.expr]] = {}
    for statement in statements[: stop_index + 1]:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            assignments.setdefault(statement.targets[0].id, []).append(statement.value)

    def inspect(expression: ast.expr, resolving: frozenset[str]) -> tuple[bool, int]:
        if _dump(expression) == _dump(logical_value):
            return True, 1
        if isinstance(expression, ast.Name):
            definitions = assignments.get(expression.id, ())
            if not definitions:
                return True, 0
            if len(definitions) != 1 or expression.id in resolving:
                return False, 0
            return inspect(definitions[0], resolving | {expression.id})
        if isinstance(expression, ast.Call) and not _is_proven_relocatable_call(
            expression,
            allow_load=False,
        ):
            return False, 0
        valid = True
        occurrences = 0
        for child in ast.iter_child_nodes(expression):
            if not isinstance(child, ast.expr):
                continue
            child_valid, child_occurrences = inspect(child, resolving)
            valid &= child_valid
            occurrences += child_occurrences
        return valid, occurrences

    valid, occurrences = inspect(value, frozenset())
    return valid and occurrences == 1


def _packed_state_replay(
    replay: DirectAffineReplay,
    scalar: DirectAffineEffectReplay,
    access: ShortAffineScanAccess,
    coordinates: DirectAffineCoordinates,
) -> DirectAffineStateStoreReplay | None:
    indices = access.indices
    if len(indices) < 3:
        return None
    slot = _result_expression(replay, indices[0])
    details = _store_details(scalar.statements)
    if slot is None or details is None:
        return None
    statement_index, pointer, value, guard = details
    slot_guard = _slot_guard(guard, slot)
    if (
        slot_guard is None
        or not _store_value_is_logical_cast(
            value,
            scalar.logical_value,
            scalar.statements,
            statement_index,
            replay.candidate.region.storage_dtype,
        )
        or not _contains_expression((pointer,), coordinates.row.source)
        or not _contains_expression((pointer,), coordinates.feature.source)
    ):
        return None
    extent, valid = slot_guard
    support = _supporting_statements(
        scalar.statements,
        (pointer, slot, extent, valid),
        statement_index,
    )
    if (
        not _flat_unique_assignments(support)
        or not _assignments_are_pure(support, allow_load=True)
        or not _dropped_assignments_are_pure(
            scalar.statements,
            support,
            statement_index,
        )
    ):
        return None
    return DirectAffineStateStoreReplay(
        statements=support,
        pointer=pointer,
        slot=slot,
        slot_extent=extent,
        valid=valid,
        bindings=_inferred_bindings(
            support,
            (pointer, slot, extent, valid),
            coordinates,
        ),
    )


def _flatten_addition(expression: ast.expr) -> tuple[ast.expr, ...]:
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        return (
            *_flatten_addition(expression.left),
            *_flatten_addition(expression.right),
        )
    return (expression,)


def _layout_stride_details(expression: ast.expr) -> tuple[str, int] | None:
    value = expression
    if (
        isinstance(value, ast.Call)
        and _qualified_name(value.func) in _INTEGER_CASTS
        and len(value.args) == 1
        and not value.keywords
    ):
        value = value.args[0]
    if not (
        isinstance(value, ast.Subscript)
        and isinstance(value.slice, ast.Constant)
        and type(value.slice.value) is int
        and isinstance(value.value, ast.Attribute)
        and value.value.attr == "stride"
        and isinstance(value.value.value, ast.Attribute)
        and value.value.value.attr == "layout"
        and isinstance(value.value.value.value, ast.Name)
    ):
        return None
    return value.value.value.value.id, value.slice.value


def _layout_stride(expression: ast.expr) -> bool:
    return _layout_stride_details(expression) is not None


def _pointer_has_tensor_layout(
    pointer: ast.expr,
    *,
    argument_name: str,
    row: ast.expr,
    feature: ast.expr | None,
    row_dimension: int,
    feature_dimension: int | None,
) -> bool:
    if row_dimension < 0 or (feature is not None and feature_dimension is None):
        return False
    terms = _flatten_addition(pointer)
    iterator = ast.Attribute(
        value=ast.Name(id=argument_name, ctx=ast.Load()),
        attr="iterator",
        ctx=ast.Load(),
    )
    rank = max(row_dimension, feature_dimension or 0) + 1
    dimension_terms: dict[int, ast.expr] = {}
    iterator_count = 0
    for term in terms:
        if _dump(term) == _dump(iterator):
            iterator_count += 1
            continue
        stripped_term = _strip_integer_cast(term)
        if (
            isinstance(stripped_term, ast.Constant)
            and type(stripped_term.value) is int
            and stripped_term.value == 0
        ):
            continue
        if not isinstance(stripped_term, ast.BinOp) or not isinstance(
            stripped_term.op, ast.Mult
        ):
            return False
        candidates = tuple(
            (index, details)
            for index, details in (
                (stripped_term.left, _layout_stride_details(stripped_term.right)),
                (stripped_term.right, _layout_stride_details(stripped_term.left)),
            )
            if details is not None
        )
        if len(candidates) != 1:
            return False
        index, (stride_argument, dimension) = candidates[0]
        if (
            stride_argument != argument_name
            or dimension < 0
            or dimension >= rank
            or dimension in dimension_terms
        ):
            return False
        dimension_terms[dimension] = _strip_integer_cast(index)

    if (
        iterator_count != 1
        or row_dimension not in dimension_terms
        or _dump(dimension_terms[row_dimension]) != _dump(_strip_integer_cast(row))
        or _expression_occurrences(pointer, row) != 1
    ):
        return False
    if feature is None:
        return True
    assert feature_dimension is not None
    return (
        feature_dimension in dimension_terms
        and _dump(dimension_terms[feature_dimension])
        == _dump(_strip_integer_cast(feature))
        and _expression_occurrences(pointer, feature) == 1
    )


def _expression_occurrences(tree: ast.AST, expression: ast.expr) -> int:
    expected = _dump(expression)
    return sum(
        isinstance(node, ast.expr) and _dump(node) == expected
        for node in ast.walk(tree)
    )


def _row_stride(pointer: ast.expr, row: ast.expr) -> ast.expr | None:
    if _expression_occurrences(pointer, row) != 1:
        return None
    matches: list[ast.expr] = []
    for term in _flatten_addition(pointer):
        if not isinstance(term, ast.BinOp) or not isinstance(term.op, ast.Mult):
            continue
        for coordinate, stride in ((term.left, term.right), (term.right, term.left)):
            if not _layout_stride(stride) or _dump(
                _strip_integer_cast(coordinate)
            ) != _dump(_strip_integer_cast(row)):
                continue
            matches.append(stride)
    return _clone_ast(matches[0]) if len(matches) == 1 else None


def _async_entry_replay(
    entry: DirectAffineValueReplay,
    load_statements: Sequence[ast.stmt],
    coordinates: DirectAffineCoordinates,
) -> DirectAffineAsyncEntryReplay | None:
    details = _load_details(load_statements)
    if details is None:
        return None
    load_statement_index, pointer, valid = details
    load_statement = load_statements[load_statement_index]
    statement_positions = tuple(
        index
        for index, statement in enumerate(entry.statements)
        if statement is load_statement
    )
    if len(statement_positions) != 1:
        return None
    statement_index = statement_positions[0]
    row_stride = _row_stride(pointer, coordinates.row.source)
    if (
        row_stride is None
        or _expression_occurrences(pointer, coordinates.row.source) != 1
        or _expression_occurrences(pointer, coordinates.feature.source) != 1
    ):
        return None
    support = _supporting_statements(
        entry.statements,
        (pointer, valid, row_stride),
        statement_index,
    )
    load_support = _supporting_statements(
        load_statements,
        (pointer, valid, row_stride),
        load_statement_index,
    )
    if (
        not _flat_unique_assignments(support)
        or not _flat_unique_assignments(load_support)
        or not _assignments_are_pure(support, allow_load=True)
        or not _assignments_are_pure(load_support, allow_load=True)
        or not _dropped_assignments_are_pure(
            entry.statements,
            support,
            statement_index,
        )
        or not _dropped_assignments_are_pure(
            load_statements,
            load_support,
            load_statement_index,
        )
    ):
        return None
    return DirectAffineAsyncEntryReplay(
        statements=support,
        pointer=pointer,
        row_stride=row_stride,
        valid=valid,
        bindings=_inferred_bindings(
            support,
            (pointer, valid, row_stride),
            coordinates,
        ),
    )


def resolve_direct_affine_templates(
    replay: DirectAffineReplay,
    coordinates: DirectAffineCoordinates,
) -> DirectAffineResolvedTemplates | None:
    """Derive all semantic templates from candidate slices and FX ancestry.

    The resolver does not recognize source names or formulas.  Separate
    unrolled steps remain separate opaque templates; only row/feature
    coordinates found structurally in their generated AST are marked for
    substitution.
    """

    if not _result_snapshots_are_intact(replay):
        return None
    region = replay.candidate.region
    replacement_nodes = frozenset((*region.producer_nodes, *region.owned_nodes))
    entry_nodes = _fx_closure(
        (region.entry_state,),
        stop=frozenset(),
        allowed=replacement_nodes,
    )
    entry_statements = _statements_for_nodes(replay, entry_nodes)
    entry_result = _result_expression(replay, region.entry_state)
    entry_load = replay.node_replay(region.entry_load)
    from .layout_propagation import _is_reduction_target

    if (
        entry_statements is None
        or entry_result is None
        or entry_load is None
        or any(
            node.op == "call_function" and _is_reduction_target(node.target)
            for node in entry_nodes
        )
        or _has_lane_reduce_marker((*entry_statements, entry_result))
    ):
        return None
    entry = DirectAffineValueReplay(
        statements=entry_statements,
        value=entry_result,
        bindings=_inferred_bindings(
            entry_statements,
            (entry_result,),
            coordinates,
        ),
    )

    source_positions = {
        node: index for index, node in enumerate(region.source_interval)
    }
    steps: list[DirectAffineStepReplay] = []
    for step in region.steps:
        values = tuple(
            _dependency_value_replay(replay, value, coordinates)
            for value in (
                step.diagonal,
                step.prediction_vector,
                step.row_input,
                step.update_scale,
                step.update_vector,
                step.observation_vector,
            )
        )
        if any(value is None for value in values):
            return None
        resolved_values = cast("tuple[DirectAffineValueReplay, ...]", values)
        if any(
            _has_lane_reduce_marker((*resolved_values[index].statements,))
            for index in (2, 3)
        ):
            return None
        allowed = frozenset((*region.producer_nodes, *step.owned_nodes))
        output = _effect_replay(
            replay,
            effect=step.output_effect,
            logical_value=step.observation,
            allowed=allowed,
            coordinates=coordinates,
        )
        scalar_state = _effect_replay(
            replay,
            effect=step.state_effect,
            logical_value=step.state,
            allowed=allowed,
            coordinates=coordinates,
        )
        if (
            output is None
            or scalar_state is None
            or _has_lane_reduce_marker(
                (
                    *output.statements,
                    output.logical_value,
                    *scalar_state.statements,
                    scalar_state.logical_value,
                )
            )
        ):
            return None
        state = _packed_state_replay(
            replay,
            scalar_state,
            step.state_access,
            coordinates,
        )
        if state is None:
            return None
        steps.append(
            DirectAffineStepReplay(
                diagonal=cast("DirectAffineValueReplay", values[0]),
                prediction_vector=cast("DirectAffineValueReplay", values[1]),
                row_input=cast("DirectAffineValueReplay", values[2]),
                update_scale=cast("DirectAffineValueReplay", values[3]),
                update_vector=cast("DirectAffineValueReplay", values[4]),
                observation_vector=cast("DirectAffineValueReplay", values[5]),
                output_effect=output,
                state_effect=state,
                output_before_state=(
                    source_positions[step.output_effect]
                    < source_positions[step.state_effect]
                ),
            )
        )
    return DirectAffineResolvedTemplates(
        entry_state=entry,
        steps=tuple(steps),
        async_entry_state=_async_entry_replay(
            entry,
            entry_load.statements,
            coordinates,
        ),
    )


def _integer_constant(expression: ast.expr) -> int | None:
    expression = _strip_integer_cast(expression)
    if isinstance(expression, ast.Constant) and type(expression.value) is int:
        return expression.value
    return None


def _ordinary_lane_expression(
    expression: ast.expr,
    feature: DirectAffineOrdinaryAxis,
) -> bool:
    expression = _strip_integer_cast(expression)
    if (
        isinstance(expression, ast.Call)
        and _qualified_name(expression.func) == "cute.arch.lane_idx"
        and not expression.args
        and not expression.keywords
    ):
        return True
    return (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Call)
        and _qualified_name(expression.value.func) == "cute.arch.thread_idx"
        and not expression.value.args
        and not expression.value.keywords
        and isinstance(expression.slice, ast.expr)
        and _integer_constant(expression.slice) == feature.thread_axis
        and feature.thread_extent == 32
    )


def _ordinary_ownership_guard(
    expression: ast.expr,
    feature: DirectAffineOrdinaryAxis,
) -> bool:
    return (
        isinstance(expression, ast.Compare)
        and len(expression.ops) == 1
        and isinstance(expression.ops[0], ast.Eq)
        and len(expression.comparators) == 1
        and isinstance(expression.comparators[0], ast.expr)
        and _integer_constant(expression.comparators[0]) == 0
        and isinstance(expression.left, ast.BinOp)
        and isinstance(expression.left.op, ast.Mod)
        and isinstance(expression.left.right, ast.expr)
        and _integer_constant(expression.left.right) == 32
        and isinstance(expression.left.left, ast.expr)
        and _ordinary_lane_expression(expression.left.left, feature)
    )


def _remove_ownership_atom(
    expression: ast.expr,
    feature: DirectAffineOrdinaryAxis,
) -> tuple[ast.expr | None, int]:
    if _ordinary_ownership_guard(expression, feature):
        return None, 1
    if not isinstance(expression, ast.BoolOp) or not isinstance(expression.op, ast.And):
        return expression, 0
    values: list[ast.expr] = []
    removed = 0
    for value in expression.values:
        if _ordinary_ownership_guard(value, feature):
            removed += 1
        else:
            values.append(value)
    if not values:
        return None, removed
    if len(values) == 1:
        return values[0], removed
    return ast.BoolOp(op=ast.And(), values=values), removed


def _store_ownership_paths(
    statements: Sequence[ast.stmt],
    feature: DirectAffineOrdinaryAxis,
) -> tuple[int, ...]:
    paths: list[int] = []

    def visit(node: ast.AST, ownership_depth: int) -> None:
        if isinstance(node, ast.If):
            _remaining, removed = _remove_ownership_atom(node.test, feature)
            for child in node.body:
                visit(child, ownership_depth + removed)
            for child in node.orelse:
                visit(child, ownership_depth)
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
        ):
            paths.append(ownership_depth)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, ownership_depth)

    for statement in statements:
        visit(statement, 0)
    return tuple(paths)


class _DischargeOutputOwnership(ast.NodeTransformer):
    def __init__(self, feature: DirectAffineOrdinaryAxis) -> None:
        self.feature = feature
        self.removed = 0
        self.invalid = False

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        node = cast("ast.If", self.generic_visit(node))
        test, removed = _remove_ownership_atom(node.test, self.feature)
        if removed == 0:
            return node
        self.removed += removed
        if removed != 1 or node.orelse:
            self.invalid = True
            return node
        if test is None:
            return node.body
        node.test = test
        return node


def _discharge_output_effect(
    effect: DirectAffineEffectReplay,
    coordinates: DirectAffineCoordinates,
    *,
    allow_pre_wrap: bool,
    expected_argument_name: str | None,
    expected_rank: int,
) -> DirectAffineEffectReplay | None:
    feature = coordinates.feature
    details = _store_details(effect.statements)
    ownership_paths = _store_ownership_paths(effect.statements, feature)
    if details is None or ownership_paths not in ((1,), (0,)):
        return None
    statement_index, pointer, value, guard = details
    residual_guard, removed_ownership = _remove_ownership_atom(guard, feature)
    if ownership_paths == (1,) and removed_ownership != 1:
        return None
    if residual_guard is not None and not _guard_calls_are_pure(residual_guard):
        return None
    expanded_pointer = _expand_expression_assignments(pointer, effect.statements)
    if (
        expanded_pointer is None
        or not _contains_expression((expanded_pointer,), coordinates.row.source)
        or _contains_expression((expanded_pointer,), coordinates.feature.source)
        or (
            expected_argument_name is not None
            and (
                not _guard_calls_are_pure(expanded_pointer, allow_load=True)
                or not _pointer_has_tensor_layout(
                    expanded_pointer,
                    argument_name=expected_argument_name,
                    row=coordinates.row.source,
                    feature=None,
                    row_dimension=expected_rank - 1,
                    feature_dimension=None,
                )
            )
        )
    ):
        return None
    if not _output_store_value_is_logical(
        value,
        effect.logical_value,
        effect.statements,
        statement_index,
    ):
        return None
    from ..tile_strategy import _is_proven_relocatable_assignment

    if any(
        index != statement_index
        and not _is_proven_relocatable_assignment(statement, allow_load=False)
        for index, statement in enumerate(effect.statements)
    ):
        return None
    if ownership_paths == (0,):
        if (
            not allow_pre_wrap
            or effect.bindings.feature is not None
            or effect.bindings.feature_local is not None
            or effect.bindings.feature_alternates
            or _contains_expression(effect.statements, feature.source)
            or _contains_expression(effect.statements, feature.local_expression)
        ):
            return None
        return effect
    transformer = _DischargeOutputOwnership(feature)
    statements: list[ast.stmt] = []
    for statement in (_clone_ast(item) for item in effect.statements):
        transformed = transformer.visit(statement)
        if isinstance(transformed, list):
            statements.extend(transformed)
        elif isinstance(transformed, ast.stmt):
            statements.append(transformed)
        else:
            return None
    if transformer.invalid or transformer.removed != 1:
        return None
    return dataclasses.replace(effect, statements=tuple(statements))


def _discharge_feature_bounds(
    state: DirectAffineStateStoreReplay,
    feature: DirectAffineOrdinaryAxis,
    proof: DirectAffineReplayProof | None,
) -> DirectAffineStateStoreReplay | None:
    expanded_slot = _expand_expression_assignments(state.slot, state.statements)
    expanded_extent = _expand_expression_assignments(
        state.slot_extent, state.statements
    )
    expanded = _expand_expression_assignments(state.valid, state.statements)
    if (
        expanded_slot is None
        or expanded_extent is None
        or expanded is None
        or not _guard_calls_are_pure(expanded_slot, allow_load=True)
        or not _guard_calls_are_pure(expanded_extent, allow_load=True)
        or not _guard_calls_are_pure(expanded)
    ):
        return None
    aliases = tuple(
        source
        for role, source in state.bindings.items()
        if role
        in {
            DirectAffineCoordinateRole.FEATURE,
            DirectAffineCoordinateRole.FEATURE_LOCAL,
        }
    ) or (feature.source, feature.local_expression)
    for term in _flatten_and(expanded):
        if not any(_contains_expression((term,), alias) for alias in aliases):
            continue
        if not _vector_bound_atom(
            term,
            aliases,
            feature.extent,
            proof.state_feature_extent_expression if proof is not None else None,
        ):
            return None
    return state


def _expand_expression_assignments(
    expression: ast.expr,
    statements: Sequence[ast.stmt],
) -> ast.expr | None:
    definitions: dict[str, list[ast.expr]] = {}
    for statement in statements:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            definitions.setdefault(statement.targets[0].id, []).append(statement.value)

    def expand(value: ast.expr, resolving: frozenset[str]) -> ast.expr | None:
        if isinstance(value, ast.Name) and value.id in definitions:
            candidates = definitions[value.id]
            if len(candidates) != 1 or value.id in resolving:
                return None
            return expand(candidates[0], resolving | {value.id})
        fields: dict[str, object] = {}
        for field in value._fields:
            original = getattr(value, field)
            if isinstance(original, ast.expr):
                replacement = expand(original, resolving)
                if replacement is None:
                    return None
                fields[field] = replacement
            elif isinstance(original, list):
                items: list[object] = []
                for item in original:
                    if isinstance(item, ast.expr):
                        replacement = expand(item, resolving)
                        if replacement is None:
                            return None
                        items.append(replacement)
                    else:
                        items.append(
                            _clone_ast(item) if isinstance(item, ast.AST) else item
                        )
                fields[field] = items
            else:
                fields[field] = (
                    _clone_ast(original) if isinstance(original, ast.AST) else original
                )
        result = _clone_ast(value)
        for field, replacement in fields.items():
            setattr(result, field, replacement)
        return result if isinstance(result, ast.expr) else None

    return expand(expression, frozenset())


def _guard_calls_are_pure(expression: ast.expr, *, allow_load: bool = False) -> bool:
    from ..tile_strategy import _is_proven_relocatable_call

    return all(
        _is_proven_relocatable_call(call, allow_load=allow_load)
        for call in ast.walk(expression)
        if isinstance(call, ast.Call)
    )


def _vector_bound_atom(
    term: ast.expr,
    aliases: Sequence[ast.expr],
    extent: int,
    extent_expression: ast.expr | None,
) -> bool:
    if not (
        isinstance(term, ast.Compare)
        and len(term.ops) == 1
        and isinstance(term.ops[0], (ast.Lt, ast.LtE))
        and len(term.comparators) == 1
        and isinstance(term.left, ast.expr)
        and isinstance(term.comparators[0], ast.expr)
    ):
        return False
    left = _dump(_strip_integer_cast(term.left))
    if sum(left == _dump(_strip_integer_cast(alias)) for alias in aliases) != 1:
        return False
    bound = _strip_integer_cast(term.comparators[0])
    expected = extent if isinstance(term.ops[0], ast.Lt) else extent - 1
    return (
        isinstance(bound, ast.Constant)
        and type(bound.value) is int
        and bound.value == expected
    ) or (
        isinstance(term.ops[0], ast.Lt)
        and extent_expression is not None
        and _dump(_strip_integer_cast(extent_expression)) == _dump(bound)
    )


def _async_guard_is_tile_uniform(
    entry: DirectAffineAsyncEntryReplay,
    coordinates: DirectAffineCoordinates,
    proof: DirectAffineReplayProof,
) -> bool:
    expanded = _expand_expression_assignments(entry.valid, entry.statements)
    if expanded is None:
        return False
    if not _guard_calls_are_pure(expanded):
        return False
    groups = (
        (
            tuple(
                source
                for role, source in entry.bindings.items()
                if role is DirectAffineCoordinateRole.ROW
            ),
            proof.runtime_state_shape[0],
            proof.state_row_extent_expression,
        ),
        (
            tuple(
                source
                for role, source in entry.bindings.items()
                if role is DirectAffineCoordinateRole.ROW_LOCAL
            ),
            coordinates.row.extent,
            None,
        ),
        (
            tuple(
                source
                for role, source in entry.bindings.items()
                if role is DirectAffineCoordinateRole.FEATURE
            ),
            proof.runtime_state_shape[1],
            proof.state_feature_extent_expression,
        ),
        (
            tuple(
                source
                for role, source in entry.bindings.items()
                if role is DirectAffineCoordinateRole.FEATURE_LOCAL
            ),
            coordinates.feature.extent,
            (
                proof.state_feature_extent_expression
                if proof.runtime_state_shape[1] == coordinates.feature.extent
                else None
            ),
        ),
    )
    for term in _flatten_and(expanded):
        dependent_groups = tuple(
            (aliases, extent, extent_expression)
            for aliases, extent, extent_expression in groups
            if aliases
            and any(_contains_expression((term,), alias) for alias in aliases)
        )
        if dependent_groups:
            matching_groups = tuple(
                (aliases, extent, extent_expression)
                for aliases, extent, extent_expression in dependent_groups
                if _vector_bound_atom(
                    term,
                    aliases,
                    extent,
                    extent_expression,
                )
            )
            if len(matching_groups) != 1:
                return False
            aliases, extent, extent_expression = matching_groups[0]
            if not _vector_bound_atom(
                term,
                aliases,
                extent,
                extent_expression,
            ):
                return False
    return True


def _template_components(
    templates: DirectAffineResolvedTemplates,
) -> Iterable[
    tuple[
        DirectAffineReplayBindings,
        tuple[ast.AST, ...],
    ]
]:
    entry = templates.entry_state
    yield entry.bindings, (*entry.statements, entry.value)
    if (async_entry := templates.async_entry_state) is not None:
        yield (
            async_entry.bindings,
            (
                *async_entry.statements,
                async_entry.pointer,
                async_entry.row_stride,
                async_entry.valid,
            ),
        )
    for step in templates.steps:
        for value in (
            step.diagonal,
            step.prediction_vector,
            step.row_input,
            step.update_scale,
            step.update_vector,
            step.observation_vector,
        ):
            yield value.bindings, (*value.statements, value.value)
        output = step.output_effect
        yield output.bindings, (*output.statements, output.logical_value)
        state = step.state_effect
        yield (
            state.bindings,
            (
                *state.statements,
                state.pointer,
                state.slot,
                state.slot_extent,
                state.valid,
            ),
        )


def _template_reduction_placement_is_supported(
    templates: DirectAffineResolvedTemplates,
    coordinates: DirectAffineCoordinates,
) -> bool:
    entry = templates.entry_state
    if _has_lane_reduce_marker((*entry.statements, entry.value)):
        return False
    if templates.async_entry_state is not None and _has_lane_reduce_marker(
        (
            *templates.async_entry_state.statements,
            templates.async_entry_state.pointer,
            templates.async_entry_state.row_stride,
            templates.async_entry_state.valid,
        )
    ):
        return False
    for step in templates.steps:
        for value in (
            step.diagonal,
            step.prediction_vector,
            step.update_vector,
            step.observation_vector,
        ):
            if not _feature_reduction_markers_are_supported(
                value.statements,
                coordinates,
            ):
                return False
        for value in (step.row_input, step.update_scale):
            if _has_lane_reduce_marker((*value.statements, value.value)):
                return False
        state = step.state_effect
        if _has_lane_reduce_marker(
            (
                *step.output_effect.statements,
                step.output_effect.logical_value,
                *state.statements,
                state.pointer,
                state.slot,
                state.slot_extent,
                state.valid,
            )
        ):
            return False
    return True


def _has_unbound_hardware_coordinate(
    templates: DirectAffineResolvedTemplates,
    coordinates: DirectAffineCoordinates,
) -> bool:
    hardware = {"cute.arch.thread_idx", "cute.arch.lane_idx", "cute.arch.warp_idx"}
    forbidden_names = {
        name
        for name in (
            coordinates.row.lane_name,
            coordinates.feature.lane_name,
            coordinates.row.source.id
            if isinstance(coordinates.row.source, ast.Name)
            else None,
            coordinates.feature.source.id
            if isinstance(coordinates.feature.source, ast.Name)
            else None,
        )
        if name is not None
    }

    def contains(node: ast.AST, bound: set[str]) -> bool:
        if isinstance(node, ast.expr) and _dump(node) in bound:
            return False
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in forbidden_names
        ):
            return True
        if isinstance(node, ast.Call) and _qualified_name(node.func) in hardware:
            return True
        return any(contains(child, bound) for child in ast.iter_child_nodes(node))

    for bindings, trees in _template_components(templates):
        bound = {_dump(source) for source in bindings.values()}
        if any(contains(tree, bound) for tree in trees):
            return True
    return False


def _validate_direct_affine_templates(
    templates: DirectAffineResolvedTemplates,
    replay: DirectAffineReplay,
    plan: DirectAffinePlan,
    coordinates: DirectAffineCoordinates,
    proof: DirectAffineReplayProof | None = None,
) -> DirectAffineResolvedTemplates | None:
    """Discharge ordinary ownership and require the packed production ABI."""

    if (
        (proof is not None and proof.source_replay is not replay)
        or len(templates.steps) != plan.step_count
        or (
            plan.state_ingress is DirectAffineStateIngress.ASYNC
            and templates.async_entry_state is None
        )
        or not _template_reduction_placement_is_supported(templates, coordinates)
    ):
        return None
    pointer_contract_required = bool(replay.nodes) and all(
        item.result_fingerprint is not None for item in replay.nodes
    )
    if pointer_contract_required and (
        proof is None
        or proof.state_argument_name is None
        or proof.state_rank < 2
        or len(proof.output_argument_names) != plan.step_count
        or len(proof.output_ranks) != plan.step_count
    ):
        return None
    async_entry_state = (
        templates.async_entry_state
        if plan.state_ingress is DirectAffineStateIngress.ASYNC
        else None
    )
    async_pointer = (
        _expand_expression_assignments(
            async_entry_state.pointer, async_entry_state.statements
        )
        if async_entry_state is not None
        else None
    )
    async_row_stride = (
        _expand_expression_assignments(
            async_entry_state.row_stride, async_entry_state.statements
        )
        if async_entry_state is not None
        else None
    )
    if async_entry_state is not None and (
        proof is None
        or async_pointer is None
        or async_row_stride is None
        or not _guard_calls_are_pure(async_pointer, allow_load=True)
        or not _guard_calls_are_pure(async_row_stride, allow_load=True)
        or (
            pointer_contract_required
            and not _pointer_has_tensor_layout(
                async_pointer,
                argument_name=cast("str", proof.state_argument_name),
                row=coordinates.row.source,
                feature=coordinates.feature.source,
                row_dimension=proof.state_rank - 2,
                feature_dimension=proof.state_rank - 1,
            )
        )
        or not _async_guard_is_tile_uniform(
            async_entry_state,
            coordinates,
            proof,
        )
    ):
        return None
    steps: list[DirectAffineStepReplay] = []
    for step_index, step in enumerate(templates.steps):
        state_pointer = _expand_expression_assignments(
            step.state_effect.pointer, step.state_effect.statements
        )
        if (
            pointer_contract_required
            and proof is not None
            and (
                state_pointer is None
                or not _guard_calls_are_pure(state_pointer, allow_load=True)
                or not _pointer_has_tensor_layout(
                    state_pointer,
                    argument_name=cast("str", proof.state_argument_name),
                    row=coordinates.row.source,
                    feature=coordinates.feature.source,
                    row_dimension=proof.state_rank - 2,
                    feature_dimension=proof.state_rank - 1,
                )
            )
        ):
            return None
        state = _discharge_feature_bounds(
            step.state_effect,
            coordinates.feature,
            proof,
        )
        if state is None:
            return None
        output = _discharge_output_effect(
            step.output_effect,
            coordinates,
            allow_pre_wrap=(
                proof is not None and proof.output_store_pre_wrap_proven is True
            ),
            expected_argument_name=(
                proof.output_argument_names[step_index]
                if pointer_contract_required and proof is not None
                else None
            ),
            expected_rank=(
                proof.output_ranks[step_index]
                if pointer_contract_required and proof is not None
                else 0
            ),
        )
        if output is None:
            return None
        steps.append(
            dataclasses.replace(step, output_effect=output, state_effect=state)
        )
    result = DirectAffineResolvedTemplates(
        entry_state=templates.entry_state,
        steps=tuple(steps),
        async_entry_state=async_entry_state,
    )
    if _has_unbound_hardware_coordinate(result, coordinates):
        return None
    return result
