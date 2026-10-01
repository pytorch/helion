# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Original typed point/collective programs executed at an admitted phase cut.

This is a lowering component, not storage or effect admission. Callers retain
their original graph, alias, domain and lifetime proofs. Inputs are scalar IR,
ownership and store descriptors, never an already rendered producer loop. Each
evaluation instance is retained, including repeated uses of the same FX node.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeAlias
from typing import TypeVar
from typing import cast

from ..ast_extension import ExtendedAST
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import AbstractSet

    from .chained_execution import ChainedExecution


_A = TypeVar("_A", bound=ast.AST)


def _copy_ir(node: _A) -> _A:
    # Copy only AST fields, retaining original source/type/loop metadata. In
    # particular, deepcopy would also copy metadata's graph and codegen state.
    fields: dict[str, object] = {}
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            fields[field] = _copy_ir(value)
        elif isinstance(value, list):
            fields[field] = [
                _copy_ir(v) if isinstance(v, ast.AST) else v for v in value
            ]
        elif isinstance(value, tuple):
            fields[field] = tuple(
                _copy_ir(v) if isinstance(v, ast.AST) else v for v in value
            )
        else:
            fields[field] = value
    result = (
        node.copy(**fields)
        if isinstance(node, ExtendedAST)
        else ast.copy_location(type(node)(**fields), node)
    )
    for name, metadata in vars(node).items():
        if name not in node._fields:
            setattr(result, name, metadata)
    return cast("_A", result)


@dataclass(frozen=True)
class PointStore:
    target: ast.expr
    pointer: bool = False
    cast_to: ast.expr | None = None

    def statement(self, value: ast.expr) -> ast.stmt:
        result = _copy_ir(value)
        if self.cast_to is not None:
            result = ast.Call(func=_copy_ir(self.cast_to), args=[result], keywords=[])
        target = _copy_ir(self.target)
        if self.pointer:
            return ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=target, attr="store", ctx=ast.Load()),
                    args=[result],
                    keywords=[],
                )
            )
        assert isinstance(target, ast.Subscript)
        target.ctx = ast.Store()
        return ast.Assign(targets=[target], value=result)


@dataclass(frozen=True)
class PointValue:
    statements: tuple[ast.stmt, ...]
    value: ast.expr
    store: PointStore


@dataclass(frozen=True)
class PointIR:
    """Original scalar math, with no ownership loop or publication hidden in it."""

    statements: tuple[ast.stmt, ...]


@dataclass(frozen=True)
class PointLoop:
    index: str
    iterator: ast.expr
    body: tuple[PointAction, ...]
    # The existing marked-lane recipe is finalized only after its exact scalar
    # math and typed stores are lowered. This is not a user-supplied callback.
    reduction_lane: str | None = None
    shared_names: frozenset[str] = frozenset()
    reserved_names: frozenset[str] = frozenset()
    # The proved marker owner replaced by this loop's index, if any.
    source_lane: str | None = None


@dataclass(frozen=True)
class PointGuard:
    test: ast.expr
    body: tuple[PointAction, ...]


PointAction: TypeAlias = PointIR | PointValue | PointLoop | PointGuard


@dataclass(frozen=True)
class LaneProducer:
    """An original marked lane loop and its scalar publications."""

    owner: ast.expr
    lane: str
    element: str
    elements: int
    values: tuple[PointValue, ...]
    # Scalar expressions execute on the original full active owner; only the
    # publication is guarded. Moving their evaluation under the guard is wrong.
    scalars: tuple[tuple[ast.expr, PointValue], ...]
    shared_names: frozenset[str]
    reserved_names: frozenset[str]
    # The validated replay's ordinary feature lane. Markers owned by it are
    # rebound to ``element``; markers owned by any other lane are rejected.
    source_lane: str | None = None


@dataclass(frozen=True)
class CollectiveProducer:
    """The original strided sum or selected serial/warp prefix program."""

    name: str
    kind: Literal["sum", "serial_scan", "warp_scan"]
    extent: int
    vectors: int
    coords: tuple[str, ...]
    output_coords: tuple[str, ...]
    statements: tuple[str, ...]
    value: str
    execution: ChainedExecution


def split_producer_reductions(
    loop: ast.For,
    shared_names: AbstractSet[str],
    lane_name: str,
    reserved_names: AbstractSet[str],
    *,
    source_lane: str | None = None,
) -> tuple[ast.stmt, ...] | None:
    """Apply the existing typed marker splitter, with exact finalization."""
    from ... import exc
    from ..tile_strategy import _find_lane_reduce_call
    from ..tile_strategy import _is_lane_reduce_marker_assign
    from ..tile_strategy import split_lane_loop_reductions

    # The splitter mutates its input. Work on private scalar IR, not the
    # caller's original proof/replay or a previously accepted body.
    loop = _copy_ir(loop)
    if source_lane is not None:
        destination_lane = getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None)
        if not isinstance(destination_lane, str):
            return None
        # The validated replay replaces the complete ordinary feature axis.
        # Its owner is a marker string, not an expression substitution; rebind
        # exactly that proved owner on this detached materialization loop.
        for statement in loop.body:
            marker = _is_lane_reduce_marker_assign(statement)
            if marker is None or marker.owner_lane is None:
                continue
            if marker.owner_lane != source_lane:
                return None
            call = _find_lane_reduce_call(statement)
            assert call is not None
            call.args[9] = ast.copy_location(
                ast.Constant(value=destination_lane), call.args[9]
            )
    markers = tuple(
        marker
        for statement in loop.body
        if (marker := _is_lane_reduce_marker_assign(statement)) is not None
    )
    tensor_names = {
        node.value.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }
    disjoint_pairs = {
        frozenset((shared, other))
        for shared in shared_names
        for other in tensor_names
        if shared != other
    }
    try:
        result = split_lane_loop_reductions(
            [ast.fix_missing_locations(loop)],
            proven_disjoint_tensor_pairs=disjoint_pairs,
            thread_axis_names={lane_name: frozenset((0,))},
        )
    except exc.BackendUnsupported:
        return None
    detached = tuple(_copy_ir(statement) for statement in result)

    def bound_names(statements: Sequence[ast.AST]) -> set[str]:
        return {
            node.id
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }

    if (bound_names(detached) - bound_names(loop.body)).intersection(reserved_names):
        return None
    expected: dict[str, int] = {}
    for marker in markers:
        expected[marker.reduction_type] = expected.get(marker.reduction_type, 0) + 1
    actual: dict[str, int] = {}
    for statement in detached:
        if _find_lane_reduce_call(statement) is not None:
            return None
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                prefix = "cute.arch.warp_reduction_"
                if name.startswith(prefix):
                    kind = name.removeprefix(prefix)
                    actual[kind] = actual.get(kind, 0) + 1
    if actual != expected:
        return None
    return cast("tuple[ast.stmt, ...]", detached)


def emit_point_actions(actions: tuple[PointAction, ...]) -> tuple[ast.stmt, ...] | None:
    """The one iteration/guard/scalar/store interpreter used by both clients."""
    result: list[ast.stmt] = []
    for action in actions:
        if isinstance(action, (PointIR, PointValue)):
            for statement in action.statements:
                for node in ast.walk(statement):
                    if (
                        isinstance(node, (ast.For, ast.AsyncFor, ast.While))
                        or (
                            isinstance(node, ast.Subscript)
                            and isinstance(node.ctx, ast.Store)
                        )
                        or (
                            isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "store"
                        )
                    ):
                        return None
            result.extend(_copy_ir(statement) for statement in action.statements)
            if isinstance(action, PointValue):
                result.append(action.store.statement(action.value))
            continue
        body = emit_point_actions(action.body)
        if body is None:
            return None
        if isinstance(action, PointGuard):
            result.append(
                ast.If(test=_copy_ir(action.test), body=list(body), orelse=[])
            )
        else:
            assert isinstance(action, PointLoop)
            loop = ast.For(
                target=ast.Name(id=action.index, ctx=ast.Store()),
                iter=_copy_ir(action.iterator),
                body=list(body),
                orelse=[],
            )
            if action.reduction_lane is None:
                result.append(loop)
            else:
                setattr(loop, HELION_LANE_LOOP_VAR_ATTR, action.index)
                split = split_producer_reductions(
                    loop,
                    action.shared_names,
                    action.reduction_lane,
                    action.reserved_names,
                    source_lane=action.source_lane,
                )
                if split is None:
                    return None
                result.extend(split)
    return tuple(result)


def _expression(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _point_ir(*lines: str) -> PointIR:
    return PointIR(tuple(ast.parse("\n".join(lines)).body))


def _lane_actions(program: LaneProducer) -> tuple[PointAction, ...]:
    body: list[PointAction] = [
        PointLoop(
            program.element,
            _expression(f"cutlass.range_constexpr({program.elements})"),
            program.values,
            program.lane,
            program.shared_names,
            program.reserved_names,
            program.source_lane,
        )
    ]
    for guard, point in program.scalars:
        body.extend(
            (
                PointIR(point.statements),
                PointGuard(guard, (PointValue((), point.value, point.store),)),
            )
        )
    return (PointGuard(program.owner, tuple(body)),)


def _collective_actions(program: CollectiveProducer) -> tuple[PointAction, ...]:
    """Describe the selected original math; shared actions own all rendering."""
    from .chained_collectives import warp_prefix_point

    name, extent, vectors = program.name, program.extent, program.vectors
    execution, value = program.execution, program.value
    vector, position = f"{name}_vector", f"{name}_position"
    acc, carry = f"{name}_acc", f"{name}_carry"
    warps = execution.threads // 32
    coords = program.output_coords if program.kind == "sum" else program.coords
    publication = PointValue(
        (), _expression(acc), PointStore(_expression(f"{name}[{', '.join(coords)}]"))
    )
    owner_count = execution.threads if program.kind == "serial_scan" else warps
    owner_index = (
        execution.thread
        if program.kind == "serial_scan"
        else f"{execution.thread} // 32"
    )
    body: list[PointAction] = []
    if program.kind == "warp_scan":
        if extent > 32:
            body.append(_point_ir(f"{carry} = cutlass.Float32(0)"))
        body.append(
            PointLoop(
                f"{name}_part",
                _expression(f"cutlass.range_constexpr({(extent + 31) // 32})"),
                (
                    _point_ir(
                        *warp_prefix_point(
                            name,
                            extent,
                            list(program.statements),
                            value,
                            execution=execution,
                        )
                    ),
                    PointGuard(_expression(f"{position} < {extent}"), (publication,)),
                ),
            )
        )
    elif program.kind == "serial_scan":
        body.extend(
            (
                _point_ir(f"{acc} = cutlass.Float32(0)"),
                PointLoop(
                    position,
                    _expression(f"cutlass.range({extent}, unroll=1)"),
                    (
                        _point_ir(*program.statements),
                        _point_ir(f"{acc} += {value}"),
                        publication,
                    ),
                ),
            )
        )
    else:
        assert program.kind == "sum"
        body.extend(
            (
                _point_ir(f"{acc} = cutlass.Float32(0)"),
                PointLoop(
                    f"{name}_part",
                    _expression(f"cutlass.range_constexpr({(extent + 31) // 32})"),
                    (
                        _point_ir(
                            f"{position} = {execution.thread} % 32 + {name}_part * 32"
                        ),
                        PointGuard(
                            _expression(f"{position} < {extent}"),
                            (
                                _point_ir(*program.statements),
                                _point_ir(f"{acc} += {value}"),
                            ),
                        ),
                    ),
                ),
            )
        )
        body.extend(
            _point_ir(f"{acc} += cute.arch.shuffle_sync_down({acc}, offset={offset})")
            for offset in (16, 8, 4, 2, 1)
        )
        body.append(
            PointGuard(_expression(f"{execution.thread} % 32 == 0"), (publication,))
        )
    return (
        PointLoop(
            f"{name}_vector_step",
            _expression(
                f"cutlass.range_constexpr({(vectors + owner_count - 1) // owner_count})"
            ),
            (
                _point_ir(
                    f"{vector} = {owner_index} + {name}_vector_step * {owner_count}"
                ),
                PointGuard(_expression(f"{vector} < {vectors}"), tuple(body)),
            ),
        ),
    )


def collective_body(program: CollectiveProducer) -> list[str]:
    """Compatibility source API, through the same shared action interpreter."""
    body = emit_point_actions(_collective_actions(program))
    assert body is not None
    return [ast.unparse(ast.fix_missing_locations(statement)) for statement in body]


def emit_producer_phase(
    programs: tuple[LaneProducer | CollectiveProducer, ...],
    completion: ast.stmt,
) -> tuple[ast.stmt, ...] | None:
    """Build one complete admitted phase, publishing nothing on failure.

    Completion is the caller's original join statement, not proof of readiness.
    The caller owns that proof and installs this body only after full success.
    """
    actions: list[PointAction] = []
    for program in programs:
        actions.extend(
            _lane_actions(program)
            if isinstance(program, LaneProducer)
            else _collective_actions(program)
        )
    actions.append(PointIR((completion,)))
    return emit_point_actions(tuple(actions))
