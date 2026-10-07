from __future__ import annotations

import ast
from typing import NoReturn
from typing import cast

from ... import exc
from .. import tile_strategy as lanes
from ..ast_read_writes import ReadWrites

_PROTECTED_GLOBALS = {
    "cutlass",
    "cute",
    "math",
    "operator",
    "range",
    "_helion_lane_reduce",
    *(name.partition(".")[0] for name in lanes._PURE_RELOCATABLE_NAMES),
}
_SCALAR_CASTS = {
    f"cutlass.{name}"
    for name in (
        "Boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Uint8",
        "Uint16",
        "Uint32",
        "Uint64",
        "Float16",
        "BFloat16",
        "Float32",
        "Float64",
    )
}


def resolve_pruned_lane_owners(
    body: list[ast.AST], fallbacks: dict[str, tuple[str, int, int, str]]
) -> None:
    """Use a concrete reshape lane only after its synthetic loop was pruned.

    Extent symbols alone cannot establish the distribution of a value. The
    complete post-wrap body must contain neither that synthetic lane loop nor
    a reference to its coordinate. The marker's exact emitted physical group
    must match its recorded proof and the actual innermost enclosing lane.
    Other markers retain their original owner and all strict validation.
    """
    live_names = {
        node.id for top in body for node in ast.walk(top) if isinstance(node, ast.Name)
    }
    live_names.update(
        lane
        for top in body
        for node in ast.walk(top)
        if (lane := getattr(node, lanes.HELION_LANE_LOOP_VAR_ATTR, None)) is not None
    )

    def visit(node: ast.AST, owners: tuple[str, ...]) -> None:
        lane = getattr(node, lanes.HELION_LANE_LOOP_VAR_ATTR, None)
        if isinstance(node, ast.For) and lane is not None:
            if not isinstance(node.target, ast.Name) or node.target.id != lane:
                return
            owners = (*owners, lane)
        marker = lanes._is_lane_reduce_marker_assign(node)
        if marker is not None and marker.owner_lane in fallbacks:
            original = marker.owner_lane
            assert original is not None
            owner, pre, span, lane_expr = fallbacks[original]
            if (
                original not in live_names
                and owners
                and owners[-1] == owner
                and (marker.group_pre, marker.group_span, marker.group_lane_expr)
                == (pre, span, lane_expr)
                and marker.group_count == marker.group_cluster_n == 1
                and not marker.matmul_contribution
            ):
                call = lanes._find_lane_reduce_call(node)
                assert call is not None and len(call.args) >= 10
                call.args[9] = ast.Constant(value=owner)
        for child in ast.iter_child_nodes(node):
            visit(child, owners)

    for statement in body:
        visit(statement, ())


def normalize_nested_lane_reductions(
    body: list[ast.AST],
    *,
    uniform_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    rename_groups: dict[str, str],
) -> list[ast.AST]:
    """Finish an inner lane reduction before an enclosing lane consumes it.

    A nested pair of reductions initially shares the innermost scalar body.
    Only a pure suffix depending on the completed inner reduction may move to
    the directly enclosing lane. The ordinary splitter proves and lowers the
    inner reduction, and its carry checks still apply to the complete body.
    Every other ownership mismatch is left for the strict owner validator.
    """
    shadowed_globals = {
        node.id
        for top in body
        for node in ast.walk(top)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    } & _PROTECTED_GLOBALS

    def visit(
        statements: list[ast.AST],
        parent_lane: str | None,
        uniform: set[str],
        thread_axes: dict[str, frozenset[int]],
        definitions: dict[str, ast.AST],
    ) -> list[ast.AST]:
        result: list[ast.AST] = []
        for statement in statements:
            lane = getattr(statement, lanes.HELION_LANE_LOOP_VAR_ATTR, None)
            is_lane = (
                isinstance(statement, ast.For)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == lane
            )
            for field in ("body", "orelse", "finalbody"):
                children = getattr(statement, field, None)
                if isinstance(children, list) and all(
                    isinstance(child, ast.stmt) for child in children
                ):
                    setattr(
                        statement,
                        field,
                        visit(
                            children,
                            lane if is_lane and field == "body" else None,
                            set(uniform),
                            dict(thread_axes),
                            dict(definitions),
                        ),
                    )
            if is_lane:
                result.extend(
                    _normalize_one(
                        cast("ast.For", statement),
                        cast("str", lane),
                        parent_lane,
                        uniform,
                        proven_disjoint_tensor_pairs,
                        proven_tensor_stride_values,
                        thread_axes,
                        definitions,
                        rename_groups,
                        shadowed_globals,
                    )
                )
            else:
                result.append(statement)
            lanes._update_proven_uniform_names(statement, uniform)
            lanes._update_thread_axis_names(statement, thread_axes)
            lanes._update_scalar_definitions(statement, definitions)
        return result

    return visit(body, None, set(uniform_names), {}, {})


def _normalize_one(
    loop: ast.For,
    lane: str,
    parent_lane: str | None,
    uniform: set[str],
    disjoint_pairs: set[frozenset[str]],
    tensor_strides: dict[tuple[str, int], int],
    thread_axes: dict[str, frozenset[int]],
    definitions: dict[str, ast.AST],
    renames: dict[str, str],
    shadowed_globals: set[str],
) -> list[ast.AST]:
    markers = {
        index: marker
        for index, statement in enumerate(loop.body)
        if (marker := lanes._is_lane_reduce_marker_assign(statement)) is not None
    }
    foreign = {
        index: marker
        for index, marker in markers.items()
        if marker.owner_lane is not None and marker.owner_lane != lane
    }
    if not foreign:
        return [loop]

    def reject() -> NoReturn:
        raise exc.BackendUnsupported(
            "cute", "nested reduction has no complete enclosing-lane schedule"
        )

    own = {
        index: marker for index, marker in markers.items() if marker.owner_lane == lane
    }
    if (
        parent_lane is None
        or bool(shadowed_globals)
        or not own
        or loop.orelse
        or lanes._static_lane_loop_extent(loop) is None
        or any(marker.owner_lane != parent_lane for marker in foreign.values())
        or any(
            marker.owner_lane is None or marker.matmul_contribution
            for marker in markers.values()
        )
    ):
        # This is not the supported pair. Keep the original ownership
        # mismatch for the unchanged strict validator to reject.
        return [loop]
    cut = max(own) + 1
    if min(foreign) < cut:
        reject()

    # This path has no conditional definitions, repeated bindings, nested
    # control flow, or mutable state. Each marker has a simple scalar input;
    # only known pure casts may wrap it. Reads of tensor data stay in the
    # complete inner reduction slice and are never moved into the suffix.
    writes: set[str] = set()
    for index, statement in enumerate(loop.body):
        name = lanes._plain_assignment_name(statement)
        if name is None or name in writes or name == lane:
            reject()
        writes.add(name)
        marker = markers.get(index)
        if marker is None:
            if not lanes._is_proven_relocatable_assignment(
                statement, allow_load=index < cut
            ):
                reject()
        else:
            marker_call = lanes._find_lane_reduce_call(statement)
            if not isinstance(ast.parse(marker.input_name, mode="eval").body, ast.Name):
                reject()
            assert isinstance(statement, ast.Assign)
            wrapper = statement.value
            while wrapper is not marker_call:
                if (
                    not isinstance(wrapper, ast.Call)
                    or lanes._qualified_name(wrapper.func) not in _SCALAR_CASTS
                    or len(wrapper.args) != 1
                    or wrapper.keywords
                ):
                    reject()
                wrapper = wrapper.args[0]
            for node in ast.walk(statement):
                if (
                    isinstance(node, ast.Call)
                    and node is not marker_call
                    and not lanes._is_proven_relocatable_call(node, allow_load=False)
                ):
                    reject()

    original: list[ast.AST] = list(loop.body)
    phase2 = [statement for index, statement in enumerate(original) if index not in own]
    varying = {lane, *lanes._lane_varying_names(phase2, lane)}
    suffix = original[cut:]
    if any(
        (
            set(ReadWrites.from_ast(statement).reads)
            | set(ReadWrites.from_ast(statement).writes)
        )
        & varying
        for statement in suffix
    ):
        reject()

    dependent = {marker.result_var for marker in own.values()}
    for index, statement in enumerate(original[cut:], start=cut):
        reads = set(ReadWrites.from_ast(statement).reads)
        if index in foreign and foreign[index].input_name not in dependent:
            reject()
        if reads & dependent:
            dependent.update(ReadWrites.from_ast(statement).writes)

    lanes._validate_renamed_lane_carries(original, lane, set(own), renames)
    prefix_loop = lanes._clone_lane_loop_with_body(loop, original[:cut])
    lowered = lanes._split_one_lane_loop(
        prefix_loop,
        lane,
        uniform,
        disjoint_pairs,
        tensor_strides,
        thread_axes,
        definitions,
        renames,
    )
    result = [*lowered, *suffix]
    lanes._validate_owned_lane_carry_schedule(original, result, lane, set(own), renames)
    return result
