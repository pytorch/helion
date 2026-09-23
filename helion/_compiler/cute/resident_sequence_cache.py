"""Reuse a completed sequence tile in later identical, read-only loads.

The producer retains only its last tile. A runtime origin/initialized guard
therefore keeps zero-trip and multi-tile loops correct without specializing
their bounds. Exact normalized expression equality includes casts and masks.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import ReadWrites
from .persistent_branch_vec import _freeze_definition
from .resident_sequence import _VECTOR_LOADS
from .resident_sequence import _assignment
from .resident_sequence import _canonical
from .resident_sequence import _clone
from .resident_sequence import _literal_loop_extent
from .resident_sequence import _qualified
from .resident_sequence import _reads
from .resident_sequence import _valid_vector_load

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .resident_sequence import SequenceRegion


@dataclasses.dataclass(frozen=True)
class SequenceCache:
    key: str
    buffer: str
    initialized: str
    origin: str
    lane_extent: int
    vector_width: int


def memory_key(
    expression: ast.expr,
    statements: list[ast.stmt],
    *,
    origin: str,
    lane: str,
    vector: str | None,
    rename_groups: Mapping[str, str],
) -> str | None:
    # These names exist only inside the proof. Reserve every source/alias name
    # before expanding definitions, including bindings that occur later.
    used = (
        {
            node.id
            for statement in [*statements, expression]
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
        }
        | set(rename_groups)
        | set(rename_groups.values())
    )
    used.update(name for name in (origin, lane, vector) if name is not None)

    def fresh(role: str) -> str:
        name = f"__sequence_{role}"
        while name in used:
            name += "_"
        used.add(name)
        return name

    coordinates = {origin: fresh("origin"), lane: fresh("lane")}
    coordinate_roles = {coordinates[origin]: "origin", coordinates[lane]: "lane"}
    if vector is not None:
        coordinates[vector] = fresh("vector")
        coordinate_roles[coordinates[vector]] = "vector"
    definitions: dict[str, ast.expr] = {
        name: cast("ast.expr", expr_from_string(replacement))
        for name, replacement in coordinates.items()
    }
    mutable = set(rename_groups) | set(rename_groups.values())
    unknown: set[str] = set()
    for ordinal, original in enumerate(statements):
        statement = _canonical(original, rename_groups)
        name = _assignment(statement)
        if name is not None:
            if name in coordinates:
                continue
            if name in mutable:
                definitions[name] = cast("ast.expr", expr_from_string(name))
                continue
            assert isinstance(statement, ast.Assign)
            value = _freeze_definition(statement.value, definitions)
            if value is None:
                return None
            definitions[name] = value
        else:
            for written in ReadWrites.from_ast(statement).writes:
                name = fresh(f"unknown_{ordinal}_{written}")
                unknown.add(name)
                definitions[written] = cast("ast.expr", expr_from_string(name))
    value = _freeze_definition(
        cast("ast.expr", _canonical(expression, rename_groups)), definitions
    )
    if value is None or _reads(value) & (mutable | unknown):
        return None
    if not any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            or _qualified(node.func) in _VECTOR_LOADS
        )
        for node in ast.walk(value)
    ):
        return None
    # Cache policies do not change a read's value. The exact pointer, type,
    # element index, and surrounding masks remain part of the key.
    for node in ast.walk(value):
        if isinstance(node, ast.Call) and _valid_vector_load(node):
            node.func = cast("ast.expr", expr_from_string("cute.arch.load"))
            node.keywords = []
        elif isinstance(node, ast.Name) and node.id in coordinate_roles:
            # Tag coordinates only in this final, non-executable comparison
            # AST. An invalid Python identifier cannot alias any source name,
            # and the role stays stable across independently named sweeps.
            node.id = f"<sequence-coordinate:{coordinate_roles[node.id]}>"
    return ast.dump(value, include_attributes=False)


def cache_for_value(
    expression: ast.expr,
    statements: list[ast.stmt],
    region: SequenceRegion,
    buffer: str,
    initialized: str,
    origin: str,
    rename_groups: Mapping[str, str],
) -> SequenceCache | None:
    assert isinstance(region.loop.target, ast.Name)
    key = memory_key(
        expression,
        statements,
        origin=region.loop.target.id,
        lane=region.lane_var,
        vector=region.vector_var,
        rename_groups=rename_groups,
    )
    if key is None:
        return None
    return SequenceCache(
        key, buffer, initialized, origin, region.lane_extent, region.vector_width
    )


def reuse_sequence_cache(
    statement: ast.For,
    caches: list[SequenceCache],
    prefix: list[ast.stmt],
    rename_groups: Mapping[str, str],
) -> ast.For | None:
    """Clone a later lane loop into cache-hit/original branches if proven."""
    if (
        not caches
        or not isinstance(statement.target, ast.Name)
        or len(statement.body) != 1
    ):
        return None
    lane_loop = statement.body[0]
    if (
        not isinstance(lane_loop, ast.For)
        or not isinstance(lane_loop.target, ast.Name)
        or getattr(lane_loop, HELION_LANE_LOOP_VAR_ATTR, None) != lane_loop.target.id
    ):
        return None
    extent = _literal_loop_extent(lane_loop)
    loops = [item for item in lane_loop.body if isinstance(item, ast.For)]
    vector = loops[0] if len(loops) == 1 else None
    if (
        len(loops) > 1
        or vector is not None
        and (
            not isinstance(vector.target, ast.Name)
            or not isinstance(vector.iter, ast.Call)
            or _qualified(vector.iter.func) != "cutlass.range_constexpr"
        )
    ):
        return None
    width = _literal_loop_extent(vector) if vector is not None else 1
    applicable = [
        cache
        for cache in caches
        if (cache.lane_extent, cache.vector_width) == (extent, width)
    ]
    if not applicable:
        return None
    vector_name = cast("ast.Name", vector.target).id if vector is not None else None
    vector_index = lane_loop.body.index(vector) if vector is not None else 0
    outer = lane_loop.body[:vector_index] if vector is not None else []
    scalar = vector.body if vector is not None else lane_loop.body
    replacement_values: dict[int, ast.expr] = {}
    selected: list[SequenceCache] = []
    for index, item in enumerate(scalar):
        if not isinstance(item, ast.Assign) or _assignment(item) is None:
            continue
        key = memory_key(
            item.value,
            [*prefix, *outer, *scalar[:index]],
            origin=statement.target.id,
            lane=lane_loop.target.id,
            vector=vector_name,
            rename_groups=rename_groups,
        )
        matches = [cache for cache in applicable if cache.key == key]
        if not matches:
            continue
        cache = matches[0]
        if selected and (cache.initialized, cache.origin) != (
            selected[0].initialized,
            selected[0].origin,
        ):
            continue
        element = (
            lane_loop.target.id
            if vector_name is None
            else f"{lane_loop.target.id} * {width} + {vector_name}"
        )
        replacement_values[id(item)] = cast(
            "ast.expr", expr_from_string(f"{cache.buffer}[{element}]")
        )
        selected.append(cache)
    if not selected:
        return None
    copied_lane = _clone(lane_loop)
    assert isinstance(copied_lane, ast.For)
    paired_nodes = list(zip(ast.walk(lane_loop), ast.walk(copied_lane), strict=True))
    for original, copied in paired_nodes:
        if id(original) in replacement_values:
            assert isinstance(copied, ast.Assign)
            copied.value = replacement_values[id(original)]
    if vector is not None:
        # Drop only original native vector loads whose packet became dead.
        # All other setup, loads, predicates, and stores retain their order.
        reads = _reads(copied_lane)
        copied_lane.body = [
            item
            for item in copied_lane.body
            if not (
                isinstance(item, ast.Assign)
                and isinstance(item.value, ast.Call)
                and _qualified(item.value.func) in _VECTOR_LOADS
                and _assignment(item) not in reads
            )
        ]
    cache = selected[0]
    branch = statement_from_string(
        f"if {cache.initialized} and {statement.target.id} == {cache.origin}:\n    pass\nelse:\n    pass"
    )
    assert isinstance(branch, ast.If)
    branch.body = [copied_lane]
    branch.orelse = [_clone(lane_loop)]
    replacement = _clone(statement)
    assert isinstance(replacement, ast.For)
    replacement.body = [branch]
    return replacement
