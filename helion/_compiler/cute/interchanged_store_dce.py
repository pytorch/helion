"""Remove the first-pass stores and atomics duplicated by CuTe reduction interchange."""

from __future__ import annotations

import ast
from typing import cast

from .. import tile_strategy
from ..ast_read_writes import ReadWrites
from .fuse_two_pass_loads import _is_store_call
from .fuse_two_pass_loads import _store_tensor_roots
from .fuse_two_pass_loads import _tensor_arg_roots
from .fuse_two_pass_loads import _tensor_roots_may_alias
from .persistent_branch_vec import _binding_write_roots
from .persistent_branch_vec import _definition_snapshots
from .persistent_branch_vec import _freeze_definition
from .persistent_branch_vec import _memory_load_calls
from .persistent_branch_vec import _plain_scalar_load_pointer
from .persistent_branch_vec import _plain_scalar_store_pointer


def _freeze_coverage(
    expressions: list[ast.expr], definitions: dict[str, ast.expr]
) -> list[ast.expr] | None:
    result = []
    for expression in expressions:
        frozen = _freeze_definition(expression, definitions)
        if frozen is None:
            return None
        result.append(frozen)
    return result


def _reads(statement: ast.AST) -> set[str]:
    result = set(ReadWrites.from_ast(statement).reads)
    marker = tile_strategy._is_lane_reduce_marker_assign(statement)
    if marker is not None and marker.group_lane_expr:
        result.update(
            ReadWrites.from_ast(ast.parse(marker.group_lane_expr, mode="eval")).reads
        )
    return result


def _live_in(statements: list[ast.stmt]) -> set[str]:
    defined: set[str] = set()
    live: set[str] = set()
    for statement in statements:
        live.update(_reads(statement) - defined)
        name = tile_strategy._plain_assignment_name(statement)
        if name is not None:
            defined.add(name)
    return live


def _single_store(statement: ast.AST) -> ast.Call | None:
    stores = [
        node
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and _is_store_call(node)
    ]
    if len(stores) != 1:
        return None
    store = stores[0]
    if _plain_scalar_store_pointer(
        store
    ) is None or not tile_strategy._is_isolated_store_statement(statement, store):
        return None
    if any(
        isinstance(node, ast.Call)
        and node is not store
        and not tile_strategy._is_proven_relocatable_call(node, allow_load=False)
        for node in ast.walk(statement)
    ):
        return None
    return store


def _known_statements(statements: list[ast.stmt]) -> bool:
    """Reject control transfers, opaque calls, atomics, and compound writes."""
    for statement in statements:
        # Only metadata/coordinate subscripts are known scalar reads. Other
        # subscripts may hide an aliasing access without an explicit load call.
        if any(
            isinstance(node, (ast.NamedExpr, ast.Lambda))
            or isinstance(node, ast.Subscript)
            and not _known_subscript(node)
            for node in ast.walk(statement)
        ):
            return False
        if isinstance(statement, ast.Pass) or _single_store(statement) is not None:
            continue
        if tile_strategy._is_lane_reduce_marker_assign(statement) is not None:
            continue
        if not tile_strategy._is_proven_relocatable_assignment(
            statement, allow_load=True, allow_reduction=True
        ):
            return False
    return True


def _known_subscript(node: ast.Subscript) -> bool:
    is_stride = (
        isinstance(node.value, ast.Attribute)
        and node.value.attr == "stride"
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == "layout"
        and isinstance(node.value.value.value, ast.Name)
    )
    is_coordinate = isinstance(node.value, ast.Call) and tile_strategy._qualified_name(
        node.value.func
    ) in {"cute.arch.thread_idx", "cute.arch.block_idx"}
    return is_stride or is_coordinate


def _coordinate_expression(
    expression: ast.expr, mutable_names: set[str], iteration_names: set[str]
) -> bool:
    """An address/bound may use immutable inputs and the two iteration IDs."""
    if set(ReadWrites.from_ast(expression).reads) & (mutable_names - iteration_names):
        return False
    for node in ast.walk(expression):
        if isinstance(node, ast.Call):
            if tile_strategy._qualified_name(node.func) in {
                "range",
                "cutlass.range_constexpr",
            }:
                continue
            if not tile_strategy._is_proven_relocatable_call(node, allow_load=False):
                return False
        elif isinstance(node, ast.Subscript):
            if not _known_subscript(node):
                return False
    return True


def _coverage_expressions(statement: ast.stmt, store: ast.Call) -> list[ast.expr]:
    pointer = _plain_scalar_store_pointer(store)
    assert pointer is not None
    return [
        pointer,
        *cast(
            "list[ast.expr]",
            tile_strategy._store_enclosing_predicates(statement, store),
        ),
    ]


def eliminate_interchanged_stores(
    lane_loop: ast.For,
    serial_loop: ast.For,
    final_nest: list[ast.AST],
    candidate_indices: set[int],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    protected_names: set[str],
) -> int:
    """Prune first-pass stores that an interchanged final pass overwrites.

    Restrict the proof to scalar stores in a rectangular pair of loops. Match
    the cloned final store, expand assignment-time address/mask definitions,
    and require immutable, identical coverage. Every load and other store must
    be disjoint from the destination using cache-safe compiler facts. Unknown
    memory operations and loop-carried coverage expressions leave both passes
    unchanged. Producers are removed only when the final pass redefines their
    scalar bindings before use and no remaining first-pass operation reads them.
    Names with pending compiler aliases remain protected until final renaming.
    """
    if not candidate_indices or lane_loop.orelse or serial_loop.orelse:
        return 0
    if not isinstance(lane_loop.target, ast.Name) or not isinstance(
        serial_loop.target, ast.Name
    ):
        return 0
    if tile_strategy._static_lane_loop_extent(lane_loop) is None:
        return 0
    if (
        not isinstance(serial_loop.iter, ast.Call)
        or tile_strategy._qualified_name(serial_loop.iter.func)
        not in {"range", "cutlass.range_constexpr"}
        or not 1 <= len(serial_loop.iter.args) <= 3
        or serial_loop.iter.keywords
    ):
        return 0
    if not final_nest or not isinstance(final_nest[-1], ast.For):
        return 0
    final_serial = final_nest[-1]
    if (
        final_serial.orelse
        or not final_serial.body
        or not isinstance(final_serial.body[-1], ast.For)
        or ast.unparse(final_serial.target) != ast.unparse(serial_loop.target)
        or ast.unparse(final_serial.iter) != ast.unparse(serial_loop.iter)
    ):
        return 0
    final_lane = final_serial.body[-1]
    if (
        final_lane.orelse
        or ast.unparse(final_lane.target) != ast.unparse(lane_loop.target)
        or ast.unparse(final_lane.iter) != ast.unparse(lane_loop.iter)
    ):
        return 0

    serial_index = lane_loop.body.index(serial_loop)
    prefix = lane_loop.body[:serial_index]
    suffix = lane_loop.body[serial_index + 1 :]
    original_flat = [*prefix, *serial_loop.body]
    final_prefix = cast("list[ast.stmt]", final_nest[:-1])
    final_flat = [*final_prefix, *final_serial.body[:-1], *final_lane.body]
    if not _known_statements([*original_flat, *suffix, *final_flat]):
        return 0
    mutable_names = _binding_write_roots(lane_loop)
    iteration_names = {lane_loop.target.id, serial_loop.target.id}
    body_writes = set(ReadWrites.from_list(serial_loop.body).writes)
    carried_names = body_writes & _live_in(serial_loop.body)

    # Freeze prefixes at the loop entry. A suffix/backedge mutation of a free
    # bound is caught by mutable_names; lane-dependent bounds are rejected too.
    before_snapshots = _definition_snapshots([*prefix, ast.Pass()])
    after_snapshots = _definition_snapshots([*final_prefix, ast.Pass()])
    if before_snapshots is None or after_snapshots is None:
        return 0
    before_bounds = before_snapshots[-1]
    after_bounds = after_snapshots[-1]
    before_bound = _freeze_definition(serial_loop.iter, before_bounds)
    after_bound = _freeze_definition(final_serial.iter, after_bounds)
    if before_bound is None or after_bound is None:
        return 0
    bounds = [before_bound, after_bound]
    if any(not _coordinate_expression(bound, mutable_names, set()) for bound in bounds):
        return 0
    if ast.unparse(bounds[0]) != ast.unparse(bounds[1]):
        return 0
    if (
        tile_strategy._dependency_names(
            cast("list[ast.AST]", original_flat), _reads(serial_loop.iter)
        )
        & protected_names
    ):
        return 0

    original_definitions = _definition_snapshots(original_flat)
    final_definitions = _definition_snapshots(final_flat)
    if original_definitions is None or final_definitions is None:
        return 0
    writes = [
        node
        for node in ast.walk(lane_loop)
        if isinstance(node, ast.Call) and _is_store_call(node)
    ]
    loads = [
        *_memory_load_calls(lane_loop),
        *(load for statement in final_flat for load in _memory_load_calls(statement)),
    ]
    tensor_names = {
        node.value.id
        for node in ast.walk(lane_loop)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }
    # Alias facts describe kernel argument bindings. A local reassignment or a
    # pending scalar rename cannot inherit the original argument's storage proof.
    if tensor_names & (mutable_names | protected_names):
        return 0
    removed: set[int] = set()
    for index in candidate_indices:
        statement = serial_loop.body[index]
        store = _single_store(statement)
        if store is None:
            continue
        roots = _store_tensor_roots(store, tensor_names)
        if roots is None or len(roots) != 1:
            continue
        if any(
            _tensor_roots_may_alias(
                roots,
                _store_tensor_roots(other, tensor_names),
                proven_disjoint_tensor_pairs,
            )
            for other in writes
            if other is not store
        ):
            continue
        if any(
            (pointer := _plain_scalar_load_pointer(load)) is None
            or _tensor_roots_may_alias(
                roots,
                _tensor_arg_roots(pointer, tensor_names),
                proven_disjoint_tensor_pairs,
            )
            for load in loads
        ):
            continue
        matches = [
            (position, other, final_store)
            for position, other in enumerate(final_flat)
            if ast.unparse(other) == ast.unparse(statement)
            and (final_store := _single_store(other)) is not None
        ]
        if len(matches) != 1:
            continue
        final_index, final_statement, final_store = matches[0]
        original_coverage = _coverage_expressions(statement, store)
        final_coverage = _coverage_expressions(final_statement, final_store)
        dependency_names = tile_strategy._dependency_names(
            cast("list[ast.AST]", original_flat),
            {name for expression in original_coverage for name in _reads(expression)},
        )
        if dependency_names & (carried_names | protected_names):
            continue
        before = _freeze_coverage(
            original_coverage, original_definitions[len(prefix) + index]
        )
        after = _freeze_coverage(final_coverage, final_definitions[final_index])
        if before is None or after is None:
            continue
        if not all(
            _coordinate_expression(expression, mutable_names, iteration_names)
            for expression in [*before, *after]
        ):
            continue
        if [ast.unparse(expression) for expression in before] != [
            ast.unparse(expression) for expression in after
        ]:
            continue
        removed.add(index)

    if not removed:
        return 0
    serial_loop.body = [
        statement
        for index, statement in enumerate(serial_loop.body)
        if index not in removed
    ]

    # The final pass executes the same nonempty Cartesian iteration set whenever
    # the first pass executes. These unconditional definitions therefore kill
    # the corresponding first-pass bindings, including after either loop exits.
    _prune_redefined_bindings(
        lane_loop, serial_loop, _redefined_names(final_flat, protected_names)
    )
    if not serial_loop.body:
        serial_loop.body = [ast.Pass()]
    return len(removed)


def _redefined_names(final_flat: list[ast.stmt], protected_names: set[str]) -> set[str]:
    """Names the final pass binds unconditionally before any read of its own."""
    return {
        name
        for statement in final_flat
        if (name := tile_strategy._plain_assignment_name(statement)) is not None
    } - (_live_in(final_flat) | protected_names)


def _prune_redefined_bindings(
    lane_loop: ast.For, owner: ast.For, redefined: set[str]
) -> None:
    """Drop ``owner``'s plain bindings that the final pass redefines and no
    statement of the first pass (``lane_loop``) still reads."""
    while True:
        reads = ReadWrites.from_ast(lane_loop).reads
        dead = [
            statement
            for statement in owner.body
            if (name := tile_strategy._plain_assignment_name(statement)) is not None
            and name in redefined
            and reads.get(name, 0) == ReadWrites.from_ast(statement).reads.get(name, 0)
            and tile_strategy._is_proven_relocatable_assignment(
                statement, allow_load=True
            )
        ]
        if not dead:
            break
        owner.body = [statement for statement in owner.body if statement not in dead]


def _is_atomic_write(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr.startswith("atomic_")
    return isinstance(func, ast.Name) and (
        func.id.startswith(("atomic_", "_cute_atomic_"))
        or func.id == "_cute_red_add_f32_vec"
    )


def atomic_write_calls(statement: ast.AST) -> list[ast.Call]:
    """The atomic read-modify-writes among ``statement``'s memory writes."""
    return [
        call
        for call in tile_strategy._memory_write_calls(statement)
        if _is_atomic_write(call)
    ]


def eliminate_interchanged_atomics(
    lane_loop: ast.For,
    serial_loop: ast.For,
    final_nest: list[ast.AST],
    candidates: list[ast.AST],
    protected_names: set[str],
) -> int:
    """Drop the first pass's atomic consumers that the final pass applies in full.

    The first pass keeps each reduction consumer over the marker's raw
    per-lane input.  A store there writes a partial value the final pass
    overwrites; an atomic read-modify-write would instead add every lane's
    partial on top of the full reduction the final pass applies once per
    serial iteration (a column sum over a lane-looped row tile added into
    ``out[column]`` came out exactly doubled).  A candidate is dropped when
    its one memory write is an atomic the statement does nothing else around
    and the final pass holds exactly one clone of it; the number dropped is
    returned so the caller can decline a nest it could not make exact.  The
    first-pass bindings the final pass redefines and nothing left in the
    first pass reads are pruned afterwards, the serial loop goes when its
    body is empty and the lane loop's body may end up empty too, which the
    caller takes as the whole first pass being dead.
    """
    if not candidates or lane_loop.orelse or serial_loop.orelse:
        return 0
    if not final_nest or not isinstance(final_nest[-1], ast.For):
        return 0
    final_serial = final_nest[-1]
    if (
        final_serial.orelse
        or not final_serial.body
        or not isinstance(final_serial.body[-1], ast.For)
        or ast.unparse(final_serial.target) != ast.unparse(serial_loop.target)
        or ast.unparse(final_serial.iter) != ast.unparse(serial_loop.iter)
    ):
        return 0
    final_lane = final_serial.body[-1]
    if (
        final_lane.orelse
        or ast.unparse(final_lane.target) != ast.unparse(lane_loop.target)
        or ast.unparse(final_lane.iter) != ast.unparse(lane_loop.iter)
    ):
        return 0
    final_prefix = cast("list[ast.stmt]", final_nest[:-1])
    final_flat = [*final_prefix, *final_serial.body[:-1], *final_lane.body]
    final_texts = [ast.unparse(statement) for statement in final_flat]
    removed: list[ast.AST] = []
    for statement in candidates:
        if statement not in serial_loop.body:
            continue
        writes = tile_strategy._memory_write_calls(statement)
        if (
            len(writes) != 1
            or not _is_atomic_write(writes[0])
            or not tile_strategy._is_isolated_store_statement(statement, writes[0])
            or final_texts.count(ast.unparse(statement)) != 1
        ):
            continue
        removed.append(statement)
    if not removed:
        return 0
    serial_loop.body = [
        statement for statement in serial_loop.body if statement not in removed
    ]

    # Both nests run the lane loop's static extent times the same serial
    # bounds, which nothing in the first pass rebinds, so the final pass's
    # unconditional definitions kill the first pass's bindings
    # (``eliminate_interchanged_stores``).
    if tile_strategy._static_lane_loop_extent(lane_loop) is None or (
        _reads(serial_loop.iter) & set(ReadWrites.from_ast(lane_loop).writes)
    ):
        if not serial_loop.body:
            serial_loop.body = [ast.Pass()]
        return len(removed)
    redefined = _redefined_names(final_flat, protected_names)
    _prune_redefined_bindings(lane_loop, serial_loop, redefined)
    if not serial_loop.body:
        lane_loop.body = [
            statement for statement in lane_loop.body if statement is not serial_loop
        ]
        _prune_redefined_bindings(lane_loop, lane_loop, redefined)
    return len(removed)
