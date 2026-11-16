"""The lane-split passes over a rendered CuTe root body.

Role: device-body pass.  ``split_lane_loop_reductions`` rewrites single-pass
lane loops that contain ``_helion_lane_reduce`` markers into the two-pass
accumulate / finalize / consume structure (register stash, register tile,
lane-invariant tail, running sums, restored per-lane markers);
``interchange_lane_outside_serial_reductions`` turns a ``for LANE: for MB``
nest whose serial loop carries markers into the accumulator / reduction nest
pair; ``sink_lane_loops_into_uniform_branches`` and ``_lift_lane_invariant_if``
move lane loops across block-uniform branches.  ``GenerateAST`` reaches them
through ``Backend.lower_owned_lane_loop`` (``root_passes.py``).
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import operator
import re
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import NoReturn
from typing import cast
import weakref

from .... import exc
from ...ast_extension import create
from ...ast_extension import expr_from_string
from ...ast_extension import statement_from_string
from ...ast_read_writes import HELION_ACCESS_REGIONS_ATTR
from ...ast_read_writes import HELION_BLOCK_UNIFORM_ATTR
from ...ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ...ast_read_writes import HELION_VEC_LANE_OF_ATTR
from ...ast_read_writes import ReadWrites
from ...device_function import DeviceFunction
from ...host_function import NoCurrentFunction
from ...tile_strategy import _clone_lane_loop_with_body
from .._ast_pass_utils import _CHUNK_REDUCTION_HELPERS
from .._ast_pass_utils import _FOLD_DOT_HELPER
from .._ast_pass_utils import _backward_slice
from .._ast_pass_utils import _canonical_name
from .._ast_pass_utils import _is_proven_relocatable_assignment
from .._ast_pass_utils import _is_proven_relocatable_call
from .._ast_pass_utils import _self_addend
from .._ast_pass_utils import assigned_name
from .._ast_pass_utils import call_path_str
from .._ast_pass_utils import deep_clone
from .._ast_pass_utils import exposed_reads
from .._ast_pass_utils import names_written
from .._ast_pass_utils import resolve_alias
from .._ast_pass_utils import tensor_roots_may_alias
from ..register_tile_admission import RegisterTileOptions
from ..register_tile_admission import RegisterTileUnsupported
from .markers import _CUTE_THREAD_VARYING_INTRINSICS
from .markers import _CUTE_UNIFORM_GLOBAL_NAMES
from .markers import HELION_VECTOR_REDUCTION_OWNER_ATTR
from .markers import _combine_expr
from .markers import _create_lane_loop
from .markers import _dtype_ctor_from_identity
from .markers import _finalize_lane_reduce_marker
from .markers import _find_lane_reduce_call
from .markers import _int_literal
from .markers import _is_constexpr_lane_iter
from .markers import _is_lane_reduce_marker_assign
from .markers import _lane_loop_extent
from .markers import _LaneReduceMarker
from .markers import _static_lane_loop_extent
from .owner_guards import _guard_stmt_with_owner
from .owner_guards import _has_impure_inline_asm
from .owner_guards import _has_observable_memory_write
from .owner_guards import _has_side_effect
from .owner_guards import _is_persistent_branch_vec_store
from .owner_guards import _is_proven_uniform_name
from .owner_guards import _lane_reduction_owner_exprs_for_statement
from .owner_guards import _loop_body_scalar_definitions
from .owner_guards import _loop_invariant_scalar_definitions
from .owner_guards import _statement_provenance_before
from .owner_guards import _update_proven_uniform_names
from .owner_guards import _update_scalar_definitions
from .owner_guards import _update_thread_axis_names
from .owner_guards import _validated_idempotent_store
from .statements import _analyze
from .statements import _depends
from .statements import _is_device_loop
from .statements import runs_once_outside_lane

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence


def split_lane_loop_reductions(
    body: list[ast.AST],
    *,
    uniform_names: set[str] | None = None,
    proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
    proven_tensor_stride_values: dict[tuple[str, int], int] | None = None,
    thread_axis_names: dict[str, frozenset[int]] | None = None,
    scalar_definitions: dict[str, ast.AST] | None = None,
    rename_groups: dict[str, str] | None = None,
    running_sums: set[str] | None = None,
    register_tile_options: RegisterTileOptions | None = None,
    cta_uniform: bool = True,
) -> list[ast.AST]:
    """Rewrite single-pass lane loops that contain ``_helion_lane_reduce``
    markers into the two-pass accumulate / finalize / consume structure.

    Operates bottom-up so nested lane loops are handled before their parents.
    Lane loops without markers are returned unchanged (their inner statements
    are still recursed into so nested markers are processed).  The split
    hoists the statements that read no lane out of its per-lane passes; an
    atomic left inside a lane loop it is uniform along is then issued at the
    loop's first lane (``pin_atomics_left_in_lane_loops``), as the lane-loop
    distribution, which leaves a body with markers to this pass, did not.
    """
    if not any(_find_lane_reduce_call(stmt) is not None for stmt in body):
        return body
    from .placement import pin_atomics_left_in_lane_loops

    result = _split_lane_loop_reductions(
        body,
        uniform_names=uniform_names,
        proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
        proven_tensor_stride_values=proven_tensor_stride_values,
        thread_axis_names=thread_axis_names,
        scalar_definitions=scalar_definitions,
        rename_groups=rename_groups,
        running_sums=running_sums,
        register_tile_options=register_tile_options,
        cta_uniform=cta_uniform,
    )
    pin_atomics_left_in_lane_loops(result)
    return result


def _split_lane_loop_reductions(
    body: list[ast.AST],
    *,
    uniform_names: set[str] | None = None,
    proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
    proven_tensor_stride_values: dict[tuple[str, int], int] | None = None,
    thread_axis_names: dict[str, frozenset[int]] | None = None,
    scalar_definitions: dict[str, ast.AST] | None = None,
    rename_groups: dict[str, str] | None = None,
    running_sums: set[str] | None = None,
    register_tile_options: RegisterTileOptions | None = None,
    cta_uniform: bool = True,
) -> list[ast.AST]:
    if not any(_find_lane_reduce_call(stmt) is not None for stmt in body):
        return body

    # A preceding conditional exit can remove CTA participants even when the
    # snapshot itself is not nested under that condition.
    cta_uniform = cta_uniform and not any(
        isinstance(node, (ast.Break, ast.Continue, ast.Return))
        for stmt in body
        for node in ast.walk(stmt)
    )

    proven_uniform = set(_CUTE_UNIFORM_GLOBAL_NAMES)
    if uniform_names is not None:
        proven_uniform.update(uniform_names)
    proven_thread_axes = dict(thread_axis_names or {})
    proven_scalar_definitions = dict(scalar_definitions or {})
    new_body: list[ast.AST] = []
    for stmt in body:
        new_body.extend(
            _split_stmt_lane_reductions(
                stmt,
                proven_uniform,
                proven_disjoint_tensor_pairs or set(),
                proven_tensor_stride_values or {},
                proven_thread_axes,
                proven_scalar_definitions,
                rename_groups or {},
                running_sums or set(),
                register_tile_options,
                cta_uniform,
            )
        )
        _update_proven_uniform_names(stmt, proven_uniform)
        _update_thread_axis_names(stmt, proven_thread_axes)
        _update_scalar_definitions(stmt, proven_scalar_definitions, rename_groups)
    return new_body


def _split_stmt_lane_reductions(
    stmt: ast.AST,
    uniform_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
    running_sums: set[str] | None = None,
    register_tile_options: RegisterTileOptions | None = None,
    cta_uniform: bool = True,
) -> list[ast.AST]:
    # Recurse into any statement-list-bearing fields first so nested lane
    # loops are rewritten before the enclosing one.
    body_definitions = _loop_body_scalar_definitions(
        stmt, scalar_definitions, thread_axis_names, rename_groups
    )
    exit_definitions = _loop_invariant_scalar_definitions(
        stmt, scalar_definitions, rename_groups
    )
    child_uniform = cta_uniform and _uniform_lane_scope(stmt, uniform_names)
    child_names = set(uniform_names)
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(
                stmt,
                field,
                _split_lane_loop_reductions(
                    old,
                    uniform_names=child_names,
                    proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                    proven_tensor_stride_values=proven_tensor_stride_values,
                    thread_axis_names=dict(thread_axis_names),
                    scalar_definitions=dict(
                        body_definitions if field == "body" else exit_definitions
                    ),
                    rename_groups=rename_groups,
                    running_sums=running_sums,
                    register_tile_options=register_tile_options,
                    cta_uniform=child_uniform,
                ),
            )
    lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
    if (
        lane_var is None
        or not isinstance(stmt, ast.For)
        or not isinstance(stmt.target, ast.Name)
        or stmt.target.id != lane_var
    ):
        return [stmt]
    return _split_one_lane_loop(
        stmt,
        lane_var,
        uniform_names,
        proven_disjoint_tensor_pairs,
        proven_tensor_stride_values,
        thread_axis_names,
        body_definitions,
        rename_groups,
        running_sums,
        register_tile_options,
        cta_uniform,
    )


@dataclasses.dataclass(frozen=True)
class _LaneRunningSum:
    """One compiler-created scalar self-add kept in the full first pass."""

    name: str
    update_index: int
    input_name: str
    phase1_indices: tuple[int, ...]
    iterator: str


def _certify_lane_running_sum(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    rename_groups: dict[str, str],
    running_sums: set[str] | None,
) -> _LaneRunningSum | None:
    """Prove the narrow mixed-matmul/owned-reduction composition.

    A name spelling is not provenance. Only the scalar accumulators recorded
    by matmul fallback may enter this path, and that provenance is necessary
    but insufficient: one unconditional self-add must have a complete, pure
    input slice independent of every marker and all other carried bindings.
    The marker inputs must also be independent of this running sum. Multiple
    definitions, forward/conditional definitions, collectives and unknown
    effects require another schedule and remain unsupported here.
    """
    from ...ast_read_writes import ReadWrites
    from ...ast_read_writes import ast_rename

    if not running_sums or any(marker.owner_lane is None for _, marker in markers):
        return None
    body = [deep_clone(statement) for statement in loop.body]
    for statement in body:
        ast_rename(statement, rename_groups)
    writes = [
        {
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        for statement in body
    ]
    all_writes = set().union(*writes)
    effects = _ordered_block_reads_writes(cast("list[ast.stmt]", body))
    candidates = (
        {_canonical_name(name, rename_groups) for name in running_sums}
        & effects.reads_before_write
        & all_writes
    )
    if not candidates:
        return None
    if _static_lane_loop_extent(loop) is None or loop.orelse:
        raise exc.BackendUnsupported(
            "cute", "certified lane running sum requires a complete static lane loop"
        )
    if len(candidates) != 1:
        raise exc.BackendUnsupported(
            "cute", "certified lane running sum requires one scalar accumulator"
        )
    name = next(iter(candidates))
    positions = [index for index, names in enumerate(writes) if name in names]
    if len(positions) != 1:
        raise exc.BackendUnsupported(
            "cute", "certified lane running sum requires one unconditional self-add"
        )
    update_index = positions[0]
    update = body[update_index]
    addend = _self_addend(update, name)
    raw_update = loop.body[update_index]
    raw_addend = _self_addend(raw_update, name)
    if (
        assigned_name(update) != name
        or writes[update_index] != {name}
        or assigned_name(raw_update) != name
        or not isinstance(addend, ast.Name)
        or not isinstance(raw_addend, ast.Name)
        or addend.id not in all_writes
        or any(name in ReadWrites.from_ast(stmt).reads for stmt in body[:update_index])
    ):
        raise exc.BackendUnsupported(
            "cute", "certified lane running sum requires one unconditional self-add"
        )

    marker_indices = {index for index, _ in markers}
    marker_results = {
        _canonical_name(marker.result_var, rename_groups) for _, marker in markers
    }
    for index, statement in enumerate(body):
        if index in marker_indices:
            continue
        if _is_proven_relocatable_assignment(statement, allow_load=True):
            continue
        if _validated_idempotent_store(statement) is None:
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum has an unproved body effect"
            )

    selected: set[int] = {update_index}

    def require_input_slice(root: str, before: int) -> None:
        if root == name or root in marker_results:
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum input depends on a carry or marker"
            )
        if root == lane_var or root not in all_writes:
            return
        producers = [index for index, names in enumerate(writes) if root in names]
        if len(producers) != 1:
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum has no complete input slice"
            )
        index = producers[0]
        statement = body[index]
        if (
            index >= before
            or index in marker_indices
            or writes[index] != {root}
            or assigned_name(statement) != root
            or sum(
                isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                for node in ast.walk(statement)
            )
            != 1
            or not _is_proven_relocatable_assignment(statement, allow_load=True)
            or _contains_unduplicatable_op(statement)
        ):
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum has no complete input slice"
            )
        if index in selected:
            return
        selected.add(index)
        for dependency in ReadWrites.from_ast(statement).reads:
            require_input_slice(dependency, index)

    require_input_slice(addend.id, update_index)
    for index, marker in markers:
        require_input_slice(_canonical_name(marker.input_name, rename_groups), index)
    return _LaneRunningSum(
        name=name,
        update_index=update_index,
        input_name=raw_addend.id,
        phase1_indices=tuple(sorted(selected)),
        iterator=ast.dump(loop.iter),
    )


def _lane_invariant_tail_after_consume(
    statements: list[ast.AST],
    *,
    invariant: set[int],
    accumulated: set[int],
    consumed: set[int],
    rename_groups: dict[str, str],
) -> set[int]:
    """The lane-invariant tail statements that must follow the consume pass.

    A lane split runs the lane-varying statements of ``statements`` inside
    its passes (``accumulated`` in an accumulate pass, ``consumed`` in the
    final consume pass) and each ``invariant`` statement once, between the
    passes.  That keeps the body's order only where the passes' accesses are
    independent of the invariant statement.  One that must stay after a
    consumed statement (a store into the tensor a per-lane store fills, a
    load or an atomic of that tensor) is emitted after the consume pass
    instead, together with every later invariant statement ordered after
    it; one that must also stay before a consumed statement, or before an
    accumulated one, has no position and the split is rejected.  The order
    is the lane-loop distribution's (``cute/lane_loops/statements.py``):
    register and tensor dependencies, with unknown effects and non-relaxed
    atomics ordered against every memory access.
    """
    analyzed = {
        index: _analyze(index, statements[index], rename_groups)
        for index in invariant | accumulated | consumed
    }
    # A statement with unknown effects may mutate in place any local name it
    # reads or writes; a register statement sharing such a name with it keeps
    # its side of it (``_prepare`` of the distribution does the same).
    defined = {
        rename_groups.get(name, name)
        for statement in statements
        for name in ReadWrites.from_ast(statement).writes
    }
    for statement in analyzed.values():
        if statement.pinned:
            statement.touched = (statement.reads | statement.writes) & defined
    after: set[int] = set()
    for index in sorted(invariant):
        statement = analyzed[index]
        source = ast.unparse(statements[index]).partition("\n")[0]
        # A statement the accumulate pass re-runs is ordered inside that pass;
        # only its tail copy is placed here.
        if index not in accumulated and any(
            _depends(statement, analyzed[later])
            for later in accumulated
            if later > index
        ):
            raise exc.BackendUnsupported(
                "cute",
                "a lane-invariant statement is ordered before the accumulate pass "
                f"of a lane reduction: {source}",
            )
        follows = any(
            _depends(analyzed[earlier], statement)
            for earlier in consumed | after
            if earlier < index
        )
        precedes = any(
            _depends(statement, analyzed[later]) for later in consumed if later > index
        )
        if follows and precedes:
            raise exc.BackendUnsupported(
                "cute",
                "a lane-invariant statement is ordered between per-lane statements "
                f"of a lane reduction's consume pass: {source}",
            )
        if follows:
            after.add(index)
    return after


def _uniform_lane_scope(stmt: ast.AST, uniform_names: set[str]) -> bool:
    """Whether a nested snapshot barrier is reached by the entire CTA."""
    from ...ast_read_writes import ReadWrites

    if isinstance(stmt, ast.If):
        return bool(getattr(stmt, HELION_BLOCK_UNIFORM_ATTR, False)) or (
            not any(isinstance(node, ast.Call) for node in ast.walk(stmt.test))
            and all(
                _is_proven_uniform_name(name, uniform_names)
                for name in ReadWrites.from_ast(stmt.test).reads
            )
        )
    # An enclosing lane has unsnapshotted elements; a device loop needs a
    # separate backedge memory-order proof. Neither enters this bounded path.
    return False


def _split_impure_producer(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    impure_indices: set[int],
    rename_groups: dict[str, str],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    *,
    cta_uniform: bool,
) -> list[ast.AST] | None:
    """Snapshot a pure input prefix before a single opaque reduction producer.

    All CTA threads finish their loads before any opaque call can overwrite a
    later lane's input. Each call then runs once, followed by a second CTA join
    before the tail may store. Typed constexpr lists retain inputs and live-outs
    without guessing a scalar or packet dtype. No opaque epilogue,
    dependent reduction, loop carry, or post-effect memory read is admitted.
    """
    from ...ast_read_writes import ReadWrites

    extent = _static_lane_loop_extent(loop)
    if (
        not cta_uniform
        or extent is None
        or not 1 <= extent <= 256
        or loop.orelse
        or len(markers) != 1
        or len(impure_indices) != 1
    ):
        return None
    marker_index, marker = markers[0]
    effect_index = next(iter(impure_indices))
    if effect_index >= marker_index or marker.owner_lane != lane_var:
        return None
    body: list[ast.AST] = list(loop.body)
    effect = body[effect_index]
    if (
        assigned_name(effect) is None
        or not isinstance(effect, ast.Assign)
        or not isinstance(effect.value, ast.Call)
        or call_path_str(effect.value.func) != "_cute_inline_asm_elementwise"
        # The generated operands are materialized scalar values. A nested load
        # here would bypass the prefix snapshot and run after the CTA join.
        or len(effect.value.args) != 1
        or not isinstance(effect.value.args[0], ast.Tuple)
        or not all(
            isinstance(operand, (ast.Name, ast.Constant))
            for operand in effect.value.args[0].elts
        )
        or sum(
            _has_impure_inline_asm(node)
            for node in ast.walk(effect.value)
            if isinstance(node, ast.Call)
        )
        != 1
    ):
        return None
    prefix = body[:effect_index]
    middle = body[effect_index + 1 : marker_index]
    tail = body[marker_index + 1 :]
    dtype = next(kw.value for kw in effect.value.keywords if kw.arg == "dtype")
    for stmt in (*middle, *tail):
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Subscript):
                continue
            # Tuple extraction from this assembly result and immutable tensor
            # layout metadata are known register/host facts. An arbitrary
            # tensor subscript could instead load shared memory after the write.
            path = call_path_str(node.value)
            tuple_result = (
                isinstance(node.value, ast.Name)
                and node.value.id == assigned_name(effect)
                and isinstance(dtype, ast.Tuple)
                and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int
                and 0 <= node.slice.value < len(dtype.elts)
            )
            metadata = path is not None and path.endswith(
                (".layout.stride", ".layout.shape")
            )
            thread_index = isinstance(node.value, ast.Call) and call_path_str(
                node.value.func
            ) in {"cute.arch.thread_idx", "cute.arch.block_idx"}
            if not (tuple_result or metadata or thread_index):
                return None
    if not all(
        _is_proven_relocatable_assignment(stmt, allow_load=True) for stmt in prefix
    ):
        return None
    if not all(
        _is_proven_relocatable_assignment(stmt, allow_load=False) for stmt in middle
    ):
        return None

    # Single definitions, including the final rename groups, exclude loop
    # carries and source values replaced by a later spelling of the same name.
    defined: set[str] = set()
    read_before_write: set[str] = set()
    for stmt in body[:marker_index]:
        name = assigned_name(stmt)
        assert name is not None
        canonical = rename_groups.get(name, name)
        read_before_write.update(
            {rename_groups.get(read, read) for read in ReadWrites.from_ast(stmt).reads}
            - defined
        )
        if canonical in defined:
            return None
        defined.add(canonical)
    if defined & read_before_write:
        return None
    if marker.result_var in defined:
        return None
    # This bounded path preserves the actual scalar names in its lists. Phi
    # aliases need an explicit carried-value schedule, not a list-name guess.
    if any(
        rename_groups.get(node.id, node.id) != node.id
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
    ):
        return None
    varying = {lane_var} | _lane_varying_names(
        body[:marker_index], lane_var, rename_groups
    )
    if not (
        {
            rename_groups.get(name, name)
            for arg in effect.value.args
            for name in ReadWrites.from_ast(arg).reads
        }
        & varying
    ):
        return None
    stores = []
    for stmt in tail:
        if not (
            _is_proven_relocatable_assignment(stmt, allow_load=False)
            or _validated_idempotent_store(stmt) is not None
        ):
            return None
        if _validated_idempotent_store(stmt) is not None:
            stores.append(stmt)
        if (name := assigned_name(stmt)) is not None:
            if name in defined or name == marker.result_var:
                return None
            defined.add(name)
            if set(ReadWrites.from_ast(stmt).reads) & varying:
                varying.add(name)
    if len(stores) != 1:
        return None
    tail_reads = set().union(*(ReadWrites.from_ast(stmt).reads for stmt in tail))
    per_lane_tail = bool(tail_reads & varying)
    if per_lane_tail:
        store_reads = _store_value_and_addr_reads(stores[0])
        if store_reads is None or not store_reads[1] & varying:
            return None
    input_slice, _ = _backward_slice(
        body[:marker_index], {marker.input_name}, rename_groups
    )
    if effect_index not in input_slice:
        return None

    occupied = {node.id for node in ast.walk(loop) if isinstance(node, ast.Name)}
    occupied.update(rename_groups)
    occupied.update(rename_groups.values())
    try:
        function = DeviceFunction.current()
    except NoCurrentFunction:
        function = None

    def fresh(hint: str) -> str:
        while True:
            name = (
                function.new_var(hint)
                if function is not None
                else f"{hint}_{_next_lane_stash_uid()}"
            )
            if name not in occupied:
                occupied.add(name)
                return name

    later_reads = set().union(
        *(ReadWrites.from_ast(stmt).reads for stmt in body[effect_index:])
    )
    stashes = {
        name: fresh("opaque_input_stash")
        for stmt in prefix
        if (name := assigned_name(stmt)) in later_reads
    }
    produced_stashes = {
        name: fresh("opaque_result_stash")
        for stmt in body[effect_index:marker_index]
        if (name := assigned_name(stmt)) in tail_reads
    }
    result: list[ast.AST] = [
        statement_from_string(f"{stash} = []")
        for stash in (*stashes.values(), *produced_stashes.values())
    ]

    def unrolled(statements: list[ast.AST]) -> ast.For:
        clone = _clone_lane_loop_with_body(loop, statements)
        clone.iter = cast(
            "ast.expr", expr_from_string(f"cutlass.range_constexpr({extent})")
        )
        return clone

    snapshot = [deep_clone(stmt) for stmt in prefix]
    snapshot.extend(
        statement_from_string(f"{stash}.append({name})")
        for name, stash in stashes.items()
    )
    result.extend(
        [unrolled(snapshot), statement_from_string("cute.arch.sync_threads()")]
    )
    acc = fresh("opaque_lane_acc")
    result.append(statement_from_string(f"{acc} = {marker.identity_expr}"))
    produce: list[ast.AST] = [
        statement_from_string(f"{name} = {stash}[{lane_var}]")
        for name, stash in stashes.items()
    ]
    produce.extend(deep_clone(stmt) for stmt in body[effect_index:marker_index])
    produce.extend(
        statement_from_string(f"{stash}.append({name})")
        for name, stash in produced_stashes.items()
    )
    ctor = _dtype_ctor_from_identity(marker.identity_expr)
    value = f"{ctor}({marker.input_name})" if ctor else marker.input_name
    produce.append(
        statement_from_string(
            f"{acc} = {_combine_expr(marker.reduction_type, acc, value, ctor)}"
        )
    )
    result.extend(
        [unrolled(produce), statement_from_string("cute.arch.sync_threads()")]
    )
    result.extend(_finalize_lane_reduce_marker(marker, acc))
    marker_dependencies = [(marker, _forward_live_names(body, {marker.result_var}))]
    axes_before, definitions_before = _statement_provenance_before(
        body, thread_axis_names, scalar_definitions, rename_groups
    )
    consume: list[ast.AST] = (
        [
            statement_from_string(f"{name} = {stash}[{lane_var}]")
            for name, stash in (stashes | produced_stashes).items()
            if name in tail_reads
        ]
        if per_lane_tail
        else []
    )
    for stmt in tail:
        owners = _lane_reduction_owner_exprs_for_statement(
            stmt,
            markers,
            marker_dependencies,
            axes_before[id(stmt)],
            definitions_before[id(stmt)],
        )
        consume.append(
            _guard_stmt_with_owner(stmt, owners) if owners else deep_clone(stmt)
        )
    if per_lane_tail:
        result.append(unrolled(consume))
    else:
        result.extend(consume)
    return result


def _split_one_lane_loop(
    loop: ast.For,
    lane_var: str,
    uniform_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
    running_sums: set[str] | None = None,
    register_tile_options: RegisterTileOptions | None = None,
    cta_uniform: bool = True,
) -> list[ast.AST]:
    from ...ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    markers: list[tuple[int, _LaneReduceMarker]] = []
    for idx, stmt in enumerate(body):
        parsed = _is_lane_reduce_marker_assign(stmt)
        if parsed is not None:
            markers.append((idx, parsed))
    impure_indices = {i for i, stmt in enumerate(body) if _has_impure_inline_asm(stmt)}
    if impure_indices and _find_lane_reduce_call(loop) is not None:
        producer = _split_impure_producer(
            loop,
            lane_var,
            markers,
            impure_indices,
            rename_groups,
            thread_axis_names,
            scalar_definitions,
            cta_uniform=cta_uniform,
        )
        if producer is not None:
            return producer
        raise exc.BackendUnsupported(
            "cute", "impure inline assembly has no proved snapshot/order schedule"
        )
    if not markers:
        nested_owners = {
            marker.owner_lane
            for stmt in body
            for node in ast.walk(stmt)
            if (marker := _is_lane_reduce_marker_assign(node)) is not None
        }
        register_tile_failure: exc.BackendUnsupported | None = None
        if nested_owners and _is_constexpr_lane_iter(loop):
            if nested_owners == {lane_var}:
                # Markers nested in trace-time tile element loops under this
                # constexpr owner: the register-tile lowering schedules every
                # lane's loads before the first store and reduces each tile
                # element once.  A body outside its shape may still be a
                # lane-invariant guard around one (lifted below).
                from ..register_tile_reductions import lower_register_tile_lane_loop

                try:
                    return lower_register_tile_lane_loop(
                        loop,
                        lane_var,
                        proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                        proven_tensor_stride_values=proven_tensor_stride_values,
                        thread_axis_names=thread_axis_names,
                        scalar_definitions=scalar_definitions,
                        rename_groups=rename_groups,
                        options=register_tile_options,
                    )
                except RegisterTileUnsupported as failure:
                    register_tile_failure = failure
            elif lane_var not in nested_owners:
                # A trace-time tile lane loop below a register-tile owner: the
                # enclosing owner lowers these markers together.
                return [loop]
        # A dynamic guard that does not depend on the synthetic lane may hide
        # reduction markers in one of its branches.  Move that guard outside
        # the lane loop so each branch owns an ordinary lane loop that can be
        # split below.  This occurs in packed recurrent kernels which guard a
        # state slot before performing several reductions over a free arange.
        lifted = _lift_lane_invariant_if(loop, lane_var, uniform_names, rename_groups)
        if lifted is not None:
            return _split_lane_loop_reductions(
                lifted,
                uniform_names=set(uniform_names),
                proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                proven_tensor_stride_values=proven_tensor_stride_values,
                thread_axis_names=dict(thread_axis_names),
                scalar_definitions=dict(scalar_definitions),
                rename_groups=rename_groups,
                running_sums=running_sums,
                register_tile_options=register_tile_options,
                cta_uniform=cta_uniform,
            )
        if register_tile_failure is not None:
            # ``generate_ast`` regenerates the kernel with the rolled lane
            # nesting; the reason stays in its debug log.
            raise register_tile_failure
        if any(_find_lane_reduce_call(node) is not None for node in ast.walk(loop)):
            folded = _split_lane_loop_around_vector_lane(
                loop,
                lane_var,
                uniform_names,
                proven_disjoint_tensor_pairs,
                proven_tensor_stride_values,
                thread_axis_names,
                scalar_definitions,
                rename_groups,
                running_sums,
            )
            if folded is not None:
                return folded
            if any(
                isinstance(node, ast.For)
                and not _is_serial_for(node)
                and getattr(node, HELION_LANE_LOOP_VAR_ATTR, None) is None
                and any(_find_lane_reduce_call(s) is not None for s in node.body)
                for node in ast.walk(loop)
            ):
                # A ``cutlass.range_constexpr`` vector lane (``cute_vector_widths``
                # > 1) holds the marker and the body is outside the shape the
                # fold above proves (a collective or a matmul contribution in
                # the vector lane, a marker input independent of the vector
                # element, statements after the vector lane, a memory write
                # anywhere in the loop, a constexpr loop that is not this
                # lane's own vector lane): its complete reduction would need
                # the vector unroll folded together with this lane loop.
                raise exc.BackendUnsupported(
                    "cute",
                    "lane reduction nested in a constexpr vector lane "
                    "(cute_vector_widths > 1) has no proved lowering",
                )
            raise exc.BackendUnsupported(
                "cute",
                "lane reduction under a guard whose thread uniformity cannot be proven",
            )
        return [loop]

    marker_indices = {i for i, _ in markers}
    if any(
        marker.owner_lane is not None and marker.owner_lane != lane_var
        for _, marker in markers
    ):
        raise exc.BackendUnsupported(
            "cute", "reduction marker is nested in a different lane owner"
        )
    running_sum = _certify_lane_running_sum(
        loop, lane_var, markers, rename_groups, running_sums
    )
    # A certified matmul self-add stays in the complete accumulation pass.
    # Only its completed value is available to the remaining carry updates;
    # keep the existing independent-carry check for every other binding.
    tail_indices = [
        index
        for index in range(len(body))
        if running_sum is None or index != running_sum.update_index
    ]
    carry_body = [body[index] for index in tail_indices]
    carry_markers = {
        index
        for index, original in enumerate(tail_indices)
        if original in marker_indices
    }
    _validate_renamed_lane_carries(carry_body, lane_var, carry_markers, rename_groups)

    if _lane_split_reorders_aliasing_memory(
        body,
        markers,
        proven_disjoint_tensor_pairs,
        lane_var,
        _static_lane_loop_extent(loop),
        proven_tensor_stride_values,
        additional_inputs=(
            [(running_sum.update_index, running_sum.input_name)]
            if running_sum is not None
            else None
        ),
        rename_groups=rename_groups,
    ):
        raise exc.BackendUnsupported(
            "cute",
            "synthetic-lane reduction would reorder a potentially aliasing write",
        )

    if any(marker.matmul_contribution for _, marker in markers):
        staged = _split_staged_matmul_lane_reductions(
            loop,
            lane_var,
            markers,
            thread_axis_names=thread_axis_names,
            scalar_definitions=scalar_definitions,
            rename_groups=rename_groups,
        )
        if staged is None:
            raise exc.BackendUnsupported(
                "cute", "matmul contribution has no complete staged lane schedule"
            )
        _validate_owned_lane_carry_schedule(
            body, staged, lane_var, marker_indices, rename_groups
        )
        return staged

    # A matmul whose *output* is reduced over a lane-distributed axis (e.g.
    # matmul_layernorm's ``acc.sum(-1)`` over the synthetic-lane N output) cannot
    # be handled by the per-lane / two-pass paths below: each lane owns a
    # distinct output column, so the reduction must combine DIFFERENT lanes, and
    # the matmul (an unduplicatable cross-thread shared-memory reduction) cannot
    # be re-run in a second lane pass.  The register-stash lowering runs the
    # matmul once, stashes each lane's output in a per-thread fragment, and
    # re-derives every downstream reduction / consumer from the stash.
    #
    # This is gated narrowly so it does NOT disturb kernels the existing paths
    # already handle correctly:
    #   * an unduplicatable op must feed the reduction, and
    #   * the marker results must NOT be consumed by a cross-lane loop-carried
    #     accumulator. A phi copy only rules out this stash path; it does not
    #     prove that an owned full-lane reduction can restore its raw input.
    if (
        running_sum is None
        and any(_contains_unduplicatable_op(stmt) for stmt in body)
        and not _markers_feed_cross_lane_carry(body, lane_var, markers)
    ):
        stashed = _split_lane_loop_with_register_stash(
            loop, lane_var, markers, rename_groups
        )
        if stashed is not None:
            _validate_owned_lane_carry_schedule(
                body, stashed, lane_var, marker_indices, rename_groups
            )
            return stashed

    # Safety: the two-pass split is only valid when the reduction marker is the
    # ONLY unproved cross-lane carried value in this lane loop. Another
    # loop-carried accumulator across the lanes (e.g. an uncertified matmul sum or a
    # plain ``extra += per_lane`` sum that already accumulates over the lanes),
    # splitting would drop or double-count it. Only legacy unowned markers can
    # use the original raw-input fallback; owned markers require a complete
    # reduction and are rejected before selecting a split subpath.
    if _has_extra_cross_lane_carry(carry_body, lane_var, carry_markers):
        return _restore_lane_markers(
            loop,
            lane_var,
            markers,
            rename_groups,
            thread_axis_names,
            scalar_definitions,
        )

    # Phase 1: the backward slices that produce the reduction inputs, read
    # through the rename groups: a device loop of the body that accumulates
    # an input writes it under its loop-output name, which only the final
    # rename pass folds into the accumulator, and the loop belongs to the
    # accumulate pass beside the initializer (the dynamic-shape nested-row
    # seeds of jagged_layer_norm reduced the fresh zero in a pass ahead of
    # the loop otherwise, and normalized by a zero mean and variance).  Each
    # input is sliced over the statements before its own marker: a later
    # statement that rewrites the input's group (a loop that keeps
    # accumulating the reduced value, an if-join) consumes the reduction's
    # input rather than producing it, and a slice over the whole body took
    # it for a producer, so the fold read the value behind it.
    if running_sum is not None:
        phase1_indices = list(running_sum.phase1_indices)
    else:
        phase1_index_union: set[int] = set()
        for marker_index, marker in markers:
            indices, _written = _backward_slice(
                body[:marker_index], {marker.input_name}, rename_groups
            )
            phase1_index_union.update(indices)
        phase1_indices = sorted(phase1_index_union)

    # A loop-output name the rename groups do not map (a body split on its
    # own, a carried value the device function has not registered) still
    # hides the update from the slice, which then selects the initializer
    # alone.  A serial loop before the marker that is not in the slice and
    # reads the marker input, or snapshots a name the slice defines in a phi
    # copy (``row_sums_copy = row_sums``: the loop carries that name and
    # rewrites it under its output name), is the sign of it: splitting that
    # shape would reduce the initializer (often zero) instead of the
    # completed accumulator.  Decline an owned reduction without a proved
    # slice; retain the legacy unowned per-lane behavior.
    phase1_index_set = set(phase1_indices)
    phase1_written = {
        name
        for index in phase1_indices
        for name in ReadWrites.from_ast(body[index]).writes
    }
    for marker_index, marker in markers:
        if any(
            index not in phase1_index_set
            and isinstance(stmt, (ast.For, ast.While))
            and (
                marker.input_name in ReadWrites.from_ast(stmt).reads
                or _carries_a_name(stmt, phase1_written)
            )
            for index, stmt in enumerate(body[:marker_index])
        ):
            return _restore_lane_markers(
                loop,
                lane_var,
                markers,
                rename_groups,
                thread_axis_names,
                scalar_definitions,
            )

    # Sequentially-dependent reductions (one marker's input depends on another
    # marker's result) require one accumulate/finalize pass per dependency
    # level.  The common single-pass path below remains preferable when all
    # markers are independent because it walks the lane extent only once.
    if set(phase1_indices) & marker_indices:
        dependent = _split_dependent_lane_reductions(
            loop,
            lane_var,
            markers,
            thread_axis_names=thread_axis_names,
            scalar_definitions=scalar_definitions,
            rename_groups=rename_groups,
        )
        if dependent is not None:
            _validate_owned_lane_carry_schedule(
                body, dependent, lane_var, marker_indices, rename_groups
            )
            return dependent
        return _restore_lane_markers(
            loop,
            lane_var,
            markers,
            rename_groups,
            thread_axis_names,
            scalar_definitions,
        )

    # The phase-1 (accumulate) and phase-2 (consume) passes both re-run the
    # reduction-input producers. That is only safe for side-effect-free
    # producers. A matmul / collective in the slice (cross-thread shared-memory
    # reductions, ``cute.gemm``, ``dot``) cannot be duplicated without racing on
    # shared memory. If the stash path did not prove a complete reduction,
    # decline owned markers rather than restoring an incomplete raw input.
    if any(_contains_unduplicatable_op(body[i]) for i in phase1_indices):
        return _restore_lane_markers(
            loop,
            lane_var,
            markers,
            rename_groups,
            thread_axis_names,
            scalar_definitions,
        )

    prefix: list[ast.AST] = []  # acc init statements (outside the lane loops)
    accumulate_body: list[ast.AST] = [body[i] for i in phase1_indices]
    marker_updates: dict[int, ast.AST] = {}
    finalize: list[ast.AST] = []
    for marker_index, m in markers:
        acc_var = f"{m.result_var}_lane_acc"
        prefix.append(statement_from_string(f"{acc_var} = {m.identity_expr}"))
        # Cast the per-lane input to the accumulator dtype before combining so
        # the CUTLASS DSL's strict ternary type check (max/min emit a Python
        # ``a if a > b else b``) does not see mixed fp32/bf16 operands.
        ctor = _dtype_ctor_from_identity(m.identity_expr)
        combine_val = f"{ctor}({m.input_name})" if ctor is not None else m.input_name
        update = statement_from_string(
            f"{acc_var} = {_combine_expr(m.reduction_type, acc_var, combine_val, ctor)}"
        )
        accumulate_body.append(update)
        marker_updates[marker_index] = update
        finalize.extend(_finalize_lane_reduce_marker(m, acc_var))
    if running_sum is not None:
        # Keep the original producer/self-add order, interleaving each marker's
        # accumulation at its original position. Never regroup the matmul sum
        # or initialize it again at a synthetic-lane boundary.
        accumulate_body = [
            marker_updates[index] if index in marker_updates else body[index]
            for index in sorted(phase1_index_set | marker_indices)
        ]

    # Phase 2: everything except the marker assignments themselves; the
    # reduced scalar is already finalized so consumers read it directly.
    phase2_indices = [index for index in tail_indices if index not in marker_indices]
    phase2_body = [body[index] for index in phase2_indices]

    # A statement is lane-varying if it (transitively) reads the lane var, or
    # writes a name a lane-varying statement writes under any spelling of its
    # rename group (the initializer of an accumulator a lane-varying device
    # loop rewrites under its loop-output name: hoisted once between the
    # passes, every lane after the first would continue from the previous
    # lane's final value).  Statements that only depend on the finalized
    # scalar(s) are lane-invariant and run once after the lane loops;
    # lane-varying consumers run in a second lane loop, but only those that
    # contribute to a side effect (a store, an if-with-store, an in-place
    # write). Pure lane-varying producers that fed only the (now-removed)
    # reduction markers are dropped.  A lane-invariant statement runs between
    # the two loops unless the consume loop's accesses order it after them
    # (``_lane_invariant_tail_after_consume``).
    lane_varying_names = _lane_varying_names(phase2_body, lane_var, rename_groups)
    marker_dependencies = [
        (marker, _forward_live_names(body, {marker.result_var}))
        for _, marker in markers
    ]
    thread_axes_before, scalar_defs_before = _statement_provenance_before(
        body,
        thread_axis_names,
        scalar_definitions,
        rename_groups,
    )

    def is_lane_varying(stmt: ast.AST) -> bool:
        rw = ReadWrites.from_ast(stmt)
        reads = {rename_groups.get(name, name) for name in rw.reads}
        writes = {rename_groups.get(name, name) for name in rw.writes}
        return (
            lane_var in reads
            or bool(reads & lane_varying_names)
            or bool(writes & lane_varying_names)
        )

    if running_sum is not None:
        # Compiler provenance identifies a completed matmul result, not a
        # prefix-sum API. An observable per-lane/conditional use of that value
        # needs a different proof; admit only pure lane-invariant finalizers.
        derived = _forward_live_names(phase2_body, {running_sum.name})
        if any(
            set(ReadWrites.from_ast(stmt).reads) & derived
            and (
                is_lane_varying(stmt)
                or not _is_proven_relocatable_assignment(stmt, allow_load=False)
            )
            for stmt in phase2_body
        ):
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum has a non-invariant consumer"
            )

    keep_indices = _live_phase2_indices(phase2_body)
    varying_indices = {i for i, s in enumerate(phase2_body) if is_lane_varying(s)}
    after_consume = _lane_invariant_tail_after_consume(
        phase2_body,
        invariant=set(range(len(phase2_body))) - varying_indices,
        accumulated={
            i for i, index in enumerate(phase2_indices) if index in phase1_index_set
        },
        consumed=varying_indices & keep_indices,
        rename_groups=rename_groups,
    )
    lane_invariant_tail: list[ast.AST] = []
    lane_invariant_after_consume: list[ast.AST] = []
    lane_varying_tail: list[ast.AST] = []
    for i, s in enumerate(phase2_body):
        if i in varying_indices and i not in keep_indices:
            continue
        owner_exprs = _lane_reduction_owner_exprs_for_statement(
            s,
            markers,
            marker_dependencies,
            thread_axes_before.get(id(s), thread_axis_names),
            scalar_defs_before.get(id(s), {}),
        )
        if phase2_indices[i] in phase1_index_set:
            # The accumulate loop holds this producer already.  Each position
            # gets its own node: later passes key facts by node identity and
            # rewrite in place (the cross-sweep load fusion, for one, saw the
            # two copies at one position and kept the consume loop's reload).
            s = _rerun_copy(s)
        if i in varying_indices:
            lane_varying_tail.append(
                _guard_stmt_with_owner(s, owner_exprs) if owner_exprs else s
            )
        else:
            if owner_exprs:
                predicate = " and ".join(
                    f"({owner_expr})" for owner_expr in owner_exprs
                )
                s = _guard_stmt_with_owner(s, [predicate])
            if i in after_consume:
                lane_invariant_after_consume.append(s)
            else:
                lane_invariant_tail.append(s)

    result: list[ast.AST] = []
    result.extend(prefix)
    result.append(_clone_lane_loop_with_body(loop, accumulate_body))
    result.extend(finalize)
    result.extend(lane_invariant_tail)
    if lane_varying_tail:
        result.append(_clone_lane_loop_with_body(loop, lane_varying_tail))
    result.extend(lane_invariant_after_consume)
    _validate_owned_lane_carry_schedule(
        body,
        result,
        lane_var,
        marker_indices,
        rename_groups,
        running_sum=running_sum,
    )
    return result


def _vector_lane_fold_shape(
    loop: ast.For, lane_var: str
) -> tuple[list[ast.AST], ast.For] | None:
    """``(prefix, vector lane)`` when every reduction marker of ``loop`` sits
    at the top level of one trailing constexpr vector lane that
    :func:`_split_lane_loop_around_vector_lane` can fold.

    The shape is the per-element body of a lane-looped tile axis with
    ``cute_vector_widths > 1``: ``prefix`` holds the lane base and the vector
    loads hoisted above the ``cutlass.range_constexpr(V)`` loop (plain
    assignments without a store, a collective or a marker), and the vector
    lane walks the elements of each packet.  The vector lane must be the
    wrapper loop the strategies build for ``lane_var`` (``vec_lane_{N}`` for
    ``lane_{N}``, see ``VecLaneWrapper``); any other constexpr loop is a
    serial unroll whose iterations the fold may not treat as lane
    coordinates.  Every marker must be a plain reduction owned by
    ``lane_var`` (no matmul contribution, no strided restore); the loop may
    not hold a collective, a nested loop or a marker anywhere else, and
    nothing may follow the vector lane.

    Nothing in the loop may write memory.  The split proves its passes
    reorder no aliasing access on the flat body, where the ``vec = lane``
    sentinel is a definition of ``vec``: the alias expansion would rewrite a
    packet-shifted address such as ``base + vec`` to ``lane * V + lane`` and
    reason in the wrong iteration space, so a store in the vector lane could
    be hoisted past a load of another lane's packet that the rolled loop
    orders after it.  With no write, the passes only re-run loads and the
    proof is trivially sound.
    """
    body: list[ast.AST] = list(loop.body)
    if not body:
        return None
    vloop = body[-1]
    if not (
        isinstance(vloop, ast.For)
        and _is_constexpr_lane_iter(vloop)
        and getattr(vloop, HELION_LANE_LOOP_VAR_ATTR, None) is None
        and isinstance(vloop.target, ast.Name)
        and getattr(vloop, HELION_VEC_LANE_OF_ATTR, None) == lane_var
        and not vloop.orelse
    ):
        return None
    prefix = body[:-1]
    if any(
        not isinstance(stmt, ast.Assign)
        or _has_side_effect(stmt)
        or _find_lane_reduce_call(stmt) is not None
        for stmt in prefix
    ):
        return None
    if any(_has_observable_memory_write(stmt) for stmt in body):
        return None
    if _contains_unduplicatable_op(loop):
        return None
    markers = [
        marker
        for stmt in vloop.body
        if (marker := _is_lane_reduce_marker_assign(stmt)) is not None
    ]
    if not markers or any(
        marker.owner_lane != lane_var
        or marker.matmul_contribution
        or marker.strided_restore
        for marker in markers
    ):
        return None
    nested_markers = sum(
        _is_lane_reduce_marker_assign(node) is not None for node in ast.walk(loop)
    )
    if nested_markers != len(markers):
        return None
    if any(
        isinstance(node, (ast.For, ast.While))
        for stmt in vloop.body
        for node in ast.walk(stmt)
    ):
        return None
    return prefix, vloop


def _split_lane_loop_around_vector_lane(
    loop: ast.For,
    lane_var: str,
    uniform_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
    running_sums: set[str] | None,
) -> list[ast.AST] | None:
    """Split a lane loop whose reduction markers sit in its constexpr vector lane.

    ``cute_vector_widths > 1`` nests the per-element body of a lane-looped
    tile axis in ``for vec in cutlass.range_constexpr(V)`` below the vector
    load of each lane step, so a reduction over that axis reduces over the
    lane steps AND the vector elements.  The vector lane is a second lane
    coordinate rather than a serial loop: the two-pass / dependent split of
    :func:`_split_one_lane_loop` applies unchanged to the flat body
    ``prefix + [vec = lane] + elements``.  The sentinel assignment makes every
    element-level value lane-varying and marks where the vector lane begins;
    every marker input must depend on it, so each accumulate pass of the
    split keeps the vector lane and folds all ``lanes x V`` elements.  Each
    lane loop the split emits is re-nested at the sentinel: the prefix stays
    per lane step and the element statements (with the accumulator updates
    that follow them) go back under the constexpr loop, while the split's
    lane-invariant tail already runs once outside both.  Returns ``None``
    when the loop is outside the shape; a split decline propagates.
    """
    shape = _vector_lane_fold_shape(loop, lane_var)
    if shape is None:
        return None
    prefix, vloop = shape
    vec_var = cast("ast.Name", vloop.target).id
    sentinel_index = len(prefix)
    flat: list[ast.AST] = [
        *prefix,
        statement_from_string(f"{vec_var} = {lane_var}"),
        *vloop.body,
    ]
    for index, stmt in enumerate(flat):
        marker = _is_lane_reduce_marker_assign(stmt)
        if marker is None:
            continue
        indices, _ = _backward_slice(flat[:index], {marker.input_name}, rename_groups)
        if sentinel_index not in indices:
            return None
    lowered = _split_one_lane_loop(
        _clone_lane_loop_with_body(loop, flat),
        lane_var,
        uniform_names,
        proven_disjoint_tensor_pairs,
        proven_tensor_stride_values,
        thread_axis_names,
        scalar_definitions,
        rename_groups,
        running_sums,
    )
    return _renest_vector_lane(lowered, lane_var, vloop, vec_var)


def _renest_vector_lane(
    statements: list[ast.AST], lane_var: str, vloop: ast.For, vec_var: str
) -> list[ast.AST]:
    """Move the statements after the ``vec = lane`` sentinel of every emitted
    ``lane_var`` loop back under a copy of the constexpr vector lane."""

    def is_sentinel(stmt: ast.AST) -> bool:
        return (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == vec_var
            and isinstance(stmt.value, ast.Name)
            and stmt.value.id == lane_var
        )

    def renest(stmt: ast.AST) -> ast.AST:
        for field in ("body", "orelse", "finalbody"):
            old = getattr(stmt, field, None)
            if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
                setattr(stmt, field, [renest(s) for s in old])
        if (
            isinstance(stmt, ast.For)
            and getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None) == lane_var
        ):
            positions = [i for i, s in enumerate(stmt.body) if is_sentinel(s)]
            if len(positions) == 1:
                split_at = positions[0]
                elements = stmt.body[split_at + 1 :]
                nested: list[ast.stmt] = []
                if elements:
                    nested.append(
                        create(
                            ast.For,
                            target=create(ast.Name, id=vec_var, ctx=ast.Store()),
                            iter=ast.parse(ast.unparse(vloop.iter), mode="eval").body,
                            body=elements,
                            orelse=[],
                            type_comment=None,
                        )
                    )
                stmt.body = [*stmt.body[:split_at], *nested]
        return stmt

    result = [renest(stmt) for stmt in statements]
    if any(is_sentinel(node) for stmt in result for node in ast.walk(stmt)):
        raise exc.BackendUnsupported(
            "cute",
            "lane reduction nested in a constexpr vector lane left the vector "
            "lane outside its lane loop",
        )
    return result


def _validate_owned_lane_carry_schedule(
    body: list[ast.AST],
    lowered: list[ast.AST],
    lane_var: str,
    marker_indices: set[int],
    rename_groups: dict[str, str],
    *,
    running_sum: _LaneRunningSum | None = None,
) -> None:
    """Require each allowed carry update once after the full-lane reductions.

    The early variation check proves that a carry can be uniform after marker
    finalization; it does not prove that every split subpath preserves it.
    In particular, the register stash drops outside-only updates and repeats
    inside-observed updates in its consume loop. Admit a carry only when the
    selected schedule retains its exact normalized assignment once, outside
    every loop and after all marker results. The explicit matmul running-sum
    certificate is the sole exception: its original self-add and full input
    slice must remain once per lane, in order, inside the original iterator.
    Compound or multiple updates remain conservatively declined.
    """
    from ...ast_read_writes import ast_rename

    if not any(
        marker.owner_lane is not None
        for index in marker_indices
        if (marker := _is_lane_reduce_marker_assign(body[index])) is not None
    ):
        return

    def normalize(statements: list[ast.AST]) -> list[ast.AST]:
        result = [deep_clone(statement) for statement in statements]
        for statement in result:
            ast_rename(statement, rename_groups)
        return result

    original = normalize(body)
    effects = _ordered_block_reads_writes(cast("list[ast.stmt]", original))
    # ReadWrites also models store(value) as an in-place memory write.
    # This proof counts assignments to scalar bindings, not stored values.
    original_writes = [names_written(statement) for statement in original]
    carried = (effects.reads_before_write & set().union(*original_writes)) - {lane_var}
    if not carried:
        return
    selected = normalize(lowered)
    marker_results = {
        marker.result_var
        for index in marker_indices
        if (marker := _is_lane_reduce_marker_assign(original[index])) is not None
    }
    selected_writes = [names_written(statement) for statement in selected]
    if running_sum is not None:
        positions = [
            index
            for index, writes in enumerate(selected_writes)
            if running_sum.name in writes
        ]
        if len(positions) != 1 or running_sum.name not in carried:
            raise exc.BackendUnsupported(
                "cute", "lane reduction cannot prove the certified per-lane update"
            )
        boundary = positions[0]
        phase1 = selected[boundary]
        expected: list[ast.AST] = []
        for index in sorted(set(running_sum.phase1_indices) | marker_indices):
            if index not in marker_indices:
                expected.append(body[index])
                continue
            marker = _is_lane_reduce_marker_assign(body[index])
            assert marker is not None
            accumulator = f"{marker.result_var}_lane_acc"
            ctor = _dtype_ctor_from_identity(marker.identity_expr)
            value = (
                f"{ctor}({marker.input_name})"
                if ctor is not None
                else marker.input_name
            )
            expected.append(
                statement_from_string(
                    f"{accumulator} = "
                    f"{_combine_expr(marker.reduction_type, accumulator, value, ctor)}"
                )
            )
        if not (
            isinstance(phase1, ast.For)
            and isinstance(phase1.target, ast.Name)
            and phase1.target.id == lane_var
            and ast.dump(phase1.iter) == running_sum.iterator
            and not phase1.orelse
            and [ast.dump(stmt) for stmt in phase1.body]
            == [ast.dump(stmt) for stmt in normalize(expected)]
        ):
            raise exc.BackendUnsupported(
                "cute", "lane reduction cannot prove the certified per-lane update"
            )
        # Every reduction must finalize after that complete lane sweep, before
        # a tail can observe the completed matmul accumulator.
        finalized_at: set[int] = set()
        for marker_result in marker_results:
            definitions = [
                index
                for index, writes in enumerate(selected_writes)
                if marker_result in writes
            ]
            if (
                len(definitions) != 1
                or definitions[0] <= boundary
                or assigned_name(selected[definitions[0]]) != marker_result
            ):
                raise exc.BackendUnsupported(
                    "cute", "certified lane update precedes no complete finalization"
                )
            finalized_at.add(definitions[0])
        from ...ast_read_writes import ReadWrites

        if any(
            index != boundary
            and index <= max(finalized_at)
            and running_sum.name in ReadWrites.from_ast(statement).reads
            for index, statement in enumerate(selected)
        ):
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum is consumed before finalization"
            )
    for name in carried:
        if running_sum is not None and name == running_sum.name:
            continue
        updates = [
            statement
            for statement, writes in zip(original, original_writes, strict=True)
            if name in writes
        ]
        positions = [
            index for index, writes in enumerate(selected_writes) if name in writes
        ]
        if (
            len(updates) == 1
            and isinstance(update := updates[0], ast.Assign)
            and len(update.targets) == 1
            and isinstance(update.targets[0], ast.Name)
            and update.targets[0].id == name
            and len(positions) == 1
            and ast.dump(selected[positions[0]]) == ast.dump(update)
        ):
            boundary = positions[0]
            finalized = set().union(*selected_writes[:boundary])
            if marker_results <= finalized and not any(
                writes & marker_results for writes in selected_writes[boundary:]
            ):
                continue
        raise exc.BackendUnsupported(
            "cute", "lane reduction cannot prove a once-per-tile carried update"
        )


def _validate_renamed_lane_carries(
    body: list[ast.AST],
    lane_var: str,
    marker_indices: set[int],
    rename_groups: dict[str, str],
) -> None:
    """Check every independent carry before any stash or split can discard it.

    Always normalize production names, even when a raw-name carry already
    exists. That carry neither protects a second renamed update from the
    earlier stash path nor proves a complete per-lane reduction. An owned
    marker needs a full reduction over its lane; no subpath below proves that
    reduction together with an independent lane-varying carry. Decline the
    composition rather than losing either observable update or the reduction.

    Unowned standalone AST markers retain their legacy raw-carry behavior;
    all production emitters attach an owner.
    """
    from ...ast_read_writes import ast_rename

    owned = any(
        marker.owner_lane is not None
        for index in marker_indices
        if (marker := _is_lane_reduce_marker_assign(body[index])) is not None
    )
    if not owned and (
        not rename_groups or _has_extra_cross_lane_carry(body, lane_var, marker_indices)
    ):
        return
    normalized = [deep_clone(statement) for statement in body]
    for statement in normalized:
        ast_rename(statement, rename_groups)
    if _has_extra_cross_lane_carry(
        normalized, lane_var, marker_indices, finalized_markers=owned
    ):
        raise exc.BackendUnsupported(
            "cute", "lane reduction cannot preserve an independent loop-carried value"
        )


def _lift_lane_invariant_if(
    loop: ast.For,
    lane_var: str,
    uniform_names: set[str],
    rename_groups: Mapping[str, str] | None = None,
    *,
    unlooped: Collection[str] = (),
    lane_setup: Collection[int] = (),
) -> list[ast.AST] | None:
    """Lift one lane-invariant guard that hides reduction markers.

    Synthetic reduction lanes wrap the complete grid body.  Consequently a
    uniform runtime guard such as ``if state_index < 0`` can sit between the
    lane loop and its reduction markers, while :func:`_split_one_lane_loop`
    intentionally only rewrites direct marker assignments.  Lift the guard
    when its complete condition slice is independent of both the sequential
    lane and CUDA thread coordinates.  Each branch then owns a lane loop and
    can be processed by the normal reduction splitter.

    This is deliberately narrow: exactly one top-level guarded tail, no
    statements after it, and no side effects in the condition slice.  The
    slice is read through ``rename_groups``, so a loop updating a value of
    the condition under its loop-output name is in it; as a side effect it
    declines the lift, and the caller then rejects the markers under the
    guard ("lane reduction under a guard whose thread uniformity cannot be
    proven") or re-raises the register-tile failure, rather than restoring
    them.

    The branches named in ``unlooped`` (``"body"``, ``"orelse"``) run without
    the lane loop and without the statements of the loop body at the
    ``lane_setup`` indices (``_sink_lane_loop_into_branches``).
    """
    from ...ast_read_writes import ReadWrites

    body: list[ast.AST] = [*loop.body]
    candidates = [
        (idx, stmt)
        for idx, stmt in enumerate(body)
        if isinstance(stmt, ast.If)
        and any(_find_lane_reduce_call(child) is not None for child in ast.walk(stmt))
    ]
    if len(candidates) != 1:
        return None
    if_index, branch = candidates[0]
    if if_index != len(body) - 1:
        return None

    prefix: list[ast.AST] = body[:if_index]
    condition_reads = set(ReadWrites.from_ast(branch.test).reads)
    condition_indices, condition_writes = _backward_slice(
        prefix, condition_reads, rename_groups
    )
    condition_nodes = [prefix[idx] for idx in condition_indices]
    external_reads = set(condition_reads)
    for stmt in condition_nodes:
        external_reads.update(ReadWrites.from_ast(stmt).reads)
    external_reads.difference_update(condition_writes)
    # Every dependency from outside the condition slice must be a kernel
    # argument, generated constexpr, module global, or an enclosing assignment
    # already proven independent of CUDA thread coordinates.
    if not all(_is_proven_uniform_name(name, uniform_names) for name in external_reads):
        return None
    condition_source = "\n".join(
        [ast.unparse(branch.test), *(ast.unparse(stmt) for stmt in condition_nodes)]
    )
    if lane_var in condition_source:
        return None
    if any(name in condition_source for name in _CUTE_THREAD_VARYING_INTRINSICS):
        return None
    if any(_has_side_effect(stmt) for stmt in condition_nodes):
        return None

    condition_index_set = set(condition_indices)
    lane_prefix = [idx for idx in range(len(prefix)) if idx not in condition_index_set]

    def branch_loop(field: str) -> list[ast.stmt]:
        statements = [deep_clone(stmt) for stmt in getattr(branch, field)]
        if field in unlooped:
            unlooped_body = [
                *(
                    deep_clone(prefix[idx])
                    for idx in lane_prefix
                    if idx not in lane_setup
                ),
                *statements,
            ]
            if unlooped_body:
                return cast("list[ast.stmt]", unlooped_body)
            return [ast.Pass()] if field == "body" else []
        branch_body = [
            *(deep_clone(prefix[idx]) for idx in lane_prefix),
            *statements,
        ]
        if not branch_body:
            return [ast.Pass()]
        return [
            cast(
                "ast.stmt",
                _clone_lane_loop_with_body(loop, branch_body),
            )
        ]

    lifted_if = create(
        ast.If,
        test=deep_clone(branch.test),
        body=branch_loop("body"),
        orelse=branch_loop("orelse"),
    )
    if getattr(branch, HELION_BLOCK_UNIFORM_ATTR, False):
        setattr(lifted_if, HELION_BLOCK_UNIFORM_ATTR, True)
    return [*(deep_clone(stmt) for stmt in condition_nodes), lifted_if]


def sink_lane_loops_into_uniform_branches(
    body: list[ast.AST],
    *,
    uniform_names: set[str],
    rename_groups: Mapping[str, str],
) -> list[ast.AST]:
    """Move each synthetic lane loop into the uniform branches it encloses.

    Synthetic reduction lanes wrap the complete grid body, so a reduction
    under a thread-uniform guard (``if pid == 0:`` in an ``hl.grid``) sits
    below the guard inside its lane loop, out of reach of the serial-loop
    interchange that runs before the splitter lifts the guard
    (``_lift_lane_invariant_if``).  The reductions of mutually exclusive
    branches also each sit in the other branch's lane loop, whose lane they
    never read: the owner check rejects such a marker, and the full nest
    would repeat its branch once per foreign lane.  Lift such a guard ahead
    of both, and run a branch that reads none of the loop's coordinates
    outside the loop.  Bottom-up, so in a nest every loop ends up around only
    the branches that read it.
    """
    if not any(_find_lane_reduce_call(stmt) is not None for stmt in body):
        return body

    def visit(statements: list[ast.AST], uniform: set[str]) -> list[ast.AST]:
        result: list[ast.AST] = []
        for stmt in statements:
            for field in ("body", "orelse", "finalbody"):
                children = getattr(stmt, field, None)
                if isinstance(children, list) and all(
                    isinstance(child, ast.stmt) for child in children
                ):
                    setattr(stmt, field, visit(children, set(uniform)))
            lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
            emitted: list[ast.AST] = [stmt]
            if (
                isinstance(stmt, ast.For)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == lane_var
            ):
                sunk = _sink_lane_loop_into_branches(
                    stmt, stmt.target.id, uniform, rename_groups
                )
                if sunk is not None:
                    # A loop kept around a branch may end in that branch's
                    # own guard (an ``elif`` chain).
                    emitted = visit(sunk, set(uniform))
            for statement in emitted:
                _update_proven_uniform_names(statement, uniform)
            result.extend(emitted)
        return result

    return visit(body, {*_CUTE_UNIFORM_GLOBAL_NAMES, *uniform_names})


def _sink_lane_loop_into_branches(
    loop: ast.For,
    lane_var: str,
    uniform_names: set[str],
    rename_groups: Mapping[str, str],
) -> list[ast.AST] | None:
    """``loop`` moved into the branches of its final guard, or None.

    The guard must hold a reduction marker.  A branch runs outside the loop,
    without the lane setup the loop body computes before the guard, when
    that setup is pure and the branch and the rest of the loop body read
    none of the loop's coordinates (its lane and the names computed from it,
    across iterations too) and keep their effect run once instead of once
    per lane (``runs_once_outside_lane``).
    """
    if not loop.body or not isinstance(guard := loop.body[-1], ast.If):
        return None
    if not any(_find_lane_reduce_call(node) is not None for node in ast.walk(guard)):
        return None
    prefix = loop.body[:-1]
    # The guard is lane-invariant: every iteration runs the same branch.
    coordinates = _forward_live_names(
        [*prefix, *guard.body, *guard.orelse], {lane_var}, rename_groups
    )

    def reads_lane(stmt: ast.AST) -> bool:
        return any(
            rename_groups.get(name, name) in coordinates
            for name in ReadWrites.from_ast(stmt).reads
        )

    def runs_outside(statements: Iterable[ast.AST]) -> bool:
        return all(
            not reads_lane(stmt) and runs_once_outside_lane(stmt, lane_var)
            for stmt in statements
        )

    lane_setup = {idx for idx, stmt in enumerate(prefix) if reads_lane(stmt)}
    unlooped: list[str] = []
    if not any(_has_side_effect(prefix[idx]) for idx in lane_setup) and runs_outside(
        stmt for idx, stmt in enumerate(prefix) if idx not in lane_setup
    ):
        unlooped = [
            field for field in ("body", "orelse") if runs_outside(getattr(guard, field))
        ]
    return _lift_lane_invariant_if(
        loop,
        lane_var,
        uniform_names,
        rename_groups,
        unlooped=unlooped,
        lane_setup=lane_setup,
    )


def _split_staged_matmul_lane_reductions(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    *,
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
) -> list[ast.AST] | None:
    """Stash a collective prefix, then finalize proved product reductions.

    This path accepts only owned markers and straight-line scalar assignments.
    Each FP32 collective in the prefix executes once per lane; later passes
    read its register fragment. The existing dependent-reduction scheduler
    keeps final, lane-invariant recurrence updates outside every lane sweep.
    Both carry validators still apply to the original body and final schedule.
    """
    if any(marker.owner_lane != lane_var for _, marker in markers):
        return None
    extent = _static_lane_loop_extent(loop)
    if extent is None or not 1 < extent <= 256:
        return None
    body = list(loop.body)
    marker_indices = {index for index, _ in markers}
    names = [assigned_name(statement) for statement in body]
    if None in names or len(set(names)) != len(names):
        return None
    for index, statement in enumerate(body):
        if index not in marker_indices and not _is_proven_relocatable_assignment(
            statement, allow_load=True, allow_reduction=True
        ):
            return None
    first_marker = min(marker_indices)
    collective_indices = [
        index
        for index, statement in enumerate(body)
        if _contains_unduplicatable_op(statement)
        # A matmul fold ahead of the reductions is computed once per lane
        # and read back from its stash, never re-run per pass.
        or (index < first_marker and _is_fold_dot_assignment(statement))
    ]
    if collective_indices and first_marker <= max(collective_indices):
        return None
    transformed: list[ast.AST] = [deep_clone(statement) for statement in body]
    prefix: list[ast.AST] = []
    if collective_indices:
        uid = _next_lane_stash_uid()
        compute: list[ast.AST] = [
            deep_clone(statement) for statement in body[: max(collective_indices) + 1]
        ]
        for index in collective_indices:
            statement = body[index]
            assert isinstance(statement, ast.Assign)
            if (
                _stashed_collective_dtype(statement.value, body[:index])
                != "cutlass.Float32"
            ):
                return None
            name = names[index]
            fragment = f"_lane_stash_{uid}_{name}"
            prefix.append(
                statement_from_string(
                    f"{fragment} = cute.make_rmem_tensor({extent}, cutlass.Float32)"
                )
            )
            compute.append(statement_from_string(f"{fragment}[{lane_var}] = {name}"))
            transformed[index] = statement_from_string(
                f"{name} = {fragment}[{lane_var}]"
            )
        prefix.append(_clone_lane_loop_with_body(loop, compute))
    staged_loop = _clone_lane_loop_with_body(loop, transformed)
    assert isinstance(staged_loop, ast.For)
    scheduled = _split_dependent_lane_reductions(
        staged_loop,
        lane_var,
        markers,
        thread_axis_names=thread_axis_names,
        scalar_definitions=scalar_definitions,
        rename_groups=rename_groups,
    )
    return [*prefix, *scheduled] if scheduled is not None else None


def _split_dependent_lane_reductions(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    *,
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
) -> list[ast.AST] | None:
    """Split an ordered chain of lane reductions into multiple passes.

    Each marker is accumulated and finalized before the next marker's input is
    recomputed.  This handles normalization/update/project chains such as
    ``sum(k*k) -> normalized_k -> sum(state*k) -> ...`` while keeping the
    existing one-pass rewrite for independent reductions.
    """
    from ...ast_read_writes import ReadWrites

    body: list[ast.AST] = [*loop.body]
    marker_indices = {idx for idx, _ in markers}
    marker_results = {marker.result_var for _, marker in markers}
    finalized: set[str] = set()
    result: list[ast.AST] = []
    marker_dependencies = [
        (marker, _forward_live_names(body, {marker.result_var}))
        for _, marker in markers
    ]
    thread_axes_before, scalar_defs_before = _statement_provenance_before(
        body,
        thread_axis_names,
        scalar_definitions,
        rename_groups,
    )

    # Every stage indexes a prefix of the marker-free body below.
    accumulated: set[int] = set()
    for marker_index, marker in markers:
        available_body: list[ast.AST] = [
            stmt
            for idx, stmt in enumerate(body[:marker_index])
            if idx not in marker_indices
        ]
        stage_indices, _ = _backward_slice(
            available_body, {marker.input_name}, rename_groups
        )
        stage = [available_body[idx] for idx in stage_indices]
        accumulated.update(stage_indices)
        stage_reads = {
            name for stmt in stage for name in ReadWrites.from_ast(stmt).reads
        }
        if stage_reads & (marker_results - finalized):
            return None
        if any(_contains_unduplicatable_op(stmt) for stmt in stage):
            return None
        if any(_has_side_effect(stmt) for stmt in stage):
            return None

        acc_var = f"{marker.result_var}_lane_acc"
        result.append(statement_from_string(f"{acc_var} = {marker.identity_expr}"))
        acc_body = [_rerun_copy(stmt) for stmt in stage]
        ctor = _dtype_ctor_from_identity(marker.identity_expr)
        combine_val = (
            f"{ctor}({marker.input_name})" if ctor is not None else marker.input_name
        )
        acc_body.append(
            statement_from_string(
                f"{acc_var} = "
                f"{_combine_expr(marker.reduction_type, acc_var, combine_val, ctor)}"
            )
        )
        result.append(_clone_lane_loop_with_body(loop, acc_body))
        result.extend(_finalize_lane_reduce_marker(marker, acc_var))
        finalized.add(marker.result_var)

    # Re-run the lane-invariant tail once with every reduction finalized, and
    # only the dependency closure of observable lane-varying side effects in a
    # final lane pass.  The invariant tail must not be pruned to memory side
    # effects: generated SSA assignments can be loop-carried outputs whose
    # physical names are restored by the later rename pass (for example the
    # ``acc_cnt``/``acc_mean``/``acc_m2`` updates in Welford).
    consume_candidates: list[ast.AST] = [
        stmt for idx, stmt in enumerate(body) if idx not in marker_indices
    ]
    keep_indices = _live_phase2_indices(consume_candidates)
    lane_varying = _lane_varying_names(consume_candidates, lane_var, rename_groups)

    def is_lane_varying(stmt: ast.AST) -> bool:
        rw = ReadWrites.from_ast(stmt)
        reads = {rename_groups.get(name, name) for name in rw.reads}
        writes = {rename_groups.get(name, name) for name in rw.writes}
        return (
            lane_var in reads
            or bool(reads & lane_varying)
            or bool(writes & lane_varying)
        )

    varying_indices = {
        idx for idx, stmt in enumerate(consume_candidates) if is_lane_varying(stmt)
    }
    after_consume = _lane_invariant_tail_after_consume(
        consume_candidates,
        invariant=set(range(len(consume_candidates))) - varying_indices,
        accumulated=accumulated,
        consumed=varying_indices & keep_indices,
        rename_groups=rename_groups,
    )
    invariant: list[ast.AST] = []
    invariant_after_consume: list[ast.AST] = []
    varying: list[ast.AST] = []
    for idx, stmt in enumerate(consume_candidates):
        lane_varying_stmt = idx in varying_indices
        if lane_varying_stmt and idx not in keep_indices:
            continue
        owner_exprs = _lane_reduction_owner_exprs_for_statement(
            stmt,
            markers,
            marker_dependencies,
            thread_axes_before.get(id(stmt), thread_axis_names),
            scalar_defs_before.get(id(stmt), {}),
        )
        cloned = _guard_stmt_with_owner(stmt, owner_exprs)
        if lane_varying_stmt:
            varying.append(cloned)
        elif idx in after_consume:
            invariant_after_consume.append(cloned)
        else:
            invariant.append(cloned)
    result.extend(invariant)
    if varying:
        result.append(_clone_lane_loop_with_body(loop, varying))
    result.extend(invariant_after_consume)
    return result


_LANE_STASH_COUNTER = itertools.count()
_LANE_STASH_COUNTERS: weakref.WeakKeyDictionary[DeviceFunction, Iterator[int]] = (
    weakref.WeakKeyDictionary()
)


def _next_lane_stash_uid() -> int:
    """Allocate the id that names one lane loop's stash fragments.

    Ids count per kernel: the rendered text must not depend on how many
    kernels this process compiled before it (the autotune cache and the render
    dumps key on that text).  The unit tests run the lane passes on bare
    bodies outside any ``DeviceFunction``; those draw from a process counter.
    """
    try:
        function = DeviceFunction.current()
    except NoCurrentFunction:
        return next(_LANE_STASH_COUNTER)
    counter = _LANE_STASH_COUNTERS.get(function)
    if counter is None:
        counter = _LANE_STASH_COUNTERS[function] = itertools.count()
    return next(counter)


def _stashed_collective_dtype(value: ast.AST, before: Sequence[ast.AST]) -> str | None:
    """The dtype constructor of a collective reduction a lane stash may hold.

    A grouped helper names its accumulator dtype in its identity argument; a
    ``cute.arch.warp_reduction_*`` returns its input's type, proven when the
    input is a dtype constructor call or a name ``before`` last binds to one.
    Anything else (a matmul, an unrecognized collective) is ``None``.
    """
    if not isinstance(value, ast.Call):
        return None
    name = call_path_str(value.func)
    if name in _CHUNK_REDUCTION_HELPERS and len(value.args) >= 3:
        return _dtype_ctor_from_identity(ast.unparse(value.args[2]))
    if name == _FOLD_DOT_HELPER and len(value.args) >= 6:
        # ``fold_dot(lhs, step, rhs, step, valid, identity, ...)`` sums in
        # its identity's dtype.
        return _dtype_ctor_from_identity(ast.unparse(value.args[5]))
    if name is None or not name.startswith("cute.arch.warp_reduction_"):
        return None
    if not value.args:
        return None
    reduced = value.args[0]
    if isinstance(reduced, ast.Name):
        definitions = [
            statement.value
            for statement in before
            if isinstance(statement, ast.Assign)
            and assigned_name(statement) == reduced.id
        ]
        if not definitions:
            return None
        reduced = definitions[-1]
    if (
        isinstance(reduced, ast.Call)
        and (ctor := call_path_str(reduced.func)) is not None
        and ctor.startswith("cutlass.")
        and len(reduced.args) == 1
    ):
        return ctor
    return None


def _stash_dtype_for_value(
    value_name: str, body: list[ast.AST], markers: list[tuple[int, _LaneReduceMarker]]
) -> str:
    """Pick a CuTe scalar dtype constructor for a stashed lane value.

    Prefer the accumulator dtype of a marker whose input transitively reads the
    stashed value (matmul outputs feed an fp32 ``sum``), then any cast that
    wraps the value's defining assignment, then ``cutlass.Float32``.
    """
    from ...ast_read_writes import ReadWrites

    for _idx, m in markers:
        slice_indices, _ = _backward_slice(body, {m.input_name})
        reads_value = any(
            value_name in ReadWrites.from_ast(body[i]).reads for i in slice_indices
        )
        if reads_value:
            ctor = _dtype_ctor_from_identity(m.identity_expr)
            if ctor is not None:
                return ctor
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == value_name
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "cutlass"
        ):
            return f"cutlass.{stmt.value.func.attr}"
    return "cutlass.Float32"


def _strip_ssa_suffix(name: str) -> str:
    """Strip Helion's SSA / loop-carry suffixes from a variable name.

    ``acc_1`` / ``acc_copy`` / ``acc_copy_0`` all collapse to ``acc`` so a
    loop-carried accumulator can be matched to its per-iteration rewrites.
    """
    base = re.sub(r"(_copy)(_\d+)*$", "", name)
    return re.sub(r"(_\d+)+$", "", base)


def _carries_a_name(loop: ast.AST, names: set[str]) -> bool:
    """Whether ``loop`` snapshots one of ``names`` in a phi copy (``X_copy = X``).

    Helion carries a value through a device loop with such a copy at the top
    of the loop's body and rewrites the value under the loop-output name,
    which only the final rename pass folds back into ``X``: a copy of a name
    inside the loop is the loop accumulating that name.
    """
    return any(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Name)
        and node.value.id in names
        and node.targets[0].id != node.value.id
        and _strip_ssa_suffix(node.targets[0].id) == _strip_ssa_suffix(node.value.id)
        for node in ast.walk(loop)
    )


def _assigns_simple_name(stmt: ast.AST, names: set[str]) -> bool:
    """Return True when ``stmt`` is ``X = ...`` for some ``X`` in ``names``."""
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id in names
    )


def _undup_stmt_rederives(stmt: ast.AST, name: str) -> bool:
    """Return True when ``stmt`` (an unduplicatable statement, typically the
    matmul K ``for`` loop) re-derives the loop-carried accumulator ``name``.

    The statement re-derives ``name`` when it both reads ``name`` (the live-in
    accumulator) and writes a value transitively dependent on ``name`` whose
    SSA-stripped name equals ``name`` (e.g. ``acc_1 = acc_copy_0 + ...``).  A
    plain input the matmul only reads (``indices_2``) is not re-derived.
    """
    from ...ast_read_writes import ReadWrites

    if not isinstance(stmt, ast.For):
        rw = ReadWrites.from_ast(stmt)
        return name in rw.reads and any(_strip_ssa_suffix(w) == name for w in rw.writes)
    if name not in ReadWrites.from_ast(stmt).reads:
        return False
    # Forward slice from ``name`` within the loop body; True when it reaches a
    # write whose stripped name is ``name``.
    inner = list(stmt.body)
    tainted = {name}
    changed = True
    while changed:
        changed = False
        for s in inner:
            rw = ReadWrites.from_ast(s)
            if set(rw.reads) & tainted:
                for w in rw.writes:
                    if w not in tainted:
                        tainted.add(w)
                        changed = True
    return any(_strip_ssa_suffix(w) == name for w in tainted if w != name)


def _store_value_and_addr_reads(stmt: ast.AST) -> tuple[set[str], set[str]] | None:
    """For a statement containing a single ``(ADDR).store(VALUE)`` call, return
    ``(value_reads, addr_reads)`` — the names read in the stored VALUE and the
    names read in the ADDRESS expression (plus any guarding condition).

    Returns ``None`` when the statement has no ``.store(...)`` call (or has more
    than one, which this analysis does not attempt to characterize)."""
    stores = [
        node
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
        and len(node.args) == 1
    ]
    if len(stores) != 1:
        return None
    store = stores[0]
    value_reads = {n.id for n in ast.walk(store.args[0]) if isinstance(n, ast.Name)}
    # Everything read in the statement that is not part of the stored value
    # belongs to the address expression / guard (e.g. an enclosing ``if mask:``).
    all_reads = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    addr_reads = all_reads - value_reads
    return value_reads, addr_reads


def _lane_axis_wrongly_collapsed(
    tail: list[ast.AST],
    lane_var: str,
    lane_varying: set[str],
    stash_set: set[str],
    marker_result_names: set[str],
) -> bool:
    """Return True when the register-stash lowering would collapse the lane axis
    incorrectly.

    The stash lowering reduces each marker over the lane axis, which is only
    correct when the lane-distributed axis IS the reduced axis (matmul_layernorm:
    the reduced output free dim is broadcast back into a per-lane store whose
    VALUE still depends on the per-lane stash output).  When a side-effecting
    store writes to a lane-varying ADDRESS but its stored VALUE depends only on
    the reduced marker scalars (never on a per-lane stash output), each lane is a
    distinct, preserved output element that the lane reduction wrongly collapses
    (e.g. ``baddbmm(...).sum(-1)`` with the reduced dim folded into the matmul).

    ``lane_varying`` is the set of lane-derived names over the FULL lane-loop body
    (the per-lane output index, e.g. ``indices_2``, lives in the compute region
    rather than the tail, so it must be supplied by the caller).
    """
    # Names in the tail that (transitively) depend on a stashed per-lane value,
    # WITHOUT crossing a marker reduction.  A reduction marker collapses the lane
    # axis: its result is a cross-lane reduced scalar, no longer a per-lane value,
    # so the taint must STOP at marker results (a store of the reduced scalar is
    # exactly the collapse-bug signature, and must not count as stash-dependent).
    stash_tainted = _forward_taint_excluding_markers(
        tail, stash_set, marker_result_names
    )
    for stmt in tail:
        parsed = _store_value_and_addr_reads(stmt)
        if parsed is None:
            continue
        value_reads, addr_reads = parsed
        addr_lane_varying = lane_var in addr_reads or bool(addr_reads & lane_varying)
        value_depends_on_stash = bool(value_reads & stash_tainted)
        value_depends_on_marker = bool(value_reads & marker_result_names)
        if addr_lane_varying and value_depends_on_marker and not value_depends_on_stash:
            return True
    return False


def _forward_taint_excluding_markers(
    body: list[ast.AST], roots: set[str], marker_result_names: set[str]
) -> set[str]:
    """Forward-slice taint of ``roots`` through ``body`` that does NOT propagate
    across a lane-reduce marker.

    A marker assigns a cross-lane reduced scalar (``R = ..._helion_lane_reduce``);
    its result no longer depends on a single lane's value, so a statement that
    only writes a marker result must not inherit the taint even when its reduction
    input was tainted.
    """
    from ...ast_read_writes import ReadWrites

    tainted = set(roots)
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if not (set(rw.reads) & tainted):
                continue
            for w in rw.writes:
                # A marker result is a reduction boundary: never taint it.
                if w in marker_result_names:
                    continue
                if w not in tainted:
                    tainted.add(w)
                    changed = True
    return tainted


def _split_lane_loop_with_register_stash(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    rename_groups: Mapping[str, str] | None = None,
) -> list[ast.AST] | None:
    """Lower a lane loop whose reduction inputs depend on an unduplicatable op.

    The standard two-pass split re-runs the reduction-input producers in a
    second pass, which is unsafe when a matmul / cross-thread reduction is in
    the slice.  Instead, run the matmul-bearing *compute region* (the prefix of
    the body up to and including the last unduplicatable statement) exactly
    once, stashing each lane's unduplicatable-derived live-out values into
    per-thread register fragments.  Every downstream reduction marker and
    consumer is then re-derived from the stash (a plain register read, safely
    duplicatable), one accumulate/finalize pass per marker in dependency order,
    plus a final consume pass for the side-effecting statements.

    Returns the replacement statement list, or ``None`` when the pattern does
    not apply (caller then falls back to the per-lane single-pass behavior).
    """
    from ...ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    marker_indices = {i for i, _ in markers}
    extent = _lane_loop_extent(loop)

    # Compute region: prefix of the body up to (and including) the last
    # unduplicatable statement.  Everything after it must be free of
    # unduplicatable ops so it can be re-derived from the stash.
    undup_indices = [
        i for i in range(len(body)) if _contains_unduplicatable_op(body[i])
    ]
    if not undup_indices:
        return None
    region_end = max(undup_indices)
    region = body[: region_end + 1]
    tail = body[region_end + 1 :]
    tail_offset = region_end + 1
    if any(_contains_unduplicatable_op(s) for s in tail):
        return None
    # A marker inside the compute region cannot be re-derived from the stash
    # (its reduction would have to run before the matmul finishes).
    if any(i <= region_end for i in marker_indices):
        return None

    if extent <= 0 or extent > 256:
        return None

    lane_varying = _lane_varying_names(body, lane_var)

    tail_reads: set[str] = set()
    for stmt in tail:
        tail_reads |= set(ReadWrites.from_ast(stmt).reads)

    # Identify the loop-carried accumulators produced by the unduplicatable
    # statements — values whose post-region magnitude depends on the matmul and
    # therefore cannot be recomputed.  These are exactly the names that MUST be
    # stashed.  Helion emits a matmul K loop as
    # ``acc = 0; for k: acc_copy = acc; ...; acc_<n> = acc_copy_0 + reduce`` and
    # an implicit phi makes the post-loop ``acc`` equal the loop output
    # ``acc_<n>`` (collapsed by a later rename pass).  A name X is carried when:
    #   * X is written by a non-unduplicatable region statement (its ``acc = 0``
    #     seed) and read after the region (lane-varying), and
    #   * an unduplicatable statement's body re-derives X — its forward slice
    #     from X reaches a write whose SSA-stripped name equals X (``acc_1`` /
    #     ``acc_copy`` -> ``acc``).
    # The forward-slice condition distinguishes a true accumulator (``acc``,
    # rewritten each K step) from a plain input that the matmul merely reads
    # (``indices_2``, unchanged through the loop).
    seed_writes: set[str] = set()
    for i, stmt in enumerate(region):
        if i in undup_indices:
            continue
        seed_writes |= set(ReadWrites.from_ast(stmt).writes)
    # The carried accumulator's *name* (``acc`` from ``acc = 0``) is not itself
    # lane-varying — only the loop output alias (``acc_1``) is — so do NOT filter
    # candidates by ``lane_varying`` here.  The re-derives check confirms the
    # matmul transforms the accumulator, and below we require the re-deriving
    # statement to be lane-varying so a genuinely lane-invariant accumulator is
    # left alone.
    candidate_carried = seed_writes & tail_reads
    carried: set[str] = set()
    for name in candidate_carried:
        for i in undup_indices:
            if not _undup_stmt_rederives(body[i], name):
                continue
            stmt_reads = set(ReadWrites.from_ast(body[i]).reads)
            if lane_var in stmt_reads or bool(stmt_reads & lane_varying):
                carried.add(name)
                break
    # Also stash any value DIRECTLY produced by an unduplicatable statement that
    # is read by the tail (e.g. a matmul whose output is a fresh name rather than
    # an accumulator phi).
    undup_writes: set[str] = set()
    for i in undup_indices:
        undup_writes |= set(ReadWrites.from_ast(body[i]).writes)
    direct = undup_writes & lane_varying & tail_reads
    stash_names = sorted(carried | direct)
    if not stash_names:
        return None

    stash_set = set(stash_names)

    # Correctness gate: the register-stash lowering reduces each marker OVER THE
    # LANE axis (stash per lane, then sum lanes + warp-reduce).  That is only
    # valid when the lane-distributed axis IS the reduced axis — i.e. the genuine
    # matmul_layernorm pattern, where the matmul OUTPUT free dim is split across
    # the lane and the layernorm reduces exactly that dim, then broadcasts the
    # reduced scalar back into a *per-lane* normalize+store (each lane keeps a
    # distinct, stash-derived output value).
    #
    # A different pattern — e.g. ``baddbmm(...).sum(-1)`` over a small static dim
    # — folds the reduced axis into the matmul itself, leaving each lane holding a
    # COMPLETE, distinct output element.  There the lane axis is a *preserved*
    # output dim, the spurious marker reduces over the wrong (lane) axis, and the
    # store writes the single reduced scalar to lane-varying addresses (every lane
    # storing the same collapsed value).  Detect that here and bail to the
    # correct per-lane path: a side-effecting store whose ADDRESS depends on the
    # lane var but whose stored VALUE depends only on reduced marker results (no
    # per-lane stash output) means the lane axis was wrongly collapsed.
    marker_result_names = {m.result_var for _, m in markers}
    if _lane_axis_wrongly_collapsed(
        tail, lane_var, lane_varying, stash_set, marker_result_names
    ):
        return None
    # Region statements that are *duplicatable* and can be recomputed cheaply in
    # the later passes (e.g. ``indices_2 = thread_idx[0] + lane * 4``).  Drop the
    # unduplicatable statements and any statement that produces a stashed name.
    recompute: list[ast.AST] = []
    for i, stmt in enumerate(region):
        if i in undup_indices:
            continue
        writes = set(ReadWrites.from_ast(stmt).writes)
        if writes & stash_set:
            continue
        recompute.append(stmt)
    # Keep only the recompute statements that (transitively) feed the tail.
    recompute_keep_idx, _ = _backward_slice(recompute, tail_reads, rename_groups)
    recompute_kept = [recompute[i] for i in recompute_keep_idx]
    # The recompute slice must not pull in an unduplicatable op or reference a
    # stashed name (which is only available from the fragment, not recomputable).
    for s in recompute_kept:
        if _contains_unduplicatable_op(s):
            return None

    # Allocate one register fragment per stashed value.
    uid = _next_lane_stash_uid()
    frag_by_name: dict[str, str] = {}
    decls: list[ast.AST] = []
    for name in stash_names:
        frag = f"_lane_stash_{uid}_{name}"
        frag_by_name[name] = frag
        dtype = _stash_dtype_for_value(name, body, markers)
        decls.append(
            statement_from_string(f"{frag} = cute.make_rmem_tensor({extent}, {dtype})")
        )

    def read_stash_stmts() -> list[ast.AST]:
        return [
            statement_from_string(f"{name} = {frag_by_name[name]}[{lane_var}]")
            for name in stash_names
        ]

    # Phase 0: run the compute region once and stash the live-out values.
    phase0_body: list[ast.AST] = list(region)
    for name in stash_names:
        phase0_body.append(
            statement_from_string(f"{frag_by_name[name]}[{lane_var}] = {name}")
        )

    # The marker assignments within the tail produce already-finalized scalars
    # (computed once after each reduction pass), so they must NOT be re-run as
    # per-lane passthroughs inside any later pass.  Each marker's input is
    # re-derived from the non-marker tail statements before it (a later
    # rewrite of the input's group consumes the reduction's input), with the
    # finalized marker result vars as pre-defined boundaries of the slices.
    marker_result_vars = {m.result_var for _, m in markers}

    result: list[ast.AST] = []
    result.extend(decls)
    result.append(_clone_lane_loop_with_body(loop, phase0_body))

    # Process each marker in source order (they are sequentially dependent: a
    # later marker's input may read an earlier marker's finalized scalar).
    for marker_index, m in markers:
        acc_var = f"{m.result_var}_lane_acc"
        result.append(statement_from_string(f"{acc_var} = {m.identity_expr}"))
        # Accumulate pass: recompute cheap region producers, read the stash,
        # then re-derive this marker's input and fold it into the accumulator.
        # The slice runs over the re-derivable tail before the marker only;
        # references to other markers' results stop at those finalized scalars.
        rederivable_before = [
            s
            for index, s in enumerate(tail, tail_offset)
            if index < marker_index and _is_lane_reduce_marker_assign(s) is None
        ]
        input_slice_idx, _ = _backward_slice(
            rederivable_before, {m.input_name}, rename_groups
        )
        input_stmts = [
            rederivable_before[i]
            for i in input_slice_idx
            if not _assigns_simple_name(rederivable_before[i], marker_result_vars)
        ]
        acc_body: list[ast.AST] = []
        acc_body.extend(_rerun_copy(s) for s in recompute_kept)
        acc_body.extend(read_stash_stmts())
        acc_body.extend(_rerun_copy(s) for s in input_stmts)
        ctor = _dtype_ctor_from_identity(m.identity_expr)
        combine_val = f"{ctor}({m.input_name})" if ctor is not None else m.input_name
        acc_body.append(
            statement_from_string(
                f"{acc_var} = {_combine_expr(m.reduction_type, acc_var, combine_val, ctor)}"
            )
        )
        result.append(_clone_lane_loop_with_body(loop, acc_body))
        result.extend(_finalize_lane_reduce_marker(m, acc_var))

    # Final consume pass: everything in the tail except the marker assignments,
    # re-derived from the stash + finalized scalars.  Only keep statements that
    # feed a side effect (a store / in-place write).
    consume_candidates = [s for i, s in enumerate(body) if i >= tail_offset]
    consume_candidates = [
        s for s in consume_candidates if _is_lane_reduce_marker_assign(s) is None
    ]
    keep_idx = _live_phase2_indices(consume_candidates)
    consume_kept = [s for i, s in enumerate(consume_candidates) if i in keep_idx]
    if consume_kept:
        consume_body: list[ast.AST] = []
        consume_body.extend(_rerun_copy(s) for s in recompute_kept)
        consume_body.extend(read_stash_stmts())
        consume_body.extend(deep_clone(s) for s in consume_kept)
        result.append(_clone_lane_loop_with_body(loop, consume_body))
    return result


def _markers_feed_cross_lane_carry(
    body: list[ast.AST],
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
) -> bool:
    """Return True when the lane loop carries an accumulator across the lanes
    that a marker result feeds (online-softmax attention's ``m_i`` / ``l_i`` /
    ``acc`` recurrence).

    Helion represents a loop-carried value with a phi ``X_copy = X`` read at the
    TOP of the loop body (before the value is rewritten) and a corresponding
    output assignment renamed back to ``X`` by a later pass. Such a copy
    conservatively excludes the register-stash path, which assumes each marker
    result is consumed only by per-lane / store consumers. It does not prove
    completeness for a raw-input restore; owned markers still require a
    complete reduction lowering.

    matmul_layernorm's N-output lane loop has no such top-level ``X_copy = X``
    carry (its ``acc`` is the matmul accumulator, carried by the *inner* K loop,
    not across the lane iterations), so this returns False and the stash path is
    free to run.
    """
    from ...ast_read_writes import ReadWrites

    if not markers:
        return False

    # Live-in names of the lane body (read before written), excluding the lane
    # var.  A loop-carried accumulator phi is live-in.
    written_so_far: set[str] = set()
    live_in: set[str] = set()
    for stmt in body:
        rw = ReadWrites.from_ast(stmt)
        for name in rw.reads:
            if name != lane_var and name not in written_so_far:
                live_in.add(name)
        written_so_far |= set(rw.writes)

    # Detect top-level phi copies ``X_copy = X`` of a live-in accumulator.
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
        ):
            src = stmt.value.id
            dst = stmt.targets[0].id
            if src in live_in and _strip_ssa_suffix(dst) == _strip_ssa_suffix(src):
                return True
    return False


def _has_extra_cross_lane_carry(
    body: list[ast.AST],
    lane_var: str,
    marker_indices: set[int],
    *,
    finalized_markers: bool = False,
) -> bool:
    """Return True when ``body`` contains a loop-carried accumulator across the
    lanes that is INDEPENDENT of the reduction markers.

    Two-pass splitting is correct whenever every cross-lane carried value
    transitively consumes a marker result (e.g. an online-softmax
    ``mi = max(mi, local_amax)`` or ``di = di + sum``): after the split the
    carried update runs once per outer iteration on the fully-reduced scalar.
    But an *independent* carried accumulator — one that does not depend on any
    marker result, such as a matmul ``dot_acc += dot_product`` — must keep
    accumulating once per lane, so the split would drop or corrupt it. In that
    case the caller falls back to the single-pass per-lane behavior.

    For owned-marker legality, ``finalized_markers`` checks variation after
    treating every marker result as a completed, lane-invariant reduction.
    Merely depending on a marker is insufficient: ``carry + partial + reduced``
    still contains an independent lane-varying contribution. Only an update
    that becomes lane-invariant after finalization can move to the tile tail.
    The default retains the legacy unowned-marker predicate.

    A carried value is detected as a name that is *live-in* to the lane body
    (read before it is written within the body, directly or through a
    ``X_copy = X`` alias) and also written within the body.
    """
    from ...ast_read_writes import ReadWrites

    definitely_written_so_far: set[str] = set()
    aliases: dict[str, str] = {}  # copy_var -> original carried name
    live_in: set[str] = set()
    for stmt in body:
        effects = _ordered_statement_reads_writes(stmt)
        for name in effects.reads_before_write:
            if name != lane_var and name not in definitely_written_so_far:
                live_in.add(name)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
        ):
            aliases[stmt.targets[0].id] = stmt.value.id
        definitely_written_so_far.update(effects.definite_writes)

    # Names that (transitively) depend on a marker result. Carried values that
    # only depend on these are fine under the two-pass split.
    marker_results = {
        m.result_var
        for i, stmt in enumerate(body)
        if i in marker_indices
        and (m := _is_lane_reduce_marker_assign(stmt)) is not None
    }
    marker_tainted = set(marker_results)
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if set(rw.reads) & marker_tainted:
                for w in rw.writes:
                    if w not in marker_tainted:
                        marker_tainted.add(w)
                        changed = True

    # Names that (transitively) depend on the lane var. A carried accumulator
    # whose per-iteration update is lane-varying (e.g. a matmul
    # ``dot_acc += dot_product(lane)``) cannot move to the once-per-tile tail,
    # so the two-pass split would break it. A lane-invariant update (e.g.
    # welford's ``acc_cnt += block_size``) is fine in the tail.
    variation_body = (
        [
            statement
            for index, statement in enumerate(body)
            if index not in marker_indices
        ]
        if finalized_markers
        else body
    )
    lane_varying = _lane_varying_names(variation_body, lane_var)

    # Bail if some carried accumulator is independent of every marker AND its
    # update consumes a lane-varying value.
    for idx, stmt in enumerate(body):
        if idx in marker_indices:
            continue
        rw = ReadWrites.from_ast(stmt)
        reads = set(rw.reads)
        update_is_lane_varying = lane_var in reads or bool(reads & lane_varying)
        if not update_is_lane_varying:
            continue
        for w in rw.writes:
            if resolve_alias(w, aliases) in live_in and (
                finalized_markers or resolve_alias(w, aliases) not in marker_tainted
            ):
                return True
    return False


class _OrderedReadWrites(NamedTuple):
    reads_before_write: set[str]
    may_writes: set[str]
    definite_writes: set[str]


def _ordered_block_reads_writes(body: list[ast.stmt]) -> _OrderedReadWrites:
    reads_before_write: set[str] = set()
    may_writes: set[str] = set()
    definitely_written_so_far: set[str] = set()
    for stmt in body:
        effects = _ordered_statement_reads_writes(stmt)
        reads_before_write.update(
            effects.reads_before_write - definitely_written_so_far
        )
        may_writes.update(effects.may_writes)
        definitely_written_so_far.update(effects.definite_writes)
    return _OrderedReadWrites(
        reads_before_write,
        may_writes,
        definitely_written_so_far,
    )


def _for_is_provably_nonempty(stmt: ast.For) -> bool:
    if any(
        isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Raise))
        for node in ast.walk(stmt)
    ):
        return False
    iterator = stmt.iter
    if not (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == "range"
        and not iterator.keywords
        and 1 <= len(iterator.args) <= 3
    ):
        return False
    values = [_int_literal(arg) for arg in iterator.args]
    if any(value is None for value in values):
        return False
    integers = cast("list[int]", values)
    try:
        return len(range(*integers)) > 0
    except ValueError:
        return False


def _ordered_statement_reads_writes(stmt: ast.AST) -> _OrderedReadWrites:
    """Return ordered reads plus may-write and definitely-write sets.

    :class:`ReadWrites` intentionally summarizes a compound statement as an
    unordered set.  That is too conservative for cross-*outer*-lane carry
    detection: locals assigned near the top of a generated serial loop and
    read later in that same loop look like live-ins, causing a valid reduction
    marker to be replaced by its per-lane value.

    A conditional write, however, cannot suppress a later read unless every
    branch definitely writes the name.  Likewise, writes in a loop only become
    definite when its generated ``range`` is statically non-empty.  Keeping
    may-writes separate preserves both properties.
    """
    from ...ast_read_writes import ReadWrites

    if isinstance(stmt, ast.If):
        test_rw = ReadWrites.from_ast(stmt.test)
        body = _ordered_block_reads_writes(stmt.body)
        orelse = _ordered_block_reads_writes(stmt.orelse)
        return _OrderedReadWrites(
            set(test_rw.reads) | body.reads_before_write | orelse.reads_before_write,
            set(test_rw.writes) | body.may_writes | orelse.may_writes,
            set(test_rw.writes) | (body.definite_writes & orelse.definite_writes),
        )

    if not isinstance(stmt, ast.For):
        rw = ReadWrites.from_ast(stmt)
        writes = set(rw.writes)
        # Other compound statements (while/try/with/match) may skip or branch
        # around writes. Treat their writes as possible only; generated plain
        # assignments and expression statements execute unconditionally.
        definite_writes = (
            set()
            if any(
                isinstance(stmt, kind)
                for kind in (ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)
            )
            else writes
        )
        return _OrderedReadWrites(set(rw.reads), writes, definite_writes)

    iter_rw = ReadWrites.from_ast(stmt.iter)
    target_rw = ReadWrites.from_ast(stmt.target)
    target_writes = set(target_rw.writes)
    body = _ordered_block_reads_writes(stmt.body)
    orelse = _ordered_block_reads_writes(stmt.orelse)
    reads_before_write = set(iter_rw.reads)
    reads_before_write.update(body.reads_before_write - target_writes)
    # A ``for ... else`` can execute its else suite without one body iteration,
    # so no body/target write is available to suppress these reads.
    reads_before_write.update(orelse.reads_before_write)
    may_writes = (
        set(iter_rw.writes) | target_writes | body.may_writes | orelse.may_writes
    )
    definite_writes = set(iter_rw.writes)
    if _for_is_provably_nonempty(stmt):
        definite_writes.update(target_writes)
        definite_writes.update(body.definite_writes)
    return _OrderedReadWrites(reads_before_write, may_writes, definite_writes)


def _identity_cast_operand(node: ast.AST, ctor: str | None) -> ast.AST:
    """``node`` below any casts to the accumulator dtype (``cutlass.Float32(x)``)."""
    while (
        ctor is not None
        and isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) == ctor
    ):
        node = node.args[0]
    return node


def _lane_carry_combine_operands(
    value: ast.AST, reduction_type: str, ctor: str | None
) -> tuple[str, str] | None:
    """The two names ``value`` combines the way the marker's reduction does.

    ``a + b`` for a sum, ``a * b`` for a product, ``cute.math.max(a, b,
    propagate_nan=...)``, ``max(a, b)`` or the ternary ``a if a > b else b``
    for a max (``min`` likewise, and a ternary spelling the other extremum
    is not a carry of this marker), each operand under any cast to the
    accumulator dtype ``ctor``; ``None`` for any other expression.
    """
    value = _identity_cast_operand(value, ctor)

    def names(left: ast.AST, right: ast.AST) -> tuple[str, str] | None:
        left = _identity_cast_operand(left, ctor)
        right = _identity_cast_operand(right, ctor)
        if isinstance(left, ast.Name) and isinstance(right, ast.Name):
            return left.id, right.id
        return None

    if reduction_type in ("sum", "prod"):
        combine = ast.Add if reduction_type == "sum" else ast.Mult
        if isinstance(value, ast.BinOp) and isinstance(value.op, combine):
            return names(value.left, value.right)
        return None
    if reduction_type not in ("max", "min"):
        return None
    if isinstance(value, ast.Call):
        if (
            ast.unparse(value.func) in (reduction_type, f"cute.math.{reduction_type}")
            and len(value.args) == 2
            and all(keyword.arg == "propagate_nan" for keyword in value.keywords)
        ):
            return names(value.args[0], value.args[1])
        return None
    if not (
        isinstance(value, ast.IfExp)
        and isinstance(value.test, ast.Compare)
        and len(value.test.ops) == 1
    ):
        return None
    compared = names(value.test.left, value.test.comparators[0])
    selected = names(value.body, value.orelse)
    if (
        compared is None
        or selected is None
        or len(set(compared)) != 2
        or set(compared) != set(selected)
    ):
        return None
    (op,) = value.test.ops
    if isinstance(op, (ast.Gt, ast.GtE)):
        body_selected_when_larger = True
    elif isinstance(op, (ast.Lt, ast.LtE)):
        body_selected_when_larger = False
    else:
        return None
    # The body is taken when the test holds: the left operand is the larger
    # one under ``>`` and the smaller under ``<``, so the ternary computes
    # the max exactly when the body names the operand the test ranks larger.
    body_is_left = selected[0] == compared[0]
    computes_max = body_is_left == body_selected_when_larger
    return compared if computes_max == (reduction_type == "max") else None


def _lane_carry_update_shape(
    stmt: ast.AST,
) -> tuple[str, list[ast.expr], set[str]] | None:
    """``(target, assigned values, names the guard reads)`` of a carry update.

    A plain assignment, or an if-join assigning one name in both branches
    (``if c: T = C + r`` / ``else: T = C``), whose test's reads are returned
    so the caller can require the join to be the same in every lane.
    """
    from ...ast_read_writes import ReadWrites

    target = assigned_name(stmt)
    if target is not None:
        assert isinstance(stmt, ast.Assign)
        return target, [stmt.value], set()
    if not (isinstance(stmt, ast.If) and len(stmt.body) == 1 and len(stmt.orelse) == 1):
        return None
    branches = [stmt.body[0], stmt.orelse[0]]
    targets = {assigned_name(branch) for branch in branches}
    if len(targets) != 1 or None in targets:
        return None
    (target,) = targets
    assert target is not None
    values = [cast("ast.Assign", branch).value for branch in branches]
    return target, values, set(ReadWrites.from_ast(stmt.test).reads)


def _restored_marker_consumers_are_lane_carries(
    body: Sequence[ast.AST],
    marker: _LaneReduceMarker,
    lane_var: str,
    rename_groups: Mapping[str, str] | None = None,
) -> bool:
    """Whether every reader of ``marker``'s result folds it into a lane carry.

    A ``strided_restore`` marker finalized per lane yields the lane's share
    of the reduction: its thread group's elements of this lane.  That share
    is a complete lowering only through a carry across the lanes: a reader
    ``T = C + result`` (``*`` for a product, ``cute.math.max`` / ``max`` /
    the ternary for a max or min, the operands under casts to the
    accumulator dtype) whose other operand ``C`` is the phi copy ``C = X``
    of a value live into the body, where ``T`` is ``X`` under the rename
    groups, and no other statement reads or writes the carried value, its
    copy or its loop-output spelling.  The update may sit in an if-join
    whose other branch keeps the carry (``T = C``) and whose condition is
    the same in every lane (it reads no lane-varying name).  Once the loop
    ends the carry holds the shares of every lane, which is the reduction
    over the tile (jagged_layer_norm's ``mean_acc = mean_acc +
    row_sums.sum(dim=1)``; the max over the lanes of the lanes' maxes).
    Any other reader -- a store of the result, a product with the per-lane
    values, a chain into a further reduction -- would observe one lane's
    share.  A result nothing reads is complete.
    """
    from ...ast_read_writes import ReadWrites

    renames = rename_groups or {}

    def canonical(names: Iterable[str]) -> set[str]:
        return {renames.get(name, name) for name in names}

    ctor = _dtype_ctor_from_identity(marker.identity_expr)
    live_in = canonical(
        _ordered_block_reads_writes(cast("list[ast.stmt]", body)).reads_before_write
    )
    varying = _lane_varying_names(
        [stmt for stmt in body if _is_lane_reduce_marker_assign(stmt) is None],
        lane_var,
        renames,
    )
    result = marker.result_var
    for index, stmt in enumerate(body):
        rw = ReadWrites.from_ast(stmt)
        if result not in rw.reads:
            continue
        shape = _lane_carry_update_shape(stmt)
        if shape is None:
            return False
        target, values, guard_reads = shape
        if set(rw.writes) != {target} or guard_reads & ({result, lane_var} | varying):
            return False
        copy_names: set[str] = set()
        for value in values:
            operands = _lane_carry_combine_operands(value, marker.reduction_type, ctor)
            if operands is None:
                # The branch of an if-join that keeps the carry.
                kept = _identity_cast_operand(value, ctor)
                if len(values) == 1 or not isinstance(kept, ast.Name):
                    return False
                copy_names.add(kept.id)
                continue
            if result not in operands or len(set(operands)) != 2:
                return False
            (copy_name,) = set(operands) - {result}
            copy_names.add(copy_name)
        if len(copy_names) != 1:
            return False
        (copy_name,) = copy_names
        # The phi copies of the carried value (``mean_acc_copy = mean_acc``,
        # ``mean_acc_copy_0 = mean_acc_copy``) lead back to a name the body
        # does not define.
        chain: list[ast.AST] = []
        names = {copy_name}
        name = copy_name
        while True:
            definitions = [
                candidate
                for candidate in body[:index]
                if assigned_name(candidate) == name
            ]
            if not definitions:
                break
            if len(definitions) != 1:
                return False
            (copy,) = definitions
            assert isinstance(copy, ast.Assign)
            if not isinstance(copy.value, ast.Name) or copy.value.id in names:
                return False
            chain.append(copy)
            name = copy.value.id
            names.add(name)
        carried = canonical({name})
        if carried != canonical({target}) or not carried <= live_in:
            return False
        group = carried | names
        if canonical(guard_reads) & group:
            return False
        for other in body:
            if other is stmt or any(other is copy for copy in chain):
                continue
            other_rw = ReadWrites.from_ast(other)
            if (canonical(other_rw.reads) | canonical(other_rw.writes)) & group:
                return False
    return True


def _restore_lane_markers(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    rename_groups: dict[str, str],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
) -> list[ast.AST]:
    """Lower a lane loop the two-pass split declined, keeping it whole.

    The per-lane restore (:func:`_restore_per_lane_markers`) is complete when
    every strided marker's result feeds a lane carry.  Otherwise the compiler
    supplies the carry itself: each lane's share is totalled across the lanes
    and the lane-invariant tail consuming the results runs once after the
    loop (:func:`_rereduce_restored_lane_markers`).  A body that needs a
    result inside the lanes (a consumer varying with the lane) has no
    whole-loop lowering and is declined.
    """
    body: list[ast.AST] = list(loop.body)
    if all(
        marker.owner_lane is None
        or (
            marker.strided_restore
            and _restored_marker_consumers_are_lane_carries(
                body, marker, lane_var, rename_groups
            )
        )
        for _, marker in markers
    ):
        return [_restore_per_lane_markers(loop, markers, rename_groups)]
    rereduced = _rereduce_restored_lane_markers(
        loop,
        lane_var,
        markers,
        rename_groups,
        thread_axis_names,
        scalar_definitions,
    )
    if rereduced is not None:
        return rereduced
    raise exc.BackendUnsupported(
        "cute", "owned lane reduction has no proved complete per-lane restore"
    )


def _rereduce_restored_lane_markers(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    rename_groups: dict[str, str],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
) -> list[ast.AST] | None:
    """Finalize the strided markers per lane and total the shares over the lanes.

    The loop runs whole, as the restore runs it (its carries, collectives
    and serial loops in place), but each marker's per-lane result is folded
    into a total the loop carries, and the body's tail -- every statement
    after the first marker, which must be lane-invariant once the results
    are final -- runs once after the loop on the totals, as the two-pass
    split runs its invariant tail (stores owner-guarded).  A pure assignment
    between the markers that reads only prefix values (the masked input of a
    second reduction of one accumulator under dynamic shapes, ``_mask_to_2 =
    row_sums if mask else -inf``) joins the prefix, so both reductions read
    inputs produced before the first marker.  ``None`` when the tail is not
    of that shape: a consumer varying with the lane (``out = row_sums *
    total[:, None]`` needs every lane's values after the total, which only
    the two-pass and stash lowerings provide), a carry update (its
    accumulation over the lanes would run once), a marker input produced
    after the first marker by a statement that cannot join the prefix, or a
    tail reading a prefix value no pure lane-invariant assignment re-derives
    after the loop.
    """
    from ...ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    marker_by_index = dict(markers)
    if any(
        marker.owner_lane is None or not marker.strided_restore
        for marker in marker_by_index.values()
    ):
        return None

    def canonical(names: Iterable[str]) -> set[str]:
        return {rename_groups.get(name, name) for name in names}

    first = min(marker_by_index)
    results = {marker.result_var for marker in marker_by_index.values()}
    hoisted = _prefix_only_statements_between_markers(
        body, marker_by_index, canonical, results
    )
    for index, marker in markers:
        produced_between = {
            name
            for between in range(first, index)
            if between not in marker_by_index and between not in hoisted
            for name in ReadWrites.from_ast(body[between]).writes
        }
        if marker.input_name in results or (
            canonical({marker.input_name}) & canonical(produced_between)
        ):
            return None
    non_markers = [
        stmt for index, stmt in enumerate(body) if index not in marker_by_index
    ]
    varying = _lane_varying_names(non_markers, lane_var, rename_groups)

    def is_varying(stmt: ast.AST) -> bool:
        rw = ReadWrites.from_ast(stmt)
        reads = canonical(rw.reads)
        writes = canonical(rw.writes)
        return lane_var in reads or bool(reads & varying) or bool(writes & varying)

    effects = _ordered_block_reads_writes(cast("list[ast.stmt]", body))
    carried = (
        canonical(effects.reads_before_write) & canonical(effects.may_writes)
    ) - {lane_var}
    tail = [
        body[index]
        for index in range(first, len(body))
        if index not in marker_by_index and index not in hoisted
    ]
    tail_reads: set[str] = set()
    for stmt in tail:
        rw = ReadWrites.from_ast(stmt)
        if is_varying(stmt) or (canonical(rw.reads) | canonical(rw.writes)) & carried:
            return None
        tail_reads |= set(rw.reads)
    prefix = [*body[:first], *(body[index] for index in sorted(hoisted))]
    rederived_indices, _ = _backward_slice(prefix, tail_reads - results, rename_groups)
    rederived = [prefix[index] for index in rederived_indices]
    for stmt in rederived:
        rw = ReadWrites.from_ast(stmt)
        if (
            is_varying(stmt)
            or (canonical(rw.reads) | canonical(rw.writes)) & carried
            or not _is_proven_relocatable_assignment(stmt, allow_load=False)
        ):
            return None

    totals: list[ast.AST] = []
    loop_body: list[ast.AST] = list(prefix)
    finalized: list[ast.AST] = []
    for index in range(first, len(body)):
        marker = marker_by_index.get(index)
        if marker is None:
            continue
        share = f"{marker.result_var}_lane_share"
        total = f"{marker.result_var}_lane_total"
        totals.append(statement_from_string(f"{total} = {marker.identity_expr}"))
        # The per-lane finalize of the restore, assigned to the share instead
        # of the result (the wrap, a cast or reshape, is applied to the total).
        per_lane = dataclasses.replace(
            marker, result_var=share, wrap_template="__HELION_FINALIZED__"
        )
        loop_body.extend(_finalize_lane_reduce_marker(per_lane, marker.input_name))
        ctor = _dtype_ctor_from_identity(marker.identity_expr)
        value = f"{ctor}({share})" if ctor is not None else share
        loop_body.append(
            statement_from_string(
                f"{total} = {_combine_expr(marker.reduction_type, total, value, ctor)}"
            )
        )
        finalized.append(
            statement_from_string(
                f"{marker.result_var} = {marker.finalize_expr(total)}"
            )
        )
    result: list[ast.AST] = [*totals, _clone_lane_loop_with_body(loop, loop_body)]
    result.extend(finalized)
    result.extend(deep_clone(stmt) for stmt in rederived)
    marker_dependencies = [
        (marker, _forward_live_names(body, {marker.result_var}))
        for _, marker in markers
    ]
    thread_axes_before, scalar_defs_before = _statement_provenance_before(
        body, thread_axis_names, scalar_definitions, rename_groups
    )
    for stmt in tail:
        owner_exprs = _lane_reduction_owner_exprs_for_statement(
            stmt,
            markers,
            marker_dependencies,
            thread_axes_before.get(id(stmt), thread_axis_names),
            scalar_defs_before.get(id(stmt), {}),
        )
        if owner_exprs:
            predicate = " and ".join(f"({owner_expr})" for owner_expr in owner_exprs)
            stmt = _guard_stmt_with_owner(stmt, [predicate])
        result.append(stmt)
    return result


def _prefix_only_statements_between_markers(
    body: Sequence[ast.AST],
    marker_by_index: Mapping[int, _LaneReduceMarker],
    canonical: Callable[[Iterable[str]], set[str]],
    results: set[str],
) -> set[int]:
    """Indices of the statements between the markers that may join the prefix.

    A statement moves before the first marker when it is a pure assignment
    (:func:`_is_proven_relocatable_assignment` without loads) that reads no
    marker result and nothing a statement staying between the markers
    writes, and writes nothing such a statement reads or writes; the
    statements that stay keep their order, so each decision only compares
    against the markers and the stayers before it.
    """
    from ...ast_read_writes import ReadWrites

    first, last = min(marker_by_index), max(marker_by_index)
    hoisted: set[int] = set()
    staying_reads: set[str] = set()
    staying_writes: set[str] = canonical(results)
    for index in range(first, last + 1):
        stmt = body[index]
        rw = ReadWrites.from_ast(stmt)
        reads, writes = canonical(rw.reads), canonical(rw.writes)
        if (
            index not in marker_by_index
            and _is_proven_relocatable_assignment(stmt, allow_load=False)
            and not reads & staying_writes
            and not writes & (staying_reads | staying_writes)
        ):
            hoisted.add(index)
            continue
        staying_reads |= reads
        staying_writes |= writes
    return hoisted


def _restore_per_lane_markers(
    loop: ast.For,
    markers: list[tuple[int, _LaneReduceMarker]],
    rename_groups: Mapping[str, str] | None = None,
    *,
    lane_var: str | None = None,
) -> ast.For:
    """Keep the lane loop whole when a two-pass split is unsafe.

    A carry or unduplicatable producer explains why a split is unsafe, not why
    its reduction can be omitted. A legacy marker's input is finalized in
    place. A production marker denotes a full reduction over the owning serial
    lane and physical thread group, which a raw per-lane input cannot complete
    -- except for a ``strided_restore`` marker whose loop body kept the
    per-element strided semantics and whose result feeds lane carries only
    (:func:`_restored_marker_consumers_are_lane_carries`: a sum or product
    update, a running max or min): finalizing this lane's raw input across
    the thread group (the strided form the split was going to replace)
    yields the lane's share, and the carry accumulates the shares of every
    lane into the reduction over the tile.  Any other consumer of a strided
    marker would read one lane's share (a stored total, a product with the
    per-lane values, an atomic add of the total), so such a marker is
    declined here; :func:`_restore_lane_markers` totals the shares across
    the lanes for a lane-invariant tail instead.  The interchange proof removes its
    already-materialized marker consumers separately.

    ``loop`` is the lane loop itself, or (``lane_var`` given) a serial loop
    nested in the lane loop ``lane_var`` whose body holds the markers.
    """
    body = list(loop.body)
    if lane_var is None:
        assert isinstance(loop.target, ast.Name)
        lane_var = loop.target.id
    if any(
        marker.owner_lane is not None
        and (
            not marker.strided_restore
            or not _restored_marker_consumers_are_lane_carries(
                body, marker, lane_var, rename_groups
            )
        )
        for _, marker in markers
    ):
        raise exc.BackendUnsupported(
            "cute", "owned lane reduction has no proved complete per-lane restore"
        )
    for idx, m in sorted(markers, key=operator.itemgetter(0), reverse=True):
        if m.owner_lane is None:
            body[idx] = statement_from_string(
                f"{m.result_var} = {m.finalize_expr(m.input_name)}"
            )
        else:
            body[idx : idx + 1] = cast(
                "list[ast.stmt]", _finalize_lane_reduce_marker(m, m.input_name)
            )
    loop.body = body
    return loop


_UNDUPLICATABLE_CALLS = (
    "_cute_grouped_reduce_shared_two_stage",
    "_cute_grouped_reduce_shared_tree",
    "_cute_grouped_reduce_warp",
    "cute.gemm",
    "warp_reduction",
)


def _contains_unduplicatable_op(stmt: ast.AST) -> bool:
    """Return True when ``stmt`` (or any nested statement) contains a matmul /
    collective whose shared-memory side effects make it unsafe to re-run in a
    second lane pass."""
    src = ast.unparse(stmt)
    return any(call in src for call in _UNDUPLICATABLE_CALLS)


def _is_fold_dot_assignment(stmt: ast.AST) -> bool:
    """``name = _cute_fold_dot(...)``: a stash-worthy pure per-lane value."""
    return (
        assigned_name(stmt) is not None
        and isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
        and call_path_str(stmt.value.func) == _FOLD_DOT_HELPER
    )


def _lane_split_reorders_aliasing_memory(
    body: list[ast.AST],
    markers: list[tuple[int, _LaneReduceMarker]],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    lane_var: str,
    lane_extent: int | None,
    proven_tensor_stride_values: dict[tuple[str, int], int],
    *,
    additional_inputs: list[tuple[int, str]] | None = None,
    rename_groups: Mapping[str, str] | None = None,
) -> bool:
    """Whether splitting the repeated lane loop can reorder an aliasing write.

    A multi-pass lane split omits stores from its accumulation passes.  Writes
    on either side of a producer load are barriers: a lexically later write
    executes before the next iteration's load.  The narrow exception is a later
    exact load/store pair with the same injective lane mapping.  Distinct
    generated tensor names do not establish disjointness because two user
    arguments may be the same tensor or overlapping views.
    """
    from ...ast_read_writes import ReadWrites
    from ..fuse_two_pass_loads import _contains_arch_attribute
    from ..fuse_two_pass_loads import _is_store_call
    from ..fuse_two_pass_loads import _store_tensor_roots
    from ..fuse_two_pass_loads import _tensor_arg_roots
    from ..persistent_branch_vec import _definition_snapshots
    from ..persistent_branch_vec import _generated_access_pointer
    from ..persistent_branch_vec import _lane_accesses_are_iteration_independent
    from ..persistent_branch_vec import _memory_load_calls

    tensor_names = {
        node.value.id
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }

    writes: list[tuple[int, ast.Call, frozenset[str] | None]] = []
    for write_index, stmt in enumerate(body):
        for call in (
            node
            for node in ast.walk(stmt)
            if isinstance(node, ast.Call) and _is_store_call(node)
        ):
            writes.append((write_index, call, _store_tensor_roots(call, tensor_names)))
    if not writes:
        # Without a write in the body the split has nothing to reorder, so a
        # load whose address is not visible below is no reason to refuse it.
        return False

    address_snapshots: dict[frozenset[str], list[dict[str, ast.expr]] | None] = {}
    definitely_written: set[str] = set()
    live_in: set[str] = set()
    may_writes: set[str] = set()
    for statement in body:
        effects = ReadWrites.from_ast(statement)
        live_in.update(set(effects.reads) - definitely_written - {lane_var})
        may_writes.update(effects.writes)
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            definitely_written.update(effects.writes)
    loop_carried_names = live_in & may_writes

    inputs = [(index, marker.input_name) for index, marker in markers]
    inputs.extend(additional_inputs or ())
    for input_index, input_name in inputs:
        stage_indices, _ = _backward_slice(
            body[:input_index], {input_name}, rename_groups
        )
        for load_index in stage_indices:
            for load_call in _memory_load_calls(body[load_index]):
                load_marker = (
                    isinstance(load_call.func, ast.Name)
                    and load_call.func.id == "_helion_persistent_branch_vec_load"
                    and len(load_call.args) == 6
                )
                if load_marker:
                    pointer = load_call.args[4]
                else:
                    load_access = _generated_access_pointer(load_call)
                    func = load_call.func
                    if load_access is not None:
                        pointer = load_access[0]
                    elif isinstance(func, ast.Attribute) and not (
                        load_call.args and _contains_arch_attribute(func.value)
                    ):
                        pointer = func.value
                    else:
                        # A byte-packed word load or an unknown helper: the
                        # element count is not visible, so no reordering.
                        return True
                load_roots = _tensor_arg_roots(pointer, tensor_names)
                for write_index, store_call, write_roots in writes:
                    if not tensor_roots_may_alias(
                        write_roots, load_roots, proven_disjoint_tensor_pairs
                    ):
                        continue
                    if _is_persistent_branch_vec_store(store_call):
                        store_pointer: ast.AST | None = store_call.args[3]
                    else:
                        store_access = _generated_access_pointer(store_call)
                        store_pointer = (
                            None if store_access is None else store_access[0]
                        )
                    if write_index <= load_index or store_pointer is None:
                        return True
                    # Freeze only the two addresses and their complete
                    # dependency closure. Large value arithmetic cannot
                    # invalidate an otherwise compact address proof.
                    address_names = frozenset(
                        ReadWrites.from_ast(pointer).reads
                        | ReadWrites.from_ast(store_pointer).reads
                    )
                    if address_names not in address_snapshots:
                        address_snapshots[address_names] = _definition_snapshots(
                            cast("list[ast.stmt]", body),
                            required_names=set(address_names),
                        )
                    definition_snapshots = address_snapshots[address_names]
                    if definition_snapshots is None:
                        return True
                    unstable_address_names = set(loop_carried_names)
                    for crossed_statement in body[load_index + 1 : write_index + 1]:
                        unstable_address_names.update(
                            ReadWrites.from_ast(crossed_statement).writes
                        )
                    # A write before this producer load is an intra-iteration
                    # dependence and cannot move. A later exact store may cross
                    # the loop backedge only when both accesses have the same
                    # injective lane mapping.
                    if _lane_accesses_are_iteration_independent(
                        load_call,
                        store_call,
                        lane_var=lane_var,
                        lane_extent=lane_extent,
                        load_definitions=definition_snapshots[load_index],
                        store_definitions=definition_snapshots[write_index],
                        proven_tensor_stride_values=proven_tensor_stride_values,
                        loop_carried_names=unstable_address_names,
                    ):
                        continue
                    return True
    return False


def _live_phase2_indices(body: list[ast.AST]) -> set[int]:
    """Indices of statements in ``body`` that contribute to a side effect
    (directly, or by feeding a later side-effecting statement)."""
    from ...ast_read_writes import ReadWrites

    needed_names: set[str] = set()
    keep: set[int] = set()
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        rw = ReadWrites.from_ast(stmt)
        writes = set(rw.writes)
        if _has_side_effect(stmt) or (writes & needed_names):
            keep.add(idx)
            needed_names |= set(rw.reads)
    return keep


def _lane_varying_names(
    body: list[ast.AST],
    lane_var: str,
    rename_groups: Mapping[str, str] | None = None,
) -> set[str]:
    """Names whose values (transitively) depend on the lane var within ``body``.

    Names are compared through ``rename_groups`` and returned under their
    canonical spelling: a carried value a lane-varying statement rewrites
    under its loop-output name (``v_8`` for ``acc``) is lane-varying under
    every spelling, so its initializer ``acc = 0`` is one of its lane-varying
    writes and runs in every pass of a lane split.
    """
    from ...ast_read_writes import ReadWrites

    renames = rename_groups or {}

    def canonical(names: Iterable[str]) -> set[str]:
        return {renames.get(name, name) for name in names}

    varying = {lane_var}
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if canonical(rw.reads) & varying:
                written = canonical(rw.writes) - varying
                if written:
                    varying |= written
                    changed = True
    varying.discard(lane_var)
    return varying


def _is_serial_for(stmt: ast.AST) -> bool:
    """Return True when ``stmt`` is an ordinary serial ``for`` loop (a device
    serial loop), NOT a per-thread lane loop.

    A ``cutlass.range_constexpr`` unroll (the constexpr vector lane of a
    ``cute_vector_widths`` partition) is a lane dimension as well, never a
    serial device loop, so markers nested in it are left to the lane split.
    """
    return (
        isinstance(stmt, ast.For)
        and getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None) is None
        and not _is_constexpr_lane_iter(stmt)
    )


def _forward_live_names(
    body: list[ast.AST],
    roots: set[str],
    rename_groups: Mapping[str, str] | None = None,
) -> set[str]:
    """Names produced by the forward slice that (transitively) consumes any
    name in ``roots`` within ``body``, compared through ``rename_groups``
    (canonical names, see ``_backward_slice``) when given."""
    from ...ast_read_writes import ReadWrites

    renames = rename_groups or {}
    tainted = {renames.get(name, name) for name in roots}
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if {renames.get(name, name) for name in rw.reads} & tainted:
                for write in rw.writes:
                    name = renames.get(write, write)
                    if name not in tainted:
                        tainted.add(name)
                        changed = True
    return tainted


def interchange_lane_outside_serial_reductions(
    body: list[ast.AST],
    *,
    proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
    protected_names: set[str] | None = None,
    proven_tensor_stride_values: Mapping[tuple[str, int], int] | None = None,
    constexpr_value: Callable[[str], int | None] | None = None,
    rename_groups: Mapping[str, str] | None = None,
) -> list[ast.AST]:
    """Interchange a ``for LANE: ... for MB: ...`` nest whose inner serial loop
    contains ``_helion_lane_reduce`` markers.

    layer_norm_bwd / rms_norm_bwd compute, inside a serial ``mb`` loop, BOTH a
    per-feature accumulator that must keep the lane loop OUTSIDE the ``mb`` loop
    (``grad_w_acc += ...``) AND a feature reduction whose result is broadcast
    back into a per-row store (``grad_x``), which needs every lane summed *per*
    ``mb`` iteration (lane INSIDE ``mb``). A single lane loop cannot satisfy
    both nestings, so emit two specialized loop nests:

    * Nest B (grad_w): the original ``for LANE: ... for MB: ...`` loop. Remove
      reduction-consuming stores and their dead producers when aliasing and
      exact overwrite coverage are proven; otherwise retain the partial stores,
      unless a load names the tensor one writes (an in-place update): such a
      store goes once its overwrite is proven, and the nest is declined when
      it cannot, or when a load of Nest B left behind would read what it
      wrote (``drop_read_back_partial_stores``). A reduction-consuming atomic
      is always removed (Nest A applies the full reduction once; the lanes'
      partials must not be added on top), and the nest is declined when it
      cannot be, or when a load of Nest B after the lane prefix may read
      what it updates (``decline_stale_reads``); Nest B itself goes when that
      leaves it empty.
    * Nest A (grad_x): a ``for MB: ... for LANE: ...`` loop carrying only the
      lane reduction and its broadcast consumer. Its inner lane loop still holds
      the markers so the subsequent ``split_lane_loop_reductions`` pass produces
      the per-``mb`` accumulate -> warp-combine -> consume structure. The nest
      is declined when Nest A would need a lane-varying value its ``mb`` loop
      carries from one iteration to the next (a lane's running total the
      consumer stores), which Nest A's order would carry from lane to lane.

    Names one pending codegen rename merges (``rename_groups``, each name's
    shared name) are one variable there: ``v_27 = scale_copy * 2.0``, which
    the loop's carry renames to ``scale``, updates ``scale``.

    Returns ``body`` unchanged when no such pattern is present.
    """
    new_body: list[ast.AST] = []
    for stmt in body:
        new_body.extend(
            _interchange_stmt(
                stmt,
                proven_disjoint_tensor_pairs or set(),
                protected_names or set(),
                proven_tensor_stride_values or {},
                constexpr_value,
                rename_groups or {},
            )
        )
    return new_body


def _interchange_stmt(
    stmt: ast.AST,
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    protected_names: set[str],
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    constexpr_value: Callable[[str], int | None] | None,
    rename_groups: Mapping[str, str],
) -> list[ast.AST]:
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(
                stmt,
                field,
                interchange_lane_outside_serial_reductions(
                    old,
                    proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                    protected_names=protected_names,
                    proven_tensor_stride_values=proven_tensor_stride_values,
                    constexpr_value=constexpr_value,
                    rename_groups=rename_groups,
                ),
            )
    lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
    if (
        lane_var is None
        or not isinstance(stmt, ast.For)
        or not isinstance(stmt.target, ast.Name)
        or stmt.target.id != lane_var
    ):
        return [stmt]
    return _interchange_one_lane_loop(
        stmt,
        lane_var,
        proven_disjoint_tensor_pairs,
        protected_names,
        proven_tensor_stride_values,
        constexpr_value,
        rename_groups,
    )


def _flatten_vector_reduction_lane(
    loop: ast.For, vector_loop: ast.For, lane_var: str
) -> ast.For:
    """Restore one complete scalar domain before the ordinary reduction split.

    The strategy records this ownership when it emits the reduction marker.
    Flattening preserves the original (outer, vector) visitation order. Only
    pure coordinate preparation and proved read-only packet loads may move from
    once per outer lane to once per element. Stores, staging, and unproved
    carries must decline. The existing splitter still checks all
    producer/consumer aliasing.
    """
    from ...ast_read_writes import ReadWrites

    def unsupported() -> NoReturn:
        raise exc.BackendUnsupported(
            "cute", "vector lane reduction has no complete scalar ownership proof"
        )

    def extent(node: ast.For) -> int:
        call = node.iter
        if not (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) in ("range", "cutlass.range_constexpr")
            and len(call.args) == 1
            and not call.keywords
            and isinstance(call.args[0], ast.Constant)
            and type(call.args[0].value) is int
            and call.args[0].value > 0
            and not node.orelse
        ):
            unsupported()
        return call.args[0].value

    outer_size, vector_size = extent(loop), extent(vector_loop)
    if not isinstance(vector_loop.target, ast.Name):
        unsupported()
    vector_var = vector_loop.target.id
    if vector_var == lane_var or loop.body[-1] is not vector_loop:
        unsupported()
    prefix = loop.body[:-1]
    body_rw = ReadWrites.from_list(vector_loop.body)
    body_writes = set(body_rw.writes) | set(body_rw.inplace_writes)
    prefix_writes = set(ReadWrites.from_list(prefix).writes)
    # A packet load may be replayed at each scalar coordinate only when the
    # complete element body is read-only. No store, atomic, synchronization,
    # unknown call, or vector-address mutation may cross the repeated read.
    readonly_body = all(
        assigned_name(statement) is not None
        and not _has_observable_memory_write(statement)
        and all(
            call_path_str(node.func) == "_helion_lane_reduce"
            or _is_proven_relocatable_call(node, allow_load=True)
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        )
        for statement in vector_loop.body
    )

    def readonly_packet(statement: ast.AST) -> bool:
        if not readonly_body or not isinstance(statement, ast.Assign):
            return False
        call = statement.value
        if not (
            isinstance(call, ast.Call)
            and call_path_str(call.func) == "cute.arch.load"
            and len(call.args) == 2
            and not call.keywords
        ):
            return False
        dtype = call.args[1]
        if not (
            isinstance(dtype, ast.Call)
            and call_path_str(dtype.func) == "ir.VectorType.get"
            and len(dtype.args) == 2
            and not dtype.keywords
            and isinstance(dtype.args[0], ast.List)
            and len(dtype.args[0].elts) == 1
            and isinstance(dtype.args[0].elts[0], ast.Constant)
            and type(dtype.args[0].elts[0].value) is int
            and dtype.args[0].elts[0].value == vector_size
            and call_path_str(dtype.args[1])
            in {"cutlass.Uint16.mlir_type", "cutlass.Uint32.mlir_type"}
        ):
            return False
        return all(
            node is call
            or node is dtype
            or _is_proven_relocatable_call(node, allow_load=False)
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        )

    prepared: set[str] = set()
    for statement in prefix:
        name = assigned_name(statement)
        packet = readonly_packet(statement)
        if (
            name is None
            or not (
                packet or _is_proven_relocatable_assignment(statement, allow_load=False)
            )
            or name in body_writes | prepared | {lane_var, vector_var}
            or any(
                isinstance(node, ast.Subscript)
                and not (
                    isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func)
                    in ("cute.arch.thread_idx", "cute.arch.block_idx")
                    or packet
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "stride"
                    and isinstance(node.value.value, ast.Attribute)
                    and node.value.value.attr == "layout"
                    and isinstance(node.value.value.value, ast.Name)
                    and isinstance(node.slice, ast.Constant)
                    and type(node.slice.value) is int
                    and node.slice.value >= 0
                )
                for node in ast.walk(statement)
            )
        ):
            unsupported()
        reads = set(ReadWrites.from_ast(statement).reads)
        # A prefix value cannot depend on state changed by an element, nor on
        # its own prior-iteration value. Earlier coordinate definitions are OK.
        if reads & (body_writes | (prefix_writes - prepared) | {vector_var}):
            unsupported()
        prepared.add(name)
    if body_writes & {lane_var, vector_var} or any(
        isinstance(
            node, (ast.Break, ast.Continue, ast.Return, ast.Yield, ast.While, ast.For)
        )
        for statement in vector_loop.body
        for node in ast.walk(statement)
    ):
        unsupported()
    for statement in vector_loop.body:
        marker = _is_lane_reduce_marker_assign(statement)
        if marker is not None and marker.owner_lane != lane_var:
            unsupported()

    class Coordinates(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load):
                if node.id == lane_var:
                    return expr_from_string(f"({lane_var} // {vector_size})")
                if node.id == vector_var:
                    return expr_from_string(f"({lane_var} % {vector_size})")
            return node

    flattened = [
        Coordinates().visit(deep_clone(statement))
        for statement in [*prefix, *vector_loop.body]
    ]
    return _create_lane_loop(lane_var, outer_size * vector_size, flattened)


def _without_access_regions(statement: ast.AST) -> ast.AST:
    """``statement``, with the access regions of its accesses dropped in
    place: pass it a copy.

    A region names the tile begins of the device loops around its access by
    the loop instance (``access_regions.tile_begin_symbol``), and the
    barrier pass cancels a begin two accesses share.  A copy placed in
    another loop over the same tiles would keep the original loop's begins,
    so ``x[..., tk.begin]`` in one loop and ``x[..., tk.begin + 1]`` in the
    other would look apart, though they meet across iterations.  Without
    regions the pass compares such a pair through the address maps.
    """
    for node in ast.walk(statement):
        vars(node).pop(HELION_ACCESS_REGIONS_ATTR, None)
    return statement


def _rerun_copy(statement: ast.AST) -> ast.AST:
    """A copy of a producer that another lane loop reruns beside the one
    holding it.

    The copy is a node of its own: later passes key facts by node identity
    and rewrite in place.  The accesses in a loop nested in the producer
    drop their regions (``_without_access_regions``), since the copy's loop
    runs apart from the original's: the original's tile begins would make
    accesses that meet across iterations look apart.
    """
    copy = deep_clone(statement)
    for node in ast.walk(copy):
        if _is_device_loop(node):
            _without_access_regions(node)
    return copy


def _interchange_one_lane_loop(
    loop: ast.For,
    lane_var: str,
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    protected_names: set[str],
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    constexpr_value: Callable[[str], int | None] | None,
    rename_groups: Mapping[str, str],
) -> list[ast.AST]:
    body: list[ast.AST] = list(loop.body)
    # Find the single inner serial ``for`` loop that carries lane-reduce
    # markers of THIS lane (or legacy unowned markers).  A vector tile's
    # constexpr V-loop under a register-tile owner also holds markers, but
    # they belong to the enclosing reduction lane and are lowered there
    # (``cute/register_tile_reductions.py``).
    mb_index: int | None = None
    for idx, stmt in enumerate(body):
        if (
            isinstance(stmt, ast.For)
            and (
                _is_serial_for(stmt)
                or getattr(stmt, HELION_VECTOR_REDUCTION_OWNER_ATTR, None) == lane_var
            )
            and any(
                (marker := _is_lane_reduce_marker_assign(s)) is not None
                and marker.owner_lane in (None, lane_var)
                for s in stmt.body
            )
        ):
            if mb_index is not None:
                # More than one candidate serial loop: not the simple pattern.
                return [loop]
            mb_index = idx
    if mb_index is None:
        return [loop]
    mb_loop = cast("ast.For", body[mb_index])
    if getattr(mb_loop, HELION_VECTOR_REDUCTION_OWNER_ATTR, None) == lane_var:
        if _is_constexpr_lane_iter(loop):
            # Only ``nest_reduction_lane_outside_vector_tiles`` makes a tile
            # lane's one-vector wrapper a trace-time loop, so this reduction
            # over the tile's own axis sits under a register-tile owner, whose
            # schedule lowers only the markers the synthetic reduction lane
            # owns (``cute/register_tile_reductions.py``).  ``generate_ast``
            # regenerates the kernel with the rolled nesting, where the
            # wrapper is a rolled loop and the flattening below applies.
            raise RegisterTileUnsupported(
                f"a reduction over the tile axis of lane {lane_var} is owned "
                "by that tile lane, not by the register-tile owner"
            )
        # These slots partition the SAME reduced axis, unlike an independent
        # serial row loop. Reducing each slot separately overwrites the output.
        return [_flatten_vector_reduction_lane(loop, mb_loop, lane_var)]
    lane_prefix = body[:mb_index]
    lane_suffix = body[mb_index + 1 :]
    mb_body: list[ast.AST] = list(mb_loop.body)

    markers = [
        (i, m)
        for i, s in enumerate(mb_body)
        if (m := _is_lane_reduce_marker_assign(s)) is not None
    ]
    if not markers:
        return [loop]
    marker_indices = {i for i, _ in markers}
    marker_results = {m.result_var for _, m in markers}

    # The mb body's only side effect that consumes a marker result is the
    # broadcast-reduction store (e.g. ``grad_x[mb] = ...``). Everything else in
    # the mb body / suffix (the per-feature accumulators and their stores) is
    # independent of the reduction and is handled correctly by Nest B alone.
    grad_x_seed = {m.input_name for _, m in markers} | marker_results
    grad_x_live = _forward_live_names(mb_body, grad_x_seed)

    def is_reduction_store(stmt: ast.AST) -> bool:
        return _has_side_effect(stmt) and bool(
            set(ReadWrites.from_ast(stmt).reads) & grad_x_live
        )

    if not any(is_reduction_store(s) for s in mb_body):
        # The lane reduction inside the serial loop is not consumed by a
        # broadcast store, so the interchange does not apply. Markers nested in
        # a serial loop are not reachable by ``split_lane_loop_reductions`` (it
        # only rewrites top-level lane loops), so finalize them in place like
        # an unsplittable lane loop would: a legacy unowned marker restores
        # its raw per-lane input and a ``strided_restore`` marker whose
        # consumers are lane carries folds it across its thread group, while
        # any other owned marker denotes a full reduction over the lane that
        # a raw per-lane input cannot complete and declines loudly.
        _restore_per_lane_markers(mb_loop, markers, lane_var=lane_var)
        return [loop]

    # A name is lane-varying if it (transitively) depends on the lane var across
    # the whole lane-loop body; lane-invariant names (mb bounds, masks, ...) are
    # recomputed once rather than per lane.
    lane_varying = _lane_varying_names([*lane_prefix, *mb_body, *lane_suffix], lane_var)

    def reads_lane(stmt: ast.AST) -> bool:
        reads = set(ReadWrites.from_ast(stmt).reads)
        return lane_var in reads or bool(reads & lane_varying)

    # --- Nest A (grad_x): for MB: for LANE: <reduction + broadcast store> ------
    # Backward slice from the reduction-broadcast stores and the marker inputs,
    # across both the mb body and the lane prefix. The per-feature accumulators
    # feed only the suffix stores (never the reduction store), so they are
    # naturally excluded — no carry analysis is required.
    def canonical(names: Iterable[str]) -> set[str]:
        return {rename_groups.get(name, name) for name in names}

    mb_bound_names = canonical(ReadWrites.from_ast(mb_loop.iter).reads)
    seeds = canonical(m.input_name for _, m in markers) | mb_bound_names
    results = canonical(marker_results)
    # The mb loop carries values from one iteration into the next (``scale =
    # scale * 2.0`` after the statement that reads ``scale``): a name the
    # slice needs that the body may read before assigning it
    # (``exposed_reads``) is needed at the end of the body too, so the slice
    # repeats until the names needed there stop growing.
    carried_in = exposed_reads(mb_body, renames=rename_groups)
    live_at_end = set(seeds)
    while True:
        needed = set(live_at_end)
        keep_mb_a: list[ast.AST] = []
        for idx in range(len(mb_body) - 1, -1, -1):
            stmt = mb_body[idx]
            rw = ReadWrites.from_ast(stmt)
            if idx in marker_indices:
                keep_mb_a.append(stmt)
                needed |= canonical(rw.reads) - results
                continue
            if is_reduction_store(stmt) or (canonical(rw.writes) & needed):
                keep_mb_a.append(stmt)
                needed |= canonical(rw.reads)
        if seeds | (needed & carried_in) == live_at_end:
            break
        live_at_end = seeds | (needed & carried_in)
    keep_mb_a.reverse()
    keep_prefix_a: list[ast.AST] = []
    for stmt in reversed(lane_prefix):
        rw = ReadWrites.from_ast(stmt)
        if canonical(rw.writes) & needed:
            keep_prefix_a.append(stmt)
            needed |= canonical(rw.reads)
    keep_prefix_a.reverse()

    if _has_impure_inline_asm(loop):
        # Both nests can retain the original producers. The in-place restore
        # and register-tile fallback above do not clone these effects.
        raise exc.BackendUnsupported(
            "cute",
            "impure inline assembly cannot be interchanged with a lane reduction",
        )

    # Partition kept statements into lane-invariant (run once per mb iteration)
    # vs lane-varying (recomputed per lane inside the inner lane loop).
    prefix_invariant_a = [_rerun_copy(s) for s in keep_prefix_a if not reads_lane(s)]
    prefix_varying_a = [_rerun_copy(s) for s in keep_prefix_a if reads_lane(s)]
    # Nest A's serial loop is another loop over the serial loop's tiles, at
    # another iteration than Nest B's at any moment: its copies drop the
    # access regions recorded in Nest B's loop (``_without_access_regions``).
    mb_head_a = [
        _without_access_regions(deep_clone(s)) for s in keep_mb_a if not reads_lane(s)
    ]
    mb_varying_a = [
        _without_access_regions(deep_clone(s)) for s in keep_mb_a if reads_lane(s)
    ]

    inner_lane_loop_a = _clone_lane_loop_with_body(
        loop, [*prefix_varying_a, *mb_varying_a]
    )
    mb_loop_a = create(
        ast.For,
        target=deep_clone(mb_loop.target),
        iter=deep_clone(mb_loop.iter),
        body=[*mb_head_a, inner_lane_loop_a],
        orelse=[],
        type_comment=None,
    )
    nest_a: list[ast.AST] = [*prefix_invariant_a, mb_loop_a]

    # --- Nest B (grad_w): the original lane loop with markers reverted to their
    # raw per-lane inputs. Its per-feature accumulators (lane-outside-mb) are
    # already correct; its reduction-broadcast store writes a partial (per-lane)
    # value that Nest A re-stores with the full reduction afterwards. Prune the
    # first store only after proving it is unobservable and fully overwritten.
    # When a load names its tensor (an in-place update) it is pruned once the
    # overwrite is proven, and the nest is declined when it cannot be or when
    # a load of Nest B still reads what it wrote.
    # An atomic consumer has no such overwrite: Nest A applies the full
    # reduction once, so Nest B must not apply the lanes' partials at all.
    restored_mb_body = list(mb_loop.body)
    for idx, m in markers:
        restored_mb_body[idx] = statement_from_string(
            f"{m.result_var} = {m.finalize_expr(m.input_name)}"
        )
    mb_loop.body = restored_mb_body
    nest_b = loop

    from ..interchanged_store_dce import _single_store
    from ..interchanged_store_dce import atomic_write_calls
    from ..interchanged_store_dce import carried_lane_updates
    from ..interchanged_store_dce import decline_stale_reads
    from ..interchanged_store_dce import drop_read_back_partial_stores
    from ..interchanged_store_dce import eliminate_interchanged_atomics
    from ..interchanged_store_dce import eliminate_interchanged_stores
    from ..interchanged_store_dce import overwrites_later_first_pass_write

    # Nest A runs the serial loop outside the lane loop, so a value a lane
    # carries from one serial iteration to the next (``running = running +
    # value``) would run on from lane to lane instead; a lane-invariant
    # update is the same in every lane, once per iteration.
    carried_updates = carried_lane_updates(loop, mb_body, rename_groups)
    if any(
        reads_lane(statement) and carried_updates(statement)
        for statement in keep_mb_a
        if _is_lane_reduce_marker_assign(statement) is None
    ):
        raise exc.BackendUnsupported(
            "cute",
            "interchanged lane reduction would carry a lane's value from one "
            "serial iteration to the next through every lane",
        )
    reduction_consumers = _forward_live_names(mb_body, marker_results)
    consumer_indices = {
        index
        for index, statement in enumerate(mb_body)
        if _has_side_effect(statement)
        and set(ReadWrites.from_ast(statement).reads) & reduction_consumers
    }
    # A memory write Nest A needs that does not consume the reduction (an
    # in-place update of the marker's input, ``y[...] = row * 0.5`` beside
    # ``row.sum()``) runs in both nests, where the tile program runs it once:
    # an atomic would apply twice, and Nest A would load what Nest B's copy
    # of a store already wrote.  Nest B's copy goes or the nest is declined,
    # as it is for a write in the lane prefix Nest A reruns.
    duplicated_indices = {
        index
        for index, statement in enumerate(mb_body)
        if index not in consumer_indices
        and statement in keep_mb_a
        and _has_observable_memory_write(statement)
    }
    duplicated_stores = {
        index for index in duplicated_indices if _single_store(mb_body[index])
    }
    if any(
        _has_observable_memory_write(statement) for statement in keep_prefix_a
    ) or any(
        index not in duplicated_stores and not atomic_write_calls(mb_body[index])
        for index in duplicated_indices
    ):
        raise exc.BackendUnsupported(
            "cute",
            "interchanged lane reduction would run a memory write both nests need "
            "twice",
        )
    atomic_consumers = [
        mb_body[index]
        for index in sorted(consumer_indices | duplicated_indices)
        if atomic_write_calls(mb_body[index])
    ]
    eliminate_interchanged_stores(
        nest_b,
        mb_loop,
        nest_a,
        consumer_indices | duplicated_stores,
        proven_disjoint_tensor_pairs,
        protected_names,
    )
    drop_read_back_partial_stores(
        nest_b,
        mb_loop,
        nest_a,
        [mb_body[index] for index in sorted(consumer_indices)],
        proven_disjoint_tensor_pairs,
        protected_names,
        proven_tensor_stride_values,
        constexpr_value,
        duplicated=[mb_body[index] for index in sorted(duplicated_stores)],
    )
    if any(
        statement is mb_body[index]
        for statement in mb_loop.body
        for index in duplicated_stores
    ):
        raise exc.BackendUnsupported(
            "cute",
            "interchanged lane reduction cannot drop the first pass's copy of a "
            "store both nests need",
        )
    if atomic_consumers:
        dropped = eliminate_interchanged_atomics(
            nest_b, mb_loop, nest_a, atomic_consumers, protected_names
        )
        if dropped != len(atomic_consumers):
            raise exc.BackendUnsupported(
                "cute",
                "interchanged lane reduction has an atomic consumer the first "
                "pass cannot drop",
            )
    # Nest A runs after all of Nest B: its writes land after every write left
    # in Nest B, where only Nest B's copies of the consumer stores it stores
    # again over the same elements are meant to be overwritten, its atomics
    # after every load of Nest B, and it reruns the lane prefix it needs, its
    # lane-varying part in every ``mb`` iteration.
    if overwrites_later_first_pass_write(
        nest_b,
        mb_loop,
        nest_a,
        [mb_body[index] for index in consumer_indices],
        proven_disjoint_tensor_pairs,
        protected_names,
    ):
        raise exc.BackendUnsupported(
            "cute",
            "a final-pass write of an interchanged lane reduction may overwrite "
            "a later first-pass write",
        )
    # Each statement of Nest A's serial loop head, with the lane-varying ones
    # it followed in the serial body: it now runs ahead of them in every
    # iteration.
    head_moves: list[tuple[ast.AST, list[ast.AST]]] = []
    heads, varying = iter(mb_head_a), 0
    for stmt in keep_mb_a:
        if reads_lane(stmt):
            varying += 1
        else:
            head_moves.append((next(heads), mb_varying_a[:varying]))
    decline_stale_reads(
        nest_b,
        mb_loop,
        nest_a,
        prefix_varying_a,
        lane_prefix,
        proven_disjoint_tensor_pairs,
        protected_names,
        head_moves,
        rename_groups,
    )
    if not nest_b.body:
        return nest_a

    return [nest_b, *nest_a]
