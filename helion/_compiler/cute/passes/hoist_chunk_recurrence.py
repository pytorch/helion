"""Hoist the lane-invariant part of a chunked recurrence out of its lane loop.

Role: device-body pass.  ``hoist_lane_invariant_chunk_recurrence``
restructures ``for chunk: for lane: <dot_acc recurrence>`` nests (gdn_fwd_h)
so the lane-invariant rescale, the chunk-entry stores and the final
accumulator combine run once per chunk; see the banner comment below for the
dataflow it recognizes.  Runs after the lane split
(``lane_loops/split.py``) from ``Backend.lower_owned_lane_loop``.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import cast

from .... import exc
from ...ast_extension import create
from ...ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ...tile_strategy import _clone_lane_loop_with_body
from .._ast_pass_utils import _CHUNK_REDUCTION_HELPERS
from .._ast_pass_utils import _backward_slice
from .._ast_pass_utils import _canonical_name
from .._ast_pass_utils import _dependency_names
from .._ast_pass_utils import _is_proven_relocatable_assignment
from .._ast_pass_utils import _self_addend
from .._ast_pass_utils import assigned_name
from .._ast_pass_utils import call_path_str
from ..lane_loops.markers import _int_literal
from ..lane_loops.owner_guards import _guard_stmt_with_owner
from ..lane_loops.owner_guards import _linear_thread_axis_strides
from ..lane_loops.owner_guards import _resolve_scalar_definition
from ..lane_loops.owner_guards import _store_enclosing_predicates
from ..lane_loops.owner_guards import _store_thread_axes
from ..lane_loops.owner_guards import _thread_axis_owner_exprs
from ..lane_loops.owner_guards import _update_scalar_definitions
from ..lane_loops.owner_guards import _update_thread_axis_names
from ..lane_loops.owner_guards import _validated_idempotent_store
from ..lane_loops.split import _is_serial_for
from ..lane_loops.split import _lane_varying_names
from ..lane_loops.split import _ordered_block_reads_writes

# ---------------------------------------------------------------------------
# Chunked-recurrence (GDN) lane-invariant accumulator hoist.
#
# A chunked recurrence such as gdn_fwd_h carries an accumulator ``b_h`` across a
# SERIAL chunk loop and, inside each chunk, contracts a matmul over the
# within-chunk position ``c`` (lowered as an inner lane loop).  The scalar
# matmul fallback emits a running sum because the per-chunk rescale is
# lane-invariant.  Depending on how the source spelled the update, its relevant
# dataflow inside the lane loop is either direct or separated by temporaries:
#
#       base = <lane-invariant rescale of b_h>       # e.g. b_h * decay
#       dot_acc = dot_acc + <product(c)>             # accumulate over c
#       reduced = <cross-thread reduction>(dot_acc)
#       b_h = base + reduced                          # WRONG: per-lane reassign
#
# with ``dot_acc = <identity>`` reset OUTSIDE the chunk loop.  Reassigning ``b_h``
# every lane iteration corrupts the recurrence (and any lane-invariant op that
# must read the chunk-ENTRY ``b_h``, e.g. the store of ``b_h`` and the matmul
# operand ``c_h = b_h``).  This pass restructures the nest to apply the rescale
# and the final add once per chunk:
#
#   for chunk:
#       dot_acc = <identity>                  # reset per chunk
#       <lane-invariant chunk-entry stores using b_h>
#       base = <rescale of frozen b_h>
#       for lane:
#           <producers; dot_acc = dot_acc + product(c)>
#       reduced = <cross-thread reduction>(dot_acc)
#       b_h = base + reduced                  # once per chunk
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ChunkRecurrence:
    """Dataflow description of one lane-folded, chunk-carried matmul."""

    dot_acc_var: str
    lane_idx: int
    lane_loop: ast.For
    lane_var: str
    sum_idx: int
    base_indices: tuple[int, ...]
    finalize_indices: tuple[int, ...]
    final_idx: int
    state_name: str


def _store_iterator_roots(store: ast.Call) -> set[str]:
    """Generated tensor iterator names that determine a store's allocation."""
    if not isinstance(store.func, ast.Attribute):
        return set()
    return {
        node.value.id
        for node in ast.walk(store.func.value)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }


def _additive_operands(expr: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    """Find the add below any generated one-argument dtype conversions."""
    while isinstance(expr, ast.Call) and len(expr.args) == 1 and not expr.keywords:
        expr = expr.args[0]
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return expr.left, expr.right
    return None


def _find_dot_acc_recurrence(
    lane_loop: ast.For,
    lane_var: str,
    rename_groups: dict[str, str],
    running_sums: set[str] | None,
) -> _ChunkRecurrence | None:
    """Find a lane-folded matmul that updates a chunk-carried accumulator.

    Matmul and DCE temporary names are deliberately not part of the contract.
    The recognizable structure is a scalar self-add over lane-varying input,
    followed by an additive combine of its finalized value with a
    lane-invariant value derived from the chunk-entry state.  The combine's
    result must alias that state according to the compiler's SSA rename groups.

    The dot-derived side may contain arbitrary plain-assignment setup and a
    cross-thread reduction.  This covers both ``dot(..., acc=state)`` and the
    equivalent ``state = state * decay; state = state + dot(...)`` spelling.
    """
    from ...ast_read_writes import ReadWrites

    body: list[ast.AST] = list(lane_loop.body)
    matches: list[_ChunkRecurrence] = []
    for sum_idx, sum_stmt in enumerate(body):
        dot_acc_var = assigned_name(sum_stmt)
        if dot_acc_var is None:
            continue
        if running_sums is not None and dot_acc_var not in running_sums:
            continue
        product = _self_addend(sum_stmt, dot_acc_var)
        if product is None:
            continue

        # The update must actually fold values across this synthetic lane.
        prefix_varying = _lane_varying_names(body[: sum_idx + 1], lane_var)
        product_reads = set(ReadWrites.from_ast(product).reads)
        if lane_var not in product_reads and not (product_reads & prefix_varying):
            continue

        dot_derived = {dot_acc_var}
        for final_idx in range(sum_idx + 1, len(body)):
            candidate = body[final_idx]
            candidate_name = assigned_name(candidate)
            if candidate_name is not None and isinstance(candidate, ast.Assign):
                operands = _additive_operands(candidate.value)
                if operands is not None:
                    left_reads = set(ReadWrites.from_ast(operands[0]).reads)
                    right_reads = set(ReadWrites.from_ast(operands[1]).reads)
                    left_is_dot = bool(left_reads & dot_derived)
                    right_is_dot = bool(right_reads & dot_derived)
                    if left_is_dot != right_is_dot:
                        dot_roots = left_reads if left_is_dot else right_reads
                        base_roots = right_reads if left_is_dot else left_reads
                        base_indices, _ = _backward_slice(
                            body[:sum_idx], set(base_roots)
                        )
                        base_dependencies = _dependency_names(
                            body[:sum_idx], set(base_roots)
                        )
                        base_external_varying = base_dependencies & (
                            {lane_var} | prefix_varying
                        )
                        state_name = _canonical_name(candidate_name, rename_groups)
                        carries_state = any(
                            _canonical_name(name, rename_groups) == state_name
                            for name in base_dependencies
                        )
                        base_is_invariant = all(
                            lane_var not in set(ReadWrites.from_ast(body[idx]).reads)
                            and not (
                                set(ReadWrites.from_ast(body[idx]).reads)
                                & prefix_varying
                            )
                            for idx in base_indices
                        )
                        base_is_pure = all(
                            _is_proven_relocatable_assignment(
                                body[idx], allow_load=True
                            )
                            for idx in base_indices
                        )

                        finalizer_body = body[sum_idx + 1 : final_idx]
                        finalize_indices_rel, _ = _backward_slice(
                            finalizer_body, set(dot_roots)
                        )
                        finalize_indices = tuple(
                            sum_idx + 1 + idx for idx in finalize_indices_rel
                        )
                        finalize_dependencies = _dependency_names(
                            finalizer_body, set(dot_roots)
                        )
                        finalize_external_varying = (
                            finalize_dependencies - {dot_acc_var}
                        ) & ({lane_var} | prefix_varying)
                        finalizer_is_pure = all(
                            _is_proven_relocatable_assignment(
                                body[idx],
                                allow_load=False,
                                allow_reduction=True,
                            )
                            for idx in finalize_indices
                        )
                        reaches_running_sum = dot_acc_var in finalize_dependencies

                        recognized_recurrence = bool(
                            base_roots and carries_state and reaches_running_sum
                        )
                        if recognized_recurrence and (
                            base_external_varying
                            or not base_is_invariant
                            or finalize_external_varying
                        ):
                            raise exc.BackendUnsupported(
                                "cute",
                                "chunk recurrence is not lane-invariant",
                            )
                        if recognized_recurrence and (
                            not base_is_pure
                            or not finalizer_is_pure
                            or not _is_proven_relocatable_assignment(
                                candidate, allow_load=False
                            )
                        ):
                            raise exc.BackendUnsupported(
                                "cute",
                                "chunk recurrence contains a call with unknown purity",
                            )
                        if recognized_recurrence:
                            matches.append(
                                _ChunkRecurrence(
                                    dot_acc_var=dot_acc_var,
                                    lane_idx=-1,
                                    lane_loop=lane_loop,
                                    lane_var=lane_var,
                                    sum_idx=sum_idx,
                                    base_indices=tuple(base_indices),
                                    finalize_indices=finalize_indices,
                                    final_idx=final_idx,
                                    state_name=state_name,
                                )
                            )

            rw = ReadWrites.from_ast(candidate)
            if set(rw.reads) & dot_derived:
                dot_derived.update(rw.writes)

    # Multiple candidates would require coordinated resets and post-lane
    # finalizers.  Fail closed instead of guessing which matmul owns the carry.
    if len(matches) > 1:
        raise exc.BackendUnsupported(
            "cute", "chunk recurrence has multiple candidate state updates"
        )
    if not matches:
        return None
    return matches[0]


def _single_lane_loop_in_body(
    body: list[ast.AST],
) -> tuple[int, ast.For, str] | None:
    """If ``body`` contains exactly one direct lane loop, return its index, the
    loop node, and its lane var; else ``None``."""
    found: tuple[int, ast.For, str] | None = None
    for idx, stmt in enumerate(body):
        lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
        if (
            lane_var is not None
            and isinstance(stmt, ast.For)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == lane_var
        ):
            if found is not None:
                return None
            found = (idx, stmt, lane_var)
    return found


def _find_reset_assign(body: list[ast.AST], var: str) -> int | None:
    """Index of the last ``var = <expr>`` plain assignment in ``body`` (the
    ``dot_acc`` reset emitted before the chunk loop), or ``None``."""
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == var
        ):
            return idx
    return None


def _collect_statement_provenance(
    body: list[ast.AST],
) -> tuple[
    dict[int, dict[str, frozenset[int]]],
    dict[int, dict[str, ast.AST]],
]:
    """Capture thread-axis and scalar-definition facts before every statement."""
    thread_axes_before: dict[int, dict[str, frozenset[int]]] = {}
    scalar_defs_before: dict[int, dict[str, ast.AST]] = {}

    def visit_block(
        statements: list[ast.AST],
        inherited_axes: dict[str, frozenset[int]],
        inherited_defs: dict[str, ast.AST],
    ) -> None:
        local_axes = dict(inherited_axes)
        local_defs = dict(inherited_defs)
        for statement in statements:
            thread_axes_before[id(statement)] = dict(local_axes)
            scalar_defs_before[id(statement)] = dict(local_defs)
            for field in ("body", "orelse", "finalbody"):
                child = getattr(statement, field, None)
                if isinstance(child, list) and all(
                    isinstance(child_stmt, ast.stmt) for child_stmt in child
                ):
                    visit_block(child, local_axes, local_defs)
            _update_thread_axis_names(statement, local_axes)
            _update_scalar_definitions(statement, local_defs)

    visit_block(body, {}, {})
    return thread_axes_before, scalar_defs_before


def _call_keyword_int(call: ast.Call, name: str) -> int | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return _int_literal(keyword.value)
    return None


def _chunk_recurrence_reduction_axes(
    info: _ChunkRecurrence,
    scalar_defs_before: dict[int, dict[str, ast.AST]],
) -> set[int]:
    """Physical thread axes folded by the moved cross-thread finalizer."""
    axes: set[int] = set()
    for idx in info.finalize_indices:
        statement = info.lane_loop.body[idx]
        definitions = scalar_defs_before.get(id(statement), {})
        for call in (
            node for node in ast.walk(statement) if isinstance(node, ast.Call)
        ):
            name = call_path_str(call.func)
            if name in _CHUNK_REDUCTION_HELPERS:
                if len(call.args) < 4:
                    raise exc.BackendUnsupported(
                        "cute", "cannot infer chunk recurrence reduction ownership"
                    )
                pre = _call_keyword_int(call, "pre")
                lane_expr = _resolve_scalar_definition(call.args[3], definitions)
                strides = _linear_thread_axis_strides(ast.unparse(lane_expr))
                matches = (
                    []
                    if pre is None or strides is None
                    else [axis for axis, stride in strides.items() if stride == pre]
                )
                if len(matches) != 1:
                    raise exc.BackendUnsupported(
                        "cute", "cannot infer chunk recurrence reduction ownership"
                    )
                axes.add(matches[0])
            elif name is not None and name.startswith("cute.arch.warp_reduction"):
                axes.add(0)
    return axes


def hoist_lane_invariant_chunk_recurrence(
    body: list[ast.AST],
    *,
    rename_groups: dict[str, str] | None = None,
    running_sums: set[str] | None = None,
    required: set[str] | None = None,
) -> list[ast.AST]:
    """Restructure ``for chunk: for lane: <dot_acc recurrence>`` nests so the
    lane-invariant rescale, chunk-entry stores, and final accumulator combine
    run once per chunk (see the module comment above).

    Tightly gated: only fires on a serial chunk loop whose single inner lane
    loop has the verified dataflow described by :func:`_find_dot_acc_recurrence`
    and whose running-sum reset sits in the same statement list before it.
    ``required`` names the running sums that are only exact restructured
    (``CuteDeviceFunctionState.chunk_recurrence_running_sums``); one left
    as emitted is rejected.
    """
    rewritten: set[str] = set()
    if running_sums is None or running_sums:
        thread_axes_before, scalar_defs_before = _collect_statement_provenance(body)
        body = _hoist_lane_invariant_chunk_recurrence(
            body,
            rename_groups or {},
            running_sums,
            thread_axes_before,
            scalar_defs_before,
            rewritten,
        )
    if required is not None and required - rewritten:
        raise exc.BackendUnsupported(
            "cute",
            "a matmul accumulator recurrence in a K loop is not restructured to "
            "update once per tile",
        )
    return body


def _hoist_lane_invariant_chunk_recurrence(
    body: list[ast.AST],
    aliases: dict[str, str],
    running_sums: set[str] | None,
    thread_axes_before: dict[int, dict[str, frozenset[int]]],
    scalar_defs_before: dict[int, dict[str, ast.AST]],
    rewritten: set[str],
) -> list[ast.AST]:
    from ...ast_read_writes import ReadWrites

    new_body: list[ast.AST] = []
    for stmt in body:
        # Recurse into nested statement-bearing fields first.
        for field in ("body", "orelse", "finalbody"):
            old = getattr(stmt, field, None)
            if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
                setattr(
                    stmt,
                    field,
                    _hoist_lane_invariant_chunk_recurrence(
                        old,
                        aliases,
                        running_sums,
                        thread_axes_before,
                        scalar_defs_before,
                        rewritten,
                    ),
                )

        info = _detect_chunk_recurrence(stmt, aliases, running_sums)
        if info is None:
            new_body.append(stmt)
            continue
        reset_idx = _find_reset_assign(new_body, info.dot_acc_var)
        if reset_idx is None:
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence has no relocatable accumulator reset"
            )
        reset_stmt = new_body[reset_idx]
        if not _is_proven_relocatable_assignment(reset_stmt, allow_load=False):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence reset has effects or unknown call purity"
            )
        chunk_writes = {
            _canonical_name(name, aliases) for name in ReadWrites.from_ast(stmt).writes
        }
        reset_reads = {
            _canonical_name(name, aliases)
            for name in ReadWrites.from_ast(reset_stmt).reads
        }
        if reset_reads & chunk_writes:
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence reset depends on chunk-carried state"
            )
        if any(
            info.dot_acc_var
            in {
                *ReadWrites.from_ast(intervening).reads,
                *ReadWrites.from_ast(intervening).writes,
            }
            for intervening in new_body[reset_idx + 1 :]
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence reset has an intervening consumer"
            )
        if any(
            {
                _canonical_name(name, aliases)
                for name in ReadWrites.from_ast(intervening).writes
            }
            & reset_reads
            for intervening in new_body[reset_idx + 1 :]
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence reset dependency is overwritten"
            )
        new_body.pop(reset_idx)
        assert isinstance(stmt, ast.For)
        rewritten.add(info.dot_acc_var)
        new_body.append(
            _rewrite_chunk_recurrence(
                stmt,
                info,
                reset_stmt,
                aliases,
                thread_axes_before,
                scalar_defs_before,
            )
        )
    return new_body


def _detect_chunk_recurrence(
    stmt: ast.AST,
    rename_groups: dict[str, str],
    running_sums: set[str] | None,
) -> _ChunkRecurrence | None:
    """Describe a serial chunk loop carrying one lane-folded recurrence."""
    if not _is_serial_for(stmt) or not isinstance(stmt, ast.For):
        return None
    found = _single_lane_loop_in_body(list(stmt.body))
    if found is None:
        return None
    lane_idx, lane_loop, lane_var = found
    recurrence = _find_dot_acc_recurrence(
        lane_loop,
        lane_var,
        rename_groups,
        running_sums,
    )
    if recurrence is None:
        return None
    return dataclasses.replace(recurrence, lane_idx=lane_idx)


def _rewrite_chunk_recurrence(
    stmt: ast.For,
    info: _ChunkRecurrence,
    reset_stmt: ast.AST,
    rename_groups: dict[str, str],
    thread_axes_before: dict[int, dict[str, frozenset[int]]],
    scalar_defs_before: dict[int, dict[str, ast.AST]],
) -> ast.For:
    """Build the restructured chunk loop (see module comment)."""
    from ...ast_read_writes import ReadWrites

    lane_idx = info.lane_idx
    lane_loop = info.lane_loop
    lane_var = info.lane_var
    chunk_body: list[ast.AST] = list(stmt.body)
    lane_body: list[ast.AST] = list(lane_loop.body)
    lane_varying = _lane_varying_names(lane_body[: info.sum_idx + 1], lane_var)

    def reads_lane(s: ast.AST) -> bool:
        reads = set(ReadWrites.from_ast(s).reads)
        return lane_var in reads or bool(reads & lane_varying)

    # The base slice was proven lane-invariant by the detector.  It freezes the
    # chunk-entry state and computes its optional per-chunk rescale.
    hoist_set = set(info.base_indices)
    reduction_axes = _chunk_recurrence_reduction_axes(info, scalar_defs_before)
    guarded_statements: dict[int, ast.AST] = {}
    all_lane_writes = {
        name
        for lane_stmt in lane_body
        for name in ReadWrites.from_ast(lane_stmt).writes
    }
    lane_effects = _ordered_block_reads_writes(cast("list[ast.stmt]", lane_body))
    canonical_live_ins = {
        _canonical_name(name, rename_groups)
        for name in lane_effects.reads_before_write
        if name != lane_var
    }
    canonical_lane_writes = {
        _canonical_name(name, rename_groups) for name in lane_effects.may_writes
    }
    loop_carried_names = canonical_live_ins & canonical_lane_writes
    allowed_recurrence_carries = {
        info.state_name,
        _canonical_name(info.dot_acc_var, rename_groups),
    }

    def reject_unrelated_carries(index: int, moved_stmt: ast.AST) -> None:
        dependencies = _dependency_names(
            lane_body[:index], set(ReadWrites.from_ast(moved_stmt).reads)
        )
        dependencies.update(ReadWrites.from_ast(moved_stmt).writes)
        if {_canonical_name(name, rename_groups) for name in dependencies} & (
            loop_carried_names - allowed_recurrence_carries
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence cannot relocate another loop carry"
            )

    def guard_idempotent_store(
        index: int,
        store_stmt: ast.AST,
        *,
        thread_axes: dict[str, frozenset[int]] | None = None,
        scalar_definitions: dict[str, ast.AST] | None = None,
    ) -> None:
        store = _validated_idempotent_store(store_stmt)
        if store is None:
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence can only relocate an ordinary store"
            )
        reject_unrelated_carries(index, store_stmt)
        store_func = cast("ast.Attribute", store.func)
        address_and_predicate_reads = set(ReadWrites.from_ast(store_func.value).reads)
        for predicate in _store_enclosing_predicates(store_stmt, store):
            address_and_predicate_reads.update(ReadWrites.from_ast(predicate).reads)
        address_and_predicate_dependencies = _dependency_names(
            lane_body[:index], address_and_predicate_reads
        )
        if {
            _canonical_name(name, rename_groups)
            for name in address_and_predicate_dependencies
        } & allowed_recurrence_carries:
            raise exc.BackendUnsupported(
                "cute",
                "chunk recurrence store address or predicate is not idempotent",
            )
        store_axes = _store_thread_axes(
            store_stmt,
            (
                thread_axes
                if thread_axes is not None
                else thread_axes_before.get(id(store_stmt), {})
            ),
            (
                scalar_definitions
                if scalar_definitions is not None
                else scalar_defs_before.get(id(store_stmt), {})
            ),
        )
        if store_axes is None:
            raise exc.BackendUnsupported(
                "cute", "cannot infer chunk recurrence store ownership"
            )
        owner_exprs = _thread_axis_owner_exprs(reduction_axes, store_axes)
        guarded_statements[index] = (
            _guard_stmt_with_owner(store_stmt, owner_exprs)
            if owner_exprs
            else store_stmt
        )

    # Lane-invariant side-effecting statements whose backward slice reaches the
    # (frozen) chunk-ENTRY accumulator (the store of ``b_h``) must hoist before
    # the lane loop so they run once per chunk on the frozen value; bring their
    # producers along.  A store that does NOT read the accumulator stays in the
    # lane loop (it is a genuinely per-lane side effect).
    entry_indices: list[int] = []
    for idx, prod in enumerate(lane_body[: info.sum_idx]):
        if idx in hoist_set or reads_lane(prod):
            continue
        slice_indices, _ = _backward_slice(
            lane_body[:idx], set(ReadWrites.from_ast(prod).reads)
        )
        slice_names = _dependency_names(
            lane_body[:idx], set(ReadWrites.from_ast(prod).reads)
        )
        if not any(
            _canonical_name(name, rename_groups) == info.state_name
            for name in slice_names
        ):
            continue
        store = _validated_idempotent_store(prod)
        if store is None:
            if not _is_proven_relocatable_assignment(prod, allow_load=True):
                raise exc.BackendUnsupported(
                    "cute",
                    "chunk recurrence state consumer has unknown side effects",
                )
            continue
        # The store + its complete, lane-invariant, side-effect-free producer
        # slice all hoist.  A live-in modified elsewhere in the lane would make
        # the repeated stores observably different and is therefore not
        # idempotent.
        if any(
            reads_lane(lane_body[i])
            or not _is_proven_relocatable_assignment(lane_body[i], allow_load=False)
            for i in slice_indices
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence store does not have a pure complete slice"
            )
        relocated_before_store = hoist_set | set(entry_indices) | set(slice_indices)
        if any(
            earlier_idx not in relocated_before_store
            and not _is_proven_relocatable_assignment(
                lane_body[earlier_idx], allow_load=False
            )
            for earlier_idx in range(idx)
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence cannot move a store across a prior effect"
            )
        entry_store_roots = _store_iterator_roots(store)
        for later_idx in range(idx + 1, info.sum_idx):
            if later_idx in relocated_before_store:
                continue
            later_stmt = lane_body[later_idx]
            try:
                later_store = _validated_idempotent_store(later_stmt)
            except exc.BackendUnsupported as error:
                raise exc.BackendUnsupported(
                    "cute",
                    "chunk recurrence cannot collapse a store before a later effect",
                ) from error
            if later_store is not None:
                later_roots = _store_iterator_roots(later_store)
                if (
                    entry_store_roots
                    and later_roots
                    and later_roots.isdisjoint(entry_store_roots)
                ):
                    continue
                raise exc.BackendUnsupported(
                    "cute",
                    "chunk recurrence cannot collapse stores to the same allocation",
                )
            if not _is_proven_relocatable_assignment(
                later_stmt,
                allow_load=True,
                allow_reduction=True,
            ):
                raise exc.BackendUnsupported(
                    "cute",
                    "chunk recurrence cannot collapse a store before a later effect",
                )
        slice_writes = {
            name
            for slice_index in slice_indices
            for name in ReadWrites.from_ast(lane_body[slice_index]).writes
        }
        slice_reads = set(ReadWrites.from_ast(prod).reads)
        for slice_index in slice_indices:
            slice_reads.update(ReadWrites.from_ast(lane_body[slice_index]).reads)
        if any(
            name in all_lane_writes
            and name not in slice_writes
            and _canonical_name(name, rename_groups) != info.state_name
            for name in slice_reads
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence store is not proven idempotent"
            )
        guard_idempotent_store(idx, prod)
        entry_indices.append(idx)
        hoist_set.update(slice_indices)

    hoist_pre = sorted(hoist_set | set(entry_indices))
    for idx in hoist_pre:
        reject_unrelated_carries(idx, lane_body[idx])
    pre_lane = [guarded_statements.get(i, lane_body[i]) for i in hoist_pre]

    # Moving the state update out of the repeated lane requires moving its
    # complete lexical suffix with it.  Otherwise a suffix consumer executes
    # before the newly finalized state exists.  Relocate only lane-invariant,
    # provably pure assignments and isolated idempotent stores; fail closed for
    # atomics, synchronization, compound effects, or calls of unknown purity.
    required_post_indices = {*info.finalize_indices, info.final_idx}
    post_lane_indices = set(range(info.sum_idx + 1, len(lane_body)))
    extra_post_indices = post_lane_indices - required_post_indices
    extra_post_reads = {
        name
        for idx in extra_post_indices
        for name in ReadWrites.from_ast(lane_body[idx]).reads
    }
    suffix_dependency_indices, _ = _backward_slice(
        lane_body[: info.sum_idx + 1], extra_post_reads
    )
    for idx in suffix_dependency_indices:
        if idx in hoist_set or idx == info.sum_idx:
            continue
        dependency = lane_body[idx]
        if reads_lane(dependency) or not _is_proven_relocatable_assignment(
            dependency, allow_load=False
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence suffix does not have a pure complete slice"
            )
    post_varying = set(lane_varying)
    first_post = info.sum_idx + 1
    effective_post_axes = dict(
        thread_axes_before.get(id(lane_body[first_post]), {})
        if first_post < len(lane_body)
        else {}
    )
    effective_post_defs = dict(
        scalar_defs_before.get(id(lane_body[first_post]), {})
        if first_post < len(lane_body)
        else {}
    )
    for idx in sorted(post_lane_indices):
        post_stmt = lane_body[idx]
        post_rw = ReadWrites.from_ast(post_stmt)
        reject_unrelated_carries(idx, post_stmt)
        if idx in required_post_indices:
            # The detector proved these statements turn the lane fold into a
            # lane-invariant final value.  Do not let their original dataflow
            # taint later consumers that are relocated with them.
            post_varying.difference_update(post_rw.writes)
            _update_thread_axis_names(post_stmt, effective_post_axes)
            _update_scalar_definitions(post_stmt, effective_post_defs)
            if any(
                call_path_str(node.func) in _CHUNK_REDUCTION_HELPERS
                or (
                    (call_path_str(node.func) or "").startswith(
                        "cute.arch.warp_reduction"
                    )
                )
                for node in ast.walk(post_stmt)
                if isinstance(node, ast.Call)
            ):
                for name in post_rw.writes:
                    effective_post_axes[name] = frozenset(
                        set(effective_post_axes.get(name, ())) - reduction_axes
                    )
            continue
        statement_is_lane_varying = lane_var in post_rw.reads or bool(
            set(post_rw.reads) & post_varying
        )
        if statement_is_lane_varying or any(
            _canonical_name(name, rename_groups) == info.state_name
            for name in post_rw.writes
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence has a lane-varying state suffix"
            )
        store = _validated_idempotent_store(post_stmt)
        if store is not None:
            if idx < info.final_idx:
                raise exc.BackendUnsupported(
                    "cute",
                    "chunk recurrence cannot relocate a store before the finalized state",
                )
            guard_idempotent_store(
                idx,
                post_stmt,
                thread_axes=effective_post_axes,
                scalar_definitions=effective_post_defs,
            )
        elif not _is_proven_relocatable_assignment(post_stmt, allow_load=False):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence suffix is not proven pure"
            )
        post_varying.difference_update(post_rw.writes)
        _update_thread_axis_names(post_stmt, effective_post_axes)
        _update_scalar_definitions(post_stmt, effective_post_defs)

    lane_kept = [
        lane_body[i]
        for i in range(len(lane_body))
        if i not in hoist_set
        and i not in set(entry_indices)
        and i not in post_lane_indices
    ]

    new_lane_loop = _clone_lane_loop_with_body(lane_loop, lane_kept)
    post_lane = [
        guarded_statements.get(i, lane_body[i]) for i in sorted(post_lane_indices)
    ]
    new_chunk_body: list[ast.AST] = [
        reset_stmt,
        *chunk_body[:lane_idx],
        *pre_lane,
        new_lane_loop,
        *post_lane,
        *chunk_body[lane_idx + 1 :],
    ]
    return create(
        ast.For,
        target=stmt.target,
        iter=stmt.iter,
        body=new_chunk_body,
        orelse=stmt.orelse,
        type_comment=None,
    )
