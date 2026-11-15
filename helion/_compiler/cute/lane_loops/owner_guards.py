"""Thread-axis ownership proofs for lane-split consumers (CuTe).

Role: analysis.  After code generation the logical block provenance of a
value is gone; these helpers rebuild the physical-thread provenance of the
generated scalars (``_update_thread_axis_names`` / ``_update_scalar_definitions``
/ ``_update_proven_uniform_names``), prove which thread axes a memory write
varies along (``_store_thread_axes``), bound generated integer scalars
(``_scalar_integer_bounds``) and derive the owner predicates
(``_thread_axis_owner_exprs`` / ``_guard_stmt_with_owner``) that keep a
reduction-broadcast consumer from racing across the redundant thread axes.
Also the store validators the lane split and the chunk-recurrence hoist share
(``_memory_write_calls``, ``_validated_owner_store``,
``_validated_idempotent_store``).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

from .... import exc
from ...ast_extension import create
from ...ast_extension import expr_from_string
from ...ast_extension import statement_from_string
from .._ast_pass_utils import _is_proven_relocatable_call
from .._ast_pass_utils import deep_clone
from ..persistent_branch_vec import PACKET_STORE_BYTES
from ..thread_budget import MAX_THREADS_PER_BLOCK
from .markers import _CUTE_THREAD_VARYING_INTRINSICS
from .markers import _int_literal
from .markers import _is_lane_reduce_marker_assign
from .markers import _LaneReduceMarker

if TYPE_CHECKING:
    from collections.abc import Mapping


def _is_proven_uniform_name(name: str, uniform_names: set[str]) -> bool:
    return name in uniform_names or name.startswith(("_BLOCK_SIZE_", "_RDIM_SIZE_"))


def _update_proven_uniform_names(stmt: ast.AST, uniform_names: set[str]) -> None:
    """Track simple assignments proven uniform across the CUDA thread block."""
    from ...ast_read_writes import ReadWrites

    rw = ReadWrites.from_ast(stmt)
    writes = set(rw.writes)
    if not writes:
        return
    source = ast.unparse(stmt)
    if not isinstance(stmt, ast.Assign) or any(
        name in source for name in _CUTE_THREAD_VARYING_INTRINSICS
    ):
        uniform_names.difference_update(writes)
        return
    reads = set(rw.reads)
    if all(_is_proven_uniform_name(name, uniform_names) for name in reads):
        uniform_names.update(writes)
    else:
        uniform_names.difference_update(writes)


def _thread_axes_read_by(
    node: ast.AST,
    thread_axis_names: dict[str, frozenset[int]],
) -> set[int]:
    """Physical thread axes that can affect an expression or statement."""
    axes: set[int] = set()

    class ThreadAxisVisitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                axes.update(thread_axis_names.get(node.id, ()))

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "lane_idx":
                # ``lane_idx`` is a linear warp coordinate, not CUDA axis 0.
                # In a multidimensional CTA it can depend on every physical
                # thread axis, and it repeats across warps.  This provenance is
                # intentionally a conservative superset so a lane-based value
                # or predicate cannot be mistaken for axis-0-only ownership.
                axes.update((0, 1, 2))
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "thread_idx"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)
            ):
                axes.add(node.slice.value)
            self.generic_visit(node)

    ThreadAxisVisitor().visit(node)
    return axes


def _update_thread_axis_names(
    stmt: ast.AST,
    thread_axis_names: dict[str, frozenset[int]],
) -> None:
    """Track physical-thread provenance through generated scalar assignments.

    The lane-reduction post-pass runs after code generation, when logical block
    provenance has otherwise been erased.  Keeping this small dataflow map lets
    the pass distinguish a real sibling output coordinate from a redundant
    launch coordinate before it emits a memory write.
    """
    from ...ast_read_writes import ReadWrites

    rw = ReadWrites.from_ast(stmt)
    writes = set(rw.writes)
    if not writes:
        return
    axes = _thread_axes_read_by(stmt, thread_axis_names)
    marker = _is_lane_reduce_marker_assign(stmt)
    if marker is not None:
        reduce_axis = _lane_reduce_axis(marker)
        if reduce_axis is not None:
            axes.discard(reduce_axis)
    frozen_axes = frozenset(axes)
    for name in writes:
        thread_axis_names[name] = frozen_axes


def _update_scalar_definitions(
    stmt: ast.AST,
    scalar_definitions: dict[str, ast.AST],
    rename_groups: Mapping[str, str] | None = None,
) -> None:
    """Track simple generated scalar assignments for predicate proofs."""
    from ...ast_read_writes import ReadWrites

    renames = rename_groups or {}
    invalidated = {renames.get(name, name) for name in ReadWrites.from_ast(stmt).writes}
    changed = True
    while changed:
        changed = False
        for name, expression in list(scalar_definitions.items()):
            if (
                renames.get(name, name) in invalidated
                or {
                    renames.get(read, read)
                    for read in ReadWrites.from_ast(expression).reads
                }
                & invalidated
            ):
                scalar_definitions.pop(name)
                canonical = renames.get(name, name)
                if canonical not in invalidated:
                    invalidated.add(canonical)
                    changed = True
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        scalar_definitions[stmt.targets[0].id] = stmt.value


def _statement_provenance_before(
    body: list[ast.AST],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: Mapping[str, str] | None = None,
) -> tuple[
    dict[int, dict[str, frozenset[int]]],
    dict[int, dict[str, ast.AST]],
]:
    """Snapshot physical-thread and scalar provenance before each statement."""
    thread_axes_before: dict[int, dict[str, frozenset[int]]] = {}
    scalar_defs_before: dict[int, dict[str, ast.AST]] = {}
    local_thread_axes = dict(thread_axis_names)
    local_scalar_defs = dict(scalar_definitions)
    for stmt in body:
        thread_axes_before[id(stmt)] = dict(local_thread_axes)
        scalar_defs_before[id(stmt)] = dict(local_scalar_defs)
        _update_thread_axis_names(stmt, local_thread_axes)
        _update_scalar_definitions(stmt, local_scalar_defs, rename_groups)
    return thread_axes_before, scalar_defs_before


def _lane_reduction_owner_exprs_for_statement(
    stmt: ast.AST,
    markers: list[tuple[int, _LaneReduceMarker]],
    marker_dependencies: list[tuple[_LaneReduceMarker, set[str]]],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
) -> list[str]:
    """Return proven owner predicates for one reduction-consume statement.

    Calling ``_store_thread_axes`` even when no predicate is ultimately needed
    is intentional: it rejects unguardable memory writes and unknown side
    effects instead of letting a reduction-broadcast consumer race.
    """
    from ...ast_read_writes import ReadWrites

    store_axes = _store_thread_axes(stmt, thread_axis_names, scalar_definitions)
    owner_exprs = _redundant_thread_axis_owner_exprs(markers, store_axes)
    if store_axes is None:
        return owner_exprs
    reads = set(ReadWrites.from_ast(stmt).reads)
    for marker, dependencies in marker_dependencies:
        owner_expr = _lane_reduce_owner_expr(marker, store_axes)
        if owner_expr is not None and reads & dependencies:
            owner_exprs.append(owner_expr)
    return list(dict.fromkeys(owner_exprs))


def _lane_reduce_owner_expr(
    marker: _LaneReduceMarker,
    store_axes: _StoreThreadAxes | None = None,
) -> str | None:
    """Predicate selecting one writer for a broadcast reduction result."""
    reduce_axis = _lane_reduce_axis(marker)
    if store_axes is not None and reduce_axis is not None:
        if reduce_axis in store_axes.address:
            return None
        if reduce_axis in store_axes.unique_predicate:
            return None
        if reduce_axis in store_axes.value or (
            reduce_axis in store_axes.predicate
            and reduce_axis not in store_axes.bounded_zero_safe_predicate
        ):
            raise exc.BackendUnsupported(
                "cute",
                "cannot infer unique ownership for a thread-varying store",
            )
    if marker.group_span > 1 and marker.group_lane_expr:
        # ``group_pre`` is the product of sibling coordinates below the
        # reduction axis.  The first ``group_pre`` lanes in each group are the
        # reduction-coordinate-zero owners, one for each sibling element.
        return (
            f"(({marker.group_lane_expr}) % {marker.group_span}) < {marker.group_pre}"
        )
    if marker.threads_in_group > 1:
        return f"(cutlass.Int32(cute.arch.lane_idx()) % {marker.threads_in_group}) == 0"
    return None


def _lane_reduce_axis(marker: _LaneReduceMarker) -> int | None:
    if marker.group_span > 1 and marker.group_lane_expr:
        strides = _linear_thread_axis_strides(marker.group_lane_expr)
        if strides is None:
            return None
        matches = [
            axis for axis, stride in strides.items() if stride == marker.group_pre
        ]
        return matches[0] if len(matches) == 1 else None
    if marker.threads_in_group > 1:
        return 0
    return None


def _linear_thread_axis_strides(expr: str | ast.expr) -> dict[int, int] | None:
    """Recover ``thread_idx`` coefficients from a flattened lane expression."""
    if isinstance(expr, str):
        try:
            root = ast.parse(expr, mode="eval").body
        except SyntaxError:
            return None
    else:
        root = expr

    def merge(
        left: tuple[dict[int, int], int],
        right: tuple[dict[int, int], int],
        sign: int = 1,
    ) -> tuple[dict[int, int], int]:
        coefficients = dict(left[0])
        for axis, coefficient in right[0].items():
            coefficients[axis] = coefficients.get(axis, 0) + sign * coefficient
        return coefficients, left[1] + sign * right[1]

    def scale(
        value: tuple[dict[int, int], int], factor: int
    ) -> tuple[dict[int, int], int]:
        return (
            {axis: coefficient * factor for axis, coefficient in value[0].items()},
            value[1] * factor,
        )

    def visit(node: ast.AST) -> tuple[dict[int, int], int] | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return {}, node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = visit(node.operand)
            return None if value is None else scale(value, -1)
        if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
            # Generated lane expressions wrap both coordinates and constants in
            # the configured index dtype constructor.
            return visit(node.args[0])
        if isinstance(node, ast.Subscript):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "thread_idx"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)
            ):
                return {node.slice.value: 1}, 0
            return None
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return merge(left, right)
            if isinstance(node.op, ast.Sub):
                return merge(left, right, -1)
            if isinstance(node.op, ast.Mult):
                if not left[0]:
                    return scale(right, left[1])
                if not right[0]:
                    return scale(left, right[1])
        return None

    result = visit(root)
    if result is None or result[1] != 0:
        return None
    return result[0]


class _StoreThreadAxes(NamedTuple):
    value: set[int]
    address: set[int]
    predicate: set[int]
    unique_predicate: set[int]
    zero_safe_predicate: set[int]
    bounded_zero_safe_predicate: frozenset[int] = frozenset()


def _contains_sync_threads(stmt: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "sync_threads"
            or isinstance(node.func, ast.Name)
            and node.func.id == "sync_threads"
        )
        for node in ast.walk(stmt)
    )


# Atomic helpers generated code calls by bare name, each taking the pointer
# first as ``cute.arch``'s atomics do: the float max / min emulated with
# integer atomics (``cute/atomic_helpers.py``) and the warp-aggregated ticket
# fetch-add (``cute/warp_atomic_runtime.py``).  Their names start with
# ``_cute_atomic_`` and the vector flushes' below with ``_cute_red_``, the
# prefixes by which ``fuse_two_pass_loads._is_store_call`` (the write check of
# the lane split and of the passes that move loads) recognizes them as writes.
_ATOMIC_HELPERS = frozenset(
    {
        "_cute_atomic_max_float32",
        "_cute_atomic_min_float32",
        "_cute_atomic_warp_fetch_add_int32",
    }
)
# The relaxed ``red.global.add`` flush of a vectorized atomic site
# (``cute/atomic_ops.py``), one packet of consecutive f32, f16 or bf16
# elements: a bare-name atomic taking the pointer first, too.
_VECTOR_ATOMIC_HELPERS = frozenset(
    {"_cute_red_add_f32_vec", "_cute_red_add_f16_vec", "_cute_red_add_bf16_vec"}
)
_HELPER_STORES = frozenset({*PACKET_STORE_BYTES, "_helion_persistent_branch_vec_store"})


def _is_atomic_helper(name: str) -> bool:
    """Whether a call of the bare name ``name`` is an atomic read-modify-write."""
    return (
        name.startswith("atomic_")
        or name in _ATOMIC_HELPERS
        or name in _VECTOR_ATOMIC_HELPERS
    )


def _is_atomic_call(call: ast.Call) -> bool:
    """``cute.arch.atomic_*``, a pointer's ``atomic_*`` method or an atomic
    called by bare name (``_is_atomic_helper``)."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr.startswith("atomic_")
    return isinstance(func, ast.Name) and _is_atomic_helper(func.id)


def _memory_write_calls(stmt: ast.AST) -> list[ast.Call]:
    result: list[ast.Call] = []
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call):
            continue
        if (
            _is_atomic_call(node)
            or isinstance(node.func, ast.Attribute)
            and node.func.attr in ("store", "__setitem__")
            or isinstance(node.func, ast.Name)
            and node.func.id in _HELPER_STORES
        ):
            result.append(node)
    return result


def _is_persistent_branch_vec_store(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == "_helion_persistent_branch_vec_store"
        and len(call.args) == 6
        and not call.keywords
    )


def _is_vector_flush_store(call: ast.Call) -> bool:
    """A ``_cute_store_*_vec(ptr, values)`` packet flush (``PACKET_STORE_BYTES``).

    The flush of a V-loop's collected per-element values into one vector
    store; its destination is the first argument and its value the second.
    """
    return (
        isinstance(call.func, ast.Name)
        and call.func.id in PACKET_STORE_BYTES
        and len(call.args) == 2
        and not call.keywords
    )


def _atomic_pointer_and_value(call: ast.Call) -> tuple[ast.AST, ast.AST] | None:
    if isinstance(call.func, ast.Name):
        return (
            _pointer_first_operands(call) if _is_atomic_helper(call.func.id) else None
        )
    if not (
        isinstance(call.func, ast.Attribute) and call.func.attr.startswith("atomic_")
    ):
        return None
    # ``cute.arch.atomic_add(ptr, value)`` carries its pointer as the first
    # argument; pointer-method atomics carry it in the callee receiver.
    if ast.unparse(call.func.value) == "cute.arch":
        return _pointer_first_operands(call)
    return (call.func.value, call.args[0]) if call.args else None


def _pointer_first_operands(call: ast.Call) -> tuple[ast.AST, ast.AST] | None:
    """The pointer and the operand of an atomic taking the pointer first."""
    if len(call.args) >= 2:
        return (call.args[0], call.args[1])
    # The emitted form spells the operand by its parameter name,
    # ``cute.arch.atomic_add(ptr, val=value, sem='relaxed')``; a
    # compare-and-swap carries two operands and is not a plain update.
    values = [keyword.value for keyword in call.keywords if keyword.arg != "sem"]
    if len(call.args) == 1 and len(values) == 1:
        return (call.args[0], values[0])
    return None


def _is_isolated_store_statement(stmt: ast.AST, store: ast.Call) -> bool:
    """Whether wrapping *stmt* predicates only one otherwise isolated store."""
    if any(isinstance(node, ast.NamedExpr) for node in ast.walk(stmt)):
        return False
    for node in ast.walk(stmt):
        if not isinstance(node, ast.stmt):
            continue
        if isinstance(node, (ast.If, ast.Pass)):
            continue
        if isinstance(node, ast.Expr) and node.value is store:
            continue
        return False
    return True


def _direct_thread_coordinate(node: ast.AST) -> int | None:
    """Return an axis for a direct thread coordinate, allowing scalar casts."""
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        is_builtin_int = isinstance(node.func, ast.Name) and node.func.id == "int"
        is_cutlass_int = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr in ("Int32", "Int64")
        )
        if is_builtin_int or is_cutlass_int:
            return _direct_thread_coordinate(node.args[0])
    if not isinstance(node, ast.Subscript):
        return None
    value = node.value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "thread_idx"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ):
        return None
    return node.slice.value


def _uniquely_predicated_thread_axes(predicate: ast.AST) -> set[int]:
    """Axes for which *predicate* fixes one exact physical thread coordinate."""
    if isinstance(predicate, ast.BoolOp) and isinstance(predicate.op, ast.And):
        result: set[int] = set()
        for value in predicate.values:
            result.update(_uniquely_predicated_thread_axes(value))
        return result
    if not (
        isinstance(predicate, ast.Compare)
        and len(predicate.ops) == 1
        and isinstance(predicate.ops[0], ast.Eq)
        and len(predicate.comparators) == 1
    ):
        return set()
    lhs_axis = _direct_thread_coordinate(predicate.left)
    rhs_axis = _direct_thread_coordinate(predicate.comparators[0])
    lhs_literal = _int_literal(predicate.left)
    rhs_literal = _int_literal(predicate.comparators[0])
    if lhs_axis is not None and rhs_literal is not None:
        return {lhs_axis}
    if rhs_axis is not None and lhs_literal is not None:
        return {rhs_axis}
    return set()


def _resolve_scalar_definition(
    node: ast.AST,
    scalar_definitions: dict[str, ast.AST],
    seen: set[str] | None = None,
) -> ast.AST:
    """Resolve generated scalar aliases without expanding arbitrary syntax."""
    if not isinstance(node, ast.Name) or node.id not in scalar_definitions:
        return node
    if seen is None:
        seen = set()
    if node.id in seen:
        return node
    return _resolve_scalar_definition(
        scalar_definitions[node.id],
        scalar_definitions,
        {*seen, node.id},
    )


def _thread_axis_linear_coefficient(
    node: ast.AST,
    axis: int,
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    seen: set[str] | None = None,
) -> int | None:
    """Return the integer coefficient of one thread axis, when affine."""
    if axis not in _thread_axes_read_by(node, thread_axis_names):
        return 0
    if isinstance(node, ast.Name):
        if node.id not in scalar_definitions:
            return None
        if seen is None:
            seen = set()
        if node.id in seen:
            return None
        return _thread_axis_linear_coefficient(
            scalar_definitions[node.id],
            axis,
            thread_axis_names,
            scalar_definitions,
            {*seen, node.id},
        )
    direct_axis = _direct_thread_coordinate(node)
    if direct_axis is not None:
        return int(direct_axis == axis)
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        is_builtin_int = isinstance(node.func, ast.Name) and node.func.id == "int"
        is_cutlass_int = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr in ("Int32", "Int64")
        )
        if is_builtin_int or is_cutlass_int:
            return _thread_axis_linear_coefficient(
                node.args[0],
                axis,
                thread_axis_names,
                scalar_definitions,
                seen,
            )
    if isinstance(node, ast.UnaryOp):
        coefficient = _thread_axis_linear_coefficient(
            node.operand,
            axis,
            thread_axis_names,
            scalar_definitions,
            seen,
        )
        if coefficient is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return coefficient
        if isinstance(node.op, ast.USub):
            return -coefficient
    if isinstance(node, ast.BinOp):
        left = _thread_axis_linear_coefficient(
            node.left,
            axis,
            thread_axis_names,
            scalar_definitions,
            seen,
        )
        right = _thread_axis_linear_coefficient(
            node.right,
            axis,
            thread_axis_names,
            scalar_definitions,
            seen,
        )
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            left_literal = _int_literal(node.left)
            right_literal = _int_literal(node.right)
            if left_literal is not None:
                return left_literal * right
            if right_literal is not None:
                return right_literal * left
    return None


def _is_generated_block_extent(
    node: ast.AST,
    scalar_definitions: dict[str, ast.AST],
) -> bool:
    """Whether *node* is a generated, statically positive block extent."""
    node = _resolve_scalar_definition(node, scalar_definitions)
    while isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        is_builtin_int = isinstance(node.func, ast.Name) and node.func.id == "int"
        is_cutlass_int = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr in ("Int32", "Int64")
        )
        if not (is_builtin_int or is_cutlass_int):
            return False
        node = _resolve_scalar_definition(node.args[0], scalar_definitions)
    return isinstance(node, ast.Name) and node.id.startswith("_BLOCK_SIZE_")


_SCALAR_INTEGER_BOUNDS_ATTR = "_helion_scalar_integer_bounds"


def _scalar_integer_bounds(
    node: ast.AST,
    scalar_definitions: dict[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> tuple[int, int] | None:
    """Bound simple generated coordinates without signed Int32 overflow.

    Loop-index ranges annotate proof-only name expressions in the scalar
    context; they never replace an expression in emitted code. Unknown loads,
    calls, cyclic aliases, and narrowing or overflowing arithmetic decline.
    """
    bounds: tuple[int, int] | None = None
    if isinstance(node, ast.Name):
        bounds = getattr(node, _SCALAR_INTEGER_BOUNDS_ATTR, None)
        if bounds is None and node.id in scalar_definitions and node.id not in seen:
            return _scalar_integer_bounds(
                scalar_definitions[node.id], scalar_definitions, seen | {node.id}
            )
    elif isinstance(node, ast.Constant) and type(node.value) is int:
        bounds = node.value, node.value
    elif _direct_thread_coordinate(node) is not None:
        # CUDA permits at most 1024 threads per block. Using that upper bound
        # also covers smaller and multidimensional launches conservatively.
        bounds = 0, MAX_THREADS_PER_BLOCK - 1
    elif isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        if ast.unparse(node.func) in ("int", "cutlass.Int32", "cutlass.Int64"):
            bounds = _scalar_integer_bounds(node.args[0], scalar_definitions, seen)
    elif isinstance(node, ast.UnaryOp):
        operand = _scalar_integer_bounds(node.operand, scalar_definitions, seen)
        if operand is not None:
            if isinstance(node.op, ast.UAdd):
                bounds = operand
            elif isinstance(node.op, ast.USub):
                bounds = -operand[1], -operand[0]
    elif isinstance(node, ast.BinOp):
        left = _scalar_integer_bounds(node.left, scalar_definitions, seen)
        right = _scalar_integer_bounds(node.right, scalar_definitions, seen)
        if left is not None and right is not None:
            if isinstance(node.op, ast.Add):
                bounds = left[0] + right[0], left[1] + right[1]
            elif isinstance(node.op, ast.Sub):
                bounds = left[0] - right[1], left[1] - right[0]
            elif isinstance(node.op, ast.Mult):
                products = [a * b for a in left for b in right]
                bounds = min(products), max(products)
    if bounds is None or not -(2**31) <= bounds[0] <= bounds[1] < 2**31:
        return None
    return bounds


def _loop_invariant_scalar_definitions(
    statement: ast.AST,
    scalar_definitions: dict[str, ast.AST],
    rename_groups: Mapping[str, str],
) -> dict[str, ast.AST]:
    """Invalidate body writes before any iteration, including carry aliases."""
    from ...ast_read_writes import ReadWrites

    if not isinstance(statement, (ast.For, ast.While)):
        return scalar_definitions
    definitions = dict(scalar_definitions)
    writes = set(ReadWrites.from_ast(statement).writes)
    if isinstance(statement, ast.For):
        # ReadWrites treats induction names as local to the loop; incoming
        # proof facts for a shadowed name must nevertheless be retired.
        writes.update(
            node.id for node in ast.walk(statement.target) if isinstance(node, ast.Name)
        )
    canonical_writes = {rename_groups.get(name, name) for name in writes}
    changed = True
    while changed:
        changed = False
        for name, expression in list(definitions.items()):
            reads = set(ReadWrites.from_ast(expression).reads)
            if any(
                rename_groups.get(value, value) in canonical_writes
                for value in (name, *reads)
            ):
                definitions.pop(name)
                canonical_writes.add(rename_groups.get(name, name))
                changed = True
    return definitions


def _loop_body_scalar_definitions(
    statement: ast.AST,
    scalar_definitions: dict[str, ast.AST],
    thread_axis_names: dict[str, frozenset[int]],
    rename_groups: Mapping[str, str] | None = None,
) -> dict[str, ast.AST]:
    """Bound a thread-uniform positive-step induction in its lexical body."""
    definitions = _loop_invariant_scalar_definitions(
        statement, scalar_definitions, rename_groups or {}
    )
    if not isinstance(statement, ast.For) or not isinstance(statement.target, ast.Name):
        return definitions
    index = ast.Name(id=statement.target.id, ctx=ast.Load())
    # Kill old aliases of the target even when this loop has no proved range.
    _update_scalar_definitions(
        ast.Assign(targets=[statement.target], value=index), definitions, rename_groups
    )
    call = statement.iter
    if not (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) in ("range", "cutlass.range_constexpr")
        and not call.keywords
        and 1 <= len(call.args) <= 3
        and not any(_thread_axes_read_by(arg, thread_axis_names) for arg in call.args)
    ):
        return definitions
    start = ast.Constant(0) if len(call.args) == 1 else call.args[0]
    stop = call.args[0] if len(call.args) == 1 else call.args[1]
    step = call.args[2] if len(call.args) == 3 else ast.Constant(1)
    begin = _scalar_integer_bounds(start, scalar_definitions)
    end = _scalar_integer_bounds(stop, scalar_definitions)
    stride = _scalar_integer_bounds(step, scalar_definitions)
    positive_step = stride is not None and stride[0] > 0
    if (
        begin is not None
        and end is not None
        and begin[0] < end[1]
        and (positive_step or _is_generated_block_extent(step, scalar_definitions))
    ):
        setattr(index, _SCALAR_INTEGER_BOUNDS_ATTR, (begin[0], end[1] - 1))
    return definitions


def _predicate_accepts_zero_thread_axis(
    predicate: ast.AST,
    axis: int,
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    *,
    require_bounded: bool = False,
) -> bool:
    """Prove a generated bound remains true after selecting axis lane zero.

    A ``<``/``<=`` bound whose left side has a positive affine coefficient for
    the nonnegative CUDA thread coordinate can only become easier to satisfy
    when that coordinate is replaced by zero. Other predicate forms fail
    closed rather than risk adding an owner condition that makes them empty.
    """
    if axis not in _thread_axes_read_by(predicate, thread_axis_names):
        return True
    resolved = _resolve_scalar_definition(predicate, scalar_definitions)
    if resolved is not predicate:
        return _predicate_accepts_zero_thread_axis(
            resolved,
            axis,
            thread_axis_names,
            scalar_definitions,
            require_bounded=require_bounded,
        )
    if isinstance(predicate, ast.BoolOp) and isinstance(predicate.op, ast.And):
        return all(
            _predicate_accepts_zero_thread_axis(
                value,
                axis,
                thread_axis_names,
                scalar_definitions,
                require_bounded=require_bounded,
            )
            for value in predicate.values
        )
    if not (
        isinstance(predicate, ast.Compare)
        and len(predicate.ops) == 1
        and isinstance(predicate.ops[0], (ast.Lt, ast.LtE))
        and len(predicate.comparators) == 1
    ):
        return False
    lhs = _resolve_scalar_definition(predicate.left, scalar_definitions)
    rhs = _resolve_scalar_definition(predicate.comparators[0], scalar_definitions)
    literal_end = _int_literal(rhs)
    bounded_coordinate = (
        literal_end is not None
        and 0 < literal_end < 2**31
        and _scalar_integer_bounds(lhs, scalar_definitions) is not None
    )
    return (
        axis not in _thread_axes_read_by(rhs, thread_axis_names)
        and (
            _is_generated_block_extent(rhs, scalar_definitions)
            or (require_bounded and bounded_coordinate)
        )
        and (
            not require_bounded
            or _scalar_integer_bounds(lhs, scalar_definitions) is not None
        )
        and (
            coefficient := _thread_axis_linear_coefficient(
                lhs,
                axis,
                thread_axis_names,
                scalar_definitions,
            )
        )
        is not None
        and coefficient > 0
    )


def _store_enclosing_predicates(stmt: ast.AST, store: ast.Call) -> list[ast.AST]:
    """Control predicates on the unique path from *stmt* to *store*."""

    def find(node: ast.AST, predicates: list[ast.AST]) -> list[ast.AST] | None:
        if node is store:
            return predicates
        if isinstance(node, ast.If):
            for child in node.body:
                found = find(child, [*predicates, node.test])
                if found is not None:
                    return found
            for child in node.orelse:
                # The else path is guarded by the negation. It can still carry
                # axis provenance, but an equality in the positive test does
                # not prove that the else selects a unique lane.
                found = find(
                    child,
                    [*predicates, create(ast.UnaryOp, op=ast.Not(), operand=node.test)],
                )
                if found is not None:
                    return found
            return None
        for child in ast.iter_child_nodes(node):
            found = find(child, predicates)
            if found is not None:
                return found
        return None

    return find(stmt, []) or []


def _store_thread_axes(
    stmt: ast.AST,
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
) -> _StoreThreadAxes | None:
    """Validate one guardable store and return its thread-axis provenance."""
    memory_calls = _memory_write_calls(stmt)
    atomic_parts = (
        _atomic_pointer_and_value(memory_calls[0])
        if len(memory_calls) == 1
        and _is_isolated_store_statement(stmt, memory_calls[0])
        else None
    )
    if atomic_parts is not None:
        if _contains_sync_threads(stmt):
            raise exc.BackendUnsupported(
                "cute", "cannot predicate a statement containing sync_threads"
            )
        store = memory_calls[0]
        pointer, value = atomic_parts
    else:
        store = _validated_owner_store(stmt)
        if store is None:
            return None
        if _is_persistent_branch_vec_store(store):
            pointer = store.args[3]
            value = store.args[4]
        elif _is_vector_flush_store(store):
            pointer = store.args[0]
            value = store.args[1]
        else:
            store_func = cast("ast.Attribute", store.func)
            pointer = store_func.value
            value = store.args[0]
    predicates = _store_enclosing_predicates(stmt, store)
    if _is_persistent_branch_vec_store(store):
        marker_mask = store.args[5]
        if not (isinstance(marker_mask, ast.Constant) and marker_mask.value is None):
            predicates.append(marker_mask)
    value_axes = _thread_axes_read_by(value, thread_axis_names)
    predicate_axes = [
        _thread_axes_read_by(predicate, thread_axis_names) for predicate in predicates
    ]
    all_predicate_axes = set().union(*predicate_axes)
    return _StoreThreadAxes(
        value_axes,
        _thread_axes_read_by(pointer, thread_axis_names),
        all_predicate_axes,
        set().union(
            *(_uniquely_predicated_thread_axes(predicate) for predicate in predicates)
        ),
        {
            axis
            for axis in all_predicate_axes
            if all(
                axis not in axes
                or _predicate_accepts_zero_thread_axis(
                    predicate,
                    axis,
                    thread_axis_names,
                    scalar_definitions,
                )
                for predicate, axes in zip(predicates, predicate_axes, strict=True)
            )
        },
        frozenset(
            axis
            for axis in all_predicate_axes
            if all(
                axis not in axes
                or _predicate_accepts_zero_thread_axis(
                    predicate,
                    axis,
                    thread_axis_names,
                    scalar_definitions,
                    require_bounded=True,
                )
                for predicate, axes in zip(predicates, predicate_axes, strict=True)
            )
        ),
    )


def _validated_owner_store(stmt: ast.AST) -> ast.Call | None:
    """Return the only isolated store, rejecting unsafe ownership rewrites."""
    if _contains_sync_threads(stmt):
        raise exc.BackendUnsupported(
            "cute", "cannot predicate a statement containing sync_threads"
        )
    # A call used as a statement discards its result, so it exists for an
    # effect.  Unknown callees must not pass through as if they were pure: the
    # finalized reduction is broadcast to every reduction thread, and an
    # unguarded custom writer would therefore race.  Recognized memory calls
    # continue through the stricter unique-store validation below.
    memory_calls = _memory_write_calls(stmt)
    memory_call_ids = {id(call) for call in memory_calls}
    if any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and id(node.value) not in memory_call_ids
        for node in ast.walk(stmt)
    ):
        raise exc.BackendUnsupported(
            "cute", "cannot infer whether a standalone call has side effects"
        )
    if not _has_observable_memory_write(stmt):
        return None
    stores = [
        node
        for node in memory_calls
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
            and len(node.args) == 1
            or _is_persistent_branch_vec_store(node)
            or _is_vector_flush_store(node)
        )
    ]
    if len(memory_calls) != 1 or len(stores) != 1:
        raise exc.BackendUnsupported(
            "cute", "cannot infer unique ownership for this memory write"
        )
    store = stores[0]
    if not _is_isolated_store_statement(stmt, store):
        raise exc.BackendUnsupported(
            "cute", "cannot predicate a compound statement containing a store"
        )
    return store


def _redundant_thread_axis_owner_exprs(
    markers: list[tuple[int, _LaneReduceMarker]],
    store_axes: _StoreThreadAxes | None,
) -> list[str]:
    """Select one writer along launch axes duplicated by a lane reduction.

    A scan feeding a reduction can expose the same logical free coordinate via
    two physical axes: one chosen by the scan/store layout and one introduced
    by the reduction layout.  The grouped reduction correctly preserves every
    sibling coordinate, but a store whose value, address, and enclosing control
    flow use none of those coordinates is otherwise emitted by every redundant
    thread. Gate only an axis proven absent from the complete statement. If the
    stored value varies on an unaddressed axis, reject instead of silently
    choosing one conflicting writer.
    """
    redundant_axes: set[int] = set()
    for _, marker in markers:
        if marker.group_span <= 1 or not marker.group_lane_expr:
            continue
        strides = _linear_thread_axis_strides(marker.group_lane_expr)
        if strides is None:
            continue
        reduce_axis = _lane_reduce_axis(marker)
        if reduce_axis is None:
            continue
        for axis in sorted(strides):
            if axis != reduce_axis:
                redundant_axes.add(axis)
    return _thread_axis_owner_exprs(redundant_axes, store_axes)


def _thread_axis_owner_exprs(
    redundant_axes: set[int],
    store_axes: _StoreThreadAxes | None,
) -> list[str]:
    """Choose one writer along physical axes absent from a logical store."""
    if store_axes is None:
        return []
    predicates: list[str] = []
    for axis in sorted(redundant_axes):
        if axis in store_axes.address or axis in store_axes.unique_predicate:
            continue
        if axis in store_axes.value:
            raise exc.BackendUnsupported(
                "cute",
                "lane reduction store aliases a value-varying thread axis",
            )
        if axis in store_axes.predicate and axis not in store_axes.zero_safe_predicate:
            raise exc.BackendUnsupported(
                "cute", "cannot infer unique ownership from a thread predicate"
            )
        predicates.append(f"cutlass.Int32(cute.arch.thread_idx()[{axis}]) == 0")
    return predicates


def _guard_stmt_with_owner(stmt: ast.AST, predicates: list[str]) -> ast.AST:
    """A copy of ``stmt`` that runs only where every predicate holds (an
    unguarded copy for none)."""
    if not predicates:
        return deep_clone(stmt)
    store = _validated_owner_store(stmt)
    if store is None:
        raise exc.BackendUnsupported("cute", "owner predicate requires one store")
    predicate = " and ".join(f"({expr})" for expr in dict.fromkeys(predicates))
    # The guarded copy keeps the statement's markers and source location
    # (``deep_clone``), as the lane split's other copies do.
    guarded = deep_clone(stmt)
    if _is_persistent_branch_vec_store(store):
        mask = store.args[5]
        mask_source = (
            None
            if isinstance(mask, ast.Constant) and mask.value is None
            else ast.unparse(mask)
        )
        # The copy's own store, which may sit under an ``if``
        # (``_is_isolated_store_statement``).
        guarded_store = _validated_owner_store(guarded)
        assert guarded_store is not None
        guarded_mask = expr_from_string(
            predicate if mask_source is None else f"({mask_source}) and ({predicate})"
        )
        assert isinstance(guarded_mask, ast.expr)
        guarded_store.args[5] = guarded_mask
        return guarded
    branch = cast("ast.If", statement_from_string(f"if {predicate}:\n    pass"))
    branch.body = [cast("ast.stmt", guarded)]
    return branch


def _has_impure_inline_asm(stmt: ast.AST) -> bool:
    """Recognize opaque assembly even when its result is an unused assignment.

    CuTe emits ``is_pure`` as a literal Boolean keyword. Inspect nested
    expressions too: a pure outer call does not make an impure argument pure.
    """
    for node in ast.walk(stmt):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_cute_inline_asm_elementwise"
        ):
            continue
        purity = next(kw.value for kw in node.keywords if kw.arg == "is_pure")
        assert isinstance(purity, ast.Constant) and type(purity.value) is bool
        if not purity.value:
            return True
    return False


def _has_side_effect(stmt: ast.AST) -> bool:
    """Return True when ``stmt`` produces an observable side effect (a store,
    an in-place / atomic write, impure assembly, or any non-plain-assignment such as
    an ``if mask: tensor[...].store(...)``)."""
    if isinstance(stmt, ast.Assign):
        # An atomic whose old value is bound to a name writes memory too
        # (``old = cute.arch.atomic_add(...)``).
        return _has_observable_memory_write(stmt) or _has_impure_inline_asm(stmt)
    # Conservatively treat structured / expression statements as
    # side-effecting (store calls live inside ``if`` blocks / bare exprs).
    return True


def _has_observable_memory_write(stmt: ast.AST) -> bool:
    """True for generated tensor stores/atomics, including guarded stores."""
    from ...ast_read_writes import ReadWrites

    if ReadWrites.from_ast(stmt).inplace_writes:
        return True
    return bool(_memory_write_calls(stmt))


def _validated_idempotent_store(stmt: ast.AST) -> ast.Call | None:
    """Return a single ordinary store whose surrounding expressions are pure.

    Atomics, helper writes, compound effects, synchronization, and calls with
    unknown purity are deliberately rejected: changing any of those from once
    per synthetic lane to once per chunk is not semantics-preserving.
    """
    store = _validated_owner_store(stmt)
    if store is None:
        return None
    if any(
        node is not store
        and isinstance(node, ast.Call)
        and not _is_proven_relocatable_call(node, allow_load=False)
        for node in ast.walk(stmt)
    ):
        raise exc.BackendUnsupported(
            "cute", "chunk recurrence store contains a call with unknown purity"
        )
    return store
