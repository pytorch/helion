"""Hoist repeated invariant reductions out of CuTe grid-lane loops.

A row-tiled persistent kernel can assign several rows to each thread while a
second thread axis performs the same reduction for every row.  Codegen then
has this shape (``row_lane`` is a compile-time loop)::

    for row_lane in range(16):
        row = row_base + row_lane
        if uniform_guard:
            ...
        else:
            key_acc = ...                         # independent of row_lane
            for k_lane in cutlass.range_constexpr(8):
                key_acc = key_acc + key[k_lane] ** 2
            key_norm = cute.arch.warp_reduction_sum(key_acc)
            ... state[row, k_lane] ...            # depends on row_lane

The invariant reduction is otherwise emitted sixteen times.  This pass first
unswitches a final, thread-uniform guard and then moves complete backward
slices for row-invariant warp reductions before the row loop.

The transform is deliberately fail closed.  It only handles generated lane
loops with statically profitable repeated work, simple assignments and
positive ``range_constexpr`` loops.  Loads may cross loop writes only when
runtime storage-specialization facts prove all source/destination tensor pairs
disjoint.  Atomics, unknown calls, ambiguous pointers, loop-carried
dependencies, and lane-dependent slices reject the rewrite.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import ExtendedAST
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from .fuse_two_pass_loads import _is_store_call
from .fuse_two_pass_loads import _store_tensor_roots
from .fuse_two_pass_loads import _tensor_arg_roots
from .licm_profitability import repeated_work_is_profitable

if TYPE_CHECKING:
    from collections.abc import Iterable


_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_PYTHON_FLOAT = "python.float"
_PYTHON_INT32 = "python.int32"
_PURE_OPERATOR_CALLS = {
    "operator.add",
    "operator.and_",
    "operator.eq",
    "operator.floordiv",
    "operator.ge",
    "operator.gt",
    "operator.index",
    "operator.invert",
    "operator.le",
    "operator.lshift",
    "operator.lt",
    "operator.mod",
    "operator.mul",
    "operator.ne",
    "operator.neg",
    "operator.or_",
    "operator.pos",
    "operator.pow",
    "operator.rshift",
    "operator.sub",
    "operator.truediv",
    "operator.xor",
}
_PURE_CUTLASS_CALLS = {
    "cutlass.BFloat16",
    "cutlass.Boolean",
    "cutlass.Float16",
    "cutlass.Float32",
    "cutlass.Float8E4M3FN",
    "cutlass.Int32",
    "cutlass.Int64",
    "cutlass.Uint16",
    "cutlass.Uint32",
    "cutlass.const_expr",
    "cutlass.range",
    "cutlass.range_constexpr",
}
_PURE_CUTE_MATH_CALLS = {
    "cute.math.absf",
    "cute.math.atan2",
    "cute.math.div",
    "cute.math.erf",
    "cute.math.exp",
    "cute.math.exp2",
    "cute.math.fma",
    "cute.math.log",
    "cute.math.log2",
    "cute.math.max",
    "cute.math.min",
    "cute.math.rcp",
    "cute.math.rsqrt",
    "cute.math.sqrt",
    "cute.math.tanh",
}
_PURE_WARP_REDUCTION_CALLS = {
    "cute.arch.warp_reduction_max",
    "cute.arch.warp_reduction_sum",
}
_PURE_LOAD_HELPERS = {"_cute_load_l2_evict_last"}
_CUTLASS_SCALAR_TYPES = {
    "cutlass.BFloat16": 2,
    "cutlass.Boolean": 1,
    "cutlass.Float16": 2,
    "cutlass.Float32": 4,
    "cutlass.Float8E4M3FN": 1,
    "cutlass.Int32": 4,
    "cutlass.Int64": 8,
    "cutlass.Uint16": 2,
    "cutlass.Uint32": 4,
}
_UNIFORM_GLOBAL_NAMES = {
    "abs",
    "bool",
    "cutlass",
    "cute",
    "float",
    "int",
    "ir",
    "len",
    "max",
    "min",
    "operator",
    "range",
}


@dataclasses.dataclass(frozen=True)
class _Effects:
    live_in: frozenset[str]
    writes: frozenset[str]


def _clone_ast(value: object) -> object:
    """Clone ordinary and Helion ExtendedAST nodes without using deepcopy."""
    if isinstance(value, list):
        return [_clone_ast(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_ast(item) for item in value)
    if not isinstance(value, ast.AST):
        return value
    fields = {field: _clone_ast(getattr(value, field)) for field in value._fields}
    if isinstance(value, ExtendedAST):
        return value.copy(**fields)
    return ast.copy_location(type(value)(**fields), value)


class _LoadedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def _names_read(node: ast.AST) -> set[str]:
    visitor = _LoadedNames()
    visitor.visit(node)
    return visitor.names


def _bound_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for elt in node.elts:
            result.update(_bound_names(elt))
        return result
    if isinstance(node, ast.Starred):
        return _bound_names(node.value)
    return set()


def _assignment_target_reads(node: ast.AST) -> set[str]:
    """Names evaluated while assigning to a non-name target."""
    if isinstance(node, ast.Name):
        return set()
    return _names_read(node)


def _assignment_target_writes(node: ast.AST) -> set[str]:
    return _bound_names(node)


def _mutation_roots(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _is_private_store_staging_append(child):
            assert isinstance(child.func, ast.Attribute)
            assert isinstance(child.func.value, ast.Name)
            result.add(child.func.value.id)
        if not isinstance(child, (ast.Attribute, ast.Subscript)) or not isinstance(
            child.ctx, ast.Store
        ):
            continue
        root: ast.AST = child.value
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if isinstance(root, ast.Name):
            result.add(root.id)
    return result


def _block_effects(body: list[ast.stmt], initially_written: set[str]) -> _Effects:
    live_in: set[str] = set()
    written = set(initially_written)
    all_writes: set[str] = set()
    for stmt in body:
        effects = _statement_effects(stmt)
        live_in.update(set(effects.live_in) - written)
        written.update(effects.writes)
        all_writes.update(effects.writes)
    return _Effects(frozenset(live_in), frozenset(all_writes))


def _statement_effects(stmt: ast.stmt) -> _Effects:
    """Return assignment-time live-ins and definite local-name writes.

    ``ReadWrites`` intentionally flattens nested statements, which makes names
    defined inside a synthetic K loop look like loop inputs.  This small
    structured analysis preserves assignment order, including loop-carried
    accumulator reads.
    """
    if isinstance(stmt, ast.Assign):
        reads = _names_read(stmt.value)
        writes: set[str] = set()
        for target in stmt.targets:
            reads.update(_assignment_target_reads(target))
            writes.update(_assignment_target_writes(target))
        return _Effects(frozenset(reads), frozenset(writes))
    if isinstance(stmt, ast.AnnAssign):
        reads = _assignment_target_reads(stmt.target)
        if stmt.value is not None:
            reads.update(_names_read(stmt.value))
        return _Effects(frozenset(reads), frozenset(_bound_names(stmt.target)))
    if isinstance(stmt, ast.AugAssign):
        reads = _names_read(stmt.value) | _names_read(stmt.target)
        if isinstance(stmt.target, ast.Name):
            reads.add(stmt.target.id)
        return _Effects(frozenset(reads), frozenset(_bound_names(stmt.target)))
    if isinstance(stmt, ast.For):
        bound = _bound_names(stmt.target)
        effects = _block_effects(stmt.body, bound)
        return _Effects(
            frozenset(_names_read(stmt.iter) | set(effects.live_in)),
            frozenset(bound | set(effects.writes)),
        )
    if isinstance(stmt, ast.If):
        body_effects = _block_effects(stmt.body, set())
        else_effects = _block_effects(stmt.orelse, set())
        return _Effects(
            frozenset(
                _names_read(stmt.test)
                | set(body_effects.live_in)
                | set(else_effects.live_in)
            ),
            frozenset(set(body_effects.writes) | set(else_effects.writes)),
        )

    # Unsupported compound statements are never selected for hoisting.  Their
    # conservative flattened effects still keep the surrounding dataflow safe.
    reads = _names_read(stmt)
    writes = {
        child.id
        for child in ast.walk(stmt)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }
    return _Effects(frozenset(reads), frozenset(writes))


def _call_path(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _is_atomic_call(call: ast.Call) -> bool:
    path = _call_path(call.func)
    return path is not None and (
        path.split(".")[-1].startswith("atomic_") or path.startswith("_cute_atomic_")
    )


def _is_proven_scalar_bitcast(call: ast.Call) -> bool:
    """Recognize the generated scalar-constructor ``.bitcast`` form only."""
    if (
        not isinstance(call.func, ast.Attribute)
        or call.func.attr != "bitcast"
        or len(call.args) != 1
        or call.keywords
        or not isinstance(call.func.value, ast.Call)
    ):
        return False
    receiver = call.func.value
    receiver_type = _call_path(receiver.func)
    target_type = _call_path(cast("ast.expr", call.args[0]))
    return (
        receiver_type in _CUTLASS_SCALAR_TYPES
        and len(receiver.args) == 1
        and not receiver.keywords
        and isinstance(call.args[0], ast.Attribute)
        and target_type in _CUTLASS_SCALAR_TYPES
        and _CUTLASS_SCALAR_TYPES[receiver_type] == _CUTLASS_SCALAR_TYPES[target_type]
    )


def _load_pointer(call: ast.Call) -> ast.AST | None:
    path = _call_path(call.func)
    if path == "cute.arch.load":
        return call.args[0] if call.args else None
    if path is not None and path.startswith("_cute_load_"):
        return call.args[0] if call.args else None
    if isinstance(call.func, ast.Attribute) and call.func.attr == "load":
        return call.func.value
    return None


def _is_generated_arch_load(call: ast.Call) -> bool:
    return (
        _call_path(call.func) == "cute.arch.load"
        and len(call.args) == 2
        and all(
            keyword.arg in {"cop", "level1_eviction_priority"}
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
            for keyword in call.keywords
        )
    )


def _is_allowed_pure_call(call: ast.Call) -> bool:
    path = _call_path(call.func)
    if path is None:
        if _is_proven_scalar_bitcast(call):
            return True
        # Generated pointer loads carry the address expression as their
        # receiver.  Alias checks below separately prove that receiver's tensor
        # root before any such load may move.
        return (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "load"
            and not call.args
            and not call.keywords
        )
    if path in {"abs", "max", "min", "range"}:
        return True
    if path in _PURE_OPERATOR_CALLS or path in _PURE_CUTLASS_CALLS:
        return True
    if path in _PURE_CUTE_MATH_CALLS:
        return True
    if path == "ir.VectorType.get":
        return True
    if path == "cute.arch.load":
        return _is_generated_arch_load(call)
    if path in {"cute.arch.block_idx", "cute.arch.lane_idx", "cute.arch.thread_idx"}:
        return True
    if path in _PURE_WARP_REDUCTION_CALLS:
        return True
    return path in _PURE_LOAD_HELPERS and len(call.args) == 2 and not call.keywords


def _is_private_store_staging_append(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id.startswith("_persistent_branch_store_values_")
    )


def _has_only_known_effects(node: ast.AST) -> bool:
    return all(
        _is_allowed_pure_call(child)
        or _is_store_call(child)
        or _is_private_store_staging_append(child)
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
    )


def _is_positive_constexpr_loop(stmt: ast.For) -> bool:
    if stmt.orelse or not isinstance(stmt.target, ast.Name):
        return False
    call = stmt.iter
    if (
        not isinstance(call, ast.Call)
        or _call_path(call.func) != "cutlass.range_constexpr"
    ):
        return False
    if len(call.args) != 1:
        return False
    extent = call.args[0]
    return (
        isinstance(extent, ast.Constant)
        and isinstance(extent.value, int)
        and extent.value > 0
    )


def _is_hoistable_statement(stmt: ast.stmt, tensor_names: set[str]) -> bool:
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                continue
            if not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id.startswith("_fuse_cache_")
            ):
                return False
    elif isinstance(stmt, ast.For):
        if not _is_positive_constexpr_loop(stmt):
            return False
        if not all(_is_hoistable_statement(child, tensor_names) for child in stmt.body):
            return False
    else:
        return False

    for child in ast.walk(stmt):
        if not isinstance(child, ast.Call):
            continue
        if _is_store_call(child) or _is_atomic_call(child):
            return False
        if not _is_allowed_pure_call(child):
            return False
        pointer = _load_pointer(child)
        if pointer is not None and _tensor_arg_roots(pointer, tensor_names) is None:
            return False
    return True


def _static_lane_extent(loop: ast.For) -> int | None:
    call = loop.iter
    if not isinstance(call, ast.Call) or _call_path(call.func) not in {
        "range",
        "cutlass.range_constexpr",
    }:
        return None
    if len(call.args) != 1:
        return None
    extent = call.args[0]
    if not isinstance(extent, ast.Constant) or not isinstance(extent.value, int):
        return None
    return extent.value


def _bounded_integer_interval(
    node: ast.AST,
    *,
    lane_var: str,
    lane_extent: int,
    thread_block_dims: tuple[int, int, int],
) -> tuple[int, int] | None:
    """Bound the small integer grammar used by generated tile masks."""

    def checked(lower: int, upper: int) -> tuple[int, int] | None:
        # Generated tile coordinates use Int32 arithmetic.  Decline to reason
        # through any intermediate that could wrap before the comparison.
        if _INT32_MIN <= lower <= upper <= _INT32_MAX:
            return lower, upper
        return None

    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return checked(node.value, node.value)
        return None
    if isinstance(node, ast.Name):
        return (0, lane_extent - 1) if node.id == lane_var else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        interval = _bounded_integer_interval(
            node.operand,
            lane_var=lane_var,
            lane_extent=lane_extent,
            thread_block_dims=thread_block_dims,
        )
        if interval is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return interval
        return checked(-interval[1], -interval[0])
    if (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and _call_path(node.func) in {"cutlass.Int32", "cutlass.Int64"}
    ):
        interval = _bounded_integer_interval(
            node.args[0],
            lane_var=lane_var,
            lane_extent=lane_extent,
            thread_block_dims=thread_block_dims,
        )
        if interval is None:
            return None
        return checked(*interval)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and _call_path(node.value.func) == "cute.arch.thread_idx"
        and not node.value.args
        and not node.value.keywords
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
        and 0 <= node.slice.value < len(thread_block_dims)
    ):
        return 0, thread_block_dims[node.slice.value] - 1
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult)
    ):
        left = _bounded_integer_interval(
            node.left,
            lane_var=lane_var,
            lane_extent=lane_extent,
            thread_block_dims=thread_block_dims,
        )
        right = _bounded_integer_interval(
            node.right,
            lane_var=lane_var,
            lane_extent=lane_extent,
            thread_block_dims=thread_block_dims,
        )
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return checked(left[0] + right[0], left[1] + right[1])
        if isinstance(node.op, ast.Sub):
            return checked(left[0] - right[1], left[1] - right[0])
        products = (
            left[0] * right[0],
            left[0] * right[1],
            left[1] * right[0],
            left[1] * right[1],
        )
        return checked(min(products), max(products))
    return None


def _comparison_is_always_true(
    node: ast.Compare,
    *,
    lane_var: str,
    lane_extent: int,
    thread_block_dims: tuple[int, int, int],
) -> bool:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    left = _bounded_integer_interval(
        node.left,
        lane_var=lane_var,
        lane_extent=lane_extent,
        thread_block_dims=thread_block_dims,
    )
    right = _bounded_integer_interval(
        node.comparators[0],
        lane_var=lane_var,
        lane_extent=lane_extent,
        thread_block_dims=thread_block_dims,
    )
    if left is None or right is None:
        return False
    op = node.ops[0]
    if isinstance(op, ast.Lt):
        return left[1] < right[0]
    if isinstance(op, ast.LtE):
        return left[1] <= right[0]
    if isinstance(op, ast.Gt):
        return left[0] > right[1]
    if isinstance(op, ast.GtE):
        return left[0] >= right[1]
    if isinstance(op, ast.Eq):
        return left[0] == left[1] == right[0] == right[1]
    if isinstance(op, ast.NotEq):
        return left[1] < right[0] or right[1] < left[0]
    return False


def _simplify_full_lane_bounds(
    loop: ast.For,
    lane_var: str,
    lane_extent: int,
    thread_block_dims: tuple[int, int, int] | None,
) -> ast.For:
    """Erase generated bounds conjuncts proven true for the full lane tile."""
    if thread_block_dims is None:
        return loop

    class _Simplifier(ast.NodeTransformer):
        def visit_Compare(self, node: ast.Compare) -> ast.AST:
            visited = self.generic_visit(node)
            assert isinstance(visited, ast.Compare)
            if _comparison_is_always_true(
                visited,
                lane_var=lane_var,
                lane_extent=lane_extent,
                thread_block_dims=thread_block_dims,
            ):
                return ast.copy_location(ast.Constant(value=True), visited)
            return visited

        def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
            visited = self.generic_visit(node)
            assert isinstance(visited, ast.BoolOp)
            if not isinstance(visited.op, ast.And):
                return visited
            values = [
                value
                for value in visited.values
                if not (isinstance(value, ast.Constant) and value.value is True)
            ]
            if not values:
                return ast.copy_location(ast.Constant(value=True), visited)
            if len(values) == 1:
                return ast.copy_location(values[0], visited)
            visited.values = values
            return visited

    result = _Simplifier().visit(loop)
    assert isinstance(result, ast.For)
    return result


def _is_warp_reduction_assignment(stmt: ast.stmt) -> bool:
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return False
    return any(
        isinstance(child, ast.Call)
        and ((_call_path(child.func) or "").startswith("cute.arch.warp_reduction_"))
        for child in ast.walk(stmt.value)
    )


def _backward_slice(body: list[ast.stmt], target_index: int) -> set[int]:
    target = body[target_index]
    effects = _statement_effects(target)
    needed = set(effects.writes)
    if not needed:
        return set()
    selected: set[int] = set()
    for index in range(target_index, -1, -1):
        stmt_effects = _statement_effects(body[index])
        if not (set(stmt_effects.writes) & needed):
            continue
        selected.add(index)
        needed.difference_update(stmt_effects.writes)
        needed.update(stmt_effects.live_in)
    return selected


def _slice_external_reads(body: list[ast.stmt], selected: set[int]) -> set[str]:
    needed: set[str] = set()
    for index in sorted(selected):
        effects = _statement_effects(body[index])
        needed.update(
            set(effects.live_in)
            - {
                name
                for prior in selected
                if prior < index
                for name in _statement_effects(body[prior]).writes
            }
        )
    return needed


def _slice_preserves_reaching_definitions(
    body: list[ast.stmt], selected: set[int]
) -> bool:
    effects = [_statement_effects(stmt) for stmt in body]
    # A selected read must not depend on a loop-carried or skipped definition.
    for index in selected:
        for name in effects[index].live_in:
            prior = [j for j in range(index) if name in effects[j].writes]
            if prior and prior[-1] not in selected:
                return False
            if not prior and any(name in effect.writes for effect in effects):
                return False
    # Hoisting a later definition must not shadow the value observed by an
    # earlier statement left in the loop.
    for index, effect in enumerate(effects):
        if index in selected:
            continue
        for name in effect.live_in:
            if any(j > index and name in effects[j].writes for j in selected):
                return False
    reaching_selected = {
        name for index in selected for name in effects[index].writes
    } | {name for index in selected for name in _mutation_roots(body[index])}
    consumed_selected: set[str] = set()
    for index, (stmt, effect) in enumerate(zip(body, effects, strict=True)):
        if index in selected:
            continue
        consumed_selected.update(set(effect.live_in) & reaching_selected)
        overwritten = set(effect.writes) | _mutation_roots(stmt)
        # A selected value used by this iteration must survive unchanged until
        # the next iteration because its defining slice no longer reruns.
        if consumed_selected & overwritten:
            return False
        reaching_selected.difference_update(overwritten)
    return True


def _canonical_name(name: str, rename_groups: dict[str, str]) -> str:
    seen: set[str] = set()
    while name in rename_groups and name not in seen:
        seen.add(name)
        name = rename_groups[name]
    return name


def _is_uniform_name(
    name: str, uniform_names: set[str], rename_groups: dict[str, str]
) -> bool:
    canonical = _canonical_name(name, rename_groups)
    return canonical in uniform_names or canonical.startswith(
        ("_BLOCK_SIZE_", "_RDIM_SIZE_")
    )


def _update_uniform_names(
    stmt: ast.stmt,
    uniform_names: set[str],
    rename_groups: dict[str, str],
) -> None:
    effects = _statement_effects(stmt)
    writes = {_canonical_name(name, rename_groups) for name in effects.writes}
    if not writes:
        return
    source = ast.unparse(stmt)
    if (
        not isinstance(stmt, ast.Assign)
        or _mutation_roots(stmt)
        or not _has_only_known_effects(stmt)
        or any(token in source for token in ("thread_idx", "lane_idx", "warp_idx"))
        or not all(
            _is_uniform_name(name, uniform_names, rename_groups)
            for name in effects.live_in
        )
    ):
        uniform_names.difference_update(writes)
        return
    uniform_names.update(writes)


def _has_rename_alias_conflict(
    body: list[ast.stmt],
    selected: set[int],
    rename_groups: dict[str, str],
) -> bool:
    """Reject hidden writes that become aliases under the final AST rename."""
    if not rename_groups:
        return False
    selected_names: set[str] = set()
    selected_writes: set[str] = set()
    other_reads: set[str] = set()
    other_writes: set[str] = set()
    for index, stmt in enumerate(body):
        effects = _statement_effects(stmt)
        names = set(effects.live_in) | set(effects.writes) | _mutation_roots(stmt)
        if index in selected:
            selected_names.update(names)
            selected_writes.update(effects.writes)
            selected_writes.update(_mutation_roots(stmt))
        else:
            other_reads.update(effects.live_in)
            other_writes.update(effects.writes)
            other_writes.update(_mutation_roots(stmt))
    selected_by_canonical: dict[str, set[str]] = {}
    for name in selected_names:
        selected_by_canonical.setdefault(
            _canonical_name(name, rename_groups), set()
        ).add(name)
    if any(len(names) > 1 for names in selected_by_canonical.values()):
        return True
    selected_touched_by_other_write = any(
        selected_name != other_name
        and _canonical_name(selected_name, rename_groups)
        == _canonical_name(other_name, rename_groups)
        for selected_name in selected_names
        for other_name in other_writes
    )
    selected_write_observed_by_other_read = any(
        selected_name != other_name
        and _canonical_name(selected_name, rename_groups)
        == _canonical_name(other_name, rename_groups)
        for selected_name in selected_writes
        for other_name in other_reads
    )
    return selected_touched_by_other_write or selected_write_observed_by_other_read


def _repeated_scalar_reaching_safe(
    body: list[ast.stmt],
    selected_names: set[str],
    direct_assignments: dict[str, list[int]],
) -> bool:
    """Ensure no statement reads a selected name before its first definition.

    Reads between selected definitions are valid because eligibility requires
    every definition to have byte-identical RHS syntax.  Moving one canonical
    definition to the preheader therefore preserves those values.
    """
    effects = [_statement_effects(stmt) for stmt in body]
    return not any(
        name in effects[index].live_in
        for name in selected_names
        for index in range(direct_assignments[name][0])
    )


def _global_store_roots(
    loop: ast.For, tensor_names: set[str]
) -> tuple[list[frozenset[str]], bool]:
    roots: list[frozenset[str]] = []
    for child in ast.walk(loop):
        if not isinstance(child, ast.Call):
            continue
        if _is_atomic_call(child):
            return [], False
        if not _is_store_call(child):
            continue
        resolved = _store_tensor_roots(child, tensor_names)
        if resolved is None:
            return [], False
        roots.append(resolved)
    return roots, True


def _loads_may_cross_loop_writes(
    statements: Iterable[ast.stmt],
    store_roots: list[frozenset[str]],
    tensor_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> bool:
    for stmt in statements:
        for child in ast.walk(stmt):
            if not isinstance(child, ast.Call):
                continue
            pointer = _load_pointer(child)
            if pointer is None:
                continue
            load_roots = _tensor_arg_roots(pointer, tensor_names)
            if load_roots is None:
                return False
            for load_root in load_roots:
                for destinations in store_roots:
                    for store_root in destinations:
                        if (
                            load_root == store_root
                            or frozenset((load_root, store_root))
                            not in proven_disjoint_tensor_pairs
                        ):
                            return False
    return True


@dataclasses.dataclass(frozen=True)
class _RmemInfo:
    extent: int
    dtype: str


@dataclasses.dataclass
class _ConstexprLoopInfo:
    outer_index: int
    loop: ast.For
    lane_var: str
    extent: int
    assignment_sites: dict[str, list[tuple[int, ast.Assign]]]


@dataclasses.dataclass(frozen=True)
class _CacheStore:
    info: _ConstexprLoopInfo
    body_index: int
    statement: ast.Assign
    target: ast.Subscript


@dataclasses.dataclass
class _VectorCandidate:
    info: _ConstexprLoopInfo
    body_index: int
    statement: ast.Assign
    target_name: str
    canonical: str
    dtype: str
    selected_statement_ids: set[int]
    cache_access_ids: dict[str, set[int]]
    cache_store_ids: set[int]
    cache_names: set[str]
    math_calls: int
    node_count: int


class _VectorCacheReject(Exception):
    pass


def _rmem_declaration(stmt: ast.stmt) -> tuple[str, _RmemInfo] | None:
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Call)
        and _call_path(stmt.value.func) == "cute.make_rmem_tensor"
        and len(stmt.value.args) == 2
        and not stmt.value.keywords
        and isinstance(stmt.value.args[0], ast.Constant)
        and isinstance(stmt.value.args[0].value, int)
        and stmt.value.args[0].value > 0
    ):
        return None
    dtype = _call_path(cast("ast.expr", stmt.value.args[1]))
    if dtype not in _CUTLASS_SCALAR_TYPES:
        return None
    return stmt.targets[0].id, _RmemInfo(stmt.value.args[0].value, dtype)


def _subscript_root(node: ast.Subscript) -> str | None:
    return node.value.id if isinstance(node.value, ast.Name) else None


def _identity_lane_index(node: ast.AST, lane_var: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id == lane_var
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.FloorDiv)
        and isinstance(node.right, ast.Constant)
        and node.right.value == 1
    ):
        node = node.left
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.Add, ast.Sub))
        and isinstance(node.right, ast.Constant)
        and node.right.value == 0
    ):
        node = node.left
    return isinstance(node, ast.Name) and node.id == lane_var


def _collect_subscript_accesses(body: list[ast.stmt]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for stmt in body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Subscript):
                continue
            root = _subscript_root(node)
            if root is not None:
                result.setdefault(root, set()).add(id(node))
    return result


def _non_subscript_access_names(body: list[ast.stmt]) -> set[str]:
    """Find names used other than as direct ``name[index]`` handles."""

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.result: set[str] = set()

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if not isinstance(node.value, ast.Name):
                self.visit(node.value)
            self.visit(node.slice)

        def visit_Name(self, node: ast.Name) -> None:
            self.result.add(node.id)

        def visit_Assign(self, node: ast.Assign) -> None:
            declaration = _rmem_declaration(node)
            if declaration is None:
                self.generic_visit(node)
                return
            self.visit(node.value)

    visitor = _Visitor()
    for stmt in body:
        visitor.visit(stmt)
    return visitor.result


def _constexpr_loop_info(stmt: ast.stmt, outer_index: int) -> _ConstexprLoopInfo | None:
    if not (
        isinstance(stmt, ast.For)
        and _is_positive_constexpr_loop(stmt)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.iter, ast.Call)
        and isinstance(stmt.iter.args[0], ast.Constant)
        and isinstance(stmt.iter.args[0].value, int)
        and not any(
            isinstance(
                node, (ast.Break, ast.Raise, ast.Return, ast.Yield, ast.YieldFrom)
            )
            for node in ast.walk(stmt)
        )
        and not any(
            stmt.target.id in _statement_effects(child).writes for child in stmt.body
        )
    ):
        return None
    sites: dict[str, list[tuple[int, ast.Assign]]] = {}
    for body_index, child in enumerate(stmt.body):
        if (
            isinstance(child, ast.Assign)
            and len(child.targets) == 1
            and isinstance(child.targets[0], ast.Name)
        ):
            sites.setdefault(child.targets[0].id, []).append((body_index, child))
    return _ConstexprLoopInfo(
        outer_index,
        stmt,
        stmt.target.id,
        stmt.iter.args[0].value,
        sites,
    )


def _cache_stores(
    infos: list[_ConstexprLoopInfo],
    rmem: dict[str, _RmemInfo],
) -> dict[str, list[_CacheStore]]:
    result: dict[str, list[_CacheStore]] = {}
    for info in infos:
        for body_index, stmt in enumerate(info.loop.body):
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Subscript)
            ):
                continue
            target = stmt.targets[0]
            root = _subscript_root(target)
            if (
                root not in rmem
                or rmem[root].extent != info.extent
                or not _identity_lane_index(target.slice, info.lane_var)
            ):
                continue
            result.setdefault(root, []).append(
                _CacheStore(info, body_index, stmt, target)
            )
    return result


@dataclasses.dataclass(frozen=True)
class _ExprContext:
    outer_index: int
    info: _ConstexprLoopInfo | None = None
    body_index: int = -1


class _VectorExpressionResolver:
    def __init__(
        self,
        *,
        outer_loop: ast.For,
        infos: list[_ConstexprLoopInfo],
        cache_stores: dict[str, list[_CacheStore]],
        rmem: dict[str, _RmemInfo],
        rename_groups: dict[str, str],
        tensor_dtypes: dict[str, str],
        tensor_names: set[str],
        float_scalar_names: set[str],
    ) -> None:
        self.outer_loop = outer_loop
        self.infos = infos
        self.cache_stores = cache_stores
        self.rmem = rmem
        self.rename_groups = rename_groups
        self.tensor_dtypes = tensor_dtypes
        self.tensor_names = tensor_names
        self.float_scalar_names = float_scalar_names
        self.outer_lane = cast("ast.Name", outer_loop.target).id
        self.outer_assignments: dict[str, list[tuple[int, ast.Assign]]] = {}
        for index, stmt in enumerate(outer_loop.body):
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                self.outer_assignments.setdefault(stmt.targets[0].id, []).append(
                    (index, stmt)
                )
        self.outer_writes = set(_statement_effects(outer_loop).writes)
        self.outer_mutations = set().union(
            *(_mutation_roots(stmt) for stmt in outer_loop.body)
        )
        aliases = [
            (stmt.targets[0].id, stmt.value.id)
            for stmt in ast.walk(outer_loop)
            if isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
        ]
        changed = True
        while changed:
            changed = False
            for left, right in aliases:
                if left in self.outer_mutations or right in self.outer_mutations:
                    previous_size = len(self.outer_mutations)
                    self.outer_mutations.update((left, right))
                    changed |= len(self.outer_mutations) != previous_size
        self.selected_statement_ids: set[int] = set()
        self.cache_access_ids: dict[str, set[int]] = {}
        self.cache_store_ids: set[int] = set()
        self.cache_names: set[str] = set()

    def normalize(self, node: ast.AST, context: _ExprContext) -> ast.AST:
        return self._normalize(node, context, set())

    def _normalize(
        self,
        node: ast.AST,
        context: _ExprContext,
        resolving: set[tuple[str, int]],
    ) -> ast.AST:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            return self._normalize_name(node.id, context, resolving)
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            cached = self._normalize_cache_read(node, context, resolving)
            if cached is not None:
                return cached

        clone = cast("ast.AST", _clone_ast(node))
        for field, value in ast.iter_fields(clone):
            if isinstance(value, ast.AST):
                setattr(clone, field, self._normalize(value, context, resolving))
            elif isinstance(value, list):
                setattr(
                    clone,
                    field,
                    [
                        self._normalize(item, context, resolving)
                        if isinstance(item, ast.AST)
                        else item
                        for item in value
                    ],
                )
        return clone

    def _normalize_name(
        self,
        name: str,
        context: _ExprContext,
        resolving: set[tuple[str, int]],
    ) -> ast.AST:
        canonical = _canonical_name(name, self.rename_groups)
        if canonical == _canonical_name(self.outer_lane, self.rename_groups):
            raise _VectorCacheReject
        if context.info is not None and name == context.info.lane_var:
            return ast.Name(id="__constexpr_lane", ctx=ast.Load())
        if context.info is not None and canonical == _canonical_name(
            context.info.lane_var, self.rename_groups
        ):
            raise _VectorCacheReject

        if context.info is not None and name in context.info.assignment_sites:
            sites = context.info.assignment_sites[name]
            if len(sites) != 1 or sites[0][0] >= context.body_index:
                raise _VectorCacheReject
            body_index, stmt = sites[0]
            if any(
                canonical
                in {
                    _canonical_name(item, self.rename_groups)
                    for item in set(_statement_effects(prior).writes)
                    | _mutation_roots(prior)
                }
                for prior in context.info.loop.body[body_index + 1 : context.body_index]
            ) or canonical in {
                _canonical_name(item, self.rename_groups)
                for item in self.outer_mutations
            }:
                raise _VectorCacheReject
            key = ("inner", id(stmt))
            if key in resolving:
                raise _VectorCacheReject
            self.selected_statement_ids.add(id(stmt))
            return self._normalize(
                stmt.value,
                _ExprContext(context.info.outer_index, context.info, body_index),
                {*resolving, key},
            )

        preceding = [
            (index, stmt)
            for index, stmt in self.outer_assignments.get(name, ())
            if index < context.outer_index
        ]
        if preceding:
            outer_index, stmt = preceding[-1]
            if any(
                canonical
                in {
                    _canonical_name(item, self.rename_groups)
                    for item in set(_statement_effects(prior).writes)
                    | _mutation_roots(prior)
                }
                for prior in self.outer_loop.body[outer_index + 1 : context.outer_index]
            ) or canonical in {
                _canonical_name(item, self.rename_groups)
                for item in self.outer_mutations
            }:
                raise _VectorCacheReject
            key = ("outer", id(stmt))
            if key in resolving:
                raise _VectorCacheReject
            self.selected_statement_ids.add(id(stmt))
            return self._normalize(
                stmt.value,
                _ExprContext(outer_index),
                {*resolving, key},
            )

        outer_mutations = self.outer_writes | self.outer_mutations
        if name in outer_mutations or canonical in {
            _canonical_name(item, self.rename_groups) for item in outer_mutations
        }:
            raise _VectorCacheReject
        return ast.Name(id=canonical, ctx=ast.Load())

    def _normalize_cache_read(
        self,
        node: ast.Subscript,
        context: _ExprContext,
        resolving: set[tuple[str, int]],
    ) -> ast.AST | None:
        root = _subscript_root(node)
        if root not in self.rmem:
            return None
        if (
            context.info is None
            or self.rmem[root].extent != context.info.extent
            or not _identity_lane_index(node.slice, context.info.lane_var)
        ):
            raise _VectorCacheReject
        stores = self.cache_stores.get(root, ())
        if len(stores) != 1:
            raise _VectorCacheReject
        producer = stores[0]
        if producer.info.extent != context.info.extent or not (
            producer.info.outer_index < context.info.outer_index
            or producer.info.outer_index == context.info.outer_index
            and producer.body_index < context.body_index
        ):
            raise _VectorCacheReject
        key = ("cache", id(producer.statement))
        if key in resolving:
            raise _VectorCacheReject
        self.cache_names.add(root)
        self.cache_store_ids.add(id(producer.statement))
        self.cache_access_ids.setdefault(root, set()).update(
            (id(node), id(producer.target))
        )
        normalized = self._normalize(
            producer.statement.value,
            _ExprContext(
                producer.info.outer_index,
                producer.info,
                producer.body_index,
            ),
            {*resolving, key},
        )
        if (
            _infer_scalar_dtype(
                normalized,
                self.tensor_dtypes,
                self.tensor_names,
                self.float_scalar_names,
            )
            != self.rmem[root].dtype
        ):
            raise _VectorCacheReject
        return normalized


def _merge_scalar_dtype(left: str | None, right: str | None) -> str | None:
    if left is None or right is None:
        return None
    python_scalars = {_PYTHON_FLOAT, _PYTHON_INT32}
    if left in python_scalars or right in python_scalars:
        other = right if left in python_scalars else left
        return "cutlass.Float32" if other == "cutlass.Float32" else None
    return left if left == right else None


def _infer_scalar_dtype(
    node: ast.AST,
    tensor_dtypes: dict[str, str],
    tensor_names: set[str],
    float_scalar_names: set[str],
) -> str | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, float):
            return _PYTHON_FLOAT
        if (
            isinstance(node.value, (bool, int))
            and _INT32_MIN <= node.value <= _INT32_MAX
        ):
            return _PYTHON_INT32
        return None
    if isinstance(node, ast.Name):
        return _PYTHON_FLOAT if node.id in float_scalar_names else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _infer_scalar_dtype(
            node.operand, tensor_dtypes, tensor_names, float_scalar_names
        )
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Mult, ast.Sub)
    ):
        return _merge_scalar_dtype(
            _infer_scalar_dtype(
                node.left, tensor_dtypes, tensor_names, float_scalar_names
            ),
            _infer_scalar_dtype(
                node.right, tensor_dtypes, tensor_names, float_scalar_names
            ),
        )
    if isinstance(node, ast.IfExp):
        body_dtype = _infer_scalar_dtype(
            node.body, tensor_dtypes, tensor_names, float_scalar_names
        )
        else_dtype = _infer_scalar_dtype(
            node.orelse, tensor_dtypes, tensor_names, float_scalar_names
        )
        return body_dtype if body_dtype == else_dtype else None
    if not isinstance(node, ast.Call):
        return None
    path = _call_path(node.func)
    if path in _CUTLASS_SCALAR_TYPES:
        return path
    if _is_proven_scalar_bitcast(node):
        return _call_path(cast("ast.expr", node.args[0]))
    pointer = _load_pointer(node)
    if pointer is not None:
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and not node.args
            and not node.keywords
        ):
            if not _is_generated_arch_load(node):
                return None
            explicit_dtype = _call_path(cast("ast.expr", node.args[1]))
            return explicit_dtype if explicit_dtype in _CUTLASS_SCALAR_TYPES else None
        roots = _tensor_arg_roots(pointer, tensor_names)
        if roots is None or len(roots) != 1:
            return None
        return tensor_dtypes.get(next(iter(roots)))
    if path in _PURE_CUTE_MATH_CALLS and node.args:
        result = _infer_scalar_dtype(
            node.args[0], tensor_dtypes, tensor_names, float_scalar_names
        )
        for arg in node.args[1:]:
            result = _merge_scalar_dtype(
                result,
                _infer_scalar_dtype(
                    arg, tensor_dtypes, tensor_names, float_scalar_names
                ),
            )
        return result if result not in {_PYTHON_FLOAT, _PYTHON_INT32} else None
    return None


def _vector_chain_expression_safe(expression: ast.expr, tensor_names: set[str]) -> bool:
    if any(
        isinstance(
            node,
            (
                ast.Await,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Lambda,
                ast.ListComp,
                ast.NamedExpr,
                ast.SetComp,
                ast.Yield,
                ast.YieldFrom,
            ),
        )
        for node in ast.walk(expression)
    ):
        return False
    for node in ast.walk(expression):
        if not isinstance(node, ast.Call):
            continue
        if not _is_allowed_pure_call(node) or _is_store_call(node):
            return False
        pointer = _load_pointer(node)
        if pointer is not None and _tensor_arg_roots(pointer, tensor_names) is None:
            return False
    return True


def _vector_chain_assignment_safe(stmt: ast.stmt, tensor_names: set[str]) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and _vector_chain_expression_safe(stmt.value, tensor_names)
    )


def _nested_rename_alias_conflict(
    loop: ast.For,
    selected_statement_ids: set[int],
    rename_groups: dict[str, str],
) -> bool:
    if not rename_groups:
        return False
    leaves = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.For, ast.If))
    ]
    selected = {
        index for index, stmt in enumerate(leaves) if id(stmt) in selected_statement_ids
    }
    return _has_rename_alias_conflict(leaves, selected, rename_groups)


def _candidate_used_later(info: _ConstexprLoopInfo, body_index: int, name: str) -> bool:
    return any(name in _names_read(stmt) for stmt in info.loop.body[body_index + 1 :])


def _build_vector_candidates(
    loop: ast.For,
    infos: list[_ConstexprLoopInfo],
    stores: dict[str, list[_CacheStore]],
    rmem: dict[str, _RmemInfo],
    tensor_dtypes: dict[str, str],
    tensor_names: set[str],
    float_scalar_names: set[str],
    rename_groups: dict[str, str],
) -> list[_VectorCandidate]:
    candidates: list[_VectorCandidate] = []
    for info in infos:
        for target_name, sites in info.assignment_sites.items():
            if len(sites) != 1:
                continue
            body_index, stmt = sites[0]
            if not all(
                isinstance(prefix_stmt, ast.Assign)
                for prefix_stmt in info.loop.body[: body_index + 1]
            ):
                continue
            if not _candidate_used_later(info, body_index, target_name):
                continue
            resolver = _VectorExpressionResolver(
                outer_loop=loop,
                infos=infos,
                cache_stores=stores,
                rmem=rmem,
                rename_groups=rename_groups,
                tensor_dtypes=tensor_dtypes,
                tensor_names=tensor_names,
                float_scalar_names=float_scalar_names,
            )
            resolver.selected_statement_ids.add(id(stmt))
            try:
                normalized = resolver.normalize(
                    stmt.value,
                    _ExprContext(info.outer_index, info, body_index),
                )
            except _VectorCacheReject:
                continue
            selected_statements = [
                node
                for node in ast.walk(loop)
                if isinstance(node, ast.stmt)
                and id(node) in resolver.selected_statement_ids
            ]
            if not selected_statements or not all(
                _vector_chain_assignment_safe(selected, tensor_names)
                for selected in selected_statements
            ):
                continue
            dtype = _infer_scalar_dtype(
                normalized, tensor_dtypes, tensor_names, float_scalar_names
            )
            # Keep the first general form deliberately Float32-only.  CuTe's
            # promotion rules for narrow floats and integers are operation
            # dependent; widening this requires a complete promotion lattice.
            if dtype != "cutlass.Float32":
                continue
            math_calls = sum(
                isinstance(node, ast.Call)
                and _call_path(node.func) in _PURE_CUTE_MATH_CALLS
                for node in ast.walk(normalized)
            )
            if math_calls == 0:
                continue
            candidates.append(
                _VectorCandidate(
                    info,
                    body_index,
                    stmt,
                    target_name,
                    ast.dump(normalized, include_attributes=False),
                    dtype,
                    resolver.selected_statement_ids,
                    resolver.cache_access_ids,
                    resolver.cache_store_ids,
                    resolver.cache_names,
                    math_calls,
                    sum(1 for _ in ast.walk(normalized)),
                )
            )
    return candidates


def _fresh_names(occupied: set[str], count: int) -> list[str]:
    result: list[str] = []
    suffix = 0
    while len(result) < count:
        candidate = f"_lane_invariant_{suffix}"
        suffix += 1
        if candidate not in occupied:
            occupied.add(candidate)
            result.append(candidate)
    return result


class _RenameNames(ast.NodeTransformer):
    def __init__(self, renames: dict[str, str]) -> None:
        self.renames = renames

    def visit_Name(self, node: ast.Name) -> ast.Name:
        replacement = self.renames.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)


def _rewrite_selected_loop_body(
    info: _ConstexprLoopInfo,
    *,
    selected_statement_ids: set[int],
    cache_store_ids: set[int],
    replacements: dict[int, ast.Assign],
    live_out: set[str],
) -> None:
    live = set(live_out)
    kept_reversed: list[ast.stmt] = []
    for stmt in reversed(info.loop.body):
        statement_id = id(stmt)
        replacement = replacements.get(statement_id)
        if replacement is not None:
            effects = _statement_effects(replacement)
            live.difference_update(effects.writes)
            live.update(effects.live_in)
            kept_reversed.append(replacement)
            continue
        if statement_id in cache_store_ids:
            continue
        if statement_id in selected_statement_ids:
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in live
            ):
                continue
        effects = _statement_effects(stmt)
        live.difference_update(effects.writes)
        live.update(effects.live_in)
        kept_reversed.append(stmt)
    info.loop.body = list(reversed(kept_reversed))


def _live_before(statements: list[ast.stmt]) -> set[str]:
    live: set[str] = set()
    for stmt in reversed(statements):
        effects = _statement_effects(stmt)
        live.difference_update(effects.writes)
        live.update(effects.live_in)
    return live


def _prune_selected_outer_assignments(
    loop: ast.For, selected_statement_ids: set[int]
) -> None:
    live: set[str] = set()
    kept_reversed: list[ast.stmt] = []
    for stmt in reversed(loop.body):
        effects = _statement_effects(stmt)
        if (
            id(stmt) in selected_statement_ids
            and isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id not in live
        ):
            continue
        live.difference_update(effects.writes)
        live.update(effects.live_in)
        kept_reversed.append(stmt)
    loop.body = list(reversed(kept_reversed))


def _selected_cache_dependencies_are_private(
    loop: ast.For,
    group: list[_VectorCandidate],
    selected_statement_ids: set[int],
    cache_store_ids: set[int],
    cache_names: set[str],
) -> bool:
    terminal_ids = {id(candidate.statement) for candidate in group}
    terminal_names = {candidate.target_name for candidate in group}
    assignments = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Assign)
        and id(node) in selected_statement_ids - terminal_ids
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    cache_dependent_names = {
        cast("ast.Name", stmt.targets[0]).id
        for stmt in assignments
        if any(
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and _subscript_root(node) in cache_names
            for node in ast.walk(stmt.value)
        )
    }
    changed = True
    while changed:
        previous_size = len(cache_dependent_names)
        cache_dependent_names.update(
            cast("ast.Name", stmt.targets[0]).id
            for stmt in assignments
            if _names_read(stmt.value) & cache_dependent_names
        )
        changed = len(cache_dependent_names) != previous_size
    if cache_dependent_names & terminal_names:
        return False
    allowed_owners = selected_statement_ids | cache_store_ids

    class _ReadOwnerVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.owner: int | None = None
            self.external_read = False

        def generic_visit(self, node: ast.AST) -> None:
            previous = self.owner
            if isinstance(node, ast.stmt):
                self.owner = id(node)
            super().generic_visit(node)
            self.owner = previous

        def visit_Name(self, node: ast.Name) -> None:
            if (
                isinstance(node.ctx, ast.Load)
                and node.id in cache_dependent_names
                and self.owner not in allowed_owners
            ):
                self.external_read = True

    visitor = _ReadOwnerVisitor()
    visitor.visit(loop)
    return not visitor.external_read


def _hoist_lane_invariant_constexpr_chain(
    loop: ast.For,
    *,
    lane_extent: int,
    tensor_names: set[str],
    tensor_dtypes: dict[str, str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    rename_groups: dict[str, str],
    rmem: dict[str, _RmemInfo],
    all_cache_access_counts: dict[str, int],
    non_subscript_access_names: set[str],
    obsolete_rmem: set[str],
    float_scalar_names: set[str],
    occupied_names: set[str],
) -> list[ast.stmt]:
    """Cache one repeated row-invariant constexpr-lane value in registers."""
    if (
        not isinstance(loop.target, ast.Name)
        or not _has_only_known_effects(loop)
        or any(
            isinstance(
                node,
                (
                    ast.Break,
                    ast.Continue,
                    ast.Raise,
                    ast.Return,
                    ast.Yield,
                    ast.YieldFrom,
                ),
            )
            for node in ast.walk(loop)
        )
    ):
        return [loop]
    infos = [
        info
        for index, stmt in enumerate(loop.body)
        if (info := _constexpr_loop_info(stmt, index)) is not None
    ]
    if len(infos) < 2:
        return [loop]
    stores = _cache_stores(infos, rmem)
    candidates = _build_vector_candidates(
        loop,
        infos,
        stores,
        rmem,
        tensor_dtypes,
        tensor_names,
        float_scalar_names,
        rename_groups,
    )
    groups: dict[tuple[int, str, str], list[_VectorCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(
            (candidate.info.extent, candidate.dtype, candidate.canonical), []
        ).append(candidate)
    viable = [
        group
        for group in groups.values()
        if len({id(candidate.info.loop) for candidate in group}) >= 2
    ]
    if not viable:
        return [loop]
    viable.sort(
        key=lambda group: (
            max(candidate.math_calls for candidate in group),
            max(candidate.node_count for candidate in group),
        ),
        reverse=True,
    )
    group = viable[0]
    selected_statement_ids = set().union(
        *(candidate.selected_statement_ids for candidate in group)
    )
    cache_store_ids = set().union(*(candidate.cache_store_ids for candidate in group))
    cache_names = set().union(*(candidate.cache_names for candidate in group))
    cache_access_ids: dict[str, set[int]] = {}
    for candidate in group:
        for name, accesses in candidate.cache_access_ids.items():
            cache_access_ids.setdefault(name, set()).update(accesses)
    if not _selected_cache_dependencies_are_private(
        loop,
        group,
        selected_statement_ids,
        cache_store_ids,
        cache_names,
    ):
        return [loop]
    if (
        not cache_names
        or any(
            name not in rmem
            or name in non_subscript_access_names
            or all_cache_access_counts.get(name, 0)
            != len(cache_access_ids.get(name, set()))
            for name in cache_names
        )
        or cache_names & set(_statement_effects(loop).writes)
    ):
        return [loop]
    extent = group[0].info.extent
    dtype = group[0].dtype
    reusable = sorted(
        name
        for name in cache_names
        if rmem[name] == _RmemInfo(extent=extent, dtype=dtype)
    )
    if not reusable:
        return [loop]
    cache_name = reusable[0]
    source = next((candidate for candidate in group if not candidate.cache_names), None)
    if source is None:
        return [loop]
    movable_statement_ids = selected_statement_ids | cache_store_ids
    if _nested_rename_alias_conflict(loop, movable_statement_ids, rename_groups):
        return [loop]
    selected_statements = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.stmt) and id(node) in movable_statement_ids
    ]
    if not all(
        _vector_chain_expression_safe(node.value, tensor_names)
        for node in selected_statements
        if id(node) in cache_store_ids and isinstance(node, ast.Assign)
    ):
        return [loop]
    store_roots, stores_resolved = _global_store_roots(loop, tensor_names)
    if not stores_resolved or not _loads_may_cross_loop_writes(
        selected_statements,
        store_roots,
        tensor_names,
        proven_disjoint_tensor_pairs,
    ):
        return [loop]

    source_top = [
        stmt for stmt in loop.body if id(stmt) in source.selected_statement_ids
    ]
    source_inner = [
        stmt
        for stmt in source.info.loop.body
        if id(stmt) in source.selected_statement_ids
    ]
    if not source_inner or source.statement not in source_inner:
        return [loop]
    assigned_names = [
        cast("ast.Name", stmt.targets[0]).id
        for stmt in [*source_top, *source_inner]
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ]
    fresh = _fresh_names(occupied_names, len(set(assigned_names)) + 1)
    fresh_lane = fresh[0]
    ordered_assigned = list(dict.fromkeys(assigned_names))
    renames = dict(zip(ordered_assigned, fresh[1:], strict=True))
    renames[source.info.lane_var] = fresh_lane
    rename = _RenameNames(renames)
    hoisted_top = [
        rename.visit(cast("ast.stmt", _clone_ast(stmt))) for stmt in source_top
    ]
    hoisted_inner = [
        rename.visit(cast("ast.stmt", _clone_ast(stmt))) for stmt in source_inner
    ]
    terminal_name = renames[source.target_name]
    hoisted_inner.append(
        statement_from_string(f"{cache_name}[{fresh_lane}] = {terminal_name}")
    )
    precompute = create(
        ast.For,
        target=create(ast.Name, id=fresh_lane, ctx=ast.Store()),
        iter=expr_from_string(f"cutlass.range_constexpr({extent})"),
        body=hoisted_inner,
        orelse=[],
        type_comment=None,
    )
    if not repeated_work_is_profitable(
        [*cast("list[ast.stmt]", hoisted_top), precompute],
        repeat_extent=lane_extent,
        # Only the cached vector remains live across the row loop; scalar
        # temporaries feeding its precompute die before that loop starts.
        live_scalars=extent,
    ):
        return [loop]

    replacements: dict[int, ast.Assign] = {}
    for candidate in group:
        replacement = statement_from_string(
            f"{candidate.target_name} = {cache_name}[{candidate.info.lane_var}]"
        )
        assert isinstance(replacement, ast.Assign)
        replacements[id(candidate.statement)] = replacement
    for info in reversed(infos):
        live_out = _live_before(loop.body[info.outer_index + 1 :])
        _rewrite_selected_loop_body(
            info,
            selected_statement_ids=selected_statement_ids,
            cache_store_ids=cache_store_ids,
            replacements=replacements,
            live_out=live_out,
        )
    _prune_selected_outer_assignments(loop, selected_statement_ids)
    obsolete_rmem.update(cache_names - {cache_name})
    return [*cast("list[ast.stmt]", hoisted_top), precompute, loop]


def _hoist_constexpr_scalar_invariants(
    loop: ast.For,
    tensor_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    rename_groups: dict[str, str],
) -> list[ast.stmt]:
    """Move scalar assignments independent of a positive constexpr loop.

    This is intentionally limited to a flat generated assignment block.  It
    primarily removes head-scalar loads, exp/rcp, and norm rsqrt operations
    that would otherwise execute once per vector lane.
    """
    if not _is_positive_constexpr_loop(loop) or not isinstance(loop.target, ast.Name):
        return [loop]
    loop_extent = _static_lane_extent(loop)
    assert loop_extent is not None
    body = list(loop.body)
    if not body or not all(isinstance(stmt, ast.Assign) for stmt in body):
        return [loop]
    # An unknown call can mutate an otherwise invariant scalar through an
    # opaque alias.  Generated arithmetic/load calls are all covered by the
    # explicit purity grammar.
    if any(
        isinstance(child, ast.Call)
        and not (_is_allowed_pure_call(child) or _is_store_call(child))
        for stmt in body
        for child in ast.walk(stmt)
    ):
        return [loop]

    effects = [_statement_effects(stmt) for stmt in body]
    write_sites: dict[str, list[int]] = {}
    for index, effect in enumerate(effects):
        for name in effect.writes:
            write_sites.setdefault(name, []).append(index)

    selected: set[int] = set()
    for index, stmt in enumerate(body):
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            continue
        target = stmt.targets[0].id
        if write_sites.get(target) != [index]:
            continue
        if not _is_hoistable_statement(stmt, tensor_names):
            continue
        dependencies_ok = True
        for name in effects[index].live_in:
            if name == loop.target.id:
                dependencies_ok = False
                break
            sites = write_sites.get(name)
            if sites is not None and (len(sites) != 1 or sites[0] not in selected):
                dependencies_ok = False
                break
        if dependencies_ok:
            selected.add(index)

    if not selected:
        return [loop]
    if not _slice_preserves_reaching_definitions(body, selected):
        return [loop]
    if _has_rename_alias_conflict(body, selected, rename_groups):
        return [loop]
    selected_external_reads = _slice_external_reads(body, selected)
    if any(selected_external_reads & _mutation_roots(stmt) for stmt in body):
        return [loop]
    store_roots, stores_resolved = _global_store_roots(loop, tensor_names)
    if not stores_resolved or not _loads_may_cross_loop_writes(
        (body[index] for index in sorted(selected)),
        store_roots,
        tensor_names,
        proven_disjoint_tensor_pairs,
    ):
        return [loop]
    invariant = [body[index] for index in sorted(selected)]
    remainder = [stmt for index, stmt in enumerate(body) if index not in selected]
    if not remainder:
        return [loop]
    invariant_writes = set().union(
        *(set(effects[index].writes) for index in selected),
        set(),
    )
    remainder_reads = _names_read(ast.Module(body=remainder, type_ignores=[]))
    if not repeated_work_is_profitable(
        invariant,
        repeat_extent=loop_extent,
        live_scalars=max(1, len(invariant_writes & remainder_reads)),
    ):
        return [loop]
    return [*invariant, _clone_lane_loop(loop, remainder)]


def _rewrite_inner_constexpr_loops(
    body: list[ast.stmt],
    tensor_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    rename_groups: dict[str, str],
) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for stmt in body:
        for field in ("body", "orelse", "finalbody"):
            child = getattr(stmt, field, None)
            if isinstance(child, list) and all(
                isinstance(item, ast.stmt) for item in child
            ):
                setattr(
                    stmt,
                    field,
                    _rewrite_inner_constexpr_loops(
                        child,
                        tensor_names,
                        proven_disjoint_tensor_pairs,
                        rename_groups,
                    ),
                )
        if isinstance(stmt, ast.For) and _is_positive_constexpr_loop(stmt):
            result.extend(
                _hoist_constexpr_scalar_invariants(
                    stmt,
                    tensor_names,
                    proven_disjoint_tensor_pairs,
                    rename_groups,
                )
            )
        else:
            result.append(stmt)
    return result


def _hoist_repeated_scalar_invariants(
    loop: ast.For,
    lane_var: str,
    lane_extent: int,
    tensor_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    rename_groups: dict[str, str],
) -> list[ast.stmt]:
    """Hoist identical scalar definitions shared by every row iteration.

    Inner-loop LICM exposes expensive scalar chains at row-loop scope.  The
    generated code often repeats the exact same definition in its projection
    and state-update sweeps.  A name is eligible only when *every* write to it
    is a simple assignment with the same RHS and all in-loop dependencies are
    themselves already proven invariant.
    """
    body = list(loop.body)
    effects = [_statement_effects(stmt) for stmt in body]
    all_write_sites: dict[str, list[int]] = {}
    direct_assignments: dict[str, list[int]] = {}
    for index, (stmt, effect) in enumerate(zip(body, effects, strict=True)):
        for name in effect.writes:
            all_write_sites.setdefault(name, []).append(index)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            direct_assignments.setdefault(stmt.targets[0].id, []).append(index)

    selected_names: set[str] = set()
    selected_indices: set[int] = set()
    ordered_names = sorted(
        direct_assignments, key=lambda name: direct_assignments[name][0]
    )
    changed = True
    while changed:
        changed = False
        for name in ordered_names:
            if name in selected_names:
                continue
            sites = direct_assignments[name]
            if all_write_sites.get(name) != sites:
                continue
            first = cast("ast.Assign", body[sites[0]])
            if isinstance(
                first.value,
                (
                    ast.Dict,
                    ast.DictComp,
                    ast.List,
                    ast.ListComp,
                    ast.Set,
                    ast.SetComp,
                    ast.Tuple,
                ),
            ):
                continue
            rhs = ast.unparse(first.value)
            if any(
                ast.unparse(cast("ast.Assign", body[index]).value) != rhs
                or not _is_hoistable_statement(body[index], tensor_names)
                for index in sites
            ):
                continue
            dependencies = set().union(*(effects[index].live_in for index in sites))
            if lane_var in dependencies or name in dependencies:
                continue
            dependencies_ok = True
            for dependency in dependencies:
                dependency_sites = all_write_sites.get(dependency)
                if dependency_sites is None:
                    continue
                if dependency not in selected_names:
                    dependencies_ok = False
                    break
                if min(dependency_sites) > min(sites):
                    dependencies_ok = False
                    break
            if not dependencies_ok:
                continue
            tentative_indices = selected_indices | set(sites)
            tentative_names = selected_names | {name}
            if not _repeated_scalar_reaching_safe(
                body, tentative_names, direct_assignments
            ):
                continue
            if _has_rename_alias_conflict(body, tentative_indices, rename_groups):
                continue
            tentative_external_reads = _slice_external_reads(body, tentative_indices)
            if any(
                tentative_external_reads & _mutation_roots(stmt)
                for index, stmt in enumerate(body)
                if index not in tentative_indices
            ):
                continue
            selected_names.add(name)
            selected_indices.update(sites)
            changed = True

    if not selected_indices:
        return [loop]
    # Opaque calls elsewhere in the row loop may mutate a loaded source behind
    # an alias.  The generated calls accepted here are explicit and audited.
    if not _has_only_known_effects(loop):
        return [loop]

    store_roots, stores_resolved = _global_store_roots(loop, tensor_names)
    if not stores_resolved or not _loads_may_cross_loop_writes(
        (body[index] for index in sorted(selected_indices)),
        store_roots,
        tensor_names,
        proven_disjoint_tensor_pairs,
    ):
        return [loop]
    # Keep one canonical definition per name, in first-definition order.
    hoisted = [
        body[direct_assignments[name][0]]
        for name in ordered_names
        if name in selected_names
    ]
    remainder = [
        stmt for index, stmt in enumerate(body) if index not in selected_indices
    ]
    if not remainder:
        return [loop]
    remainder_reads = _names_read(ast.Module(body=remainder, type_ignores=[]))
    if not repeated_work_is_profitable(
        hoisted,
        repeat_extent=lane_extent,
        live_scalars=max(1, len(selected_names & remainder_reads)),
    ):
        return [loop]
    return [*hoisted, _clone_lane_loop(loop, remainder)]


def _clone_lane_loop(loop: ast.For, body: list[ast.stmt]) -> ast.For:
    assert isinstance(loop.target, ast.Name)
    result = create(
        ast.For,
        target=create(ast.Name, id=loop.target.id, ctx=ast.Store()),
        iter=expr_from_string(ast.unparse(loop.iter)),
        body=body,
        orelse=[],
        type_comment=loop.type_comment,
    )
    lane_var = getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None)
    if lane_var is not None:
        setattr(result, HELION_LANE_LOOP_VAR_ATTR, lane_var)
    return result


def _hoist_reductions_from_loop(
    loop: ast.For,
    lane_var: str,
    lane_extent: int,
    tensor_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    rename_groups: dict[str, str],
) -> list[ast.stmt]:
    if not _has_only_known_effects(loop):
        return [loop]
    body = list(loop.body)
    selected: set[int] = set()
    for index, stmt in enumerate(body):
        if not _is_warp_reduction_assignment(stmt):
            continue
        candidate = _backward_slice(body, index)
        if not candidate:
            continue
        external_reads = _slice_external_reads(body, candidate)
        if lane_var in external_reads:
            continue
        if not all(_is_hoistable_statement(body[i], tensor_names) for i in candidate):
            continue
        combined = selected | candidate
        if not _slice_preserves_reaching_definitions(body, combined):
            continue
        if _has_rename_alias_conflict(body, combined, rename_groups):
            continue
        combined_external_reads = _slice_external_reads(body, combined)
        if any(
            combined_external_reads & _mutation_roots(other)
            for other_index, other in enumerate(body)
            if other_index not in combined
        ):
            continue
        marginal = candidate - selected
        marginal_writes = set().union(
            *(set(_statement_effects(body[i]).writes) for i in marginal),
            set(),
        )
        remaining_reads = _names_read(
            ast.Module(
                body=[
                    other
                    for other_index, other in enumerate(body)
                    if other_index not in combined
                ],
                type_ignores=[],
            )
        )
        if not repeated_work_is_profitable(
            (body[i] for i in sorted(marginal)),
            repeat_extent=lane_extent,
            live_scalars=max(1, len(marginal_writes & remaining_reads)),
        ):
            continue
        selected = combined

    if not selected:
        return [loop]
    store_roots, stores_resolved = _global_store_roots(loop, tensor_names)
    if not stores_resolved or not _loads_may_cross_loop_writes(
        (body[index] for index in sorted(selected)),
        store_roots,
        tensor_names,
        proven_disjoint_tensor_pairs,
    ):
        return [loop]
    hoisted = [body[index] for index in sorted(selected)]
    remainder = [stmt for index, stmt in enumerate(body) if index not in selected]
    if not remainder:
        return [loop]
    remaining_loop = _clone_lane_loop(loop, remainder)
    return [
        *hoisted,
        *_hoist_repeated_scalar_invariants(
            remaining_loop,
            lane_var,
            lane_extent,
            tensor_names,
            proven_disjoint_tensor_pairs,
            rename_groups,
        ),
    ]


def _try_unswitch_and_hoist(
    loop: ast.For,
    lane_var: str,
    lane_extent: int,
    tensor_names: set[str],
    tensor_dtypes: dict[str, str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    rename_groups: dict[str, str],
    uniform_names: set[str],
    float_scalar_names: set[str],
    rmem: dict[str, _RmemInfo],
    all_cache_access_counts: dict[str, int],
    non_subscript_access_names: set[str],
    obsolete_rmem: set[str],
    occupied_names: set[str],
) -> list[ast.stmt] | None:
    if not _has_only_known_effects(loop):
        return None
    body = list(loop.body)
    if not body or not isinstance(body[-1], ast.If):
        return None
    branch = body[-1]
    if not isinstance(branch.test, ast.Name):
        return None
    prefix = body[:-1]
    condition_names = _names_read(branch.test)
    selected: set[int] = set()
    needed = set(condition_names)
    for index in range(len(prefix) - 1, -1, -1):
        effects = _statement_effects(prefix[index])
        if not (set(effects.writes) & needed):
            continue
        selected.add(index)
        needed.difference_update(effects.writes)
        needed.update(effects.live_in)
    if lane_var in needed:
        return None
    if not all(_is_uniform_name(name, uniform_names, rename_groups) for name in needed):
        return None
    condition_slice = [prefix[index] for index in sorted(selected)]
    if not all(_is_hoistable_statement(stmt, tensor_names) for stmt in condition_slice):
        return None
    if not _slice_preserves_reaching_definitions(prefix, selected):
        return None
    if _has_rename_alias_conflict(prefix, selected, rename_groups):
        return None
    protected_condition_names = set(condition_names) | set(needed)
    for index in selected:
        effects = _statement_effects(prefix[index])
        protected_condition_names.update(effects.live_in)
        protected_condition_names.update(effects.writes)
    remaining_statements = [
        *(stmt for index, stmt in enumerate(prefix) if index not in selected),
        *branch.body,
        *branch.orelse,
    ]
    protected_condition_canonical = {
        _canonical_name(name, rename_groups) for name in protected_condition_names
    }
    if any(
        protected_condition_canonical
        & {
            _canonical_name(name, rename_groups)
            for name in (set(_statement_effects(stmt).writes) | _mutation_roots(stmt))
        }
        for stmt in remaining_statements
    ):
        return None
    # A guard may be unswitched only when its condition is uniform across the
    # thread block.  The generated prefix is self-contained, so direct CUDA
    # thread-coordinate use is a sufficient fail-closed rejection here.
    condition_text = "\n".join(
        [ast.unparse(branch.test), *(ast.unparse(stmt) for stmt in condition_slice)]
    )
    if any(token in condition_text for token in ("thread_idx", "lane_idx", "warp_idx")):
        return None
    store_roots, stores_resolved = _global_store_roots(loop, tensor_names)
    if not stores_resolved or not _loads_may_cross_loop_writes(
        condition_slice,
        store_roots,
        tensor_names,
        proven_disjoint_tensor_pairs,
    ):
        return None

    selected_set = set(selected)
    lane_prefix = [
        statement_from_string(ast.unparse(stmt))
        for index, stmt in enumerate(prefix)
        if index not in selected_set
    ]

    def make_branch(statements: list[ast.stmt]) -> ast.For:
        branch_body = _rewrite_inner_constexpr_loops(
            [
                *(statement_from_string(ast.unparse(stmt)) for stmt in lane_prefix),
                *(statement_from_string(ast.unparse(stmt)) for stmt in statements),
            ],
            tensor_names,
            proven_disjoint_tensor_pairs,
            rename_groups,
        )
        return _clone_lane_loop(
            loop,
            branch_body,
        )

    body_loop = make_branch(branch.body)
    else_loop = make_branch(branch.orelse)
    lifted = create(
        ast.If,
        test=expr_from_string(ast.unparse(branch.test)),
        body=[body_loop],
        orelse=[else_loop],
    )
    old_access_counts = {
        name: len(accesses)
        for name, accesses in _collect_subscript_accesses([loop]).items()
    }
    new_access_counts = {
        name: len(accesses)
        for name, accesses in _collect_subscript_accesses(
            [*condition_slice, lifted]
        ).items()
    }
    effective_access_counts = dict(all_cache_access_counts)
    for name in old_access_counts.keys() | new_access_counts.keys():
        effective_access_counts[name] = (
            effective_access_counts.get(name, 0)
            - old_access_counts.get(name, 0)
            + new_access_counts.get(name, 0)
        )

    def optimize_branch(branch_loop: ast.For) -> tuple[list[ast.stmt], bool]:
        vectorized = _hoist_lane_invariant_constexpr_chain(
            branch_loop,
            lane_extent=lane_extent,
            tensor_names=tensor_names,
            tensor_dtypes=tensor_dtypes,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
            rename_groups=rename_groups,
            rmem=rmem,
            all_cache_access_counts=effective_access_counts,
            non_subscript_access_names=non_subscript_access_names,
            obsolete_rmem=obsolete_rmem,
            float_scalar_names=float_scalar_names,
            occupied_names=occupied_names,
        )
        rewritten_loop = cast("ast.For", vectorized[-1])
        rewritten = [
            *vectorized[:-1],
            *_hoist_reductions_from_loop(
                rewritten_loop,
                lane_var,
                lane_extent,
                tensor_names,
                proven_disjoint_tensor_pairs,
                rename_groups,
            ),
        ]
        return rewritten, len(rewritten) > 1

    old_obsolete_rmem = set(obsolete_rmem)
    old_occupied_names = set(occupied_names)
    lifted.body, body_changed = optimize_branch(body_loop)
    lifted.orelse, else_changed = optimize_branch(else_loop)

    def profitable_path(rewritten: list[ast.stmt]) -> bool:
        moved = [*condition_slice, *rewritten[:-1]]
        moved_writes = set().union(
            *(set(_statement_effects(statement).writes) for statement in moved),
            set(),
        )
        remaining_reads = _names_read(rewritten[-1]) | _names_read(branch.test)
        return repeated_work_is_profitable(
            moved,
            repeat_extent=lane_extent,
            live_scalars=max(1, len(moved_writes & remaining_reads)),
            # Unswitching duplicates a branch and loop header in the static
            # kernel even though only one path executes.  Charge a small fixed
            # cost so a two-row loop is not rewritten for a trivial predicate.
            fixed_work=2,
        )

    changed_paths = [
        rewritten
        for rewritten, changed in (
            (lifted.body, body_changed),
            (lifted.orelse, else_changed),
        )
        if changed
    ]
    if changed_paths:
        profitable = all(profitable_path(rewritten) for rewritten in changed_paths)
    else:
        profitable = bool(condition_slice) and repeated_work_is_profitable(
            condition_slice,
            repeat_extent=lane_extent,
            live_scalars=1,
            fixed_work=2,
        )
    if not profitable:
        obsolete_rmem.clear()
        obsolete_rmem.update(old_obsolete_rmem)
        occupied_names.clear()
        occupied_names.update(old_occupied_names)
        return None
    return [*condition_slice, lifted]


def _rewrite_block(
    body: list[ast.stmt],
    tensor_names: set[str],
    tensor_dtypes: dict[str, str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    rename_groups: dict[str, str],
    uniform_names: set[str],
    float_scalar_names: set[str],
    thread_block_dims: tuple[int, int, int] | None,
    rmem: dict[str, _RmemInfo],
    all_cache_access_counts: dict[str, int],
    non_subscript_access_names: set[str],
    obsolete_rmem: set[str],
    occupied_names: set[str],
) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    known_uniform = set(uniform_names)
    known_rmem = dict(rmem)
    for stmt in body:
        lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
        extent = _static_lane_extent(stmt) if isinstance(stmt, ast.For) else None
        if (
            isinstance(stmt, ast.For)
            and isinstance(lane_var, str)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == lane_var
            and not stmt.orelse
            and extent is not None
            and extent > 1
        ):
            stmt = _simplify_full_lane_bounds(
                stmt,
                lane_var,
                extent,
                thread_block_dims,
            )
            replacement = _try_unswitch_and_hoist(
                stmt,
                lane_var,
                extent,
                tensor_names,
                tensor_dtypes,
                proven_disjoint_tensor_pairs,
                rename_groups,
                known_uniform,
                float_scalar_names,
                known_rmem,
                all_cache_access_counts,
                non_subscript_access_names,
                obsolete_rmem,
                occupied_names,
            )
            if replacement is not None:
                result.extend(replacement)
                for rewritten in replacement:
                    _update_uniform_names(rewritten, known_uniform, rename_groups)
                continue
            vectorized = _hoist_lane_invariant_constexpr_chain(
                stmt,
                lane_extent=extent,
                tensor_names=tensor_names,
                tensor_dtypes=tensor_dtypes,
                proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                rename_groups=rename_groups,
                rmem=known_rmem,
                all_cache_access_counts=all_cache_access_counts,
                non_subscript_access_names=non_subscript_access_names,
                obsolete_rmem=obsolete_rmem,
                float_scalar_names=float_scalar_names,
                occupied_names=occupied_names,
            )
            rewritten_loop = cast("ast.For", vectorized[-1])
            replacement = [
                *vectorized[:-1],
                *_hoist_reductions_from_loop(
                    rewritten_loop,
                    lane_var,
                    extent,
                    tensor_names,
                    proven_disjoint_tensor_pairs,
                    rename_groups,
                ),
            ]
            if len(replacement) > 1:
                result.extend(replacement)
                for rewritten in replacement:
                    _update_uniform_names(rewritten, known_uniform, rename_groups)
                continue
        for field in ("body", "orelse", "finalbody"):
            child = getattr(stmt, field, None)
            if isinstance(child, list) and all(
                isinstance(item, ast.stmt) for item in child
            ):
                setattr(
                    stmt,
                    field,
                    _rewrite_block(
                        child,
                        tensor_names,
                        tensor_dtypes,
                        proven_disjoint_tensor_pairs,
                        rename_groups,
                        set(known_uniform),
                        float_scalar_names,
                        thread_block_dims,
                        dict(known_rmem),
                        all_cache_access_counts,
                        non_subscript_access_names,
                        obsolete_rmem,
                        occupied_names,
                    ),
                )
        result.append(stmt)
        _update_uniform_names(stmt, known_uniform, rename_groups)
        declaration = _rmem_declaration(stmt)
        written = set(_statement_effects(stmt).writes)
        for name in written:
            known_rmem.pop(name, None)
        if declaration is not None:
            known_rmem[declaration[0]] = declaration[1]
    return result


def _prune_obsolete_rmem_declarations(
    body: list[ast.stmt], obsolete_rmem: set[str]
) -> list[ast.stmt]:
    loaded = {
        node.id
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    result: list[ast.stmt] = []
    for stmt in body:
        declaration = _rmem_declaration(stmt)
        if (
            declaration is not None
            and declaration[0] in obsolete_rmem
            and declaration[0] not in loaded
        ):
            continue
        for field in ("body", "orelse", "finalbody"):
            child = getattr(stmt, field, None)
            if isinstance(child, list) and all(
                isinstance(item, ast.stmt) for item in child
            ):
                setattr(
                    stmt,
                    field,
                    _prune_obsolete_rmem_declarations(child, obsolete_rmem),
                )
        result.append(stmt)
    return result


def hoist_lane_invariant_reductions(
    body: list[ast.stmt],
    *,
    tensor_names: set[str],
    tensor_dtypes: dict[str, str] | None = None,
    proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
    rename_groups: dict[str, str] | None = None,
    uniform_names: set[str] | None = None,
    float_scalar_names: set[str] | None = None,
    thread_block_dims: tuple[int, int, int] | None = None,
) -> list[ast.stmt]:
    """Hoist profitable row-invariant reductions from grid lane loops."""
    renames = rename_groups or {}
    initial_uniform = {
        _canonical_name(name, renames)
        for name in {*_UNIFORM_GLOBAL_NAMES, *tensor_names, *(uniform_names or set())}
    }
    float_scalars = {
        _canonical_name(name, renames) for name in (float_scalar_names or set())
    }
    all_cache_access_counts = {
        name: len(accesses)
        for name, accesses in _collect_subscript_accesses(body).items()
    }
    non_subscript_access_names = _non_subscript_access_names(body)
    occupied_names = {
        node.id
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
    }
    occupied_names.update(renames)
    occupied_names.update(renames.values())
    occupied_names.update(_canonical_name(name, renames) for name in renames)
    obsolete_rmem: set[str] = set()
    result = _rewrite_block(
        body,
        tensor_names,
        tensor_dtypes or {},
        proven_disjoint_tensor_pairs or set(),
        renames,
        initial_uniform,
        float_scalars,
        thread_block_dims,
        {},
        all_cache_access_counts,
        non_subscript_access_names,
        obsolete_rmem,
        occupied_names,
    )
    result = _prune_obsolete_rmem_declarations(result, obsolete_rmem)
    return [ast.fix_missing_locations(stmt) for stmt in result]
