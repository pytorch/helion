"""Pipeline a narrow in-place vector state-update pattern through shared memory.

This pass is intentionally stricter than a general load scheduler.  It recognizes
one compiler-generated runtime row loop in which every thread loads one 16-byte
vector from a tensor and later stores a 16-byte vector to the exact same address.
The row and column coordinates must form a provably disjoint two-dimensional
thread/lane partition, and the mutated tensor must not alias another argument.

Eligible loads are staged with ``cp.async`` in fixed-size row groups.  Each thread
owns private shared-memory slots, so no CTA barrier is needed.  The pass otherwise
leaves the loop body byte-for-byte intact; in particular it does not perform CSE
or delete arithmetic.  When the same accepted row loop also has one exact,
read-only BF16 scalar stream contiguous in the row coordinate, adjacent elements
may be staged in Uint32 register pairs before the first async wait.  Masked streams
retain their original scalar expression as the partial-pair fallback.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import replace
import textwrap
from typing import NamedTuple

import torch

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from .cutedsl_compat import L2_EVICT_LAST_STORE_ABI_VERSION
from .cutedsl_compat import cp_async_supported
from .cutedsl_compat import fixed_l2_evict_last_store_policy_supported


class TensorMetadata(NamedTuple):
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _Pointer:
    tensor_name: str
    expression: ast.expr
    # Exact offset added to ``tensor.iterator``.  Keeping the original
    # conditional offset preserves the generated masked-load fallback.
    offset: ast.expr
    active_offset: ast.expr
    predicate: ast.expr | None


@dataclass(frozen=True)
class _PackedScalarLoad:
    tensor_name: str
    target_name: str
    statements: tuple[ast.Assign, ...]
    pointer: _Pointer
    predicate: ast.expr | None
    original_values: tuple[ast.expr, ...]


@dataclass(frozen=True)
class _Region:
    owner: list[ast.stmt]
    loop_index: int
    loop: ast.For
    loop_var: str
    trip_count: int
    row_name: str
    row_rhs: ast.expr
    row_axis: int
    column_axis: int
    load_index: int
    load_name: str
    load_pointer: _Pointer
    vector_type: ast.expr
    vector_width: int
    tensor_name: str
    store_top_level: ast.stmt
    insertion_index: int
    allowed_source_nodes: frozenset[int]
    packed_scalar_load: _PackedScalarLoad | None


def _call_path(node: ast.AST) -> tuple[str, ...] | None:
    path: list[str] = []
    while isinstance(node, ast.Attribute):
        path.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    path.append(node.id)
    return tuple(reversed(path))


def _is_cutlass_int_cast(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_path(node.func) in {
        ("cutlass", "Int32"),
        ("cutlass", "Int64"),
        ("cutlass", "Uint32"),
        ("cutlass", "Uint64"),
    }


def _unwrap_int_cast(node: ast.expr) -> ast.expr:
    while (
        isinstance(node, ast.Call)
        and _is_cutlass_int_cast(node)
        and len(node.args) == 1
        and not node.keywords
    ):
        inner = node.args[0]
        if not isinstance(inner, ast.expr):
            break
        node = inner
    return node


def _const_int(node: ast.expr) -> int | None:
    node = _unwrap_int_cast(node)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def _thread_axis(node: ast.expr) -> int | None:
    node = _unwrap_int_cast(node)
    if not isinstance(node, ast.Subscript):
        return None
    if not (
        isinstance(node.value, ast.Call)
        and _call_path(node.value.func) == ("cute", "arch", "thread_idx")
        and not node.value.args
        and not node.value.keywords
    ):
        return None
    return _const_int(node.slice)


def _block_axis(node: ast.expr) -> int | None:
    node = _unwrap_int_cast(node)
    if not isinstance(node, ast.Subscript):
        return None
    if not (
        isinstance(node.value, ast.Call)
        and _call_path(node.value.func) == ("cute", "arch", "block_idx")
        and not node.value.args
        and not node.value.keywords
    ):
        return None
    return _const_int(node.slice)


def _has_thread_or_lane(node: ast.AST, lane_var: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == lane_var:
            return True
        if isinstance(child, ast.Subscript) and _thread_axis(child) is not None:
            return True
    return False


def _linear_coefficients(
    node: ast.expr,
    lane_var: str,
    definitions: dict[str, ast.expr],
    visiting: frozenset[str] = frozenset(),
) -> tuple[int, dict[int, int]] | None:
    """Return coefficients of ``lane_var`` and ``thread_idx()[axis]``.

    Expressions independent of both are treated as an opaque affine base.  Only
    addition/subtraction, multiplication by an integer literal, and generated
    integer casts are admitted around the varying terms.
    """
    node = _unwrap_int_cast(node)
    if isinstance(node, ast.Name):
        if node.id == lane_var:
            return 1, {}
        if node.id in definitions and node.id not in visiting:
            return _linear_coefficients(
                definitions[node.id], lane_var, definitions, visiting | {node.id}
            )
        return 0, {}
    axis = _thread_axis(node)
    if axis is not None:
        return 0, {axis: 1}
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _linear_coefficients(node.left, lane_var, definitions, visiting)
        right = _linear_coefficients(node.right, lane_var, definitions, visiting)
        if left is None or right is None:
            return None
        sign = -1 if isinstance(node.op, ast.Sub) else 1
        axes = dict(left[1])
        for key, value in right[1].items():
            axes[key] = axes.get(key, 0) + sign * value
        return left[0] + sign * right[0], {k: v for k, v in axes.items() if v}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_int = _const_int(node.left)
        right_int = _const_int(node.right)
        if left_int is not None:
            value = _linear_coefficients(node.right, lane_var, definitions, visiting)
            factor = left_int
        elif right_int is not None:
            value = _linear_coefficients(node.left, lane_var, definitions, visiting)
            factor = right_int
        else:
            return None if _has_thread_or_lane(node, lane_var) else (0, {})
        if value is None:
            return None
        return value[0] * factor, {key: val * factor for key, val in value[1].items()}
    if _has_thread_or_lane(node, lane_var):
        return None
    return 0, {}


def _captured_definitions(
    statements: list[ast.stmt], write_counts: dict[str, int]
) -> dict[str, ast.expr]:
    """Model straight-line scalar assignment values at their assignment time."""
    definitions: dict[str, ast.expr] = {}
    for stmt in statements:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            value = _expand_aliases(stmt.value, definitions)
            name = stmt.targets[0].id
            has_unresolved_rebound_dependency = value is not None and any(
                write_counts.get(node.id, 0)
                for node in ast.walk(value)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in {"cutlass", "cute", "ir"}
            )
            if value is None or has_unresolved_rebound_dependency:
                definitions.pop(name, None)
            else:
                definitions[name] = value
            continue
        for name in _assigned_names(stmt):
            definitions.pop(name, None)
    return definitions


class _Substitute(ast.NodeTransformer):
    def __init__(self, replacements: dict[str, ast.expr]) -> None:
        self.replacements = replacements

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.replacements:
            return ast.copy_location(
                expr_from_string(ast.unparse(self.replacements[node.id])), node
            )
        return node


def _substitute(node: ast.expr, replacements: dict[str, ast.expr]) -> ast.expr:
    copied = expr_from_string(ast.unparse(node))
    result = _Substitute(replacements).visit(copied)
    assert isinstance(result, ast.expr)
    return ast.fix_missing_locations(result)


def _flatten_add(node: ast.expr) -> list[ast.expr]:
    node = _unwrap_int_cast(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_flatten_add(node.left), *_flatten_add(node.right)]
    return [node]


def _stride_dimension(node: ast.expr, tensor_name: str) -> int | None:
    node = _unwrap_int_cast(node)
    if not isinstance(node, ast.Subscript):
        return None
    if _call_path(node.value) != (tensor_name, "layout", "stride"):
        return None
    return _const_int(node.slice)


def _indices_by_stride(
    offset: ast.expr, tensor_name: str, rank: int
) -> dict[int, ast.expr] | None:
    """Split a generated flat offset into one exact term per tensor dim."""
    result: dict[int, ast.expr] = {}
    for term in _flatten_add(offset):
        term = _unwrap_int_cast(term)
        if not isinstance(term, ast.BinOp) or not isinstance(term.op, ast.Mult):
            return None
        left_dim = _stride_dimension(term.left, tensor_name)
        right_dim = _stride_dimension(term.right, tensor_name)
        if (left_dim is None) == (right_dim is None):
            return None
        dim = right_dim if left_dim is None else left_dim
        index = term.left if left_dim is None else term.right
        assert dim is not None
        if not 0 <= dim < rank or dim in result:
            return None
        result[dim] = index
    return result if set(result) == set(range(rank)) else None


def _vector_type_info(node: ast.expr) -> tuple[int, str] | None:
    if not (
        isinstance(node, ast.Call)
        and _call_path(node.func) == ("ir", "VectorType", "get")
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[0], ast.List)
        and len(node.args[0].elts) == 1
    ):
        return None
    width = _const_int(node.args[0].elts[0])
    element = _call_path(node.args[1])
    if width is None or element is None or element[-1] != "mlir_type":
        return None
    return width, ".".join(element[:-1])


def _pointer_info(node: ast.expr) -> _Pointer | None:
    terms = _flatten_add(node)
    if len(terms) < 2:
        return None
    base = terms[0]
    if not (
        isinstance(base, ast.Attribute)
        and base.attr == "iterator"
        and isinstance(base.value, ast.Name)
    ):
        return None
    tensor_name = base.value.id
    offset = terms[1]
    for term in terms[2:]:
        offset = ast.BinOp(left=offset, op=ast.Add(), right=term)
    if isinstance(offset, ast.IfExp):
        if _const_int(offset.orelse) != 0:
            return None
        return _Pointer(tensor_name, node, offset, offset.body, offset.test)
    return _Pointer(tensor_name, node, offset, offset, None)


def _load_info(stmt: ast.stmt) -> tuple[str, _Pointer, ast.expr, int] | None:
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Call)
        and _call_path(stmt.value.func) == ("cute", "arch", "load")
        and len(stmt.value.args) == 2
        and not stmt.value.keywords
    ):
        return None
    pointer = _pointer_info(stmt.value.args[0])
    vector = _vector_type_info(stmt.value.args[1])
    if pointer is None or vector is None:
        return None
    width, element_type = vector
    if element_type != "cutlass.Uint16" or width * 2 != 16:
        return None
    return stmt.targets[0].id, pointer, stmt.value.args[1], width


def _name_write_counts(node: ast.AST) -> dict[str, int]:
    writes: dict[str, int] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            writes[child.id] = writes.get(child.id, 0) + 1
    return writes


def _immutable_trivial_aliases(
    statements: list[ast.stmt], write_counts: dict[str, int]
) -> dict[str, ast.expr]:
    """Collect only earlier scalar aliases whose bindings never change.

    The order check matters: replacing ``snapshot = selector`` with
    ``selector`` is invalid if ``selector`` is assigned again before the use.
    Requiring one write in the whole kernel also makes every transitive alias
    target immutable for the lifetime of the matched region.
    """
    aliases: dict[str, ast.expr] = {}
    definitions: dict[str, ast.expr] = {}

    def immutable_dependencies(name: str, visiting: frozenset[str]) -> bool:
        count = write_counts.get(name, 0)
        if count == 0:
            return True
        if count != 1 or name in visiting or name not in definitions:
            return False
        return all(
            immutable_dependencies(dependency.id, visiting | {name})
            for dependency in ast.walk(definitions[name])
            if isinstance(dependency, ast.Name) and isinstance(dependency.ctx, ast.Load)
        )

    for stmt in statements:
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            continue
        name = stmt.targets[0].id
        value = _unwrap_int_cast(stmt.value)
        if (
            write_counts.get(name) == 1
            and isinstance(value, (ast.Name, ast.Constant))
            and (
                isinstance(value, ast.Constant)
                or immutable_dependencies(value.id, frozenset())
            )
        ):
            aliases[name] = value
        if write_counts.get(name) == 1:
            definitions[name] = stmt.value
    return aliases


def _immutable_scalar_definitions(
    statements: list[ast.stmt], write_counts: dict[str, int]
) -> dict[str, ast.expr]:
    return {
        stmt.targets[0].id: stmt.value
        for stmt in statements
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and write_counts.get(stmt.targets[0].id) == 1
    }


def _scalar_tensor_load(stmt: ast.stmt) -> tuple[str, _Pointer] | None:
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "load"
        and not stmt.value.args
        and not stmt.value.keywords
    ):
        return None
    pointer = _pointer_info(stmt.value.func.value)
    if pointer is None or pointer.predicate is not None:
        return None
    return stmt.targets[0].id, pointer


def _scalar_bf16_tensor_load(
    stmt: ast.stmt,
) -> tuple[str, _Pointer, ast.expr | None, ast.expr] | None:
    """Match a direct BF16 scalar load with an optional zero fallback.

    The optional predicate remains part of the returned original expression;
    callers may move the load only when they preserve that exact fallback.
    """
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return None
    value = stmt.value
    predicate: ast.expr | None = None
    load_value = value
    if isinstance(value, ast.IfExp):
        if not (
            isinstance(value.orelse, ast.Call)
            and _call_path(value.orelse.func) == ("cutlass", "BFloat16")
            and len(value.orelse.args) == 1
            and not value.orelse.keywords
            and _const_int(value.orelse.args[0]) == 0
        ):
            return None
        predicate = value.test
        load_value = value.body
    if not (
        isinstance(load_value, ast.Call)
        and isinstance(load_value.func, ast.Attribute)
        and load_value.func.attr == "load"
        and not load_value.args
        and not load_value.keywords
    ):
        return None
    pointer = _pointer_info(load_value.func.value)
    if pointer is None or pointer.predicate is not None:
        return None
    return stmt.targets[0].id, pointer, predicate, value


def _index_expr_is_cta_uniform(
    node: ast.expr,
    *,
    stride_source: str | None,
    definitions: dict[str, ast.expr],
    uniform_names: frozenset[str],
    uniform_scalar_load_names: frozenset[str],
    write_counts: dict[str, int],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    visiting: frozenset[str] = frozenset(),
) -> bool:
    """Prove an integer expression is identical for every CTA thread.

    Unknown calls and names fail closed.  The only admitted coordinates are
    block indices, immutable launch scalars, and explicitly proven uniform
    scalar loads.  Tensor strides are admitted only while proving an address
    for ``stride_source`` and only when their exact runtime value specializes
    the compiled kernel.
    """
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.Name):
        if node.id in visiting:
            return False
        if node.id in uniform_scalar_load_names and write_counts.get(node.id) == 1:
            return True
        if node.id in definitions:
            if write_counts.get(node.id) != 1:
                return False
            return _index_expr_is_cta_uniform(
                definitions[node.id],
                stride_source=stride_source,
                definitions=definitions,
                uniform_names=uniform_names,
                uniform_scalar_load_names=uniform_scalar_load_names,
                write_counts=write_counts,
                proven_tensor_stride_values=proven_tensor_stride_values,
                visiting=visiting | {node.id},
            )
        return node.id in uniform_names and write_counts.get(node.id, 0) == 0
    if isinstance(node, ast.Subscript):
        block_axis = _block_axis(node)
        if block_axis is not None:
            return 0 <= block_axis < 3
        if stride_source is None:
            return False
        stride_dim = _stride_dimension(node, stride_source)
        return (
            stride_dim is not None
            and (stride_source, stride_dim) in proven_tensor_stride_values
        )
    if isinstance(node, ast.Call):
        if not _is_cutlass_int_cast(node) or len(node.args) != 1 or node.keywords:
            return False
        argument = node.args[0]
        return isinstance(argument, ast.expr) and _index_expr_is_cta_uniform(
            argument,
            stride_source=stride_source,
            definitions=definitions,
            uniform_names=uniform_names,
            uniform_scalar_load_names=uniform_scalar_load_names,
            write_counts=write_counts,
            proven_tensor_stride_values=proven_tensor_stride_values,
            visiting=visiting,
        )
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ):
        return _index_expr_is_cta_uniform(
            node.left,
            stride_source=stride_source,
            definitions=definitions,
            uniform_names=uniform_names,
            uniform_scalar_load_names=uniform_scalar_load_names,
            write_counts=write_counts,
            proven_tensor_stride_values=proven_tensor_stride_values,
            visiting=visiting,
        ) and _index_expr_is_cta_uniform(
            node.right,
            stride_source=stride_source,
            definitions=definitions,
            uniform_names=uniform_names,
            uniform_scalar_load_names=uniform_scalar_load_names,
            write_counts=write_counts,
            proven_tensor_stride_values=proven_tensor_stride_values,
            visiting=visiting,
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _index_expr_is_cta_uniform(
            node.operand,
            stride_source=stride_source,
            definitions=definitions,
            uniform_names=uniform_names,
            uniform_scalar_load_names=uniform_scalar_load_names,
            write_counts=write_counts,
            proven_tensor_stride_values=proven_tensor_stride_values,
            visiting=visiting,
        )
    return False


def _proven_uniform_scalar_load_names(
    statements: list[ast.stmt],
    *,
    root_body: list[ast.stmt],
    state_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    uniform_names: frozenset[str],
    write_counts: dict[str, int],
) -> frozenset[str]:
    """Find immutable CTA-uniform scalar loads with a non-mutable source.

    The tensor source may occur only in this one direct scalar load, and must
    be runtime-specialized disjoint from every other tensor argument. This
    rules out both local aliases and mutation through another argument while
    future state rows are in flight.
    """
    result: set[str] = set()
    for index, stmt in enumerate(statements):
        loaded = _scalar_tensor_load(stmt)
        if loaded is None:
            continue
        target, pointer = loaded
        source = pointer.tensor_name
        if (
            write_counts.get(target) != 1
            or source == state_tensor
            or source not in tensor_names
            or any(
                frozenset((source, other)) not in proven_disjoint_tensor_pairs
                for other in tensor_names
                if other != source
            )
        ):
            continue
        assert isinstance(stmt, ast.Assign)
        allowed_source_nodes = frozenset(
            id(node)
            for node in ast.walk(stmt.value)
            if isinstance(node, ast.Name) and node.id == source
        )
        if not _only_allowed_source_uses(
            root_body, source, allowed_source_nodes
        ) or not _index_expr_is_cta_uniform(
            pointer.active_offset,
            stride_source=source,
            definitions=_immutable_scalar_definitions(statements[:index], write_counts),
            uniform_names=uniform_names,
            uniform_scalar_load_names=frozenset(),
            write_counts=write_counts,
            proven_tensor_stride_values=proven_tensor_stride_values,
        ):
            continue
        result.add(target)
    return frozenset(result)


def _expand_aliases(node: ast.expr, aliases: dict[str, ast.expr]) -> ast.expr | None:
    result = node
    for _ in range(len(aliases) + 1):
        updated = _substitute(result, aliases)
        if ast.dump(updated, include_attributes=False) == ast.dump(
            result, include_attributes=False
        ):
            return updated
        result = updated
    return None


def _store_pointer(
    loop: ast.For, tensor_name: str, vector_width: int
) -> tuple[_Pointer, ast.expr | None, ast.stmt] | None:
    """Match the exact generated vector-store tail.

    Requiring ``list init; constexpr append loop; vector store`` as the final
    three statements keeps the proof tied to the codegen protocol rather than
    attempting to infer arbitrary container mutation semantics.
    """
    if len(loop.body) < 3:
        return None
    init, append_loop, top_level = loop.body[-3:]
    if not (
        isinstance(init, ast.Assign)
        and len(init.targets) == 1
        and isinstance(init.targets[0], ast.Name)
        and isinstance(init.value, ast.List)
        and not init.value.elts
        and isinstance(append_loop, ast.For)
        and isinstance(append_loop.target, ast.Name)
        and isinstance(append_loop.iter, ast.Call)
        and _call_path(append_loop.iter.func) == ("cutlass", "range_constexpr")
        and len(append_loop.iter.args) == 1
        and not append_loop.iter.keywords
        and _const_int(append_loop.iter.args[0]) == vector_width
        and not append_loop.orelse
        and append_loop.body
    ):
        return None
    list_name = init.targets[0].id
    append_stmt = append_loop.body[-1]
    if not (
        isinstance(append_stmt, ast.Expr)
        and isinstance(append_stmt.value, ast.Call)
        and isinstance(append_stmt.value.func, ast.Attribute)
        and append_stmt.value.func.attr == "append"
        and isinstance(append_stmt.value.func.value, ast.Name)
        and append_stmt.value.func.value.id == list_name
        and len(append_stmt.value.args) == 1
        and not append_stmt.value.keywords
    ):
        return None
    append_calls = [
        node
        for node in ast.walk(append_loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == list_name
    ]
    if append_calls != [append_stmt.value]:
        return None

    predicate: ast.expr | None = None
    store_stmt: ast.stmt = top_level
    if isinstance(top_level, ast.If):
        if not (
            len(top_level.body) == 1
            and (
                not top_level.orelse
                or len(top_level.orelse) == 1
                and isinstance(top_level.orelse[0], ast.Pass)
            )
        ):
            return None
        predicate = top_level.test
        store_stmt = top_level.body[0]
    if not (
        isinstance(store_stmt, ast.Expr)
        and isinstance(store_stmt.value, ast.Call)
        and isinstance(store_stmt.value.func, ast.Name)
        and store_stmt.value.func.id == "_cute_store_u16_vec"
        and len(store_stmt.value.args) == 2
        and not store_stmt.value.keywords
        and isinstance(store_stmt.value.args[1], ast.Name)
        and store_stmt.value.args[1].id == list_name
    ):
        return None
    pointer = _pointer_info(store_stmt.value.args[0])
    if (
        pointer is None
        or pointer.tensor_name != tensor_name
        or pointer.predicate is not None
    ):
        return None
    list_uses = [
        node
        for node in ast.walk(ast.Module(body=loop.body, type_ignores=[]))
        if isinstance(node, ast.Name) and node.id == list_name
    ]
    allowed_list_uses = {
        id(init.targets[0]),
        id(append_stmt.value.func.value),
        id(store_stmt.value.args[1]),
    }
    if {id(node) for node in list_uses} != allowed_list_uses:
        return None
    return pointer, predicate, top_level


def _assigned_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del))
    }


_PURE_CUTLASS_SCALAR_CALLS = frozenset(
    (
        "BFloat16",
        "Boolean",
        "Float16",
        "Float32",
        "Float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Uint8",
        "Uint16",
        "Uint32",
        "Uint64",
    )
)
_PURE_CUTE_ARCH_CALLS = frozenset(
    (
        "block_idx",
        "lane_idx",
        "thread_idx",
        "warp_idx",
        "warp_reduction_max",
        "warp_reduction_min",
        "warp_reduction_sum",
    )
)
_PURE_CUTE_MATH_CALLS = frozenset(
    (
        "absf",
        "div",
        "erf",
        "exp",
        "exp2",
        "fma",
        "log",
        "log2",
        "max",
        "min",
        "rcp",
        "rsqrt",
        "sqrt",
        "tanh",
    )
)
_PURE_OPERATOR_CALLS = frozenset(
    (
        "add",
        "and_",
        "eq",
        "floordiv",
        "ge",
        "gt",
        "le",
        "lshift",
        "lt",
        "mod",
        "mul",
        "ne",
        "neg",
        "not_",
        "or_",
        "pos",
        "pow",
        "rshift",
        "sub",
        "truediv",
        "xor",
    )
)


def _is_cutlass_numeric_type(node: ast.AST) -> bool:
    path = _call_path(node)
    return (
        path is not None
        and len(path) == 2
        and path[0] == "cutlass"
        and path[1] in _PURE_CUTLASS_SCALAR_CALLS
    )


def _is_disjoint_tensor_read(
    call: ast.Call,
    *,
    state_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> bool:
    """Prove that ``call`` is a generated read from another tensor argument."""
    path = _call_path(call.func)
    if path == ("cute", "arch", "load"):
        if len(call.args) != 2 or call.keywords:
            return False
        pointer = _pointer_info(call.args[0])
    elif (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "load"
        and not call.args
        and not call.keywords
    ):
        pointer = _pointer_info(call.func.value)
    else:
        return False
    if pointer is None:
        return False
    source = pointer.tensor_name
    return (
        source in tensor_names
        and source != state_tensor
        and frozenset((state_tensor, source)) in proven_disjoint_tensor_pairs
    )


def _prefetch_overlap_call_is_safe(
    call: ast.Call,
    *,
    state_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> bool:
    path = _call_path(call.func)
    if path is not None:
        if len(path) == 2 and path[0] == "cutlass":
            return path[1] in _PURE_CUTLASS_SCALAR_CALLS
        if len(path) == 3 and path[:2] == ("cute", "math"):
            return path[2] in _PURE_CUTE_MATH_CALLS
        if path == ("ir", "VectorType", "get"):
            return True
        if len(path) == 3 and path[:2] == ("cute", "arch"):
            if path[2] in _PURE_CUTE_ARCH_CALLS:
                return True
            if path[2] == "load":
                return _is_disjoint_tensor_read(
                    call,
                    state_tensor=state_tensor,
                    tensor_names=tensor_names,
                    proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                )
        if len(path) == 2 and path[0] == "operator":
            return path[1] in _PURE_OPERATOR_CALLS
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "bitcast"
        and len(call.args) == 1
        and not call.keywords
        and _is_cutlass_numeric_type(call.args[0])
    ):
        receiver = call.func.value
        return isinstance(receiver, ast.Call) and (
            (
                (receiver_path := _call_path(receiver.func)) is not None
                and len(receiver_path) == 2
                and receiver_path[0] == "cutlass"
                and receiver_path[1] in _PURE_CUTLASS_SCALAR_CALLS
            )
            or _is_disjoint_tensor_read(
                receiver,
                state_tensor=state_tensor,
                tensor_names=tensor_names,
                proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
            )
        )
    return _is_disjoint_tensor_read(
        call,
        state_tensor=state_tensor,
        tensor_names=tensor_names,
        proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
    )


def _prefetch_overlap_expr_is_safe(
    expression: ast.expr,
    *,
    state_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> bool:
    """Accept only compiler-emitted register math and disjoint tensor reads."""
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
    return all(
        _prefetch_overlap_call_is_safe(
            node,
            state_tensor=state_tensor,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
        )
        for node in ast.walk(expression)
        if isinstance(node, ast.Call)
    )


def _private_register_tensor_names(
    statements: list[ast.stmt], write_counts: dict[str, int]
) -> frozenset[str]:
    """Return directly bound register fragments on one lexical path."""
    return frozenset(
        stmt.targets[0].id
        for stmt in statements
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and write_counts.get(stmt.targets[0].id) == 1
        and isinstance(stmt.value, ast.Call)
        and _call_path(stmt.value.func) == ("cute", "make_rmem_tensor")
    )


def _prefetch_overlap_target_is_safe(
    target: ast.expr,
    *,
    private_register_tensors: set[str],
    state_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> bool:
    if isinstance(target, ast.Name):
        return True
    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id in private_register_tensors
        and _prefetch_overlap_expr_is_safe(
            target.slice,
            state_tensor=state_tensor,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
        )
    )


def _prefetch_overlap_statement_is_safe(
    stmt: ast.stmt,
    *,
    private_register_tensors: set[str],
    state_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> bool:
    """Whether an async read may move before one generated statement.

    Control flow and opaque calls deliberately form barriers.  The one admitted
    compound statement is a fixed ``range_constexpr`` register loop, which is
    how generated vector unpack/reduction preparation is represented.
    """
    if isinstance(stmt, ast.Assign):
        return all(
            _prefetch_overlap_target_is_safe(
                target,
                private_register_tensors=private_register_tensors,
                state_tensor=state_tensor,
                tensor_names=tensor_names,
                proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
            )
            for target in stmt.targets
        ) and _prefetch_overlap_expr_is_safe(
            stmt.value,
            state_tensor=state_tensor,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
        )
    if isinstance(stmt, ast.For):
        return (
            isinstance(stmt.target, ast.Name)
            and isinstance(stmt.iter, ast.Call)
            and _call_path(stmt.iter.func) == ("cutlass", "range_constexpr")
            and len(stmt.iter.args) == 1
            and not stmt.iter.keywords
            and _const_int(stmt.iter.args[0]) is not None
            and not stmt.orelse
            and all(
                _prefetch_overlap_statement_is_safe(
                    child,
                    private_register_tensors=private_register_tensors,
                    state_tensor=state_tensor,
                    tensor_names=tensor_names,
                    proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                )
                for child in stmt.body
            )
        )
    return isinstance(stmt, ast.Pass)


def _is_disjoint_tensor_store(
    call: ast.Call,
    *,
    source_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> tuple[ast.expr, ...] | None:
    """Return evaluated expressions for one proven-disjoint direct store."""
    pointer: _Pointer | None = None
    values: tuple[ast.expr, ...] = ()
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "store"
        and len(call.args) == 1
        and not call.keywords
    ):
        pointer = _pointer_info(call.func.value)
        values = (call.args[0],)
    elif (
        isinstance(call.func, ast.Name)
        and call.func.id == "_cute_store_u16_vec"
        and len(call.args) == 2
        and not call.keywords
    ):
        pointer = _pointer_info(call.args[0])
        values = (call.args[1],)
    if pointer is None:
        return None
    target = pointer.tensor_name
    if (
        target not in tensor_names
        or target == source_tensor
        or frozenset((source_tensor, target)) not in proven_disjoint_tensor_pairs
    ):
        return None
    return (pointer.expression, *values)


def _scalar_preload_target_is_local(
    target: ast.expr,
    *,
    local_containers: frozenset[str],
    source_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    reaching_definitions: dict[str, ast.expr],
    constexpr_values: dict[str, int],
    loop_assigned_names: frozenset[str],
) -> bool:
    if isinstance(target, ast.Name):
        return True
    if isinstance(target, (ast.List, ast.Tuple)):
        return all(
            _scalar_preload_target_is_local(
                element,
                local_containers=local_containers,
                source_tensor=source_tensor,
                tensor_names=tensor_names,
                proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                reaching_definitions=reaching_definitions,
                constexpr_values=constexpr_values,
                loop_assigned_names=loop_assigned_names,
            )
            for element in target.elts
        )
    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id in local_containers
        and _scalar_preload_expr_is_safe(
            target.slice,
            source_tensor=source_tensor,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
            reaching_definitions=reaching_definitions,
            constexpr_values=constexpr_values,
            loop_assigned_names=loop_assigned_names,
        )
    )


def _scalar_preload_expr_is_safe(
    expression: ast.expr,
    *,
    source_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    reaching_definitions: dict[str, ast.expr],
    constexpr_values: dict[str, int],
    loop_assigned_names: frozenset[str],
) -> bool:
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
    extra_pure_calls = {
        ("cute", "arch", "fma_packed_f32x2"),
        ("cute", "arch", "mul_packed_f32x2"),
        ("cute", "arch", "shuffle_sync"),
    }
    if any(
        _call_path(call.func) not in extra_pure_calls
        and not _prefetch_overlap_call_is_safe(
            call,
            state_tensor=source_tensor,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
        )
        for call in ast.walk(expression)
        if isinstance(call, ast.Call)
    ):
        return False
    replacements: dict[str, ast.expr] = {
        name: ast.Constant(value=value) for name, value in constexpr_values.items()
    }
    for node in ast.walk(expression):
        if isinstance(node, ast.BinOp) and not isinstance(
            node.op,
            (
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.FloorDiv,
                ast.Mod,
                ast.LShift,
                ast.RShift,
                ast.BitAnd,
                ast.BitOr,
                ast.BitXor,
            ),
        ):
            return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            if any(
                isinstance(child, ast.Name) and child.id in loop_assigned_names
                for child in ast.walk(node.right)
            ):
                return False
            denominator = _expand_aliases(
                _substitute(node.right, replacements), reaching_definitions
            )
            if denominator is None or not _constant_int_expr(denominator):
                return False
        if isinstance(node, ast.Call) and _call_path(node.func) in {
            ("cute", "math", "div"),
            ("operator", "floordiv"),
            ("operator", "mod"),
            ("operator", "truediv"),
        }:
            return False
    return True


def _scalar_preload_statement_is_safe(
    stmt: ast.stmt,
    *,
    matched_statements: frozenset[int],
    local_containers: frozenset[str],
    source_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    reaching_definitions: dict[str, ast.expr],
    constexpr_values: dict[str, int],
    loop_assigned_names: frozenset[str],
) -> bool:
    """Prove one statement can be crossed by every hoisted scalar read."""
    if id(stmt) in matched_statements:
        return True

    def expression_is_safe(expression: ast.expr) -> bool:
        return _scalar_preload_expr_is_safe(
            expression,
            source_tensor=source_tensor,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
            reaching_definitions=reaching_definitions,
            constexpr_values=constexpr_values,
            loop_assigned_names=loop_assigned_names,
        )

    if isinstance(stmt, ast.Assign):
        return all(
            _scalar_preload_target_is_local(
                target,
                local_containers=local_containers,
                source_tensor=source_tensor,
                tensor_names=tensor_names,
                proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                reaching_definitions=reaching_definitions,
                constexpr_values=constexpr_values,
                loop_assigned_names=loop_assigned_names,
            )
            for target in stmt.targets
        ) and expression_is_safe(stmt.value)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "append"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in local_containers
            and len(call.args) == 1
            and not call.keywords
        ):
            return expression_is_safe(call.args[0])
        stored_values = _is_disjoint_tensor_store(
            call,
            source_tensor=source_tensor,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
        )
        return stored_values is not None and all(
            expression_is_safe(value) for value in stored_values
        )
    if isinstance(stmt, ast.For):
        return (
            isinstance(stmt.target, ast.Name)
            and isinstance(stmt.iter, ast.Call)
            and _call_path(stmt.iter.func) == ("cutlass", "range_constexpr")
            and len(stmt.iter.args) == 1
            and not stmt.iter.keywords
            and _const_int(stmt.iter.args[0]) is not None
            and not stmt.orelse
            and all(
                _scalar_preload_statement_is_safe(
                    child,
                    matched_statements=matched_statements,
                    local_containers=local_containers,
                    source_tensor=source_tensor,
                    tensor_names=tensor_names,
                    proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                    reaching_definitions=reaching_definitions,
                    constexpr_values=constexpr_values,
                    loop_assigned_names=loop_assigned_names,
                )
                for child in stmt.body
            )
        )
    if isinstance(stmt, ast.If):
        return expression_is_safe(stmt.test) and all(
            _scalar_preload_statement_is_safe(
                child,
                matched_statements=matched_statements,
                local_containers=local_containers,
                source_tensor=source_tensor,
                tensor_names=tensor_names,
                proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                reaching_definitions=reaching_definitions,
                constexpr_values=constexpr_values,
                loop_assigned_names=loop_assigned_names,
            )
            for child in (*stmt.body, *stmt.orelse)
        )
    return isinstance(stmt, ast.Pass)


def _earliest_prefetch_insertion_index(
    owner: list[ast.stmt],
    loop_index: int,
    *,
    dependencies: set[str],
    dominating_prefix: list[ast.stmt],
    write_counts: dict[str, int],
    state_tensor: str,
    tensor_names: frozenset[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> int:
    """Find the earliest same-branch point that dominates every prefetch use."""
    private_register_tensors = set(
        _private_register_tensor_names(dominating_prefix, write_counts)
    )
    insertion_index = 0
    for index, stmt in enumerate(owner[:loop_index]):
        if _assigned_names(
            stmt
        ) & dependencies or not _prefetch_overlap_statement_is_safe(
            stmt,
            private_register_tensors=private_register_tensors,
            state_tensor=state_tensor,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
        ):
            insertion_index = index + 1
        private_register_tensors.update(
            _private_register_tensor_names([stmt], write_counts)
        )
    return insertion_index


def _only_allowed_source_uses(
    body: list[ast.stmt], source: str, allowed_nodes: frozenset[int]
) -> bool:
    """Reject every source reference outside the matched load/store pointers.

    This deliberately forbids local aliases, pointer snapshots, extra loads,
    and rebinding. Runtime-specialized disjointness facts separately prove the
    source cannot be mutated through another tensor argument.
    """
    uses = {
        id(node)
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and node.id == source
    }
    return uses == allowed_nodes


def _is_generated_index_expr(node: ast.expr) -> bool:
    """Accept only side-effect-free scalar forms emitted for CuTe addresses."""
    node = _unwrap_int_cast(node)
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load)
    if isinstance(node, ast.Constant):
        return type(node.value) in (bool, int)
    if _thread_axis(node) is not None:
        return True
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ):
        return _is_generated_index_expr(node.left) and _is_generated_index_expr(
            node.right
        )
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Not)
    ):
        return _is_generated_index_expr(node.operand)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        return all(_is_generated_index_expr(value) for value in node.values)
    if isinstance(node, ast.Compare):
        return (
            len(node.ops) == len(node.comparators)
            and all(
                isinstance(
                    op,
                    (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE),
                )
                for op in node.ops
            )
            and _is_generated_index_expr(node.left)
            and all(_is_generated_index_expr(value) for value in node.comparators)
        )
    return False


def _stable_generated_expr(
    node: ast.expr,
    *,
    definitions: dict[str, ast.expr],
    uniform_names: frozenset[str],
    constexpr_values: dict[str, int],
    loop_var: str,
    forbidden_names: frozenset[str],
    visiting: frozenset[str] = frozenset(),
) -> bool:
    """Prove a generated scalar expression is pure and stable for one row loop."""
    node = _unwrap_int_cast(node)
    if isinstance(node, ast.Constant):
        return type(node.value) in (bool, int)
    if isinstance(node, ast.Name):
        if node.id in forbidden_names:
            return False
        if node.id == loop_var or node.id in constexpr_values:
            return True
        if node.id in visiting:
            return False
        definition = definitions.get(node.id)
        if definition is not None:
            return _stable_generated_expr(
                definition,
                definitions=definitions,
                uniform_names=uniform_names,
                constexpr_values=constexpr_values,
                loop_var=loop_var,
                forbidden_names=forbidden_names,
                visiting=visiting | {node.id},
            )
        return node.id in uniform_names
    if isinstance(node, ast.Subscript):
        return _thread_axis(node) is not None or _block_axis(node) is not None
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ):
        return _stable_generated_expr(
            node.left,
            definitions=definitions,
            uniform_names=uniform_names,
            constexpr_values=constexpr_values,
            loop_var=loop_var,
            forbidden_names=forbidden_names,
            visiting=visiting,
        ) and _stable_generated_expr(
            node.right,
            definitions=definitions,
            uniform_names=uniform_names,
            constexpr_values=constexpr_values,
            loop_var=loop_var,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Not)
    ):
        return _stable_generated_expr(
            node.operand,
            definitions=definitions,
            uniform_names=uniform_names,
            constexpr_values=constexpr_values,
            loop_var=loop_var,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        return all(
            _stable_generated_expr(
                value,
                definitions=definitions,
                uniform_names=uniform_names,
                constexpr_values=constexpr_values,
                loop_var=loop_var,
                forbidden_names=forbidden_names,
                visiting=visiting,
            )
            for value in node.values
        )
    if isinstance(node, ast.Compare):
        return (
            len(node.ops) == len(node.comparators)
            and all(
                isinstance(
                    op,
                    (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE),
                )
                for op in node.ops
            )
            and _stable_generated_expr(
                node.left,
                definitions=definitions,
                uniform_names=uniform_names,
                constexpr_values=constexpr_values,
                loop_var=loop_var,
                forbidden_names=forbidden_names,
                visiting=visiting,
            )
            and all(
                _stable_generated_expr(
                    value,
                    definitions=definitions,
                    uniform_names=uniform_names,
                    constexpr_values=constexpr_values,
                    loop_var=loop_var,
                    forbidden_names=forbidden_names,
                    visiting=visiting,
                )
                for value in node.comparators
            )
        )
    return False


def _expression_is_integer(
    node: ast.expr,
    *,
    definitions: dict[str, ast.expr],
    constexpr_values: dict[str, int],
    forbidden_names: frozenset[str],
    visiting: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.Name):
        if node.id in forbidden_names:
            return False
        if node.id in constexpr_values:
            return True
        if node.id in visiting or node.id not in definitions:
            return False
        return _expression_is_integer(
            definitions[node.id],
            definitions=definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
            visiting=visiting | {node.id},
        )
    if isinstance(node, ast.Subscript):
        return _thread_axis(node) is not None or _block_axis(node) is not None
    if isinstance(node, ast.Call):
        return _is_cutlass_int_cast(node) and len(node.args) == 1 and not node.keywords
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _expression_is_integer(
            node.operand,
            definitions=definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
    return (
        isinstance(node, ast.BinOp)
        and isinstance(
            node.op,
            (
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.FloorDiv,
                ast.Mod,
                ast.LShift,
                ast.RShift,
                ast.BitAnd,
                ast.BitOr,
                ast.BitXor,
            ),
        )
        and _expression_is_integer(
            node.left,
            definitions=definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
        and _expression_is_integer(
            node.right,
            definitions=definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
    )


def _expression_parity(
    node: ast.expr,
    *,
    definitions: dict[str, ast.expr],
    constexpr_values: dict[str, int],
    forbidden_names: frozenset[str],
    visiting: frozenset[str] = frozenset(),
) -> int | None:
    """Return a proven expression value modulo two, or ``None``."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value & 1
    if isinstance(node, ast.Name):
        if node.id in forbidden_names:
            return None
        if node.id in constexpr_values:
            return constexpr_values[node.id] & 1
        if node.id in visiting or node.id not in definitions:
            return None
        return _expression_parity(
            definitions[node.id],
            definitions=definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
            visiting=visiting | {node.id},
        )
    if isinstance(node, ast.Call):
        if not (
            _is_cutlass_int_cast(node)
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.expr)
            and _expression_is_integer(
                node.args[0],
                definitions=definitions,
                constexpr_values=constexpr_values,
                forbidden_names=forbidden_names,
                visiting=visiting,
            )
        ):
            return None
        return _expression_parity(
            node.args[0],
            definitions=definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            return None
        return _expression_parity(
            node.operand,
            definitions=definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
    if not isinstance(node, ast.BinOp):
        return None
    left = _expression_parity(
        node.left,
        definitions=definitions,
        constexpr_values=constexpr_values,
        forbidden_names=forbidden_names,
        visiting=visiting,
    )
    right = _expression_parity(
        node.right,
        definitions=definitions,
        constexpr_values=constexpr_values,
        forbidden_names=forbidden_names,
        visiting=visiting,
    )
    if isinstance(node.op, (ast.Add, ast.Sub)):
        return None if left is None or right is None else left ^ right
    if isinstance(node.op, ast.Mult):
        left_integer = _expression_is_integer(
            node.left,
            definitions=definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
        right_integer = _expression_is_integer(
            node.right,
            definitions=definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
        if left == 0 and right_integer or right == 0 and left_integer:
            return 0
        return None if left is None or right is None else left & right
    constant = (
        _constant_int_expr(node)
        if _expression_is_integer(
            node,
            definitions=definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
        else None
    )
    return None if constant is None else constant & 1


_INT32_MAX = (1 << 31) - 1


def _nonnegative_int32_bounds(
    node: ast.expr,
    *,
    definitions: dict[str, ast.expr],
    constexpr_values: dict[str, int],
    loop_bounds: dict[str, tuple[int, int]],
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int | None, int | None, int | None],
    forbidden_names: frozenset[str],
    visiting: frozenset[str] = frozenset(),
) -> tuple[int, int] | None:
    """Bound a generated nonnegative Int32 expression without overflow."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return (node.value, node.value) if 0 <= node.value <= _INT32_MAX else None
    if isinstance(node, ast.Name):
        if node.id in forbidden_names:
            return None
        if node.id in loop_bounds:
            return loop_bounds[node.id]
        if node.id in constexpr_values:
            value = constexpr_values[node.id]
            return (value, value) if 0 <= value <= _INT32_MAX else None
        if node.id in visiting or node.id not in definitions:
            return None
        return _nonnegative_int32_bounds(
            definitions[node.id],
            definitions=definitions,
            constexpr_values=constexpr_values,
            loop_bounds=loop_bounds,
            thread_block_dims=thread_block_dims,
            block_grid_dims=block_grid_dims,
            forbidden_names=forbidden_names,
            visiting=visiting | {node.id},
        )
    if isinstance(node, ast.Subscript):
        thread_axis = _thread_axis(node)
        if thread_axis is not None and 0 <= thread_axis < 3:
            return 0, thread_block_dims[thread_axis] - 1
        block_axis = _block_axis(node)
        if block_axis is not None and 0 <= block_axis < 3:
            extent = block_grid_dims[block_axis]
            return None if extent is None or extent <= 0 else (0, extent - 1)
        return None
    if isinstance(node, ast.Call):
        if (
            _call_path(node.func) != ("cutlass", "Int32")
            or len(node.args) != 1
            or node.keywords
            or not isinstance(node.args[0], ast.expr)
        ):
            return None
        return _nonnegative_int32_bounds(
            node.args[0],
            definitions=definitions,
            constexpr_values=constexpr_values,
            loop_bounds=loop_bounds,
            thread_block_dims=thread_block_dims,
            block_grid_dims=block_grid_dims,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _nonnegative_int32_bounds(
            node.operand,
            definitions=definitions,
            constexpr_values=constexpr_values,
            loop_bounds=loop_bounds,
            thread_block_dims=thread_block_dims,
            block_grid_dims=block_grid_dims,
            forbidden_names=forbidden_names,
            visiting=visiting,
        )
    if not isinstance(node, ast.BinOp):
        return None
    left = _nonnegative_int32_bounds(
        node.left,
        definitions=definitions,
        constexpr_values=constexpr_values,
        loop_bounds=loop_bounds,
        thread_block_dims=thread_block_dims,
        block_grid_dims=block_grid_dims,
        forbidden_names=forbidden_names,
        visiting=visiting,
    )
    right = _nonnegative_int32_bounds(
        node.right,
        definitions=definitions,
        constexpr_values=constexpr_values,
        loop_bounds=loop_bounds,
        thread_block_dims=thread_block_dims,
        block_grid_dims=block_grid_dims,
        forbidden_names=forbidden_names,
        visiting=visiting,
    )
    if left is None or right is None:
        return None
    if isinstance(node.op, ast.Add):
        result = left[0] + right[0], left[1] + right[1]
    elif isinstance(node.op, ast.Mult):
        result = left[0] * right[0], left[1] * right[1]
    elif isinstance(node.op, ast.FloorDiv) and right[0] == right[1] and right[0] > 0:
        result = left[0] // right[0], left[1] // right[0]
    else:
        return None
    return result if result[1] <= _INT32_MAX else None


def _packed_scalar_predicate_is_always_true(
    predicate: ast.expr,
    *,
    source: str,
    indices: dict[int, ast.expr],
    definitions: dict[str, ast.expr],
    constexpr_values: dict[str, int],
    loop_var: str,
    trip_count: int,
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int | None, int | None, int | None],
    proven_tensor_size_values: dict[tuple[str, int], tuple[str, int]],
    forbidden_names: frozenset[str],
    write_counts: dict[str, int],
) -> bool:
    """Prove every evaluation of an exact generated scalar mask is true."""
    terms = (
        predicate.values
        if isinstance(predicate, ast.BoolOp) and isinstance(predicate.op, ast.And)
        else [predicate]
    )
    for term in terms:
        if not (
            isinstance(term, ast.Compare)
            and len(term.ops) == 1
            and isinstance(term.ops[0], ast.Lt)
            and len(term.comparators) == 1
            and isinstance(term.comparators[0], ast.Name)
        ):
            return False
        size_name = term.comparators[0].id
        matching_bounds = [
            value
            for (tensor_name, dim), (argument_name, value) in (
                proven_tensor_size_values.items()
            )
            if tensor_name == source
            and argument_name == size_name
            and dim in indices
            and _same_expr(_unwrap_int_cast(term.left), _unwrap_int_cast(indices[dim]))
        ]
        if len(matching_bounds) != 1 or write_counts.get(size_name):
            return False
        upper = matching_bounds[0]
        if not 0 <= upper <= _INT32_MAX:
            return False
        value_bounds = _nonnegative_int32_bounds(
            term.left,
            definitions=definitions,
            constexpr_values=constexpr_values,
            loop_bounds={loop_var: (0, trip_count - 1)},
            thread_block_dims=thread_block_dims,
            block_grid_dims=block_grid_dims,
            forbidden_names=forbidden_names,
        )
        if value_bounds is None or value_bounds[1] >= upper:
            return False
    return True


def _direct_read_only_tensor_source(body: list[ast.stmt], source: str) -> bool:
    """Require every source reference to belong to a direct tensor load."""
    module = ast.Module(body=body, type_ignores=[])
    allowed: set[int] = set()
    for call in (node for node in ast.walk(module) if isinstance(node, ast.Call)):
        pointer_node: ast.expr | None = None
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "load"
            and not call.args
            and not call.keywords
        ):
            pointer_node = call.func.value
        elif (
            _call_path(call.func) == ("cute", "arch", "load")
            and len(call.args) == 2
            and not call.keywords
        ):
            pointer_node = call.args[0]
        if pointer_node is None:
            continue
        pointer = _pointer_info(pointer_node)
        if pointer is None or pointer.tensor_name != source:
            continue
        allowed.update(
            id(node)
            for node in ast.walk(pointer_node)
            if isinstance(node, ast.Name) and node.id == source
        )
    uses = {
        id(node)
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and node.id == source
    }
    return bool(uses) and uses == allowed


def _loop_local_definitions_before(
    statements: list[ast.stmt],
    stop: int,
    loop_written_names: frozenset[str],
) -> dict[str, ast.expr]:
    """Capture pure straight-line aliases as they exist at one load site."""
    definitions: dict[str, ast.expr] = {}
    for stmt in statements[:stop]:
        assigned = _assigned_names(stmt)
        for name in assigned:
            definitions.pop(name, None)
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and _is_generated_index_expr(stmt.value)
        ):
            continue
        expanded = _expand_aliases(stmt.value, definitions)
        if expanded is not None and not any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in loop_written_names
            for node in ast.walk(expanded)
        ):
            definitions[stmt.targets[0].id] = expanded
    return definitions


def _name_loads_outside(node: ast.AST, subtree: ast.AST, name: str) -> bool:
    inside = {id(child) for child in ast.walk(subtree)}
    return any(
        id(child) not in inside
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and child.id == name
    )


def _match_packed_scalar_load(
    *,
    loop: ast.For,
    loop_var: str,
    trip_count: int,
    state_tensor: str,
    root_body: list[ast.stmt],
    lexical_prefix: list[ast.stmt],
    reaching_definitions: dict[str, ast.expr],
    write_counts: dict[str, int],
    uniform_names: frozenset[str],
    constexpr_values: dict[str, int],
    tensor_metadata: dict[str, TensorMetadata],
    tensor_names: frozenset[str],
    proven_tensor_base_alignments: frozenset[str],
    proven_tensor_size_values: dict[tuple[str, int], tuple[str, int]],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int | None, int | None, int | None],
) -> _PackedScalarLoad | None:
    """Match one exact row-contiguous BF16 scalar read for pair preloading."""
    if trip_count % 2:
        return None
    loop_writes = frozenset(
        _assigned_names(ast.Module(body=loop.body, type_ignores=[]))
    )
    sites: list[tuple[ast.Assign, str, _Pointer, ast.expr | None, ast.expr]] = []
    for index, stmt in enumerate(loop.body):
        loaded = _scalar_bf16_tensor_load(stmt)
        if loaded is None:
            continue
        assert isinstance(stmt, ast.Assign)
        target, pointer, predicate, original_value = loaded
        local_definitions = _loop_local_definitions_before(
            loop.body, index, loop_writes
        )
        offset = _expand_aliases(pointer.active_offset, local_definitions)
        expanded_predicate = (
            None if predicate is None else _expand_aliases(predicate, local_definitions)
        )
        if offset is None or predicate is not None and expanded_predicate is None:
            continue
        sites.append(
            (
                stmt,
                target,
                replace(pointer, active_offset=offset, offset=offset),
                expanded_predicate,
                original_value,
            )
        )
    if not sites:
        return None

    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for index, (_, target, pointer, predicate, _) in enumerate(sites):
        key = (
            pointer.tensor_name,
            target,
            ast.dump(pointer.active_offset, include_attributes=False),
            "" if predicate is None else ast.dump(predicate, include_attributes=False),
        )
        groups.setdefault(key, []).append(index)
    eligible_groups: list[list[int]] = []
    for indices in groups.values():
        source = sites[indices[0]][2].tensor_name
        source_sites = [
            index for index, site in enumerate(sites) if site[2].tensor_name == source
        ]
        if indices == source_sites and len(indices) <= 2:
            eligible_groups.append(indices)
    if len(eligible_groups) != 1:
        return None
    matched = [sites[index] for index in eligible_groups[0]]
    _, target_name, pointer, predicate, _ = matched[0]
    source = pointer.tensor_name
    metadata = tensor_metadata.get(source)
    if (
        source == state_tensor
        or source not in tensor_names
        or source not in proven_tensor_base_alignments
        or metadata is None
        or metadata.dtype != "cutlass.BFloat16"
        or write_counts.get(target_name) != len(matched)
        or _name_loads_outside(
            ast.Module(body=root_body, type_ignores=[]), loop, target_name
        )
        or any(
            frozenset((source, other)) not in proven_disjoint_tensor_pairs
            for other in tensor_names
            if other != source
        )
        or not _direct_read_only_tensor_source(root_body, source)
    ):
        return None
    local_containers = set(_private_register_tensor_names(lexical_prefix, write_counts))
    local_containers.update(
        stmt.targets[0].id
        for stmt in ast.walk(loop)
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.List)
    )
    matched_statements = frozenset(id(site[0]) for site in matched)
    if not all(
        _scalar_preload_statement_is_safe(
            stmt,
            matched_statements=matched_statements,
            local_containers=frozenset(local_containers),
            source_tensor=source,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
            reaching_definitions=reaching_definitions,
            constexpr_values=constexpr_values,
            loop_assigned_names=loop_writes,
        )
        for stmt in loop.body
    ):
        return None

    rank = len(metadata.shape)
    strides = tuple(
        proven_tensor_stride_values.get((source, dim), 0) for dim in range(rank)
    )
    indices = _indices_by_stride(pointer.active_offset, source, rank)
    if (
        rank == 0
        or indices is None
        or any(size <= 0 for size in metadata.shape)
        or any(stride <= 0 for stride in strides)
        or strides[-1] != 1
        or any(stride % 2 for stride in strides[:-1])
    ):
        return None
    forbidden_names = frozenset(
        name for name in (*uniform_names, *constexpr_values) if write_counts.get(name)
    )
    if (
        not _scalar_preload_expr_is_safe(
            pointer.active_offset,
            source_tensor=source,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
            reaching_definitions=reaching_definitions,
            constexpr_values=constexpr_values,
            loop_assigned_names=loop_writes,
        )
        or predicate is not None
        and not _scalar_preload_expr_is_safe(
            predicate,
            source_tensor=source,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
            reaching_definitions=reaching_definitions,
            constexpr_values=constexpr_values,
            loop_assigned_names=loop_writes,
        )
    ):
        return None
    dependencies: set[str] = set()
    for dim, index in indices.items():
        linear = _linear_coefficients(index, loop_var, {})
        expected = 1 if dim == rank - 1 else 0
        if (
            linear is None
            or linear[0] != expected
            or not _stable_generated_expr(
                index,
                definitions=reaching_definitions,
                uniform_names=uniform_names,
                constexpr_values=constexpr_values,
                loop_var=loop_var,
                forbidden_names=forbidden_names,
            )
        ):
            return None
        dependencies.update(
            node.id for node in ast.walk(index) if isinstance(node, ast.Name)
        )
    if predicate is not None:
        if not _stable_generated_expr(
            predicate,
            definitions=reaching_definitions,
            uniform_names=uniform_names,
            constexpr_values=constexpr_values,
            loop_var=loop_var,
            forbidden_names=forbidden_names,
        ):
            return None
        dependencies.update(
            node.id for node in ast.walk(predicate) if isinstance(node, ast.Name)
        )
    dependencies -= {loop_var, "cutlass", "cute", "ir", source}
    if loop_writes & dependencies:
        return None

    first_index = _substitute(indices[rank - 1], {loop_var: ast.Constant(value=0)})
    # The caller supplied a cache-key-backed 4-byte base-pointer fact. Even
    # non-final strides plus an even first final index therefore prove every
    # pair-start address is Uint32-aligned.
    if (
        _expression_parity(
            first_index,
            definitions=reaching_definitions,
            constexpr_values=constexpr_values,
            forbidden_names=forbidden_names,
        )
        != 0
    ):
        return None
    if predicate is not None and _packed_scalar_predicate_is_always_true(
        predicate,
        source=source,
        indices=indices,
        definitions=reaching_definitions,
        constexpr_values=constexpr_values,
        loop_var=loop_var,
        trip_count=trip_count,
        thread_block_dims=thread_block_dims,
        block_grid_dims=block_grid_dims,
        proven_tensor_size_values=proven_tensor_size_values,
        forbidden_names=forbidden_names,
        write_counts=write_counts,
    ):
        predicate = None
    return _PackedScalarLoad(
        tensor_name=source,
        target_name=target_name,
        statements=tuple(site[0] for site in matched),
        pointer=pointer,
        predicate=predicate,
        original_values=tuple(site[4] for site in matched),
    )


def _exact_axis_partition(
    node: ast.expr,
    *,
    lane_var: str,
    definitions: dict[str, ast.expr],
    axis: int,
    coefficient: int,
) -> bool:
    """Prove an expression is exactly ``thread_idx()[axis] * coefficient``."""
    expanded = _expand_aliases(node, definitions)
    if expanded is None:
        return False
    linear = _linear_coefficients(expanded, lane_var, {})
    if linear != (0, {axis: coefficient}):
        return False

    class _ZeroAxis(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if _thread_axis(node) is not None:
                return ast.copy_location(ast.Constant(value=0), node)
            return self.generic_visit(node)

    zeroed = _ZeroAxis().visit(expanded)
    assert isinstance(zeroed, ast.expr)
    return _constant_int_expr(zeroed) == 0


def _row_partition_axis(
    node: ast.expr,
    *,
    lane_var: str,
    trip_count: int,
    definitions: dict[str, ast.expr],
    uniform_names: frozenset[str],
    write_counts: dict[str, int],
    proven_tensor_stride_values: dict[tuple[str, int], int],
) -> int | None:
    """Prove one row per loop lane and row-axis thread, plus a CTA base."""
    expanded = _expand_aliases(node, definitions)
    if expanded is None:
        return None
    linear = _linear_coefficients(expanded, lane_var, {})
    if linear is None or linear[0] != 1 or len(linear[1]) != 1:
        return None
    row_axis, row_coefficient = next(iter(linear[1].items()))
    if not 0 <= row_axis < 3 or row_coefficient != trip_count:
        return None

    class _ZeroPartitionCoordinates(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id == lane_var:
                return ast.copy_location(ast.Constant(value=0), node)
            return node

        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if _thread_axis(node) is not None:
                return ast.copy_location(ast.Constant(value=0), node)
            return self.generic_visit(node)

    residual = _ZeroPartitionCoordinates().visit(expanded)
    assert isinstance(residual, ast.expr)
    if not _index_expr_is_cta_uniform(
        residual,
        stride_source=None,
        definitions={},
        uniform_names=uniform_names,
        uniform_scalar_load_names=frozenset(),
        write_counts=write_counts,
        proven_tensor_stride_values=proven_tensor_stride_values,
    ):
        return None
    return row_axis


def _constant_int_expr(node: ast.expr) -> int | None:
    node = _unwrap_int_cast(node)
    value = _const_int(node)
    if value is not None:
        return value
    if isinstance(node, ast.UnaryOp):
        operand = _constant_int_expr(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        return None
    if not isinstance(node, ast.BinOp):
        return None
    left = _constant_int_expr(node.left)
    right = _constant_int_expr(node.right)
    if left is None or right is None:
        return None
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.FloorDiv) and right:
        return left // right
    if isinstance(node.op, ast.Mod) and right:
        return left % right
    return None


def _same_expr(left: ast.expr, right: ast.expr) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def _predicate_is_cta_uniform(
    node: ast.expr,
    *,
    definitions: dict[str, ast.expr],
    uniform_names: frozenset[str],
    uniform_scalar_load_names: frozenset[str],
    write_counts: dict[str, int],
    visiting: frozenset[str] = frozenset(),
) -> bool:
    """Positively prove that every thread in the CTA sees one predicate.

    Only literals, immutable scalar arguments/constexprs, and immutable local
    definitions transitively composed from those values are accepted. Tensor
    loads, thread/block coordinates, opaque calls, and unknown names all fail
    closed. This is intentionally narrower than the address-expression grammar:
    a false masked load aliases element zero, whose value may change while the
    async ring is prefetching future rows.
    """
    if isinstance(node, ast.Constant):
        return type(node.value) in (bool, int)
    if isinstance(node, ast.Name):
        if node.id in visiting:
            return False
        if node.id in uniform_scalar_load_names and write_counts.get(node.id, 0) == 1:
            return True
        if node.id in definitions:
            return _predicate_is_cta_uniform(
                definitions[node.id],
                definitions=definitions,
                uniform_names=uniform_names,
                uniform_scalar_load_names=uniform_scalar_load_names,
                write_counts=write_counts,
                visiting=visiting | {node.id},
            )
        return node.id in uniform_names and write_counts.get(node.id, 0) == 0
    if isinstance(node, ast.Call):
        if not _is_cutlass_int_cast(node) or len(node.args) != 1 or node.keywords:
            return False
        argument = node.args[0]
        return isinstance(argument, ast.expr) and _predicate_is_cta_uniform(
            argument,
            definitions=definitions,
            uniform_names=uniform_names,
            uniform_scalar_load_names=uniform_scalar_load_names,
            write_counts=write_counts,
            visiting=visiting,
        )
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ):
        return _predicate_is_cta_uniform(
            node.left,
            definitions=definitions,
            uniform_names=uniform_names,
            uniform_scalar_load_names=uniform_scalar_load_names,
            write_counts=write_counts,
            visiting=visiting,
        ) and _predicate_is_cta_uniform(
            node.right,
            definitions=definitions,
            uniform_names=uniform_names,
            uniform_scalar_load_names=uniform_scalar_load_names,
            write_counts=write_counts,
            visiting=visiting,
        )
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Not)
    ):
        return _predicate_is_cta_uniform(
            node.operand,
            definitions=definitions,
            uniform_names=uniform_names,
            uniform_scalar_load_names=uniform_scalar_load_names,
            write_counts=write_counts,
            visiting=visiting,
        )
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        return all(
            _predicate_is_cta_uniform(
                value,
                definitions=definitions,
                uniform_names=uniform_names,
                uniform_scalar_load_names=uniform_scalar_load_names,
                write_counts=write_counts,
                visiting=visiting,
            )
            for value in node.values
        )
    if isinstance(node, ast.Compare):
        return (
            len(node.ops) == len(node.comparators)
            and all(
                isinstance(
                    op,
                    (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE),
                )
                for op in node.ops
            )
            and _predicate_is_cta_uniform(
                node.left,
                definitions=definitions,
                uniform_names=uniform_names,
                uniform_scalar_load_names=uniform_scalar_load_names,
                write_counts=write_counts,
                visiting=visiting,
            )
            and all(
                _predicate_is_cta_uniform(
                    value,
                    definitions=definitions,
                    uniform_names=uniform_names,
                    uniform_scalar_load_names=uniform_scalar_load_names,
                    write_counts=write_counts,
                    visiting=visiting,
                )
                for value in node.comparators
            )
        )
    return False


def _nonnegative_int32_column_affine(
    node: ast.expr,
    *,
    column_axis: int,
    thread_block_dims: tuple[int, int, int],
) -> tuple[int, int] | None:
    """Return ``(constant, thread coefficient)`` for a tiny exact grammar."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        if 0 <= node.value <= (1 << 31) - 1:
            return node.value, 0
        return None
    if isinstance(node, ast.Subscript):
        axis = _thread_axis(node)
        if axis == column_axis:
            return 0, 1
        return None
    if isinstance(node, ast.Call):
        if (
            _call_path(node.func) != ("cutlass", "Int32")
            or len(node.args) != 1
            or node.keywords
            or not isinstance(node.args[0], ast.expr)
        ):
            return None
        return _nonnegative_int32_column_affine(
            node.args[0],
            column_axis=column_axis,
            thread_block_dims=thread_block_dims,
        )
    if not isinstance(node, ast.BinOp):
        return None
    if isinstance(node.op, ast.Add):
        left = _nonnegative_int32_column_affine(
            node.left,
            column_axis=column_axis,
            thread_block_dims=thread_block_dims,
        )
        right = _nonnegative_int32_column_affine(
            node.right,
            column_axis=column_axis,
            thread_block_dims=thread_block_dims,
        )
        if left is None or right is None:
            return None
        result = left[0] + right[0], left[1] + right[1]
    elif isinstance(node.op, ast.Mult):
        left_constant = (
            node.left.value
            if isinstance(node.left, ast.Constant) and type(node.left.value) is int
            else None
        )
        right_constant = (
            node.right.value
            if isinstance(node.right, ast.Constant) and type(node.right.value) is int
            else None
        )
        if left_constant is not None and left_constant >= 0:
            value = _nonnegative_int32_column_affine(
                node.right,
                column_axis=column_axis,
                thread_block_dims=thread_block_dims,
            )
            factor = left_constant
        elif right_constant is not None and right_constant >= 0:
            value = _nonnegative_int32_column_affine(
                node.left,
                column_axis=column_axis,
                thread_block_dims=thread_block_dims,
            )
            factor = right_constant
        else:
            return None
        if value is None:
            return None
        result = value[0] * factor, value[1] * factor
    else:
        return None
    maximum = result[0] + result[1] * (thread_block_dims[column_axis] - 1)
    return result if maximum <= (1 << 31) - 1 else None


def _column_bound_is_always_true(
    node: ast.expr,
    *,
    definitions: dict[str, ast.expr],
    column_axis: int,
    vector_width: int,
    thread_block_dims: tuple[int, int, int],
) -> bool:
    """Recognize the generated, statically true final-dimension bounds."""
    if not (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and len(node.comparators) == 1
        and isinstance(node.ops[0], ast.Lt)
        and isinstance(node.comparators[0], ast.Constant)
        and type(node.comparators[0].value) is int
        and 0 <= node.comparators[0].value <= (1 << 31) - 1
    ):
        return False
    expanded = _expand_aliases(node.left, definitions)
    if expanded is None:
        return False
    affine = _nonnegative_int32_column_affine(
        expanded,
        column_axis=column_axis,
        thread_block_dims=thread_block_dims,
    )
    if affine is None or affine[1] != vector_width:
        return False
    base = affine[0]
    maximum = base + vector_width * (thread_block_dims[column_axis] - 1)
    return maximum < node.comparators[0].value


def _predicate_is_safe_for_fallback(
    node: ast.expr,
    *,
    immutable_definitions: dict[str, ast.expr],
    range_definitions: dict[str, ast.expr],
    uniform_names: frozenset[str],
    uniform_scalar_load_names: frozenset[str],
    write_counts: dict[str, int],
    column_axis: int,
    vector_width: int,
    thread_block_dims: tuple[int, int, int],
) -> bool:
    if _predicate_is_cta_uniform(
        node,
        definitions=immutable_definitions,
        uniform_names=uniform_names,
        uniform_scalar_load_names=uniform_scalar_load_names,
        write_counts=write_counts,
    ):
        return True
    if isinstance(node, ast.Name) and node.id in immutable_definitions:
        remaining_definitions = dict(immutable_definitions)
        definition = remaining_definitions.pop(node.id)
        return _predicate_is_safe_for_fallback(
            definition,
            immutable_definitions=remaining_definitions,
            range_definitions=range_definitions,
            uniform_names=uniform_names,
            uniform_scalar_load_names=uniform_scalar_load_names,
            write_counts=write_counts,
            column_axis=column_axis,
            vector_width=vector_width,
            thread_block_dims=thread_block_dims,
        )
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return all(
            _predicate_is_safe_for_fallback(
                value,
                immutable_definitions=immutable_definitions,
                range_definitions=range_definitions,
                uniform_names=uniform_names,
                uniform_scalar_load_names=uniform_scalar_load_names,
                write_counts=write_counts,
                column_axis=column_axis,
                vector_width=vector_width,
                thread_block_dims=thread_block_dims,
            )
            for value in node.values
        )
    return _column_bound_is_always_true(
        node,
        definitions=range_definitions,
        column_axis=column_axis,
        vector_width=vector_width,
        thread_block_dims=thread_block_dims,
    )


def _strip_always_true_column_bounds(
    node: ast.expr,
    *,
    definitions: dict[str, ast.expr],
    column_axis: int,
    vector_width: int,
    thread_block_dims: tuple[int, int, int],
) -> ast.expr:
    """Remove exact vector-column bounds already proven by the tile shape."""
    if _column_bound_is_always_true(
        node,
        definitions=definitions,
        column_axis=column_axis,
        vector_width=vector_width,
        thread_block_dims=thread_block_dims,
    ):
        return ast.Constant(value=True)
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.And):
        return node
    values: list[ast.expr] = []
    for value in node.values:
        simplified = _strip_always_true_column_bounds(
            value,
            definitions=definitions,
            column_axis=column_axis,
            vector_width=vector_width,
            thread_block_dims=thread_block_dims,
        )
        if isinstance(simplified, ast.Constant) and simplified.value is True:
            continue
        values.append(simplified)
    if not values:
        return ast.Constant(value=True)
    if len(values) == 1:
        return values[0]
    return ast.BoolOp(op=ast.And(), values=values)


def _name_dependency_closure(
    node: ast.AST, definitions: dict[str, ast.expr]
) -> set[str]:
    """Return direct and transitively defined names used by ``node``."""
    pending = [
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    ]
    result: set[str] = set()
    while pending:
        name = pending.pop()
        if name in result:
            continue
        result.add(name)
        definition = definitions.get(name)
        if definition is not None:
            pending.extend(
                child.id
                for child in ast.walk(definition)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            )
    return result


def _match_loop(
    owner: list[ast.stmt],
    loop_index: int,
    lexical_prefix: list[ast.stmt],
    dominating_prefix: list[ast.stmt],
    root_body: list[ast.stmt],
    definitions: dict[str, ast.expr],
    immutable_definitions: dict[str, ast.expr],
    aliases: dict[str, ast.expr],
    write_counts: dict[str, int],
    uniform_names: frozenset[str],
    constexpr_values: dict[str, int],
    tensor_metadata: dict[str, TensorMetadata],
    tensor_names: frozenset[str],
    proven_tensor_base_alignments: frozenset[str],
    proven_tensor_size_values: dict[tuple[str, int], tuple[str, int]],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int | None, int | None, int | None],
) -> _Region | None:
    loop = owner[loop_index]
    if not (
        isinstance(loop, ast.For)
        and isinstance(loop.target, ast.Name)
        and isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Name)
        and loop.iter.func.id == "range"
        and len(loop.iter.args) == 1
        and not loop.iter.keywords
        and not loop.orelse
    ):
        return None
    trip_count = _const_int(loop.iter.args[0])
    if trip_count is None or trip_count < 4 or not loop.body:
        return None
    loop_var = loop.target.id
    loop_module = ast.Module(body=loop.body, type_ignores=[])
    if any(
        isinstance(
            node,
            (
                ast.AsyncFor,
                ast.Break,
                ast.Continue,
                ast.Raise,
                ast.Return,
                ast.Try,
                ast.While,
                ast.With,
                ast.Yield,
                ast.YieldFrom,
            ),
        )
        for node in ast.walk(loop_module)
    ):
        return None
    writes = [
        node.id
        for node in ast.walk(loop_module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    if loop_var in writes:
        return None

    row_stmt = loop.body[0]
    if not (
        isinstance(row_stmt, ast.Assign)
        and len(row_stmt.targets) == 1
        and isinstance(row_stmt.targets[0], ast.Name)
    ):
        return None
    row_name = row_stmt.targets[0].id
    row_rhs = row_stmt.value
    if writes.count(row_name) != 1 or not _is_generated_index_expr(row_rhs):
        return None
    row_axis = _row_partition_axis(
        row_rhs,
        lane_var=loop_var,
        trip_count=trip_count,
        definitions=immutable_definitions,
        uniform_names=uniform_names,
        write_counts=write_counts,
        proven_tensor_stride_values=proven_tensor_stride_values,
    )
    if row_axis is None:
        return None

    loads = [
        (index, info)
        for index, stmt in enumerate(loop.body)
        if (info := _load_info(stmt)) is not None
    ]
    candidates: list[_Region] = []
    for load_index, load in loads:
        load_name, load_pointer, vector_type, vector_width = load
        tensor_name = load_pointer.tensor_name
        metadata = tensor_metadata.get(tensor_name)
        if (
            tensor_name not in tensor_names
            or metadata is None
            or metadata.dtype
            not in {
                "cutlass.BFloat16",
                "cutlass.Float16",
            }
        ):
            continue
        if any(
            frozenset((tensor_name, other_name)) not in proven_disjoint_tensor_pairs
            for other_name in tensor_names
            if other_name != tensor_name
        ):
            continue
        uniform_scalar_load_names = _proven_uniform_scalar_load_names(
            lexical_prefix,
            root_body=root_body,
            state_tensor=tensor_name,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
            proven_tensor_stride_values=proven_tensor_stride_values,
            uniform_names=uniform_names,
            write_counts=write_counts,
        )
        rank = len(metadata.shape)
        stride = tuple(
            proven_tensor_stride_values.get((tensor_name, dim), 0)
            for dim in range(rank)
        )
        if rank < 2:
            continue
        indices = _indices_by_stride(load_pointer.active_offset, tensor_name, rank)
        if indices is None or not all(
            _is_generated_index_expr(index) for index in indices.values()
        ):
            continue
        last_dim = len(metadata.shape) - 1
        if (
            stride[last_dim] != 1
            or any(size <= 0 for size in metadata.shape)
            or any(value <= 0 for value in stride)
            or any(
                value % vector_width
                for dim, value in enumerate(stride)
                if dim != last_dim
            )
        ):
            continue

        row_dims = []
        for dim, index in indices.items():
            unwrapped = _unwrap_int_cast(index)
            if isinstance(unwrapped, ast.Name) and unwrapped.id == row_name:
                row_dims.append(dim)
        if len(row_dims) != 1:
            continue
        row_dim = row_dims[0]
        column = indices.get(last_dim)
        if column is None or row_dim == last_dim:
            continue
        column_linear = _linear_coefficients(column, loop_var, definitions)
        if column_linear is None or column_linear[0] != 0 or len(column_linear[1]) != 1:
            continue
        column_axis, column_coefficient = next(iter(column_linear[1].items()))
        if (
            not 0 <= column_axis < 3
            or column_axis == row_axis
            or column_coefficient != vector_width
            or not _exact_axis_partition(
                column,
                lane_var=loop_var,
                definitions=definitions,
                axis=column_axis,
                coefficient=vector_width,
            )
            or thread_block_dims[column_axis] * vector_width != metadata.shape[last_dim]
            or thread_block_dims[row_axis] * trip_count != metadata.shape[row_dim]
            or stride[row_dim] < metadata.shape[last_dim]
            or any(
                thread_block_dims[axis] != 1
                for axis in range(3)
                if axis not in (row_axis, column_axis)
            )
        ):
            continue
        if any(
            not _index_expr_is_cta_uniform(
                index,
                stride_source=None,
                definitions=immutable_definitions,
                uniform_names=uniform_names,
                uniform_scalar_load_names=uniform_scalar_load_names,
                write_counts=write_counts,
                proven_tensor_stride_values=proven_tensor_stride_values,
            )
            for dim, index in indices.items()
            if dim not in (row_dim, last_dim)
        ):
            continue

        stored = _store_pointer(loop, tensor_name, vector_width)
        if stored is None:
            continue
        store_pointer, store_predicate, store_top_level = stored
        if loop.body.index(store_top_level) <= load_index:
            continue
        load_offset = _expand_aliases(load_pointer.active_offset, aliases)
        store_offset = _expand_aliases(store_pointer.active_offset, aliases)
        load_predicate = (
            None
            if load_pointer.predicate is None
            else _expand_aliases(load_pointer.predicate, aliases)
        )
        expanded_store_predicate = (
            None
            if store_predicate is None
            else _expand_aliases(store_predicate, aliases)
        )
        if (
            load_offset is None
            or store_offset is None
            or not _same_expr(load_offset, store_offset)
            or (load_predicate is None) != (expanded_store_predicate is None)
            or load_predicate is not None
            and expanded_store_predicate is not None
            and not _same_expr(load_predicate, expanded_store_predicate)
        ):
            continue
        if load_predicate is not None:
            if not _predicate_is_safe_for_fallback(
                load_predicate,
                immutable_definitions=immutable_definitions,
                range_definitions=definitions,
                uniform_names=uniform_names,
                uniform_scalar_load_names=uniform_scalar_load_names,
                write_counts=write_counts,
                column_axis=column_axis,
                vector_width=vector_width,
                thread_block_dims=thread_block_dims,
            ):
                continue
            if set(writes) & _name_dependency_closure(load_predicate, definitions):
                continue

            simplified_predicate = _strip_always_true_column_bounds(
                load_predicate,
                definitions=definitions,
                column_axis=column_axis,
                vector_width=vector_width,
                thread_block_dims=thread_block_dims,
            )
            if isinstance(simplified_predicate, ast.Constant) and (
                simplified_predicate.value is True
            ):
                load_pointer = replace(
                    load_pointer,
                    offset=load_pointer.active_offset,
                    predicate=None,
                )
            else:
                load_pointer = replace(
                    load_pointer,
                    offset=ast.IfExp(
                        test=simplified_predicate,
                        body=load_pointer.active_offset,
                        orelse=ast.Constant(value=0),
                    ),
                    predicate=simplified_predicate,
                )

        pointer_for_iteration = _substitute(load_pointer.offset, {row_name: row_rhs})
        dependencies = {
            node.id
            for node in ast.walk(pointer_for_iteration)
            if isinstance(node, ast.Name)
        } - {loop_var, tensor_name, "cutlass", "cute", "ir"}
        dependencies.update(
            node.id for node in ast.walk(row_rhs) if isinstance(node, ast.Name)
        )
        dependencies -= {loop_var, "cutlass", "cute", "ir"}
        if _assigned_names(loop_module) & dependencies:
            continue
        # Start the asynchronous copies as soon as their address dependencies
        # dominate, but remain in the loop's original control-flow branch and
        # never cross opaque/effectful statements.  This overlaps the state
        # reads with independent compiler-generated register preparation while
        # leaving the first wait and shared-memory load at the loop boundary.
        insertion_index = _earliest_prefetch_insertion_index(
            owner,
            loop_index,
            dependencies=dependencies,
            dominating_prefix=dominating_prefix,
            write_counts=write_counts,
            state_tensor=tensor_name,
            tensor_names=tensor_names,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
        )
        allowed_source_nodes = frozenset(
            id(node)
            for pointer_expr in (
                load_pointer.expression,
                store_pointer.expression,
            )
            for node in ast.walk(pointer_expr)
            if isinstance(node, ast.Name) and node.id == tensor_name
        )
        packed_scalar_load = _match_packed_scalar_load(
            loop=loop,
            loop_var=loop_var,
            trip_count=trip_count,
            state_tensor=tensor_name,
            root_body=root_body,
            lexical_prefix=lexical_prefix,
            reaching_definitions=definitions,
            write_counts=write_counts,
            uniform_names=uniform_names,
            constexpr_values=constexpr_values,
            tensor_metadata=tensor_metadata,
            tensor_names=tensor_names,
            proven_tensor_base_alignments=proven_tensor_base_alignments,
            proven_tensor_size_values=proven_tensor_size_values,
            proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
            proven_tensor_stride_values=proven_tensor_stride_values,
            thread_block_dims=thread_block_dims,
            block_grid_dims=block_grid_dims,
        )
        candidates.append(
            _Region(
                owner=owner,
                loop_index=loop_index,
                loop=loop,
                loop_var=loop_var,
                trip_count=trip_count,
                row_name=row_name,
                row_rhs=row_rhs,
                row_axis=row_axis,
                column_axis=column_axis,
                load_index=load_index,
                load_name=load_name,
                load_pointer=load_pointer,
                vector_type=vector_type,
                vector_width=vector_width,
                tensor_name=tensor_name,
                store_top_level=store_top_level,
                insertion_index=insertion_index,
                allowed_source_nodes=allowed_source_nodes,
                packed_scalar_load=packed_scalar_load,
            )
        )
    return candidates[0] if len(candidates) == 1 else None


def _find_regions(
    body: list[ast.stmt],
    tensor_metadata: dict[str, TensorMetadata],
    tensor_names: frozenset[str],
    proven_tensor_base_alignments: frozenset[str],
    proven_tensor_size_values: dict[tuple[str, int], tuple[str, int]],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int | None, int | None, int | None],
    write_counts: dict[str, int],
    uniform_names: frozenset[str],
    constexpr_values: dict[str, int],
    inherited_prefix: list[ast.stmt] | None = None,
    root_body: list[ast.stmt] | None = None,
) -> list[_Region]:
    prefix = list(inherited_prefix or [])
    root_body = body if root_body is None else root_body
    regions: list[_Region] = []
    for index, stmt in enumerate(body):
        lexical_prefix = [*prefix, *body[:index]]
        if isinstance(stmt, ast.For):
            immutable_definitions = _immutable_scalar_definitions(
                lexical_prefix, write_counts
            )
            match = _match_loop(
                body,
                index,
                lexical_prefix,
                prefix,
                root_body,
                _captured_definitions(lexical_prefix, write_counts),
                immutable_definitions,
                _immutable_trivial_aliases(lexical_prefix, write_counts),
                write_counts,
                uniform_names,
                constexpr_values,
                tensor_metadata,
                tensor_names,
                proven_tensor_base_alignments,
                proven_tensor_size_values,
                proven_disjoint_tensor_pairs,
                proven_tensor_stride_values,
                thread_block_dims,
                block_grid_dims,
            )
            if match is not None:
                regions.append(match)
        elif isinstance(stmt, ast.If):
            regions.extend(
                _find_regions(
                    stmt.body,
                    tensor_metadata,
                    tensor_names,
                    proven_tensor_base_alignments,
                    proven_tensor_size_values,
                    proven_disjoint_tensor_pairs,
                    proven_tensor_stride_values,
                    thread_block_dims,
                    block_grid_dims,
                    write_counts,
                    uniform_names,
                    constexpr_values,
                    lexical_prefix,
                    root_body,
                )
            )
            regions.extend(
                _find_regions(
                    stmt.orelse,
                    tensor_metadata,
                    tensor_names,
                    proven_tensor_base_alignments,
                    proven_tensor_size_values,
                    proven_disjoint_tensor_pairs,
                    proven_tensor_stride_values,
                    thread_block_dims,
                    block_grid_dims,
                    write_counts,
                    uniform_names,
                    constexpr_values,
                    lexical_prefix,
                    root_body,
                )
            )
    return regions


def _raw_pointer_for_iteration(region: _Region, iteration_name: str) -> str:
    row = _substitute(
        region.row_rhs,
        {region.loop_var: ast.Name(id=iteration_name, ctx=ast.Load())},
    )
    offset = _substitute(region.load_pointer.offset, {region.row_name: row})
    return (
        f"{region.tensor_name}.iterator.raw_ptr() + "
        f"({ast.unparse(ast.fix_missing_locations(offset))})"
    )


def _parse_statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def _wait_code(prefix: str, total_chunks: int, lookahead: int) -> str:
    maximum = lookahead - 2
    full_until = total_chunks - lookahead
    lines = [f"if {prefix}chunk <= {full_until}:"]
    lines.append(f"    cute.arch.cp_async_wait_group({maximum})")
    for target in range(maximum - 1, -1, -1):
        chunk = total_chunks - target - 2
        lines.extend(
            (
                f"elif {prefix}chunk == {chunk}:",
                f"    cute.arch.cp_async_wait_group({target})",
            )
        )
    return "\n".join(lines)


def _packed_scalar_preload_statements(
    packed: _PackedScalarLoad, *, loop_var: str, trip_count: int, prefix: str
) -> list[ast.stmt]:
    pair_name = f"{prefix}scalar_pair"
    pair_expr = ast.Name(id=pair_name, ctx=ast.Load())
    low_iteration = ast.BinOp(
        left=pair_expr, op=ast.Mult(), right=ast.Constant(value=2)
    )
    high_iteration = ast.BinOp(
        left=low_iteration, op=ast.Add(), right=ast.Constant(value=1)
    )
    low_offset = _substitute(packed.pointer.active_offset, {loop_var: low_iteration})
    load = (
        f"cute.recast_ptr({packed.tensor_name}.iterator + "
        f"({ast.unparse(ast.fix_missing_locations(low_offset))}), "
        "dtype=cutlass.Uint32).load()"
    )
    cache = f"{prefix}scalar_pairs"
    if packed.predicate is None:
        return _parse_statements(
            f"""
for {pair_name} in cutlass.range_constexpr({trip_count // 2}):
    {cache}[{pair_name}] = {load}
"""
        )
    low_predicate = _substitute(packed.predicate, {loop_var: low_iteration})
    high_predicate = _substitute(packed.predicate, {loop_var: high_iteration})
    return _parse_statements(
        f"""
for {pair_name} in cutlass.range_constexpr({trip_count // 2}):
    {cache}[{pair_name}] = cutlass.Uint32(0)
    if ({ast.unparse(ast.fix_missing_locations(low_predicate))}) and ({ast.unparse(ast.fix_missing_locations(high_predicate))}):
        {cache}[{pair_name}] = {load}
"""
    )


def _rewrite_packed_scalar_loads(
    loop: ast.For, packed: _PackedScalarLoad, *, loop_var: str, prefix: str
) -> None:
    pair_base = ast.BinOp(
        left=ast.BinOp(
            left=ast.Name(id=loop_var, ctx=ast.Load()),
            op=ast.FloorDiv(),
            right=ast.Constant(value=2),
        ),
        op=ast.Mult(),
        right=ast.Constant(value=2),
    )
    pair_high = ast.BinOp(left=pair_base, op=ast.Add(), right=ast.Constant(value=1))
    pair_predicate: ast.expr | None = None
    if packed.predicate is not None:
        pair_predicate = ast.BoolOp(
            op=ast.And(),
            values=[
                _substitute(packed.predicate, {loop_var: pair_base}),
                _substitute(packed.predicate, {loop_var: pair_high}),
            ],
        )
    for site, (statement, original) in enumerate(
        zip(packed.statements, packed.original_values, strict=True)
    ):
        bits_name = f"{prefix}scalar_pair_bits_{site}"
        packed_value = (
            "cutlass.Uint16("
            f"{bits_name} if {loop_var} % 2 == 0 else "
            f"{bits_name} >> cutlass.Uint32(16)"
            ").bitcast(cutlass.BFloat16)"
        )
        value = packed_value
        if pair_predicate is not None:
            value = (
                f"({packed_value} if "
                f"{ast.unparse(ast.fix_missing_locations(pair_predicate))} "
                f"else {ast.unparse(original)})"
            )
        replacement = _parse_statements(
            f"""
{bits_name} = {prefix}scalar_pairs[{loop_var} // 2]
{packed.target_name} = {value}
"""
        )
        index = loop.body.index(statement)
        loop.body[index : index + 1] = replacement


def _rewrite_region(
    body: list[ast.stmt],
    region: _Region,
    *,
    stages: int,
    lookahead: int,
    group_rows: int,
    cache_policy: str,
    store_policy: str,
    thread_block_dims: tuple[int, int, int],
    suffix: int,
) -> None:
    prefix = f"_async_state_{suffix}_"
    total_threads = thread_block_dims[0] * thread_block_dims[1] * thread_block_dims[2]
    total_chunks = region.trip_count // group_rows
    shared_elements = stages * total_threads * group_rows * region.vector_width
    stage_elements = total_threads * group_rows * region.vector_width
    column_extent = thread_block_dims[region.column_axis] * region.vector_width
    copy_bytes = region.vector_width * 2
    raw_pointer = _raw_pointer_for_iteration(region, f"{prefix}iteration")

    allocations = [
        statement_from_string(
            f"{prefix}smem = cutlass.Array(cutlass.Uint16, {shared_elements}, "
            "space=cutlass.AddressSpace.smem, alignment=128)"
        )
    ]
    if store_policy == "l2_evict_last":
        allocations.append(
            statement_from_string(
                f"{prefix}l2_store_abi_version = {L2_EVICT_LAST_STORE_ABI_VERSION}"
            )
        )
    if region.packed_scalar_load is not None:
        allocations.append(
            statement_from_string(
                f"{prefix}scalar_pairs = cute.make_rmem_tensor("
                f"{region.trip_count // 2}, cutlass.Uint32)"
            )
        )
    body[0:0] = allocations

    # Root-owned region indices move with the allocations. Nested branch
    # owners are unaffected.
    if region.owner is body:
        region = replace(
            region,
            loop_index=region.loop_index + len(allocations),
            insertion_index=region.insertion_index + len(allocations),
        )

    prologue = _parse_statements(
        f"""
{prefix}row_thread = cutlass.Int32(cute.arch.thread_idx()[{region.row_axis}])
{prefix}column = (
    cutlass.Int32(cute.arch.thread_idx()[{region.column_axis}]) * {region.vector_width}
)
for {prefix}prefetch_chunk in cutlass.range_constexpr({lookahead}):
    for {prefix}row_local in cutlass.range_constexpr({group_rows}):
        {prefix}iteration = (
            {prefix}prefetch_chunk * {group_rows} + {prefix}row_local
        )
        {prefix}destination = (
            ({prefix}prefetch_chunk % {stages}) * {stage_elements}
            + ({prefix}row_thread * {group_rows} + {prefix}row_local) * {column_extent}
            + {prefix}column
        )
        cute.arch.cp_async_shared_global(
            {prefix}smem.data_ptr() + {prefix}destination,
            {raw_pointer},
            {copy_bytes},
            {cache_policy!r},
        )
    cute.arch.cp_async_commit_group()
"""
    )
    region.owner[region.insertion_index : region.insertion_index] = prologue
    loop_index = region.loop_index + len(prologue)

    initial_wait = max(lookahead - 2, 0)
    scalar_preload = (
        []
        if region.packed_scalar_load is None
        else _packed_scalar_preload_statements(
            region.packed_scalar_load,
            loop_var=region.loop_var,
            trip_count=region.trip_count,
            prefix=prefix,
        )
    )
    prefix_before_loop = _parse_statements(
        f"""
cute.arch.cp_async_wait_group({initial_wait})
{prefix}prefetch_chunk = {lookahead}
for {prefix}row_local in cutlass.range_constexpr({group_rows}):
    {prefix}iteration = (
        {prefix}prefetch_chunk * {group_rows} + {prefix}row_local
    )
    {prefix}destination = (
        ({prefix}prefetch_chunk % {stages}) * {stage_elements}
        + ({prefix}row_thread * {group_rows} + {prefix}row_local) * {column_extent}
        + {prefix}column
    )
    cute.arch.cp_async_shared_global(
        {prefix}smem.data_ptr() + {prefix}destination,
        {raw_pointer},
        {copy_bytes},
        {cache_policy!r},
    )
cute.arch.cp_async_commit_group()
{prefix}values = cute.arch.load(
    {prefix}smem.data_ptr()
    + {prefix}row_thread * {group_rows * column_extent}
    + {prefix}column,
    {ast.unparse(region.vector_type)},
)
"""
    )
    region.owner[loop_index:loop_index] = [*scalar_preload, *prefix_before_loop]
    loop_index += len(scalar_preload) + len(prefix_before_loop)
    loop = region.owner[loop_index]
    assert loop is region.loop

    wait_code = _wait_code(prefix, total_chunks, lookahead)
    indented_wait = textwrap.indent(wait_code, "    ")
    loop_prefix = _parse_statements(
        f"""
{prefix}chunk = {region.loop_var} // {group_rows}
{prefix}row_local = {region.loop_var} % {group_rows}
if {prefix}row_local == 0 and {prefix}chunk > 0:
{indented_wait}
    {prefix}prefetch_chunk = {prefix}chunk + {lookahead}
    if {prefix}prefetch_chunk < {total_chunks}:
        for {prefix}prefetch_row in cutlass.range_constexpr({group_rows}):
            {prefix}iteration = (
                {prefix}prefetch_chunk * {group_rows} + {prefix}prefetch_row
            )
            {prefix}destination = (
                ({prefix}prefetch_chunk % {stages}) * {stage_elements}
                + ({prefix}row_thread * {group_rows} + {prefix}prefetch_row) * {column_extent}
                + {prefix}column
            )
            cute.arch.cp_async_shared_global(
                {prefix}smem.data_ptr() + {prefix}destination,
                {raw_pointer},
                {copy_bytes},
                {cache_policy!r},
            )
        cute.arch.cp_async_commit_group()
"""
    )
    load_replacement = _parse_statements(
        f"""
{region.load_name} = {prefix}values
if {region.loop_var} + 1 < {region.trip_count}:
    {prefix}next_lane = {region.loop_var} + 1
    {prefix}next_chunk = {prefix}next_lane // {group_rows}
    {prefix}next_row = {prefix}next_lane % {group_rows}
    {prefix}next_offset = (
        ({prefix}next_chunk % {stages}) * {stage_elements}
        + ({prefix}row_thread * {group_rows} + {prefix}next_row) * {column_extent}
        + {prefix}column
    )
    {prefix}values = cute.arch.load(
        {prefix}smem.data_ptr() + {prefix}next_offset,
        {ast.unparse(region.vector_type)},
    )
"""
    )
    loop.body[0:0] = loop_prefix
    load_index = region.load_index + len(loop_prefix)
    loop.body[load_index : load_index + 1] = load_replacement
    if region.packed_scalar_load is not None:
        _rewrite_packed_scalar_loads(
            loop,
            region.packed_scalar_load,
            loop_var=region.loop_var,
            prefix=prefix,
        )

    # Preserve the exact state-vector provenance for later arithmetic
    # lowering.  These attributes live only on the generated AST and are
    # consumed in the same compilation; they never become runtime state.
    loop.__dict__["_helion_async_state_vector"] = region.load_name
    loop.__dict__["_helion_async_state_vector_width"] = region.vector_width
    loop.__dict__["_helion_async_state_valid_predicate"] = region.load_pointer.predicate
    loop.__dict__["_helion_async_state_tensor"] = region.tensor_name

    if store_policy == "l2_evict_last":
        store_stmt = region.store_top_level
        if isinstance(store_stmt, ast.If):
            assert len(store_stmt.body) == 1
            store_stmt = store_stmt.body[0]
        assert isinstance(store_stmt, ast.Expr)
        store_call = store_stmt.value
        assert isinstance(store_call, ast.Call)
        assert isinstance(store_call.func, ast.Name)
        assert store_call.func.id == "_cute_store_u16_vec"
        store_call.func.id = "_cute_store_u16x8_l2_evict_last"

    # The matcher proved that this is a finite, literal-trip-count loop with no
    # early exits or loop-variable writes.  Preserve that proof in the emitted
    # CuTe DSL so tracing can specialize every pipeline step (wait depth, ring
    # slot, and next-row preload) instead of lowering a rolled runtime loop.
    loop.iter = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="cutlass", ctx=ast.Load()),
            attr="range_constexpr",
            ctx=ast.Load(),
        ),
        args=[ast.Constant(value=region.trip_count)],
        keywords=[],
    )


def pipeline_state_loads(
    body: list[ast.stmt],
    *,
    stages: int,
    lookahead: int,
    group_rows: int,
    cache_policy: str,
    thread_block_dims: tuple[int, int, int],
    tensor_metadata: dict[str, TensorMetadata],
    tensor_names: frozenset[str],
    proven_tensor_base_alignments: frozenset[str],
    proven_tensor_size_values: dict[tuple[str, int], tuple[str, int]],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    block_grid_dims: tuple[int | None, int | None, int | None],
    constexpr_values: dict[str, int],
    uniform_names: frozenset[str],
    target_device_capability: tuple[int, int] | None,
    store_policy: str = "default",
) -> list[ast.stmt]:
    """Apply the exact in-place state-load ring when one region is proven safe."""
    if stages == 0 or not cp_async_supported(target_device_capability):
        return body
    if (
        store_policy == "l2_evict_last"
        and not fixed_l2_evict_last_store_policy_supported(
            target_device_capability, torch.version.cuda
        )
    ):
        return body
    if (
        stages < 3
        or lookahead < 2
        or lookahead >= stages
        or group_rows < 2
        or cache_policy not in {"cg", "ca"}
        or store_policy not in {"default", "l2_evict_last"}
        or any(dim <= 0 for dim in thread_block_dims)
        or any(
            isinstance(node, ast.Name) and node.id.startswith("_async_state_")
            for stmt in body
            for node in ast.walk(stmt)
        )
    ):
        return body
    module = ast.Module(body=body, type_ignores=[])
    regions = _find_regions(
        body,
        tensor_metadata,
        tensor_names,
        proven_tensor_base_alignments,
        proven_tensor_size_values,
        proven_disjoint_tensor_pairs,
        proven_tensor_stride_values,
        thread_block_dims,
        block_grid_dims,
        _name_write_counts(module),
        uniform_names,
        constexpr_values,
    )
    if len(regions) != 1:
        return body
    region = regions[0]
    shared_bytes = (
        stages
        * thread_block_dims[0]
        * thread_block_dims[1]
        * thread_block_dims[2]
        * group_rows
        * region.vector_width
        * 2
    )
    if (
        region.trip_count % group_rows
        or region.trip_count // group_rows <= lookahead
        or shared_bytes > 48 * 1024
        or not _only_allowed_source_uses(
            body, region.tensor_name, region.allowed_source_nodes
        )
    ):
        return body
    _rewrite_region(
        body,
        region,
        stages=stages,
        lookahead=lookahead,
        group_rows=group_rows,
        cache_policy=cache_policy,
        store_policy=store_policy,
        thread_block_dims=thread_block_dims,
        suffix=0,
    )
    ast.fix_missing_locations(module)
    return body
