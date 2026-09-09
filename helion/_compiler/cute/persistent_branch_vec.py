"""Late branch-local vectorization for exact persistent-reduction fragments.

Persistent reduction lanes wrap the whole grid body.  A load whose address is
defined inside a runtime branch therefore cannot use the normal wrapper-level
vector hoist.  Memory lowering marks only sites that already passed the
stride-1, dtype, and exact-fragment checks; this pass places their vector
transaction immediately outside the materialized constexpr lane loop, where
branch-local indices dominate it.
"""

from __future__ import annotations

import ast
import math
import re
from typing import TYPE_CHECKING

from ..ast_read_writes import ReadWrites

if TYPE_CHECKING:
    from collections.abc import Mapping

_LOAD_MARKER = "_helion_persistent_branch_vec_load"
_STORE_MARKER = "_helion_persistent_branch_vec_store"
_SYNTHETIC_LANE_RE = re.compile(r"synthetic_lane_(\d+)")
_DTYPE_INFO = {
    "cutlass.BFloat16": ("cutlass.Uint16", 2, "_cute_store_u16_vec"),
    "cutlass.Float16": ("cutlass.Uint16", 2, "_cute_store_u16_vec"),
    "cutlass.Float32": ("cutlass.Uint32", 4, "_cute_store_u32_vec"),
}


def _clone_expr(node: ast.AST) -> ast.expr:
    """Clone generated/extended AST without relying on ``deepcopy``."""
    return ast.parse(ast.unparse(node), mode="eval").body


def _marker_call(node: ast.AST, name: str, nargs: int) -> ast.Call | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and len(node.args) == nargs
        and not node.keywords
    ):
        return node
    return None


def _literal(node: ast.AST, typ: type[int | str]) -> int | str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, typ):
        return node.value
    return None


def _range_extent(loop: ast.For) -> int | None:
    call = loop.iter
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "range_constexpr"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "cutlass"
        and len(call.args) == 1
        and not call.keywords
    ):
        return None
    value = _literal(call.args[0], int)
    return value if isinstance(value, int) and value > 1 else None


def _single_name_definitions(statements: list[ast.stmt]) -> dict[str, ast.expr]:
    definitions: dict[str, ast.expr] = {}
    for stmt in statements:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            definitions[stmt.targets[0].id] = stmt.value
    return definitions


def _binding_write_roots(node: ast.AST) -> set[str]:
    """Return names whose bindings or reachable values may be mutated."""

    def root_name(target: ast.AST) -> str | None:
        while isinstance(target, (ast.Attribute, ast.Subscript)):
            target = target.value
        return target.id if isinstance(target, ast.Name) else None

    result: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            result.add(child.id)
        elif isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(
            child.ctx, (ast.Store, ast.Del)
        ):
            if (name := root_name(child)) is not None:
                result.add(name)
        elif isinstance(child, ast.ExceptHandler) and child.name is not None:
            result.add(child.name)
    return result


def _mutated_value_roots(node: ast.AST) -> set[str]:
    """Return base names mutated through an attribute or subscript."""

    def root_name(target: ast.AST) -> str | None:
        while isinstance(target, (ast.Attribute, ast.Subscript)):
            target = target.value
        return target.id if isinstance(target, ast.Name) else None

    return {
        name
        for child in ast.walk(node)
        if isinstance(child, (ast.Attribute, ast.Subscript))
        and isinstance(child.ctx, (ast.Store, ast.Del))
        and (name := root_name(child)) is not None
    }


def _freeze_definition(
    node: ast.expr,
    definitions: dict[str, ast.expr],
) -> ast.expr:
    """Inline known values so a definition keeps assignment-time semantics."""

    class _FreezeNames(ast.NodeTransformer):
        def __init__(self) -> None:
            super().__init__()
            self.expanding: set[str] = set()

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if not isinstance(node.ctx, ast.Load) or node.id in self.expanding:
                return node
            definition = definitions.get(node.id)
            if definition is None:
                return node
            self.expanding.add(node.id)
            replacement = self.visit(_clone_expr(definition))
            self.expanding.remove(node.id)
            return ast.copy_location(replacement, node)

    result = _FreezeNames().visit(_clone_expr(node))
    assert isinstance(result, ast.expr)
    return result


def _definition_snapshots(
    statements: list[ast.stmt],
) -> list[dict[str, ast.expr]]:
    """Return the definitions dominating each top-level statement."""
    definitions: dict[str, ast.expr] = {}
    result: list[dict[str, ast.expr]] = []
    for statement in statements:
        result.append(dict(definitions))
        exact_definitions = _single_name_definitions([statement])
        # Definitions retain source expressions rather than assignment-time
        # values.  Invalidate every definition that transitively depends on a
        # written name before recording the new exact definitions.  Otherwise
        # a later reassignment could make two historically different address
        # aliases expand to the same current expression.  Unmodelled writes
        # (AugAssign, structured assignment, and control flow) also kill the
        # name itself because they have no unconditional post-value to inline.
        invalidated = {
            *ReadWrites.from_ast(statement).writes,
            *_binding_write_roots(statement),
        }
        # A structured write through ``alias`` also mutates the object named
        # by ``alias = original``.  Frozen definitions retain that referent as
        # a read, so invalidate it (and all definitions depending on it) too.
        # This is deliberately conservative for non-name RHS expressions.
        for name in _mutated_value_roots(statement):
            definition = definitions.get(name)
            if definition is not None:
                invalidated.update(ReadWrites.from_ast(definition).reads)
        while invalidated:
            newly_invalidated: set[str] = set()
            for name, value in tuple(definitions.items()):
                if name in invalidated or (
                    set(ReadWrites.from_ast(value).reads) & invalidated
                ):
                    definitions.pop(name)
                    newly_invalidated.add(name)
            invalidated = newly_invalidated
        definitions.update(
            {
                name: _freeze_definition(value, definitions)
                for name, value in exact_definitions.items()
            }
        )
    return result


def _unstable_address_names(
    body: list[ast.stmt], lane_var: str, load_index: int, store_index: int
) -> set[str]:
    """Names whose value may differ across a load/store loop backedge."""
    definitely_written: set[str] = set()
    live_in: set[str] = set()
    may_writes: set[str] = set()
    for statement in body:
        effects = ReadWrites.from_ast(statement)
        live_in.update(set(effects.reads) - definitely_written - {lane_var})
        may_writes.update(effects.writes)
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            definitely_written.update(effects.writes)
    result = live_in & may_writes
    if store_index > load_index:
        for crossed_statement in body[load_index + 1 : store_index + 1]:
            result.update(ReadWrites.from_ast(crossed_statement).writes)
    return result


class _CollectMemoryLoads(ast.NodeVisitor):
    """Collect logical loads without double-counting a marker's scalar fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        if _marker_call(node, _LOAD_MARKER, 6) is not None:
            self.calls.append(node)
            return
        if isinstance(node.func, ast.Attribute) and node.func.attr == "load":
            self.calls.append(node)
            return
        self.generic_visit(node)


def _memory_load_calls(node: ast.AST) -> list[ast.Call]:
    collector = _CollectMemoryLoads()
    collector.visit(node)
    return collector.calls


def _lane_dependent_alias_expansion(
    node: ast.expr,
    definitions: dict[str, ast.expr],
    lane_var: str,
) -> ast.expr:
    dependency_cache: dict[str, bool] = {}
    resolving: set[str] = set()

    def depends_on_lane(name: str) -> bool:
        if name == lane_var:
            return True
        cached = dependency_cache.get(name)
        if cached is not None:
            return cached
        definition = definitions.get(name)
        if definition is None or name in resolving:
            dependency_cache[name] = False
            return False
        resolving.add(name)
        result = any(
            depends_on_lane(read) for read in ReadWrites.from_ast(definition).reads
        )
        resolving.remove(name)
        dependency_cache[name] = result
        return result

    class _ExpandLaneAliases(ast.NodeTransformer):
        def __init__(self) -> None:
            super().__init__()
            self.expanding: set[str] = set()

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if (
                not isinstance(node.ctx, ast.Load)
                or node.id == lane_var
                or not depends_on_lane(node.id)
                or node.id in self.expanding
            ):
                return node
            definition = definitions.get(node.id)
            if definition is None:
                return node
            self.expanding.add(node.id)
            replacement = self.visit(_clone_expr(definition))
            self.expanding.remove(node.id)
            return ast.copy_location(replacement, node)

    expanded = _ExpandLaneAliases().visit(_clone_expr(node))
    assert isinstance(expanded, ast.expr)
    return expanded


def _is_affine_in_lane(node: ast.AST, lane_var: str) -> tuple[bool, bool]:
    """Return ``(depends_on_lane, is_affine)`` for pointer arithmetic."""
    if isinstance(node, ast.Name):
        return node.id == lane_var, True
    if isinstance(node, ast.UnaryOp):
        depends, valid = _is_affine_in_lane(node.operand, lane_var)
        return depends, valid and isinstance(node.op, (ast.UAdd, ast.USub))
    if isinstance(node, ast.BinOp):
        left_depends, left_valid = _is_affine_in_lane(node.left, lane_var)
        right_depends, right_valid = _is_affine_in_lane(node.right, lane_var)
        depends = left_depends or right_depends
        if not depends:
            return False, left_valid and right_valid
        if not left_valid or not right_valid or left_depends and right_depends:
            return True, False
        if isinstance(node.op, ast.Add):
            return True, True
        if isinstance(node.op, ast.Sub):
            return True, left_depends
        if isinstance(node.op, ast.Mult):
            return True, True
        return True, False
    if isinstance(node, ast.Call):
        dependencies = [_is_affine_in_lane(arg, lane_var) for arg in node.args]
        depends = any(item[0] for item in dependencies)
        if not depends:
            return False, all(item[1] for item in dependencies)
        is_cutlass_cast = (
            len(node.args) == 1
            and not node.keywords
            and ast.unparse(node.func).startswith("cutlass.")
        )
        return True, is_cutlass_cast and all(item[1] for item in dependencies)
    dependencies = [
        _is_affine_in_lane(child, lane_var) for child in ast.iter_child_nodes(node)
    ]
    depends = any(item[0] for item in dependencies)
    return depends, not depends and all(item[1] for item in dependencies)


def _plain_scalar_load_pointer(call: ast.Call) -> ast.expr | None:
    """Return the pointer of a generated scalar ``PTR.load()`` call."""
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "load"
        and not call.args
        and not call.keywords
    ):
        return call.func.value
    return None


def _plain_scalar_store_pointer(call: ast.Call) -> ast.expr | None:
    """Return the pointer of a generated scalar ``PTR.store(value)`` call."""
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "store"
        and len(call.args) == 1
        and not call.keywords
    ):
        return call.func.value
    return None


def _single_tensor_iterator_root(node: ast.AST) -> str | None:
    roots = {
        child.value.id
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and child.attr == "iterator"
        and isinstance(child.value, ast.Name)
    }
    return next(iter(roots)) if len(roots) == 1 else None


def _tensor_stride_reference(node: ast.AST) -> tuple[str, int] | None:
    while (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) in ("cutlass.Int32", "cutlass.Int64")
    ):
        node = node.args[0]
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "stride"
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == "layout"
        and isinstance(node.value.value.value, ast.Name)
    ):
        return None
    return node.value.value.value.id, node.slice.value


def _lane_scale_value(
    node: ast.AST,
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
) -> int | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _lane_scale_value(node.operand, proven_tensor_stride_values)
        if value is None:
            return None
        return value if isinstance(node.op, ast.UAdd) else -value
    if (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) in ("cutlass.Int32", "cutlass.Int64")
    ):
        return _lane_scale_value(node.args[0], proven_tensor_stride_values)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult)
    ):
        left = _lane_scale_value(node.left, proven_tensor_stride_values)
        right = _lane_scale_value(node.right, proven_tensor_stride_values)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    stride = _tensor_stride_reference(node)
    return proven_tensor_stride_values.get(stride) if stride is not None else None


def _scalar_pointer_calls_are_known(node: ast.AST) -> bool:
    """Whether every call has transparent integer-coordinate semantics."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = ast.unparse(child.func)
        if name in ("cutlass.Int32", "cutlass.Int64"):
            if len(child.args) != 1 or child.keywords:
                return False
            continue
        if name in (
            "cute.arch.block_idx",
            "cute.arch.thread_idx",
            "cute.arch.warp_idx",
        ):
            if child.args or child.keywords:
                return False
            continue
        return False
    return True


def _scalar_pointer_is_known_affine(node: ast.AST, lane_var: str) -> bool:
    """Reject pointer expressions whose affine semantics are not transparent."""
    if not _scalar_pointer_calls_are_known(node):
        return False
    depends, affine = _is_affine_in_lane(node, lane_var)
    return depends and affine


def _scalar_lane_mapping_is_injective(
    node: ast.AST,
    lane_var: str,
    lane_extent: int,
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
) -> bool:
    """Prove distinct lanes produce distinct scalar pointer offsets."""

    def visit(expr: ast.AST) -> tuple[bool, int | None]:
        if isinstance(expr, ast.Name):
            return (True, 1) if expr.id == lane_var else (False, 0)
        if isinstance(expr, ast.UnaryOp):
            depends, coefficient = visit(expr.operand)
            if not isinstance(expr.op, (ast.UAdd, ast.USub)):
                return depends, None if depends else 0
            if coefficient is None:
                return depends, None
            return depends, coefficient if isinstance(
                expr.op, ast.UAdd
            ) else -coefficient
        if isinstance(expr, ast.BinOp):
            left_depends, left_coefficient = visit(expr.left)
            right_depends, right_coefficient = visit(expr.right)
            if not left_depends and not right_depends:
                return False, 0
            if left_depends and right_depends:
                return True, None
            coefficient = left_coefficient if left_depends else right_coefficient
            if coefficient is None:
                return True, None
            if isinstance(expr.op, ast.Add):
                return True, coefficient
            if isinstance(expr.op, ast.Sub):
                return True, coefficient if left_depends else -coefficient
            if isinstance(expr.op, ast.Mult):
                scale = expr.right if left_depends else expr.left
                scale_value = _lane_scale_value(scale, proven_tensor_stride_values)
                return (
                    True,
                    None if scale_value is None else coefficient * scale_value,
                )
            return True, None
        if isinstance(expr, ast.Call):
            dependencies = [visit(arg) for arg in expr.args]
            depends = any(item[0] for item in dependencies)
            if not depends:
                return False, 0
            is_integer_cast = (
                len(expr.args) == 1
                and not expr.keywords
                and ast.unparse(expr.func) in ("cutlass.Int32", "cutlass.Int64")
            )
            return True, dependencies[0][1] if is_integer_cast else None
        dependencies = [visit(child) for child in ast.iter_child_nodes(expr)]
        depends = any(item[0] for item in dependencies)
        return depends, 0 if not depends else None

    depends, coefficient = visit(node)
    if not depends or coefficient in (None, 0):
        return False
    # Generated scalar indices may narrow through Int32.  Checking every
    # finite lane delta modulo 2**32 proves injectivity even when a known
    # coefficient itself wraps; it is conservative for wider arithmetic.
    return all(
        (coefficient * delta) % (1 << 32) != 0 for delta in range(1, lane_extent)
    )


def _pointer_integer_affine_form(
    node: ast.AST,
    lane_var: str,
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    unstable_names: set[str],
    site: str,
) -> tuple[int, dict[str, int], int] | None:
    """Return ``(lane coefficient, opaque coefficients, constant)``.

    Generated pointer offsets are integer-affine combinations of coordinates
    and exact tensor strides.  Lane-independent coordinates need not be equal
    at two accesses: retaining them as opaque integer atoms lets the caller
    prove that every possible base difference is a multiple of a safe row
    stride.  Unknown tensor strides and lane-dependent nonlinear expressions
    fail closed.
    """

    def add_forms(
        left: tuple[int, dict[str, int], int],
        right: tuple[int, dict[str, int], int],
        sign: int = 1,
    ) -> tuple[int, dict[str, int], int]:
        lane = left[0] + sign * right[0]
        opaque = dict(left[1])
        for key, coefficient in right[1].items():
            updated = opaque.get(key, 0) + sign * coefficient
            if updated:
                opaque[key] = updated
            else:
                opaque.pop(key, None)
        return lane, opaque, left[2] + sign * right[2]

    def scale_form(
        form: tuple[int, dict[str, int], int], coefficient: int
    ) -> tuple[int, dict[str, int], int]:
        return (
            form[0] * coefficient,
            {
                key: value * coefficient
                for key, value in form[1].items()
                if value * coefficient
            },
            form[2] * coefficient,
        )

    def opaque_atom(expr: ast.AST) -> tuple[int, dict[str, int], int] | None:
        if lane_var in ReadWrites.from_ast(expr).reads:
            return None
        if not _scalar_pointer_calls_are_known(expr):
            return None
        key = ast.dump(expr, include_attributes=False)
        if set(ReadWrites.from_ast(expr).reads) & unstable_names:
            key = f"{site}:{key}"
        return 0, {key: 1}, 0

    def visit(expr: ast.AST) -> tuple[int, dict[str, int], int] | None:
        stride = _tensor_stride_reference(expr)
        if stride is not None:
            value = proven_tensor_stride_values.get(stride)
            return None if value is None else (0, {}, value)
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, int) and not isinstance(expr.value, bool):
                return 0, {}, expr.value
            return None
        if isinstance(expr, ast.Name):
            if expr.id == lane_var:
                return 1, {}, 0
            return opaque_atom(expr)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            operand = visit(expr.operand)
            if operand is None:
                return None
            return operand if isinstance(expr.op, ast.UAdd) else scale_form(operand, -1)
        if (
            isinstance(expr, ast.Call)
            and len(expr.args) == 1
            and not expr.keywords
            and ast.unparse(expr.func) in ("cutlass.Int32", "cutlass.Int64")
        ):
            return visit(expr.args[0])
        if isinstance(expr, ast.BinOp) and isinstance(
            expr.op, (ast.Add, ast.Sub, ast.Mult)
        ):
            if isinstance(expr.op, ast.Mult):
                left_constant = _lane_scale_value(
                    expr.left, proven_tensor_stride_values
                )
                right_constant = _lane_scale_value(
                    expr.right, proven_tensor_stride_values
                )
                if left_constant is not None:
                    right = visit(expr.right)
                    return None if right is None else scale_form(right, left_constant)
                if right_constant is not None:
                    left = visit(expr.left)
                    return None if left is None else scale_form(left, right_constant)
                return None
            left = visit(expr.left)
            right = visit(expr.right)
            if left is None or right is None:
                return None
            return add_forms(left, right, -1 if isinstance(expr.op, ast.Sub) else 1)
        return opaque_atom(expr)

    return visit(node)


def _scalar_pointers_are_modularly_lane_disjoint(
    load_pointer: ast.expr,
    store_pointer: ast.expr,
    *,
    lane_var: str,
    lane_extent: int,
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    unstable_names: set[str],
) -> bool:
    """Prove cross-lane disjointness despite different row/checkpoint bases.

    If both pointers advance by the same nonzero lane step and every possible
    difference between their lane-independent bases is a multiple of ``M``, a
    cross-iteration dependence is impossible when ``step * delta`` is nonzero
    modulo ``M`` for every live lane delta.  Including the Int32 modulus keeps
    the proof sound under generated index narrowing.
    """
    load_root = _single_tensor_iterator_root(load_pointer)
    store_root = _single_tensor_iterator_root(store_pointer)
    if load_root is None or load_root != store_root:
        return False
    if not _scalar_pointer_calls_are_known(
        load_pointer
    ) or not _scalar_pointer_calls_are_known(store_pointer):
        return False
    load_form = _pointer_integer_affine_form(
        load_pointer,
        lane_var,
        proven_tensor_stride_values,
        unstable_names,
        "load",
    )
    store_form = _pointer_integer_affine_form(
        store_pointer,
        lane_var,
        proven_tensor_stride_values,
        unstable_names,
        "store",
    )
    if load_form is None or store_form is None:
        return False
    lane_coefficient = load_form[0]
    if lane_coefficient == 0 or lane_coefficient != store_form[0]:
        return False
    base_coefficients = dict(load_form[1])
    for key, coefficient in store_form[1].items():
        updated = base_coefficients.get(key, 0) - coefficient
        if updated:
            base_coefficients[key] = updated
        else:
            base_coefficients.pop(key, None)
    modulus = 1 << 32
    modulus = math.gcd(modulus, abs(load_form[2] - store_form[2]))
    for coefficient in base_coefficients.values():
        modulus = math.gcd(modulus, abs(coefficient))
    return modulus > 1 and all(
        (lane_coefficient * delta) % modulus != 0 for delta in range(1, lane_extent)
    )


def _same_stable_address_aliases(
    node: ast.AST,
    load_definitions: dict[str, ast.expr],
    store_definitions: dict[str, ast.expr],
    lane_var: str,
    unstable_names: set[str],
) -> bool:
    """Ensure opaque lane-independent locals mean the same value at both sites."""
    visiting: set[str] = set()
    checked: set[str] = set()

    def check(name: str) -> bool:
        if name == lane_var or name in checked:
            return True
        if name in unstable_names:
            return False
        load_definition = load_definitions.get(name)
        store_definition = store_definitions.get(name)
        if load_definition is None or store_definition is None:
            return load_definition is store_definition
        if name in visiting:
            return False
        if ast.dump(load_definition, include_attributes=False) != ast.dump(
            store_definition, include_attributes=False
        ):
            return False
        if not _scalar_pointer_calls_are_known(load_definition):
            return False
        visiting.add(name)
        result = all(
            check(dependency)
            for dependency in ReadWrites.from_ast(load_definition).reads
        )
        visiting.remove(name)
        if result:
            checked.add(name)
        return result

    return all(
        check(name) for name in ReadWrites.from_ast(node).reads if name != lane_var
    )


def _lane_accesses_are_iteration_independent(
    load_call: ast.Call,
    store_call: ast.Call,
    *,
    lane_var: str,
    lane_extent: int | None,
    load_definitions: dict[str, ast.expr],
    store_definitions: dict[str, ast.expr],
    proven_tensor_stride_values: Mapping[tuple[str, int], int] | None = None,
    loop_carried_names: set[str] | None = None,
) -> bool:
    """Prove an exact load/store pair cannot communicate across lane iterations.

    Persistent vector markers prove injectivity through their matching exact-
    fragment metadata.  Plain scalar accesses instead require one known tensor
    iterator, stable address aliases, an identical affine pointer function, and
    an injective lane step backed by literals and cache-safe exact tensor-stride
    values.  Both paths preserve same-iteration load-before-store order while
    proving that iteration ``i`` cannot access iteration ``j``'s element.
    Shifted/unknown scalar accesses and atomics deliberately fail closed.
    """
    if lane_extent is None:
        return False
    load_marker = _marker_call(load_call, _LOAD_MARKER, 6)
    store_marker = _marker_call(store_call, _STORE_MARKER, 6)
    if (load_marker is None) != (store_marker is None):
        return False
    if load_marker is None:
        load_pointer = _plain_scalar_load_pointer(load_call)
        store_pointer = _plain_scalar_store_pointer(store_call)
        if load_pointer is None or store_pointer is None:
            return False
        unstable_names = loop_carried_names or set()
        expanded_load = _lane_dependent_alias_expansion(
            load_pointer, load_definitions, lane_var
        )
        expanded_store = _lane_dependent_alias_expansion(
            store_pointer, store_definitions, lane_var
        )
        exact_mapping = (
            ast.dump(expanded_load, include_attributes=False)
            == ast.dump(expanded_store, include_attributes=False)
            and _single_tensor_iterator_root(expanded_load) is not None
            and not bool(
                (set(ReadWrites.from_ast(expanded_load).reads) - {lane_var})
                & unstable_names
            )
            and _same_stable_address_aliases(
                expanded_load,
                load_definitions,
                store_definitions,
                lane_var,
                unstable_names,
            )
            and _scalar_pointer_is_known_affine(expanded_load, lane_var)
        )
        if exact_mapping and _scalar_lane_mapping_is_injective(
            expanded_load,
            lane_var,
            lane_extent,
            proven_tensor_stride_values or {},
        ):
            return True
        return _scalar_pointers_are_modularly_lane_disjoint(
            expanded_load,
            expanded_store,
            lane_var=lane_var,
            lane_extent=lane_extent,
            proven_tensor_stride_values=proven_tensor_stride_values or {},
            unstable_names=unstable_names,
        )
    assert store_marker is not None
    load_block = _literal(load_marker.args[0], int)
    load_width = _literal(load_marker.args[1], int)
    load_dtype = _literal(load_marker.args[2], str)
    store_block = _literal(store_marker.args[0], int)
    store_width = _literal(store_marker.args[1], int)
    store_dtype = _literal(store_marker.args[2], str)
    lane_match = _SYNTHETIC_LANE_RE.fullmatch(lane_var)
    if (
        lane_match is None
        or not isinstance(load_block, int)
        or load_block != store_block
        or load_block != int(lane_match.group(1))
        or not isinstance(load_width, int)
        or load_width != store_width
        or load_width != lane_extent
        or load_dtype != store_dtype
    ):
        return False

    load_pointer = load_marker.args[4]
    store_pointer = store_marker.args[3]
    if not isinstance(load_pointer, ast.expr) or not isinstance(
        store_pointer, ast.expr
    ):
        return False
    expanded_load = _lane_dependent_alias_expansion(
        load_pointer, load_definitions, lane_var
    )
    expanded_store = _lane_dependent_alias_expansion(
        store_pointer, store_definitions, lane_var
    )
    unstable_names = loop_carried_names or set()
    if ast.dump(expanded_load, include_attributes=False) == ast.dump(
        expanded_store, include_attributes=False
    ) and _same_stable_address_aliases(
        expanded_load,
        load_definitions,
        store_definitions,
        lane_var,
        unstable_names,
    ):
        depends, affine = _is_affine_in_lane(expanded_load, lane_var)
        if depends and affine:
            return True
    return _scalar_pointers_are_modularly_lane_disjoint(
        expanded_load,
        expanded_store,
        lane_var=lane_var,
        lane_extent=lane_extent,
        proven_tensor_stride_values=proven_tensor_stride_values or {},
        unstable_names=unstable_names,
    )


def _store_accesses_are_iteration_independent(
    first_call: ast.Call,
    second_call: ast.Call,
    *,
    lane_var: str,
    lane_extent: int,
    first_definitions: dict[str, ast.expr],
    second_definitions: dict[str, ast.expr],
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    unstable_names: set[str],
) -> bool:
    """Prove two ordinary exact-fragment stores independent across lanes."""
    first_marker = _marker_call(first_call, _STORE_MARKER, 6)
    second_marker = _marker_call(second_call, _STORE_MARKER, 6)
    if first_marker is None or second_marker is None:
        return False
    lane_match = _SYNTHETIC_LANE_RE.fullmatch(lane_var)
    signatures = [
        (
            _literal(marker.args[0], int),
            _literal(marker.args[1], int),
            _literal(marker.args[2], str),
        )
        for marker in (first_marker, second_marker)
    ]
    if (
        lane_match is None
        or signatures[0] != signatures[1]
        or signatures[0][0] != int(lane_match.group(1))
        or signatures[0][1] != lane_extent
    ):
        return False
    first_pointer = first_marker.args[3]
    second_pointer = second_marker.args[3]
    if not isinstance(first_pointer, ast.expr) or not isinstance(
        second_pointer, ast.expr
    ):
        return False
    expanded_first = _lane_dependent_alias_expansion(
        first_pointer, first_definitions, lane_var
    )
    expanded_second = _lane_dependent_alias_expansion(
        second_pointer, second_definitions, lane_var
    )
    if ast.dump(expanded_first, include_attributes=False) == ast.dump(
        expanded_second, include_attributes=False
    ) and _same_stable_address_aliases(
        expanded_first,
        first_definitions,
        second_definitions,
        lane_var,
        unstable_names,
    ):
        depends, affine = _is_affine_in_lane(expanded_first, lane_var)
        if depends and affine:
            return True
    return _scalar_pointers_are_modularly_lane_disjoint(
        expanded_first,
        expanded_second,
        lane_var=lane_var,
        lane_extent=lane_extent,
        proven_tensor_stride_values=proven_tensor_stride_values,
        unstable_names=unstable_names,
    )


class _ExpandLocalAliases(ast.NodeTransformer):
    def __init__(
        self,
        definitions: dict[str, ast.expr],
        lane_var: str,
        lane_replacement: ast.expr | None,
    ) -> None:
        super().__init__()
        self.definitions = definitions
        self.lane_var = lane_var
        self.lane_replacement = lane_replacement
        self.expanding: set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return node
        if node.id == self.lane_var:
            if self.lane_replacement is None:
                return node
            return ast.copy_location(_clone_expr(self.lane_replacement), node)
        value = self.definitions.get(node.id)
        if value is None or node.id in self.expanding:
            return node
        self.expanding.add(node.id)
        replacement = self.visit(_clone_expr(value))
        self.expanding.remove(node.id)
        return ast.copy_location(replacement, node)


def _expand(
    node: ast.expr,
    definitions: dict[str, ast.expr],
    lane_var: str,
    lane_replacement: ast.expr | None = None,
) -> ast.expr:
    result = _ExpandLocalAliases(definitions, lane_var, lane_replacement).visit(
        _clone_expr(node)
    )
    assert isinstance(result, ast.expr)
    return result


def _and_terms(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return [term for value in node.values for term in _and_terms(value)]
    return [node]


def _uniform_guard(
    mask: ast.expr | None,
    definitions: dict[str, ast.expr],
    lane_var: str,
    width: int,
    local_writes: set[str],
) -> tuple[bool, ast.expr | None]:
    if mask is None:
        return True, None
    expanded = _expand(mask, definitions, lane_var)
    uniform: list[ast.expr] = []
    for term in _and_terms(expanded):
        reads = set(ReadWrites.from_ast(term).reads)
        if lane_var in reads:
            if not isinstance(term, ast.Compare):
                return False, None
            # Prove the complete contiguous fragment is live by checking both
            # endpoints outside the loop.  The candidate's earlier unit-affine
            # proof makes this sufficient for generated tensor-bound compares.
            endpoints = (
                _expand(term, definitions, lane_var, ast.Constant(value=0)),
                _expand(
                    term,
                    definitions,
                    lane_var,
                    ast.Constant(value=width - 1),
                ),
            )
            if any(
                set(ReadWrites.from_ast(endpoint).reads) & local_writes
                for endpoint in endpoints
            ):
                return False, None
            uniform.extend(endpoints)
            continue
        if reads & local_writes:
            return False, None
        uniform.append(term)
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("load", "store")
        for term in uniform
        for node in ast.walk(term)
    ):
        return False, None
    if not uniform:
        return True, None
    if len(uniform) == 1:
        return True, uniform[0]
    return True, ast.BoolOp(op=ast.And(), values=uniform)


def _flatten_add(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_flatten_add(node.left), *_flatten_add(node.right)]
    return [node]


def _sum_terms(terms: list[ast.expr]) -> ast.expr:
    assert terms
    result = _clone_expr(terms[0])
    for term in terms[1:]:
        result = ast.BinOp(left=result, op=ast.Add(), right=_clone_expr(term))
    return result


def _safe_base_pointer(
    pointer: ast.expr,
    definitions: dict[str, ast.expr],
    lane_var: str,
    local_writes: set[str],
    guard: ast.expr | None,
) -> ast.expr | None:
    expanded_with_lane = _expand(pointer, definitions, lane_var)
    if lane_var not in set(ReadWrites.from_ast(expanded_with_lane).reads):
        return None
    base = _expand(pointer, definitions, lane_var, ast.Constant(value=0))
    if set(ReadWrites.from_ast(base).reads) & local_writes:
        return None
    if any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("load", "store")
            or isinstance(node.func, ast.Name)
            and node.func.id in (_LOAD_MARKER, _STORE_MARKER)
        )
        for node in ast.walk(base)
    ):
        return None
    if guard is None:
        return base

    terms = _flatten_add(base)
    iterator_indices = [
        index
        for index, term in enumerate(terms)
        if isinstance(term, ast.Attribute)
        and term.attr == "iterator"
        and isinstance(term.value, ast.Name)
    ]
    if len(iterator_indices) != 1 or len(terms) == 1:
        return None
    iterator_index = iterator_indices[0]
    iterator = terms[iterator_index]
    offset = _sum_terms(
        [term for index, term in enumerate(terms) if index != iterator_index]
    )
    return ast.BinOp(
        left=_clone_expr(iterator),
        op=ast.Add(),
        right=ast.IfExp(
            test=_clone_expr(guard),
            body=offset,
            orelse=ast.parse("cutlass.Int32(0)", mode="eval").body,
        ),
    )


def _scalar_load_and_mask(node: ast.expr) -> tuple[ast.Call, ast.expr | None] | None:
    if isinstance(node, ast.IfExp):
        value = node.body
        mask = node.test
    else:
        value = node
        mask = None
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "load"
    ):
        return value, mask
    return None


class _ReplaceCall(ast.NodeTransformer):
    def __init__(self, target: ast.Call, replacement: ast.expr) -> None:
        super().__init__()
        self.target = ast.dump(target, include_attributes=False)
        self.replacement = replacement
        self.replaced = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (
            not self.replaced
            and ast.dump(node, include_attributes=False) == self.target
        ):
            self.replaced = True
            return _clone_expr(self.replacement)
        return self.generic_visit(node)


class _RestoreLoadMarkers(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        marker = _marker_call(node, _LOAD_MARKER, 6)
        if marker is not None:
            return self.visit(_clone_expr(marker.args[5]))
        return self.generic_visit(node)


class _BranchLocalPersistentVectorizer:
    def __init__(
        self,
        body: list[ast.stmt],
        proven_disjoint_tensor_pairs: set[frozenset[str]],
        proven_tensor_stride_values: Mapping[tuple[str, int], int],
    ) -> None:
        self.used_names = {
            node.id
            for stmt in body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Name)
        }
        self.counter = 0
        self.proven_disjoint_tensor_pairs = proven_disjoint_tensor_pairs
        self.proven_tensor_stride_values = proven_tensor_stride_values

    def _fresh(self, prefix: str) -> str:
        while True:
            name = f"{prefix}_{self.counter}"
            self.counter += 1
            if name not in self.used_names:
                self.used_names.add(name)
                return name

    @staticmethod
    def _store_marker(stmt: ast.stmt) -> ast.Call | None:
        if not isinstance(stmt, ast.Expr):
            return None
        return _marker_call(stmt.value, _STORE_MARKER, 6)

    @staticmethod
    def _restore_store(marker: ast.Call) -> list[ast.stmt]:
        pointer, value, mask = marker.args[3:]
        store = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=_clone_expr(pointer), attr="store", ctx=ast.Load()
                ),
                args=[_clone_expr(value)],
                keywords=[],
            )
        )
        if isinstance(mask, ast.Constant) and mask.value is None:
            return [store]
        return [
            ast.If(test=_clone_expr(mask), body=[store], orelse=[]),
        ]

    @staticmethod
    def _pointer_roots(node: ast.AST) -> frozenset[str] | None:
        roots: set[str] = set()
        unresolved = False
        for child in ast.walk(node):
            if not isinstance(child, ast.Attribute) or child.attr != "iterator":
                continue
            if isinstance(child.value, ast.Name):
                roots.add(child.value.id)
            else:
                unresolved = True
        if unresolved:
            return None
        return frozenset(roots) if roots else None

    def _may_alias(
        self,
        left: frozenset[str] | None,
        right: frozenset[str] | None,
    ) -> bool:
        if left is None or right is None:
            return True
        return any(
            left_name == right_name
            or frozenset((left_name, right_name))
            not in self.proven_disjoint_tensor_pairs
            for left_name in left
            for right_name in right
        )

    @staticmethod
    def _contains_arch_attribute(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Attribute) and child.attr == "arch"
            for child in ast.walk(node)
        )

    @classmethod
    def _load_pointer(cls, node: ast.Call) -> ast.AST | None:
        marker = _marker_call(node, _LOAD_MARKER, 6)
        if marker is not None:
            return marker.args[4]
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "load"):
            return None
        if node.args and cls._contains_arch_attribute(node.func.value):
            return node.args[0]
        return node.func.value

    @classmethod
    def _store_pointer(cls, node: ast.Call) -> ast.AST | None:
        marker = _marker_call(node, _STORE_MARKER, 6)
        if marker is not None:
            return marker.args[3]
        func = node.func
        if isinstance(func, ast.Name) and func.id.startswith(
            ("_cute_store_", "_cute_atomic_")
        ):
            return node.args[0] if node.args else None
        if not isinstance(func, ast.Attribute):
            return None
        if func.attr.startswith("atomic_"):
            return node.args[0] if node.args else None
        if func.attr == "store":
            if node.args and cls._contains_arch_attribute(func.value):
                return node.args[0]
            return func.value
        if func.attr == "__setitem__":
            return func.value
        return None

    def _has_aliasing_write_in_loop(
        self,
        body: list[ast.stmt],
        load_marker: ast.Call,
        load_index: int,
        lane_var: str,
        lane_extent: int,
    ) -> bool:
        """Whether any loop iteration can write storage read by this load.

        A write lexically after the load still runs before that load in the
        next constexpr-lane iteration.  Hoisting the vector load above the
        loop would observe the pre-write value for every lane. Such a write is
        a barrier unless its tensor is disjoint or both exact accesses have the
        same injective lane mapping.
        """
        load_pointer = load_marker.args[4]
        load_roots = self._pointer_roots(load_pointer)
        definition_snapshots = _definition_snapshots(body)
        for store_index, stmt in enumerate(body):
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Call):
                    continue
                store_pointer = self._store_pointer(node)
                if store_pointer is None or not self._may_alias(
                    self._pointer_roots(store_pointer), load_roots
                ):
                    continue
                # A later exact store preserves this load's same-iteration
                # ordering. Its identical injective lane mapping additionally
                # rules out a dependence through the loop backedge.
                if (
                    store_index > load_index
                    and _lane_accesses_are_iteration_independent(
                        load_marker,
                        node,
                        lane_var=lane_var,
                        lane_extent=lane_extent,
                        load_definitions=definition_snapshots[load_index],
                        store_definitions=definition_snapshots[store_index],
                        proven_tensor_stride_values=(self.proven_tensor_stride_values),
                        loop_carried_names=_unstable_address_names(
                            body,
                            lane_var,
                            load_index,
                            store_index,
                        ),
                    )
                ):
                    continue
                return True
        return False

    def _has_other_aliasing_access_in_loop(
        self,
        body: list[ast.stmt],
        store_marker: ast.Call,
        store_index: int,
        lane_var: str,
        lane_extent: int,
    ) -> bool:
        """Whether sinking a store can cross an aliasing memory access.

        A load lexically before the store executes after that store on the
        next lane iteration.  Likewise, another write on either side can race
        the sunk vector transaction across iterations.  Exclude only the
        candidate marker itself; every other load/store/atomic in the repeated
        loop must be proven disjoint, except a preceding exact load with the
        same injective lane mapping.
        """
        store_roots = self._pointer_roots(store_marker.args[3])
        definition_snapshots = _definition_snapshots(body)
        for access_index, stmt in enumerate(body):
            for node in _memory_load_calls(stmt):
                pointer = self._load_pointer(node)
                if pointer is not None and self._may_alias(
                    store_roots, self._pointer_roots(pointer)
                ):
                    # Sinking the store retains a preceding load's within-lane
                    # order. Matching exact lane mappings rule out only the
                    # cross-iteration dependence introduced by the sink.
                    if (
                        access_index < store_index
                        and _lane_accesses_are_iteration_independent(
                            node,
                            store_marker,
                            lane_var=lane_var,
                            lane_extent=lane_extent,
                            load_definitions=definition_snapshots[access_index],
                            store_definitions=definition_snapshots[store_index],
                            proven_tensor_stride_values=(
                                self.proven_tensor_stride_values
                            ),
                            loop_carried_names=_unstable_address_names(
                                body,
                                lane_var,
                                access_index,
                                store_index,
                            ),
                        )
                    ):
                        continue
                    return True
            for node in ast.walk(stmt):
                if node is store_marker or not isinstance(node, ast.Call):
                    continue
                pointer = self._store_pointer(node)
                if pointer is not None and self._may_alias(
                    store_roots, self._pointer_roots(pointer)
                ):
                    if _store_accesses_are_iteration_independent(
                        store_marker,
                        node,
                        lane_var=lane_var,
                        lane_extent=lane_extent,
                        first_definitions=definition_snapshots[store_index],
                        second_definitions=definition_snapshots[access_index],
                        proven_tensor_stride_values=(self.proven_tensor_stride_values),
                        unstable_names=_unstable_address_names(
                            body,
                            lane_var,
                            min(store_index, access_index),
                            max(store_index, access_index),
                        ),
                    ):
                        continue
                    return True
        return False

    def _vectorize_loop(
        self, loop: ast.For, block_id: int, width: int
    ) -> tuple[list[ast.stmt], ast.For, list[ast.stmt]]:
        assert isinstance(loop.target, ast.Name)
        lane_var = loop.target.id
        all_local_writes = {
            name for stmt in loop.body for name in ReadWrites.from_ast(stmt).writes
        }
        definitions: dict[str, ast.expr] = {}
        before: list[ast.stmt] = []
        after: list[ast.stmt] = []
        rewritten_body: list[ast.stmt] = []

        for stmt_index, stmt in enumerate(loop.body):
            load_markers: list[ast.Call] = []
            for node in ast.walk(stmt):
                marker_call = _marker_call(node, _LOAD_MARKER, 6)
                if marker_call is not None:
                    load_markers.append(marker_call)
            if len(load_markers) == 1:
                marker = load_markers[0]
                marker_block = _literal(marker.args[0], int)
                marker_width = _literal(marker.args[1], int)
                dtype = _literal(marker.args[2], str)
                eviction = _literal(marker.args[3], str)
                original = marker.args[5]
                load_info = (
                    _scalar_load_and_mask(original)
                    if isinstance(original, ast.expr)
                    else None
                )
                dtype_info = _DTYPE_INFO.get(dtype) if isinstance(dtype, str) else None
                if (
                    marker_block == block_id
                    and marker_width == width
                    and isinstance(eviction, str)
                    and dtype_info is not None
                    and load_info is not None
                    and not self._has_aliasing_write_in_loop(
                        loop.body,
                        marker,
                        stmt_index,
                        lane_var,
                        width,
                    )
                ):
                    scalar_load, mask = load_info
                    guard_ok, guard = _uniform_guard(
                        mask, definitions, lane_var, width, all_local_writes
                    )
                    base = (
                        _safe_base_pointer(
                            marker.args[4],
                            definitions,
                            lane_var,
                            all_local_writes,
                            guard,
                        )
                        if guard_ok and isinstance(marker.args[4], ast.expr)
                        else None
                    )
                    if base is not None:
                        carrier, itemsize, _store_helper = dtype_info
                        vec_name = self._fresh("_persistent_branch_vec")
                        vector_type = (
                            f"ir.VectorType.get([{width}], {carrier}.mlir_type)"
                        )
                        if eviction == "__l2_last__":
                            load_source = (
                                f"_cute_load_l2_evict_last({ast.unparse(base)}, "
                                f"{vector_type})"
                                if width * itemsize == 16
                                else f"cute.arch.load({ast.unparse(base)}, {vector_type})"
                            )
                        else:
                            load_source = (
                                f"cute.arch.load({ast.unparse(base)}, {vector_type}"
                                f"{eviction})"
                            )
                        vector_load = ast.parse(load_source, mode="eval").body
                        before.append(
                            ast.Assign(
                                targets=[ast.Name(id=vec_name, ctx=ast.Store())],
                                value=vector_load,
                            )
                        )
                        extract = ast.parse(
                            f"{carrier}({vec_name}[{lane_var}]).bitcast({dtype})",
                            mode="eval",
                        ).body
                        replacement = _ReplaceCall(scalar_load, extract).visit(
                            _clone_expr(original)
                        )
                        assert isinstance(replacement, ast.expr)
                        stmt = _ReplaceCall(marker, replacement).visit(stmt)
                        assert isinstance(stmt, ast.stmt)

            store_marker = self._store_marker(stmt)
            if store_marker is not None:
                marker_block = _literal(store_marker.args[0], int)
                marker_width = _literal(store_marker.args[1], int)
                dtype = _literal(store_marker.args[2], str)
                dtype_info = _DTYPE_INFO.get(dtype) if isinstance(dtype, str) else None
                mask_arg = store_marker.args[5]
                mask = (
                    None
                    if isinstance(mask_arg, ast.Constant) and mask_arg.value is None
                    else mask_arg
                )
                guard_ok, guard = (
                    _uniform_guard(mask, definitions, lane_var, width, all_local_writes)
                    if isinstance(mask, (ast.expr, type(None)))
                    else (False, None)
                )
                base = (
                    _safe_base_pointer(
                        store_marker.args[3],
                        definitions,
                        lane_var,
                        all_local_writes,
                        None,
                    )
                    if guard_ok and isinstance(store_marker.args[3], ast.expr)
                    else None
                )
                if (
                    marker_block == block_id
                    and marker_width == width
                    and dtype_info is not None
                    and base is not None
                    and not self._has_other_aliasing_access_in_loop(
                        loop.body,
                        store_marker,
                        stmt_index,
                        lane_var,
                        width,
                    )
                ):
                    carrier, _itemsize, store_helper = dtype_info
                    values_name = self._fresh("_persistent_branch_store_values")
                    before.append(
                        ast.Assign(
                            targets=[ast.Name(id=values_name, ctx=ast.Store())],
                            value=ast.List(elts=[], ctx=ast.Load()),
                        )
                    )
                    append = ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=values_name, ctx=ast.Load()),
                                attr="append",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Call(
                                    func=ast.Attribute(
                                        value=_clone_expr(store_marker.args[4]),
                                        attr="bitcast",
                                        ctx=ast.Load(),
                                    ),
                                    args=[
                                        ast.parse(carrier, mode="eval").body,
                                    ],
                                    keywords=[],
                                )
                            ],
                            keywords=[],
                        )
                    )
                    stmt = append
                    flush = ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id=store_helper, ctx=ast.Load()),
                            args=[
                                base,
                                ast.Name(id=values_name, ctx=ast.Load()),
                            ],
                            keywords=[],
                        )
                    )
                    after.append(
                        flush
                        if guard is None
                        else ast.If(test=_clone_expr(guard), body=[flush], orelse=[])
                    )
                else:
                    rewritten_body.extend(self._restore_store(store_marker))
                    continue

            rewritten_body.append(stmt)
            definitions.update(_single_name_definitions([stmt]))

        loop.body = self.transform_body(rewritten_body)
        return before, loop, after

    def transform_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for stmt in body:
            if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                match = _SYNTHETIC_LANE_RE.fullmatch(stmt.target.id)
                extent = _range_extent(stmt)
                if match is not None and extent is not None:
                    before, loop, after = self._vectorize_loop(
                        stmt, int(match.group(1)), extent
                    )
                    result.extend(before)
                    result.append(loop)
                    result.extend(after)
                    continue

            store_marker = self._store_marker(stmt)
            if store_marker is not None:
                result.extend(self._restore_store(store_marker))
                continue
            for field in ("body", "orelse", "finalbody"):
                child = getattr(stmt, field, None)
                if isinstance(child, list) and all(
                    isinstance(child_stmt, ast.stmt) for child_stmt in child
                ):
                    setattr(stmt, field, self.transform_body(child))
            result.append(stmt)
        return result


def vectorize_branch_local_persistent_fragments(
    body: list[ast.stmt],
    *,
    proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
    proven_tensor_stride_values: Mapping[tuple[str, int], int] | None = None,
) -> list[ast.stmt]:
    """Vectorize marked exact fragments and erase every unhandled marker."""
    vectorizer = _BranchLocalPersistentVectorizer(
        body,
        proven_disjoint_tensor_pairs or set(),
        proven_tensor_stride_values or {},
    )
    transformed = vectorizer.transform_body(body)
    # Most kernels have no persistent-fragment load markers.  Avoid running an
    # AST ``NodeTransformer`` over those bodies: FX can preserve an
    # ``immutable_list`` in constructs such as inline-asm operand lists, while
    # ``NodeTransformer.generic_visit`` mutates list fields in place.
    if not any(
        isinstance(node, ast.Call) and _marker_call(node, _LOAD_MARKER, 6) is not None
        for stmt in transformed
        for node in ast.walk(stmt)
    ):
        module = ast.Module(body=transformed, type_ignores=[])
        ast.fix_missing_locations(module)
        return module.body
    restored = _RestoreLoadMarkers().visit(
        ast.Module(body=transformed, type_ignores=[])
    )
    assert isinstance(restored, ast.Module)
    ast.fix_missing_locations(restored)
    return restored.body
