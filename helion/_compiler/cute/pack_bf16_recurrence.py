"""Lower an async BF16 rank-1 recurrence to native BF16x2 arithmetic.

This is a deliberately late and fail-closed source transform.  The async-state
pipeline has already proved a private, aligned, in-place 16-byte state vector
and records that proof on the row loop.  This pass additionally proves the
generated rank-1 update graph (decay, key projection, residual update, query
projection, state store) before changing either arithmetic or storage types.

No tensor, function, or generated temporary name participates in matching.
The lowering is opt-in because BF16x2 changes reduction association and rounds
the recurrent state update after each native packed operation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import replace

from ..ast_extension import ExtendedAST

_ABI_VERSION = 1
_PACK = "_cute_rank1_pack_bf16x2"
_MUL = "_cute_rank1_mul_bf16x2"
_ADD = "_cute_rank1_add_bf16x2"
_FMA = "_cute_rank1_fma_bf16x2"


def _clone_ast(value: object) -> object:
    """Clone ordinary and Helion AST nodes while retaining proof metadata."""
    if isinstance(value, list):
        return [_clone_ast(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_ast(item) for item in value)
    if not isinstance(value, ast.AST):
        return value
    fields = {field: _clone_ast(getattr(value, field)) for field in value._fields}
    if isinstance(value, ExtendedAST):
        result = value.copy(**fields)
    else:
        result = ast.copy_location(type(value)(**fields), value)
    for name, item in value.__dict__.items():
        if name.startswith("_helion_"):
            result.__dict__[name] = _clone_ast(item)
    return result


def _clone_expression(value: ast.expr) -> ast.expr:
    result = _clone_ast(value)
    assert isinstance(result, ast.expr)
    return result


def _call_path(node: ast.AST) -> tuple[str, ...] | None:
    path: list[str] = []
    while isinstance(node, ast.Attribute):
        path.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    path.append(node.id)
    return tuple(reversed(path))


def _calls(node: ast.AST, path: tuple[str, ...]) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_path(child.func) == path
    ]


def _name_target(stmt: ast.stmt) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    return None


def _assigned_names(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            result.add(child.id)
    return result


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _name_loads(node: ast.AST, name: str) -> set[int]:
    return {
        id(child)
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and child.id == name
    }


def _loads_only_within(node: ast.AST, name: str, allowed: ast.AST) -> bool:
    return _name_loads(node, name) <= _name_loads(allowed, name)


def _loaded_before_redefinition(
    statements: list[ast.stmt], start: int, name: str
) -> bool:
    """Whether a straight-line generated suffix reads ``name`` before killing it."""

    def statement_access(statement: ast.stmt) -> tuple[bool, bool]:
        if isinstance(statement, ast.Assign):
            if name in _loaded_names(statement.value):
                return True, False
            return False, name in _assigned_names(statement)
        if (
            isinstance(statement, ast.For)
            and (_constexpr_trip_count(statement) or 0) > 0
        ):
            if name in _loaded_names(statement.iter):
                return True, False
            if isinstance(statement.target, ast.Name) and statement.target.id == name:
                return False, True
            for child in statement.body:
                read, written = statement_access(child)
                if read or written:
                    return read, written
            return False, False
        return name in _loaded_names(statement), name in _assigned_names(statement)

    for statement in statements[start:]:
        read, written = statement_access(statement)
        if read:
            return True
        if written:
            return False
    return False


def _assignment_rhs_is_pure(statement: ast.Assign) -> bool:
    """Conservatively allow generated scalar computations that may be erased."""
    if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
        return False
    return _expression_is_pure(statement.value)


def _expression_is_pure(node: ast.expr) -> bool:
    """Accept only side-effect-free calls used by generated scalar arithmetic."""
    for call in (child for child in ast.walk(node) if isinstance(child, ast.Call)):
        path = _call_path(call.func)
        if path is not None and (
            path[0] == "operator"
            or path[:2] == ("cute", "math")
            or path
            in {
                ("cute", "arch", "fma_packed_f32x2"),
                ("cute", "arch", "lane_idx"),
                ("cute", "arch", "mul_packed_f32x2"),
                ("cute", "arch", "shuffle_sync"),
                ("cute", "arch", "thread_idx"),
                ("cute", "arch", "warp_reduction_sum"),
            }
            or path[0] == "cutlass"
        ):
            continue
        if isinstance(call.func, ast.Attribute) and call.func.attr in {
            "bitcast",
            "load",
        }:
            continue
        return False
    return True


def _recurrence_loop_is_pure(
    loop: ast.For,
    *,
    subscript_containers: frozenset[str],
    append_containers: frozenset[str],
) -> bool:
    """Prove a loop replaced wholesale has no unmodeled observable effects."""

    def local_target(target: ast.expr) -> bool:
        if isinstance(target, ast.Name):
            return True
        if isinstance(target, (ast.List, ast.Tuple)):
            return all(local_target(element) for element in target.elts)
        return (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id in subscript_containers
        )

    for statement in loop.body:
        if isinstance(statement, ast.Assign):
            if not all(local_target(target) for target in statement.targets) or not (
                _expression_is_pure(statement.value)
            ):
                return False
            continue
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "append"
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id in append_containers
            and len(statement.value.args) == 1
            and not statement.value.keywords
            and _expression_is_pure(statement.value.args[0])
        ):
            return False
    return True


def _has_unique_name_writes(node: ast.AST) -> bool:
    writes = [
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del))
    ]
    return len(writes) == len(set(writes))


def _cache_expansion_loop_is_pure(loop: ast.For, cache_names: set[str]) -> bool:
    """Prove a cache-expansion loop contains only local register writes."""
    for statement in loop.body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and (
                isinstance(statement.targets[0], ast.Name)
                or (
                    isinstance(statement.targets[0], ast.Subscript)
                    and isinstance(statement.targets[0].value, ast.Name)
                    and statement.targets[0].value.id in cache_names
                )
            )
            and _expression_is_pure(statement.value)
        ):
            return False
    return True


def _packed_cache_expression_is_pure(node: ast.expr) -> bool:
    for call in (child for child in ast.walk(node) if isinstance(child, ast.Call)):
        path = _call_path(call.func)
        if path == ("cutlass", "Float16"):
            return False
        if path is not None and path[0] == "cutlass":
            continue
        if isinstance(call.func, ast.Attribute) and call.func.attr == "bitcast":
            continue
        return False
    return True


def _constexpr_trip_count(loop: ast.For) -> int | None:
    if not (
        isinstance(loop.target, ast.Name)
        and isinstance(loop.iter, ast.Call)
        and _call_path(loop.iter.func) == ("cutlass", "range_constexpr")
        and len(loop.iter.args) == 1
        and not loop.iter.keywords
        and isinstance(loop.iter.args[0], ast.Constant)
        and type(loop.iter.args[0].value) is int
    ):
        return None
    return loop.iter.args[0].value


def _unwrap_value_casts(node: ast.expr) -> ast.expr:
    """Remove only the scalar representation casts emitted around pair lanes."""
    while True:
        if (
            isinstance(node, ast.Call)
            and _call_path(node.func)
            in {
                ("cutlass", "BFloat16"),
                ("cutlass", "Float16"),
                ("cutlass", "Float32"),
                ("cutlass", "Uint16"),
                ("cutlass", "Uint32"),
            }
            and len(node.args) == 1
            and not node.keywords
        ):
            node = node.args[0]
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "bitcast"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.func.value, ast.expr)
        ):
            node = node.func.value
            continue
        return node


def _direct_cache_subscript(node: ast.expr) -> ast.Subscript | None:
    node = _unwrap_value_casts(node)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node
    return None


def _expected_pair_indices(loop_var: str) -> tuple[object | None, object | None]:
    low = ast.BinOp(
        left=ast.Name(id=loop_var, ctx=ast.Load()),
        op=ast.Mult(),
        right=ast.Constant(value=2),
    )
    high = ast.BinOp(left=low, op=ast.Add(), right=ast.Constant(value=1))
    return _canonical_index(low), _canonical_index(high)


def _pair_cache(
    node: ast.expr,
    *,
    loop_var: str,
    definitions: dict[str, ast.expr],
) -> str | None:
    """Match an exact low/high cache pair with optional common scale factors.

    Factor-affine lowering wraps generated cache reads in scalar casts and may
    multiply both lanes by the same row-uniform normalization factors.  Accept
    precisely that grammar.  In particular, walking for an arbitrary nested
    subscript is unsafe: an addition such as ``cache[lo] + 1`` would otherwise
    be silently discarded by the packed rewrite.
    """
    node = _inline_names(node, definitions)
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 2:
        return None
    expected_indices = _expected_pair_indices(loop_var)
    cache_name: str | None = None
    common_factors: list[ast.expr] | None = None
    for lane, element in enumerate(node.elts):
        element = _unwrap_value_casts(element)
        factors = _multiplication_factors(element)
        cache_factors = [
            (position, subscript)
            for position, factor in enumerate(factors)
            if (subscript := _direct_cache_subscript(factor)) is not None
        ]
        if len(cache_factors) != 1:
            return None
        position, subscript = cache_factors[0]
        assert isinstance(subscript.value, ast.Name)
        if _canonical_index(subscript.slice) != expected_indices[lane]:
            return None
        if cache_name is None:
            cache_name = subscript.value.id
        elif cache_name != subscript.value.id:
            return None
        other_factors = [
            factor for index, factor in enumerate(factors) if index != position
        ]
        if common_factors is None:
            common_factors = other_factors
        elif len(common_factors) != len(other_factors) or any(
            not _same_expression(left, right)
            for left, right in zip(common_factors, other_factors, strict=True)
        ):
            return None
    return cache_name


def _normalized_pair_expression(
    node: ast.expr,
    *,
    loop_var: str,
    definitions: dict[str, ast.expr],
) -> ast.expr:
    resolved = _inline_names(node, definitions)
    return _substitute_names(
        resolved,
        {loop_var: ast.Name(id="_pair_lane", ctx=ast.Load())},
    )


def _same_pair_expression(
    left: ast.expr,
    left_definitions: dict[str, ast.expr],
    left_loop_var: str,
    right: ast.expr,
    right_definitions: dict[str, ast.expr],
    right_loop_var: str,
) -> bool:
    return _same_expression(
        _normalized_pair_expression(
            left,
            loop_var=left_loop_var,
            definitions=left_definitions,
        ),
        _normalized_pair_expression(
            right,
            loop_var=right_loop_var,
            definitions=right_definitions,
        ),
    )


def _pair_is_direct_cache(node: ast.expr, cache_name: str) -> bool:
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 2:
        return False
    subscripts = [_direct_cache_subscript(element) for element in node.elts]
    return all(
        subscript is not None
        and isinstance(subscript.value, ast.Name)
        and subscript.value.id == cache_name
        for subscript in subscripts
    )


def _zero_value(node: ast.expr) -> bool:
    node = _unwrap_value_casts(node)
    return (
        isinstance(node, ast.Constant)
        and type(node.value) in {int, float}
        and node.value == 0
    )


def _zero_initialized_before(
    statements: list[ast.stmt], end: int, names: tuple[str, ...]
) -> bool:
    """Prove that the last top-level write to each accumulator is exact zero."""
    remaining = set(names)
    for statement in reversed(statements[:end]):
        written = remaining & _assigned_names(statement)
        if not written:
            continue
        target = _name_target(statement)
        if (
            target is None
            or target not in written
            or not isinstance(statement, ast.Assign)
            or not _zero_value(statement.value)
        ):
            return False
        remaining.remove(target)
        if not remaining:
            return True
    return False


def _state_element_subscript(node: ast.expr) -> ast.Subscript | None:
    """Recover a state lane only through an exact BF16 bitcast chain."""
    saw_bf16_bitcast = False
    while True:
        if isinstance(node, ast.IfExp):
            if not _zero_value(node.orelse):
                return None
            node = node.body
            continue
        if not isinstance(node, ast.Call):
            break
        path = _call_path(node.func)
        if (
            path
            in {
                ("cutlass", "BFloat16"),
                ("cutlass", "Float32"),
                ("cutlass", "Uint16"),
                ("cutlass", "Uint32"),
            }
            and len(node.args) == 1
            and not node.keywords
        ):
            node = node.args[0]
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "bitcast"
            and len(node.args) == 1
            and not node.keywords
            and _call_path(node.args[0]) == ("cutlass", "BFloat16")
            and not saw_bf16_bitcast
            and isinstance(node.func.value, ast.expr)
        ):
            saw_bf16_bitcast = True
            node = node.func.value
            continue
        return None
    subscript = _direct_cache_subscript(node)
    return subscript if saw_bf16_bitcast else None


def _state_pair_cache(
    node: ast.expr,
    *,
    loop_var: str,
    definitions: dict[str, ast.expr],
) -> str | None:
    node = _inline_names(node, definitions)
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 2:
        return None
    expected_indices = _expected_pair_indices(loop_var)
    cache_name: str | None = None
    for lane, element in enumerate(node.elts):
        subscript = _state_element_subscript(element)
        if (
            subscript is None
            or _canonical_index(subscript.slice) != expected_indices[lane]
        ):
            return None
        assert isinstance(subscript.value, ast.Name)
        if cache_name is None:
            cache_name = subscript.value.id
        elif cache_name != subscript.value.id:
            return None
    return cache_name


def _tuple_target_names(call: ast.Call, loop: ast.For) -> tuple[str, str] | None:
    for child in ast.walk(loop):
        if not isinstance(child, ast.Assign) or child.value is not call:
            continue
        if len(child.targets) != 1 or not isinstance(child.targets[0], ast.Tuple):
            return None
        elts = child.targets[0].elts
        if len(elts) != 2 or not all(isinstance(elt, ast.Name) for elt in elts):
            return None
        return elts[0].id, elts[1].id  # type: ignore[union-attr]
    return None


def _scalar_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Call):
        if (
            _call_path(node.func)
            in {
                ("cutlass", "BFloat16"),
                ("cutlass", "Float16"),
                ("cutlass", "Float32"),
                ("cutlass", "Uint16"),
                ("cutlass", "Uint32"),
            }
            and len(node.args) == 1
            and not node.keywords
        ):
            node = node.args[0]
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "bitcast"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.func.value, ast.expr)
        ):
            node = node.func.value
            continue
        break
    return node.id if isinstance(node, ast.Name) else None


def _name_pair(node: ast.expr) -> tuple[str, str] | None:
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 2:
        return None
    names = tuple(_scalar_name(element) for element in node.elts)
    if names[0] is None or names[1] is None:
        return None
    return names[0], names[1]  # type: ignore[return-value]


def _same_expression(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def _canonical_index(node: ast.expr) -> object | None:
    while (
        isinstance(node, ast.Call)
        and _call_path(node.func)
        in {
            ("cutlass", "Int32"),
            ("cutlass", "Int64"),
            ("cutlass", "Uint32"),
            ("cutlass", "Uint64"),
        }
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return ("constant", node.value)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and _call_path(node.value.func) == ("cute", "arch", "thread_idx")
        and not node.value.args
        and not node.value.keywords
        and isinstance(node.slice, ast.Constant)
        and type(node.slice.value) is int
    ):
        return ("thread", node.slice.value)
    if isinstance(node, ast.BinOp):
        left = _canonical_index(node.left)
        right = _canonical_index(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Sub) and right == ("constant", 0):
            return left
        if isinstance(node.op, ast.FloorDiv) and right == ("constant", 1):
            return left
        if isinstance(node.op, ast.Add):
            return ("add", *sorted((left, right), key=repr))
        if isinstance(node.op, ast.Mult):
            return ("multiply", *sorted((left, right), key=repr))
        if isinstance(node.op, ast.FloorDiv):
            return ("floor_divide", left, right)
        if isinstance(node.op, ast.Mod):
            return ("modulo", left, right)
    return None


def _warp_reduction_target(
    statements: list[ast.stmt],
    start: int,
    end: int,
    accumulators: tuple[str, str],
) -> str | None:
    matches: list[str] = []
    for index in range(start, end):
        statement = statements[index]
        target = _name_target(statement)
        if target is None or not isinstance(statement, ast.Assign):
            continue
        value = statement.value
        if not (
            isinstance(value, ast.Call)
            and _call_path(value.func) == ("cutlass", "Float32")
            and len(value.args) == 1
            and not value.keywords
            and isinstance(value.args[0], ast.Call)
        ):
            continue
        reduction = value.args[0]
        if not (
            _call_path(reduction.func) == ("cute", "arch", "warp_reduction_sum")
            and len(reduction.args) == 1
            and isinstance(reduction.args[0], ast.Name)
            and len(reduction.keywords) == 1
            and reduction.keywords[0].arg == "threads_in_group"
            and isinstance(reduction.keywords[0].value, ast.Constant)
            and reduction.keywords[0].value.value == 16
        ):
            continue
        lane_name = reduction.args[0].id
        lane_definitions = [
            previous
            for previous in statements[start:index]
            if _name_target(previous) == lane_name and isinstance(previous, ast.Assign)
        ]
        if len(lane_definitions) != 1:
            continue
        lane_value = lane_definitions[0].value
        if not isinstance(lane_value, ast.BinOp) or not isinstance(
            lane_value.op, ast.Add
        ):
            continue
        lane_pair = (
            _scalar_name(lane_value.left),
            _scalar_name(lane_value.right),
        )
        if lane_pair == accumulators or lane_pair == tuple(reversed(accumulators)):
            matches.append(target)
    return matches[0] if len(matches) == 1 else None


def _resolve_value_aliases(
    node: ast.expr, definitions: dict[str, ast.expr]
) -> ast.expr:
    result = node
    visited: set[str] = set()
    while True:
        if isinstance(result, ast.Name) and result.id in definitions:
            if result.id in visited:
                return result
            visited.add(result.id)
            result = definitions[result.id]
            continue
        if (
            isinstance(result, ast.Call)
            and _call_path(result.func)
            in {
                ("cutlass", "BFloat16"),
                ("cutlass", "Float16"),
                ("cutlass", "Float32"),
            }
            and len(result.args) == 1
            and not result.keywords
        ):
            result = result.args[0]
            continue
        return result


def _is_named(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_original_output(
    expression: ast.expr,
    definitions: dict[str, ast.expr],
    *,
    delta: str,
    qdot: str,
    base: str,
) -> bool:
    expression = _resolve_value_aliases(expression, definitions)
    if (
        isinstance(expression, ast.Call)
        and _call_path(expression.func) == ("cute", "math", "fma")
        and len(expression.args) == 3
        and not expression.keywords
    ):
        return (
            _is_named(expression.args[0], delta)
            and _is_named(expression.args[1], qdot)
            and _is_named(expression.args[2], base)
        )
    if not isinstance(expression, ast.BinOp) or not isinstance(expression.op, ast.Add):
        return False
    for base_term, product in (
        (expression.left, expression.right),
        (expression.right, expression.left),
    ):
        if not _is_named(base_term, base) or not isinstance(product, ast.BinOp):
            continue
        if not isinstance(product.op, ast.Mult):
            continue
        if (_is_named(product.left, delta) and _is_named(product.right, qdot)) or (
            _is_named(product.left, qdot) and _is_named(product.right, delta)
        ):
            return True
    return False


def _is_bf16_output(expression: ast.expr, definitions: dict[str, ast.expr]) -> bool:
    """Require the value passed to the output store to round to BF16."""
    visited: set[str] = set()
    while isinstance(expression, ast.Name) and expression.id in definitions:
        if expression.id in visited:
            return False
        visited.add(expression.id)
        expression = definitions[expression.id]
    return (
        isinstance(expression, ast.Call)
        and _call_path(expression.func) == ("cutlass", "BFloat16")
        and len(expression.args) == 1
        and not expression.keywords
        and not any(
            isinstance(node, ast.Call)
            and _call_path(node.func) == ("cutlass", "Float16")
            for node in ast.walk(expression.args[0])
        )
    )


def _stored_bf16_name(
    expression: ast.expr, definitions: dict[str, ast.expr]
) -> str | None:
    """Recover a scalar stored through only BF16 rounding and Uint16 bitcast."""
    expression = _inline_names(expression, definitions)
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "bitcast"
        and len(expression.args) == 1
        and not expression.keywords
        and _call_path(expression.args[0]) == ("cutlass", "Uint16")
    ):
        return None
    value = expression.func.value
    saw_bf16_cast = False
    while (
        isinstance(value, ast.Call)
        and _call_path(value.func) == ("cutlass", "BFloat16")
        and len(value.args) == 1
        and not value.keywords
    ):
        saw_bf16_cast = True
        value = value.args[0]
    return value.id if saw_bf16_cast and isinstance(value, ast.Name) else None


def _find_owner(
    statements: list[ast.stmt], target: ast.stmt
) -> tuple[list[ast.stmt], int] | None:
    for index, statement in enumerate(statements):
        if statement is target:
            return statements, index
        for _field, value in ast.iter_fields(statement):
            if isinstance(value, list) and all(
                isinstance(item, ast.stmt) for item in value
            ):
                found = _find_owner(value, target)
                if found is not None:
                    return found
    return None


def _lexical_statements_before(
    statements: list[ast.stmt], target: ast.stmt
) -> list[ast.stmt] | None:
    """Collect statements that can define names before a nested target.

    The returned prefix follows the lexical path to ``target``.  Compound
    statements that precede that path remain intact so the local analyses can
    conservatively invalidate names written under control flow.
    """
    for index, statement in enumerate(statements):
        if statement is target:
            return list(statements[:index])
        for _field, value in ast.iter_fields(statement):
            if not (
                isinstance(value, list)
                and all(isinstance(item, ast.stmt) for item in value)
            ):
                continue
            nested = _lexical_statements_before(value, target)
            if nested is not None:
                return [*statements[:index], *nested]
    return None


def _store_call(statement: ast.stmt) -> ast.Call | None:
    matches = [
        node
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
    ]
    return matches[0] if len(matches) == 1 else None


def _state_store_call(statement: ast.stmt) -> ast.Call | None:
    matches = [
        node
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"_cute_store_u16_vec", "_cute_store_u16x8_l2_evict_last"}
    ]
    return matches[0] if len(matches) == 1 else None


def _substitute_names(node: ast.expr, replacements: dict[str, ast.expr]) -> ast.expr:
    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id in replacements:
                result = _clone_ast(replacements[node.id])
                assert isinstance(result, ast.expr)
                return result
            return node

    cloned = _clone_ast(node)
    assert isinstance(cloned, ast.expr)
    result = Substitute().visit(cloned)
    assert isinstance(result, ast.expr)
    return result


def _substitute_names_in_statement(
    node: ast.stmt, name: str, replacement: ast.expr
) -> ast.stmt:
    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id == name:
                return _clone_expression(replacement)
            return node

    result = Substitute().visit(node)
    assert isinstance(result, ast.stmt)
    return result


def _inline_names(node: ast.expr, definitions: dict[str, ast.expr]) -> ast.expr:
    result = _clone_ast(node)
    assert isinstance(result, ast.expr)
    for _ in range(len(definitions) + 1):
        before = ast.dump(result, include_attributes=False)
        result = _substitute_names(result, definitions)
        if ast.dump(result, include_attributes=False) == before:
            return result
    return result


class _ReplaceShuffle(ast.NodeTransformer):
    def __init__(self) -> None:
        self.count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        visited = self.generic_visit(node)
        assert isinstance(visited, ast.Call)
        if _call_path(visited.func) == ("cute", "arch", "shuffle_sync"):
            if len(visited.args) != 2 or visited.keywords:
                return visited
            self.count += 1
            result = _clone_ast(visited.args[0])
            assert isinstance(result, ast.expr)
            return result
        return visited


def _source_expression(
    expression: ast.expr, element_name: str, replacement: ast.expr
) -> ast.expr | None:
    replacer = _ReplaceShuffle()
    cloned = _clone_ast(expression)
    assert isinstance(cloned, ast.expr)
    result = replacer.visit(cloned)
    if replacer.count != 1 or not isinstance(result, ast.expr):
        return None
    return _substitute_names(result, {element_name: replacement})


_UNIFORM_GLOBALS = frozenset({"cutlass", "cute", "ir", "operator"})


def _uniform_names_before(
    statements: list[ast.stmt], end: int, initial: frozenset[str]
) -> set[str]:
    """Conservatively propagate scalar values uniform across a warp."""
    uniform = set(initial) | set(_UNIFORM_GLOBALS)
    for statement in statements[:end]:
        # A write hidden under control flow is not a straight-line definition
        # that this local proof can model.  Invalidate every such name before
        # considering the one direct assignment form below.
        uniform.difference_update(_assigned_names(statement))
        target = _name_target(statement)
        if target is None or not isinstance(statement, ast.Assign):
            continue
        if not _expression_is_pure(statement.value):
            continue
        warp_reductions = _calls(
            statement.value, ("cute", "arch", "warp_reduction_sum")
        )
        if warp_reductions and all(
            any(
                keyword.arg == "threads_in_group"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == 16
                for keyword in call.keywords
            )
            for call in warp_reductions
        ):
            uniform.add(target)
            continue
        if _calls(statement.value, ("cute", "arch", "thread_idx")) or _calls(
            statement.value, ("cute", "arch", "lane_idx")
        ):
            continue
        if any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "load"
            for call in (
                node for node in ast.walk(statement.value) if isinstance(node, ast.Call)
            )
        ):
            continue
        if _loaded_names(statement.value) <= uniform:
            uniform.add(target)
    return uniform


def _packed_source_is_safe(
    original: ast.expr,
    packed: ast.expr,
    *,
    loop_var: str,
    replacement_name: str,
    uniform_names: set[str],
) -> bool:
    shuffles = _calls(original, ("cute", "arch", "shuffle_sync"))
    if len(shuffles) != 1:
        return False
    source = _substitute_names(
        shuffles[0].args[0],
        {loop_var: ast.Name(id=replacement_name, ctx=ast.Load())},
    )
    source_names = _loaded_names(source)
    moved_names = _loaded_names(packed) - source_names - _UNIFORM_GLOBALS
    return (
        moved_names <= uniform_names
        and not _calls(original, ("cutlass", "Float16"))
        and not _calls(packed, ("cutlass", "Float16"))
        and not _calls(packed, ("cute", "arch", "thread_idx"))
        and not _calls(packed, ("cute", "arch", "lane_idx"))
    )


def _multiplication_factors(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return [
            *_multiplication_factors(node.left),
            *_multiplication_factors(node.right),
        ]
    return [node]


def _multiply(factors: list[ast.expr]) -> ast.expr:
    assert factors
    result = _clone_expression(factors[0])
    for factor in factors[1:]:
        result = ast.BinOp(
            left=result,
            op=ast.Mult(),
            right=_clone_expression(factor),
        )
    return result


def _hoist_common_scale(
    low: ast.expr,
    high: ast.expr,
    *,
    low_name: str,
    high_name: str,
    scale_name: str,
) -> tuple[ast.expr, ast.expr, ast.expr] | None:
    low_factors = _multiplication_factors(low)
    high_factors = _multiplication_factors(high)
    if len(low_factors) != len(high_factors):
        return None
    low_source = [factor for factor in low_factors if low_name in _loaded_names(factor)]
    high_source = [
        factor for factor in high_factors if high_name in _loaded_names(factor)
    ]
    low_uniform = [
        factor for factor in low_factors if low_name not in _loaded_names(factor)
    ]
    high_uniform = [
        factor for factor in high_factors if high_name not in _loaded_names(factor)
    ]
    if (
        len(low_source) != 1
        or len(high_source) != 1
        or len(low_uniform) < 2
        or len(low_uniform) != len(high_uniform)
        or any(
            not _same_expression(left, right)
            for left, right in zip(low_uniform, high_uniform, strict=True)
        )
    ):
        return None
    scale = _multiply(low_uniform)
    return (
        ast.BinOp(
            left=_clone_expression(low_source[0]),
            op=ast.Mult(),
            right=ast.Name(id=scale_name, ctx=ast.Load()),
        ),
        ast.BinOp(
            left=_clone_expression(high_source[0]),
            op=ast.Mult(),
            right=ast.Name(id=scale_name, ctx=ast.Load()),
        ),
        scale,
    )


def _parse_statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def _replace_store_value(statement: ast.stmt, expression: ast.expr) -> ast.stmt:
    result = _clone_ast(statement)
    assert isinstance(result, ast.stmt)
    call = _store_call(result)
    assert call is not None and len(call.args) == 1 and not call.keywords
    call.args[0] = expression
    return result


def _replace_state_store(
    statement: ast.stmt, helper: str, pointer: ast.expr, values_name: str
) -> ast.stmt:
    result = _clone_ast(statement)
    assert isinstance(result, ast.stmt)
    call = _state_store_call(result)
    assert call is not None
    call.func = ast.Name(id=helper, ctx=ast.Load())
    call.args = [
        _clone_expression(pointer),
        ast.Name(id=values_name, ctx=ast.Load()),
    ]
    return result


def _backward_slice(
    statements: list[ast.stmt], required_name: str, defined_name: str
) -> list[ast.stmt] | None:
    required = {required_name}
    selected: list[ast.stmt] = []
    for statement in reversed(statements):
        written = _assigned_names(statement)
        if not (required & written):
            continue
        cloned = _clone_ast(statement)
        assert isinstance(cloned, ast.stmt)
        selected.append(cloned)
        required.difference_update(written)
        required.update(_loaded_names(statement))
    # The new packed reduction supplies this value.  Everything else in the
    # slice must be defined within the selected statements or outside the row.
    required.discard(defined_name)
    if required_name in required:
        return None
    selected.reverse()
    return selected


def _is_uint16_vector_type(call: ast.Call, width: int) -> bool:
    if _call_path(call.func) != ("ir", "VectorType", "get") or len(call.args) != 2:
        return False
    shape, dtype = call.args
    return (
        isinstance(shape, ast.List)
        and len(shape.elts) == 1
        and isinstance(shape.elts[0], ast.Constant)
        and shape.elts[0].value == width
        and _call_path(dtype) == ("cutlass", "Uint16", "mlir_type")
    )


@dataclass(frozen=True)
class _Match:
    root_body: list[ast.stmt]
    row: ast.For
    owner: list[ast.stmt]
    row_index: int
    first_index: int
    update_index: int
    output_index: int
    store_index: int
    state_vector: str
    pipe_vector: str
    pipe_loads: tuple[ast.Assign, ast.Assign]
    key_cache: str
    query_cache: str
    decay_cache: str
    decayed_cache: str
    prediction_name: str
    base_name: str
    qdot_name: str
    delta_name: str
    pair_loop_var: str
    key_pair_expression: ast.expr
    query_pair_expression: ast.expr
    decay_pair_expression: ast.expr
    first_loop: ast.For
    update_loop: ast.For
    output_statement: ast.stmt
    state_store_statement: ast.stmt
    state_pointer: ast.expr
    state_store_helper: str


def _match(body: list[ast.stmt]) -> _Match | None:
    rows = [
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
        and getattr(node, "_helion_async_state_vector", None) is not None
        and getattr(node, "_helion_async_state_vector_width", None) == 8
    ]
    if len(rows) != 1:
        return None
    row = rows[0]
    state_vector = row.__dict__["_helion_async_state_vector"]
    if not isinstance(state_vector, str):
        return None
    owner_info = _find_owner(body, row)
    if owner_info is None:
        return None
    owner, row_index = owner_info

    pairs: list[tuple[int, ast.For, int, int, int]] = []
    for index, statement in enumerate(row.body):
        if not isinstance(statement, ast.For) or _constexpr_trip_count(statement) != 4:
            continue
        mul_count = len(_calls(statement, ("cute", "arch", "mul_packed_f32x2")))
        fma_count = len(_calls(statement, ("cute", "arch", "fma_packed_f32x2")))
        append_count = sum(
            1
            for call in (
                child for child in ast.walk(statement) if isinstance(child, ast.Call)
            )
            if isinstance(call.func, ast.Attribute) and call.func.attr == "append"
        )
        pairs.append((index, statement, mul_count, fma_count, append_count))
    firsts = [item for item in pairs if item[2:] == (1, 2, 0)]
    updates = [item for item in pairs if item[2:] == (1, 1, 2)]
    if len(firsts) != 1 or len(updates) != 1 or firsts[0][0] >= updates[0][0]:
        return None
    first_index, first_loop, _, _, _ = firsts[0]
    update_index, update_loop, _, _, _ = updates[0]

    first_mul = _calls(first_loop, ("cute", "arch", "mul_packed_f32x2"))[0]
    first_fmas = _calls(first_loop, ("cute", "arch", "fma_packed_f32x2"))
    update_mul = _calls(update_loop, ("cute", "arch", "mul_packed_f32x2"))[0]
    update_fma = _calls(update_loop, ("cute", "arch", "fma_packed_f32x2"))[0]
    if not (
        len(first_mul.args) == 2
        and len(update_fma.args) == 3
        and all(len(call.args) == 3 for call in first_fmas)
    ):
        return None
    assert isinstance(first_loop.target, ast.Name)
    assert isinstance(update_loop.target, ast.Name)
    first_loop_var = first_loop.target.id
    update_loop_var = update_loop.target.id
    first_definitions: dict[str, ast.expr] = {}
    for statement in first_loop.body:
        target = _name_target(statement)
        if target is not None and isinstance(statement, ast.Assign):
            first_definitions[target] = _inline_names(
                statement.value, first_definitions
            )
    update_definitions: dict[str, ast.expr] = {}
    for statement in update_loop.body:
        target = _name_target(statement)
        if target is not None and isinstance(statement, ast.Assign):
            update_definitions[target] = _inline_names(
                statement.value, update_definitions
            )
    state_base = _state_pair_cache(
        first_mul.args[0],
        loop_var=first_loop_var,
        definitions=first_definitions,
    )
    decay_cache = _pair_cache(
        first_mul.args[1],
        loop_var=first_loop_var,
        definitions=first_definitions,
    )
    key_cache = _pair_cache(
        update_fma.args[1],
        loop_var=update_loop_var,
        definitions=update_definitions,
    )
    delta_pair = _name_pair(_inline_names(update_fma.args[0], update_definitions))
    delta_name = (
        delta_pair[0]
        if delta_pair is not None and delta_pair[0] == delta_pair[1]
        else None
    )
    if None in {state_base, decay_cache, key_cache, delta_name}:
        return None
    if state_base != state_vector:
        return None

    fma_caches = [
        _pair_cache(
            call.args[1],
            loop_var=first_loop_var,
            definitions=first_definitions,
        )
        for call in first_fmas
    ]
    if any(cache is None for cache in fma_caches) or key_cache not in fma_caches:
        return None
    query_candidates = [cache for cache in fma_caches if cache != key_cache]
    if len(query_candidates) != 1 or not isinstance(query_candidates[0], str):
        return None
    query_cache = query_candidates[0]
    key_fma = first_fmas[fma_caches.index(key_cache)]
    query_fma = first_fmas[fma_caches.index(query_cache)]
    if not _same_expression(key_fma.args[0], query_fma.args[0]):
        return None
    decayed_pair = _tuple_target_names(first_mul, first_loop)
    prediction_acc = _tuple_target_names(key_fma, first_loop)
    base_acc = _tuple_target_names(query_fma, first_loop)
    if (
        decayed_pair is None
        or prediction_acc is None
        or base_acc is None
        or prediction_acc == base_acc
        or _name_pair(key_fma.args[0]) != decayed_pair
        or _name_pair(query_fma.args[0]) != decayed_pair
        or _name_pair(key_fma.args[2]) != prediction_acc
        or _name_pair(query_fma.args[2]) != base_acc
        or not _zero_initialized_before(
            row.body,
            first_index,
            (*prediction_acc, *base_acc),
        )
    ):
        return None

    # The row's temporary cache is written from the state load in pass one and
    # read in pass two.  It becomes the packed decayed-state cache.  Prove both
    # lane writes exactly; merely noticing that a cache mentions the state is
    # insufficient because a swapped or later overwrite changes the recurrence.
    state_pair_expression = _inline_names(first_mul.args[0], first_definitions)
    assert isinstance(state_pair_expression, (ast.Tuple, ast.List))
    expected_indices = _expected_pair_indices(first_loop_var)
    cache_writes: dict[str, list[tuple[object | None, ast.expr]]] = {}
    for statement in first_loop.body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Subscript)
            and isinstance(statement.targets[0].value, ast.Name)
        ):
            continue
        target = statement.targets[0]
        assert isinstance(target, ast.Subscript)
        assert isinstance(target.value, ast.Name)
        cache_writes.setdefault(target.value.id, []).append(
            (
                _canonical_index(_inline_names(target.slice, first_definitions)),
                _unwrap_value_casts(_inline_names(statement.value, first_definitions)),
            )
        )
    expected_state_writes = [
        (expected_indices[lane], _unwrap_value_casts(value))
        for lane, value in enumerate(state_pair_expression.elts)
    ]
    written_caches = {
        name
        for name, writes in cache_writes.items()
        if len(writes) == 2
        and all(
            index == expected_index and _same_expression(value, expected_value)
            for (index, value), (expected_index, expected_value) in zip(
                writes, expected_state_writes, strict=True
            )
        )
    }
    update_reads = _loaded_names(update_loop)
    decayed_candidates = written_caches & update_reads
    if len(decayed_candidates) != 1:
        return None
    decayed_cache = next(iter(decayed_candidates))
    if len({key_cache, query_cache, decay_cache, decayed_cache}) != 4:
        return None
    update_decayed_pair = _tuple_target_names(update_mul, update_loop)
    updated_pair = _tuple_target_names(update_fma, update_loop)
    if (
        update_decayed_pair is None
        or updated_pair is None
        or _name_pair(update_fma.args[2]) != update_decayed_pair
        or {
            _pair_cache(
                update_mul.args[0],
                loop_var=update_loop_var,
                definitions=update_definitions,
            ),
            _pair_cache(
                update_mul.args[1],
                loop_var=update_loop_var,
                definitions=update_definitions,
            ),
        }
        != {decayed_cache, decay_cache}
        or not _same_pair_expression(
            key_fma.args[1],
            first_definitions,
            first_loop_var,
            update_fma.args[1],
            update_definitions,
            update_loop_var,
        )
    ):
        return None

    prediction_name = _warp_reduction_target(
        row.body, first_index + 1, update_index, prediction_acc
    )
    base_name = _warp_reduction_target(
        row.body, first_index + 1, update_index, base_acc
    )
    if prediction_name is None or base_name is None or prediction_name == base_name:
        return None

    qdot_candidates: list[str] = []
    for index, statement in enumerate(owner[:row_index]):
        if not isinstance(statement, ast.For) or _constexpr_trip_count(statement) != 4:
            continue
        assert isinstance(statement.target, ast.Name)
        loop_var = statement.target.id
        calls = _calls(statement, ("cute", "arch", "fma_packed_f32x2"))
        if len(calls) != 1 or len(calls[0].args) != 3:
            continue
        definitions: dict[str, ast.expr] = {}
        for child in statement.body:
            target = _name_target(child)
            if target is not None and isinstance(child, ast.Assign):
                definitions[target] = _inline_names(child.value, definitions)
        pair_caches = [
            _pair_cache(
                argument,
                loop_var=loop_var,
                definitions=definitions,
            )
            for argument in calls[0].args[:2]
        ]
        caches = set(pair_caches)
        accumulators = _tuple_target_names(calls[0], statement)
        if caches != {key_cache, query_cache} or accumulators is None:
            continue
        expressions_by_cache = dict(zip(pair_caches, calls[0].args[:2], strict=True))
        if not _same_pair_expression(
            key_fma.args[1],
            first_definitions,
            first_loop_var,
            expressions_by_cache[key_cache],
            definitions,
            loop_var,
        ) or not _same_pair_expression(
            query_fma.args[1],
            first_definitions,
            first_loop_var,
            expressions_by_cache[query_cache],
            definitions,
            loop_var,
        ):
            continue
        if not _zero_initialized_before(owner, index, accumulators):
            continue
        target = _warp_reduction_target(owner, index + 1, row_index, accumulators)
        if target is not None:
            qdot_candidates.append(target)
    if len(qdot_candidates) != 1:
        return None
    qdot_name = qdot_candidates[0]

    output_sites = [
        (index, statement)
        for index, statement in enumerate(
            row.body[first_index + 1 : update_index], first_index + 1
        )
        if _store_call(statement) is not None
    ]
    state_sites = [
        (index, statement, _state_store_call(statement))
        for index, statement in enumerate(
            row.body[update_index + 1 :], update_index + 1
        )
        if _state_store_call(statement) is not None
    ]
    if len(output_sites) != 1 or len(state_sites) != 1:
        return None
    output_index, output_statement = output_sites[0]
    store_index, state_store_statement, state_call = state_sites[0]
    assert state_call is not None
    if len(state_call.args) != 2 or state_call.keywords:
        return None
    if not isinstance(state_call.args[1], ast.Name):
        return None
    store_values_name = state_call.args[1].id
    append_values = [
        call.args[0]
        for call in (
            node for node in ast.walk(update_loop) if isinstance(node, ast.Call)
        )
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == store_values_name
        and len(call.args) == 1
        and not call.keywords
    ]
    if (
        len(append_values) != 2
        or tuple(
            _stored_bf16_name(value, update_definitions) for value in append_values
        )
        != updated_pair
        or not _recurrence_loop_is_pure(
            first_loop,
            subscript_containers=frozenset((decayed_cache,)),
            append_containers=frozenset(),
        )
        or not _recurrence_loop_is_pure(
            update_loop,
            subscript_containers=frozenset(),
            append_containers=frozenset((store_values_name,)),
        )
        or not _has_unique_name_writes(first_loop)
        or not _has_unique_name_writes(update_loop)
    ):
        return None
    output_call = _store_call(output_statement)
    assert output_call is not None
    if len(output_call.args) != 1 or output_call.keywords:
        return None
    output_definitions: dict[str, ast.expr] = {}
    for statement in row.body[first_index + 1 : output_index]:
        target = _name_target(statement)
        if target is not None and isinstance(statement, ast.Assign):
            output_definitions[target] = statement.value
    if not _is_bf16_output(output_call.args[0], output_definitions) or not (
        _is_original_output(
            output_call.args[0],
            output_definitions,
            delta=delta_name,  # type: ignore[arg-type]
            qdot=qdot_name,
            base=base_name,
        )
    ):
        return None

    # The staged vector is copied into the row-local name.  Recover the ring
    # value name so both initial and lookahead LDS vector types can be packed.
    pipe_names = {
        statement.value.id
        for statement in row.body[:first_index]
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == state_vector
        and isinstance(statement.value, ast.Name)
    }
    if len(pipe_names) != 1:
        return None
    pipe_vector = next(iter(pipe_names))
    body_module = ast.Module(body=body, type_ignores=[])
    pipe_loads = [
        assignment
        for assignment in (
            node for node in ast.walk(body_module) if isinstance(node, ast.Assign)
        )
        if _name_target(assignment) == pipe_vector
        and isinstance(assignment.value, ast.Call)
        and _call_path(assignment.value.func) == ("cute", "arch", "load")
        and len(assignment.value.args) == 2
        and isinstance(assignment.value.args[1], ast.Call)
        and _is_uint16_vector_type(assignment.value.args[1], 8)
    ]
    pipe_aliases = [
        statement
        for statement in row.body[:first_index]
        if isinstance(statement, ast.Assign)
        and _name_target(statement) == state_vector
        and isinstance(statement.value, ast.Name)
        and statement.value.id == pipe_vector
    ]
    if (
        len(pipe_loads) != 2
        or len(pipe_aliases) != 1
        or sum(assignment in owner[:row_index] for assignment in pipe_loads) != 1
        or sum(assignment in ast.walk(row) for assignment in pipe_loads) != 1
        or _name_loads(body_module, pipe_vector)
        != _name_loads(pipe_aliases[0], pipe_vector)
    ):
        return None
    rewritten_region = ast.Module(
        body=row.body[first_index : store_index + 1], type_ignores=[]
    )
    if not _loads_only_within(body_module, qdot_name, rewritten_region) or not (
        _loads_only_within(body_module, decayed_cache, rewritten_region)
    ):
        return None
    return _Match(
        root_body=body,
        row=row,
        owner=owner,
        row_index=row_index,
        first_index=first_index,
        update_index=update_index,
        output_index=output_index,
        store_index=store_index,
        state_vector=state_vector,
        pipe_vector=pipe_vector,
        pipe_loads=(pipe_loads[0], pipe_loads[1]),
        key_cache=key_cache,  # type: ignore[arg-type]
        query_cache=query_cache,
        decay_cache=decay_cache,  # type: ignore[arg-type]
        decayed_cache=decayed_cache,
        prediction_name=prediction_name,
        base_name=base_name,
        qdot_name=qdot_name,
        delta_name=delta_name,  # type: ignore[arg-type]
        pair_loop_var=first_loop_var,
        key_pair_expression=_inline_names(key_fma.args[1], first_definitions),
        query_pair_expression=_inline_names(query_fma.args[1], first_definitions),
        decay_pair_expression=_inline_names(first_mul.args[1], first_definitions),
        first_loop=first_loop,
        update_loop=update_loop,
        output_statement=output_statement,
        state_store_statement=state_store_statement,
        state_pointer=_clone_expression(state_call.args[0]),
        state_store_helper=state_call.func.id,  # type: ignore[union-attr]
    )


def _find_allocations(
    body: list[ast.stmt],
    dtypes: dict[str, str | frozenset[str]],
    *,
    width: int,
) -> dict[str, ast.Call] | None:
    found: dict[str, ast.Call] = {}
    for statement in body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in dtypes:
                continue
            if target.id in found:
                return None
            value = node.value
            expected_dtypes = dtypes[target.id]
            if isinstance(expected_dtypes, str):
                expected_dtypes = frozenset((expected_dtypes,))
            if not (
                isinstance(value, ast.Call)
                and _call_path(value.func) == ("cute", "make_rmem_tensor")
                and len(value.args) == 2
                and isinstance(value.args[0], ast.Constant)
                and value.args[0].value == width
                and _call_path(value.args[1])
                in {("cutlass", dtype) for dtype in expected_dtypes}
            ):
                return None
            found[target.id] = value
    if set(found) != set(dtypes):
        return None
    return found


def _replace_allocations(
    body: list[ast.stmt], dtypes: dict[str, str | frozenset[str]]
) -> bool:
    found = _find_allocations(body, dtypes, width=8)
    if found is None:
        return False
    for value in found.values():
        value.args = [
            ast.Constant(value=4),
            ast.Attribute(
                value=ast.Name(id="cutlass", ctx=ast.Load()),
                attr="Uint32",
                ctx=ast.Load(),
            ),
        ]
    return True


def _pack_expansion(
    body: list[ast.stmt],
    match: _Match,
    prefix: str,
    initial_uniform_names: frozenset[str],
) -> bool:
    if not (
        _pair_is_direct_cache(match.key_pair_expression, match.key_cache)
        and _pair_is_direct_cache(match.query_pair_expression, match.query_cache)
        and _pair_is_direct_cache(match.decay_pair_expression, match.decay_cache)
    ):
        return False
    candidates: list[
        tuple[int, ast.For, dict[str, ast.expr], str, ast.expr, ast.expr, str]
    ] = []
    cache_names = {match.key_cache, match.query_cache, match.decay_cache}
    row_index = match.owner.index(match.row)
    for index, statement in enumerate(match.owner[:row_index]):
        if not isinstance(statement, ast.For) or _constexpr_trip_count(statement) != 8:
            continue
        if not isinstance(statement.target, ast.Name):
            continue
        loop_var = statement.target.id
        definitions: dict[str, ast.expr] = {}
        writes: dict[str, ast.expr] = {}
        element_name: str | None = None
        lane_expression: ast.expr | None = None
        for child in statement.body:
            target = _name_target(child)
            if target is not None and isinstance(child, ast.Assign):
                definitions[target] = _inline_names(child.value, definitions)
                if (
                    isinstance(child.value, ast.BinOp)
                    and isinstance(child.value.op, ast.Mod)
                    and isinstance(child.value.left, ast.Name)
                    and child.value.left.id == loop_var
                    and isinstance(child.value.right, ast.Constant)
                    and child.value.right.value == 4
                ):
                    element_name = target
                if _calls(child.value, ("cute", "arch", "thread_idx")) and any(
                    isinstance(node, ast.Name) and node.id == loop_var
                    for node in ast.walk(child.value)
                ):
                    lane_expression = definitions[target]
            if not isinstance(child, ast.Assign) or len(child.targets) != 1:
                continue
            target_node = child.targets[0]
            if (
                isinstance(target_node, ast.Subscript)
                and isinstance(target_node.value, ast.Name)
                and target_node.value.id in cache_names
            ):
                if _canonical_index(target_node.slice) != ("name", loop_var):
                    return False
                writes[target_node.value.id] = _inline_names(child.value, definitions)
        if (
            set(writes) == cache_names
            and element_name is not None
            and lane_expression is not None
        ):
            candidates.append(
                (
                    index,
                    statement,
                    writes,
                    element_name,
                    definitions[element_name],
                    lane_expression,
                    loop_var,
                )
            )
    if len(candidates) != 1:
        return False
    (
        index,
        loop,
        writes,
        element_name,
        element_expression,
        lane_expression,
        loop_var,
    ) = candidates[0]
    if not _cache_expansion_loop_is_pure(loop, cache_names):
        return False
    if _canonical_index(element_expression) != (
        "modulo",
        ("name", loop_var),
        ("constant", 4),
    ) or _canonical_index(lane_expression) != (
        "add",
        ("floor_divide", ("name", loop_var), ("constant", 4)),
        ("multiply", ("constant", 2), ("thread", 0)),
    ):
        return False
    lexical_prefix = _lexical_statements_before(match.root_body, loop)
    if lexical_prefix is None:
        return False
    uniform_names = _uniform_names_before(
        lexical_prefix, len(lexical_prefix), initial_uniform_names
    )
    packed_expressions: dict[str, tuple[ast.expr, ast.expr]] = {}
    for cache_name, expression in writes.items():
        shuffles = _calls(expression, ("cute", "arch", "shuffle_sync"))
        if (
            len(shuffles) != 1
            or len(shuffles[0].args) != 2
            or shuffles[0].keywords
            or _canonical_index(shuffles[0].args[1])
            != _canonical_index(lane_expression)
        ):
            return False
        low = _source_expression(
            expression,
            loop_var,
            ast.Name(id=f"{prefix}source_low", ctx=ast.Load()),
        )
        high = _source_expression(
            expression,
            loop_var,
            ast.Name(id=f"{prefix}source_high", ctx=ast.Load()),
        )
        if low is None or high is None:
            return False
        if not _packed_source_is_safe(
            expression,
            low,
            loop_var=loop_var,
            replacement_name=f"{prefix}source_low",
            uniform_names=uniform_names,
        ) or not _packed_source_is_safe(
            expression,
            high,
            loop_var=loop_var,
            replacement_name=f"{prefix}source_high",
            uniform_names=uniform_names,
        ):
            return False
        packed_expressions[cache_name] = low, high

    cache_order = [match.key_cache, match.query_cache, match.decay_cache]
    scale_assignments: list[str] = []
    query_low, query_high = packed_expressions[match.query_cache]
    query_scale_name = f"{prefix}query_scale"
    hoisted_query_scale = _hoist_common_scale(
        query_low,
        query_high,
        low_name=f"{prefix}source_low",
        high_name=f"{prefix}source_high",
        scale_name=query_scale_name,
    )
    if hoisted_query_scale is not None:
        query_low, query_high, query_scale = hoisted_query_scale
        packed_expressions[match.query_cache] = query_low, query_high
        scale_assignments.append(f"{query_scale_name} = {ast.unparse(query_scale)}")
    source_names = {
        cache: f"{prefix}source_{position}"
        for position, cache in enumerate(cache_order)
    }
    source = [
        f"{prefix}abi_version = {_ABI_VERSION}",
        *scale_assignments,
        *[
            f"{source_names[cache]} = cute.make_rmem_tensor(2, cutlass.Uint32)"
            for cache in cache_order
        ],
        f"for {prefix}source_pair in cutlass.range_constexpr(2):\n"
        f"    {prefix}source_low = {prefix}source_pair * 2\n"
        f"    {prefix}source_high = {prefix}source_low + 1\n"
        + "\n".join(
            f"    {source_names[cache]}[{prefix}source_pair] = {_PACK}(\n"
            f"        {ast.unparse(packed_expressions[cache][0])},\n"
            f"        {ast.unparse(packed_expressions[cache][1])},\n"
            "    )"
            for cache in cache_order
        ),
        f"for {prefix}destination_pair in cutlass.range_constexpr(4):\n"
        f"    {prefix}source_lane = (\n"
        "        cutlass.Int32(cute.arch.thread_idx()[0]) * 2\n"
        f"        + cutlass.Int32({prefix}destination_pair // 2)\n"
        "    )\n"
        f"    {prefix}source_pair = {prefix}destination_pair % 2\n"
        + "\n".join(
            f"    {cache}[{prefix}destination_pair] = cutlass.Uint32(\n"
            "        cute.arch.shuffle_sync(\n"
            f"            {source_names[cache]}[{prefix}source_pair],\n"
            f"            {prefix}source_lane,\n"
            "        )\n"
            "    )"
            for cache in cache_order
        ),
    ]
    replacement: list[ast.stmt] = []
    for fragment in source:
        replacement.extend(_parse_statements(fragment))
    body_module = ast.Module(body=body, type_ignores=[])
    rewritten_region = ast.Module(
        body=[loop, *match.row.body[match.first_index : match.store_index + 1]],
        type_ignores=[],
    )
    if any(
        not _loads_only_within(body_module, cache, rewritten_region)
        for cache in cache_names
    ) or any(
        not _loads_only_within(body_module, name, loop)
        for name in _assigned_names(loop) - cache_names
    ):
        return False
    expected_dtypes: dict[str, str | frozenset[str]] = {
        match.key_cache: frozenset(("BFloat16", "Float32")),
        match.query_cache: frozenset(("BFloat16", "Float32")),
        match.decay_cache: frozenset(("BFloat16", "Float32")),
    }
    if match.decayed_cache in expected_dtypes:
        return False
    expected_dtypes[match.decayed_cache] = "BFloat16"
    if not _replace_allocations(body, expected_dtypes):
        return False
    match.owner[index : index + 1] = replacement
    return True


def _has_combined_cache_expansion(match: _Match) -> bool:
    cache_names = {match.key_cache, match.query_cache, match.decay_cache}
    row_index = match.owner.index(match.row)
    for statement in match.owner[:row_index]:
        if not isinstance(statement, ast.For) or _constexpr_trip_count(statement) != 8:
            continue
        written_caches = {
            target.value.id
            for child in statement.body
            if isinstance(child, ast.Assign)
            for target in child.targets
            if isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id in cache_names
        }
        if written_caches == cache_names:
            return True
    return False


@dataclass(frozen=True)
class _Producer:
    loop: ast.For
    loop_index: int
    vector_load: ast.Assign
    vector_index: int
    base_name: str
    accumulator: str | None
    reduction: ast.Assign | None


def _producer_loop_is_pure(loop: ast.For, cache_name: str) -> bool:
    for statement in loop.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return False
        target = statement.targets[0]
        if isinstance(target, ast.Subscript):
            if not (
                isinstance(target.value, ast.Name) and target.value.id == cache_name
            ):
                return False
        elif not isinstance(target, ast.Name):
            return False
        for call in (
            node for node in ast.walk(statement.value) if isinstance(node, ast.Call)
        ):
            path = _call_path(call.func)
            if path is not None and (
                path[0] in {"cutlass", "operator"} or path[:2] == ("cute", "math")
            ):
                continue
            if isinstance(call.func, ast.Attribute) and call.func.attr in {
                "bitcast",
                "load",
            }:
                continue
            return False
    return True


def _reaching_definitions_before(
    statements: list[ast.stmt], end: int
) -> tuple[dict[str, ast.expr], frozenset[str]]:
    """Collect straight-line reaching definitions and every earlier write.

    Half-warp producer sharing combines values from two physical thread rows.
    A source address is therefore safe only when every earlier local on which it
    depends can be expanded far enough to prove that it has no hidden thread
    coordinate.  Unknown kernel arguments remain admissible because they are
    uniform; unresolved locals fail closed via the returned write set.
    """
    definitions: dict[str, ast.expr] = {}
    written: set[str] = set()
    for statement in statements[:end]:
        assigned = _assigned_names(statement)
        written.update(assigned)
        for name in assigned:
            definitions.pop(name, None)
        target = _name_target(statement)
        if target is not None and isinstance(statement, ast.Assign):
            definitions[target] = statement.value
    return definitions, frozenset(written)


def _source_address_is_half_warp_uniform(
    node: ast.expr,
    *,
    base_name: str,
    definitions: dict[str, ast.expr],
    written_names: frozenset[str],
) -> bool:
    """Prove an address is CTA-uniform apart from the canonical x-lane base."""
    without_base = _substitute_names(node, {base_name: ast.Constant(value=0)})
    expanded = _inline_names(without_base, definitions)
    if _loaded_names(expanded) & written_names:
        return False
    return not any(
        _call_path(call.func)
        in {
            ("cute", "arch", "lane_idx"),
            ("cute", "arch", "thread_idx"),
            ("cute", "arch", "warp_idx"),
        }
        for call in ast.walk(expanded)
        if isinstance(call, ast.Call)
    )


def _vector_high_bound_literal(
    pointer: ast.expr, base_name: str, width: int
) -> ast.Constant | None:
    """Find the exact generated ``base + width - 1 < bound`` guard."""
    expected_left = (
        "add",
        ("constant", width - 1),
        ("name", base_name),
    )
    candidates: list[ast.Constant] = []
    for comparison in (
        node for node in ast.walk(pointer) if isinstance(node, ast.Compare)
    ):
        if (
            len(comparison.ops) != 1
            or not isinstance(comparison.ops[0], ast.Lt)
            or len(comparison.comparators) != 1
            or _canonical_index(comparison.left) != expected_left
            or base_name in _loaded_names(comparison.comparators[0])
        ):
            continue
        literals = [
            node
            for node in ast.walk(comparison.left)
            if isinstance(node, ast.Constant)
            and type(node.value) is int
            and node.value == width - 1
        ]
        if len(literals) == 1:
            candidates.append(literals[0])
    return candidates[0] if len(candidates) == 1 else None


def _find_cache_producer(match: _Match, cache_name: str) -> _Producer | None:
    row_index = match.owner.index(match.row)
    candidates: list[tuple[int, ast.For]] = []
    for index, statement in enumerate(match.owner[:row_index]):
        if not isinstance(statement, ast.For) or _constexpr_trip_count(statement) != 8:
            continue
        writes = [
            target
            for child in statement.body
            if isinstance(child, ast.Assign)
            for target in child.targets
            if isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == cache_name
        ]
        if len(writes) == 1:
            candidates.append((index, statement))
    if len(candidates) != 1:
        return None
    loop_index, loop = candidates[0]
    assert isinstance(loop.target, ast.Name)
    loop_var = loop.target.id
    write = next(
        target
        for child in loop.body
        if isinstance(child, ast.Assign)
        for target in child.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == cache_name
    )
    if (
        _canonical_index(write.slice) != ("name", loop_var)
        or not _producer_loop_is_pure(loop, cache_name)
        or _calls(loop, ("cute", "arch", "thread_idx"))
        or _calls(loop, ("cute", "arch", "lane_idx"))
    ):
        return None

    base_names = {
        name
        for name in _loaded_names(loop)
        if any(
            _name_target(statement) == name
            and isinstance(statement, ast.Assign)
            and _canonical_index(statement.value)
            == ("multiply", ("constant", 8), ("thread", 0))
            for statement in match.owner[:loop_index]
        )
    }
    if len(base_names) != 1:
        return None
    base_name = next(iter(base_names))
    vector_candidates: list[tuple[int, ast.Assign]] = []
    for index, statement in enumerate(match.owner[:loop_index]):
        if not isinstance(statement, ast.Assign):
            continue
        target = _name_target(statement)
        if target is None or target not in _loaded_names(loop):
            continue
        calls = _calls(statement.value, ("cute", "arch", "load"))
        if (
            len(calls) == 1
            and len(calls[0].args) == 2
            and isinstance(calls[0].args[1], ast.Call)
            and _is_uint16_vector_type(calls[0].args[1], 8)
            and base_name in _loaded_names(statement.value)
        ):
            vector_candidates.append((index, statement))
    if len(vector_candidates) != 1:
        return None
    vector_index, vector_load = vector_candidates[0]
    vector_name = _name_target(vector_load)
    assert vector_name is not None
    load_call = _calls(vector_load.value, ("cute", "arch", "load"))[0]
    lexical_prefix = _lexical_statements_before(match.root_body, vector_load)
    if lexical_prefix is None:
        return None
    definitions, written_names = _reaching_definitions_before(
        lexical_prefix, len(lexical_prefix)
    )
    written_names |= frozenset(
        _assigned_names(ast.Module(body=match.root_body, type_ignores=[]))
    )
    base_definition = definitions.get(base_name)
    high_bound_literal = _vector_high_bound_literal(load_call.args[0], base_name, 8)
    if not _loads_only_within(
        ast.Module(body=match.root_body, type_ignores=[]), vector_name, loop
    ) or (
        base_definition is None
        or _canonical_index(base_definition)
        != ("multiply", ("constant", 8), ("thread", 0))
        or high_bound_literal is None
        or not _source_address_is_half_warp_uniform(
            load_call.args[0],
            base_name=base_name,
            definitions=definitions,
            written_names=written_names,
        )
    ):
        return None

    self_updates = [
        target
        for statement in loop.body
        if (target := _name_target(statement)) is not None
        and isinstance(statement, ast.Assign)
        and target in _loaded_names(statement.value)
    ]
    expected_reductions = 0 if cache_name == match.decay_cache else 1
    if len(self_updates) != expected_reductions:
        return None
    accumulator = self_updates[0] if self_updates else None
    reduction: ast.Assign | None = None
    if accumulator is not None:
        if not _zero_initialized_before(match.owner, loop_index, (accumulator,)):
            return None
        reductions = [
            statement
            for statement in match.owner[loop_index + 1 : row_index]
            if isinstance(statement, ast.Assign)
            and len(
                calls := _calls(statement.value, ("cute", "arch", "warp_reduction_sum"))
            )
            == 1
            and len(calls[0].args) == 1
            and isinstance(calls[0].args[0], ast.Name)
            and calls[0].args[0].id == accumulator
            and len(calls[0].keywords) == 1
            and calls[0].keywords[0].arg == "threads_in_group"
            and isinstance(calls[0].keywords[0].value, ast.Constant)
            and calls[0].keywords[0].value.value == 16
        ]
        if len(reductions) != 1:
            return None
        reduction = reductions[0]

    allowed_external_names = {cache_name}
    if accumulator is not None:
        allowed_external_names.add(accumulator)
    assigned_temporaries = _assigned_names(loop) - allowed_external_names - {loop_var}
    root = ast.Module(body=match.root_body, type_ignores=[])
    owner_module = ast.Module(body=match.owner, type_ignores=[])
    cache_region = ast.Module(
        body=[loop, match.first_loop, match.update_loop], type_ignores=[]
    )
    if any(
        _loaded_before_redefinition(match.owner, loop_index + 1, name)
        or bool(_name_loads(root, name) - _name_loads(owner_module, name))
        for name in assigned_temporaries
    ) or not _loads_only_within(root, cache_name, cache_region):
        return None
    return _Producer(
        loop=loop,
        loop_index=loop_index,
        vector_load=vector_load,
        vector_index=vector_index,
        base_name=base_name,
        accumulator=accumulator,
        reduction=reduction,
    )


def _resize_allocations(
    body: list[ast.stmt],
    names: set[str],
    *,
    old_width: int,
    new_width: int,
    dtype: str,
) -> bool:
    found = _find_allocations(body, dict.fromkeys(names, dtype), width=old_width)
    if found is None:
        return False
    for value in found.values():
        value.args[0] = ast.Constant(value=new_width)
    return True


def _share_separate_cache_producers(
    body: list[ast.stmt],
    match: _Match,
    prefix: str,
    thread_block_dims: tuple[int, int, int] | None,
) -> bool:
    """Split row-invariant 8-lane producers across paired half-warps."""
    if (
        thread_block_dims is None
        or thread_block_dims[0] != 16
        or thread_block_dims[1] < 2
        or thread_block_dims[1] % 2
        or thread_block_dims[2] != 1
    ):
        return False
    caches = (match.key_cache, match.query_cache, match.decay_cache)
    producers = [_find_cache_producer(match, cache) for cache in caches]
    if any(producer is None for producer in producers):
        return False
    proven = [producer for producer in producers if producer is not None]
    if len({id(producer.loop) for producer in proven}) != len(proven):
        return False
    load_calls = [
        _calls(producer.vector_load.value, ("cute", "arch", "load"))[0]
        for producer in proven
    ]
    high_bound_literals = [
        _vector_high_bound_literal(load_call.args[0], producer.base_name, 8)
        for producer, load_call in zip(proven, load_calls, strict=True)
    ]
    if any(literal is None for literal in high_bound_literals):
        return False
    if not _resize_allocations(
        body,
        set(caches),
        old_width=8,
        new_width=4,
        dtype="Float32",
    ):
        return False

    source_base_name = f"{prefix}producer_base"
    for producer, load_call, high_bound_literal in zip(
        proven, load_calls, high_bound_literals, strict=True
    ):
        assert high_bound_literal is not None
        vector_type = load_call.args[1]
        assert isinstance(vector_type, ast.Call)
        vector_type.args[0] = ast.List(elts=[ast.Constant(value=4)], ctx=ast.Load())
        high_bound_literal.value = 3
        producer.vector_load.value = _substitute_names(
            producer.vector_load.value,
            {
                producer.base_name: ast.Name(
                    id=source_base_name,
                    ctx=ast.Load(),
                )
            },
        )
        replacement_loop = _clone_ast(producer.loop)
        assert isinstance(replacement_loop, ast.For)
        assert isinstance(replacement_loop.iter, ast.Call)
        replacement_loop.iter.args[0] = ast.Constant(value=4)
        replacement_loop = _substitute_names_in_statement(
            replacement_loop,
            producer.base_name,
            ast.Name(id=source_base_name, ctx=ast.Load()),
        )
        match.owner[match.owner.index(producer.loop)] = replacement_loop
        if producer.reduction is not None:
            reduction_call = _calls(
                producer.reduction.value,
                ("cute", "arch", "warp_reduction_sum"),
            )[0]
            reduction_call.keywords[0].value = ast.Constant(value=32)

    insertion_index = min(
        match.owner.index(producer.vector_load) for producer in proven
    )
    match.owner[insertion_index:insertion_index] = _parse_statements(
        f"""
{source_base_name} = (
    (
        cutlass.Int32(cute.arch.thread_idx()[0])
        + cutlass.Int32(cute.arch.thread_idx()[1]) * 16
    ) % 32
) * 4
{prefix}half_warp_producer_abi_version = {_ABI_VERSION}
"""
    )
    return True


def _pack_separate_caches(
    body: list[ast.stmt],
    match: _Match,
    prefix: str,
    initial_uniform_names: frozenset[str],
    insertion_index: int,
    thread_block_dims: tuple[int, int, int] | None,
) -> _Match | None:
    """Pack factored caches after their independent source/reduction loops.

    Generic factor-affine lowering keeps raw key/query/decay fragments in
    separate register tensors and applies normalization at each consumer.  A
    late packed recurrence instead materializes three normalized BF16x2 caches
    once, immediately before the row loop.  The exact pair expressions were
    already proved equal at prediction, update, and qdot sites by ``_match``.
    """
    insertion_anchor = match.owner[insertion_index]
    lexical_prefix = _lexical_statements_before(match.root_body, insertion_anchor)
    if lexical_prefix is None:
        return None
    uniform_names = _uniform_names_before(
        lexical_prefix, len(lexical_prefix), initial_uniform_names
    )
    original_caches = {
        match.key_cache,
        match.query_cache,
        match.decay_cache,
    }
    if (
        _find_allocations(
            body,
            dict.fromkeys(original_caches, frozenset(("BFloat16", "Float32"))),
            width=8,
        )
        is None
    ):
        return None
    pair_expressions = {
        match.key_cache: _clone_expression(match.key_pair_expression),
        match.query_cache: _clone_expression(match.query_pair_expression),
        match.decay_cache: _clone_expression(match.decay_pair_expression),
    }
    for expression in pair_expressions.values():
        moved_names = (
            _loaded_names(expression)
            - original_caches
            - {match.pair_loop_var}
            - _UNIFORM_GLOBALS
        )
        if (
            not moved_names <= uniform_names
            or not _packed_cache_expression_is_pure(expression)
            or _calls(expression, ("cute", "arch", "thread_idx"))
            or _calls(expression, ("cute", "arch", "lane_idx"))
        ):
            return None

    query_pair = pair_expressions[match.query_cache]
    assert isinstance(query_pair, (ast.Tuple, ast.List))
    query_scale_name = f"{prefix}query_scale"
    hoisted_query_scale = _hoist_common_scale(
        _unwrap_value_casts(query_pair.elts[0]),
        _unwrap_value_casts(query_pair.elts[1]),
        low_name=match.query_cache,
        high_name=match.query_cache,
        scale_name=query_scale_name,
    )
    scale_assignment: str | None = None
    if hoisted_query_scale is not None:
        low, high, scale = hoisted_query_scale
        pair_expressions[match.query_cache] = ast.Tuple(
            elts=[low, high], ctx=ast.Load()
        )
        scale_assignment = f"{query_scale_name} = {ast.unparse(scale)}"

    packed_names = {
        match.key_cache: f"{prefix}key_cache",
        match.query_cache: f"{prefix}query_cache",
        match.decay_cache: f"{prefix}decay_cache",
    }
    shared = _share_separate_cache_producers(
        body,
        match,
        prefix,
        thread_block_dims,
    )
    insertion_index = match.owner.index(insertion_anchor)
    source = [
        f"{prefix}abi_version = {_ABI_VERSION}",
        *([scale_assignment] if scale_assignment is not None else []),
        *[
            f"{packed_names[cache]} = cute.make_rmem_tensor(4, cutlass.Uint32)"
            for cache in (
                match.key_cache,
                match.query_cache,
                match.decay_cache,
            )
        ],
    ]
    if shared:
        packed_source_names = {
            cache: f"{prefix}source_{position}"
            for position, cache in enumerate(
                (match.key_cache, match.query_cache, match.decay_cache)
            )
        }
        source.extend(
            f"{packed_source_names[cache]} = cute.make_rmem_tensor(2, cutlass.Uint32)"
            for cache in (match.key_cache, match.query_cache, match.decay_cache)
        )
        source_pair = f"{prefix}source_pair"
        source_loop = [
            f"for {source_pair} in cutlass.range_constexpr(2):",
        ]
        for cache in (match.key_cache, match.query_cache, match.decay_cache):
            expression = pair_expressions[cache]
            assert isinstance(expression, (ast.Tuple, ast.List))
            low = _substitute_names(
                expression.elts[0],
                {match.pair_loop_var: ast.Name(id=source_pair, ctx=ast.Load())},
            )
            high = _substitute_names(
                expression.elts[1],
                {match.pair_loop_var: ast.Name(id=source_pair, ctx=ast.Load())},
            )
            source_loop.extend(
                (
                    f"    {packed_source_names[cache]}[{source_pair}] = {_PACK}(",
                    f"        {ast.unparse(low)},",
                    f"        {ast.unparse(high)},",
                    "    )",
                )
            )
        source.append("\n".join(source_loop))
        destination_pair = f"{prefix}destination_pair"
        source_lane = f"{prefix}source_lane"
        source.append(
            f"for {destination_pair} in cutlass.range_constexpr(4):\n"
            f"    {source_lane} = (\n"
            "        cutlass.Int32(cute.arch.thread_idx()[0]) * 2\n"
            f"        + cutlass.Int32({destination_pair} // 2)\n"
            "    )\n"
            f"    {source_pair} = {destination_pair} % 2\n"
            + "\n".join(
                f"    {packed_names[cache]}[{destination_pair}] = cutlass.Uint32(\n"
                "        cute.arch.shuffle_sync(\n"
                f"            {packed_source_names[cache]}[{source_pair}],\n"
                f"            {source_lane},\n"
                "        )\n"
                "    )"
                for cache in (
                    match.key_cache,
                    match.query_cache,
                    match.decay_cache,
                )
            )
        )
    else:
        pair_name = f"{prefix}source_pair"
        loop_lines = [f"for {pair_name} in cutlass.range_constexpr(4):"]
        for cache in (match.key_cache, match.query_cache, match.decay_cache):
            expression = pair_expressions[cache]
            assert isinstance(expression, (ast.Tuple, ast.List))
            low = _substitute_names(
                expression.elts[0],
                {match.pair_loop_var: ast.Name(id=pair_name, ctx=ast.Load())},
            )
            high = _substitute_names(
                expression.elts[1],
                {match.pair_loop_var: ast.Name(id=pair_name, ctx=ast.Load())},
            )
            loop_lines.extend(
                (
                    f"    {packed_names[cache]}[{pair_name}] = {_PACK}(",
                    f"        {ast.unparse(low)},",
                    f"        {ast.unparse(high)},",
                    "    )",
                )
            )
        source.append("\n".join(loop_lines))
    replacement: list[ast.stmt] = []
    for fragment in source:
        replacement.extend(_parse_statements(fragment))
    if not _replace_allocations(body, {match.decayed_cache: "BFloat16"}):
        return None
    match.owner[insertion_index:insertion_index] = replacement
    return replace(
        match,
        key_cache=packed_names[match.key_cache],
        query_cache=packed_names[match.query_cache],
        decay_cache=packed_names[match.decay_cache],
    )


def _remove_qdot(match: _Match) -> int | None:
    candidates: list[tuple[int, ast.For, tuple[str, str]]] = []
    for index, statement in enumerate(match.owner[: match.row_index]):
        if not isinstance(statement, ast.For) or _constexpr_trip_count(statement) != 4:
            continue
        assert isinstance(statement.target, ast.Name)
        loop_var = statement.target.id
        calls = _calls(statement, ("cute", "arch", "fma_packed_f32x2"))
        if len(calls) != 1 or len(calls[0].args) != 3:
            continue
        definitions: dict[str, ast.expr] = {}
        for child in statement.body:
            target = _name_target(child)
            if target is not None and isinstance(child, ast.Assign):
                definitions[target] = _inline_names(child.value, definitions)
        caches = {
            _pair_cache(
                argument,
                loop_var=loop_var,
                definitions=definitions,
            )
            for argument in calls[0].args[:2]
        }
        targets = _tuple_target_names(calls[0], statement)
        if caches == {match.key_cache, match.query_cache} and targets is not None:
            candidates.append((index, statement, targets))
    if len(candidates) != 1:
        return None
    index, loop, targets = candidates[0]
    if not _recurrence_loop_is_pure(
        loop,
        subscript_containers=frozenset(),
        append_containers=frozenset(),
    ):
        return None
    start = index
    while start > 0 and _name_target(match.owner[start - 1]) in targets:
        start -= 1
    if start != index - 2:
        return None
    end = index + 1
    reduced_names: set[str] = set(targets)
    while end < match.row_index and end < index + 4:
        statement = match.owner[end]
        if not (_loaded_names(statement) & reduced_names):
            break
        reduced_names.update(_assigned_names(statement))
        end += 1
    if end != index + 3 or not _calls(
        match.owner[end - 1], ("cute", "arch", "warp_reduction_sum")
    ):
        return None
    if any(
        not isinstance(statement, ast.Assign) or not _assignment_rhs_is_pure(statement)
        for statement in (*match.owner[start:index], *match.owner[index + 1 : end])
    ):
        return None
    removed_module = ast.Module(body=match.owner[start:end], type_ignores=[])
    root_module = ast.Module(body=match.root_body, type_ignores=[])
    if any(
        not _loads_only_within(root_module, name, removed_module)
        for name in _assigned_names(removed_module) - {match.qdot_name}
    ):
        return None
    del match.owner[start:end]
    return start


def _pack_state_loads(match: _Match) -> None:
    for assignment in match.pipe_loads:
        assert isinstance(assignment.value, ast.Call)
        vector_type = assignment.value.args[1]
        assert isinstance(vector_type, ast.Call)
        vector_type.args = [
            ast.List(elts=[ast.Constant(value=4)], ctx=ast.Load()),
            ast.Attribute(
                value=ast.Name(id="cutlass", ctx=ast.Load()),
                attr="Uint32",
                ctx=ast.Load(),
            ),
        ]
        # VectorType.get wants the MLIR type, not the scalar wrapper.
        vector_type.args[1] = ast.Attribute(
            value=vector_type.args[1], attr="mlir_type", ctx=ast.Load()
        )


def _rewrite_row(match: _Match, prefix: str) -> bool:
    # Match indices were captured before owner-level edits, which do not alter
    # the row object or its own statement positions.
    row = match.row
    prediction_stmt_index: int | None = None
    for index in range(match.first_index + 1, match.update_index):
        if _name_target(row.body[index]) == match.prediction_name:
            prediction_stmt_index = index
            break
    if prediction_stmt_index is None:
        return False
    delta_slice = _backward_slice(
        row.body[prediction_stmt_index + 1 : match.update_index],
        match.delta_name,
        match.prediction_name,
    )
    if delta_slice is None:
        return False

    predicate = getattr(row, "_helion_async_state_valid_predicate", None)
    if predicate is not None and not isinstance(predicate, ast.expr):
        return False
    state_mask = ""
    if predicate is not None:
        state_mask = (
            f"\n    {prefix}state_pair = (\n"
            f"        {prefix}state_pair if {ast.unparse(predicate)} else cutlass.Uint32(0)\n"
            "    )"
        )
    first_source = f"""
{prefix}prediction = cutlass.Uint32(0)
for {prefix}pair in cutlass.range_constexpr(4):
    {prefix}state_pair = cutlass.Uint32({match.state_vector}[{prefix}pair]){state_mask}
    {prefix}decayed = {_MUL}(
        {prefix}state_pair, {match.decay_cache}[{prefix}pair]
    )
    {match.decayed_cache}[{prefix}pair] = {prefix}decayed
    {prefix}prediction = {_FMA}(
        {prefix}decayed,
        {match.key_cache}[{prefix}pair],
        {prefix}prediction,
    )
for {prefix}reduce_step in cutlass.range_constexpr(4):
    {prefix}other = cutlass.Uint32(
        cute.arch.shuffle_sync_bfly(
            {prefix}prediction, cutlass.Int32(8 >> {prefix}reduce_step)
        )
    )
    {prefix}prediction = {_ADD}({prefix}prediction, {prefix}other)
{match.prediction_name} = cutlass.Float32(
    cutlass.Uint16({prefix}prediction).bitcast(cutlass.BFloat16)
) + cutlass.Float32(
    cutlass.Uint16(
        {prefix}prediction >> cutlass.Uint32(16)
    ).bitcast(cutlass.BFloat16)
)
"""
    replacement = _parse_statements(first_source)
    replacement.extend(delta_slice)

    store_helper = (
        "_cute_store_u32x4_l2_evict_last"
        if match.state_store_helper == "_cute_store_u16x8_l2_evict_last"
        else "_cute_store_u32_vec"
    )
    second_source = f"""
{prefix}delta = {_PACK}({match.delta_name}, {match.delta_name})
{prefix}output = cutlass.Uint32(0)
{prefix}store_values = []
for {prefix}pair in cutlass.range_constexpr(4):
    {prefix}updated = {_FMA}(
        {match.key_cache}[{prefix}pair],
        {prefix}delta,
        {match.decayed_cache}[{prefix}pair],
    )
    {prefix}output = {_FMA}(
        {prefix}updated,
        {match.query_cache}[{prefix}pair],
        {prefix}output,
    )
    {prefix}store_values.append({prefix}updated)
"""
    replacement.extend(_parse_statements(second_source))
    replacement.append(
        _replace_state_store(
            match.state_store_statement,
            store_helper,
            match.state_pointer,
            f"{prefix}store_values",
        )
    )
    reduction_source = f"""
for {prefix}reduce_step in cutlass.range_constexpr(4):
    {prefix}other = cutlass.Uint32(
        cute.arch.shuffle_sync_bfly(
            {prefix}output, cutlass.Int32(8 >> {prefix}reduce_step)
        )
    )
    {prefix}output = {_ADD}({prefix}output, {prefix}other)
{prefix}output = {_ADD}(
    {prefix}output, {prefix}output >> cutlass.Uint32(16)
)
"""
    replacement.extend(_parse_statements(reduction_source))
    output_value = ast.parse(
        f"cutlass.Uint16({prefix}output).bitcast(cutlass.BFloat16)", mode="eval"
    ).body
    replacement.append(_replace_store_value(match.output_statement, output_value))

    start = match.first_index
    accumulator_names = set()
    for call in _calls(match.first_loop, ("cute", "arch", "fma_packed_f32x2")):
        targets = _tuple_target_names(call, match.first_loop)
        if targets is not None:
            accumulator_names.update(targets)
    while start > 0 and _name_target(row.body[start - 1]) in accumulator_names:
        start -= 1
    removed = row.body[start : match.store_index + 1]
    explicitly_replaced = {
        id(match.first_loop),
        id(match.update_loop),
        id(match.output_statement),
        id(match.state_store_statement),
    }
    if any(
        id(statement) not in explicitly_replaced
        and not (
            isinstance(statement, ast.Assign) and _assignment_rhs_is_pure(statement)
        )
        for statement in removed
    ):
        return False
    removed_module = ast.Module(body=removed, type_ignores=[])
    root_module = ast.Module(body=match.root_body, type_ignores=[])
    if any(
        not _loads_only_within(root_module, name, removed_module)
        for name in _assigned_names(removed_module)
    ):
        return False
    row.body[start : match.store_index + 1] = replacement
    return True


def _transform(
    body: list[ast.stmt],
    uniform_names: frozenset[str],
    thread_block_dims: tuple[int, int, int] | None,
) -> bool:
    match = _match(body)
    if match is None:
        return False
    occupied_names = {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    suffix = 0
    while any(name.startswith(f"_bf16x2_{suffix}_") for name in occupied_names):
        suffix += 1
    prefix = f"_bf16x2_{suffix}_"
    # Do the proof-sensitive steps before row replacement.  The entire caller
    # works on a deep copy, so a late failure still returns the original body.
    _pack_state_loads(match)
    qdot_index = _remove_qdot(match)
    if qdot_index is None:
        return False
    if _has_combined_cache_expansion(match):
        if not _pack_expansion(body, match, prefix, uniform_names):
            return False
        packed_match = match
    else:
        packed_match = _pack_separate_caches(
            body,
            match,
            prefix,
            uniform_names,
            qdot_index,
            thread_block_dims,
        )
        if packed_match is None:
            return False
    return _rewrite_row(packed_match, prefix)


def pack_bf16_recurrences(
    body: list[ast.stmt],
    *,
    enabled: bool,
    fast_math: bool,
    target_device_capability: tuple[int, int] | None,
    uniform_names: frozenset[str] = frozenset(),
    thread_block_dims: tuple[int, int, int] | None = None,
) -> list[ast.stmt]:
    """Pack one proven async BF16 rank-1 recurrence, or return ``body``.

    Native ``bf16x2`` arithmetic requires SM80.  Exactly one complete graph
    must match; partial matches are discarded transactionally.
    """
    if (
        not enabled
        or not fast_math
        or target_device_capability is None
        or target_device_capability < (8, 0)
    ):
        return body
    candidate = _clone_ast(body)
    assert isinstance(candidate, list)
    if not _transform(candidate, uniform_names, thread_block_dims):
        return body
    module = ast.Module(body=candidate, type_ignores=[])
    ast.fix_missing_locations(module)
    return candidate
