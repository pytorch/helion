"""Late, proof-driven cleanup of generated CuTe bounds checks.

The pass intentionally understands only nonnegative integer affine arithmetic
over exact launch dimensions, constexprs, and cache-specialized tensor sizes.
Unknown values, reassigned proof inputs, non-XYZ launches, and arithmetic that
could overflow are left untouched.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import cast

_INT32_MAX = (1 << 31) - 1
_ABI_VERSION = 1
_MARKER_PREFIX = "_cute_proven_bounds_"


def _call_path(node: ast.AST) -> tuple[str, ...] | None:
    path: list[str] = []
    while isinstance(node, ast.Attribute):
        path.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    path.append(node.id)
    return tuple(reversed(path))


def _axis_index(node: ast.AST, function: str) -> int | None:
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and _call_path(node.value.func) == ("cute", "arch", function)
        and not node.value.args
        and not node.value.keywords
        and isinstance(node.slice, ast.Constant)
        and type(node.slice.value) is int
        and 0 <= node.slice.value < 3
    ):
        return None
    return node.slice.value


def _assigned_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del))
    }


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _literal_int(node: ast.expr, bounds: dict[str, tuple[int, int]]) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        value = bounds.get(node.id)
        if value is not None and value[0] == value[1]:
            return value[0]
    return None


def _checked_interval(low: int, high: int) -> tuple[int, int] | None:
    # Generated scalar indices are routinely wrapped in cutlass.Int32.  Keep
    # the whole proof domain in signed int32 even when an outer cast spells
    # Int64/Uint64: proving a larger intermediate would otherwise overlook an
    # inner Int32 wrap.  Large-index kernels simply retain their guards.
    if 0 <= low <= high <= _INT32_MAX:
        return low, high
    return None


def _integer_interval(
    node: ast.expr,
    *,
    bounds: dict[str, tuple[int, int]],
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int, int, int],
) -> tuple[int, int] | None:
    """Return a nonnegative interval, rejecting unsupported or wrapping math."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return _checked_interval(node.value, node.value)
    if isinstance(node, ast.Name):
        return bounds.get(node.id)
    thread_axis = _axis_index(node, "thread_idx")
    if thread_axis is not None:
        return 0, thread_block_dims[thread_axis] - 1
    block_axis = _axis_index(node, "block_idx")
    if block_axis is not None:
        return 0, block_grid_dims[block_axis] - 1
    if isinstance(node, ast.Call):
        path = _call_path(node.func)
        if (
            path
            not in {
                ("cutlass", "Int32"),
                ("cutlass", "Int64"),
                ("cutlass", "Uint32"),
                ("cutlass", "Uint64"),
            }
            or len(node.args) != 1
            or node.keywords
        ):
            return None
        value = _integer_interval(
            node.args[0],
            bounds=bounds,
            thread_block_dims=thread_block_dims,
            block_grid_dims=block_grid_dims,
        )
        if value is None:
            return None
        return _checked_interval(value[0], value[1])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _integer_interval(
            node.operand,
            bounds=bounds,
            thread_block_dims=thread_block_dims,
            block_grid_dims=block_grid_dims,
        )
    if not isinstance(node, ast.BinOp):
        return None
    left = _integer_interval(
        node.left,
        bounds=bounds,
        thread_block_dims=thread_block_dims,
        block_grid_dims=block_grid_dims,
    )
    right = _integer_interval(
        node.right,
        bounds=bounds,
        thread_block_dims=thread_block_dims,
        block_grid_dims=block_grid_dims,
    )
    if left is None or right is None:
        return None
    if isinstance(node.op, ast.Add):
        low, high = left[0] + right[0], left[1] + right[1]
    elif isinstance(node.op, ast.Sub):
        low, high = left[0] - right[1], left[1] - right[0]
    elif isinstance(node.op, ast.Mult):
        products = (
            left[0] * right[0],
            left[0] * right[1],
            left[1] * right[0],
            left[1] * right[1],
        )
        low, high = min(products), max(products)
    elif isinstance(node.op, ast.FloorDiv) and right[0] == right[1] > 0:
        low, high = left[0] // right[0], left[1] // right[0]
    elif isinstance(node.op, ast.Mod) and right[0] == right[1] > 0:
        low, high = 0, min(left[1], right[0] - 1)
    else:
        return None
    return _checked_interval(low, high)


def _comparison_is_true(
    node: ast.Compare,
    *,
    bounds: dict[str, tuple[int, int]],
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int, int, int],
) -> bool:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    left = _integer_interval(
        node.left,
        bounds=bounds,
        thread_block_dims=thread_block_dims,
        block_grid_dims=block_grid_dims,
    )
    right = _integer_interval(
        node.comparators[0],
        bounds=bounds,
        thread_block_dims=thread_block_dims,
        block_grid_dims=block_grid_dims,
    )
    if left is None or right is None:
        return False
    operation = node.ops[0]
    if isinstance(operation, ast.Lt):
        return left[1] < right[0]
    if isinstance(operation, ast.LtE):
        return left[1] <= right[0]
    if isinstance(operation, ast.Gt):
        return left[0] > right[1]
    if isinstance(operation, ast.GtE):
        return left[0] >= right[1]
    if isinstance(operation, ast.Eq):
        return left[0] == left[1] == right[0] == right[1]
    if isinstance(operation, ast.NotEq):
        return left[1] < right[0] or right[1] < left[0]
    return False


def _range_interval(
    node: ast.expr,
    *,
    bounds: dict[str, tuple[int, int]],
) -> tuple[int, int] | None:
    if not isinstance(node, ast.Call) or _call_path(node.func) not in {
        ("range",),
        ("cutlass", "range_constexpr"),
    }:
        return None
    if node.keywords or not 1 <= len(node.args) <= 3:
        return None
    values = [_literal_int(argument, bounds) for argument in node.args]
    if any(value is None for value in values):
        return None
    exact = [value for value in values if value is not None]
    if len(exact) == 1:
        start, stop, step = 0, exact[0], 1
    elif len(exact) == 2:
        start, stop, step = exact[0], exact[1], 1
    else:
        start, stop, step = exact
    if start < 0 or stop <= start or step <= 0:
        return None
    last = start + ((stop - start - 1) // step) * step
    return _checked_interval(start, last)


def _supported_body(statements: list[ast.stmt]) -> bool:
    """Reject control flow whose scalar dataflow this small pass cannot model."""
    for statement in statements:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0], (ast.Name, ast.Subscript)
            ):
                return False
        elif isinstance(statement, ast.If):
            if not _supported_body(statement.body) or not _supported_body(
                statement.orelse
            ):
                return False
        elif isinstance(statement, ast.For):
            if not isinstance(statement.target, ast.Name):
                return False
            if not _supported_body(statement.body) or not _supported_body(
                statement.orelse
            ):
                return False
        elif not isinstance(statement, (ast.Expr, ast.Pass)):
            return False
    module = ast.Module(body=statements, type_ignores=[])
    return not any(
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
        for node in ast.walk(module)
    )


@dataclass
class _Simplifier:
    thread_block_dims: tuple[int, int, int]
    block_grid_dims: tuple[int, int, int]
    changed: int = 0

    def expression(
        self, node: ast.expr, bounds: dict[str, tuple[int, int]]
    ) -> ast.expr:
        parent = self

        class Transformer(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                visited = self.generic_visit(node)
                assert isinstance(visited, ast.Subscript)
                axis = _axis_index(visited, "block_idx")
                if axis is not None and parent.block_grid_dims[axis] == 1:
                    parent.changed += 1
                    return ast.copy_location(ast.Constant(value=0), visited)
                return visited

            def visit_Compare(self, node: ast.Compare) -> ast.AST:
                visited = self.generic_visit(node)
                assert isinstance(visited, ast.Compare)
                if _comparison_is_true(
                    visited,
                    bounds=bounds,
                    thread_block_dims=parent.thread_block_dims,
                    block_grid_dims=parent.block_grid_dims,
                ):
                    parent.changed += 1
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
                if len(values) == len(visited.values):
                    return visited
                parent.changed += 1
                if not values:
                    return ast.copy_location(ast.Constant(value=True), visited)
                if len(values) == 1:
                    return ast.copy_location(values[0], visited)
                visited.values = values
                return visited

            def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
                visited = self.generic_visit(node)
                assert isinstance(visited, ast.IfExp)
                if (
                    isinstance(visited.test, ast.Constant)
                    and visited.test.value is True
                ):
                    parent.changed += 1
                    return ast.copy_location(visited.body, visited)
                return visited

        result = Transformer().visit(node)
        assert isinstance(result, ast.expr)
        return result

    def block(
        self,
        statements: list[ast.stmt],
        bounds: dict[str, tuple[int, int]],
    ) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for statement in statements:
            if isinstance(statement, ast.Assign):
                statement.value = self.expression(statement.value, bounds)
                if len(statement.targets) == 1 and isinstance(
                    statement.targets[0], ast.Name
                ):
                    target = statement.targets[0].id
                    if target in _loaded_names(statement.value):
                        bounds.pop(target, None)
                    else:
                        interval = _integer_interval(
                            statement.value,
                            bounds=bounds,
                            thread_block_dims=self.thread_block_dims,
                            block_grid_dims=self.block_grid_dims,
                        )
                        if interval is None:
                            bounds.pop(target, None)
                        else:
                            bounds[target] = interval
                result.append(statement)
                continue
            if isinstance(statement, ast.For):
                statement.iter = self.expression(statement.iter, bounds)
                loop_bounds = dict(bounds)
                # A value written in the loop is loop-carried unless a local
                # assignment re-establishes its interval before use.  Do not
                # let an incoming value justify a guard in a later iteration.
                loop_assigned = _assigned_names(
                    ast.Module(body=statement.body, type_ignores=[])
                )
                for name in loop_assigned:
                    loop_bounds.pop(name, None)
                interval = _range_interval(statement.iter, bounds=bounds)
                if isinstance(statement.target, ast.Name):
                    loop_bounds.pop(statement.target.id, None)
                    if interval is not None:
                        loop_bounds[statement.target.id] = interval
                statement.body = self.block(statement.body, loop_bounds)
                post_loop_bounds = dict(bounds)
                for name in _assigned_names(statement):
                    post_loop_bounds.pop(name, None)
                statement.orelse = self.block(statement.orelse, dict(post_loop_bounds))
                bounds.clear()
                bounds.update(post_loop_bounds)
                result.append(statement)
                continue
            if isinstance(statement, ast.If):
                statement.test = self.expression(statement.test, bounds)
                if isinstance(statement.test, ast.Constant) and (
                    statement.test.value is True
                ):
                    self.changed += 1
                    result.extend(self.block(statement.body, bounds))
                    continue
                statement.body = self.block(statement.body, dict(bounds))
                statement.orelse = self.block(statement.orelse, dict(bounds))
                for name in _assigned_names(statement):
                    bounds.pop(name, None)
                result.append(statement)
                continue
            for field, value in ast.iter_fields(statement):
                if isinstance(value, ast.expr):
                    setattr(statement, field, self.expression(value, bounds))
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, ast.expr):
                            value[index] = self.expression(item, bounds)
            result.append(statement)
        return result


def simplify_proven_bounds(
    body: list[ast.stmt],
    *,
    enabled: bool,
    xyz_grid: bool,
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int | None, int | None, int | None],
    constexpr_values: dict[str, int],
    proven_tensor_size_values: dict[tuple[str, int], tuple[str, int]],
) -> list[ast.stmt]:
    """Simplify only bounds proven by exact launch and input metadata."""
    if not enabled or not xyz_grid:
        return body
    if len(thread_block_dims) != 3 or len(block_grid_dims) != 3:
        return body
    if any(type(value) is not int or value <= 0 for value in thread_block_dims):
        return body
    if any(type(value) is not int or value <= 0 for value in block_grid_dims):
        return body
    grid = cast("tuple[int, int, int]", block_grid_dims)
    if (
        any(value > _INT32_MAX for value in thread_block_dims)
        or any(value > _INT32_MAX for value in grid)
        or thread_block_dims[0] * thread_block_dims[1] * thread_block_dims[2] > 1024
    ):
        return body
    protected_names = set(constexpr_values)
    exact_sizes: dict[str, int] = {}
    for name, value in proven_tensor_size_values.values():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or type(value) is not int
            or not 0 <= value <= _INT32_MAX
            or (name in exact_sizes and exact_sizes[name] != value)
            or (name in constexpr_values and constexpr_values[name] != value)
        ):
            return body
        exact_sizes[name] = value
        protected_names.add(name)
    module = ast.Module(body=body, type_ignores=[])
    if not _supported_body(body):
        return body
    if _assigned_names(module) & protected_names:
        return body
    if any(
        not isinstance(name, str)
        or not name.isidentifier()
        or type(value) is not int
        or not 0 <= value <= _INT32_MAX
        for name, value in constexpr_values.items()
    ):
        return body
    occupied_names = {
        node.id for node in ast.walk(module) if isinstance(node, ast.Name)
    }
    marker_suffix = 0
    while f"{_MARKER_PREFIX}{marker_suffix}_abi_version" in occupied_names:
        marker_suffix += 1
    initial_bounds = {name: (value, value) for name, value in constexpr_values.items()}
    initial_bounds.update({name: (value, value) for name, value in exact_sizes.items()})
    simplifier = _Simplifier(thread_block_dims, grid)
    rewritten = simplifier.block(body, initial_bounds)
    if not simplifier.changed:
        return body
    rewritten.insert(
        0,
        ast.Assign(
            targets=[
                ast.Name(
                    id=f"{_MARKER_PREFIX}{marker_suffix}_abi_version",
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=_ABI_VERSION),
        ),
    )
    ast.fix_missing_locations(ast.Module(body=rewritten, type_ignores=[]))
    return rewritten
