"""Late, proof-driven cleanup of generated CuTe bounds checks.

The pass intentionally understands only nonnegative integer affine arithmetic
over exact launch dimensions, constexprs, and cache-specialized tensor sizes.
Unknown values, reassigned proof inputs, launches without an exact grid, and
arithmetic that could overflow are left untouched; names ``ast_rename`` merges
later count as one variable.  The same proofs let ``//`` and ``%`` of a
nonnegative traced ``Int32`` run unsigned.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import expr_from_string
from ._ast_pass_utils import call_path as _call_path
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import comparison_is_always_true
from ._ast_pass_utils import is_staged_loop
from ._ast_pass_utils import merged_names
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_stored_or_deleted as _assigned_names
from ._ast_pass_utils import with_merged_names
from .scalar_integer import _INTEGER_CASTS

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping

_INT32_MAX = (1 << 31) - 1
_ABI_VERSION = 1
_MARKER_PREFIX = "_cute_proven_bounds_"
_PYTHON_MOD = "_cute_python_mod"
# The kinds of integer the unsigned rewrite tells apart (``_integer_kind``).
_INT32 = "int32"
_PYTHON = "python"


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


def _boolean_value(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is bool
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    return isinstance(node, ast.BoolOp) and all(
        _boolean_value(value) for value in node.values
    )


def _remove_true_and_operands(values: list[ast.expr]) -> list[ast.expr]:
    remaining = [
        value
        for value in values
        if not (isinstance(value, ast.Constant) and value.value is True)
    ]
    # ``and`` returns an operand, not its truth value. A final True changes
    # the result of ``7 and True`` from 7 to True and must remain unless the
    # preceding result is already Boolean.
    if (
        remaining
        and isinstance(values[-1], ast.Constant)
        and values[-1].value is True
        and not _boolean_value(remaining[-1])
    ):
        remaining.append(values[-1])
    return remaining


def _python_mod_operands(node: ast.AST) -> list[ast.expr] | None:
    """The operands of a ``_cute_python_mod(a, b)`` call, else None."""
    if (
        isinstance(node, ast.Call)
        and _call_path(node.func) == (_PYTHON_MOD,)
        and len(node.args) == 2
        and not node.keywords
    ):
        return node.args
    return None


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
        operands = _python_mod_operands(node)
        if operands is not None:
            # Over the nonnegative domain Python's remainder is C's.
            dividend, divisor = (
                _integer_interval(
                    argument,
                    bounds=bounds,
                    thread_block_dims=thread_block_dims,
                    block_grid_dims=block_grid_dims,
                )
                for argument in operands
            )
            if dividend is None or divisor is None or not divisor[0] == divisor[1] > 0:
                return None
            return _checked_interval(0, min(dividend[1], divisor[0] - 1))
        if (
            call_path_str(node.func) not in _INTEGER_CASTS
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


def _integer_kind(node: ast.expr, kinds: dict[str, str]) -> str | None:
    """``_INT32`` for a traced ``cutlass.Int32`` value, ``_PYTHON`` for a
    trace-time Python int (constants, constexprs, ``range_constexpr`` lanes),
    None for anything else (wider or unsigned DSL integers, floats, unknown
    values).  ``kinds`` maps names to the kinds of their values.

    The DSL converts a Python int within Int32 beside an Int32 operand to
    Int32, so arithmetic mixing the two stays Int32.  A Python int outside
    Int32 would widen it, so a kind only holds where the interval proof also
    bounds the expression.
    """
    if isinstance(node, ast.Constant):
        return _PYTHON if type(node.value) is int else None
    if isinstance(node, ast.Name):
        return kinds.get(node.id)
    if (
        _axis_index(node, "thread_idx") is not None
        or _axis_index(node, "block_idx") is not None
    ):
        return _INT32
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _integer_kind(node.operand, kinds)
    if isinstance(node, ast.Call):
        if (
            call_path_str(node.func) == "cutlass.Int32"
            and len(node.args) == 1
            and not node.keywords
        ):
            return _INT32
        operands = _python_mod_operands(node)
        if operands is None:
            return None
    elif isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ):
        operands = [node.left, node.right]
    else:
        return None
    found = {_integer_kind(operand, kinds) for operand in operands}
    if found == {_PYTHON}:
        return _PYTHON
    if found <= {_INT32, _PYTHON}:
        return _INT32
    return None


def _loop_target_kind(
    loop: ast.For,
    kinds: dict[str, str],
    interval: Callable[[ast.expr], tuple[int, int] | None],
) -> str | None:
    """The kind of a loop's counter: a staged loop counts in the integer its
    bounds promote to (Int32 for Int32 bounds and Python ints the interval
    proof keeps within Int32; a larger Python int widens it), an unrolled
    ``range_constexpr`` loop in a Python int."""
    iterator = loop.iter
    if not isinstance(iterator, ast.Call):
        return None
    if is_staged_loop(loop):
        if all(
            (kind := _integer_kind(argument, kinds)) == _INT32
            or (kind == _PYTHON and interval(argument) is not None)
            for argument in iterator.args
        ):
            return _INT32
        return None
    if _call_path(iterator.func) == ("cutlass", "range_constexpr"):
        return _PYTHON
    return None


def _bind_kind(kinds: dict[str, str], name: str, kind: str | None) -> None:
    if kind is None:
        kinds.pop(name, None)
    else:
        kinds[name] = kind


def _unsigned(operation: str, dividend: ast.expr, divisor: ast.expr) -> ast.expr:
    """``dividend // divisor`` or ``dividend % divisor`` in unsigned 32-bit
    arithmetic, back as ``Int32``: signed floor division and remainder carry
    sign corrections that a proven nonnegative dividend never needs.  The
    dividend is a traced Int32 and the divisor an Int32 or a Python int, so
    the result has the type the signed operation had."""
    expression = expr_from_string(
        f"cutlass.Int32(cutlass.Uint32({{a}}) {operation} cutlass.Uint32({{b}}))",
        a=dividend,
        b=divisor,
    )
    assert isinstance(expression, ast.expr)
    return expression


@dataclass
class _Simplifier:
    thread_block_dims: tuple[int, int, int]
    block_grid_dims: tuple[int, int, int]
    # Each name ``ast_rename`` merges with others (a loop-carried value and
    # its aliases), with all of them: once renamed they are one variable, so
    # a write to any is a write to all.
    merged: Mapping[str, frozenset[str]]
    changed: int = 0

    def with_merged(self, names: Iterable[str]) -> set[str]:
        return with_merged_names(names, self.merged)

    def expression(
        self,
        node: ast.expr,
        bounds: dict[str, tuple[int, int]],
        kinds: dict[str, str],
    ) -> ast.expr:
        parent = self
        interval = functools.partial(
            _integer_interval,
            bounds=bounds,
            thread_block_dims=self.thread_block_dims,
            block_grid_dims=self.block_grid_dims,
        )

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
                if comparison_is_always_true(visited, interval):
                    parent.changed += 1
                    return ast.copy_location(ast.Constant(value=True), visited)
                return visited

            def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
                visited = self.generic_visit(node)
                assert isinstance(visited, ast.BoolOp)
                if not isinstance(visited.op, ast.And):
                    return visited
                values = _remove_true_and_operands(visited.values)
                if len(values) == len(visited.values):
                    return visited
                parent.changed += 1
                if not values:
                    return ast.copy_location(ast.Constant(value=True), visited)
                if len(values) == 1:
                    return ast.copy_location(values[0], visited)
                visited.values = values
                return visited

            def _unsigned_operands(self, dividend: ast.expr, divisor: ast.expr) -> bool:
                # Trace-time Python ints are never rewritten, and wider or
                # unsigned operands keep their type (see ``_unsigned``).
                divisor_range = interval(divisor)
                return (
                    interval(dividend) is not None
                    and divisor_range is not None
                    and divisor_range[0] == divisor_range[1] > 0
                    and _integer_kind(dividend, kinds) == _INT32
                    and _integer_kind(divisor, kinds) in (_INT32, _PYTHON)
                )

            def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
                visited = self.generic_visit(node)
                assert isinstance(visited, ast.BinOp)
                # isinstance: codegen's operator nodes are ExtendedAST
                # subclasses of the ast classes.
                if isinstance(visited.op, ast.FloorDiv):
                    operation = "//"
                elif isinstance(visited.op, ast.Mod):
                    operation = "%"
                else:
                    return visited
                if not self._unsigned_operands(visited.left, visited.right):
                    return visited
                parent.changed += 1
                return ast.copy_location(
                    _unsigned(operation, visited.left, visited.right), visited
                )

            def visit_Call(self, node: ast.Call) -> ast.AST:
                visited = self.generic_visit(node)
                assert isinstance(visited, ast.Call)
                operands = _python_mod_operands(visited)
                if operands is None or not self._unsigned_operands(*operands):
                    return visited
                parent.changed += 1
                return ast.copy_location(_unsigned("%", *operands), visited)

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
        kinds: dict[str, str],
    ) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for statement in statements:
            if isinstance(statement, ast.Assign):
                statement.value = self.expression(statement.value, bounds, kinds)
                if len(statement.targets) == 1 and isinstance(
                    statement.targets[0], ast.Name
                ):
                    targets = self.with_merged((statement.targets[0].id,))
                    kind = _integer_kind(statement.value, kinds)
                    interval = (
                        None
                        if targets & names_read(statement.value)
                        else _integer_interval(
                            statement.value,
                            bounds=bounds,
                            thread_block_dims=self.thread_block_dims,
                            block_grid_dims=self.block_grid_dims,
                        )
                    )
                    for target in targets:
                        _bind_kind(kinds, target, kind)
                        if interval is None:
                            bounds.pop(target, None)
                        else:
                            bounds[target] = interval
                result.append(statement)
                continue
            if isinstance(statement, ast.For):
                statement.iter = self.expression(statement.iter, bounds, kinds)
                loop_bounds = dict(bounds)
                loop_kinds = dict(kinds)
                # A value written in the loop is loop-carried unless a local
                # assignment re-establishes its interval before use.  Do not
                # let an incoming value justify a guard in a later iteration.
                loop_assigned = self.with_merged(
                    _assigned_names(ast.Module(body=statement.body, type_ignores=[]))
                )
                for name in loop_assigned:
                    loop_bounds.pop(name, None)
                    loop_kinds.pop(name, None)
                interval = _range_interval(statement.iter, bounds=bounds)
                if isinstance(statement.target, ast.Name):
                    kind = _loop_target_kind(
                        statement,
                        kinds,
                        functools.partial(
                            _integer_interval,
                            bounds=bounds,
                            thread_block_dims=self.thread_block_dims,
                            block_grid_dims=self.block_grid_dims,
                        ),
                    )
                    for target in self.with_merged((statement.target.id,)):
                        _bind_kind(loop_kinds, target, kind)
                        loop_bounds.pop(target, None)
                        if interval is not None:
                            loop_bounds[target] = interval
                statement.body = self.block(statement.body, loop_bounds, loop_kinds)
                post_loop_bounds = dict(bounds)
                for name in self.with_merged(_assigned_names(statement)):
                    post_loop_bounds.pop(name, None)
                    kinds.pop(name, None)
                statement.orelse = self.block(
                    statement.orelse, dict(post_loop_bounds), dict(kinds)
                )
                bounds.clear()
                bounds.update(post_loop_bounds)
                result.append(statement)
                continue
            if isinstance(statement, ast.If):
                statement.test = self.expression(statement.test, bounds, kinds)
                if isinstance(statement.test, ast.Constant) and (
                    statement.test.value is True
                ):
                    self.changed += 1
                    result.extend(self.block(statement.body, bounds, kinds))
                    continue
                statement.body = self.block(statement.body, dict(bounds), dict(kinds))
                statement.orelse = self.block(
                    statement.orelse, dict(bounds), dict(kinds)
                )
                for name in self.with_merged(_assigned_names(statement)):
                    bounds.pop(name, None)
                    kinds.pop(name, None)
                result.append(statement)
                continue
            for field, value in ast.iter_fields(statement):
                if isinstance(value, ast.expr):
                    setattr(statement, field, self.expression(value, bounds, kinds))
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, ast.expr):
                            value[index] = self.expression(item, bounds, kinds)
            result.append(statement)
        return result


def simplify_proven_bounds(
    body: list[ast.stmt],
    *,
    enabled: bool,
    exact_grid: bool,
    thread_block_dims: tuple[int, int, int] | None,
    block_grid_dims: tuple[int | None, int | None, int | None],
    constexpr_values: dict[str, int],
    proven_tensor_size_values: dict[tuple[str, int], tuple[str, int]],
    rename_groups: Mapping[str, str] | None = None,
) -> list[ast.stmt]:
    """Simplify only bounds proven by exact launch and input metadata.

    ``thread_block_dims`` is the proven static launch shape; ``None`` (an
    argument-sized or otherwise unproven axis) proves nothing.  ``exact_grid``
    says ``block_grid_dims`` is the launch's block grid.  ``rename_groups``
    maps the names ``ast_rename`` will merge to their shared name; the body
    still spells them apart, so a loop-carried value updated under an alias
    must count as updated.
    """
    if not enabled or not exact_grid or thread_block_dims is None:
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
    simplifier = _Simplifier(thread_block_dims, grid, merged_names(rename_groups or {}))
    if simplifier.with_merged(_assigned_names(module)) & protected_names:
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
    rewritten = simplifier.block(
        body, initial_bounds, dict.fromkeys(constexpr_values, _PYTHON)
    )
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
