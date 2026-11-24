"""Evaluate runtime strides in their original host or grid scope."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Node

from ... import exc
from ..ast_extension import statement_from_string
from ..device_function import SymbolArgument
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction
from ..variable_origin import GridOrigin

if TYPE_CHECKING:
    from .fragment_context import FragmentContext


def _static_int(ctx: FragmentContext, value: object) -> int | None:
    if isinstance(value, Node):
        value = value.meta["val"]
    if isinstance(value, (int, torch.SymInt, sympy.Expr)):
        return ctx.static_extent(value)
    return None


def _endpoints(
    ctx: FragmentContext, begin: object, end: object
) -> tuple[int, int] | None:
    start, stop = _static_int(ctx, begin), _static_int(ctx, end)
    return (start, stop) if start is not None and stop is not None else None


def _coordinates(start: int, stop: int, step: int | None) -> range:
    # Without a known stride, any coordinate between the endpoints may occur.
    if step is None:
        step = 1 if start <= stop else -1
    return range(start, stop, step) if step else range(0)


def device_range_expr(ctx: FragmentContext, expr: sympy.Basic, operand: str) -> str:
    """Reject statically bounded Int64 overflow; retain wide native math for unknowns."""
    source = ctx.sym(expr, wide=True)
    i64 = torch.iinfo(torch.int64)
    operation = ast.parse(source, mode="eval").body
    names = {node.id for node in ast.walk(operation) if isinstance(node, ast.Name)}
    bounds: dict[str, range] = {}
    for name, (begin, end, step) in ctx.stride_operands.items():
        if endpoints := _endpoints(ctx, begin, end):
            domain = _coordinates(*endpoints, _static_int(ctx, step))
            if not domain:
                # No operand in the body of an empty Cartesian product is read.
                return "1"
            bounds[name] = domain
    grid = ctx.cg.current_grid_state
    if grid is not None:
        # Scalar origins follow the declared grid stride. A flattened tile
        # can walk elements, so use its stride only to establish emptiness.
        origins = HostFunction.current().expr_to_origin
        scalar_axes = {
            origin.block_id
            for symbol in expr.free_symbols
            if (entry := origins.get(symbol))
            and type(origin := entry.origin) is GridOrigin
        }
        for bid, info in grid.block_id_to_info.items():
            name = grid.strategy.grid_origin_var(bid)
            endpoints = _endpoints(ctx, info.begin_expr, info.grid_end_expr)
            if endpoints is not None:
                step = _static_int(ctx, info.grid_step_expr)
                if not _coordinates(*endpoints, step):
                    return "1"
                if name not in names and info.end_var_name not in names:
                    continue
                bounds[name] = _coordinates(
                    *endpoints, step if bid in scalar_axes else None
                )
                if info.end_var_name is not None:
                    bounds[info.end_var_name] = range(endpoints[1], endpoints[1] + 1)
    # A configured block size can appear as a named constant in the source.
    for symbol in expr.free_symbols:
        configured = ctx.configured_expr(symbol)
        if isinstance(configured, sympy.Integer):
            bounds[ctx.df.literal_expr(symbol)] = range(
                int(configured), int(configured) + 1
            )

    def spacing(n: ast.expr) -> int:
        """A divisor of all differences between values of this expression."""
        if isinstance(n, ast.Name) and (domain := bounds.get(n.id)):
            return abs(domain.step) if domain[0] != domain[-1] else 0
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return 0
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "cutlass.Int64":
            return spacing(n.args[0])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return spacing(n.operand)
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, (ast.Add, ast.Sub)):
                return math.gcd(spacing(n.left), spacing(n.right))
            if isinstance(n.op, ast.Mult):
                for factor, term in ((n.left, n.right), (n.right, n.left)):
                    if isinstance(factor, ast.Constant) and type(factor.value) is int:
                        return abs(factor.value) * spacing(term)
        return 1

    guarded = False

    def check_divisor(node: ast.expr, divisor: tuple[int, int] | None) -> ast.expr:
        nonlocal guarded
        if divisor == (0, 0):
            raise exc.InvalidConfig(f"fragment grid {operand} has a zero divisor")
        if divisor is not None and not divisor[0] <= 0 <= divisor[1]:
            return node
        # Guard in the enclosing device scope, before the arithmetic uses it.
        # The fallback makes the division defined even when speculated by LLVM.
        name = ctx.df.new_var("fragment_divisor")
        ctx.emit(f"{name} = cutlass.Int64({ast.unparse(node)})")
        ctx.emit(
            f"if {name} == 0:\n"
            "    _cute_inline_asm_elementwise((), asm='mov.u32 $0, 0; trap;', "
            "constraints='=r', dtype=cutlass.Int32, is_pure=False)\n"
            f"    {name} = cutlass.Int64(1)"
        )
        guarded = True
        return ast.Name(id=name, ctx=ast.Load())

    def remainder_bounds(
        lhs: ast.expr,
        left: tuple[int, int] | None,
        rhs: ast.expr,
        right: tuple[int, int],
    ) -> tuple[int, int]:
        if left is not None:
            dividends = range(left[0], left[1] + 1, max(1, spacing(lhs)))
            divisors = range(right[0], right[1] + 1, max(1, spacing(rhs)))
            # Small strided domains can settle e.g. 2 % {-1, 2} exactly.
            # Slice before len: even a single Int64 range may exceed sys.maxsize.
            limit = 32
            if len(dividends[: limit + 1]) * len(divisors[: limit + 1]) <= limit:
                remainders = [x % y for x in dividends for y in divisors if y]
                return min(remainders), max(remainders)
        # Zero was guarded. Python's remainder follows each surviving divisor's sign.
        return min(0, right[0] + 1), max(0, right[1] - 1)

    def interval(n: ast.expr) -> tuple[int, int] | None:
        a: int
        b: int
        if isinstance(n, ast.Constant) and type(n.value) is int:
            a = b = n.value
        elif isinstance(n, ast.Name):
            value = bounds.get(n.id)
            if value is None:
                return None
            a, b = sorted((value[0], value[-1]))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            if isinstance(n.operand, ast.Constant) and type(n.operand.value) is int:
                a = b = -n.operand.value
            else:
                value = interval(n.operand)
                if value is None:
                    return None
                a, b = -value[1], -value[0]
        elif isinstance(n, ast.Call):
            args = [interval(arg) for arg in n.args]
            name = ast.unparse(n.func)
            if name == "cutlass.Int64" and len(args) == 1:
                return args[0]
            if name in ("min", "max") and args and all(v is not None for v in args):
                select = min if name == "min" else max
                a = select(v[0] for v in args if v is not None)
                b = select(v[1] for v in args if v is not None)
            elif name == "_cute_python_mod" and len(args) == 2:
                lhs, rhs = n.args
                left, right = args
                n.args[1] = check_divisor(rhs, right)
                if right is None:
                    return None
                a, b = remainder_bounds(lhs, left, rhs, right)
            else:
                return None
        elif isinstance(n, ast.BinOp):
            # Visit both subexpressions even if only one has a known domain.
            left, right = interval(n.left), interval(n.right)
            rhs = n.right
            if isinstance(n.op, (ast.Mod, ast.FloorDiv)):
                n.right = check_divisor(n.right, right)
            if isinstance(n.op, ast.Mod) and right is not None and right[0] > 0:
                a, b = remainder_bounds(n.left, left, rhs, right)
            elif left is None or right is None:
                return None
            elif isinstance(n.op, ast.Add):
                a, b = left[0] + right[0], left[1] + right[1]
            elif isinstance(n.op, ast.Sub):
                a, b = left[0] - right[1], left[1] - right[0]
            elif isinstance(n.op, ast.Mult):
                products = [x * y for x in left for y in right]
                a, b = min(products), max(products)
            elif isinstance(n.op, ast.FloorDiv):
                # Floor division is monotone on each side of zero. Check
                # the closest reachable candidates there, respecting the
                # divisor's congruence rather than inventing a divisor of -1.
                step = max(1, spacing(rhs))
                lo, hi = right
                negative = -1 - (-1 - lo) % step
                positive = 1 + (lo - 1) % step
                divisors = {
                    y for y in (lo, negative, positive, hi) if y and lo <= y <= hi
                }
                results = [x // y for x in left for y in divisors]
                a, b = min(results), max(results)
            else:
                return None
        else:
            return None
        if not i64.min <= a <= b <= i64.max:
            raise exc.InvalidConfig(
                f"fragment grid {operand} is not proved to fit Int64"
            )
        return a, b

    interval(operation)
    return ast.unparse(operation) if guarded else source


def dynamic_fragment_step(
    ctx: FragmentContext, node: Node, step: object
) -> tuple[str, bool]:
    """Return a signed device operand and whether zero needs a check at loop entry."""
    value = step.meta["val"] if isinstance(step, Node) else step
    expr = value._sympy_() if isinstance(value, torch.SymInt) else value
    host = HostFunction.current()
    if not isinstance(expr, sympy.Expr):
        raise exc.InvalidConfig("a dynamic fragment loop step must be an integer")
    if not host.is_host_scalar_expr(expr):
        # Render the lexical coordinates before their arithmetic, including
        # nested grid indices, tile origins and non-affine expressions.
        return f"cutlass.Int64({device_range_expr(ctx, expr, 'stride')})", True

    # Never recompute a host expression using fixed-width device arithmetic.
    # A compound step can fit even though one intermediate exceeds signed Int64.
    grid = ctx.cg.current_grid_state
    reached = (
        grid is not None
        # The root bounds alone do not encode the sign of its step. Only a
        # canonical (zero origin, unit step) axis proves that it will execute.
        and not host.device_ir.noncanonical_task_origin_block_ids.intersection(
            grid.block_id_to_info
        )
        and any(
            isinstance(g, RootGraphInfo) and g.graph is node.graph for g in ctx.graphs
        )
        and all(
            (span := _endpoints(ctx, info.begin_expr, info.end_expr)) is not None
            and span[0] < span[1]
            for info in grid.block_id_to_info.values()
        )
    )
    operand = host.sympy_expr(expr, ctx.df.config)
    # This name lives in both the device signature and the Python wrapper.
    code = host.fn.__code__
    host_names = {
        *code.co_varnames,
        *code.co_names,
        *code.co_freevars,
        *code.co_cellvars,
    }
    name = ctx.df.new_var("fragment_host_step")
    while name in host_names:
        name = ctx.df.new_var("fragment_host_step")
    i64 = torch.iinfo(torch.int64)
    if reached or isinstance(expr, sympy.Symbol):
        ctx.cg.host_statements.append(statement_from_string(f"{name} = {operand}"))
    else:
        # A conditional range need not evaluate e.g. n // divisor when dead.
        ctx.cg.host_statements.append(
            statement_from_string(
                f"try:\n    {name} = {operand}\nexcept ArithmeticError:\n    {name} = 0"
            )
        )
    if reached:
        ctx.cg.host_statements.append(
            statement_from_string(
                f"if {name} == 0:\n    raise ValueError('range() arg 3 must not be zero')"
                f"\nelif not {i64.min} <= {name} <= {i64.max}:"
                "\n    raise OverflowError('fragment loop step exceeds signed Int64 range')"
            )
        )
    else:
        # A false device branch or an empty enclosing range must not raise.
        # Encode an invalid host result as zero; only a reached loop rejects it.
        ctx.cg.host_statements.append(
            statement_from_string(
                f"if not {i64.min} <= {name} <= {i64.max}:\n    {name} = 0"
            )
        )
    arg = SymbolArgument(name, name)
    ctx.df.arguments.append(arg)
    return f"cutlass.Int64({arg.name})", not reached
