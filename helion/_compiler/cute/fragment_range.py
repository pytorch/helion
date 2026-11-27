"""Evaluate runtime strides in their original host or grid scope."""

from __future__ import annotations

import ast
import builtins
import math
import operator
from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Node
from torch.utils._sympy.value_ranges import SymPyValueRangeAnalysis
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._sympy.value_ranges import bound_sympy

from ... import exc
from ..ast_extension import statement_from_string
from ..device_function import SymbolArgument
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction

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
    bounds: dict[str, range] = {}
    grid = ctx.cg.current_grid_state
    if grid is not None:
        for bid, info in grid.block_id_to_info.items():
            endpoints = _endpoints(ctx, info.begin_expr, info.grid_end_expr)
            if endpoints is None:
                continue
            # A scalar root advances by its declared grid step; tile.begin
            # advances by the physical tile width, never by individual lanes.
            width = _static_int(ctx, ctx.df.resolved_block_size(bid))
            step = _static_int(ctx, info.grid_step_expr) if width == 1 else width
            domain = _coordinates(*endpoints, step)
            if not domain:
                return "1"
            bounds[grid.strategy.grid_origin_var(bid)] = domain
            if info.end_var_name is not None:
                bounds[info.end_var_name] = range(endpoints[1], endpoints[1] + 1)
    # A configured block size can appear as a named constant in the source.
    for symbol in expr.free_symbols:
        configured = ctx.configured_expr(symbol)
        if isinstance(configured, sympy.Integer):
            bounds[ctx.df.literal_expr(symbol)] = range(
                int(configured), int(configured) + 1
            )

    def congruence(n: ast.expr) -> tuple[int, int]:
        """Every reached value is residue + k*modulus; zero modulus is exact."""
        if isinstance(n, ast.Name) and (domain := bounds.get(n.id)):
            return domain[0], abs(domain.step) if domain[0] != domain[-1] else 0
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value, 0
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "cutlass.Int64":
            return congruence(n.args[0])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            residue, modulus = congruence(n.operand)
            return -residue, modulus
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)):
            left, m = congruence(n.left)
            right, nmod = congruence(n.right)
            if isinstance(n.op, ast.Mult):
                # (left + m*x)*(right + n*y), for arbitrary integers x,y.
                return left * right, math.gcd(left * nmod, right * m, m * nmod)
            result = left + right if isinstance(n.op, ast.Add) else left - right
            return result, math.gcd(m, nmod)
        return 0, 1

    def reachable(n: ast.expr, interval: tuple[int, int]) -> range:
        # An interval endpoint is a bound, not necessarily a reachable
        # representative of the stride (e.g. after tightening a polynomial).
        lo, hi = interval
        residue, modulus = congruence(n)
        if not modulus:
            return range(residue, residue + 1) if lo <= residue <= hi else range(0)
        return range(lo + (residue - lo) % modulus, hi + 1, modulus)

    guarded = False
    analyzing_header = True

    def check_divisor(node: ast.expr, divisor: tuple[int, int] | None) -> ast.expr:
        nonlocal guarded
        if analyzing_header:
            # An enclosing header was checked where it is evaluated; replay
            # only its range proof, not a guard at the nested use.
            return node
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
            dividends = reachable(lhs, left)
            divisors = reachable(rhs, right)
            # Small strided domains can settle e.g. 2 % {-1, 2} exactly.
            # Slice before len: even a single Int64 range may exceed sys.maxsize.
            limit = 32
            if len(dividends[: limit + 1]) * len(divisors[: limit + 1]) <= limit:
                remainders = [x % y for x in dividends for y in divisors if y]
                if remainders:
                    return min(remainders), max(remainders)
        # Zero was guarded. Python's remainder follows each surviving divisor's sign.
        return min(0, right[0] + 1), max(0, right[1] - 1)

    aliases: dict[str, sympy.Expr] = {}

    def polynomial(n: ast.expr) -> sympy.Expr | None:
        # Preserve shared coordinates through integer arithmetic. For example
        # the printer emits -x*x as (-1*x)*x; independent operand intervals
        # would incorrectly allow that product to be positive.
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return sympy.Integer(n.value)
        if isinstance(n, ast.Name) and n.id in bounds:
            return aliases.get(n.id, sympy.Symbol(n.id, integer=True))
        if (
            isinstance(n, ast.Call)
            and ast.unparse(n.func) == "cutlass.Int64"
            and len(n.args) == 1
        ):
            return polynomial(n.args[0])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            value = polynomial(n.operand)
            return -value if value is not None else None
        if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult):
            left, right = polynomial(n.left), polynomial(n.right)
            if left is not None and right is not None:
                if isinstance(n.op, ast.Mult):
                    return sympy.Mul(left, right)
                return sympy.Add(left, -right if isinstance(n.op, ast.Sub) else right)
        return None

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
            elif isinstance(n.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
                bitop, estimate = {
                    ast.BitAnd: (operator.and_, SymPyValueRangeAnalysis.bitwise_and),
                    ast.BitOr: (operator.or_, SymPyValueRangeAnalysis.bitwise_or),
                    ast.BitXor: (operator.xor, SymPyValueRangeAnalysis.bitwise_xor),
                }[type(n.op)]
                if left[0] == left[1] and right[0] == right[1]:
                    a = b = bitop(left[0], right[0])
                else:
                    result = estimate(ValueRanges(*left), ValueRanges(*right))
                    # Bitwise operations on signed words always fit Int64.
                    a = (
                        int(result.lower)
                        if isinstance(result.lower, sympy.Integer)
                        else i64.min
                    )
                    b = (
                        int(result.upper)
                        if isinstance(result.upper, sympy.Integer)
                        else i64.max
                    )
                    # A bitwise result cannot use more non-sign bits than
                    # either operand (also when the estimator returns oo).
                    bits = max(
                        (x if x >= 0 else ~x).bit_length() for x in (*left, *right)
                    )
                    word = 1 << bits
                    # Infinite sign bits obey the very same bit operation.
                    # If every pair agrees, keep that known result sign.
                    signs = {bitop(x < 0, y < 0) for x in left for y in right}
                    a = max(a, -word if True in signs else 0)
                    b = min(b, word - 1 if False in signs else -1)
                    if isinstance(n.op, ast.BitAnd):
                        for low, high in (left, right):
                            if low >= 0:
                                a, b = max(a, 0), min(b, high)
            elif isinstance(n.op, ast.FloorDiv):
                # Floor division is monotone on each side of zero. Check
                # the closest reachable candidates there, respecting the
                # divisor's congruence rather than inventing a divisor of -1.
                candidates = reachable(rhs, right)
                if not candidates:
                    return None
                lo, hi, step = candidates[0], candidates[-1], candidates.step
                negative = -1 - (-1 - lo) % step
                positive = 1 + (lo - 1) % step
                divisors = {
                    y for y in (lo, negative, positive, hi) if y and lo <= y <= hi
                }
                if not divisors:
                    return None  # The enclosing header already guarded zero.
                results = [x // y for x in left for y in divisors]
                a, b = min(results), max(results)
            else:
                return None
        else:
            return None
        if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult):
            form = polynomial(n)
            if form is not None:
                domain = {}
                for symbol in form.free_symbols:
                    assert isinstance(symbol, sympy.Symbol)
                    reached = bounds[symbol.name]
                    domain[symbol] = ValueRanges(*sorted((reached[0], reached[-1])))
                exact = bound_sympy(form, domain)
                if isinstance(exact.lower, sympy.Integer) and isinstance(
                    exact.upper, sympy.Integer
                ):
                    a, b = max(a, int(exact.lower)), min(b, int(exact.upper))
        if not i64.min <= a <= b <= i64.max:
            raise exc.InvalidConfig(
                f"fragment grid {operand} is not proved to fit Int64"
            )
        return a, b

    # Headers are rendered in their enclosing lexical scope before a nested
    # block ID can shadow it. Propagate their reachable coordinate intervals
    # from outer to inner, retaining a stride only when its start is fixed.
    for name, header in ctx.stride_operands.items():
        start, stop, stride = [ast.parse(source, mode="eval").body for source in header]
        left, right, delta = interval(start), interval(stop), interval(stride)
        if left is None or right is None:
            continue
        step = delta[0] if delta is not None and delta[0] == delta[1] else None
        first, last = polynomial(start), polynomial(stop)
        travel_lo, travel_hi = right[0] - left[1], right[1] - left[0]
        if first is not None and last is not None:
            distance = sympy.Add(last, -first)
            if isinstance(distance, sympy.Integer):
                travel_lo = travel_hi = int(distance)
        # A checked nonzero stride has at least unit magnitude. Its interval
        # can prove a singleton without itself being a singleton.
        forward = travel_hi > 0 and (delta is None or delta[1] > 0)
        backward = travel_lo < 0 and (delta is None or delta[0] < 0)
        if not forward and not backward:
            return "1"
        positive = max(1, delta[0]) if delta is not None else 1
        negative = min(-1, delta[1]) if delta is not None else -1
        only_start = (not forward or travel_hi <= positive) and (
            not backward or travel_lo >= negative
        )
        if only_start:
            domain = reachable(start, left)
            if not domain:
                return "1"
            bounds[name] = domain
            if first is not None:
                aliases[name] = first
            continue
        if first is not None:
            # The nested coordinate is its enclosing start plus the progress
            # within this interval; treating both as independent can invent
            # a difference larger than the entire loop.
            relative = sympy.Symbol(f"{name}#relative", integer=True)
            low = travel_lo + 1 if backward else 0
            high = travel_hi - 1 if forward else 0
            if step is not None:
                bounds[relative.name] = range(
                    0, travel_hi if step > 0 else travel_lo, step
                )
            else:
                bounds[relative.name] = range(low, high + 1)
            aliases[name] = sympy.Add(first, relative)
        if left[0] == left[1] and right[0] == right[1]:
            domain = _coordinates(left[0], right[0], step)
        elif step is None:
            domain = range(min(left[0], right[0]), max(left[1], right[1]) + 1)
        elif step > 0:
            domain = range(left[0], right[1], step if left[0] == left[1] else 1)
        else:
            domain = range(left[1], right[0], step if left[0] == left[1] else -1)
        if not domain:
            return "1"
        bounds[name] = domain
    analyzing_header = False
    interval(operation)
    return ast.unparse(operation) if guarded else source


def dynamic_fragment_step(
    ctx: FragmentContext, node: Node, step: object
) -> tuple[str, bool, str]:
    """Return the device stride, its zero-check flag and its proof in nested loops."""
    value = step.meta["val"] if isinstance(step, Node) else step
    expr = value._sympy_() if isinstance(value, torch.SymInt) else value
    host = HostFunction.current()
    if not isinstance(expr, sympy.Expr):
        raise exc.InvalidConfig("a dynamic fragment loop step must be an integer")
    if not host.is_host_scalar_expr(expr):
        # Render the lexical coordinates before their arithmetic, including
        # nested grid indices, tile origins and non-affine expressions.
        return (
            f"cutlass.Int64({device_range_expr(ctx, expr, 'stride')})",
            True,
            ctx.sym(expr, wide=True),
        )

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
        host.name,
        *code.co_varnames,
        *code.co_names,
        *code.co_freevars,
        *code.co_cellvars,
    }

    def exception_name(name: str) -> str:
        if name in host_names:
            return (
                HostFunction.current()
                .import_from_module(vars(builtins), name)
                .host_str()
            )
        return name

    name = ctx.df.new_var("fragment_host_step")
    while name in host_names:
        name = ctx.df.new_var("fragment_host_step")
    i64 = torch.iinfo(torch.int64)
    if reached or isinstance(expr, sympy.Symbol):
        ctx.host_step_statements.append(statement_from_string(f"{name} = {operand}"))
    else:
        # A conditional range need not evaluate e.g. n // divisor when dead.
        ctx.host_step_statements.append(
            statement_from_string(
                f"try:\n    {name} = {operand}\nexcept {exception_name('ArithmeticError')}:\n    {name} = 0"
            )
        )
    if reached:
        ctx.host_step_statements.append(
            statement_from_string(
                f"if {name} == 0:\n    raise {exception_name('ValueError')}('range() arg 3 must not be zero')"
                f"\nelif not {i64.min} <= {name} <= {i64.max}:"
                f"\n    raise {exception_name('OverflowError')}('fragment loop step exceeds signed Int64 range')"
            )
        )
    else:
        # A false device branch or an empty enclosing range must not raise.
        # Encode an invalid host result as zero; only a reached loop rejects it.
        ctx.host_step_statements.append(
            statement_from_string(
                f"if not {i64.min} <= {name} <= {i64.max}:\n    {name} = 0"
            )
        )
    arg = SymbolArgument(name, name)
    ctx.host_step_arguments.append(arg)
    # Python intermediates need not fit Int64. Only the resulting argument
    # is evaluated on the device and can constrain a nested range.
    return f"cutlass.Int64({arg.name})", not reached, arg.name
