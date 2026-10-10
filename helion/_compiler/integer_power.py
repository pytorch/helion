"""Exact bounded integer powers in scalar device expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import sympy
from torch.utils._sympy.functions import PowByNatural
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._sympy.value_ranges import bound_sympy

from .. import exc
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .tile_strategy import DeviceLoopState
from .variable_origin import GridOrigin
from .variable_origin import TileBeginOrigin

if TYPE_CHECKING:
    from .device_function import DeviceFunction


class IntegerPowerOfTwo(sympy.Function):
    """A power of two proved to fit a positive signed Int64 value."""

    nargs = (1,)
    is_integer = True
    is_positive = True


def lower_integer_powers(
    expression: sympy.Expr, ranges: dict[sympy.Symbol, ValueRanges], *, backend: str
) -> sympy.Expr:
    """Lower only powers whose entire integer result range is representable.

    PowByNatural is integer shape arithmetic, not floating exponentiation.
    A constant power-of-two base admits an exact shift when its effective
    exponent lies in [0, 62]. CuTe requires this proof for scalar integer
    powers; other backends keep their existing printer for unproved powers.
    """
    replacements: dict[sympy.Basic, sympy.Basic] = {}
    for power in sympy.postorder_traversal(expression):
        if not isinstance(power, PowByNatural):
            continue
        base, exponent = power.args
        if base == 1:
            replacements[power] = sympy.Integer(1)
            continue
        if (
            not isinstance(base, sympy.Integer)
            or base <= 0
            or int(base).bit_count() != 1
        ):
            if backend == "cute":
                raise exc.BackendUnsupported(backend, "unproved scalar integer power")
            continue
        shift = sympy.Mul(sympy.Integer(int(base).bit_length() - 1), exponent)
        bounds = bound_sympy(shift, ranges)
        if not (bounds.lower >= 0 and bounds.upper <= 62):
            if backend == "cute":
                raise exc.BackendUnsupported(
                    backend,
                    "scalar integer power requires a proved exponent in [0, 62]",
                )
            continue
        replacements[power] = cast(
            "sympy.Expr", IntegerPowerOfTwo(shift.xreplace(replacements))
        )
    return expression.xreplace(replacements)


def prepare_integer_powers(
    expression: sympy.Expr, function: DeviceFunction
) -> sympy.Expr:
    """Use original symbols and active scalar-loop bounds before renaming."""
    if not expression.has(PowByNatural):
        return expression
    env = CompileEnvironment.current()
    ranges = dict(env.shape_env.var_to_range)
    origins = HostFunction.current().expr_to_origin
    for symbol in expression.free_symbols:
        assert isinstance(symbol, sympy.Symbol)
        info = origins.get(symbol)
        if info is None or not isinstance(info.origin, GridOrigin):
            continue
        if type(info.origin) not in (GridOrigin, TileBeginOrigin):
            continue
        block = env.resolve_codegen_block_id(info.origin.block_id, function.codegen)
        active = function.codegen.active_device_loops.get(block)
        if not active:
            continue
        # GridOrigin is a scalar range index; its registered block size may
        # encode a nonunit (or negative) step. A TileBeginOrigin must instead
        # prove a single-element tile before using the scalar endpoint proof.
        if (
            isinstance(info.origin, TileBeginOrigin)
            and function.resolved_block_size(block) != 1
        ):
            continue
        loop = active[-1]
        if not isinstance(loop, DeviceLoopState):
            continue
        dimension = loop.block_id_to_info[block]
        begin, end = dimension.begin_expr, dimension.end_expr
        if begin is None or end is None:
            continue
        begin, end = env.specialize_expr(begin), env.specialize_expr(end)
        if not isinstance(begin, sympy.Integer) or not isinstance(end, sympy.Integer):
            continue
        # Any executed iteration is inside these endpoint bounds, including
        # stepped and descending ranges. Empty ranges contribute no proof.
        if begin < end:
            interval = ValueRanges(begin, end - 1)
        elif begin > end:
            interval = ValueRanges(end + 1, begin)
        else:
            continue
        ranges[symbol] = ranges.get(symbol, ValueRanges.unknown_int()) & interval
    return lower_integer_powers(expression, ranges, backend=env.backend.name)
