from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

import sympy

from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import _to_sympy
from . import _decorators

if TYPE_CHECKING:
    import torch


def compute_symbolic_min_max(
    args: tuple[int | torch.SymInt, ...], op: object
) -> torch.SymInt | int:
    env = CompileEnvironment.current()
    shape_env = env.shape_env
    sympy_op = sympy.Min if op is builtins.min else sympy.Max
    hint_fn = min if op is builtins.min else max

    expr = _to_sympy(args[0])
    hint = env.size_hint(args[0])

    for arg in args[1:]:
        rhs_expr = _to_sympy(arg)
        rhs_hint = env.size_hint(arg)
        expr = sympy_op(expr, rhs_expr)  # type: ignore[call-arg]
        hint = hint_fn(hint, rhs_hint)  # type: ignore[arg-type]

    return shape_env.create_symintnode(expr, hint=hint)  # type: ignore[return-value]


@_decorators.device_func_replacement(builtins.min)
def _builtin_min(*args: int | torch.SymInt) -> torch.SymInt | int:
    """Device replacement for builtin min() that supports symbolic integers.

    Returns the minimum value among the provided arguments, preserving
    symbolic integer expressions when present.

    Args:
        *args: Integer arguments, which may be concrete ints or symbolic SymInts

    Returns:
        The minimum value, as a SymInt if any argument is symbolic, otherwise int
    """
    return compute_symbolic_min_max(args, op=builtins.min)


@_decorators.device_func_replacement(builtins.max)
def _builtin_max(*args: int | torch.SymInt) -> torch.SymInt | int:
    """Device replacement for builtin max() that supports symbolic integers.

    Returns the maximum value among the provided arguments, preserving
    symbolic integer expressions when present.

    Args:
        *args: Integer arguments, which may be concrete ints or symbolic SymInts

    Returns:
        The maximum value, as a SymInt if any argument is symbolic, otherwise int
    """
    return compute_symbolic_min_max(args, op=builtins.max)
