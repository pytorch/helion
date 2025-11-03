from __future__ import annotations

import builtins

import sympy
import torch

from .._compiler.compile_environment import CompileEnvironment
from . import _decorators


def compute_symbolic_min_max(
    args: tuple[int | torch.SymInt, ...], is_min: bool
) -> torch.SymInt | int:
    """Core logic for computing symbolic min/max over integer arguments.

    Used by both device code execution and type propagation.
    """
    env = CompileEnvironment.current()
    shape_env = env.shape_env
    sympy_op = sympy.Min if is_min else sympy.Max
    hint_fn = min if is_min else max

    if isinstance(args[0], torch.SymInt):
        expr, hint = args[0]._sympy_(), env.size_hint(args[0])
    else:
        expr, hint = sympy.Integer(args[0]), args[0]

    for arg in args[1:]:
        if isinstance(arg, torch.SymInt):
            rhs_expr, rhs_hint = arg._sympy_(), env.size_hint(arg)
        else:
            rhs_expr, rhs_hint = sympy.Integer(arg), arg
        expr = sympy_op(expr, rhs_expr)  # type: ignore[call-arg]
        hint = hint_fn(hint, rhs_hint)  # type: ignore[arg-type]

    with shape_env.ignore_fresh_unbacked_symbols():
        return shape_env.create_symintnode(expr, hint=hint)  # type: ignore[return-value]


@_decorators.device_func_replacement(builtins.min)
def _device_min(*args: int | torch.SymInt) -> torch.SymInt | int:
    return compute_symbolic_min_max(args, is_min=True)


@_decorators.device_func_replacement(builtins.max)
def _device_max(*args: int | torch.SymInt) -> torch.SymInt | int:
    return compute_symbolic_min_max(args, is_min=False)
