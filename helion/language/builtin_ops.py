from __future__ import annotations

import builtins
from typing import cast

import sympy
import torch

from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import _to_sympy
from . import _decorators


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


def _compute_scalar_tensor_min(
    args: tuple[int | torch.SymInt | torch.Tensor, ...],
) -> torch.Tensor:
    reference = next(arg for arg in args if isinstance(arg, torch.Tensor))
    assert isinstance(reference, torch.Tensor)
    if any(isinstance(arg, torch.Tensor) and arg.ndim != 0 for arg in args):
        raise TypeError("device min/max only supports scalar tensor arguments")
    tensor_args = [
        arg
        if isinstance(arg, torch.Tensor)
        else torch.full_like(reference, cast("int", arg))
        for arg in args
    ]
    result = tensor_args[0]
    for arg in tensor_args[1:]:
        result = torch.minimum(result, arg)
    return result


@_decorators.device_func_replacement(builtins.min)
def _builtin_min(
    *args: int | torch.SymInt | torch.Tensor,
) -> torch.SymInt | torch.Tensor | int:
    """Device replacement for min() over symbolic ints or scalar tensors.

    A scalar tensor result is used when any input is a scalar tensor; otherwise
    symbolic integer expressions are preserved.

    Args:
        *args: Concrete ints, symbolic SymInts, or scalar tensors.

    Returns:
        The minimum value with the corresponding scalar representation.
    """
    if any(isinstance(arg, torch.Tensor) for arg in args):
        return _compute_scalar_tensor_min(args)
    return compute_symbolic_min_max(args, op=builtins.min)  # type: ignore[arg-type]


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
