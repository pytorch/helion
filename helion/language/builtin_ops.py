from __future__ import annotations

import builtins
from typing import TYPE_CHECKING
from typing import cast

import torch

from .._compiler.compile_environment import compute_symbolic_min_max
from . import _decorators

if TYPE_CHECKING:
    from collections.abc import Callable


def _compute_scalar_tensor_min_max(
    args: tuple[int | torch.SymInt | torch.Tensor, ...],
    elementwise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Reduce mixed int/SymInt/0-D tensor args with ``elementwise``.

    Non-tensor args are materialized against the first tensor, so the result
    keeps that tensor's dtype and device.
    """
    reference = next(arg for arg in args if isinstance(arg, torch.Tensor))
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
        result = elementwise(result, arg)
    return result


def _unpack_one_sequence(
    args: tuple[int | torch.SymInt | torch.Tensor, ...],
) -> tuple[int | torch.SymInt | torch.Tensor, ...]:
    """The items of a single list or tuple argument (``min([a, b])``), which
    Python's min and max compare as the separate arguments."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return tuple(args[0])
    return args


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
    args = _unpack_one_sequence(args)
    if any(isinstance(arg, torch.Tensor) for arg in args):
        return _compute_scalar_tensor_min_max(args, torch.minimum)
    return compute_symbolic_min_max(args, op=builtins.min)  # type: ignore[arg-type]


@_decorators.device_func_replacement(builtins.max)
def _builtin_max(
    *args: int | torch.SymInt | torch.Tensor,
) -> torch.SymInt | torch.Tensor | int:
    """Device replacement for max() over symbolic ints or scalar tensors.

    A scalar tensor result is used when any input is a scalar tensor; otherwise
    symbolic integer expressions are preserved.

    Args:
        *args: Concrete ints, symbolic SymInts, or scalar tensors.

    Returns:
        The maximum value with the corresponding scalar representation.
    """
    args = _unpack_one_sequence(args)
    if any(isinstance(arg, torch.Tensor) for arg in args):
        return _compute_scalar_tensor_min_max(args, torch.maximum)
    return compute_symbolic_min_max(args, op=builtins.max)  # type: ignore[arg-type]
