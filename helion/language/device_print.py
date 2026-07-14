from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from torch.fx import has_side_effect

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.type_info import TypeInfo
    from .._compiler.variable_origin import Origin


@has_side_effect
@_decorators.device_func_replacement(builtins.print)
@_decorators.api(is_device_only=False)
def device_print(prefix: str, *values: object) -> None:
    """
    Print values from device code.

    Args:
        prefix: A string prefix for the print statement
        values: Tensor values to print

    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(device_print)
def _(*values: object, sep: str = " ", end: str = "\n") -> None:
    return None


@_decorators.type_propagation(device_print)
def _(*args: object, origin: Origin, **kwargs: object) -> TypeInfo:
    from .._compiler.type_info import LiteralType
    from .._compiler.type_info import NoType
    from .._compiler.type_info import TensorType

    # Check that we have at least one argument (prefix)
    if len(args) == 0:
        raise ValueError("print() requires at least one argument (prefix)")

    # First argument must be the prefix string
    if not (isinstance(args[0], LiteralType) and isinstance(args[0].value, str)):
        raise TypeError(
            f"First argument to print() must be a string prefix, got {args[0]}"
        )

    # For compile-time values like tensor shapes, we should error out
    for i, arg in enumerate(args[1:]):
        if not isinstance(arg, TensorType):
            raise TypeError(
                f"print() only supports runtime tensor values. "
                f"Argument {i + 1} is {arg}, not a tensor. "
                f"Compile-time values like tensor shapes are not supported yet."
            )

    return NoType(origin)


@_decorators.ref(device_print)
def _(prefix: str, *values: object) -> None:
    print(prefix, *values)


# ---------------------------------------------------------------------------
# Backend-specific codegens for these ops live in per-backend modules under
# helion/_compiler/<backend>/.  Import them here (at module import time) so the
# @_decorators.codegen(op, "<backend>") registrations run with the same eager
# timing as when the bodies lived in this file -- no behavior change.
import helion._compiler.cute.device_print  # noqa: E402, F401
import helion._compiler.triton.device_print  # noqa: E402, F401
