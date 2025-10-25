from __future__ import annotations

import builtins
import os
from typing import TYPE_CHECKING

from torch.fx import has_side_effect

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.type_propagation import TypeInfo
    from .._compiler.variable_origin import Origin


@has_side_effect
@_decorators.device_func_replacement(builtins.breakpoint)
@_decorators.api(is_device_only=True)
def breakpoint() -> None:  # noqa: A001
    """Breakpoint that works inside device loops.

    Maps to Python's breakpoint() in the generated Triton code when TRITON_INTERPRET=1 is set,
    or in the PyTorch eager code when HELION_INTERPRET=1 is set.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(breakpoint)
def _() -> None:
    return None


@_decorators.type_propagation(breakpoint)
def _(*args: object, origin: Origin, **kwargs: object) -> TypeInfo:
    from .._compiler.type_propagation import LiteralType

    if args or kwargs:
        raise exc.TypeInferenceError("breakpoint() does not take any arguments")
    assert origin.is_device()
    if not (
        os.environ.get("TRITON_INTERPRET") == "1"
        or os.environ.get("HELION_INTERPRET") == "1"
    ):
        raise exc.BreakpointInDeviceLoopRequiresInterpret
    return LiteralType(origin, None)


@_decorators.codegen(breakpoint)
def _(state: CodegenState) -> None:
    state.add_statement("breakpoint()")


@_decorators.ref(breakpoint)
def _() -> None:
    builtins.breakpoint()
