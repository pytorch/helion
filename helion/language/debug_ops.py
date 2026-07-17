from __future__ import annotations

import builtins
import os
from typing import TYPE_CHECKING

from torch.fx import has_side_effect

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.type_info import TypeInfo
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
    from .._compiler.type_info import LiteralType

    if args or kwargs:
        raise exc.TypeInferenceError("breakpoint() does not take any arguments")
    assert origin.is_device()
    if not (
        os.environ.get("TRITON_INTERPRET") == "1"
        or os.environ.get("HELION_INTERPRET") == "1"
    ):
        raise exc.BreakpointInDeviceLoopRequiresInterpret
    return LiteralType(origin, None)


@_decorators.ref(breakpoint)
def _() -> None:
    builtins.breakpoint()


# ---------------------------------------------------------------------------
# Backend-specific codegens for these ops live in per-backend modules under
# helion/_compiler/<backend>/.  Import them here (at module import time) so the
# @_decorators.codegen(op, "<backend>") registrations run with the same eager
# timing as when the bodies lived in this file -- no behavior change.
import helion._compiler.triton.debug_ops  # noqa: E402, F401
