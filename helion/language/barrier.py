from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from .. import exc
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.type_info import BarrierResultType
from .._compiler.type_info import LiteralType
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.variable_origin import Origin

__all__ = ["barrier"]


@_decorators.api(
    is_device_loop=False,
    is_device_only=False,
    cache_type=True,
    signature=inspect.signature(lambda: None),
)
def barrier() -> None:
    """Grid-wide barrier separating top-level `hl.tile` / `hl.grid` loops."""
    raise exc.NotInsideKernel


@_decorators.type_propagation(barrier)
def _(origin: Origin, **kwargs: object) -> LiteralType:
    # Only allowed on the host between top-level device loops.
    if origin.is_device():
        raise exc.BarrierOnlyAllowedAtTopLevel

    env = CompileEnvironment.current()
    # A barrier introduces a sequential phase boundary between top-level loops,
    # so force persistent kernels (other PID choices are incompatible).
    env.has_barrier = True
    for disallowed in ("flat", "xyz", "persistent_interleaved"):
        env.config_spec.disallow_pid_type(disallowed)

    # Return None literal with a dedicated marker type.
    return BarrierResultType(origin=origin, value=None)


@_decorators.ref(barrier)
def _() -> None:
    # No-op in ref/interpret mode
    return None


# ---------------------------------------------------------------------------
# Backend-specific codegens for these ops live in per-backend modules under
# helion/_compiler/<backend>/.  Import them here (at module import time) so the
# @_decorators.codegen(op, "<backend>") registrations run with the same eager
# timing as when the bodies lived in this file -- no behavior change.
import helion._compiler.cute.barrier  # noqa: E402, F401
import helion._compiler.triton.barrier  # noqa: E402, F401
