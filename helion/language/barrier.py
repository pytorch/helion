from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.type_propagation import BarrierResultType
from .._compiler.type_propagation import LiteralType
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState
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

    # A barrier introduces a sequential phase boundary between top-level loops,
    # so force persistent kernels (other PID choices are incompatible).
    env = CompileEnvironment.current()
    env.has_barrier = True
    for disallowed in ("flat", "xyz", "persistent_interleaved"):
        env.config_spec.disallow_pid_type(disallowed)

    # Return None literal with a dedicated marker type.
    return BarrierResultType(origin=origin, value=None)


@_decorators.codegen(barrier, "triton")
def _(state: CodegenState) -> object:
    # No device code emitted; barrier only affects host-side scheduling.
    return expr_from_string("None")


@_decorators.ref(barrier)
def _() -> None:
    # No-op in ref/interpret mode
    return None
