from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from .. import exc
from .._compiler.type_info import PdlResultType
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.variable_origin import Origin

__all__ = ["code_warm", "pdl_launch_dependents", "pdl_wait"]


@_decorators.api(
    is_device_loop=False,
    is_device_only=False,
    cache_type=True,
    signature=inspect.signature(lambda: None),
)
def pdl_wait() -> None:
    """
    Wait until the previous kernel on the stream has finished and flushed memory.

    Call it on the host before the first ``hl.tile`` / ``hl.grid`` loop; the kernel
    waits at entry. Any PDL op launches the kernel with programmatic dependent
    launch, so its launch overlaps the previous kernel's tail on supported
    NVIDIA targets. Triton uses ordinary stream order on non-NVIDIA targets
    and before sm_90. Other backends may reject PDL operations; support is not
    implied by a non-NVIDIA target.
    """
    raise exc.NotInsideKernel


@_decorators.api(
    is_device_loop=False,
    is_device_only=False,
    cache_type=True,
    signature=inspect.signature(lambda: None),
)
def pdl_launch_dependents() -> None:
    """
    Let the next PDL-launched kernel on the stream start launching.

    Call it on the host before the first loop to trigger at kernel entry, or after
    the last loop to trigger at exit.
    """
    raise exc.NotInsideKernel


@_decorators.api(
    is_device_loop=False,
    is_device_only=False,
    cache_type=True,
    signature=inspect.signature(lambda: None),
)
def code_warm() -> None:
    """
    L2-warm the kernel's own native code at entry.

    Call it on the host before the first loop. A megakernel's code outgrows the
    launch prefetch, and the weights streamed between its launches evict it from
    L2, so its first tasks would fetch their instructions from DRAM; the warm
    loads the code into L2 ahead of the body (and of any PDL wait). It does
    nothing outside the Triton and Gluon backends on NVIDIA GPUs of compute
    capability 9.0 or newer.
    """
    raise exc.NotInsideKernel


def _pdl_type(origin: Origin, kind: str) -> PdlResultType:
    if origin.is_device():
        raise exc.PdlPlacement
    return PdlResultType(origin=origin, value=kind)


@_decorators.type_propagation(pdl_wait)
def _(origin: Origin, **kwargs: object) -> PdlResultType:
    return _pdl_type(origin, "wait")


@_decorators.type_propagation(pdl_launch_dependents)
def _(origin: Origin, **kwargs: object) -> PdlResultType:
    return _pdl_type(origin, "launch_dependents")


@_decorators.type_propagation(code_warm)
def _(origin: Origin, **kwargs: object) -> PdlResultType:
    return _pdl_type(origin, "code_warm")


@_decorators.ref(pdl_wait)
def _() -> None:
    return None


@_decorators.ref(code_warm)
def _() -> None:
    return None


@_decorators.ref(pdl_launch_dependents)
def _() -> None:
    return None
