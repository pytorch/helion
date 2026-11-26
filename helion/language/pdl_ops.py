from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.type_info import PdlResultType
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.variable_origin import Origin

__all__ = ["pdl_launch_dependents", "pdl_wait", "prefetch_code"]


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

    Call it on the host before the first device loop to trigger at kernel entry,
    or after the last one to trigger at exit (host loops do not count). Between
    two top level device loops of a kernel whose loops run as one persistent
    megakernel with static or claimed task dispatch, each program triggers at
    the later loop's block (or, when that loop's tasks continue an earlier
    loop's, the first resident block after it, at exit if none follows);
    elsewhere (packet dispatch, the barrier pipeline, no megakernel) such a call
    triggers at exit. The next kernel's launch then overlaps the rest of this
    one, while its wait still follows this kernel's end.
    """
    raise exc.NotInsideKernel


@_decorators.api(
    is_device_loop=False,
    is_device_only=False,
    cache_type=True,
    signature=inspect.signature(lambda: None),
)
def prefetch_code() -> None:
    """
    Prefetch the kernel's own native code into L2.

    Call it on the host before the first loop or between two loops, as often as
    needed. A megakernel's code outgrows the launch prefetch, and the weights
    streamed between its launches evict it from L2, so its first tasks would
    fetch their instructions from DRAM; the prefetch loads the code into L2
    ahead of the body (and of any PDL wait). Between two loops it runs where the
    megakernel's workers reach the second one (in a Gluon ring, its idle MMA
    partition), so the code of the roots after a large weight stream is warm
    when they run; where workers do not run that root's tasks as a block of
    their own (packet dispatch, barrier phases, a non-persistent launch, a root
    continuing an earlier root's tasks) it runs at entry. Every 16th CTA reads
    a share of the text. It does nothing outside the Triton and Gluon backends
    on NVIDIA GPUs of compute capability 9.0 or newer.
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


@_decorators.type_propagation(prefetch_code)
def _(origin: Origin, **kwargs: object) -> PdlResultType:
    return _pdl_type(origin, "prefetch_code")


@_decorators.ref(pdl_wait)
def _() -> None:
    return None


@_decorators.codegen(prefetch_code, "common")
def _(state: CodegenState) -> ast.AST:
    # DeviceFunction emits the prefetch (Triton and Gluon on sm_90+); every other
    # backend compiles the call to nothing.
    return expr_from_string("None")


@_decorators.ref(prefetch_code)
def _() -> None:
    return None


@_decorators.ref(pdl_launch_dependents)
def _() -> None:
    return None
