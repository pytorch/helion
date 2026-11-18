from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.type_info import PdlResultType
from .._compiler.type_info import TypeInfo
from . import _decorators

if TYPE_CHECKING:
    import ast
    from collections.abc import Sequence

    import torch

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.variable_origin import Origin

__all__ = ["pdl_launch_dependents", "pdl_wait", "prefetch_code"]


@_decorators.api(
    is_device_loop=False,
    is_device_only=False,
    cache_type=True,
    signature=inspect.signature(lambda independent=(): None),
)
def pdl_wait(independent: torch.Tensor | Sequence[torch.Tensor] = ()) -> None:
    """
    Wait until the previous kernel on the stream has finished and flushed memory.

    Call it on the host before the first ``hl.tile`` / ``hl.grid`` loop; the kernel
    waits at entry. Any PDL op launches the kernel with programmatic dependent
    launch, so its launch overlaps the previous kernel's tail on supported
    NVIDIA targets. Triton uses ordinary stream order on non-NVIDIA targets
    and before sm_90. Other backends may reject PDL operations; support is not
    implied by a non-NVIDIA target.

    Args:
        independent: A tensor argument of the kernel, or a sequence of them
            (weights, for example), that no kernel still running when this one
            starts writes. Code that reads only these may run before the wait:
            a Gluon ring's producers start loading them while the previous
            kernel finishes. Every other access still follows the wait. The
            kernel rejects them (``exc.PdlIndependentArgument``) when its own
            code may write them: through a store or atomic (also to a target a
            host branch picks), an op it gives them to (``hl.inline_triton``,
            distributed ops), device pointers (a stack tensor store rejects
            every one), an in-place op of its host code (which runs as the
            kernel's predecessor), or another argument sharing their memory
            (where the binding keys arguments by overlap, as Triton's does).
            What it cannot check: such an argument elsewhere (the caller must
            not write them through one), a snippet (``hl.inline_triton``)
            writing through an address it computes from values it is given
            (an integer argument, a loaded word), the previous kernel
            triggering after
            its own wait (a kernel triggering at entry, before its wait, lets
            this one start while that kernel's predecessor still runs), and
            other ranks (PDL orders nothing across ranks, so a symmetric tensor
            must not be written by any rank's running kernel). Any of these is
            a data race. Call ``hl.pdl_wait`` through a global or a module
            (``hl.pdl_wait``, an imported alias) to pass it: the compiler finds
            the call in the kernel's source.
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
    the later loop's block (or the first resident block after it, when that
    loop's tasks continue an earlier loop's); elsewhere (packet dispatch, the
    barrier pipeline, no megakernel) such a call triggers at exit. The next
    kernel's launch then overlaps the rest of this one, while its wait still
    follows this kernel's end.
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
    Prefetch the kernel's own native code into L2 at entry.

    Call it on the host before the first loop. A megakernel's code outgrows the
    launch prefetch, and the weights streamed between its launches evict it from
    L2, so its first tasks would fetch their instructions from DRAM; the prefetch
    loads the code into L2 ahead of the body (and of any PDL wait). It does
    nothing outside the Triton and Gluon backends on NVIDIA GPUs of compute
    capability 9.0 or newer.
    """
    raise exc.NotInsideKernel


def _pdl_type(
    origin: Origin, kind: str, independent: TypeInfo | None = None
) -> PdlResultType:
    if origin.is_device():
        raise exc.PdlPlacement
    return PdlResultType(origin=origin, value=kind, independent=independent)


@_decorators.type_propagation(pdl_wait)
def _(independent: object = (), *, origin: Origin) -> PdlResultType:
    # The bound argument, also through ``*`` or ``**`` unpacking.
    return _pdl_type(
        origin, "wait", independent if isinstance(independent, TypeInfo) else None
    )


@_decorators.type_propagation(pdl_launch_dependents)
def _(origin: Origin, **kwargs: object) -> PdlResultType:
    return _pdl_type(origin, "launch_dependents")


@_decorators.type_propagation(prefetch_code)
def _(origin: Origin, **kwargs: object) -> PdlResultType:
    return _pdl_type(origin, "prefetch_code")


@_decorators.ref(pdl_wait)
def _(independent: torch.Tensor | Sequence[torch.Tensor] = ()) -> None:
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
