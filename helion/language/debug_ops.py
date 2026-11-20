from __future__ import annotations

import builtins
import os
from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from .._compiler.compile_environment import CompileEnvironment
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


@has_side_effect
@_decorators.api(is_device_only=True)
def device_assert(condition: torch.Tensor | bool, message: str = "") -> None:
    """Check a condition in device code, as ``tl.device_assert`` does: a
    kernel in which an element of ``condition`` is false stops, and the host
    sees the failure at its next synchronization, after which the process's
    CUDA context is unusable.  The lanes past the end of a partial tile are
    not checked.  CuTe may run the check after the loads that follow it in
    the tile (a gather it guards can fault first), but before the stores.

    Args:
        condition: A bool tensor, or a bool computed in device code
        message: A string printed by a failing thread

    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(device_assert)
def _(condition: object, message: str = "") -> None:
    if not (
        isinstance(condition, (bool, torch.SymBool))
        or (isinstance(condition, torch.Tensor) and condition.dtype == torch.bool)
    ) or not isinstance(message, str):
        raise exc.TypeInferenceError(
            "hl.device_assert() takes a bool condition and a string message"
        )


@_decorators.post_create_proxy(device_assert)
def _(node: torch.fx.Node, condition: object, message: str = "") -> None:
    from .._compiler.node_masking import apply_masking

    CompileEnvironment.current().has_device_assert = True
    # The lanes past the end of a partial tile read as true.  For
    # hl.device_assert(x[tile] != 0) over 1000 elements in tiles of 256:
    #     load = tl.load(x + indices_0 * 1, mask_0, other=0)
    #     v_1 = load != v_0
    #     _mask_to = tl.where(mask_0, v_1, tl.full([], True, tl.int1))
    #     tl.device_assert(_mask_to, 'zero element')
    # Without the mask, tl.device_assert(v_1, ...) fails on lanes 1000 to 1023,
    # which the masked load filled with 0.
    cond = node.args[0]
    if isinstance(cond, torch.fx.Node) and isinstance(cond.meta["val"], torch.Tensor):
        node.args = (apply_masking(cond, base_node=node, other=True), message)


@_decorators.ref(device_assert)
def _(condition: object, message: str = "") -> None:
    if not torch.as_tensor(condition).all():
        raise AssertionError(message)
