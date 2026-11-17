from __future__ import annotations

import builtins
import contextlib
import os
import threading
from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    """Check ``condition`` on the device: a kernel in which an element of it is
    false (zero) stops, as ``tl.device_assert`` and CUDA's ``assert`` do.

    The host sees the failure at its next synchronization, as "device-side
    assert triggered" from a Triton kernel and "unspecified launch failure"
    from a CuTe one, and the process's CUDA context is unusable after it.
    The failing threads print ``message``.

    Only the elements of the tiles are checked: the lanes past the end of a
    partial tile, and those past the length of an ``hl.arange``, are not.

    ``Settings.device_asserts`` (``HELION_DEVICE_ASSERTS``) set to False
    compiles the checks out, for performance.  A condition on host values
    (sizes, scalar arguments) is better checked by a Python ``assert`` on the
    host, once per call.

    The Triton and CuTe backends check it; the others compile a kernel with
    one only with the checks compiled out.  It is not allowed in the combine
    function of ``hl.reduce`` or ``hl.associative_scan``.

    Args:
        condition: A tensor (of any shape and dtype; nonzero is true) or a bool
            computed in device code
        message: A string literal printed by a failing thread

    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(device_assert)
def _(condition: object, message: str = "") -> None:
    return None


@_decorators.type_propagation(device_assert)
def _(
    condition: TypeInfo, message: TypeInfo | None = None, *, origin: Origin
) -> TypeInfo:
    from .._compiler.type_info import LiteralType
    from .._compiler.type_info import NoType
    from .._compiler.type_info import SymBoolType
    from .._compiler.type_info import TensorType

    if not (
        isinstance(condition, (TensorType, SymBoolType))
        or (isinstance(condition, LiteralType) and isinstance(condition.value, bool))
    ):
        raise exc.TypeInferenceError(
            f"hl.device_assert() takes a tensor or bool condition, got {condition}"
        )
    if message is not None and not (
        isinstance(message, LiteralType) and isinstance(message.value, str)
    ):
        raise exc.TypeInferenceError(
            f"hl.device_assert() takes a string literal message, got {message}"
        )
    return NoType(origin)


class _CombineFunctionTracing(threading.local):
    depth = 0


_combine_function = _CombineFunctionTracing()


@contextlib.contextmanager
def tracing_combine_function() -> Iterator[None]:
    """Trace the combine function of ``hl.reduce`` or ``hl.associative_scan``,
    which runs on pairs of partial results that no lane mask covers: an
    ``hl.device_assert`` in it is rejected."""
    _combine_function.depth += 1
    try:
        yield
    finally:
        _combine_function.depth -= 1


@_decorators.register_to_device_ir(device_assert)
def _(tracer: object, condition: object, message: str = "") -> None:
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.node_masking import apply_masking

    if _combine_function.depth:
        raise exc.InvalidAPIUsage(
            "hl.device_assert() in the combine function of hl.reduce or "
            "hl.associative_scan, which runs on pairs of partial results"
        )
    # Settings.device_asserts=False traces no node, and the condition, which
    # nothing else reads, is dead code: nothing of the check is emitted.
    env = CompileEnvironment.current()
    if not env.settings.device_asserts:
        return
    if not env.backend.supports_device_asserts:
        raise exc.BackendUnsupported(
            env.backend_name, "hl.device_assert (compile it out: device_asserts=False)"
        )
    proxy_out = tracer.create_proxy(  # type: ignore[attr-defined]
        "call_function",
        device_assert,
        *_decorators.args_to_proxies(
            tracer,  # pyrefly: ignore[bad-argument-type]
            (condition, message),
            {},
        ),
    )
    node = proxy_out.node
    cond = node.args[0]
    if not isinstance(cond, torch.fx.Node) or not isinstance(
        cond.meta.get("val"), torch.Tensor
    ):
        return
    # The lanes past the end of a partial tile hold no element: a condition
    # computed from them, as idx < n of the values a masked load gives them,
    # would fail there.  They read as true, as a reduction's padding reads as
    # its identity.
    node.args = (apply_masking(cond, base_node=node, other=True), *node.args[1:])


@_decorators.ref(device_assert)
def _(condition: object, message: str = "") -> None:
    from .._compiler.compile_environment import CompileEnvironment

    if (
        CompileEnvironment.current().settings.device_asserts
        and not torch.as_tensor(condition).all()
    ):
        raise AssertionError(message)
