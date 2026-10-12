"""Spyre codegen for ``helion.language.memory_ops``.

Plan 2: memory nodes produce **no OpSpec**. SuperDSC has no load/store op -- an
``OpSpec`` reads and writes buffers directly through
``TensorArg.device_coordinates`` -- so a ``load`` only *fixes an address
expression*, and a ``store`` only tells the producing OpSpec where to write.

Both therefore return an ``Operand`` (a buffer handle), which is legal because
the per-node lowering seam passes non-``ast.AST`` results straight through
(inductor_lowering.py:1657-1662).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _decorators
from ...language.memory_ops import load
from ...language.memory_ops import store

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState
    from .backend import SpyreBackend
    from .emitter import Operand


def host_operand(state: CodegenState, tensor: torch.Tensor) -> Operand:
    """The buffer for a kernel-argument tensor, registering it on first use.

    Argument order follows Helion's own ``tensor_arg`` table, so the emitted
    ``arg_index`` matches the launcher's positional order.
    """
    from ..compile_environment import CompileEnvironment

    backend = cast("SpyreBackend", CompileEnvironment.current().backend)
    name = state.device_function.tensor_arg(tensor).name
    return backend.emitter.add_arg(name, tensor)


@_decorators.codegen(load, "spyre")
def _(state: CodegenState) -> object:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    _reject_narrowing(subscript)
    return host_operand(state, tensor)


@_decorators.codegen(store, "spyre")
def _(state: CodegenState) -> None:
    """Redirect the producing OpSpec's output to this tensor's buffer.

    Plan 2: a store "makes ``v``'s producing OpSpec write ``out`` directly".
    The value node already emitted its OpSpec, so rather than emit a copy we
    rewrite that OpSpec's output ``TensorArg`` in place to address ``out``.
    """
    from ..compile_environment import CompileEnvironment

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    _reject_narrowing(subscript)

    backend = cast("SpyreBackend", CompileEnvironment.current().backend)
    emitter = backend.emitter
    dest = host_operand(state, tensor)

    value = state.ast_args[2] if len(state.ast_args) > 2 else None
    from .emitter import Operand as _Operand

    if not isinstance(value, _Operand):
        raise exc.BackendUnsupported(
            "spyre", f"store of a non-buffer value: {type(value).__name__}"
        )
    emitter.redirect_output(value, dest)


def _reject_narrowing(subscript: object) -> None:
    """Fail loudly on indexing this backend does not model yet.

    A Spyre tile is an address range, so a narrowing subscript would change
    ``device_coordinates`` rather than select registers. Silently ignoring it
    is wrong numerics, so plan 2's "never a silent fallback" applies.
    """
    assert isinstance(subscript, (list, tuple))
    for index in subscript:
        if isinstance(index, slice):
            if index != slice(None):
                raise exc.BackendUnsupported("spyre", f"narrowing slice {index!r}")
        elif not isinstance(index, (torch.SymInt, int)):
            raise exc.BackendUnsupported(
                "spyre", f"subscript of type {type(index).__name__}"
            )
