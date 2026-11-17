"""Triton-backend codegen for the ops defined in ``helion.language.debug_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``debug_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.debug_ops import breakpoint  # noqa: A004
from ...language.debug_ops import device_assert
from ..device_assert_codegen import emit_device_assert

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(breakpoint, "triton")
def _(state: CodegenState) -> None:
    state.add_statement("breakpoint()")


@_decorators.codegen(device_assert, "triton")
def _(state: CodegenState) -> None:
    # tl.device_assert checks the elements of the tensor each thread holds and
    # calls CUDA's __assertfail for a false one; Triton keeps it only in debug
    # mode, which the launch then asks for (TritonBackend.build_launcher_args).
    emit_device_assert(state, "tl.device_assert")
