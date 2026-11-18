"""Triton-backend codegen for the ops defined in ``helion.language.debug_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``debug_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ... import exc
from ...language import _decorators
from ...language.debug_ops import breakpoint  # noqa: A004
from ...language.debug_ops import device_assert
from ..ast_extension import create
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(breakpoint, "triton")
def _(state: CodegenState) -> None:
    state.add_statement("breakpoint()")


@_decorators.codegen(device_assert, "triton")
def _(state: CodegenState) -> None:
    if CompileEnvironment.current().backend_name == "tileir":
        # TileIR compiles Triton's code with its own pipeline, which is not
        # known to keep tl.device_assert or to take the debug launch options.
        raise exc.BackendImplementationMissing("tileir", "hl.device_assert")
    # tl.device_assert(_mask_to, 'zero element'), which Triton keeps only in
    # debug mode (TritonBackend.build_launcher_args).
    state.add_statement(
        statement_from_string(
            "tl.device_assert({condition}, {message})",
            condition=state.ast_arg(0),
            message=create(ast.Constant, value=state.proxy_arg(1)),
        )
    )
