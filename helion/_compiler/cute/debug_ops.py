"""CuTe-backend codegen for the ops defined in ``helion.language.debug_ops``.

Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.debug_ops import device_assert
from ..ast_extension import create
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState

# MLIR's cf.assert: a false one prints the message and traps.
CUTE_DEVICE_ASSERT = "cute.testing.assert_"


@_decorators.codegen(device_assert, "cute")
def _(state: CodegenState) -> None:
    # cute.testing.assert_(_mask_to, 'zero element'), which the DSL keeps
    # only under --enable-assertions (CuteBackend.launcher_keyword_args).
    state.add_statement(
        statement_from_string(
            f"{CUTE_DEVICE_ASSERT}({{condition}}, {{message}})",
            condition=state.ast_arg(0),
            message=create(ast.Constant, value=state.proxy_arg(1)),
        )
    )
