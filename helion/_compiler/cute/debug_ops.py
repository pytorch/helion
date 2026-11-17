"""CuTe-backend codegen for the ops defined in ``helion.language.debug_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.debug_ops import device_assert
from ..device_assert_codegen import emit_device_assert

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


# The CuTe DSL's device assert, which the lane passes name too.
CUTE_DEVICE_ASSERT = "cute.testing.assert_"
# Calls with an effect that touch no memory: an hl.device_assert checks its
# condition wherever its reads let it run, once or once per lane alike (a lane
# split moves it; a reduction's result is checked on every reduction thread).
# The kernel's dead-code elimination keeps them (``ast_read_writes``).
CUTE_MEMORYLESS_EFFECT_CALLS = frozenset({CUTE_DEVICE_ASSERT})


@_decorators.codegen(device_assert, "cute")
def _(state: CodegenState) -> None:
    # Each thread checks its own element.  cute.testing.assert_ is MLIR's
    # cf.assert, which the DSL compiles to a printf of the message and a trap
    # under --enable-assertions, the option the launch then asks for
    # (CuteBackend.launcher_keyword_args), and drops otherwise.
    emit_device_assert(state, CUTE_DEVICE_ASSERT)
