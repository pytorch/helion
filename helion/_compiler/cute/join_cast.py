"""Control-flow join helper for CuTe codegen.

Called during ``@cute.kernel`` tracing (plain Python), so no ``@cute.jit``
wrapper is needed.
"""

from __future__ import annotations

import cutlass


def join_cast(value: object, like: object) -> object:
    """Give ``value`` the numeric type ``like`` has.

    CuTe DSL requires a variable reassigned inside a dynamic ``if`` to keep
    the type it had before the ``if``.  That type is only known while
    tracing: emitted values keep their DSL type (fp32 math on 16-bit
    tensors, index math on Int64 shape arguments), which can differ from the
    FX dtype.
    """
    if isinstance(like, cutlass.Numeric):
        return type(like)(value)
    return value
