from __future__ import annotations

import contextlib

__all__ = ["device_scope"]


def device_scope() -> contextlib.AbstractContextManager[None]:
    """Run the body as device code inside a single program.
    Inside an ``@helion.kernel``, ``with hl.device_scope():`` is rewritten
    into a one-program ``hl.grid(1)`` loop before compilation.  This lets a
    kernel contain whole-tensor device operations (``x[:, :]`` subscripts,
    ``hl.arange``, ...) between top-level ``hl.tile``/``hl.grid`` loops, or
    wrap the whole kernel body to force single-program execution when loops
    must be serialized (e.g. gather/scatter hazards on shared buffers).
    Stores performed in the scope become dependencies that later top-level
    loops are scheduled after.

    In ref eager mode (and plain Python) the body simply runs eagerly.

    See Also:
        - :func:`~helion.language.grid`: The underlying one-program loop.
        - :func:`~helion.language.barrier`: Explicit phase separation
          between top-level loops.
    """
    return contextlib.nullcontext()
