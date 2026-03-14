from __future__ import annotations

import functools
from typing import TypeVar

F = TypeVar("F")

_DEVICE_FUNCTION_ATTR = "_helion_device_function"


def device_function(fn: F) -> F:
    """Mark a function as callable from inside ``hl.tile()`` device loops.

    Device functions must be **pure computations** over their tensor
    arguments — they may use standard PyTorch operators (``torch.add``,
    ``torch.mul``, ``torch.where``, etc.) but must **not** contain
    ``hl.tile()`` or ``hl.grid()`` loops, memory allocation, or
    side effects.

    During tracing, the function body is inlined into the caller's
    computation graph via ``make_fx``.  In future releases the compiler
    may emit device functions as separate Triton ``@triton.jit`` helpers
    to improve code reuse across kernels.

    Example::

        @hl.device_function
        def gelu(x: torch.Tensor) -> torch.Tensor:
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


        @helion.kernel()
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = gelu(x[tile])
            return out
    """
    setattr(fn, _DEVICE_FUNCTION_ATTR, True)

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        return fn(*args, **kwargs)

    setattr(wrapper, _DEVICE_FUNCTION_ATTR, True)
    return wrapper  # type: ignore[return-value]


def is_device_function(fn: object) -> bool:
    """Return True if *fn* was decorated with ``@device_function``."""
    return getattr(fn, _DEVICE_FUNCTION_ATTR, False) is True
