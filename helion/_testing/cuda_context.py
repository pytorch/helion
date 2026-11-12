"""CUDA context helpers for the CuTe launcher tests: a new thread starts
without a current context, as PyTorch's autograd device thread does."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from typing import TypeVar

# pyrefly: ignore [missing-import]
import cuda.bindings.driver as cuda_driver

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")


def on_new_thread(fn: Callable[[], _T]) -> _T:
    """``fn()`` on a new thread, which starts without a current CUDA context;
    its exception is re-raised."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn).result()


def current_context() -> int:
    """The calling thread's current CUDA context handle (0: none)."""
    return int(cuda_driver.cuCtxGetCurrent()[1])


def clear_current_context() -> None:
    """Leave the calling thread without a current CUDA context."""
    (error,) = cuda_driver.cuCtxSetCurrent(cuda_driver.CUcontext(0))
    assert error == cuda_driver.CUresult.CUDA_SUCCESS, error
