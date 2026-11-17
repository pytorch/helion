"""An MLIR module that a test emits CuTe DSL operations into, without a kernel."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def sdk_module() -> Iterator[SimpleNamespace]:
    """An empty MLIR module with its body as the insertion point, and the SDK
    namespaces a layout test builds in it. Nothing inside may initialize
    CUDA.

    The SDK is imported here and not at module level: the modules that use
    this context are collected in installations without it, where their
    tests skip (``cute_codegen.require_cute_codegen``).
    """
    import cutlass
    from cutlass._mlir import ir
    from cutlass._mlir import passmanager
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute
    from cutlass.utils.gemm import sm100

    initialized = torch.cuda.is_initialized()
    # pyrefly: ignore [missing-attribute]
    with ir.Context(), ir.Location.unknown():
        # pyrefly: ignore [missing-attribute]
        module = ir.Module.create()
        # pyrefly: ignore [missing-attribute]
        with ir.InsertionPoint(module.body):
            yield SimpleNamespace(
                cutlass=cutlass,
                cute=cute,
                ir=ir,
                func=func,
                passmanager=passmanager,
                epi=sm100,
                module=module,
            )
    assert torch.cuda.is_initialized() == initialized
