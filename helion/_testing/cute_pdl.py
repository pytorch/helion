"""Host helpers for dependent-launch correctness tests.

Triton stays lazy so CPU source tests can import these helpers without it.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from torch._inductor.codecache import PyCodeCache

from helion._compiler.cute.pdl_release import ZERO_FILL_KERNEL_PREFIX

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import torch

    from helion.runtime.kernel import BoundKernel


WAIT_CALL = "cute.arch.griddepcontrol_wait()"

PRODUCER_GRID = 128
PRODUCER_BLOCK = 1024
# A long producer must outlive host launch latency for the missing-wait
# negative control to expose its race reliably.
CONTROL_DELAY = 4096


def triton_kernels() -> ModuleType:
    """Import the native producer/consumer kernels only when executed."""
    import helion._testing.cute_pdl_kernels as kernels

    return kernels


def produce(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    delay: int = 64,
    release: bool = True,
    grid: int = PRODUCER_GRID,
    block: int = PRODUCER_BLOCK,
) -> None:
    """Copy through an optionally releasing, delayed Triton producer."""
    n = dst.numel()
    elements = (n + grid - 1) // grid
    per_program = (elements + block - 1) // block * block
    triton_kernels()._late_writer[(grid,)](
        src.view(-1),
        dst.view(-1),
        n,
        per_program,
        delay,
        RELEASE=release,
        BLOCK=block,
    )


def attribute_only_control(bound: BoundKernel, source: str) -> Callable[..., object]:
    """Keep the real dependent-launch attribute but remove its single wait."""
    lines = source.splitlines()
    assert sum(WAIT_CALL in line for line in lines) == 1
    # A guarded wait may be the only statement in an if-body. Preserve that
    # empty control arm while removing the synchronization being tested.
    kept = [line.replace(WAIT_CALL, "pass") for line in lines]
    control_source = "\n".join(kept) + "\n"
    module = PyCodeCache.load(control_source)
    bound.env.backend.annotate_compiled_module(
        module, control_source, bound.kernel.name
    )
    return getattr(module, bound.kernel.name)


def kernel_definition(code: str) -> ast.FunctionDef:
    """Select the kernel proper, excluding auxiliary zero-fill kernels."""
    module = ast.parse(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
        and not node.name.startswith(ZERO_FILL_KERNEL_PREFIX)
    ]
    return kernel
