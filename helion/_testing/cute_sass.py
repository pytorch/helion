"""The SASS of the kernel a compiled CuTe host function launches.

For tests that read the instructions ptxas emitted
(``test/test_cute_exact_division.py``'s load and MUFU.EX2 counts,
``test/test_cute_mma_dag.py``'s hinted-decode checks).  The SASS checker
itself (``helion.runtime.cute.sass_check``) enters the disk-cache key of every
kernel that declares a check, so these test-only readers live here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..runtime.cute import sass_check

if TYPE_CHECKING:
    from collections.abc import Callable


def cute_launcher_cubins(compiled: Callable[..., object]) -> list[bytes]:
    """The cubins of the kernel the host function ``compiled`` launches.

    ``compiled`` is ``BoundKernel.compile_config``'s host function, called at
    least once.  Its kernel is ``_helion_<name>`` in its globals (as
    ``CuteBackend._compiled_kernels`` finds it), with one compiled launcher.
    """
    name = compiled.__name__  # pyrefly: ignore[missing-attribute]
    kernel = compiled.__globals__[f"_helion_{name}"]  # pyrefly: ignore[missing-attribute]
    (launcher,) = kernel._helion_cute_compiled_launchers.values()
    return sass_check._cubin_blobs(launcher._compiled.ir_module)


def cute_launcher_sass(compiled: Callable[..., object]) -> str:
    """``nvdisasm -c`` of every cubin of the kernel ``compiled`` launches."""
    return "\n".join(
        sass_check.disassemble(cubin) for cubin in cute_launcher_cubins(compiled)
    )
