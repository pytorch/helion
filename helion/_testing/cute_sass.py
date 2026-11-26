"""The SASS of the kernel a compiled CuTe host function launches.

For tests that read the instructions ptxas emitted
(``test/test_cute_exact_division.py``'s load and MUFU.EX2 counts,
``test/test_cute_mma_dag.py``'s hinted-decode checks).  The SASS checker
itself (``helion.runtime.cute.sass_check``) enters the disk-cache key of every
kernel that declares a check, so these test-only readers live here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

import pytest

from .._compiler.cute import cutedsl_compat
from ..runtime.cute import sass_check

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


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


def skip_unless_nvdisasm_runs() -> None:
    """Skip a test that reads SASS on a host whose ``nvdisasm`` cannot run
    (``cutedsl_compat.nvdisasm_unusable_reason``: the aarch64 wheel's binary
    on the GB300 hosts dies at startup)."""
    reason = cutedsl_compat.nvdisasm_unusable_reason()
    if reason is not None:
        raise unittest.SkipTest(f"nvdisasm cannot run on this host: {reason}")


@pytest.fixture
def hinted_packets_render() -> Iterator[None]:
    """Render hinted cp.async packets whatever the host's disassembler.

    For CPU tests of the emitted packets and seeds, which the launcher's SASS
    check verifies at launch: a host whose ``nvdisasm`` cannot run (the
    aarch64 GB300 wheel's) renders plain packets instead
    (``test_cute_mma_dag.py::test_hinted_packets_fall_back_where_nvdisasm_cannot_run``).
    Import it into a test module and name it in ``pytest.mark.usefixtures``.
    """
    with patch.object(cutedsl_compat, "nvdisasm_path", return_value="nvdisasm"):
        yield


def cute_launcher_sass(compiled: Callable[..., object]) -> str:
    """``nvdisasm -c`` of every cubin of the kernel ``compiled`` launches."""
    skip_unless_nvdisasm_runs()
    return "\n".join(
        sass_check.disassemble(cubin) for cubin in cute_launcher_cubins(compiled)
    )
