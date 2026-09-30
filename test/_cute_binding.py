from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
import importlib.util
from typing import TYPE_CHECKING
from unittest import SkipTest
from unittest.mock import patch

from helion._compiler.cute.cutedsl_compat import _cute_backend_requirement_error
from helion._utils import triton_is_available

if TYPE_CHECKING:
    from collections.abc import Iterator
    from unittest.mock import MagicMock


def _require_cute_codegen() -> None:
    # Source generation imports SDK types even when native compilation is
    # forbidden. Keep these tests runnable in Triton-only installations.
    if reason := _cute_backend_requirement_error():
        raise SkipTest(reason)


def _cpu_bind(kernel, args):
    if kernel.settings.backend == "cute":
        _require_cute_codegen()
    elif kernel.settings.backend in ("triton", "tileir") and not triton_is_available():
        raise SkipTest("Triton is required for source generation")
    with ExitStack() as stack:
        for name, value in (
            ("helion.runtime.kernel.target_device_capability", (10, 0)),
            ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
            ("helion.runtime.get_num_sm", 148),
            ("helion._compat._is_hip", False),
        ):
            stack.enter_context(patch(name, return_value=value))
        return kernel._bind_isolated(args)


@contextmanager
def _mock_cuda_unavailable() -> Iterator[None]:
    """Model CPU code generation without filling process-wide device caches."""
    with (
        patch(
            "helion._compiler.cute.backend.CuteBackend.validate_environment",
            side_effect=_require_cute_codegen,
        ),
        patch("torch.cuda.is_available", return_value=False),
        patch("helion._compat._supports_maxnreg", return_value=False),
        patch("helion._compat._supports_tensor_descriptor", return_value=False),
        patch("helion._compat._is_hip", return_value=False),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.autotuner.config_spec.num_compute_units", return_value=128),
    ):
        yield


@contextmanager
def _forbid_native_compile() -> Iterator[list[MagicMock]]:
    """Fail if CuTe DSL compilation runs; a no-op when cutlass is not installed."""
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        if importlib.util.find_spec("cutlass") is not None:
            mocks.extend(
                stack.enter_context(
                    patch(target, side_effect=AssertionError("native forbidden"))
                )
                for target in ("cutlass.cute.compile", "cutlass.compile")
            )
        yield mocks
