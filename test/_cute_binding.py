from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
import importlib.util
from typing import TYPE_CHECKING
from unittest import SkipTest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from torch.utils._pytree import tree_leaves

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


def _codegen_skip_reason(backend: str) -> str | None:
    """Report why ``backend`` cannot generate source, without raising."""
    if backend == "cute":
        return _cute_backend_requirement_error()
    if backend in ("triton", "tileir") and not triton_is_available():
        return "Triton is required for source generation"
    return None


def _cpu_bind(kernel, args, *, capability: tuple[int, int] | None = (10, 0)):
    """Bind ``kernel`` for code generation on a CPU-only process.

    ``capability`` is the compute capability the bound kernel targets; ``None``
    models an unknown target (``target_device_capability`` without CUDA).
    """
    if reason := _codegen_skip_reason(kernel.settings.backend):
        raise SkipTest(reason)
    with ExitStack() as stack:
        for name, value in (
            ("helion.runtime.kernel.target_device_capability", capability),
            (
                "helion._compiler.compile_environment.target_device_capability",
                capability,
            ),
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


@contextmanager
def _forbid_cuda_work() -> Iterator[None]:
    """Guard this test even when an earlier test initialized the CUDA context."""
    initialized = torch.cuda.is_initialized()
    with ExitStack() as stack:
        stack.enter_context(_RejectNativeCuda())
        guards = stack.enter_context(_forbid_native_compile())
        guards.extend(
            stack.enter_context(
                patch(target, side_effect=AssertionError("CUDA work forbidden"))
            )
            for target in (
                "torch.cuda._lazy_init",
                "torch.cuda.init",
                "torch.cuda.synchronize",
                "torch.cuda.current_stream",
                "torch.cuda.default_stream",
                "torch.cuda.Stream",
                "torch.cuda.Event",
                "torch.cuda.CUDAGraph",
            )
        )
        yield
        for guard in guards:
            guard.assert_not_called()
        assert torch.cuda.is_initialized() == initialized


class _RejectNativeCuda(TorchDispatchMode):
    """Stop direct ATen dispatch before it can reach a live CUDA context."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        inputs = tree_leaves((args, kwargs))
        fake_modes = [
            mode
            for mode in _get_current_dispatch_mode_stack()
            if isinstance(mode, FakeTensorMode)
        ]
        for value in inputs:
            if isinstance(value, FakeTensor) and value.fake_mode not in fake_modes:
                fake_modes.append(value.fake_mode)
        for value in inputs:
            if (
                isinstance(value, torch.device)
                and value.type == "cuda"
                and not fake_modes
            ):
                raise AssertionError("CUDA device forbidden by dispatch guard")
            if (
                isinstance(value, torch.Tensor)
                and not isinstance(value, FakeTensor)
                and value.device.type == "cuda"
            ):
                raise AssertionError("CUDA tensor forbidden by dispatch guard")
        # Fake metadata operations need an installed mode or a typed fake input;
        # native fallback must not escape while this dispatch guard is popped.
        with ExitStack() as stack:
            for mode in fake_modes:
                stack.enter_context(patch.object(mode, "allow_fallback_kernels", False))
            result = func(*args, **kwargs)
        for value in tree_leaves(result):
            if (
                isinstance(value, torch.Tensor)
                and not isinstance(value, FakeTensor)
                and value.device.type == "cuda"
            ):
                raise AssertionError("CUDA result forbidden by dispatch guard")
        return result
