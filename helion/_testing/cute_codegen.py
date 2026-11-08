"""CPU-only CuTe code generation helpers shared by the test suite."""

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

from .._compiler.cute.cutedsl_compat import _cute_backend_requirement_error
from .._utils import triton_is_available
from . import patch_cute_mma_support

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from unittest.mock import MagicMock

    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import Kernel


def require_cute_codegen() -> None:
    # Source generation imports SDK types even when native compilation is
    # forbidden. Keep these tests runnable in Triton-only installations.
    if reason := _cute_backend_requirement_error():
        raise SkipTest(reason)


def codegen_skip_reason(backend: str) -> str | None:
    """Report why ``backend`` cannot generate source, without raising."""
    if backend == "cute":
        return _cute_backend_requirement_error()
    if backend in ("triton", "tileir") and not triton_is_available():
        return "Triton is required for source generation"
    return None


def cpu_bind(
    kernel: Kernel[object],
    args: tuple[object, ...],
    *,
    capability: tuple[int, int] | None = (10, 0),
) -> BoundKernel[object]:
    """Bind ``kernel`` for code generation on a CPU-only process.

    ``capability`` is the compute capability the bound kernel targets; ``None``
    models an unknown target (``target_device_capability`` without CUDA).
    """
    if reason := codegen_skip_reason(kernel.settings.backend):
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
def mock_cuda_unavailable() -> Generator[None, None, None]:
    """Model CPU code generation without filling process-wide device caches."""
    with (
        patch(
            "helion._compiler.cute.backend.CuteBackend.validate_environment",
            side_effect=require_cute_codegen,
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
def forbid_native_compile() -> Generator[list[MagicMock], None, None]:
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
def forbid_cuda_work() -> Generator[None, None, None]:
    """Guard this test even when an earlier test initialized the CUDA context."""
    initialized = torch.cuda.is_initialized()
    with ExitStack() as stack:
        stack.enter_context(RejectNativeCuda())
        guards = stack.enter_context(forbid_native_compile())
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


class RejectNativeCuda(TorchDispatchMode):
    """Stop direct ATen dispatch before it can reach a live CUDA context."""

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
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


@contextmanager
def cpu_target() -> Generator[None, None, None]:
    """Generate tcgen05 CuTe source for a 148-SM sm100 target without CUDA.

    Extends ``mock_cuda_unavailable`` with the compute capability, SM count and
    tcgen05 shared-memory budget the schedule heuristics read, and fails the
    test if anything initializes CUDA.
    """
    with ExitStack() as stack:
        stack.enter_context(patch_cute_mma_support())
        stack.enter_context(mock_cuda_unavailable())
        stack.enter_context(
            patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU test"))
        )
        for name, value in (
            ("helion.runtime.kernel.target_device_capability", (10, 0)),
            ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
            ("helion.runtime.get_num_sm", 148),
            ("helion._compat._is_hip", False),
            (
                (
                    "helion._compiler.cute.tcgen05_config."
                    "CuteTcgen05Config.per_cta_smem_budget_bytes"
                ),
                232448,
            ),
        ):
            stack.enter_context(patch(name, return_value=value))
        yield


def cuda_fake(backend: object, tensor: torch.Tensor) -> torch.Tensor:
    """Stand-in for ``CuteBackend.normalize_input_fake_tensor`` on a CPU process.

    Re-homes a CPU fake tensor on ``cuda:0`` so CuTe code generation sees device
    inputs without a CUDA context.
    """
    assert isinstance(tensor, FakeTensor)
    with torch._C._DisableTorchDispatch():
        meta = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device=torch.device("meta"),
            requires_grad=tensor.requires_grad,
        )
    return FakeTensor(tensor.fake_mode, meta, torch.device("cuda:0"))


def full_and_edge_store_bodies(code: str) -> tuple[str, str]:
    """The full-tile and edge-tile bodies of a hybrid tcgen05 store: the
    ``if tcgen05_full_tile:`` branch and its ``else:`` branch, at the
    indentation the generated code gives them."""
    lines = code.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.strip() == "if tcgen05_full_tile:"
    )
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == f"{indent}else:")
    stop = next(
        (
            i
            for i in range(end + 1, len(lines))
            if lines[i].strip()
            and len(lines[i]) - len(lines[i].lstrip()) <= len(indent)
        ),
        len(lines),
    )
    return "\n".join(lines[start + 1 : end]), "\n".join(lines[end + 1 : stop])
