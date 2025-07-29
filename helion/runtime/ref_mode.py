from __future__ import annotations

import enum
import os
import threading
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.overrides import BaseTorchFunctionMode

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import tls as ce_tls

if TYPE_CHECKING:
    from typing_extensions import Self

    from .settings import Settings

_thread_local = threading.local()


class RefMode(enum.Enum):
    """Reference mode for kernel execution."""

    OFF = "off"
    EAGER = "eager"


def _get_ref_mode_from_env() -> RefMode:
    """Get reference mode from environment variables."""
    # Check for environment variables
    ref_eager = os.environ.get("HELION_REF_EAGER", "").lower() in ("1", "true", "yes")

    if ref_eager:
        return RefMode.EAGER
    return RefMode.OFF


def is_in_ref_mode_context() -> bool:
    """Check if ref mode is currently active."""
    return getattr(_thread_local, "in_ref_mode_context", False)


def is_ref_mode_enabled(settings: Settings) -> bool:
    return (
        is_in_ref_mode_context()
        or _get_ref_mode_from_env() != RefMode.OFF
        or settings.ref_mode != RefMode.OFF
    )


class RefModeContext:
    """Context manager to enable ref mode execution."""

    def __init__(self, env: CompileEnvironment) -> None:
        self.env = env
        self.func_mode = RefModeTorchFunctionMode()
        self._old_value: bool | None = None

    def __enter__(self) -> Self:
        self._old_value = getattr(_thread_local, "in_ref_mode_context", False)
        _thread_local.in_ref_mode_context = True
        ce_tls.env = self.env
        self.func_mode.__enter__()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.func_mode.__exit__(exc_type, exc_val, exc_tb)
        ce_tls.env = None
        _thread_local.in_ref_mode_context = self._old_value
        return False


class RefModeTorchFunctionMode(BaseTorchFunctionMode):
    """Torch function mode for Helion ref mode operations."""

    def __torch_function__(
        self,
        func: object,
        types: list[type[object]],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if kwargs is None:
            kwargs = {}

        # Handle matrix multiplication operations
        if func == torch.addmm:
            return self._handle_addmm(args, kwargs)
        if func == torch.baddbmm:
            return self._handle_baddbmm(args, kwargs)

        return super().__torch_function__(func, types, args, kwargs)

    def _handle_addmm(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.addmm with mixed precision support (e.g. torch.addmm(fp32, bf16, bf16))."""
        assert len(args) >= 3, "addmm requires at least 3 arguments"
        bias = cast("torch.Tensor", args[0])
        mat1 = cast("torch.Tensor", args[1])
        mat2 = cast("torch.Tensor", args[2])
        beta = cast("float", kwargs.get("beta", 1))
        alpha = cast("float", kwargs.get("alpha", 1))

        assert mat1.dtype == mat2.dtype, (
            f"Matrix dtypes must match for torch.addmm: "
            f"mat1.dtype={mat1.dtype}, mat2.dtype={mat2.dtype}"
        )

        result = torch.mm(mat1, mat2, out_dtype=bias.dtype)
        if alpha != 1:
            result = result * alpha
        if result.dtype != bias.dtype:
            result = result.to(bias.dtype)
        if beta == 0:
            return result
        return result + (beta * bias)

    def _handle_baddbmm(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.baddbmm with mixed precision support (e.g. torch.baddbmm(fp32, bf16, bf16))."""
        assert len(args) >= 3, "baddbmm requires at least 3 arguments"
        bias = cast("torch.Tensor", args[0])
        batch1 = cast("torch.Tensor", args[1])
        batch2 = cast("torch.Tensor", args[2])
        beta = cast("float", kwargs.get("beta", 1))
        alpha = cast("float", kwargs.get("alpha", 1))

        assert batch1.dtype == batch2.dtype, (
            f"Matrix dtypes must match for torch.baddbmm: "
            f"mat1.dtype={batch1.dtype}, mat2.dtype={batch2.dtype}"
        )

        result = torch.bmm(batch1, batch2, out_dtype=bias.dtype)
        if alpha != 1:
            result = result * alpha
        if result.dtype != bias.dtype:
            result = result.to(bias.dtype)
        if beta == 0:
            return result
        return result + (beta * bias)
