from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.overrides import BaseTorchFunctionMode

if TYPE_CHECKING:
    from typing_extensions import Self

_thread_local = threading.local()


def is_ref_mode_enabled() -> bool:
    """Check if ref mode is currently active."""
    return getattr(_thread_local, "ref_mode_enabled", False)


class RefModeContext:
    """Context manager to enable ref mode execution."""

    def __enter__(self) -> Self:
        self._old_value = getattr(_thread_local, "ref_mode_enabled", False)
        _thread_local.ref_mode_enabled = True
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        _thread_local.ref_mode_enabled = self._old_value
        return False


class HelionTorchFunctionMode(BaseTorchFunctionMode):
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

        # Replace torch.addmm with _helion_mixed_addmm
        if func == torch.addmm:
            # Cast args to expected types
            assert len(args) >= 3, "addmm requires at least 3 arguments"
            bias = cast("torch.Tensor", args[0])
            mat1 = cast("torch.Tensor", args[1])
            mat2 = cast("torch.Tensor", args[2])
            return _helion_mixed_addmm(bias, mat1, mat2, *args[3:], **kwargs)

        # Replace torch.baddbmm with _helion_mixed_baddbmm
        if func == torch.baddbmm:
            # Cast args to expected types
            assert len(args) >= 3, "baddbmm requires at least 3 arguments"
            bias = cast("torch.Tensor", args[0])
            batch1 = cast("torch.Tensor", args[1])
            batch2 = cast("torch.Tensor", args[2])
            return _helion_mixed_baddbmm(bias, batch1, batch2, *args[3:], **kwargs)

        return super().__torch_function__(func, types, args, kwargs)


def _helion_mixed_addmm(
    bias: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float = 1,
    alpha: float = 1,
) -> torch.Tensor:
    """Mixed precision addmm that handles dtype mismatches."""
    # Ensure both matrices have the same dtype
    if mat1.dtype != mat2.dtype:
        raise ValueError(
            f"Matrix dtypes must match for torch.addmm: mat1.dtype={mat1.dtype}, mat2.dtype={mat2.dtype}"
        )

    # Use torch.mm with out_dtype to perform mixed precision computation
    # out_dtype must be the same as bias dtype or fp32 for fp16/bf16 inputs
    if (
        mat1.dtype in (torch.float16, torch.bfloat16) and bias.dtype == torch.float32
    ) or mat1.dtype == bias.dtype:
        result = torch.mm(mat1, mat2, out_dtype=bias.dtype)
    else:
        raise ValueError(
            f"Unsupported dtype combination for torch.addmm: bias.dtype={bias.dtype}, "
            f"mat1.dtype={mat1.dtype}. out_dtype must be the same as bias dtype or "
            f"fp32 for fp16/bf16 inputs."
        )

    # Scale the result
    if alpha != 1:
        result = result * alpha

    # Add the bias term, converting result to bias's dtype if needed
    if result.dtype != bias.dtype:
        result = result.to(bias.dtype)

    if beta == 0:
        return result
    return result + (beta * bias)


def _helion_mixed_baddbmm(
    bias: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    *,
    beta: float = 1,
    alpha: float = 1,
) -> torch.Tensor:
    """Mixed precision baddbmm that handles dtype mismatches."""
    # Ensure both batch matrices have the same dtype
    if batch1.dtype != batch2.dtype:
        raise ValueError(
            f"Batch matrix dtypes must match for torch.baddbmm: batch1.dtype={batch1.dtype}, batch2.dtype={batch2.dtype}"
        )

    # Use torch.bmm with out_dtype to perform mixed precision computation
    # out_dtype must be the same as bias dtype or fp32 for fp16/bf16 inputs
    if (
        batch1.dtype in (torch.float16, torch.bfloat16) and bias.dtype == torch.float32
    ) or batch1.dtype == bias.dtype:
        result = torch.bmm(batch1, batch2, out_dtype=bias.dtype)
    else:
        raise ValueError(
            f"Unsupported dtype combination for torch.baddbmm: bias.dtype={bias.dtype}, "
            f"batch1.dtype={batch1.dtype}. out_dtype must be the same as bias dtype or "
            f"fp32 for fp16/bf16 inputs."
        )

    # Scale the result
    if alpha != 1:
        result = result * alpha

    # Add the bias term, converting result to bias's dtype if needed
    if result.dtype != bias.dtype:
        result = result.to(bias.dtype)

    if beta == 0:
        return result
    return result + (beta * bias)
