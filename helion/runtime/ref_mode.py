from __future__ import annotations

import threading

import torch
from torch.overrides import BaseTorchFunctionMode

_thread_local = threading.local()


def is_ref_mode_enabled() -> bool:
    """Check if ref mode is currently active."""
    return getattr(_thread_local, "ref_mode_enabled", False)


class RefModeContext:
    """Context manager to enable ref mode execution."""

    def __enter__(self) -> RefModeContext:
        self._old_value = getattr(_thread_local, "ref_mode_enabled", False)
        _thread_local.ref_mode_enabled = True
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        _thread_local.ref_mode_enabled = self._old_value
        return False


class HelionTorchFunctionMode(BaseTorchFunctionMode):
    """Torch function mode for Helion ref mode operations."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        # Replace torch.addmm with _helion_mixed_addmm
        if func == torch.addmm:
            return _helion_mixed_addmm(*args, **kwargs)
        
        # Replace torch.baddbmm with _helion_mixed_baddbmm
        if func == torch.baddbmm:
            return _helion_mixed_baddbmm(*args, **kwargs)
        
        return super().__torch_function__(func, types, args, kwargs)


def _helion_mixed_addmm(input, mat1, mat2, *, beta=1, alpha=1):
    """Mixed precision addmm that handles dtype mismatches."""
    # Ensure both matrices have the same dtype
    if mat1.dtype != mat2.dtype:
        raise ValueError(
            f"Matrix dtypes must match for torch.addmm: mat1.dtype={mat1.dtype}, mat2.dtype={mat2.dtype}"
        )

    # Use torch.mm with out_dtype to perform mixed precision computation
    # out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs
    if (
        mat1.dtype in (torch.float16, torch.bfloat16)
        and input.dtype == torch.float32
    ) or mat1.dtype == input.dtype:
        result = torch.mm(mat1, mat2, out_dtype=input.dtype)
    else:
        raise ValueError(
            f"Unsupported dtype combination for torch.addmm: input.dtype={input.dtype}, "
            f"mat1.dtype={mat1.dtype}. out_dtype must be the same as input dtype or "
            f"fp32 for fp16/bf16 inputs."
        )

    # Scale the result
    if alpha != 1:
        result = result * alpha

    # Add the bias term, converting result to bias's dtype if needed
    if result.dtype != input.dtype:
        result = result.to(input.dtype)

    if beta == 0:
        return result
    return result + (beta * input)


def _helion_mixed_baddbmm(input, batch1, batch2, *, beta=1, alpha=1):
    """Mixed precision baddbmm that handles dtype mismatches."""
    # Ensure both batch matrices have the same dtype
    if batch1.dtype != batch2.dtype:
        raise ValueError(
            f"Batch matrix dtypes must match for torch.baddbmm: batch1.dtype={batch1.dtype}, batch2.dtype={batch2.dtype}"
        )

    # Use torch.bmm with out_dtype to perform mixed precision computation
    # out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs
    if (
        batch1.dtype in (torch.float16, torch.bfloat16)
        and input.dtype == torch.float32
    ) or batch1.dtype == input.dtype:
        result = torch.bmm(batch1, batch2, out_dtype=input.dtype)
    else:
        raise ValueError(
            f"Unsupported dtype combination for torch.baddbmm: input.dtype={input.dtype}, "
            f"batch1.dtype={batch1.dtype}. out_dtype must be the same as input dtype or "
            f"fp32 for fp16/bf16 inputs."
        )

    # Scale the result
    if alpha != 1:
        result = result * alpha

    # Add the bias term, converting result to bias's dtype if needed
    if result.dtype != input.dtype:
        result = result.to(input.dtype)

    if beta == 0:
        return result
    return result + (beta * input)


