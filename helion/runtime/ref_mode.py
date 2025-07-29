from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch.overrides import BaseTorchFunctionMode
from ..language.ref_tile import RefTile

if TYPE_CHECKING:
    from typing_extensions import Self

_thread_local = threading.local()


def is_ref_mode_enabled() -> bool:
    """Check if ref mode is currently active."""
    return getattr(_thread_local, "ref_mode_enabled", False)


def get_ref_mode_config() -> dict[str, object] | None:
    """Get the current ref mode configuration."""
    return getattr(_thread_local, "ref_mode_config", None)


def set_ref_mode_config(config: dict[str, object] | None) -> None:
    """Set the ref mode configuration."""
    _thread_local.ref_mode_config = config


class RefModeContext:
    """Context manager to enable ref mode execution."""

    def __init__(self, config: dict[str, object] | None = None) -> None:
        self.config = config
        self._old_value: bool | None = None
        self._old_config: dict[str, object] | None = None

    def __enter__(self) -> Self:
        self._old_value = getattr(_thread_local, "ref_mode_enabled", False)
        self._old_config = getattr(_thread_local, "ref_mode_config", None)
        _thread_local.ref_mode_enabled = True
        _thread_local.ref_mode_config = self.config
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        _thread_local.ref_mode_enabled = self._old_value
        _thread_local.ref_mode_config = self._old_config
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

        # Handle tensor indexing operations
        if func == torch.Tensor.__getitem__:
            result = self._handle_tensor_getitem(args)
            if result is not None:
                return result

        # Handle matrix multiplication operations
        if func == torch.addmm:
            return self._handle_addmm(args, kwargs)
        if func == torch.baddbmm:
            return self._handle_baddbmm(args, kwargs)

        # Handle tensor reshaping operations
        result = self._handle_tensor_reshaping(func, args)
        if result is not None:
            return result

        return super().__torch_function__(func, types, args, kwargs)

    def _handle_tensor_getitem(
        self, args: tuple[object, ...]
    ) -> torch.Tensor | None:
        """Handle tensor indexing with special cases for RefTile objects."""
        if len(args) != 2:
            return None

        tensor = cast("torch.Tensor", args[0])
        index = args[1]

        # Handle list of RefTile objects
        if isinstance(index, list) and all(isinstance(idx, RefTile) for idx in index):
            # Convert RefTile indices to slices
            slices = [idx._slice for idx in index]
            
            # Build the final index, handling singleton dimensions
            final_index = []
            for i in range(min(len(slices), tensor.ndim)):
                # For singleton dimensions (size=1), use full slice to preserve dimension
                if tensor.shape[i] == 1:
                    final_index.append(slice(None))
                else:
                    final_index.append(slices[i])
            
            # If we have fewer indices than dimensions, slice remaining dimensions fully
            if len(slices) < tensor.ndim:
                final_index.extend([slice(None)] * (tensor.ndim - len(slices)))
            
            return tensor[tuple(final_index)]

        # Handle tuple of tensors for broadcasting
        if isinstance(index, tuple) and len(index) >= 2:
            return self._handle_tensor_tuple_indexing(tensor, index)

        return None

    def _handle_tensor_tuple_indexing(
        self, tensor: torch.Tensor, index: tuple[Any, ...]
    ) -> torch.Tensor | None:
        """Handle tensor indexing with tuple of tensors."""
        # Check that all indices are tensors (not RefTile objects)
        if not all(
            isinstance(idx, torch.Tensor) and not isinstance(idx, RefTile)
            for idx in index
        ):
            return None

        # Special case for 2D indexing with 1D tensors (original behavior)
        if len(index) == 2 and all(idx.ndim == 1 for idx in index):
            return tensor[index[0][:, None], index[1]]

        # General case: let PyTorch handle the advanced indexing
        # This will work for any number of tensor indices with proper broadcasting
        try:
            return tensor[index]
        except (IndexError, RuntimeError):
            # If PyTorch can't handle it, return None to fall back to default behavior
            return None

    def _handle_addmm(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.addmm with mixed precision support."""
        assert len(args) >= 3, "addmm requires at least 3 arguments"
        bias = cast("torch.Tensor", args[0])
        mat1 = cast("torch.Tensor", args[1])
        mat2 = cast("torch.Tensor", args[2])
        return _helion_mixed_addmm(bias, mat1, mat2, *args[3:], **kwargs)

    def _handle_baddbmm(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.baddbmm with mixed precision support."""
        assert len(args) >= 3, "baddbmm requires at least 3 arguments"
        bias = cast("torch.Tensor", args[0])
        batch1 = cast("torch.Tensor", args[1])
        batch2 = cast("torch.Tensor", args[2])
        return _helion_mixed_baddbmm(bias, batch1, batch2, *args[3:], **kwargs)

    def _handle_tensor_reshaping(
        self, func: object, args: tuple[object, ...]
    ) -> torch.Tensor | None:
        """Handle tensor reshaping operations (expand, view, reshape)."""
        # Handle tensor methods
        if func in (torch.Tensor.expand, torch.Tensor.view, torch.Tensor.reshape):
            return self._handle_tensor_reshape_with_tiles(func, args)

        # Handle torch.reshape function
        if func == torch.reshape:
            return self._handle_torch_reshape_with_tiles(args)

        return None

    def _handle_tensor_reshape_with_tiles(
        self, func: object, args: tuple[object, ...]
    ) -> torch.Tensor | None:
        """Handle tensor methods (expand, view, reshape) with Tile arguments."""
        if len(args) < 2:
            return None

        # Check if any arguments are RefTile objects
        has_tiles = any(isinstance(arg, RefTile) for arg in args[1:])
        if not has_tiles:
            return None

        tensor = cast("torch.Tensor", args[0])
        new_sizes = self._convert_tiles_to_sizes(args[1:])

        if func == torch.Tensor.expand:
            return tensor.expand(*new_sizes)
        if func == torch.Tensor.view:
            return tensor.view(*new_sizes)
        if func == torch.Tensor.reshape:
            return tensor.reshape(*new_sizes)

        return None

    def _handle_torch_reshape_with_tiles(
        self, args: tuple[object, ...]
    ) -> torch.Tensor | None:
        """Handle torch.reshape function with Tile arguments."""
        if len(args) < 2:
            return None

        tensor = cast("torch.Tensor", args[0])
        shape = args[1]

        if not isinstance(shape, (list, tuple)):
            return None

        has_tiles = any(isinstance(s, RefTile) for s in shape)
        if not has_tiles:
            return None

        new_shape = self._convert_tiles_to_sizes(shape)
        return torch.reshape(tensor, new_shape)

    def _convert_tiles_to_sizes(
        self, args: tuple[object, ...] | list[object]
    ) -> list[int]:
        """Convert RefTile objects to their block sizes."""
        new_sizes = []
        for arg in args:
            if isinstance(arg, RefTile):
                new_sizes.append(arg.block_size)
            elif isinstance(arg, int):
                new_sizes.append(arg)
            else:
                # For other numeric types, try to convert to int
                new_sizes.append(int(arg))  # type: ignore[arg-type]
        return new_sizes


def _helion_mixed_addmm(
    bias: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float = 1,
    alpha: float = 1,
) -> torch.Tensor:
    """Mixed precision addmm that handles dtype mismatches.

    Args:
        bias: Bias tensor
        mat1: First matrix for multiplication
        mat2: Second matrix for multiplication
        beta: Scaling factor for bias
        alpha: Scaling factor for matrix multiplication result

    Returns:
        Result of beta * bias + alpha * (mat1 @ mat2)
    """
    _validate_matrix_dtypes(mat1, mat2, "torch.addmm")

    result = _compute_mixed_precision_mm(mat1, mat2, bias.dtype)
    return _apply_scaling_and_bias(result, bias, alpha, beta)


def _helion_mixed_baddbmm(
    bias: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    *,
    beta: float = 1,
    alpha: float = 1,
) -> torch.Tensor:
    """Mixed precision baddbmm that handles dtype mismatches.

    Args:
        bias: Bias tensor
        batch1: First batch of matrices
        batch2: Second batch of matrices
        beta: Scaling factor for bias
        alpha: Scaling factor for matrix multiplication result

    Returns:
        Result of beta * bias + alpha * (batch1 @ batch2)
    """
    _validate_matrix_dtypes(batch1, batch2, "torch.baddbmm")

    result = _compute_mixed_precision_bmm(batch1, batch2, bias.dtype)
    return _apply_scaling_and_bias(result, bias, alpha, beta)


def _validate_matrix_dtypes(
    mat1: torch.Tensor, mat2: torch.Tensor, op_name: str
) -> None:
    """Validate that matrix dtypes match."""
    if mat1.dtype != mat2.dtype:
        raise ValueError(
            f"Matrix dtypes must match for {op_name}: "
            f"mat1.dtype={mat1.dtype}, mat2.dtype={mat2.dtype}"
        )


def _compute_mixed_precision_mm(
    mat1: torch.Tensor, mat2: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Compute matrix multiplication with mixed precision support."""
    if _is_valid_mixed_precision_dtype(mat1.dtype, target_dtype):
        return torch.mm(mat1, mat2, out_dtype=target_dtype)
    raise ValueError(
        f"Unsupported dtype combination: mat.dtype={mat1.dtype}, "
        f"target.dtype={target_dtype}. out_dtype must be the same as target dtype "
        f"or fp32 for fp16/bf16 inputs."
    )


def _compute_mixed_precision_bmm(
    batch1: torch.Tensor, batch2: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Compute batch matrix multiplication with mixed precision support."""
    if _is_valid_mixed_precision_dtype(batch1.dtype, target_dtype):
        return torch.bmm(batch1, batch2, out_dtype=target_dtype)
    raise ValueError(
        f"Unsupported dtype combination: batch.dtype={batch1.dtype}, "
        f"target.dtype={target_dtype}. out_dtype must be the same as target dtype "
        f"or fp32 for fp16/bf16 inputs."
    )


def _is_valid_mixed_precision_dtype(
    input_dtype: torch.dtype, target_dtype: torch.dtype
) -> bool:
    """Check if the dtype combination is valid for mixed precision operations."""
    # Valid if input is fp16/bf16 and target is fp32
    if input_dtype in (torch.float16, torch.bfloat16) and target_dtype == torch.float32:
        return True
    # Valid if input and target dtypes match
    return input_dtype == target_dtype


def _apply_scaling_and_bias(
    result: torch.Tensor, bias: torch.Tensor, alpha: float, beta: float
) -> torch.Tensor:
    """Apply scaling factor and add bias term."""
    # Scale the result
    if alpha != 1:
        result = result * alpha

    # Convert to bias dtype if needed
    if result.dtype != bias.dtype:
        result = result.to(bias.dtype)

    # Add bias term
    if beta == 0:
        return result
    return result + (beta * bias)
