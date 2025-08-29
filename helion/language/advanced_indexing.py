"""Advanced indexing support for Helion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..exc import NotInsideKernel
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.type_propagation import TypeInfo
    from .._compiler.variable_origin import Origin

__all__ = ["advanced_index"]


@_decorators.api(allow_host_tensor=True)
def advanced_index(tensor: torch.Tensor, *indices: torch.Tensor) -> torch.Tensor:
    """
    Perform advanced indexing with integer tensor indices.
    
    This function enables PyTorch-style advanced indexing where indices
    are tensors rather than integers or slices.
    
    Args:
        tensor: The tensor to index into
        *indices: Variable number of index tensors
    
    Returns:
        torch.Tensor: The indexed result
    """
    raise NotInsideKernel


@_decorators.device_func_replacement(advanced_index)
def _device_advanced_index(tensor: torch.Tensor, *indices: torch.Tensor) -> torch.Tensor:
    """Device replacement for advanced_index that directly calls aten.index.Tensor."""
    # Convert indices to a list for aten.index.Tensor
    indices_list = list(indices)
    return torch.ops.aten.index.Tensor(tensor, indices_list)


@_decorators.register_fake(advanced_index)
def _(tensor: torch.Tensor, *indices: torch.Tensor) -> torch.Tensor:
    """Fake implementation that creates aten.index.Tensor operation."""
    # Handle the case where indices might be wrapped in a tuple
    if len(indices) == 1 and isinstance(indices[0], (list, tuple)):
        indices = indices[0]
    
    # Convert indices to a list for aten.index.Tensor
    indices_list = list(indices)
    # Use torch.ops.aten.index to create the operation in the FX graph
    # This will be handled by Inductor's fallthrough path
    return torch.ops.aten.index.Tensor(tensor, indices_list)