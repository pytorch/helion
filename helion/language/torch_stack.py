"""
Replacement for torch.stack operation in Helion kernels.

This module provides a custom implementation of torch.stack that uses
hl.arange, broadcasting, and torch.where to create the stacked tensor
in a way that's compatible with Triton's functional programming model.
"""

from __future__ import annotations

import torch
from . import _decorators
from . import arange

__all__ = []  # Don't export anything, just register the replacement


@_decorators.device_func_replacement(torch.stack)
def torch_stack(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Stack tensors along a new dimension using broadcasting and where.
    
    This is a replacement for torch.stack that works in Helion kernels.
    It creates the stacked tensor using hl.arange for indexing and torch.where
    for selection, avoiding the slice assignment issues in Triton.
    
    Args:
        tensors: List of tensors to stack (currently only supports 2 tensors)
        dim: Dimension along which to stack
    
    Returns:
        Stacked tensor
    """
    # Only support 2 tensors for now
    if len(tensors) != 2:
        raise NotImplementedError("torch_stack currently only supports 2 tensors")
    
    tensor0, tensor1 = tensors
    
    # Create index for the new dimension
    mid_idx = arange(0, 2)
    
    # Add unsqueeze operations to match the output shape
    for i in range(dim):
        mid_idx = mid_idx.unsqueeze(0)
    
    for i in range(tensor0.ndim - dim):
        mid_idx = mid_idx.unsqueeze(i + dim + 1)
    
    # Unsqueeze the input tensors at the stack dimension
    tensor0_expanded = tensor0.unsqueeze(dim)
    tensor1_expanded = tensor1.unsqueeze(dim)
    
    # Expand to the full output shape
    # The output shape is the input shape with size 2 inserted at dim
    output_shape = list(tensor0.shape)
    output_shape.insert(dim, 2)
    
    tensor0_broadcast = tensor0_expanded.expand(*output_shape)
    tensor1_broadcast = tensor1_expanded.expand(*output_shape)
    
    # Use where to select values based on the index
    result = torch.where(mid_idx == 0, tensor0_broadcast, tensor1_broadcast)
    
    return result