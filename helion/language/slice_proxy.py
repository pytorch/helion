"""SliceProxy implementation for handling symbolic slices in Helion."""

import torch
from typing import Optional, Union, Any, Mapping, Tuple
from torch.utils._pytree import tree_map

from . import _decorators
from .._compiler import type_propagation as tp


class SliceProxy(torch.Tensor):
    """A tensor subclass that represents a slice with potentially symbolic bounds."""
    
    _start: Optional[Union[int, torch.SymInt]]
    _stop: Optional[Union[int, torch.SymInt]]
    _step: Optional[Union[int, torch.SymInt]]
    
    @staticmethod
    def __new__(
        cls,
        start: Optional[Union[int, torch.SymInt]] = None,
        stop: Optional[Union[int, torch.SymInt]] = None,
        step: Optional[Union[int, torch.SymInt]] = None,
    ):
        # Create a meta tensor similar to how Tile does it
        return torch.Tensor._make_wrapper_subclass(
            cls, size=(), dtype=torch.int64, device="meta", requires_grad=False
        )
    
    def __init__(
        self,
        start: Optional[Union[int, torch.SymInt]] = None,
        stop: Optional[Union[int, torch.SymInt]] = None,
        step: Optional[Union[int, torch.SymInt]] = None,
    ):
        self._start = start
        self._stop = stop
        self._step = step
    
    @property
    def start(self):
        return self._start
    
    @property
    def stop(self):
        return self._stop
    
    @property
    def step(self):
        return self._step
    
    def to_slice(self) -> slice:
        """Convert to a regular Python slice object."""
        return slice(self._start, self._stop, self._step)
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Handle torch operations on SliceProxy objects."""
        if kwargs is None:
            kwargs = {}
        
        # For now, we don't expect SliceProxy to participate in tensor operations
        # It should only be used as an index
        return NotImplemented
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Handle torch dispatch for SliceProxy objects."""
        # SliceProxy shouldn't participate in any tensor operations
        # It's only meant to be used as an index
        return NotImplemented
    
    def __repr__(self):
        return f"SliceProxy(start={self._start}, stop={self._stop}, step={self._step})"


@_decorators.api(is_device_only=False)
def make_slice(
    start: Optional[Union[int, torch.SymInt]] = None,
    stop: Optional[Union[int, torch.SymInt]] = None,
    step: Optional[Union[int, torch.SymInt]] = None,
) -> SliceProxy:
    """Create a slice with potentially symbolic bounds.
    
    This function creates a SliceProxy object that can be used in tensor subscripts
    with symbolic values. It works similarly to Python's built-in slice() but supports
    SymInt values from register_block_size().
    
    Args:
        start: The starting index (inclusive), can be None, int, or SymInt
        stop: The stopping index (exclusive), can be None, int, or SymInt  
        step: The step size, can be None, int, or SymInt
        
    Returns:
        A SliceProxy object that can be used in tensor subscripts
        
    Example:
        >>> block_size = hl.register_block_size(128)
        >>> slice0 = hl.make_slice(0, block_size)
        >>> tensor[slice0, :] = data  # Use slice with symbolic bound
    """
    return SliceProxy(start, stop, step)


# Type propagation for make_slice API
@_decorators.type_propagation(make_slice)
def _make_slice_type_prop(
    start: tp.TypeInfo = None, stop: tp.TypeInfo = None, step: tp.TypeInfo = None, *, origin: tp.Origin
) -> tp.SliceProxyType:
    """Type propagation for make_slice() API."""
    # Use the provided origin instead of combining from arguments
    return tp.SliceProxyType(origin, start, stop, step)


# Reference implementation for make_slice API
@_decorators.ref(make_slice)
def _make_slice_ref(start=None, stop=None, step=None):
    """Reference implementation for make_slice() API."""
    return SliceProxy(start, stop, step)


# Fake implementation for make_slice API
@_decorators.register_fake(make_slice)
def _make_slice_fake(start=None, stop=None, step=None):
    """Fake implementation for make_slice() API."""
    return SliceProxy(start, stop, step)