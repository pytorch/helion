"""SliceProxy implementation for handling symbolic slices in Helion."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

import torch

from .. import exc
from .._compiler import type_propagation as tp
from .._compiler.compile_environment import CompileEnvironment
from . import _decorators

if TYPE_CHECKING:
    from collections.abc import Callable


def has_symbolic_bounds(lower: object, upper: object, step: object) -> bool:
    """Check if any slice bound is symbolic (SymInt)."""
    return isinstance(lower, torch.SymInt) or isinstance(upper, torch.SymInt) or isinstance(step, torch.SymInt)


class SliceProxy(torch.Tensor):
    """A tensor subclass that represents a slice with potentially symbolic bounds.

    Like Tile, SliceProxy stores only an ID. The actual slice bounds are stored
    in CompileEnvironment to avoid FX tracing issues with symbolic values.
    """

    slice_id: int

    @staticmethod
    def __new__(cls, slice_id: int):
        # Create a meta tensor similar to how Tile does it
        return torch.Tensor._make_wrapper_subclass(
            cls, size=(), dtype=torch.int64, device="meta", requires_grad=False
        )

    def __init__(self, slice_id: int):
        self.slice_id = slice_id

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        # Handle attribute access for tensor properties
        if getattr(func, "__name__", None) == "__get__":
            with torch._C.DisableTorchFunction():
                return func(args[0])
        
        from ..language.memory_ops import load, store

        # Handle indexing operations
        if func is torch.Tensor.__getitem__:
            if len(args) != 2 or kwargs:
                raise exc.IncorrectTileUsage(f"SliceProxy cannot be used with {func}")
            return load(args[0], cls._prepare_index(args[1]))
        
        if func is torch.Tensor.__setitem__:
            if len(args) != 3 or kwargs:
                raise exc.IncorrectTileUsage(f"SliceProxy cannot be used with {func}")
            return store(args[0], cls._prepare_index(args[1]), args[2])

        # Handle __index__ for tracking
        if func is torch.Tensor.__index__:
            if index_calls := getattr(_decorators._tls, "index_calls", None):
                index_calls.count += 1

        # Allow formatting
        if func is torch.Tensor.__format__:
            return repr(args[0])
        
        raise exc.IncorrectTileUsage(f"SliceProxy cannot be used with {func}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Handle torch dispatch for SliceProxy objects."""
        # SliceProxy shouldn't participate in any tensor operations
        # It's only meant to be used as an index
        return NotImplemented

    def __repr__(self):
        return f"SliceProxy({self.slice_id!r})"

    def to_slice(self) -> builtins.slice:
        """Convert to regular slice using stored bounds."""
        try:
            env = CompileEnvironment.current()
            bounds = env.slice_bounds[self.slice_id]
            return builtins.slice(bounds.start, bounds.stop, bounds.step)
        except:
            return builtins.slice(None, None, None)

    @staticmethod
    def _prepare_index(index: object) -> list[object]:
        """Prepare index for load/store operations."""
        return [*index] if isinstance(index, (list, tuple)) else [index]


def _register_slice_bounds(env: CompileEnvironment, start, stop, step) -> SliceProxy:
    """Common helper to register slice and create proxy."""
    slice_id = env.register_slice(start, stop, step)
    return SliceProxy(slice_id)


@_decorators.api(is_device_only=False)
def slice(
    start: int | torch.SymInt | None = None,
    stop: int | torch.SymInt | None = None,
    step: int | torch.SymInt | None = None,
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
        >>> # Can use regular Python slice notation - will be auto-converted
        >>> tensor[0:block_size, :] = data  # Use slice with symbolic bound
        >>> # Or explicitly create a SliceProxy if needed
        >>> slice0 = hl.slice(0, block_size)
        >>> tensor[slice0, :] = data
    """
    return _register_slice_bounds(CompileEnvironment.current(), start, stop, step)


@_decorators.type_propagation(slice)
def _slice_type_prop(
    start: tp.TypeInfo = None,
    stop: tp.TypeInfo = None,
    step: tp.TypeInfo = None,
    *,
    origin: tp.Origin,
) -> tp.SliceProxyType:
    # Store bounds for later registration during device IR
    return tp.SliceProxyType(origin, -1, start, stop, step)


@_decorators.ref(slice)
def _slice_ref(start=None, stop=None, step=None):
    return builtins.slice(start, stop, step)


@_decorators.register_fake(slice)
def _slice_fake(start=None, stop=None, step=None):
    return _register_slice_bounds(CompileEnvironment.current(), start, stop, step)


@_decorators.codegen(slice)
def _slice_codegen(state):
    # No code generation needed - slice is just a data structure
    return None


@_decorators.register_to_device_ir(slice)
def _slice_to_device_ir(tracer, start=None, stop=None, step=None):
    proxy_args, _ = _decorators.args_to_proxies(tracer, (start, stop, step), {})
    proxy_out = tracer.create_proxy("call_function", slice, proxy_args, {})

    slice_proxy = _register_slice_bounds(
        CompileEnvironment.current(), start, stop, step
    )

    proxy_out.node.meta["val"] = slice_proxy
    tracer.tensor_tracker[slice_proxy] = proxy_out
    return slice_proxy
