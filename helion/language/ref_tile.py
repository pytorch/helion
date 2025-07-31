from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import exc
from .tile_interface import TileInterface

if TYPE_CHECKING:
    from collections.abc import Callable


class RefTile(TileInterface, torch.Tensor):
    _slice: slice
    _block_size: int

    def __init__(self, begin: int, end: int, block_size: int) -> None:
        super().__init__()

        from ..runtime.ref_mode import is_in_ref_mode_context

        assert is_in_ref_mode_context()
        self._slice = slice(begin, end, None)
        self._block_size = block_size

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func is torch.Tensor.__getitem__:
            return cls._handle_getitem(func, args, kwargs)

        if func is torch.Tensor.__setitem__:
            return cls._handle_setitem(func, args, kwargs)

        if func is torch.Tensor.__format__:
            return repr(args[0])

        if func is torch.Tensor.expand:
            return cls._handle_expand(func, args, kwargs)
        
        if func is torch.Tensor.view:
            return cls._handle_view(func, args, kwargs)
        
        if func is torch.reshape:
            return cls._handle_reshape(func, args, kwargs)
        
        if func is torch.zeros:
            return cls._handle_zeros(func, args, kwargs)
        
        if func is torch.ones:
            return cls._handle_ones(func, args, kwargs)
        
        if func is torch.full:
            return cls._handle_full(func, args, kwargs)

        raise exc.IncorrectTileUsage(func)

    @classmethod
    def _handle_getitem(
        cls,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object] | None,
    ) -> object:
        """Handle tensor[index] operations."""
        tensor, index = args
        assert isinstance(tensor, torch.Tensor)

        if isinstance(index, RefTile):
            return tensor[index._slice]

        if isinstance(index, tuple):
            new_index = cls._convert_tile_indices_to_slices(index)
            return tensor[tuple(new_index)]  # pyright: ignore[reportArgumentType]

        # Non-tile index in ref mode
        return tensor[index]  # pyright: ignore[reportArgumentType]

    @classmethod
    def _handle_setitem(
        cls,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object] | None,
    ) -> object:
        """Handle tensor[index] = value operations."""
        tensor, index, value = args
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(value, (int, float, bool, torch.Tensor))

        if isinstance(index, RefTile):
            tensor[index._slice] = value
            return None

        if isinstance(index, tuple):
            new_index = cls._convert_tile_indices_to_slices(index)
            tensor[tuple(new_index)] = value  # pyright: ignore[reportArgumentType]
            return None

        # Non-tile index in ref mode
        tensor[index] = value  # pyright: ignore[reportArgumentType]
        return None

    @classmethod
    def _convert_tile_indices_to_slices(
        cls, indices: tuple[object, ...]
    ) -> list[object]:
        """Convert RefTile objects in a tuple of indices to slices."""
        new_index = []
        for idx in indices:
            if isinstance(idx, RefTile):
                new_index.append(idx._slice)
            else:
                new_index.append(idx)
        return new_index

    @classmethod
    def _handle_expand(
        cls,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object] | None,
    ) -> object:
        """Handle tensor.expand() operations with RefTile arguments."""
        if kwargs is None:
            kwargs = {}
        
        # args[0] is the tensor, args[1:] are the sizes
        tensor = args[0]
        sizes = args[1:]
        
        # Convert RefTile objects to their sizes
        new_sizes = []
        for size in sizes:
            if isinstance(size, RefTile):
                # Use the size of the slice (end - begin)
                new_sizes.append(size._slice.stop - size._slice.start)
            else:
                new_sizes.append(size)
        
        # Call expand with the converted sizes
        return tensor.expand(*new_sizes, **kwargs)
    
    @classmethod
    def _handle_view(
        cls,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object] | None,
    ) -> object:
        """Handle tensor.view() operations with RefTile arguments."""
        if kwargs is None:
            kwargs = {}
        
        # args[0] is the tensor, args[1:] are the sizes
        tensor = args[0]
        sizes = args[1:]
        
        # Convert RefTile objects to their sizes
        new_sizes = []
        for size in sizes:
            if isinstance(size, RefTile):
                # Use the size of the slice (end - begin)
                new_sizes.append(size._slice.stop - size._slice.start)
            else:
                new_sizes.append(size)
        
        # Call view with the converted sizes
        return tensor.view(*new_sizes, **kwargs)
    
    @classmethod
    def _handle_reshape(
        cls,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object] | None,
    ) -> object:
        """Handle torch.reshape() operations with RefTile arguments."""
        if kwargs is None:
            kwargs = {}
        
        # args[0] is the tensor, args[1] is the shape (which may be a list/tuple)
        tensor = args[0]
        if len(args) > 1:
            shape = args[1]
            # If shape is a list/tuple, convert any RefTile objects in it
            if isinstance(shape, (list, tuple)):
                new_shape = []
                for size in shape:
                    if isinstance(size, RefTile):
                        new_shape.append(size._slice.stop - size._slice.start)
                    else:
                        new_shape.append(size)
                return torch.reshape(tensor, new_shape, **kwargs)
            else:
                # Single dimension reshape
                if isinstance(shape, RefTile):
                    shape = shape._slice.stop - shape._slice.start
                return torch.reshape(tensor, shape, **kwargs)
        
        return torch.reshape(tensor, **kwargs)

    def __repr__(self, tensor_contents: None = None) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"RefTile({self._slice!r})"

    def __index__(self) -> int:
        return self.block_size
    
    @property
    def index(self) -> RefTile:
        """Return self for .index attribute access in ref mode."""
        return self
    
    @classmethod
    def _handle_zeros(
        cls,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object] | None,
    ) -> torch.Tensor:
        """Handle torch.zeros() with RefTile in size argument."""
        if kwargs is None:
            kwargs = {}
        
        # args[0] is the size argument
        size = args[0]
        other_args = args[1:]
        
        # Convert RefTile objects to their sizes
        if isinstance(size, (list, tuple)):
            new_size = []
            for s in size:
                if isinstance(s, RefTile):
                    new_size.append(s._slice.stop - s._slice.start)
                else:
                    new_size.append(s)
            return torch.zeros(new_size, *other_args, **kwargs)
        elif isinstance(size, RefTile):
            # Single RefTile as size
            new_size = size._slice.stop - size._slice.start
            return torch.zeros(new_size, *other_args, **kwargs)
        else:
            return torch.zeros(size, *other_args, **kwargs)
    
    @classmethod
    def _handle_ones(
        cls,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object] | None,
    ) -> torch.Tensor:
        """Handle torch.ones() with RefTile in size argument."""
        if kwargs is None:
            kwargs = {}
        
        # args[0] is the size argument
        size = args[0]
        other_args = args[1:]
        
        # Convert RefTile objects to their sizes
        if isinstance(size, (list, tuple)):
            new_size = []
            for s in size:
                if isinstance(s, RefTile):
                    new_size.append(s._slice.stop - s._slice.start)
                else:
                    new_size.append(s)
            return torch.ones(new_size, *other_args, **kwargs)
        elif isinstance(size, RefTile):
            # Single RefTile as size
            new_size = size._slice.stop - size._slice.start
            return torch.ones(new_size, *other_args, **kwargs)
        else:
            return torch.ones(size, *other_args, **kwargs)
    
    @classmethod
    def _handle_full(
        cls,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object] | None,
    ) -> torch.Tensor:
        """Handle torch.full() with RefTile in size argument."""
        if kwargs is None:
            kwargs = {}
        
        # args[0] is the size argument, args[1] is the fill value
        size = args[0]
        fill_value = args[1]
        other_args = args[2:]
        
        # Convert RefTile objects to their sizes
        if isinstance(size, (list, tuple)):
            new_size = []
            for s in size:
                if isinstance(s, RefTile):
                    new_size.append(s._slice.stop - s._slice.start)
                else:
                    new_size.append(s)
            return torch.full(new_size, fill_value, *other_args, **kwargs)
        elif isinstance(size, RefTile):
            # Single RefTile as size
            new_size = size._slice.stop - size._slice.start
            return torch.full(new_size, fill_value, *other_args, **kwargs)
        else:
            return torch.full(size, fill_value, *other_args, **kwargs)
