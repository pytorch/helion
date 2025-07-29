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

    def __repr__(self, tensor_contents: None = None) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"RefTile({self._slice!r})"

    def __index__(self) -> int:
        return self.block_size
