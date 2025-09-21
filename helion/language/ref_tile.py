from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import exc
from .._utils import convert_tile_indices_to_slices
from .._utils import create_shape_matching_slices
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

        slice_index = convert_tile_indices_to_slices(index)
        return tensor[slice_index]  # pyright: ignore[reportArgumentType]

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

        slice_index = convert_tile_indices_to_slices(index)
        target_shape = tensor[slice_index].shape  # pyright: ignore[reportArgumentType]

        # Slice value tensor to match target shape if needed
        if (
            isinstance(value, torch.Tensor)
            and value.shape != target_shape
            and len(value.shape) == len(target_shape)
        ):
            slices = create_shape_matching_slices(value.shape, target_shape)
            value = value[slices]

        tensor[slice_index] = value  # pyright: ignore[reportArgumentType]
        return None

    def __repr__(self, tensor_contents: None = None) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"RefTile({self._slice!r})"

    def __index__(self) -> int:
        return self.block_size

    @property
    def index(self) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return tensor of indices for .index attribute access in ref mode."""
        from .._compiler.compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        return torch.arange(
            self._slice.start, self._slice.stop, dtype=torch.int32, device=env.device
        )
