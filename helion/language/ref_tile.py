from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from .. import exc
from .._compiler.compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable


class RefTile(torch.Tensor):
    _slice: slice
    _block_size: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()

        from ..runtime.ref_mode import is_in_ref_mode_context

        assert is_in_ref_mode_context()
        start: int = -1
        stop: int = -1
        step: int | None = None
        if len(args) >= 2 and all(isinstance(a, int) for a in args[:2]):
            # Two or three int args: RefTile(start, stop, step=None)
            start = cast("int", args[0])
            stop = cast("int", args[1])
            step = cast("int", args[2]) if len(args) > 2 else None
        elif "start" in kwargs and "stop" in kwargs:
            # Keyword args: RefTile(start=..., stop=..., step=...)
            start = cast("int", kwargs["start"])
            stop = cast("int", kwargs["stop"])
            step = cast("int | None", kwargs.get("step"))
        else:
            raise ValueError("RefTile must be created with start/stop parameters")
        self._slice = slice(start, stop, step)

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

    @property
    def index(self) -> torch.Tensor:
        env = CompileEnvironment.current()
        return torch.arange(self.start, self.stop, dtype=torch.int32, device=env.device)

    @property
    def begin(self) -> int:
        return self._slice.start

    @property
    def end(self) -> int:
        return self._slice.stop

    @property
    def start(self) -> int:
        return self._slice.start

    @property
    def stop(self) -> int:
        return self._slice.stop

    @property
    def step(self) -> int:
        return self._slice.step

    @property
    def block_size(self) -> int:
        if hasattr(self, "_block_size"):
            return self._block_size
        return self.stop - self.start

    def set_block_size(self, block_size: int) -> None:
        self._block_size = block_size

    @property
    def id(self) -> int:
        # In ref mode, tiles are created with consistent block sizes
        # The ID is the index of the tile (start / block_size)
        assert self.block_size > 0
        return self.start // self.block_size

    def __index__(self) -> int:
        return self.block_size
