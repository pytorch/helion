from __future__ import annotations

import itertools
import traceback
from typing import TYPE_CHECKING
from typing import TypeVar

import torch
from torch.utils._pytree import tree_map_only

from .. import exc
from .._utils import convert_tile_indices_to_slices
from .._utils import create_shape_matching_slices
from .tile_interface import TileInterface

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")

# Counter for generating unique block_ids in ref mode
_ref_mode_block_id_counter = itertools.count()

# Dict to map tensor id -> block_ids for tracking (cleared at kernel start)
_tensor_block_ids: dict[int, tuple[int | None, ...]] = {}

# Patterns indicating library/framework code (not user code)
_LIBRARY_PATH_PATTERNS = (
    "/helion/helion/",
    "/torch/",
    "/unittest/",
    "/pytest/",
    "/site-packages/",
    "<frozen",
)


_ADD_OPS: set[object] = {
    torch.add,
    torch.Tensor.add,
    torch.Tensor.add_,
    torch.Tensor.__add__,
    torch.Tensor.__radd__,
}
_SUB_OPS: set[object] = {
    torch.sub,
    torch.Tensor.sub,
    torch.Tensor.sub_,
    torch.Tensor.__sub__,
    torch.Tensor.__rsub__,
}

try:
    _ADD_OPS.add(torch.ops.aten.add.Tensor)
    _SUB_OPS.add(torch.ops.aten.sub.Tensor)
except AttributeError:  # pragma: no cover - aten fallback not always defined
    pass


class RefTile(TileInterface, torch.Tensor):
    _slice: slice
    _block_size: int
    _block_id: int

    def __init__(
        self, begin: int, end: int, block_size: int, block_id: int | None = None
    ) -> None:
        super().__init__()

        from ..runtime.ref_mode import is_in_ref_mode_context

        assert is_in_ref_mode_context()
        self._slice = slice(begin, end, None)
        self._block_size = block_size
        self._block_id = block_id if block_id is not None else next(
            _ref_mode_block_id_counter
        )

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

        if func in _ADD_OPS:
            return cls._handle_add(args)

        if func in _SUB_OPS:
            return cls._handle_sub(args)

        # For any other torch.* function or torch.Tensor.* method, convert tiles to sizes
        is_torch_func = getattr(func, "__module__", "") == "torch"
        is_tensor_method = hasattr(torch.Tensor, getattr(func, "__name__", ""))
        if is_torch_func or is_tensor_method:
            new_args = cls._tiles_to_sizes(args)
            new_kwargs = cls._tiles_to_sizes(kwargs) if kwargs else {}
            return func(*new_args, **new_kwargs)

        raise exc.IncorrectTileUsage(func)

    @classmethod
    def _tiles_to_sizes(cls, it: _T) -> _T:
        return tree_map_only(RefTile, cls._tile_to_size, it)

    @staticmethod
    def _tile_to_size(tile: RefTile) -> int:
        return tile.block_size

    @classmethod
    def _handle_add(cls, args: tuple[object, ...]) -> torch.Tensor:
        tile, offset, flipped = cls._extract_tile_and_offset(args, torch.add)
        return tile.index + offset if not flipped else offset + tile.index

    @classmethod
    def _handle_sub(cls, args: tuple[object, ...]) -> torch.Tensor:
        tile, offset, flipped = cls._extract_tile_and_offset(args, torch.sub)
        return (
            tile.index - offset
            if not flipped
            else offset - tile.index  # pragma: no cover - defensive
        )

    @classmethod
    def _extract_tile_and_offset(
        cls, args: tuple[object, ...], func: object
    ) -> tuple[RefTile, int, bool]:
        if len(args) != 2:
            raise exc.IncorrectTileUsage(func)

        lhs, rhs = args
        flipped = False

        if isinstance(lhs, RefTile) and cls._is_valid_offset(rhs):
            tile = lhs
            offset = cls._to_int(rhs, func)
        elif isinstance(rhs, RefTile) and cls._is_valid_offset(lhs):
            tile = rhs
            offset = cls._to_int(lhs, func)
            flipped = True
        else:
            raise exc.IncorrectTileUsage(func)

        return tile, offset, flipped

    @staticmethod
    def _is_valid_offset(value: object) -> bool:
        if isinstance(value, int):
            return True
        return bool(isinstance(value, torch.Tensor) and value.ndim == 0)

    @staticmethod
    def _to_int(value: object, func: object) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            return int(value.item())
        raise exc.IncorrectTileUsage(func)

    @classmethod
    def _handle_getitem(
        cls,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object] | None,
    ) -> object:
        """Handle tensor[index] operations with tile indices."""
        tensor, index = args
        assert isinstance(tensor, torch.Tensor)

        # Extract block_ids from RefTile indices
        indices = index if isinstance(index, tuple) else (index,)
        block_ids: list[int | None] = []
        for idx in indices:
            if isinstance(idx, RefTile):
                block_ids.append(idx._block_id)
            elif not isinstance(idx, int):  # slice or other -> adds a dim
                block_ids.append(None)
            # int indices reduce dims, so don't append

        slice_index = convert_tile_indices_to_slices(index)
        # pyrefly: ignore [bad-index]
        result = tensor[slice_index]

        # Register result with block_ids for tracking
        if block_ids and isinstance(result, torch.Tensor) and result.ndim > 0:
            if len(block_ids) == result.ndim:
                _tensor_block_ids[id(result)] = tuple(block_ids)

        return result

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
        # pyrefly: ignore [bad-index]
        target_shape = tensor[slice_index].shape

        if (
            isinstance(value, torch.Tensor)
            and value.shape != target_shape
            and len(value.shape) == len(target_shape)
        ):
            slices = create_shape_matching_slices(value.shape, target_shape)
            value = value[slices]

        # pyrefly: ignore [unsupported-operation]
        tensor[slice_index] = value
        return None

    def __repr__(self, tensor_contents: None = None) -> str:
        return f"RefTile({self._slice!r})"

    def __index__(self) -> int:
        return self.block_size

    @property
    def index(self) -> torch.Tensor:  # pyrefly: ignore [bad-override]
        """Return tensor of indices for .index attribute access in ref mode."""
        from .._compiler.compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        data = torch.arange(
            self._slice.start, self._slice.stop, dtype=torch.int32, device=env.device
        )
        _tensor_block_ids[id(data)] = (self._block_id,)
        return data


def reset_ref_mode_block_id_counter() -> None:
    """Reset the block_id counter and tracking dict. Called at the start of each ref mode kernel execution."""
    global _ref_mode_block_id_counter
    _ref_mode_block_id_counter = itertools.count()
    _tensor_block_ids.clear()


def get_block_ids(tensor: torch.Tensor) -> tuple[int | None, ...] | None:
    """Get block_ids for a tensor if tracked."""
    return _tensor_block_ids.get(id(tensor))


def maybe_set_block_ids(tensor: object, block_ids: tuple[int | None, ...] | None) -> None:
    """Set block_ids for a tensor if block_ids is non-empty and matches tensor ndim."""
    if block_ids and isinstance(tensor, torch.Tensor) and len(block_ids) == tensor.ndim:
        _tensor_block_ids[id(tensor)] = block_ids


def check_broadcast_and_get_result_block_ids(
    tensors: list[torch.Tensor],
) -> tuple[int | None, ...] | None:
    """Check broadcast compatibility and return result block_ids."""
    # Get tracked tensors (those with block_ids)
    tracked: list[tuple[torch.Tensor, tuple[int | None, ...]]] = []
    for t in tensors:
        bids = _tensor_block_ids.get(id(t))
        if bids is not None:
            tracked.append((t, bids))

    if not tracked:
        return None

    shapes = [[*t.shape] for t, _ in tracked]
    bids = [[*b] for _, b in tracked]
    max_rank = max(len(s) for s in shapes)

    # Right-align with padding
    for i in range(len(shapes)):
        pad = max_rank - len(shapes[i])
        shapes[i] = [1] * pad + shapes[i]
        bids[i] = [None] * pad + bids[i]

    result: list[int | None] = []
    for d in range(max_rank):
        ids_in_dim = {bids[i][d] for i in range(len(tracked)) if shapes[i][d] != 1 and bids[i][d] is not None}
        if len(ids_in_dim) >= 2:
            _raise_mismatch(d, shapes, bids, ids_in_dim)
        result.append(next(iter(ids_in_dim)) if ids_in_dim else None)
    return tuple(result)


def _raise_mismatch(
    dim: int, shapes: list[list[int]], bids: list[list[int | None]], ids_in_dim: set[int],
) -> None:
    """Raise ShapeMismatch with location info."""
    fmt = lambda s, b: "[" + ", ".join(f"u{x}" if x is not None else str(y) for y, x in zip(s, b, strict=False)) + "]"
    descs = [f"tensor with shape {fmt(s, b)}" for s, b in zip(shapes, bids, strict=False)
             if s[dim] != 1 and b[dim] in ids_in_dim][:2]

    loc = ""
    for f in reversed(traceback.extract_stack()):
        if not any(p in f.filename for p in _LIBRARY_PATH_PATTERNS):
            loc = f"\n  at {f.filename}:{f.lineno}: {f.line}"
            break

    raise exc.ShapeMismatch(descs[0] if descs else "unknown", (descs[1] if len(descs) > 1 else "unknown") + loc)
