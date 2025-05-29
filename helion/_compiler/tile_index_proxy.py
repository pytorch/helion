from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from typing import Protocol
from typing import Sequence
from typing import TypeVar
from typing import cast
from typing_extensions import Self

import torch
from torch.utils._pytree import tree_map_only

from .. import exc
from .compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable

    _T = TypeVar("_T")

    class _TLS(Protocol):
        index_calls: CheckForIndexCalls | None


tls: _TLS = cast("_TLS", threading.local())


class TileIndexProxy(torch.Tensor):
    def __init__(self, block_size_index: int) -> None:
        super().__init__()
        self.block_size_index = block_size_index

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        from ..language.memory_ops import load
        from ..language.memory_ops import store

        if func is torch.Tensor.__getitem__:
            if len(args) != 2 or kwargs:
                raise exc.IncorrectTileUsage(func)
            tensor, index = args
            assert isinstance(tensor, torch.Tensor)
            return load(tensor, cls.prepare_index(index))
        if func is torch.Tensor.__setitem__:
            if len(args) != 3 or kwargs:
                raise exc.IncorrectTileUsage(func)
            tensor, index, value = args
            assert isinstance(tensor, torch.Tensor)
            assert isinstance(value, torch.Tensor)
            return store(tensor, cls.prepare_index(index), value)
        if (
            func is torch.Tensor.__index__
            and (index_calls := getattr(tls, "index_calls", None)) is not None
        ):
            index_calls.count += 1
        if func is torch.Tensor.__format__:
            return repr(args[0])
        raise exc.IncorrectTileUsage(func)

    @staticmethod
    def prepare_index(index: object) -> list[object]:
        if isinstance(index, (list, tuple)):
            return [*index]
        assert isinstance(index, TileIndexProxy)
        return [index]

    def __repr__(self, tensor_contents: None = None) -> str:
        return f"TileIndexProxy({self.block_size_index!r})"

    @classmethod
    def tiles_to_sizes(cls, it: _T) -> _T:
        return tree_map_only(TileIndexProxy, cls._tile_to_size, it)

    @staticmethod
    def _tile_to_size(x: TileIndexProxy) -> torch.SymInt:
        return CompileEnvironment.current().block_sizes[x.block_size_index].var

    @property
    def index(self) -> torch.Tensor:
        """
        Alias for hl.tile_index, which retrieves a tensor containing the offsets for a tile.
        """
        from ..language.tiles import tile_index

        return tile_index(self)


class CheckForIndexCalls:
    """
    Unfortunately, the `__torch_function__` method of `TileIndexProxy` does not work
    properly when operations like view() are called on a `TileIndexProxy` object.  It calls
    `__torch_function__(Tensor.__index__, ...)` but then discards the result because it is not
    an integer (if a SymInt is returned).

    This class is a workaround to detect this case and turn tiles to sizes in the caller.
    """

    @classmethod
    def retry_call(
        cls,
        fn: Callable[..., object],
        proxy_args: Sequence[object],
        proxy_kwargs: dict[str, object],
    ) -> object:
        index_calls = cls()
        try:
            with index_calls:
                return fn(*proxy_args, **proxy_kwargs)
        except TypeError:
            if index_calls.count == 0:
                raise
        # This is likely a view op, try again with tiles_to_sizes
        proxy_args = TileIndexProxy.tiles_to_sizes(proxy_args)
        proxy_kwargs = TileIndexProxy.tiles_to_sizes(proxy_kwargs)
        return fn(*proxy_args, **proxy_kwargs)

    def __init__(self) -> None:
        self.count = 0

    def __enter__(self) -> Self:
        assert getattr(tls, "index_calls", None) is None
        tls.index_calls = self
        return self

    def __exit__(self, *args: object) -> None:
        tls.index_calls = None
