from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

import torch
from torch.utils._pytree import tree_map_only

from .. import exc
from .compile_environment import CompileEnvironment

OpOverload = torch._ops.OpOverload

if TYPE_CHECKING:
    from collections.abc import Callable

    _T = TypeVar("_T")


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
