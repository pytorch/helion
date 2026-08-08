from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable


_RESHAPE_OPS = {
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
}


class TileShapeConstraintMode(TorchDispatchMode):
    """Record block-size constraints before FakeTensor validates shape ops."""

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func in _RESHAPE_OPS and len(args) >= 2:
            tensor, shape = args[:2]
            if isinstance(tensor, torch.Tensor) and isinstance(shape, (list, tuple)):
                if all(isinstance(dim, (int, torch.SymInt)) for dim in shape):
                    CompileEnvironment.current().prepare_tile_reshape(
                        tensor.shape,
                        shape,
                    )
        return func(*args, **(kwargs or {}))
