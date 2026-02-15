from __future__ import annotations

from typing import Callable
from typing import ClassVar

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from .._utils import next_power_of_2


class _PadTensorFactoryMode(TorchDispatchMode):
    """Dispatch mode that pads tensor factory size arguments."""

    _SIZE_ARG_INDEX: ClassVar[dict[Callable[..., torch.Tensor], int]] = {
        torch.ops.aten.zeros.default: 0,
        torch.ops.aten.ones.default: 0,
        torch.ops.aten.empty.memory_format: 0,
        torch.ops.aten.full.default: 0,
        torch.ops.aten.new_empty.default: 1,
        torch.ops.aten.new_full.default: 1,
        torch.ops.aten.new_zeros.default: 1,
        torch.ops.aten.new_ones.default: 1,
    }

    # pyrefly: ignore [bad-override]
    def __torch_dispatch__(
        self,
        func: Callable[..., torch.Tensor],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> torch.Tensor:
        def _pad_shape(shape: object) -> object:
            """Pad positive integer dimension sizes to the next power of 2."""

            def _pad_dim(dim_size: object) -> object:
                if isinstance(dim_size, int) and dim_size > 0:
                    return next_power_of_2(dim_size)
                return dim_size

            return tree_map(_pad_dim, shape)

        kwargs = dict(kwargs or {})
        size_index = self._SIZE_ARG_INDEX.get(func)
        if size_index is not None:
            if "size" in kwargs:
                kwargs["size"] = _pad_shape(kwargs["size"])
            elif size_index < len(args):
                args_list = list(args)
                args_list[size_index] = _pad_shape(args_list[size_index])
                args = tuple(args_list)
        return func(*args, **kwargs)


patch_tensor_factories = _PadTensorFactoryMode


__all__ = ["patch_tensor_factories"]
