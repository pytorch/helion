from __future__ import annotations

from typing import Callable
from typing import ClassVar

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from triton import next_power_of_2


class _PadTensorFactoryMode(TorchDispatchMode):
    """Dispatch mode that pads tensor factory size arguments."""

    _SIZE_ARG_INDEX: ClassVar[dict[Callable[..., torch.Tensor], int]] = {
        torch.ops.aten.zeros.default: 0,  # pyright: ignore[reportAttributeAccessIssue]
        torch.ops.aten.ones.default: 0,  # pyright: ignore[reportAttributeAccessIssue]
        torch.ops.aten.empty.memory_format: 0,  # pyright: ignore[reportAttributeAccessIssue]
        torch.ops.aten.full.default: 0,  # pyright: ignore[reportAttributeAccessIssue]
        torch.ops.aten.new_empty.default: 1,  # pyright: ignore[reportAttributeAccessIssue]
        torch.ops.aten.new_full.default: 1,  # pyright: ignore[reportAttributeAccessIssue]
        torch.ops.aten.new_zeros.default: 1,  # pyright: ignore[reportAttributeAccessIssue]
        torch.ops.aten.new_ones.default: 1,  # pyright: ignore[reportAttributeAccessIssue]
    }

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


class _BroadcastBatchMatmulMode(TorchDispatchMode):
    """Dispatch mode that broadcasts batch dimensions for batched matmul operations."""

    def __torch_dispatch__(
        self,
        func: Callable[..., torch.Tensor],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> torch.Tensor:
        kwargs = dict(kwargs or {})
        if func == torch.ops.aten.baddbmm.default and len(args) >= 3:  # pyright: ignore[reportAttributeAccessIssue]
            input_tensor, batch1, batch2 = args[:3]
            if (
                isinstance(batch1, torch.Tensor)
                and isinstance(batch2, torch.Tensor)
                and batch1.ndim == 3
                and batch2.ndim == 3
            ):
                b1, b2 = batch1.size(0), batch2.size(0)
                if b1 == 1 and b2 != 1:
                    batch1 = batch1.expand(b2, -1, -1)
                    args = (input_tensor, batch1, batch2, *args[3:])
                elif b2 == 1 and b1 != 1:
                    batch2 = batch2.expand(b1, -1, -1)
                    args = (input_tensor, batch1, batch2, *args[3:])
        return func(*args, **kwargs)


patch_tensor_factories = _PadTensorFactoryMode
broadcast_batch_matmul = _BroadcastBatchMatmulMode


__all__ = ["broadcast_batch_matmul", "patch_tensor_factories"]
