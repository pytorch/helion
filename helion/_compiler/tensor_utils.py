from __future__ import annotations

import inspect
from typing import Callable
from typing import ClassVar

import torch
from torch._ops import _len_torch_dispatch_stack_pre_dispatch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._device import DeviceContext
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from torch.utils._pytree import tree_map

from .._utils import next_power_of_2
from .compile_environment import CompileEnvironment

# Dispatch modes Helion pushes around host code (fake tensors and peer-view
# capture while binding, its CPU codegen harness's CUDA guard), each with the
# __torch_dispatch__ it had when registered: their Python changes no existing
# tensor's layout (host_modes_trusted).
HOST_DISPATCH_MODES: dict[type[TorchDispatchMode], object] = {
    FakeTensorMode: inspect.getattr_static(FakeTensorMode, "__torch_dispatch__")
}
_DEVICE_CONTEXT_HOOK = DeviceContext.__dict__["__torch_function__"]


def host_dispatch_mode(kind: type[TorchDispatchMode]) -> type[TorchDispatchMode]:
    """Register ``kind`` in HOST_DISPATCH_MODES."""
    HOST_DISPATCH_MODES[kind] = inspect.getattr_static(kind, "__torch_dispatch__")
    return kind


def function_modes_trusted() -> bool:
    """Whether every active torch function mode is a stock torch.device context:
    its class keeps its import-time __torch_function__ and the instance holds
    none of its own. Dynamo traces this: it keeps the function mode stack
    symbolically (without torch.device contexts)."""
    return (
        all(
            type(mode := torch._C._get_function_stack_at(i)) is DeviceContext
            and "__torch_function__" not in vars(mode)
            for i in range(torch._C._len_torch_function_stack())
        )
        and DeviceContext.__dict__.get("__torch_function__") is _DEVICE_CONTEXT_HOOK
    )


def host_modes_trusted() -> bool:
    """Whether no torch mode can run Python of its own inside a host torch call:
    function_modes_trusted, every dispatch mode is an instance of one in
    HOST_DISPATCH_MODES whose class keeps its registered __torch_dispatch__
    and which holds none of its own, and the pre-dispatch stack is empty."""
    if _len_torch_dispatch_stack_pre_dispatch():
        return False
    if not (
        torch._C._len_torch_function_stack() or torch._C._len_torch_dispatch_stack()
    ):
        # The common case, on every call of a kernel whose binding asked. The
        # torch.device hook is checked even with no context active, as under
        # capture, where Dynamo's stack leaves torch.device contexts out.
        return DeviceContext.__dict__.get("__torch_function__") is _DEVICE_CONTEXT_HOOK
    return function_modes_trusted() and all(
        (kind := type(mode)) in HOST_DISPATCH_MODES
        and inspect.getattr_static(kind, "__torch_dispatch__")
        is HOST_DISPATCH_MODES[kind]
        and "__torch_dispatch__" not in vars(mode)
        for mode in _get_current_dispatch_mode_stack()
    )


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

            leaves: list[object] = []
            tree_map(lambda leaf: leaves.append(leaf), shape)
            fully_concrete = bool(leaves) and all(
                isinstance(leaf, int) for leaf in leaves
            )

            def _pad_dim(dim_size: object) -> object:
                if not isinstance(dim_size, int) or dim_size <= 0:
                    return dim_size
                if fully_concrete:
                    return next_power_of_2(dim_size)
                if dim_size < 16:
                    return dim_size
                # The padded dim of a tile is the full-slice reduction dim of
                # the same size (``hl.zeros([tile, 400])`` and ``y[tile, :]``
                # over 400 columns): its var keeps the padded tile and its mask
                # hides the padding columns.
                return CompileEnvironment.current().padded_extent_dim(dim_size)

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
