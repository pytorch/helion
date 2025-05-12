from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.utils import triton_type

from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from ..exc import NotInsideKernel
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["full", "zeros"]


def zeros(shape: list[object], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Return a device-tensor filled with zeros

    :param shape: a list of sizes (or tile indices which are implicitly converted to sizes)
    :param dtype: torch.dtype, default is torch.float32
    :return: a device tensor of the given shape and dtype
    """
    return full(shape, 0.0 if dtype.is_floating_point else 0, dtype=dtype)


@_decorators.api(tiles_as_sizes=True)
def full(
    shape: list[object], value: float, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create a device-tensor filled with a specified value.

    :param shape: A list of sizes (or tile indices which are implicitly converted to sizes).
    :param value: The value to fill the tensor with.
    :param dtype: The data type of the tensor, default is torch.float32.
    :return: A device tensor of the given shape and dtype.
    """
    raise NotInsideKernel


@_decorators.register_fake(full)
def _full_fake(
    shape: list[int | torch.SymInt],
    value: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")
    env = CompileEnvironment.current()
    env.add_kernel_tensor_size(shape)
    return torch.empty(
        [*shape],
        dtype=dtype,
        device=env.device,
    )


@_decorators.codegen(full)
def _full_codegen(state: CodegenState) -> ast.AST:
    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)
    shape_str = state.device_function.tile_strategy.shape_str(fake_value.size())
    type_str = triton_type(fake_value.dtype)
    value_str = state.device_function.literal_expr(state.proxy_arg(1))
    return expr_from_string(f"tl.full({shape_str}, {value_str}, {type_str})")
