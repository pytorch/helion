
from __future__ import annotations

import ast
from collections.abc import Mapping
from collections.abc import Sequence
import inspect
import textwrap
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

import torch
from torch._inductor.codegen.wrapper import (
    user_defined_triton_kernel_transitive_closure_source_code,
)
from torch._inductor.utils import triton_type
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import convert
from .._compiler.ast_extension import create
from .._compiler.ast_extension import expr_from_string
from .._compiler.ast_extension import statement_from_string
from .._compiler.host_function import HostFunction
from .._compiler.output_header import SOURCE_MODULE
from . import _decorators

if TYPE_CHECKING:
    from types import FunctionType

    from .._compiler.inductor_lowering import CodegenState

    _T = TypeVar("_T")

__all__ = ["start_async_remote_copy"]

class AsyncCopyDescriptor:
    pass

@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def start_async_remote_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    device_id: int | tuple[int, ...],
) -> AsyncCopyDescriptor:
    raise exc.NotInsideKernel

@_decorators.type_propagation(start_async_remote_copy)
def _(src: torch.Tensor,
    dst: torch.Tensor,
    device_id: int | tuple[int, ...], *, origin: Origin) -> TypeInfo:

    from .._compiler.type_info import AsyncCopyDescriptorType

    element_types = {}

    return AsyncCopyDescriptorType(origin, element_types)

@_decorators.register_fake(start_async_remote_copy)
def _(
    src: torch.Tensor,
    dst: torch.Tensor,
    device_id: int | tuple[int, ...]
) -> object:
    return AsyncCopyDescriptor()