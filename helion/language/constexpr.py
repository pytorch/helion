from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.type_propagation import TypeInfo
    from .._compiler.variable_origin import Origin


class ConstExpr(NamedTuple):
    """
    Typically used as a type annotation for kernels:

    .. code-block:: python

        @helion.kernel()
        def fn(v: hl.constexpr, ...):
            ...

    Can also be used when calling a kernel:

    .. code-block:: python

        some_kernel(..., hl.constexpr(5.0))

    Causes the generated code to specialize on the value of `v`, where a different
    kernel, hardcoding the value of v, will be generated every time `v` changes.

    See Also:
        - :func:`specialize`: Convert dynamic shapes to compile-time constants
    """

    value: object

    def __index__(self) -> int:
        if isinstance(self.value, int):
            return self.value
        raise TypeError(f"ConstExpr cannot be indexed: {self.value}")

    def __bool__(self) -> bool:
        return bool(self.value)


@_decorators.api(is_device_only=False)
def specialize(value: int | torch.SymInt) -> int:
    """
    Turn a dynamic shape into a compile-time constant.  Example:

           hl.specialize(tensor.size(1))

    Args:
        value: The symbolic value to specialize on.

    Returns:
        int: The specialized value.

    See Also:
        - :class:`ConstExpr`: Create compile-time constants for kernel parameters
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(specialize)
def _(value: TypeInfo, *, origin: Origin) -> TypeInfo:
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.type_propagation import TypeInfo

    if origin.is_device():
        raise exc.SpecializeOnDevice
    proxy = value.proxy()
    if isinstance(proxy, torch.SymInt):
        CompileEnvironment.current().specialized_vars.update(
            proxy._sympy_().free_symbols
        )
        return TypeInfo.from_example(proxy.__int__(), origin=origin)
    if isinstance(proxy, int):
        return TypeInfo.from_example(proxy, origin=origin)  # already specialized
    raise exc.SpecializeArgType(value)


@_decorators.codegen(specialize)
def _(state: CodegenState) -> ast.AST:
    value = state.proxy_arg(0)
    if isinstance(value, torch.SymInt):
        value = value.__int__()
    assert isinstance(value, int)
    return expr_from_string(repr(value))


@_decorators.ref(specialize)
def _(value: int | torch.SymInt) -> int:
    return int(value)
