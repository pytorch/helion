from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import NamedTuple
from typing_extensions import TypeVar

import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.type_propagation import TypeInfo
    from .._compiler.variable_origin import Origin

    _T = TypeVar("_T")


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
def specialize(value: _T) -> _T:
    """
    Turn dynamic shapes into compile-time constants. Examples::

           channels = hl.specialize(tensor.size(1))
           height, width = hl.specialize(tensor.shape[-2:])

    Args:
        value: The symbolic value or sequence of symbolic values to specialize on.

    Returns:
        A Python int or a sequence containing only Python ints.

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
    env = CompileEnvironment.current()

    def handle_symint(symint: torch.SymInt) -> int:
        env.specialized_vars.update(symint._sympy_().free_symbols)
        return symint.__int__()

    specialized = _convert_specializable(proxy, on_symint=handle_symint)
    return TypeInfo.from_example(specialized, origin=origin)


@_decorators.codegen(specialize)
def _(state: CodegenState) -> ast.AST:
    value = state.proxy_arg(0)
    specialized = _convert_specializable(value)
    return expr_from_string(repr(specialized))


@_decorators.ref(specialize)
def _(value: _T) -> _T:
    return _convert_specializable(value)


def _convert_specializable(
    value: _T,
    *,
    on_symint: Callable[[torch.SymInt], int] = lambda symint: symint.__int__(),
) -> _T:
    if isinstance(value, torch.SymInt):
        return on_symint(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (torch.Size, tuple, list)):
        try:
            return type(value)(
                [_convert_specializable(x, on_symint=on_symint) for x in value]
            )
        except exc.SpecializeArgType:
            raise exc.SpecializeArgType(value) from None
    raise exc.SpecializeArgType(value)
