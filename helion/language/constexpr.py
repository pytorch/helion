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
    from torch._dynamo.source import LocalSource
    from torch._dynamo.source import TensorProperty
    from torch._dynamo.source import TensorPropertySource

    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.type_propagation import TypeInfo
    from .._compiler.variable_origin import ArgumentOrigin
    from .._compiler.variable_origin import TensorStrideOrigin

    if origin.is_device():
        raise exc.SpecializeOnDevice

    proxy = value.proxy()
    env = CompileEnvironment.current()

    newly_specialized: set[object] = set()

    def handle_symint(symint: torch.SymInt) -> int:
        new_syms = symint._sympy_().free_symbols - env.specialized_vars
        env.specialized_vars.update(new_syms)
        newly_specialized.update(new_syms)
        return symint.__int__()

    specialized = _convert_specializable(proxy, on_symint=handle_symint)

    # Detect stride specialization via TensorStrideOrigin set by propagate_call.
    def _add_stride_specializations(ti: TypeInfo) -> None:
        if isinstance(ti.origin, TensorStrideOrigin) and isinstance(ti.origin.value, ArgumentOrigin):
            env.specialized_strides.add((ti.origin.value.name, ti.origin.key))
        for elem in getattr(ti, "element_types", ()):
            _add_stride_specializations(elem)

    _add_stride_specializations(value)

    # Record explicit stride sources for newly specialized symbols so
    # _specialize_extra() can use them instead of the idx-1 heuristic.
    if env.specialized_strides and newly_specialized:
        for sym in newly_specialized:
            if sym in env.specialized_stride_sources:
                continue
            for src in env.shape_env.var_to_sources.get(sym, []):
                if (isinstance(src, TensorPropertySource)
                        and src.prop is TensorProperty.STRIDE
                        and isinstance(src.base, LocalSource)
                        and src.idx is not None
                        and (src.base.local_name, src.idx) in env.specialized_strides):
                    env.specialized_stride_sources[sym] = src
                    break

    return TypeInfo.from_example(specialized, origin=origin)


@_decorators.codegen(specialize, "triton")
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
        # pyrefly: ignore [bad-return]
        return on_symint(value)
    if isinstance(value, int):
        # pyrefly: ignore [bad-return]
        return value
    if isinstance(value, (torch.Size, tuple, list)):
        try:
            return type(value)(
                [_convert_specializable(x, on_symint=on_symint) for x in value]
            )
        except exc.SpecializeArgType:
            raise exc.SpecializeArgType(value) from None
    raise exc.SpecializeArgType(value)
