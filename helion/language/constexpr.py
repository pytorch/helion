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
    from torch._inductor.runtime.runtime_utils import next_power_of_2

    if origin.is_device():
        raise exc.SpecializeOnDevice

    proxy = value.proxy()
    env = CompileEnvironment.current()

    def handle_symint(symint: torch.SymInt) -> torch.SymInt:
        # Record that we're specializing on these symbols
        env.specializations.mark_source_symbols(symint._sympy_().free_symbols)

        # Get concrete value for constraint
        concrete_val = symint.__int__()

        # Create new unbacked symbol with power-of-2 hint
        hint = next_power_of_2(concrete_val)
        new_symbol = env.create_unbacked_symint(hint=hint)

        # Add constraint: new_symbol == concrete_val
        import sympy
        env.shape_env._add_assertion(
            sympy.Eq(new_symbol._sympy_(), concrete_val)
        )

        # Register the new symbol as specialized (allow it in allocations)
        # Note: We also add the original symint's free_symbols to track what we're specializing on
        env.specializations.register_symint(new_symbol, concrete_val)

        # Don't immediately allocate a reduction dimension - only allocate when actually needed
        # This prevents unnecessary power-of-2 indexing for host tensor slicing

        return new_symbol

    specialized = _convert_specializable(proxy, on_symint=handle_symint)
    return TypeInfo.from_example(specialized, origin=origin)


@_decorators.codegen(specialize)
def _(state: CodegenState) -> ast.AST:
    # In HOST context, use the concrete value, not the power-of-2 hint
    # The power-of-2 hint is only for device-side allocations via _get_shape_string
    # Slicing of specialized tensors now reuses the dimension instead of allocating new reduction dims
    value = state.proxy_arg(0)
    return expr_from_string(repr(value))


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
