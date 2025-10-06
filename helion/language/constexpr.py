from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import NamedTuple
from typing_extensions import TypeVar

import contextlib
import functools
import torch
from torch.utils._pytree import tree_map

from .. import exc
from .._compiler.ast_extension import expr_from_string
from . import _decorators

if TYPE_CHECKING:
    import ast
    from collections.abc import Generator

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.type_propagation import TypeInfo
    from .._compiler.variable_origin import Origin

_T = TypeVar("_T")


class SpecializedInt(int):
    """Marker int indicating the value arose from hl.specialize."""

    __slots__ = ()

    def __new__(cls, value: int) -> "SpecializedInt":
        return int.__new__(cls, value)


def _pad_shape(shape: object) -> object:
    """Recursively pad shape dimensions to next power of 2 for SpecializedInt values."""
    from triton import next_power_of_2

    def _pad_dim(dim: object) -> object:
        if isinstance(dim, SpecializedInt) and dim > 0:
            padded = next_power_of_2(int(dim))
            return padded if padded != dim else dim
        return dim

    return tree_map(_pad_dim, shape)


def _should_pad() -> bool:
    """Check if we should pad dimensions (inside device loop during compilation)."""
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.type_propagation import _device_loop_depth

    return CompileEnvironment.has_current() and _device_loop_depth.get() > 0


def _wrap_factory(original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Wrap a torch module factory function to pad SpecializedInt dimensions."""
    @functools.wraps(original)
    def wrapper(*args: object, **kwargs: object) -> torch.Tensor:
        if not _should_pad():
            return original(*args, **kwargs)
        # Extract and pad size argument
        if args and isinstance(args[0], (int, list, tuple, torch.Size)):
            args = (_pad_shape(args[0]),) + args[1:]
        elif "size" in kwargs:
            kwargs = dict(kwargs)
            kwargs["size"] = _pad_shape(kwargs["size"])
        return original(*args, **kwargs)
    return wrapper


# Patch module-level tensor factories at import time
torch.zeros = _wrap_factory(torch.zeros)
torch.ones = _wrap_factory(torch.ones)
torch.empty = _wrap_factory(torch.empty)
torch.full = _wrap_factory(torch.full)


@contextlib.contextmanager
def _patch_tensor_factories() -> Generator[None, None, None]:
    """Context manager that patches tensor instance methods to pad SpecializedInt dimensions."""

    def _make_padded_method(original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        @functools.wraps(original)
        def wrapper(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            # Check at call time, not at patch time
            if not _should_pad():
                return original(self, *args, **kwargs)
            padded_args = (_pad_shape(args[0]),) + args[1:] if args else args
            padded_kwargs = dict(kwargs) if "size" in kwargs else kwargs
            if "size" in kwargs:
                padded_kwargs["size"] = _pad_shape(kwargs["size"])
            return original(self, *padded_args, **padded_kwargs)
        return wrapper

    methods_to_patch = [
        ("new_zeros", torch.Tensor.new_zeros),
        ("new_ones", torch.Tensor.new_ones),
        ("new_empty", torch.Tensor.new_empty),
        ("new_full", torch.Tensor.new_full),
    ]

    for name, original in methods_to_patch:
        setattr(torch.Tensor, name, _make_padded_method(original))

    try:
        yield
    finally:
        for name, original in methods_to_patch:
            setattr(torch.Tensor, name, original)


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
        return SpecializedInt(on_symint(value))
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
