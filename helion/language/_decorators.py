from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING
from typing import Generic
from typing import Literal
from typing import Protocol
from typing import TypeGuard
from typing import TypeVar
from typing import cast
from typing_extensions import Never

import torch
from torch.fx.experimental import proxy_tensor
from torch.utils._pytree import tree_map
from torch.utils._pytree import tree_map_only
from torch.utils._thunk import Thunk

from helion import exc

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.inductor_lowering import CodegenState
    from helion._compiler.type_propagation import TypeInfo
    from helion._compiler.variable_origin import Origin

    _T = TypeVar("_T")
    _C = TypeVar("_C", bound=Callable[..., object])

    class _Decorator(Protocol):
        def __call__(self, fn: _C) -> _C: ...

    class _NoReturnDecorator(Protocol, Generic[_T]):
        def __call__(self, fn: Callable[..., _T]) -> object: ...


class APIFunc(Protocol):
    """Protocol for Helion API functions that define operations within kernel code.

    This protocol defines the interface for functions decorated with @api. These functions
    represent operations that can be called in Helion kernel code and are compiled
    into the final device code.

    Attributes:
        __qualname__: The qualified name of the function.
        _helion_api: A literal True marker indicating this is a Helion API function.
        _is_device_loop: Whether this API function can transition between host and device code.
            When True, the function can contain both host and device code sections.
        _is_device_only: Whether this API function is intended for device code only.
            When True, the function can only be used within device code sections.
        _tiles_as_sizes: Whether tile indices should be converted to sizes automatically.
            Used primarily with tiling operations to transform indices to dimensions.
        _cache_type: Whether to cache the type information for repeated calls.
        _type_function: A callable that determines the return type of this function
            during type propagation phase.
        _codegen: A callable that generates the device code for this function.
        _fake_fn: A callable that provides a "fake" implementation used during
            tracing and compilation.
        _prepare_args: A callable that preprocesses the arguments before they're
            passed to the actual function implementation.
        _signature: The function signature for binding and validating arguments.
    """

    __qualname__: str
    _helion_api: Literal[True]
    # a device loop can transition between host and device code
    _is_device_loop: bool
    _is_device_only: bool
    _tiles_as_sizes: bool
    _cache_type: bool
    _type_function: Callable[..., TypeInfo] | None
    _codegen: Callable[[CodegenState], object] | None
    _fake_fn: Callable[..., object] | None
    _prepare_args: Callable[[tuple[object, ...]], tuple[object, ...]]
    _signature: inspect.Signature

    def __call__(self, *args: object, **kwargs: object) -> object: ...


def _no_call(*args: object, **kwargs: object) -> Never:
    raise TypeError("type_prop/codegen functions cannot be called directly")


def is_api_func(fn: object) -> TypeGuard[APIFunc]:
    return getattr(fn, "_helion_api", False)


def args_to_proxies(
    tracer: proxy_tensor.PythonKeyTracer,
    args: _T,
    kwargs: dict[str, object] | None = None,
) -> tuple[_T, dict[str, object]]:
    def unpack(x: object) -> object:
        if isinstance(x, (torch.Tensor, torch.SymInt, torch.SymBool, torch.SymFloat)):
            # pyre-ignore[6]
            return unpack(proxy_tensor.get_proxy_slot(x, tracer=tracer))
        if isinstance(x, proxy_tensor._ProxyTensor):
            return x.proxy
        if isinstance(x, Thunk):
            return x.force()
        return x

    return tree_map(
        unpack,
        (args, kwargs or {}),
    )


def tiles_as_sizes_prepare_args(*args: object) -> tuple[object, ...]:
    from helion._compiler.tile_index_proxy import TileIndexProxy

    return TileIndexProxy.tiles_to_sizes(args)


def no_op_prepare_args(*args: object) -> tuple[object, ...]:
    return args


def api(
    *,
    is_device_loop: bool = False,
    is_device_only: bool = True,
    tiles_as_sizes: bool = False,
    cache_type: bool = False,
    signature: inspect.Signature | None = None,
) -> _Decorator:
    def _impl(fn: _C) -> _C:
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            bound = api._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            flat_args = api._prepare_args(*bound.arguments.values())
            del args, kwargs

            mode = proxy_tensor.get_proxy_mode()
            if mode is None:
                from helion._compiler.compile_environment import CompileEnvironment

                if CompileEnvironment.has_current():
                    assert api._fake_fn is not None
                    return api._fake_fn(*flat_args)
                return fn(*flat_args)
            assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
            tracer = mode.tracer
            assert isinstance(tracer, proxy_tensor.PythonKeyTracer)
            # We hit type errors if we use the regular custom_op overload, instead we
            # intercept the call and fake the custom op.
            with proxy_tensor.disable_proxy_modes_tracing():
                proxy_out = tracer.create_proxy(
                    "call_function",
                    wrapper,
                    *args_to_proxies(tracer, flat_args, {}),
                )
                assert api._fake_fn is not None
                out = api._fake_fn(*flat_args)
                proxy_tensor.track_tensor_tree(
                    out, proxy_out, constant=None, tracer=tracer
                )
            return out

        api: APIFunc = cast("APIFunc", wrapper)
        api._helion_api = True
        api._is_device_loop = is_device_loop
        api._is_device_only = is_device_only
        api._tiles_as_sizes = tiles_as_sizes
        if tiles_as_sizes:
            api._prepare_args = tiles_as_sizes_prepare_args
        else:
            api._prepare_args = no_op_prepare_args
        api._cache_type = cache_type
        api._type_function = None
        api._codegen = None
        api._fake_fn = None
        api._signature = signature or inspect.signature(
            cast("Callable[..., object]", fn)
        )
        return wrapper

    return _impl


def register_fake(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[object]:
    def _impl(fake_fn: Callable[..., object]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{register_fake.__qualname__} can only be used on API functions"
        )
        assert original_fn._fake_fn is None
        original_fn._fake_fn = fake_fn
        if original_fn._type_function is None:
            original_fn._type_function = _default_type_function(
                fake_fn, original_fn._tiles_as_sizes
            )
        return _no_call

    return _impl


def type_propagation(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[TypeInfo]:
    def _impl(type_fn: Callable[..., TypeInfo]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{type_propagation.__qualname__} can only be used on API functions"
        )
        original_fn._type_function = type_fn
        return _no_call

    return _impl


def prepare_args(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[tuple[object, ...]]:
    def _impl(
        prep_fn: Callable[
            ...,
            tuple[object, ...],
        ],
    ) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{type_propagation.__qualname__} can only be used on API functions"
        )
        original_fn._prepare_args = prep_fn
        return _no_call

    return _impl


def codegen(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[object]:
    def _impl(codegen_fn: Callable[[CodegenState], object]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{type_propagation.__qualname__} can only be used on API functions"
        )
        assert original_fn._codegen is None, (
            "codegen can only be used once per function"
        )
        original_fn._codegen = codegen_fn
        return _no_call

    return _impl


def _default_type_function(
    fake_fn: Callable[..., object], tiles_as_sizes: bool
) -> Callable[..., TypeInfo]:
    from .._compiler.tile_index_proxy import TileIndexProxy
    from .._compiler.type_propagation import TypeInfo

    def type_prop_with_fake_fn(
        *args: object, origin: Origin, **kwargs: object
    ) -> TypeInfo:
        args, kwargs = tree_map_only(TypeInfo, _to_proxy, (args, kwargs))
        if tiles_as_sizes:
            args, kwargs = TileIndexProxy.tiles_to_sizes((args, kwargs))
        return TypeInfo.from_example(fake_fn(*args, **kwargs), origin)

    return type_prop_with_fake_fn


def _to_proxy(arg: TypeInfo) -> object:
    try:
        return arg.proxy()
    except NotImplementedError:
        raise exc.TracedArgNotSupported(arg) from None
