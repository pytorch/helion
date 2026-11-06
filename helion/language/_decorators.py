from __future__ import annotations

import contextlib
import functools
import inspect
import threading
from typing import TYPE_CHECKING
from typing import Any
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

from .. import exc

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.type_info import TypeInfo
    from .._compiler.variable_origin import Origin

    _T = TypeVar("_T")
    _C = TypeVar("_C", bound=Callable[..., object])

    class _Decorator(Protocol):
        def __call__(self, fn: _C) -> _C: ...

    class _NoReturnDecorator(Protocol, Generic[_T]):
        def __call__(self, fn: Callable[..., _T]) -> object: ...


_CodegenT = TypeVar("_CodegenT")
_CODEGEN_REPAIR_STATE = threading.local()


class _CodegenRecording(threading.local):
    """Per-thread stack of active plugin loads' recorders."""

    def __init__(self) -> None:  # runs once per thread
        self.stack: list[list[tuple[dict[str, Any], str, object, object]]] = []


_CODEGEN_RECORDING = _CodegenRecording()
_NO_PREVIOUS_HANDLER = object()


@contextlib.contextmanager
def _record_codegen_registrations() -> Iterator[
    list[tuple[dict[str, Any], str, object, object]]
]:
    """Collect ``(implementations, backend, previous, handler)`` per registration.

    Nests: a registration is recorded by every active recorder on this thread.
    Recording is thread-local: a plugin that spawns threads during import has
    their registrations recorded nowhere, so they survive a rollback of the
    plugin's load.
    """
    recorded: list[tuple[dict[str, Any], str, object, object]] = []
    _CODEGEN_RECORDING.stack.append(recorded)
    try:
        yield recorded
    finally:
        _CODEGEN_RECORDING.stack.pop()


def _forget_recorded_registrations(
    entries: list[tuple[dict[str, Any], str, object, object]],
) -> None:
    """Drop ``entries`` from every enclosing recorder (a nested load succeeded)."""
    ids = {id(entry) for entry in entries}
    for recorded in _CODEGEN_RECORDING.stack:
        recorded[:] = [entry for entry in recorded if id(entry) not in ids]


def _codegen_repair_in_progress() -> bool:
    return getattr(_CODEGEN_REPAIR_STATE, "codegen_names", None) is not None


def _begin_codegen_repair(codegen_names: frozenset[str]) -> None:
    assert not _codegen_repair_in_progress()
    _CODEGEN_REPAIR_STATE.codegen_names = codegen_names
    _CODEGEN_REPAIR_STATE.registrations = set()


def _end_codegen_repair() -> None:
    del _CODEGEN_REPAIR_STATE.codegen_names
    del _CODEGEN_REPAIR_STATE.registrations


def _register_codegen_handler(
    implementations: dict[str, _CodegenT], backend: str, handler: _CodegenT
) -> None:
    codegen_names: frozenset[str] | None = getattr(
        _CODEGEN_REPAIR_STATE, "codegen_names", None
    )
    # Explicit raises (not asserts) so -O cannot let a handler be overwritten.
    if codegen_names is None:
        if backend in implementations:
            raise AssertionError(f"codegen already registered for backend {backend!r}")
    else:
        if backend not in codegen_names:
            return
        key = (id(implementations), backend)
        registrations: set[tuple[int, str]] = _CODEGEN_REPAIR_STATE.registrations
        if key in registrations:
            raise AssertionError(f"codegen already registered for backend {backend!r}")
        registrations.add(key)
    stack = _CODEGEN_RECORDING.stack
    if stack:
        previous = dict.get(implementations, backend, _NO_PREVIOUS_HANDLER)
        # The *same* tuple object is appended to every active recorder;
        # _forget_recorded_registrations matches entries by id(), so do not
        # copy the tuple per list (a copy would break nested-load rollback).
        entry = (implementations, backend, previous, handler)
        for recorded in stack:
            recorded.append(entry)
    implementations[backend] = handler


class BackendDispatchTable(dict[str, _CodegenT]):
    """Codegen handlers keyed by a backend's ``codegen_name``, with the
    ``"common"`` fallback.

    ``table[name]`` first lets ``repair_backend_codegen`` re-register a
    backend whose codegen modules a reload dropped (on a hit too: a partial
    module may have registered a handler before defining helpers it calls),
    then returns the backend's handler or the ``"common"`` one; a miss is a
    ``KeyError`` (``lookup`` raises ``BackendImplementationMissing`` instead).
    Both codegen registries -- ``APIFunc._codegen`` and
    ``AtenLowering.codegen_impls`` -- are one of these.
    """

    def __getitem__(self, key: str) -> _CodegenT:
        if key != "common":
            from .._compiler.backend_registry import repair_backend_codegen

            repair_backend_codegen(key)
        return dict.__getitem__(self, key)

    def __missing__(self, key: str) -> _CodegenT:
        if key != "common" and "common" in self:
            return dict.__getitem__(self, "common")
        raise KeyError(key)

    # pyrefly: ignore[bad-override]
    def get(self, key: str, default: _CodegenT | None = None) -> _CodegenT | None:
        try:
            return self[key]
        except KeyError:
            return default

    def lookup(self, codegen_name: str, *, backend_name: str, what: str) -> _CodegenT:
        """The handler for ``codegen_name``; ``BackendImplementationMissing``
        naming ``backend_name`` and ``what`` when there is none."""
        try:
            return self[codegen_name]
        except KeyError:
            raise exc.BackendImplementationMissing(backend_name, what) from None


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
        _codegen: Mapping of backend names to callables that generate device code.
        _fake_fn: A callable that provides a "fake" implementation used during
            tracing and compilation.
        _prepare_args: A callable that preprocesses the arguments before they're
            passed to the actual function implementation.
        _post_create_proxy: An optional callback invoked with the newly created FX
            node and prepared arguments.
        _get_masked_value: A callable that retrieves the masked value for a node,
        _signature: The function signature for binding and validating arguments.
    """

    __name__: str
    __qualname__: str
    _helion_api: Literal[True]
    # a device loop can transition between host and device code
    _is_device_loop: bool
    _is_device_only: bool
    _tiles_as_sizes: bool
    _cache_type: bool
    _type_function: Callable[..., TypeInfo] | None
    _codegen: BackendDispatchTable[Callable[[CodegenState], object]]
    _fake_fn: Callable[..., object] | None
    _prepare_args: Callable[[tuple[object, ...]], tuple[object, ...]]
    _post_create_proxy: Callable[..., None] | None
    _get_masked_value: Callable[[torch.fx.Node], float | bool | None] | None
    _to_device_ir: Callable[..., object] | None
    _allow_host_tensor: bool
    _signature: inspect.Signature
    _ref_fn: Callable[..., object] | None

    def __call__(self, *args: object, **kwargs: object) -> object: ...


def _no_call(*args: object, **kwargs: object) -> Never:
    raise TypeError("type_prop/codegen functions cannot be called directly")


def is_api_func(fn: object) -> TypeGuard[APIFunc]:
    return getattr(fn, "_helion_api", False)


def args_to_proxies(
    tracer: proxy_tensor.PythonKeyTracer | proxy_tensor._GraphAppendingTracerEx,
    args: _T,
    kwargs: dict[str, object] | None = None,
) -> tuple[_T, dict[str, object]]:
    def unpack(x: object) -> object:
        if isinstance(x, (torch.Tensor, torch.SymInt, torch.SymBool, torch.SymFloat)):
            return unpack(proxy_tensor.get_proxy_slot(x, tracer=tracer))
        if isinstance(x, proxy_tensor._ProxyTensor):
            return x.proxy
        if isinstance(x, Thunk):
            return x.force()
        if isinstance(x, slice):
            # pytree treats a slice as a leaf; its symbolic bounds need proxies.
            return slice(unpack(x.start), unpack(x.stop), unpack(x.step))
        return x

    return tree_map(
        unpack,
        (args, kwargs or {}),
    )


def tiles_as_sizes_prepare_args(*args: object) -> tuple[object, ...]:
    from .tile_proxy import Tile

    return Tile._tiles_to_sizes(args)


def no_op_prepare_args(*args: object) -> tuple[object, ...]:
    return args


def api(
    *,
    is_device_loop: bool = False,
    is_device_only: bool = True,
    tiles_as_sizes: bool = False,
    cache_type: bool = False,
    allow_host_tensor: bool = False,
    signature: inspect.Signature | None = None,
) -> _Decorator:
    def _impl(fn: _C) -> _C:
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            from ..runtime.ref_mode import is_in_ref_mode_context

            bound = api._signature.bind(*args, **kwargs)
            bound.apply_defaults()

            if is_in_ref_mode_context():
                assert api._ref_fn is not None, (
                    f"{fn.__qualname__} does not have a ref mode implementation yet"
                )
                flat_args = api._prepare_args(*bound.arguments.values())
                from ..runtime.ref_mode import dispatch_reference

                return dispatch_reference(wrapper, api._ref_fn, flat_args)

            flat_args = api._prepare_args(*bound.arguments.values())

            mode = proxy_tensor.get_proxy_mode()
            if mode is None:
                from .._compiler.compile_environment import CompileEnvironment

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
                # Use _to_device_ir if available, otherwise use _fake_fn with proxy creation
                if api._to_device_ir is not None:
                    out = api._to_device_ir(tracer, *flat_args)
                else:
                    proxy_out = tracer.create_proxy(
                        "call_function",
                        wrapper,
                        *args_to_proxies(tracer, flat_args, {}),
                    )
                    if api._post_create_proxy is not None:
                        api._post_create_proxy(proxy_out.node, *flat_args)
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
        api._codegen = BackendDispatchTable()
        api._fake_fn = None
        api._post_create_proxy = None
        api._get_masked_value = None
        api._to_device_ir = None
        api._allow_host_tensor = allow_host_tensor
        api._signature = signature or inspect.signature(
            cast("Callable[..., object]", fn)
        )
        api._ref_fn = None
        # pyrefly: ignore [bad-return]
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

    # pyrefly: ignore [bad-return]
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

    # pyrefly: ignore [bad-return]
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

    # pyrefly: ignore [bad-return]
    return _impl


def post_create_proxy(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[None]:
    """Register a callback that annotates an API call's newly created FX node."""

    def _impl(callback: Callable[..., None]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{post_create_proxy.__qualname__} can only be used on API functions"
        )
        assert original_fn._post_create_proxy is None
        original_fn._post_create_proxy = callback
        return _no_call

    # pyrefly: ignore [bad-return]
    return _impl


def codegen(
    original_fn: Callable[..., object],
    backend: str,
) -> _NoReturnDecorator[object]:
    def _impl(codegen_fn: Callable[[CodegenState], object]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{type_propagation.__qualname__} can only be used on API functions"
        )
        _register_codegen_handler(original_fn._codegen, backend, codegen_fn)
        return _no_call

    # pyrefly: ignore [bad-return]
    return _impl


def get_masked_value(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[object]:
    def _impl(
        mask_value_fn: Callable[[torch.fx.Node], float | bool | None],
    ) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{type_propagation.__qualname__} can only be used on API functions"
        )
        assert original_fn._get_masked_value is None, (
            "get_masked_value can only be used once per function"
        )
        original_fn._get_masked_value = mask_value_fn
        return _no_call

    # pyrefly: ignore [bad-return]
    return _impl


def register_to_device_ir(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[object]:
    def _impl(to_device_ir_fn: Callable[..., object]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{register_to_device_ir.__qualname__} can only be used on API functions"
        )
        assert original_fn._to_device_ir is None
        original_fn._to_device_ir = to_device_ir_fn
        return _no_call

    # pyrefly: ignore [bad-return]
    return _impl


def ref(
    original_fn: Callable[..., object],
) -> _NoReturnDecorator[object]:
    def _impl(ref_fn: Callable[..., object]) -> Callable[..., Never]:
        assert is_api_func(original_fn), (
            f"{ref.__qualname__} can only be used on API functions"
        )
        assert original_fn._ref_fn is None, (
            "ref mode implementation can only be registered once per function"
        )
        original_fn._ref_fn = ref_fn
        return _no_call

    # pyrefly: ignore [bad-return]
    return _impl


def _default_type_function(
    fake_fn: Callable[..., object], tiles_as_sizes: bool
) -> Callable[..., TypeInfo]:
    def type_prop_with_fake_fn(
        *args: object, origin: Origin, **kwargs: object
    ) -> TypeInfo:
        from .._compiler.type_info import TypeInfo
        from .tile_proxy import Tile

        args, kwargs = tree_map_only(TypeInfo, _to_proxy, (args, kwargs))
        if tiles_as_sizes:
            args, kwargs = Tile._tiles_to_sizes((args, kwargs))
        return TypeInfo.from_example(fake_fn(*args, **kwargs), origin)

    return type_prop_with_fake_fn


def _to_proxy(arg: TypeInfo) -> object:
    try:
        return arg.proxy()
    except NotImplementedError:
        raise exc.TracedArgNotSupported(arg) from None


# Tracks 1-1 mapping between Python functions and their Helion API counterparts within device function.
_DEVICE_FUNC_REPLACEMENTS: dict[object, Callable[..., object]] = {}
_TENSOR_METHOD_REPLACEMENTS = frozenset({"chunk", "unbind"})


def device_func_replacement(python_func: object) -> _Decorator:
    def _impl(fn: _C) -> _C:
        assert callable(fn)
        _DEVICE_FUNC_REPLACEMENTS[python_func] = fn
        return fn

    return _impl


def get_device_func_replacement(func: object) -> Callable[..., object] | None:
    return _DEVICE_FUNC_REPLACEMENTS.get(func)
