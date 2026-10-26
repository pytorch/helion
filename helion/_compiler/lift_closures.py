from __future__ import annotations

import functools
from types import FunctionType
import weakref

from torch._dynamo.utils import make_cell
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

from .. import exc
from .host_function import HostFunction
from .variable_origin import ClosureOrigin
from .variable_origin import Origin


class CaptureGlobals(dict[str, object]):
    """Globals of a lifted function: every read registers the value as a fake.

    A global is registered once per host function and per bound object, like
    a closure cell is registered once at lift time.  The lifted function runs
    again while the device body is traced; re-registering there would create
    a fresh fake under the tracer, which records its allocation as a device
    ``empty_strided`` node that carries no host origin (the lowering then
    cannot name the tensor), so a value first seen under tracing is
    registered with proxy tracing disabled.
    """

    def __init__(self, _globals: dict[str, object]) -> None:
        super().__init__(_globals)
        self._globals = _globals
        self._fakes: weakref.WeakKeyDictionary[
            HostFunction, dict[str, tuple[object, object]]
        ] = weakref.WeakKeyDictionary()

    def __getitem__(self, key: str) -> object:
        if key == "__builtins__":
            return self._globals[key]
        value = self._globals[key]
        host_function = HostFunction.current()
        fakes = self._fakes.setdefault(host_function, {})
        cached = fakes.get(key)
        if cached is not None and cached[0] is value:
            return cached[1]
        with disable_proxy_modes_tracing():
            fake = host_function.register_fake(
                value, host_function.import_from_module(self._globals, key)
            )
        fakes[key] = (value, fake)
        return fake

    def __delitem__(self, key: str) -> None:
        raise exc.GlobalMutation(key)

    def __setitem__(self, key: str, value: object) -> None:
        raise exc.GlobalMutation(key)


def lift_closures(func: FunctionType, origin: Origin) -> FunctionType:
    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        nonlocal new_func, closure_contents
        if new_func is None:
            host_function = HostFunction.current()
            closure = None
            if func.__closure__ is not None:
                closure_contents = [
                    host_function.register_fake(
                        obj.cell_contents, ClosureOrigin(origin, i)
                    )
                    for i, obj in enumerate(func.__closure__)
                ]
                closure = (*map(make_cell, closure_contents),)
            new_func = FunctionType(
                code=func.__code__,
                globals=(CaptureGlobals(func.__globals__)),
                name=func.__name__,
                argdefs=func.__defaults__,
                closure=closure,
            )
        result = new_func(*args, **kwargs)
        if closure_contents:
            for cell, expected, varname in zip(
                new_func.__closure__ or (),
                closure_contents,
                new_func.__code__.co_freevars,
                strict=True,
            ):
                if cell.cell_contents is not expected:
                    raise exc.ClosureMutation(varname)
        return result

    new_func: FunctionType | None = None
    closure_contents: list[object] = []
    return wrapper
