from __future__ import annotations

import functools
from types import FunctionType

from torch._dynamo.utils import make_cell

from helion import exc
from helion._compiler.host_function import HostFunction
from helion._compiler.variable_origin import ClosureOrigin
from helion._compiler.variable_origin import Origin


class CaptureGlobals(dict[str, object]):
    def __init__(self, _globals: dict[str, object]) -> None:
        super().__init__(_globals)
        self._globals = _globals

    def __getitem__(self, key: str) -> object:
        if key == "__builtins__":
            return self._globals[key]
        return HostFunction.current().register_fake(
            self._globals[key],
            HostFunction.current().import_from_module(self._globals, key),
        )

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
