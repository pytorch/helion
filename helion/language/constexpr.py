from __future__ import annotations

from typing import NamedTuple


class ConstExpr(NamedTuple):
    """
    Typically used as a type annotation for kernels:

        @helion.kernel()
        def fn(v: hl.constexpr, ...):
            ...

    Causes the generated code to specialize on the value of `v`, where a different
    kernel, hardcoding the value of v, will be generated every time `v` changes.
    """

    value: object

    def __index__(self) -> int:
        if isinstance(self.value, int):
            return self.value
        raise TypeError(f"ConstExpr cannot be indexed: {self.value}")

    def __bool__(self) -> bool:
        return bool(self.value)
