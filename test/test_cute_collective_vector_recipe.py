"""Memory and arithmetic oracles for cooperative register-vector recipes."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import itertools
import operator
import struct
from types import SimpleNamespace
from typing import cast

import pytest

from helion._compiler.cute.collective_vector_recipe import emit_vector_recipe
from helion._compiler.cute.contiguous_copy import CopyTensorFacts


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _i32(value: int) -> int:
    return (int(value) + 2**31) % 2**32 - 2**31


@dataclass
class _Pointer:
    tensor: _Tensor
    offset: int = 0

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.tensor, self.offset + offset)

    def toint(self) -> _Pointer:
        return self

    def load(self) -> int | float:
        assert 0 <= self.offset < len(self.tensor.values)
        self.tensor.loads.append(self.offset)
        return self.tensor.values[self.offset]


class _Tensor:
    def __init__(
        self,
        values: list[int | float],
        shape: tuple[int, ...],
        *,
        offset: int = 0,
    ) -> None:
        self.values = list(values)
        self.loads: list[int] = []
        self.iterator = _Pointer(self, offset)
        self.element_type = float
        self.layout = SimpleNamespace(
            shape=shape, stride=(shape[1], 1) if len(shape) == 2 else (1,)
        )

    def _offset(self, index: tuple[int, ...] | int) -> int:
        if isinstance(index, int):
            index = (index,)
        assert all(
            0 <= value < size
            for value, size in zip(index, self.layout.shape, strict=True)
        )
        return sum(
            value * stride
            for value, stride in zip(index, self.layout.stride, strict=True)
        )

    def __getitem__(self, index: tuple[int, ...] | int) -> int | float:
        return (self.iterator + self._offset(index)).load()

    def __setitem__(self, index: tuple[int, ...] | int, value: float) -> None:
        self.values[self.iterator.offset + self._offset(index)] = value


_RECIPE = """
valid_row = m < count
row = (indices.iterator + m).load() if valid_row else cutlass.Int32(0)
column = cutlass.Int32(origin + k)
active = valid_row and lower <= column and column < upper
a = (A.iterator + cutlass.Int32(row) * cutlass.Int32(A.layout.stride[0]) + column).load() if active else cutlass.Float16(0.0)
scale = (scales.iterator + row).load() if valid_row else cutlass.Float16(1.0)
b = (B.iterator + cutlass.Int32(row) * cutlass.Int32(B.layout.stride[0]) + column).load() if active else cutlass.Float16(0.0)
product = cutlass.Float32(a) * cutlass.Float32(scale)
value = cutlass.Float16(product + cutlass.Float32(b))
"""


def _run_recipe(
    source: str,
    *,
    optimized: bool,
    width: int,
    m: int,
    k: int,
    origin: int,
    lower: int,
    upper: int,
    alignment: int,
    dtype: str,
    mixed_dtype: bool = False,
) -> tuple[list[int | float], dict[str, set[int]], int]:
    offset = 0 if alignment == 16 else 1
    tensors = {
        "A": _Tensor(
            [0.25 * (index - 43) for index in range(97)], (4, 24), offset=offset
        ),
        "B": _Tensor(
            [0.5 * (index - 29) for index in range(97)], (4, 24), offset=offset
        ),
        "indices": _Tensor([3, 1], (2,)),
        "scales": _Tensor([2.0, -4.0, 0.5, -1.0], (4,)),
    }
    destination = _Tensor([-999.0] * 48, (3, 16))
    vector_copies = 0
    transaction_width = 4 if dtype == "cutlass.Float32" else 8

    def make_ptr(
        element_type: type, pointer: _Pointer, space: str, *, assumed_align: int
    ) -> _Pointer:
        assert element_type is float and space == "gmem" and assumed_align == 16
        assert pointer.offset % transaction_width == 0
        return pointer

    def autovec_copy(source_tensor: SimpleNamespace, registers: _Tensor) -> None:
        nonlocal vector_copies
        vector_copies += 1
        assert source_tensor.layout.shape == (transaction_width,)
        for lane in range(transaction_width):
            registers[lane] = (source_tensor.iterator + lane).load()

    namespace = {
        **tensors,
        "shared": destination,
        "m": m,
        "k": k,
        "count": 2,
        "origin": origin,
        "lower": lower,
        "upper": upper,
        "operator": operator,
        "cutlass": SimpleNamespace(
            Float16=float,
            BFloat16=float,
            Float32=float,
            Int32=_i32,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            make_ptr=make_ptr,
            make_layout=lambda shape, stride=None: SimpleNamespace(
                shape=shape, stride=stride
            ),
            make_tensor=lambda pointer, layout: SimpleNamespace(
                iterator=pointer, layout=layout
            ),
            make_rmem_tensor=lambda layout, element_type: _Tensor(
                [float("nan")] * layout.shape[0], layout.shape
            ),
            AddressSpace=SimpleNamespace(gmem="gmem"),
            autovec_copy=autovec_copy,
        ),
    }
    statements = cast("list[ast.Assign]", ast.parse(source).body)
    if optimized:
        counter = itertools.count()
        generated = emit_vector_recipe(
            statements,
            _expr("value"),
            coordinate="k",
            width=width,
            tensors={
                name: CopyTensorFacts(
                    "cutlass.Float32" if mixed_dtype and name == "B" else dtype,
                    (24, 1),
                    alignment,
                )
                for name in ("A", "B")
            },
            aligned_names={"k": width, "origin": width},
            destination="shared",
            destination_indices=(_expr("m"), _expr("k")),
            fresh_name=lambda hint: f"vector_{next(counter)}_{hint}",
        )
        exec(
            compile(ast.Module(body=generated, type_ignores=[]), "<vector>", "exec"),
            namespace,
        )
    else:
        code = compile(ast.Module(body=statements, type_ignores=[]), "<scalar>", "exec")
        for lane in range(width):
            namespace["k"] = k + lane
            exec(code, namespace)
            destination[m, k + lane] = namespace["value"]
    return (
        destination.values,
        {name: set(tensor.loads) for name, tensor in tensors.items()},
        vector_copies,
    )


@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("alignment", [2, 16])
@pytest.mark.parametrize(
    "dtype,width",
    [("cutlass.Float16", 8), ("cutlass.BFloat16", 8), ("cutlass.Float32", 4)],
)
def test_vector_recipe_matches_scalar_values_and_memory_reads(
    m: int, alignment: int, dtype: str, width: int
) -> None:
    source = _RECIPE.replace("cutlass.Float16", dtype)
    for origin, lower, upper, k in (
        (0, 0, 24, 0),
        (0, 0, 24, 8),
        (8, 0, 24, 8),
        (0, 2, 6, 0),
        (0, 0, 5, 0),
        (-8, 0, 7, 0),
        (-8, 0, 7, 8),
        (2**32 - 8, 0, 7, 8),
        (0, 1, 1, 0),
    ):
        arguments = {
            "width": width,
            "m": m,
            "k": k,
            "origin": origin,
            "lower": lower,
            "upper": upper,
            "alignment": alignment,
            "dtype": dtype,
        }
        expected, expected_reads, scalar_copies = _run_recipe(
            source, optimized=False, **arguments
        )
        actual, actual_reads, vector_copies = _run_recipe(
            source, optimized=True, **arguments
        )
        assert [struct.pack("d", value) for value in actual] == [
            struct.pack("d", value) for value in expected
        ]
        assert actual_reads == expected_reads
        assert scalar_copies == 0
        if alignment == 2 or m == 2:
            assert vector_copies == 0
        elif origin == 0 and lower == 0 and upper == 24:
            assert vector_copies == 2


def test_mixed_width_loads_keep_scalar_fallback() -> None:
    arguments = {
        "width": 8,
        "m": 1,
        "k": 0,
        "origin": 0,
        "lower": 0,
        "upper": 24,
        "alignment": 16,
        "dtype": "cutlass.Float16",
        "mixed_dtype": True,
    }
    expected, reads, scalar_copies = _run_recipe(_RECIPE, optimized=False, **arguments)
    actual, actual_reads, vector_copies = _run_recipe(
        _RECIPE, optimized=True, **arguments
    )
    assert actual == expected and actual_reads == reads
    assert scalar_copies == 0 and vector_copies == 1


def test_masked_negative_zero_keeps_its_sign() -> None:
    source = _RECIPE[: _RECIPE.index("scale =")] + "value = a\n"
    source = source.replace("cutlass.Float16(0.0)", "cutlass.Float16(-0.0)")
    arguments = {
        "width": 8,
        "m": 1,
        "k": 0,
        "origin": 0,
        "lower": 2,
        "upper": 6,
        "alignment": 16,
        "dtype": "cutlass.Float16",
    }
    expected, reads, scalar_copies = _run_recipe(source, optimized=False, **arguments)
    actual, actual_reads, vector_copies = _run_recipe(
        source, optimized=True, **arguments
    )
    assert [struct.pack("d", value) for value in actual] == [
        struct.pack("d", value) for value in expected
    ]
    assert actual_reads == reads
    assert scalar_copies == vector_copies == 0
