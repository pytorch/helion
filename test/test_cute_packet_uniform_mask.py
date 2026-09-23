"""Execute emitted vector loads against the original scalar masked recipe."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import operator
from types import SimpleNamespace
from typing import cast

import pytest

from test.test_cute_contiguous_copy import _body
from test.test_cute_contiguous_copy import _expr
from test.test_cute_contiguous_copy import _fresh
from test.test_cute_contiguous_copy import _i32

from helion._compiler.ast_extension import convert
from helion._compiler.cute.collective_register_chain import _packet_alignment
from helion._compiler.cute.collective_register_chain import _packet_plan
from helion._compiler.cute.contiguous_copy import ContiguousCopy
from helion._compiler.cute.contiguous_copy import CopyTensorFacts
from helion._compiler.cute.contiguous_copy import plan_contiguous_copy


@dataclass
class _Pointer:
    allowed: set[int]
    loads: list[int]
    offset: int = 0

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.allowed, self.loads, self.offset + offset)

    def toint(self) -> _Pointer:
        return self

    def load(self) -> float:
        assert self.offset in self.allowed, "masked element was dereferenced"
        self.loads.append(self.offset)
        return self.offset * 0.125 + 2.0


class _Registers(list[float]):
    def fill(self, value: float) -> None:
        self[:] = [value] * len(self)


def _plan(
    predicate: str,
    *,
    kind: str = "generic",
    width: int = 8,
    dtype: str = "cutlass.Float16",
    alignments: dict[str, int] | None = None,
    negative_zero: bool = False,
    extended: bool = False,
) -> tuple[ContiguousCopy, list[ast.Assign]]:
    original = _body(
        f"value = (A.iterator + cutlass.Int32(k)).load() if {predicate} else {dtype}(0)"
    )
    if negative_zero:
        value = original[0].value
        assert isinstance(value, ast.IfExp) and isinstance(value.orelse, ast.Call)
        value.orelse.args[0] = ast.Constant(-0.0)
    if extended:
        original = [cast("ast.Assign", convert(statement)) for statement in original]
    arguments = {
        "coordinate": "k",
        "tensors": {"A": CopyTensorFacts(dtype, (1,), 16)},
        "aligned_names": {"k": width, **(alignments or {})},
    }
    plan = (
        plan_contiguous_copy(original, _expr("value"), **arguments)
        if kind == "generic"
        else _packet_plan(original, _expr("value"), width=width, **arguments)
    )
    assert plan is not None and plan.width == width
    return plan, original


def _exercise(
    plan: ContiguousCopy,
    original: list[ast.Assign],
    predicate: str,
    parameters: dict[str, int | float | bool],
) -> int:
    width = plan.width
    cutlass = SimpleNamespace(
        Int32=_i32, Float16=float, BFloat16=float, Float32=float, range_constexpr=range
    )
    base = parameters["k"]
    assert type(base) is int
    allowed = {
        _i32(base + lane)
        for lane in range(width)
        if eval(
            predicate,
            {"cutlass": cutlass, "operator": operator, **parameters, "k": base + lane},
        )
    }
    loads: list[int] = []
    tensor = SimpleNamespace(iterator=_Pointer(allowed, loads), element_type=float)
    namespace = {"cutlass": cutlass, "operator": operator, "A": tensor, **parameters}
    scalar = compile(
        ast.fix_missing_locations(ast.Module(body=[*original], type_ignores=[])),
        "scalar",
        "exec",
    )
    expected: list[float] = []
    for lane in range(width):
        values = {**namespace, "k": base + lane}
        exec(scalar, values)
        expected.append(cast("float", values["value"]))
    scalar_loads = list(loads)
    loads.clear()
    copies = []

    def make_ptr(
        dtype: type, pointer: _Pointer, space: str, *, assumed_align: int
    ) -> _Pointer:
        assert dtype is float and space == "gmem"
        assert pointer.offset % width == 0 and assumed_align in (8, 16)
        return pointer

    def copy(source: tuple[_Pointer, tuple[int]], destination: _Registers) -> None:
        pointer, shape = source
        assert shape == (width,) and len(destination) == width
        copies.append(pointer.offset)
        for lane in range(width):
            destination[lane] = (pointer + lane).load()

    namespace["cute"] = SimpleNamespace(
        make_rmem_tensor=lambda shape, dtype: _Registers([float("nan")] * shape[0]),
        make_layout=lambda shape, stride=None: shape,
        make_tensor=lambda pointer, shape: (pointer, shape),
        make_ptr=make_ptr,
        autovec_copy=copy,
        AddressSpace=SimpleNamespace(gmem="gmem"),
    )
    body, name = plan.emit_to_registers(_fresh())
    if width == 4:
        body = _packet_alignment(body, 8)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            "vector",
            "exec",
        ),
        namespace,
    )
    actual = cast("_Registers", namespace[name])
    assert actual == expected
    assert [math.copysign(1.0, value) for value in actual] == [
        math.copysign(1.0, value) for value in expected
    ]
    assert loads == scalar_loads
    return len(copies)


@pytest.mark.parametrize(
    ("kind", "width", "dtype"),
    [
        ("generic", 8, "cutlass.Float16"),
        ("generic", 4, "cutlass.Float32"),
        ("chain", 8, "cutlass.BFloat16"),
        ("chain", 4, "cutlass.Float16"),
    ],
)
@pytest.mark.parametrize(
    ("predicate", "uniform"),
    [
        ("k < limit", True),
        ("k >= limit", True),
        ("limit > k", True),
        ("limit <= k", True),
        ("k <= limit", False),
        ("k > limit", False),
        ("limit >= k", False),
        ("limit < k", False),
    ],
)
def test_aligned_comparison_boundaries_match_scalar_loads(
    kind: str, width: int, dtype: str, predicate: str, uniform: bool
) -> None:
    plan, original = _plan(
        predicate, kind=kind, width=width, dtype=dtype, alignments={"limit": width}
    )
    assert plan._uniform_positive_zero is uniform
    for limit in range(0, 5 * width, width):
        for k in range(0, 5 * width, width):
            _exercise(plan, original, predicate, {"k": k, "limit": limit})


@pytest.mark.parametrize("kind", ["generic", "chain"])
def test_unaligned_dynamic_bound_keeps_every_partial_tail(kind: str) -> None:
    predicate = "k < limit"
    plan, original = _plan(predicate, kind=kind)
    assert not plan._uniform_positive_zero
    for limit in range(33):
        for k in (0, 8, 16, 24, 32):
            _exercise(plan, original, predicate, {"k": k, "limit": limit})


@pytest.mark.parametrize("kind", ["generic", "chain"])
def test_integer_constexpr_bound_and_compound_mask(kind: str) -> None:
    predicate = (
        "active and operator.ge(k, lower) and k < BLOCK + BLOCK * (row // BLOCK)"
    )
    plan, original = _plan(
        predicate, kind=kind, alignments={"lower": 8, "BLOCK": 16, "row": 1}
    )
    assert plan._uniform_positive_zero
    for active in (False, True):
        for row in (0, 16, 32, 48):
            for k in range(0, 72, 8):
                copies = _exercise(
                    plan,
                    original,
                    predicate,
                    {"k": k, "lower": 8, "BLOCK": 16, "row": row, "active": active},
                )
                assert copies == int(active and 8 <= k < row + 16)


@pytest.mark.parametrize("kind", ["generic", "chain"])
def test_float_to_integer_bound_does_not_supply_alignment(kind: str) -> None:
    predicate = "k < cutlass.Int32(scale * 8)"
    plan, original = _plan(predicate, kind=kind)
    assert not plan._uniform_positive_zero
    for scale in (0.125, 1.125, 2.125):
        for k in (0, 8, 16, 24):
            _exercise(plan, original, predicate, {"k": k, "scale": scale})


@pytest.mark.parametrize("kind", ["generic", "chain"])
def test_signed_cast_extremes_preserve_whole_packet_mask(kind: str) -> None:
    predicate = "cutlass.Int32(k) < limit"
    plan, original = _plan(predicate, kind=kind, alignments={"limit": 8})
    assert plan._uniform_positive_zero
    for k in (-(1 << 31), -(1 << 31) - 8, (1 << 31) - 8, 1 << 31):
        for limit in (-(1 << 31), -(1 << 31) + 8, 0, (1 << 31) - 8):
            _exercise(plan, original, predicate, {"k": k, "limit": limit})


@pytest.mark.parametrize("kind", ["generic", "chain"])
def test_negative_zero_inactive_value_is_not_positive_zero_fill(kind: str) -> None:
    predicate = "k < limit"
    plan, original = _plan(
        predicate, kind=kind, alignments={"limit": 8}, negative_zero=True
    )
    assert not plan._uniform_positive_zero
    for k in (0, 8, 16):
        _exercise(plan, original, predicate, {"k": k, "limit": 8})


@pytest.mark.parametrize("kind", ["generic", "chain"])
def test_compiler_extended_comparisons_retain_uniformity_proof(kind: str) -> None:
    predicate = "k < limit"
    plan, original = _plan(predicate, kind=kind, alignments={"limit": 8}, extended=True)
    assert plan._uniform_positive_zero
    for k in (0, 8, 16):
        _exercise(plan, original, predicate, {"k": k, "limit": 8})
