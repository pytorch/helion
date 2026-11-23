"""A pointer model for the affine lane vectorizer's tests.

``rewrite`` runs ``affine_vector_io._Vectorizer`` over the lane loops of a
kernel written as generated CuTe source (``SOURCE``: four fp32 lanes loaded,
scaled and stored under a tail mask), and ``execute`` runs the result on
Python lists behind ``Pointer`` objects that count scalar and packet loads
and stores, so a test compares the stored values and the access counts of
the rewritten loop with the original's.  ``Float32``, ``Uint32``, ``Tensor``,
``vector_load`` and ``vector_store`` stand in for the DSL's types and packet
helpers; the tests of other lane passes build their own namespaces from
them.
"""

from __future__ import annotations

import ast
import struct
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

from .._compiler.cute.affine_vector_io import _Vectorizer

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


class Float32(float):
    def bitcast(self, target: type) -> Uint32:
        assert target is Uint32
        return Uint32(struct.unpack("I", struct.pack("f", self))[0])


class Uint32(int):
    mlir_type: str

    def bitcast(self, target: type) -> Float32:
        assert target is Float32
        return Float32(struct.unpack("f", struct.pack("I", self))[0])


class Pointer:
    def __init__(
        self, values: list[float], stats: dict[str, int], offset: int = 0
    ) -> None:
        self.values = values
        self.offset = offset
        self.stats = stats

    def __add__(self, offset: int) -> Pointer:
        # Pointer arithmetic is 64-bit: a (modelled) Int32 offset is
        # sign-extended and no longer wraps once it is added.
        return Pointer(self.values, self.stats, self.offset + int(offset))

    def load(self) -> Float32:
        assert 0 <= self.offset < len(self.values)
        self.stats["scalar_loads"] += 1
        return Float32(self.values[self.offset])

    def store(self, value: float) -> None:
        assert 0 <= self.offset < len(self.values)
        self.stats["scalar_stores"] += 1
        self.values[self.offset] = float(value)


class Tensor(SimpleNamespace):
    def __getitem__(self, index: int) -> Float32:
        return (self.iterator + index).load()


def vector_load(
    pointer: Pointer, vector_type: tuple[list[int], object]
) -> list[Uint32]:
    pointer.stats["vector_loads"] += 1
    width = vector_type[0][0]
    assert pointer.offset >= 0 and pointer.offset + width <= len(pointer.values)
    # A packet is loaded from its own boundary only.
    assert pointer.offset % width == 0
    return [
        Float32(value).bitcast(Uint32)
        for value in pointer.values[pointer.offset : pointer.offset + width]
    ]


def vector_store(pointer: Pointer, values: Sequence[Uint32]) -> None:
    pointer.stats["vector_stores"] += 1
    assert pointer.offset >= 0 and pointer.offset + len(values) <= len(pointer.values)
    # ... and stored to it: what a store tier's guard has to establish.
    assert pointer.offset % len(values) == 0
    for index, value in enumerate(values):
        pointer.values[pointer.offset + index] = float(value.bitcast(Float32))


# Model the generated Int32 coordinates explicitly; unknown scalar arithmetic
# is tested separately and must not inherit an integer type from Python inputs.
SOURCE = """
def kernel(src, out, start, length):
    for tile_offset_1 in range(0, cutlass.Int32(length), 16):
        for lane_1 in range(4):
            lane_base_1 = cutlass.Int32(tile_offset_1) + cutlass.Int32(lane_1) * 4
            for vec_lane_1 in cutlass.range_constexpr(4):
                indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
                mask_1 = indices_1 < cutlass.Int32(length)
                address = cutlass.Int32(start) + indices_1
                value = (src.iterator + cutlass.Int32(address) * cutlass.Int32(src.layout.stride[0])).load() if mask_1 else cutlass.Float32(0)
                result = value * 1.25 + 2
                if mask_1:
                    (out.iterator + cutlass.Int32(address) * cutlass.Int32(out.layout.stride[0])).store(cutlass.Float32(result))
"""


def rewrite(
    source: str,
    *,
    aligned: bool = True,
    disjoint: bool = True,
    integer_tensors: frozenset[str] = frozenset(),
    strides: dict[tuple[str, int], int] | None = None,
    builtin_abs: bool = True,
    store_tiers: bool = False,
) -> tuple[ast.Module, int]:
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    rewriter = _Vectorizer(
        function.body,
        strides=strides if strides is not None else {("src", 0): 1, ("out", 0): 1},
        alignments={"src": 16 if aligned else 1, "out": 16},
        dtypes={"src": "cutlass.Float32", "out": "cutlass.Float32"},
        disjoint={frozenset(("src", "out"))} if disjoint else set(),
        constexpr={},
        integer_tensors=integer_tensors,
        unshadowed_abs=builtin_abs
        and all(arg.arg != "abs" for arg in function.args.args),
        store_tiers=store_tiers,
    )
    function.body = rewriter.body(function.body)
    return ast.fix_missing_locations(tree), rewriter.changed


def execute(
    tree: ast.Module,
    start: int,
    length: int,
    *,
    int32: Callable[..., Any] = int,
    int64: Callable[..., Any] = int,
    extra_args: tuple[object, ...] = (),
    out_stride: tuple[int, ...] = (1,),
) -> tuple[list[float], dict[str, int]]:
    stats = dict.fromkeys(
        ("scalar_loads", "scalar_stores", "vector_loads", "vector_stores"), 0
    )
    data = [float(value) for value in range(80)]
    output = [-123.0] * 80
    src = Tensor(iterator=Pointer(data, stats), layout=SimpleNamespace(stride=(1,)))
    out = Tensor(
        iterator=Pointer(output, stats), layout=SimpleNamespace(stride=out_stride)
    )
    namespace: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            Int32=int32,
            Int64=int64,
            Float32=Float32,
            Uint32=Uint32,
            Boolean=bool,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(arch=SimpleNamespace(load=vector_load)),
        "ir": SimpleNamespace(
            VectorType=SimpleNamespace(get=lambda shape, dtype: (shape, dtype))
        ),
        "_cute_store_u32_vec": vector_store,
    }
    Uint32.mlir_type = "i32"
    exec(compile(tree, "<affine-lane-check>", "exec"), namespace)
    namespace["kernel"](src, out, start, length, *extra_args)
    return output, stats
