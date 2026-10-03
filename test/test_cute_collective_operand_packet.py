from __future__ import annotations

import ast
from collections import Counter
import copy
import itertools
import math
import struct
from types import SimpleNamespace
from typing import Any

import pytest

from helion._compiler.cute.collective_operand_packet import (
    plan_collective_operand_packet,
)
from helion._compiler.cute.collective_vector_recipe import emit_vector_recipe


def _plan(**changes: Any):
    return plan_collective_operand_packet(
        **(
            {
                "dtype": "cutlass.TFloat32",
                "native": True,
                "bm": 128,
                "bn": 64,
                "bk": 16,
                "operand_a": True,
                "k_major": False,
                "coordinate_k": True,
                "stages": 1,
                "width": 4,
                "base_alignment": 1024,
            }
            | changes
        )
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype": "cutlass.Float16"},
        {"dtype": "cutlass.BFloat16"},
        {"dtype": "cutlass.Float32"},
        {"stages": 2},
        {"stages": 0},
        {"width": 8},
        {"width": 3},
        {"base_alignment": 16},
        {"base_alignment": 1025},
        {"bm": 32},
        {"bm": 256},
        {"bn": 16},
        {"bn": 128},
        {"bk": 8},
        {"bk": 48},
        {"coordinate_k": False},
        {"operand_a": False},
        {"operand_a": False, "k_major": True, "coordinate_k": False},
    ],
)
def test_unknown_or_noncontiguous_operand_packets_decline(
    changes: dict[str, Any],
) -> None:
    assert _plan(**changes) is None


def _size(shape: Any) -> int:
    return (
        math.prod(_size(item) for item in shape)
        if isinstance(shape, tuple)
        else int(shape)
    )


def _offset(index: int, shape: Any, stride: Any) -> int:
    if not isinstance(shape, tuple):
        return index * int(stride)
    result = 0
    for extent, step in zip(shape, stride, strict=True):
        result += _offset(index % _size(extent), extent, step)
        index //= _size(extent)
    assert index == 0
    return result


def _swizzle(address: int, bits: int, base: int, shift: int) -> int:
    return address ^ (((address >> (base + shift)) & ((1 << bits) - 1)) << base)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bm", [32, 64, 128])
@pytest.mark.parametrize("bn", [32, 64])
@pytest.mark.parametrize("bk", [16, 32, 64, 128])
def test_actual_sdk_operand_packet_coordinates_and_bounds(
    native: bool, bm: int, bn: int, bk: int
) -> None:
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    ir = pytest.importorskip("cutlass._mlir.ir")
    if native and bm == 32:
        assert _plan(native=native, bm=bm, bn=bn, bk=bk) is None
        return
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            for operand_a, k_major in ((True, False), (False, False), (False, True)):
                mn = bm if operand_a else bn
                contiguous_k = operand_a or k_major
                extent = bk if contiguous_k else mn
                assert (
                    _plan(
                        native=native,
                        bm=bm,
                        bn=bn,
                        bk=bk,
                        operand_a=operand_a,
                        k_major=k_major,
                        coordinate_k=contiguous_k,
                    )
                    is not None
                )
                if native:
                    kind_name = (
                        f"K_SW{min(128, 4 * bk)}" if contiguous_k else "MN_SW128_32B"
                    )
                    kind = getattr(cute.nvgpu.tcgen05.SmemLayoutAtomKind, kind_name)
                    layout = cute.tile_to_shape(
                        cute.nvgpu.tcgen05.make_smem_layout_atom(
                            kind, cutlass.TFloat32
                        ),
                        (mn, bk),
                        order=(0, 1) if contiguous_k else (1, 0),
                    )
                    outer, byte_swizzle = layout.outer, layout.inner
                else:
                    layout = cute.make_composed_layout(
                        cute.make_swizzle(min(3, (extent // 8).bit_length() - 1), 3, 3),
                        0,
                        cute.make_layout(
                            (mn, bk), stride=(bk, 1) if contiguous_k else (1, mn)
                        ),
                    )
                    outer = layout.outer
                    byte_swizzle = cute.recast_layout(8, 32, layout).inner
                shape, stride = outer.shape, outer.stride
                bits = byte_swizzle.num_bits
                base = byte_swizzle.num_base
                shift = byte_swizzle.num_shift
                assert base >= 4 and shift > 0
                swizzle_layout = cute.make_composed_layout(
                    byte_swizzle, 0, cute.make_layout((262144,), stride=(1,))
                )
                writers: Counter[tuple[int, int]] = Counter()
                wrong_without_swizzle = 0
                for flat in range(0, mn * bk, 4):
                    coordinates = [
                        (flat // bk, flat % bk + lane)
                        if contiguous_k
                        else (flat % mn + lane, flat // mn)
                        for lane in range(4)
                    ]
                    offsets = [
                        sum(
                            itertools.starmap(
                                _offset, zip(coord, shape, stride, strict=True)
                            )
                        )
                        for coord in coordinates
                    ]
                    assert offsets[0] % 4 == 0
                    assert offsets == list(range(offsets[0], offsets[0] + 4))
                    for allocation in (0, 1024, 2048, 3072):
                        addresses = [
                            _swizzle(allocation + 4 * value, bits, base, shift)
                            for value in offsets
                        ]
                        assert addresses[0] % 16 == 0
                        assert addresses == list(
                            range(addresses[0], addresses[0] + 16, 4)
                        )
                        assert all(
                            allocation <= value < allocation + mn * bk * 4
                            for value in addresses
                        )
                    writers.update(coordinates)
                    wrong_without_swizzle += sum(
                        _swizzle(4 * value, bits, base, shift) != 4 * value
                        for value in offsets
                    )
                    # Validate the independent integer evaluator against actual
                    # SDK static indexing at every physical packet boundary.
                    assert int(outer(coordinates[0])) == offsets[0]
                    assert int(outer(coordinates[-1])) == offsets[-1]
                    assert int(swizzle_layout(1024 + 4 * offsets[0])) == _swizzle(
                        1024 + 4 * offsets[0], bits, base, shift
                    )
                assert set(writers.values()) == {1}
                assert set(writers) == {(m, k) for m in range(mn) for k in range(bk)}
                assert wrong_without_swizzle
                # Dropping swizzle or starting at lane one is not a packet proof.
                assert _swizzle(4, bits, base, shift) % 16 != 0


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _emit(packet: bool) -> list[ast.stmt]:
    counter = itertools.count()
    statements = ast.parse("""
valid = m < rows and 0 <= k and k < columns and (active_mask & (1 << k)) != 0
value = (source.iterator + m * pitch + k).load() if valid else cutlass.Float32(-0.0)
term = cutlass.Float32(value / divisor) if valid and divisor != 0 else value
""").body
    return emit_vector_recipe(
        statements,
        _expr(
            "cutlass.TFloat32(cutlass.Uint32(cute.arch.cvt_f32_tf32(term)).bitcast(cutlass.Float32))"
        ),
        coordinate="k",
        width=4,
        tensors={},
        aligned_names={"k": 4},
        destination="shared",
        destination_indices=(_expr("m"), _expr("k")),
        fresh_name=lambda hint: f"{hint}_{next(counter)}",
        shared_packet=_plan() if packet else None,
    )


def test_packet_recovers_entire_scalar_recipe_and_exact_rna_expression() -> None:
    scalar = _emit(False)
    packet = _emit(True)
    assert len(packet) == len(scalar) + 3
    recovered = copy.deepcopy(packet)
    original_loop = next(node for node in scalar if isinstance(node, ast.For))
    packet_loop = next(node for node in recovered if isinstance(node, ast.For))
    assert ast.dump(original_loop.body[-1].value) == ast.dump(
        packet_loop.body[-1].value
    )
    packet_loop.body[-1].targets = copy.deepcopy(original_loop.body[-1].targets)
    recovered = [packet_loop]
    assert ast.dump(ast.Module(body=recovered, type_ignores=[])) == ast.dump(
        ast.Module(body=scalar, type_ignores=[])
    )
    source = ast.unparse(ast.Module(body=packet, type_ignores=[]))
    assert "shared.iterator + cute.assume" in source
    assert "shared.layout" in source and "divby=4" in source
    assert ".align(" not in source and ".toint(" not in source
    assert "num_bits_per_copy=128" in source
    assert "divisor != 0" in source and "cutlass.Float32(-0.0)" in source


@pytest.mark.parametrize("active_mask", range(16))
@pytest.mark.parametrize("columns,divisor", [(0, 0.0), (3, 0.0), (4, 2.0)])
def test_every_lane_mask_initializes_packet_without_evaluating_inactive_loads(
    active_mask: int, columns: int, divisor: float
) -> None:
    from test.test_cute_collective_vector_recipe import _Tensor

    def float_bits(value: float) -> bytes:
        return struct.pack("f", value)

    def rna(value: float) -> int:
        bits = struct.unpack("I", float_bits(value))[0]
        return (bits + 0x1000) & 0xFFFFE000 if bits & 0x7F800000 != 0x7F800000 else bits

    class Bits(int):
        def bitcast(self, dtype: type) -> float:
            assert dtype is float
            return struct.unpack("f", struct.pack("I", self))[0]

    class Registers(_Tensor):
        def __init__(self) -> None:
            super().__init__([float("nan")] * 4, (4,))
            self.written: set[int] = set()

        def __setitem__(self, index: int, value: float) -> None:
            super().__setitem__(index, value)
            self.written.add(index)

    results = []
    for packet in (False, True):
        source = _Tensor([-0.0, float("nan"), float("inf"), 1.00048828125], (1, 4))
        shared = _Tensor([901.0] * 4, (1, 4))
        copied = []

        def copy_packet(
            atom: object,
            registers: Registers,
            destination: SimpleNamespace,
            *,
            copies: list[bool] = copied,
        ) -> None:
            assert atom == (float, 128)
            assert registers.written == set(range(4))
            assert destination.iterator.offset % 4 == 0
            for lane in range(4):
                destination.iterator.tensor.values[
                    destination.iterator.offset + lane
                ] = registers[lane]
            copies.append(True)

        namespace = {
            "source": source,
            "shared": shared,
            "m": 0,
            "k": 0,
            "rows": 1,
            "columns": columns,
            "pitch": 4,
            "divisor": divisor,
            "active_mask": active_mask,
            "cutlass": SimpleNamespace(
                Float32=float, TFloat32=float, Uint32=Bits, range_constexpr=range
            ),
            "cute": SimpleNamespace(
                arch=SimpleNamespace(cvt_f32_tf32=rna),
                make_layout=lambda shape, stride: SimpleNamespace(
                    shape=shape, stride=stride
                ),
                make_rmem_tensor=lambda layout, dtype: Registers(),
                make_tensor=lambda pointer, layout: SimpleNamespace(
                    iterator=pointer, layout=layout
                ),
                crd2idx=lambda index, layout: sum(
                    c * s for c, s in zip(index, layout.stride, strict=True)
                ),
                assume=lambda value, divby: (
                    value
                    if value % divby == 0
                    else (_ for _ in ()).throw(AssertionError("false alignment"))
                ),
                make_copy_atom=lambda operation, dtype, num_bits_per_copy: (
                    dtype,
                    num_bits_per_copy,
                ),
                nvgpu=SimpleNamespace(CopyUniversalOp=lambda: None),
                copy=copy_packet,
            ),
        }
        exec(
            compile(
                ast.Module(body=_emit(packet), type_ignores=[]),
                "<operand-packet>",
                "exec",
            ),
            namespace,
        )
        assert len(copied) == int(packet)
        assert source.loads == [
            lane for lane in range(columns) if active_mask & (1 << lane)
        ]
        results.append([float_bits(value) for value in shared.values])
    assert results[0] == results[1]
