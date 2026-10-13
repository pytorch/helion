"""CPU proofs for scalar typed-cache layouts, not native operand aliases."""

from __future__ import annotations

from collections import Counter
import importlib
from itertools import pairwise
from unittest.mock import patch

import cutlass
import cutlass.cute as cute
import pytest
import sympy
import torch

from helion import exc
from helion._compiler.cute.chained_cache_layout import PointwiseCacheLayouts
from helion._compiler.cute.chained_cache_layout import cache_xor_swizzle
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts
from helion._compiler.cute.chained_scratch_layout import xor_swizzle

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_ROWS = (2, 3, 16, 32, 128)
_COLUMNS = (2, 4, 8, 16, 32, 64, 128, 256)
ir = importlib.import_module("cutlass._mlir.ir")
passmanager = importlib.import_module("cutlass._mlir.passmanager")


def _physical(row, column, columns, parameters):
    bits, base, shift = parameters
    linear = row * columns + column
    return linear ^ (((linear >> (base + shift)) & ((1 << bits) - 1)) << base)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("columns", _COLUMNS)
def test_every_coordinate_is_a_row_local_bijection_with_unchanged_bytes(
    dtype, rows, columns
):
    parameters = cache_xor_swizzle((rows, columns), dtype)
    assert parameters is not None
    bits, base, shift = parameters
    column_bits = columns.bit_length() - 1
    assert base == int(dtype != torch.float32 and columns >= 64)
    assert base + shift == column_bits
    assert 1 <= bits <= 5 and base + bits <= column_bits
    source_mask = ((1 << bits) - 1) << (base + shift)
    target_mask = ((1 << bits) - 1) << base
    assert source_mask & target_mask == 0
    physical = [
        _physical(row, column, columns, parameters)
        for row in range(rows)
        for column in range(columns)
    ]
    assert sorted(physical) == list(range(rows * columns))
    assert physical != list(range(rows * columns))
    for logical, offset in enumerate(physical):
        assert logical // columns == offset // columns
        # Disjoint source/target fields also make the permutation its inverse.
        assert _physical(offset // columns, offset % columns, columns, parameters) == (
            logical
        )
        if base == 1:
            assert offset % 2 == logical % 2
    element_bytes = torch.empty((), dtype=dtype).element_size()
    raw_bytes = rows * columns * element_bytes
    allocated_bytes = (raw_bytes + 127) // 128 * 128
    assert (max(physical) + 1) * element_bytes == raw_bytes
    assert ((max(physical) + 1) * element_bytes + 127) // 128 * 128 == allocated_bytes
    # Pointer origin and frame stride are outside the logical permutation.
    # Non-power-of-two frame strides need no additional swizzle alignment.
    for origin in (128, 1152, 17152):
        intervals = [
            (origin + slot * (allocated_bytes + 128), allocated_bytes)
            for slot in range(3)
        ]
        for start, size in intervals:
            assert start + min(physical) * element_bytes == start
            assert start + (max(physical) + 1) * element_bytes <= start + size
        assert all(a + size <= b for (a, size), (b, _) in pairwise(intervals))


@pytest.mark.parametrize(
    "shape", [(rows, columns) for rows in _ROWS for columns in _COLUMNS]
)
def test_fp32_matches_existing_scratch_policy_and_source_exactly(shape):
    assert cache_xor_swizzle(shape, torch.float32) == xor_swizzle(shape)
    assert PointwiseCacheLayouts("xor").layout(shape, torch.float32, "unused") == (
        ScratchLayouts("xor").layout("scratch", shape)
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_half_32_layout_matches_validated_manual_layout(dtype):
    assert PointwiseCacheLayouts("xor").layout((32, 32), dtype, "unused") == (
        "cute.make_composed_layout(cute.make_swizzle(5, 0, 5), 0, "
        "cute.make_layout((32, 32), stride=(32, 1)))"
    )


@pytest.mark.parametrize(
    "shape",
    (
        (),
        (32,),
        (2, 4, 8),
        (0, 32),
        (-2, 32),
        (1, 32),
        (32, 0),
        (32, -2),
        (32, 1),
        (32, 3),
        (32, 24),
        (32, 96),
        (True, 32),
        (32, False),
        (2.0, 32),
        (32, 8.0),
        (sympy.Integer(2), 32),
        (32, sympy.Integer(8)),
        (None, 32),
        (32, None),
    ),
)
def test_unsupported_shapes_do_not_activate_or_change_fallback(shape):
    for dtype in _DTYPES:
        assert cache_xor_swizzle(shape, dtype) is None
        attempt = PointwiseCacheLayouts("xor")
        default = "cute.make_layout((32, 32), stride=(36, 1))"
        assert attempt.layout(shape, dtype, default) is default
        assert not attempt.activated
        with pytest.raises(
            exc.BackendUnsupported, match="eligible materialized typed cache"
        ):
            attempt.validate()


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
        torch.complex64,
        None,
    ),
)
def test_unsupported_dtypes_keep_exact_fallback(dtype):
    attempt = PointwiseCacheLayouts("xor")
    assert cache_xor_swizzle((32, 32), dtype) is None
    default = "existing_layout"
    assert attempt.layout((32, 32), dtype, default) is default
    assert not attempt.activated
    with pytest.raises(
        exc.BackendUnsupported, match="eligible materialized typed cache"
    ):
        attempt.validate()


@pytest.mark.parametrize(
    "mode", (None, True, False, 0, 1, "", "row_major", "XOR", [], {})
)
def test_invalid_modes_reject(mode):
    with pytest.raises(ValueError, match="auto or xor"):
        PointwiseCacheLayouts(mode)


def test_auto_is_byte_identical_without_even_consulting_eligibility():
    attempt = PointwiseCacheLayouts()
    defaults = (
        "cute.make_layout((32, 32))",
        "cute.make_layout((32, 32), stride=(36, 1))",
        "  preexisting_layout  ",
    )
    with patch(
        "helion._compiler.cute.chained_cache_layout.cache_xor_swizzle",
        side_effect=AssertionError("auto must preserve the supplied layout"),
    ):
        for default in defaults:
            assert attempt.layout((32, 32), torch.bfloat16, default) is default
    assert not attempt.activated
    attempt.validate()


def test_activation_is_attempt_local_and_requires_an_emitted_eligible_layout():
    first, second = PointwiseCacheLayouts("xor"), PointwiseCacheLayouts("xor")
    with pytest.raises(
        exc.BackendUnsupported, match="eligible materialized typed cache"
    ):
        first.validate()
    first.layout((3, 64), torch.float16, "unused")
    assert first.activated and not second.activated
    assert first.layout((1, 64), torch.float16, "original") == "original"
    first.validate()
    with pytest.raises(
        exc.BackendUnsupported, match="eligible materialized typed cache"
    ):
        second.validate()


def _bank_words(addresses):
    words = [address // 4 for address in addresses]
    banks = {}
    for word in words:
        banks.setdefault(word % 32, set()).add(word)
    return (
        len(set(words)),
        len(banks),
        max(map(len, banks.values())),
        max(Counter(words).values()),
    )


def test_manual_half32_reference_counts_distinct_transposed_words():
    origin = 17152
    row_major = [origin + 2 * (row * 32) for row in range(32)]
    parameters = cache_xor_swizzle((32, 32), torch.bfloat16)
    xor = [origin + 2 * _physical(row, 0, 32, parameters) for row in range(32)]
    assert _bank_words(row_major) == (32, 2, 16, 1)
    assert _bank_words(xor) == (32, 32, 1, 1)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("columns", (32, 64, 128, 256))
def test_bank_model_counts_distinct_words_not_halfword_lanes(dtype, columns):
    parameters = cache_xor_swizzle((32, columns), dtype)
    element_bytes = torch.empty((), dtype=dtype).element_size()
    for origin in (128, 1152, 17152):
        for fixed in range(32):
            row_addresses = [
                origin + element_bytes * _physical(fixed, lane, columns, parameters)
                for lane in range(32)
            ]
            column_addresses = [
                origin + element_bytes * _physical(lane, fixed, columns, parameters)
                for lane in range(32)
            ]
            assert len(set(row_addresses)) == len(set(column_addresses)) == 32
            # Two distinct half elements in one 32-bit word are not two
            # conflicting read requests. These are read-bank facts, not claims
            # about vector accesses, write throughput or measured speed.
            assert _bank_words(row_addresses) == (
                (32, 32, 1, 1) if dtype == torch.float32 else (16, 16, 1, 2)
            )
            assert _bank_words(column_addresses) == (32, 32, 1, 1)
    assert _bank_words([1152] * 32) == (1, 1, 1, 32)
    assert _bank_words([1152 + lane * 128 for lane in range(32)]) == (32, 1, 32, 1)


def _lowered_memory(module):
    passmanager.PassManager.parse(
        "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,"
        "convert-cute-to-core,canonicalize)"
    ).run(module.operation)
    assert module.operation.verify()
    values, memory, reads, writes = {}, {}, [], []
    for view in module.body.operations:
        op = view.operation
        if op.name == "arith.constant":
            value = op.attributes["value"].value
        elif op.name in ("llvm.inttoptr", "llvm.ptrtoint"):
            value = values[op.operands[0]]
        elif op.name == "llvm.getelementptr":
            indices = list(op.attributes["rawConstantIndices"])
            assert len(indices) == 1
            scale = {"i8": 1, "bf16": 2, "f16": 2, "f32": 4}[
                str(op.attributes["elem_type"])
            ]
            value = values[op.operands[0]] + scale * indices[0]
        elif op.name == "llvm.load":
            address = values[op.operands[0]]
            reads.append(address)
            value = memory[address]
        elif op.name == "llvm.store":
            address, value = values[op.operands[1]], values[op.operands[0]]
            writes.append(address)
            memory[address] = value
            continue
        else:
            assert op.name == "llvm.intr.assume", op.name
            continue
        values[op.results[0]] = value
    return reads, writes, memory


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "shape",
    ((3, 2), (3, 4), (3, 8), (16, 16), (32, 32), (3, 64), (16, 128), (128, 256)),
)
def test_actual_cute_scalar_load_store_uses_relative_offsets_in_each_slot(dtype, shape):
    rows, columns = shape
    parameters = cache_xor_swizzle(shape, dtype)
    cute_dtype = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }[dtype]
    element_bytes = torch.empty((), dtype=dtype).element_size()
    expression = PointwiseCacheLayouts("xor").layout(shape, dtype, "unused")
    # Exhaust small geometries; sample row/column field boundaries for larger
    # tensors. The all-coordinate, all-geometry proof above covers their rest.
    row_samples = sorted({0, 1, rows // 2, rows - 1})
    column_samples = sorted(
        {0, 1, columns // 2 - 1, columns // 2, columns - 2, columns - 1}
    )
    points = [(r, c) for r in row_samples for c in column_samples]
    if rows * columns <= 256:
        points = [(r, c) for r in range(rows) for c in range(columns)]
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        for frame_base, region_offset, slot in (
            (1024, 17152, 0),
            (1152, 128, 1),
            (1280, 384, 2),
        ):
            raw_bytes = rows * columns * element_bytes
            slot_stride = (region_offset + raw_bytes + 383) // 128 * 128
            frame_origin = frame_base + slot * slot_stride
            origin = frame_origin + region_offset
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                frame = cute.make_ptr(
                    cutlass.Uint8,
                    frame_origin,
                    cute.AddressSpace.smem,
                    assumed_align=128,
                )
                pointer = cute.recast_ptr(frame + region_offset, dtype=cute_dtype)
                layout = eval(expression, {"cute": cute})
                assert cute.size(layout) == rows * columns
                for row, column in points:
                    assert int(layout((row, column))) == _physical(
                        row, column, columns, parameters
                    )
                producer = cute.make_tensor(pointer, layout)
                consumer = cute.make_tensor(pointer, eval(expression, {"cute": cute}))
                output = cute.make_tensor(
                    cute.make_ptr(cute_dtype, 1000000, cute.AddressSpace.gmem),
                    cute.make_layout((len(points),)),
                )
                for index, (row, column) in enumerate(points):
                    producer[row, column] = cute_dtype(index % 128)
                for index, (row, column) in enumerate(points):
                    output[index] = consumer[row, column]
            reads, writes, memory = _lowered_memory(module)
            expected = [
                origin + element_bytes * _physical(r, c, columns, parameters)
                for r, c in points
            ]
            assert reads == writes[: len(points)] == expected
            assert len(set(expected)) == len(points)
            assert all(origin <= address < origin + raw_bytes for address in expected)
            assert writes[len(points) :] == [
                1000000 + element_bytes * i for i in range(len(points))
            ]
            assert [
                memory[1000000 + element_bytes * i] for i in range(len(points))
            ] == [i % 128 for i in range(len(points))]
    assert torch.cuda.is_initialized() is before
