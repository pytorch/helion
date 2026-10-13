from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute.chained_register_fragments import _derive_map
from helion._compiler.cute.chained_register_fragments import plan_warp_fragment_map

_CASES = [
    (role, dtype, transpose)
    for role, dtypes in (
        ("a", (torch.float16, torch.bfloat16)),
        ("b", (torch.float16, torch.bfloat16)),
        ("c", (torch.float32,)),
    )
    for dtype in dtypes
    for transpose in (False, True)
]


def _apply(mapping, values):
    return [
        [values[source.lane][source.slot] for source in lane]
        for lane in mapping.sources
    ]


def _transpose_word(values, word):
    # Independent PTX m8n8 b16 movmatrix definition, not a copied lane formula.
    matrix = [[None] * 8 for _ in range(8)]
    for lane in range(32):
        for half in range(2):
            matrix[lane // 4][2 * (lane % 4) + half] = values[lane][2 * word + half]
    return [
        [matrix[2 * (lane % 4) + half][lane // 4] for half in range(2)]
        for lane in range(32)
    ]


@pytest.mark.parametrize("role,dtype,transpose", _CASES)
def test_typed_maps_are_exact_permutations_with_actual_element_unit_recasts(
    role, dtype, transpose
):
    mapping = plan_warp_fragment_map(role, dtype, transpose=transpose)
    assert mapping is not None
    assert mapping.dtype is dtype and mapping.element_bits == dtype.itemsize * 8
    assert len(mapping.sources) == 32 and all(
        len(lane) == 8 for lane in mapping.sources
    )
    assert {
        (source.lane, source.slot) for lane in mapping.sources for source in lane
    } == {(lane, slot) for lane in range(32) for slot in range(8)}
    values = [[(lane, slot) for slot in range(8)] for lane in range(32)]
    result = _apply(mapping, values)
    for lane in range(32):
        for slot in range(8):
            source_lane, source_slot = result[lane][slot]
            assert (
                mapping.source_coordinates[source_lane][source_slot]
                == mapping.destination_coordinates[lane][slot]
            )
    if role == "c":
        assert mapping.source_word_pairs == mapping.destination_word_pairs == ()
    else:
        assert (
            mapping.source_word_pairs
            == mapping.destination_word_pairs
            == ((0, 1), (2, 3), (4, 5), (6, 7))
        )
    assert plan_warp_fragment_map(role, dtype, transpose=transpose) is mapping
    with pytest.raises(FrozenInstanceError):
        mapping.transpose = not transpose  # pyrefly: ignore [read-only]


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "role,transpose", (("a", False), ("b", False), ("a", True), ("b", True))
)
def test_b16_mapping_matches_raw_movmatrix_or_original_same_lane_slots(
    dtype, role, transpose
):
    mapping = plan_warp_fragment_map(role, dtype, transpose=transpose)
    assert mapping is not None
    values = [[(lane, slot) for slot in range(8)] for lane in range(32)]
    if (role, transpose) == ("a", False):
        expected = values
        assert mapping.same_lane
    elif (role, transpose) == ("b", True):
        expected = [
            [row[index] for index in (0, 1, 4, 5, 2, 3, 6, 7)] for row in values
        ]
        assert mapping.same_lane
        # The previously rejected permutation-free transport is a negative control.
        assert expected != values
    else:
        order = (0, 2, 1, 3) if transpose else (0, 1, 2, 3)
        words = [_transpose_word(values, word) for word in order]
        expected = [
            [value for word in words for value in word[lane]] for lane in range(32)
        ]
        assert not mapping.same_lane
    assert _apply(mapping, values) == expected


@pytest.mark.parametrize("role,dtype,transpose", _CASES)
def test_transport_preserves_raw_signed_zero_subnormal_nan_and_infinity_bits(
    role, dtype, transpose
):
    mapping = plan_warp_fragment_map(role, dtype, transpose=transpose)
    assert mapping is not None
    if dtype is torch.float16:
        patterns = (
            0,
            0x8000,
            1,
            0x8001,
            0x03FF,
            0x0400,
            0x7BFF,
            0x7C00,
            0xFC00,
            0x7E01,
            0xFE15,
        )
    elif dtype is torch.bfloat16:
        patterns = (
            0,
            0x8000,
            1,
            0x8001,
            0x007F,
            0x0080,
            0x7F7F,
            0x7F80,
            0xFF80,
            0x7FC1,
            0xFFC5,
        )
    else:
        patterns = (
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0xFFC00005,
        )
    by_coordinate = {
        (row, col): patterns[(row * 16 + col) % len(patterns)]
        for row in range(16)
        for col in range(16)
    }
    values = [
        [by_coordinate[coord] for coord in coords]
        for coords in mapping.source_coordinates
    ]
    actual = _apply(mapping, values)
    expected = [
        [by_coordinate[coord] for coord in coords]
        for coords in mapping.destination_coordinates
    ]
    assert actual == expected
    # These are transport bit equalities, not a promise about NaN arithmetic.
    if role == "c" and transpose:
        inverse = _apply(mapping, actual)
        assert inverse == values


@pytest.mark.parametrize(
    "role,dtype,transpose",
    (
        ("a", torch.float32, False),
        ("b", torch.int16, False),
        ("c", torch.float16, False),
        ("c", torch.bfloat16, True),
        ("other", torch.float16, False),
        ("a", torch.float16, 1),
        ("a", torch.float16, None),
    ),
)
def test_unsupported_dtype_role_or_nonboolean_orientation_is_not_a_map(
    role, dtype, transpose
):
    assert plan_warp_fragment_map(role, dtype, transpose=transpose) is None


def test_private_layout_context_does_not_emit_into_caller_or_initialize_cuda():
    ir = importlib.import_module("cutlass._mlir.ir")
    initialized = torch.cuda.is_initialized()
    _derive_map.cache_clear()
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            before = str(module)
            with patch(
                "torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")
            ):
                assert plan_warp_fragment_map("a", torch.bfloat16) is not None
            assert str(module) == before
        assert module.operation.verify()
    assert torch.cuda.is_initialized() is initialized


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("transpose", (False, True))
def test_actual_ordered_original_group_atom_inputs_and_fp32_output_slots(
    dtype, transpose
):
    cutlass = pytest.importorskip("cutlass")
    cute = importlib.import_module("cutlass.cute")
    ir = importlib.import_module("cutlass._mlir.ir")
    maps = {
        role: plan_warp_fragment_map(
            role, torch.float32 if role == "c" else dtype, transpose=transpose
        )
        for role in ("a", "b", "c")
    }
    input_type = cutlass.Float16 if dtype is torch.float16 else cutlass.BFloat16
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            atom = cute.make_mma_atom(
                cute.nvgpu.warp.MmaF16BF16Op(input_type, cutlass.Float32, (16, 8, 16))
            )
            local = cute.make_tiled_mma(atom, atom_layout_mnk=(1, 1, 1))
            original = cute.make_tiled_mma(atom, atom_layout_mnk=(1, 8, 1))

            def coords(tensor):
                return [
                    tuple(int(value) for value in tensor[index])
                    for index in range(cute.size(tensor))
                ]

            canonical = [
                coords(
                    local.get_slice(lane).partition_C(
                        cute.make_identity_tensor((16, 16))
                    )
                )
                for lane in range(32)
            ]
            for role, mapping in maps.items():
                assert mapping is not None
                actual = _apply(mapping, canonical)
                for lane in range(32):
                    thread = local.get_slice(lane)
                    if role == "a":
                        expected = coords(
                            thread.partition_A(cute.make_identity_tensor((16, 16)))
                        )
                    elif role == "b":
                        expected = coords(
                            thread.partition_B(cute.make_identity_tensor((16, 16)))
                        )
                    else:
                        expected = canonical[lane]
                    if transpose if role != "b" else not transpose:
                        expected = [(column, row) for row, column in expected]
                    assert actual[lane] == expected
            # Include nonzero member offsets and two distinct K panels. No
            # reduction reordering is authorized by splitting physical ownership.
            for offset in (0, 32):
                for origin in (0, 16):
                    for n_atom in (0, 1):
                        old_warp = (offset + origin + 8 * n_atom) // 8
                        for lane in range(32):
                            current = local.get_slice(lane)
                            previous = original.get_slice(old_warp * 32 + lane)
                            a = current.partition_A(cute.make_identity_tensor((16, 16)))
                            b = current.partition_B(cute.make_identity_tensor((16, 16)))
                            c = current.partition_C(cute.make_identity_tensor((16, 16)))
                            old_a = previous.partition_A(
                                cute.make_identity_tensor((32, 32))
                            )
                            old_b = previous.partition_B(
                                cute.make_identity_tensor((64, 32))
                            )
                            old_c = previous.partition_C(
                                cute.make_identity_tensor((32, 64))
                            )
                            assert coords(old_a[None, origin // 16, origin // 16]) == [
                                (origin + row, origin + k)
                                for row, k in coords(a[None, 0, 0])
                            ]
                            assert coords(old_b[None, 0, origin // 16]) == [
                                (offset + origin + column, origin + k)
                                for column, k in coords(b[None, n_atom, 0])
                            ]
                            assert coords(old_c[None, origin // 16, 0]) == [
                                (origin + row, offset + origin + column)
                                for row, column in coords(c[None, 0, n_atom])
                            ]
        assert module.operation.verify()
