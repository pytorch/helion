from __future__ import annotations

import ctypes
from dataclasses import replace
import itertools
import random
from typing import Literal
from unittest.mock import patch

import pytest

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import cutlass
import cutlass.cute as cute

from helion._compiler.cute.tcgen05_flattened_prefix import FlatGroupedProviderPlan
from helion._compiler.cute.tcgen05_flattened_prefix import FlatRowLayout
from helion._compiler.cute.tcgen05_flattened_prefix import build_flat_tile_prefix
from helion._compiler.cute.tcgen05_flattened_prefix import project_flat_interval
from helion._compiler.cute.tcgen05_flattened_prefix import resolve_flat_nm_work


def signed(value: int, bits: int) -> int:
    word = value % (1 << bits)
    return word - (1 << bits) if word >= 1 << (bits - 1) else word


def original_addresses(
    start: int, end: int, plan: FlatGroupedProviderPlan
) -> tuple[list[list[int]], list[list[int]]] | None:
    """Enumerate the original row-add/multiply/inner-add, not the span formula."""
    count = signed(end - start, plan.offset_bits)
    if count <= 0:
        return [], []
    if count > plan.max_rows + 1:
        # One more row than either buffer can hold suffices to find an invalid
        # address, without executing an enormous source-width positive loop.
        count = plan.max_rows + 1
    results = []
    for layout in (plan.a, plan.output):
        rows = []
        for row in range(count):
            logical_row = signed(start + row, plan.offset_bits)
            scaled_row = signed(logical_row * layout.row_stride, plan.offset_bits)
            values = [
                signed(scaled_row + column, plan.offset_bits)
                for column in range(layout.width)
            ]
            if any(value < 0 or value >= layout.elements for value in values):
                return None
            rows.append(values)
        results.append(rows)
    return results[0], results[1]


@cute.jit
def host_project(
    input_address: cutlass.Int64,
    output_address: cutlass.Int64,
    count: cutlass.Int32,
    bits: cutlass.Constexpr[int],
    sizes: cutlass.Constexpr,
) -> None:
    integer_type = cutlass.Int32 if cutlass.const_expr(bits == 32) else cutlass.Int64
    inputs = cute.make_tensor(
        cute.make_ptr(
            integer_type, input_address, cute.AddressSpace.generic, assumed_align=4
        ),
        cute.make_layout((count, 2), stride=(2, 1)),
    )
    outputs = cute.make_tensor(
        cute.make_ptr(
            cutlass.Int32, output_address, cute.AddressSpace.generic, assumed_align=4
        ),
        cute.make_layout((count, 4), stride=(4, 1)),
    )
    for index in cutlass.range(count, unroll=1):
        a_base, d_base, extent, supported = project_flat_interval(
            inputs[index, 0], inputs[index, 1], integer_type, *sizes
        )
        outputs[index, 0] = a_base
        outputs[index, 1] = d_base
        outputs[index, 2] = extent
        outputs[index, 3] = cutlass.Int32(supported)


@cute.jit
def host_prefix(
    input_address: cutlass.Int64,
    stride: cutlass.Int64,
    prefix_address: cutlass.Int64,
    output_address: cutlass.Int64,
    queries: cutlass.Int32,
    groups: cutlass.Constexpr[int],
    bits: cutlass.Constexpr[int],
    sizes: cutlass.Constexpr,
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
) -> None:
    integer_type = cutlass.Int32 if cutlass.const_expr(bits == 32) else cutlass.Int64
    inputs = cute.make_tensor(
        cute.make_ptr(
            integer_type, input_address, cute.AddressSpace.generic, assumed_align=4
        ),
        cute.make_layout((groups + 1,), stride=(stride,)),
    )
    prefix = cute.make_tensor(
        cute.make_ptr(
            cutlass.Int32, prefix_address, cute.AddressSpace.generic, assumed_align=4
        ),
        cute.make_layout((groups,), stride=(1,)),
    )
    outputs = cute.make_tensor(
        cute.make_ptr(
            cutlass.Int32, output_address, cute.AddressSpace.generic, assumed_align=4
        ),
        cute.make_layout((queries, 9), stride=(9, 1)),
    )
    build_flat_tile_prefix(inputs, prefix, groups, *sizes, tile_m, tile_n)
    for query in cutlass.range(queries, unroll=1):
        record = resolve_flat_nm_work(
            inputs,
            prefix,
            cutlass.Int64(query - 1),
            groups,
            *sizes,
            tile_m,
            tile_n,
        )
        for field in cutlass.range_constexpr(9):
            outputs[query, field] = record[field]


def sizes(plan: FlatGroupedProviderPlan) -> tuple[int, ...]:
    return (
        plan.a.elements,
        plan.a.width,
        plan.a.row_stride,
        plan.output.elements,
        plan.output.width,
        plan.output.row_stride,
    )


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize(
    ("a", "output"),
    [
        (FlatRowLayout(33 * 128, 128, 128, 16), FlatRowLayout(33 * 128, 128, 128, 16)),
        (FlatRowLayout(33 * 96, 96, 96, 16), FlatRowLayout(33 * 192, 192, 192, 16)),
        (
            FlatRowLayout(32 * 132 + 128, 128, 132, 16),
            FlatRowLayout(32 * 140 + 132, 132, 140, 16),
        ),
        (FlatRowLayout(0, 128, 128, 16), FlatRowLayout(0, 132, 132, 16)),
    ],
)
def test_actual_host_jit_matches_all_original_touched_elements(
    bits: Literal[32, 64], a: FlatRowLayout, output: FlatRowLayout
) -> None:
    plan = FlatGroupedProviderPlan(1, a, output, 128, 128, bits)
    rng = random.Random(2026091501 + bits)
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    starts = [
        low,
        low + 1,
        high,
        -1,
        0,
        1,
        17,
        32,
        33,
        1 << 25,
        1 << (bits - 7),
        -(1 << (bits - 7)),
        pow(3, -1, 1 << (bits - 5)),
    ]
    starts += [rng.randrange(low, high + 1) for _ in range(64)]
    pairs = [
        (signed(start, bits), signed(start + extent, bits))
        for start, extent in itertools.product(starts, (-35, -1, 0, 1, 2, 9, 34))
    ]
    ctype = ctypes.c_int32 if bits == 32 else ctypes.c_int64
    inputs = (ctype * (2 * len(pairs)))(*(v for pair in pairs for v in pair))
    actual = (ctypes.c_int32 * (4 * len(pairs) + 2))(*([-777] * (4 * len(pairs) + 2)))
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")):
        compiled = cute.compile(
            host_project,
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int32(1),
            bits,
            sizes(plan),
            options="--gpu-arch sm_100a",
        )
        compiled(ctypes.addressof(inputs), ctypes.addressof(actual) + 4, len(pairs))
    assert actual[0] == actual[-1] == -777
    for index, (start, end) in enumerate(pairs):
        original = original_addresses(start, end, plan)
        a_base, d_base, extent, supported = actual[1 + 4 * index : 5 + 4 * index]
        assert bool(supported) == (original is not None), (plan, start, end)
        if original is None:
            continue
        a_rows, d_rows = original
        assert extent == len(a_rows) == len(d_rows)
        assert a_base % 4 == d_base % 4 == 0
        for layout, base, rows in ((a, a_base, a_rows), (output, d_base, d_rows)):
            for row, expected in enumerate(rows):
                assert expected == list(
                    range(
                        base + row * layout.row_stride,
                        base + row * layout.row_stride + layout.width,
                    )
                )


@pytest.mark.parametrize("bits", [32, 64])
def test_prefix_preserves_holes_overlap_empty_and_wrapped_element_bases(
    bits: Literal[32, 64],
) -> None:
    plan = FlatGroupedProviderPlan(
        8,
        FlatRowLayout(273 * 96, 96, 96, 16),
        FlatRowLayout(273 * 192, 192, 192, 16),
        128,
        128,
        bits,
    )
    period = 1 << (bits - 5)
    nonrow_start = pow(3, -1, period)
    corpora = [
        [0, 129, 17, 146, 146, 3, 270, 270, 273],
        [nonrow_start, nonrow_start + 3] + [nonrow_start + 3] * 7,
        [period, period + 129] + [period + 129] * 7,
        [0] * 9,
        [0, 129, 100000, 100001] + [100001] * 5,
    ]
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")):
        compiled = cute.compile(
            host_prefix,
            *(cutlass.Int64(0) for _ in range(4)),
            cutlass.Int32(1),
            plan.groups,
            bits,
            sizes(plan),
            plan.tile_m,
            plan.tile_n,
            options="--gpu-arch sm_100a",
        )
        for turn, values in enumerate(corpora):
            ctype = ctypes.c_int32 if bits == 32 else ctypes.c_int64
            stride = 1 + turn % 3
            routing = (ctype * (len(values) * stride + 2))()
            for index, value in enumerate(values):
                routing[1 + index * stride] = signed(value, bits)
            prefix = (ctypes.c_int32 * (plan.groups + 2))(*([-777] * (plan.groups + 2)))
            records = []
            all_supported = True
            for group, (start, end) in enumerate(itertools.pairwise(values)):
                original = original_addresses(
                    signed(start, bits), signed(end, bits), plan
                )
                if original is None:
                    all_supported = False
                    break
                a_rows, d_rows = original
                row_tiles = (len(a_rows) + 127) // 128
                for local in range(row_tiles * 2):
                    row, column = local // 2, local % 2
                    if 0 < row_tiles <= 2:
                        row, column = local % row_tiles, local // row_tiles
                    records.append(
                        (
                            row,
                            column,
                            1,
                            group,
                            a_rows[0][0],
                            len(a_rows),
                            192,
                            96,
                            d_rows[0][0],
                        )
                    )
            queries = len(records) + 2
            result = (ctypes.c_int32 * (queries * 9 + 2))(*([-888] * (queries * 9 + 2)))
            compiled(
                ctypes.addressof(routing) + ctypes.sizeof(ctype),
                stride,
                ctypes.addressof(prefix) + 4,
                ctypes.addressof(result) + 4,
                queries,
            )
            assert prefix[0] == prefix[-1] == -777
            assert result[0] == result[-1] == -888
            assert (prefix[-2] >= 0) == all_supported
            for query in range(queries):
                actual = tuple(result[1 + 9 * query : 10 + 9 * query])
                if not all_supported:
                    assert actual[2] == -1
                elif 0 < query <= len(records):
                    assert actual == records[query - 1]
                else:
                    assert actual[2] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"elements": 1 << 31},
        {"elements": -1},
        {"width": 0},
        {"row_stride": 127},
        {"row_stride": 129},
        {"base_alignment": 8},
        {"base_alignment": 24},
    ],
)
def test_unsupported_static_layouts_decline(kwargs: dict[str, int]) -> None:
    with pytest.raises(AssertionError):
        replace(FlatRowLayout(4096, 128, 128, 16), **kwargs)


def test_cache_identity_binds_both_element_bases_and_original_integer_width() -> None:
    plan = FlatGroupedProviderPlan(
        3,
        FlatRowLayout(4096, 128, 128, 16),
        FlatRowLayout(8192, 256, 256, 16),
        128,
        128,
        64,
    )
    alternatives = [
        replace(plan, offset_bits=32),
        replace(plan, groups=4),
        replace(plan, tile_n=64),
        replace(plan, a=replace(plan.a, elements=8192)),
        replace(plan, a=replace(plan.a, row_stride=132)),
        replace(plan, output=replace(plan.output, row_stride=260)),
    ]
    assert all(item.cache_identity() != plan.cache_identity() for item in alternatives)
    assert plan.prefix_bytes == 12


@pytest.mark.parametrize("which", ["tile_m", "tile_n"])
def test_tile_dimension_int32_boundary_remains_checked(which: str) -> None:
    plan = FlatGroupedProviderPlan(
        1,
        FlatRowLayout(0, 128, 128, 16),
        FlatRowLayout(0, 128, 128, 16),
        128,
        128,
        64,
    )
    with pytest.raises(AssertionError):
        if which == "tile_m":
            replace(plan, tile_m=1 << 31)
        else:
            replace(plan, tile_n=1 << 31)
