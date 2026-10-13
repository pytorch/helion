from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from . import test_cute_chained_native_read_mapping as mapping
from helion._compiler.cute.chained_native_reads import plan_native_vector_read
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership

_CASES = (
    ((32, 128), (32, 128), 0, 128, 32),
    ((64, 128), (32, 128), 32, 128, 32),
    ((48, 64), (33, 64), 7, 64, 16),
    ((128, 256), (97, 256), 7, 512, 64),
)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("case", _CASES)
def test_actual_column_major_copy_order_and_full_native_views(case, dtype):
    full, shape, offset, threads, columns = case
    owner = plan_vector_ownership(
        shape, threads, tile_columns=columns, thread_order="column_major"
    )
    assert owner is not None
    read = plan_native_vector_read(full, shape, offset, dtype, owner)
    assert read is not None
    # Reuse the existing dynamic CuTe fixture and fail-closed LLVM interpreter,
    # not a coordinate-only approximation. Bypass its row-major result cache.
    with patch.object(mapping, "_read", return_value=read):
        compiled, before, lowered = mapping._lower_copy.__wrapped__(case, dtype)
    assert "!cute" in before
    assert ("vector<8xbf16>" if dtype == torch.bfloat16 else "vector<8xf16>") in lowered
    phases = range(0, 1024, 128) if shape == (32, 128) else (0, 896)
    slots = range(4) if shape == (32, 128) else (0, 3)
    for phase in phases:
        for slot in slots:
            origin = 2048 + phase + slot * 49920
            cells, addresses = set(), set()
            for thread in range(threads):
                for step in range(owner.trips):
                    row_tile, column_tile = divmod(step, owner.column_tiles)
                    row = thread % owner.thread_rows + row_tile * owner.thread_rows
                    base = (
                        thread // owner.thread_rows * 8
                        + column_tile * owner.tile_columns
                    )
                    if row >= shape[0]:
                        continue
                    payload, loaded = mapping._evaluate(compiled, thread, step, origin)
                    vector, scalar, whole = payload[:8], payload[8:16], payload[16:]
                    assert vector == scalar == whole == loaded
                    assert vector != tuple(reversed(scalar))
                    assert vector == tuple(
                        mapping._native_address(
                            origin, full[0], row + offset, base + element
                        )
                        for element in range(8)
                    )
                    assert loaded[0] % 16 == 0
                    assert loaded == tuple(
                        loaded[0] + 2 * element for element in range(8)
                    )
                    for element, address in enumerate(loaded):
                        cell = row, base + element
                        assert cell not in cells and address not in addresses
                        cells.add(cell)
                        addresses.add(address)
            assert cells == {
                (row, column) for row in range(shape[0]) for column in range(shape[1])
            }


def test_default_native_source_and_explicit_row_major_are_identical():
    for case in _CASES:
        full, shape, offset, threads, columns = case
        before = plan_vector_ownership(shape, threads, tile_columns=columns)
        after = plan_vector_ownership(
            shape, threads, tile_columns=columns, thread_order="row_major"
        )
        assert before == after and before is not None
        read = plan_native_vector_read(full, shape, offset, torch.bfloat16, before)
        assert read is not None
        source = read.emit("native", "read", "thread", "step")
        rows, columns = before.thread_rows, before.thread_columns
        assert source.setup[0] == (
            "read_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128), "
            f"cute.make_layout(({rows}, {columns}), stride=({columns}, 1)), cute.make_layout((1, 8)))"
        )
