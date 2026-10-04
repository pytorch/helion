from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from helion._compiler.cute.chained_native_reads import plan_native_vector_read
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("threads", (32, 64, 128, 256))
@pytest.mark.parametrize("columns", (8, 32, 64, 128))
@pytest.mark.parametrize("offset,height", ((0, 32), (32, 32), (7, 33)))
def test_native_read_keeps_full_owner_and_exact_logical_ownership(
    dtype, threads, columns, offset, height
):
    ownership = plan_vector_ownership((height, 128), threads, tile_columns=columns)
    assert ownership is not None
    read = plan_native_vector_read((64, 128), (height, 128), offset, dtype, ownership)
    assert read is not None and read.matches()
    emission = read.emit("original_alias", "read", "producer_thread", "step")
    assert emission.setup[2] == "read_source = read_thread.partition_S(original_alias)"
    assert emission.copy == (
        f"cute.copy(read_copy, read_source[{ownership.copy_indices('step')}], read_values)"
    )
    assert emission.values == "read_values"
    # The logical eight-value vector cannot cross a K64 panel. Native swizzling
    # preserves its element order for every 128-byte-aligned frame phase.
    for row in range(height):
        for column in range(0, 128, 8):
            for phase in range(0, 1024, 128):
                offsets = []
                for element in range(8):
                    col = column + element
                    linear = phase + 2 * (
                        col % 64 + 64 * (offset + row) + col // 64 * 64 * 64
                    )
                    offsets.append(linear ^ ((linear >> 3) & 0x70))
                assert offsets == list(range(offsets[0], offsets[0] + 16, 2))
                assert offsets[0] % 16 == 0


@pytest.mark.parametrize(
    "full,shape,offset,dtype",
    (
        ((64, 128), (32, 128), True, torch.bfloat16),
        ((64, 128), (32, 128), -1, torch.bfloat16),
        ((64, 128), (32, 128), 33, torch.bfloat16),
        ((64, 128), (32, 64), 0, torch.bfloat16),
        ((63, 128), (32, 128), 0, torch.bfloat16),
        ((64, 32), (32, 32), 0, torch.bfloat16),
        ((64, 128), (32, 128), 0, torch.float32),
        ((64, 128), (32, 128), 0, torch.int16),
    ),
)
def test_native_read_rejects_unproved_geometry(full, shape, offset, dtype):
    ownership = plan_vector_ownership(shape, 128)
    assert ownership is not None
    assert plan_native_vector_read(full, shape, offset, dtype, ownership) is None


def test_native_read_rejects_stale_dependent_ownership_before_emission():
    ownership = plan_vector_ownership((32, 128), 128, tile_columns=32)
    assert ownership is not None
    read = plan_native_vector_read((64, 128), (32, 128), 32, torch.bfloat16, ownership)
    assert read is not None
    invalid = replace(read, ownership=replace(ownership, trips=1))
    assert not invalid.matches()
    with pytest.raises(ValueError, match="geometry changed"):
        invalid.emit("original_alias", "read", "thread", "step")
