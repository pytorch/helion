from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from helion._compiler.cute.chained_native_stores import plan_native_stmatrix_store
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("threads", (32, 64, 128, 256, 1024))
@pytest.mark.parametrize(
    "owner,member,offset", ((32, 32, 0), (80, 32, 16), (160, 128, 32))
)
def test_stmatrix_preserves_full_native_member_and_complete_warp_owners(
    dtype, threads, owner, member, offset
):
    shape = (32, member)
    ownership = plan_vector_ownership(shape, threads, tile_columns=32)
    assert ownership is not None
    store = plan_native_stmatrix_store((owner, 32), shape, offset, dtype, ownership)
    assert store is not None and store.matches()
    emission = store.emit("original_alias", "store", "thread", "step")
    assert "partition_D(original_alias)" in emission.setup[1]
    assert "StMatrix8x8x16bOp(num_matrices=4, transpose=True)" in emission.setup[0]
    assert "domain_offset" not in "\n".join(emission.setup)
    assert "assumed_align" not in "\n".join(emission.setup)
    assert emission.copy.endswith(f"store_target[{ownership.copy_indices('step')}])")
    cells = set()
    for step in range(ownership.trips):
        for warp in range(threads // 32):
            active = []
            for lane in range(32):
                thread = warp * 32 + lane
                row = (
                    thread // 4 + step // ownership.column_tiles * ownership.thread_rows
                )
                base = thread % 4 * 8 + step % ownership.column_tiles * 32
                active.append(row < shape[0])
                if row < shape[0]:
                    for element in range(8):
                        cell = row, base + element
                        assert cell not in cells
                        cells.add(cell)
            assert len(set(active)) == 1
    assert len(cells) == 32 * member
    for phase in range(0, 1024, 128):
        for native_row in range(offset, offset + member):
            for column in range(0, 32, 8):
                addresses = []
                for element in range(8):
                    raw = phase + 2 * (native_row * 32 + column + element)
                    addresses.append(raw ^ ((raw & 384) >> 3))
                assert addresses == list(range(addresses[0], addresses[0] + 16, 2))
                assert addresses[0] % 16 == 0


@pytest.mark.parametrize(
    "full,shape,offset,dtype,threads,columns",
    (
        ((160, 32), (32, 128), True, torch.bfloat16, 128, 32),
        ((160, 32), (32, 128), -16, torch.bfloat16, 128, 32),
        ((160, 32), (32, 128), 1, torch.bfloat16, 128, 32),
        ((160, 32), (32, 128), 48, torch.bfloat16, 128, 32),
        ((159, 32), (32, 128), 0, torch.bfloat16, 128, 32),
        ((160, 64), (64, 128), 32, torch.bfloat16, 128, 32),
        ((160, 128), (128, 128), 32, torch.bfloat16, 128, 32),
        ((160, 32), (13, 128), 32, torch.bfloat16, 128, 32),
        ((160, 32), (32, 128), 32, torch.float32, 128, 32),
        ((160, 32), (32, 128), 32, torch.int16, 128, 32),
        ((160, 32), (32, 128), 32, torch.bfloat16, 128, 8),
        ((160, 32), (32, 128), 32, torch.bfloat16, 128, 128),
        ((160, 32), (32, 128), 32, torch.bfloat16, 16, 32),
        ((160, 32), (32, 128), 32, torch.bfloat16, 96, 32),
    ),
)
def test_unproved_store_layout_partial_warp_and_k_panel_decline(
    full, shape, offset, dtype, threads, columns
):
    ownership = plan_vector_ownership(shape, threads, tile_columns=columns)
    assert ownership is not None
    assert plan_native_stmatrix_store(full, shape, offset, dtype, ownership) is None


@pytest.mark.parametrize(
    "field,value",
    (("trips", 1), ("thread_rows", 1), ("row_tiles", 2), ("column_tiles", 2)),
)
def test_corrupted_ownership_is_rejected_before_emission(field, value):
    ownership = plan_vector_ownership((32, 128), 128, tile_columns=32)
    assert ownership is not None
    store = plan_native_stmatrix_store(
        (160, 32), (32, 128), 32, torch.bfloat16, ownership
    )
    assert store is not None
    invalid = replace(store, ownership=replace(ownership, **{field: value}))
    assert not invalid.matches()
    with pytest.raises(ValueError, match="geometry changed"):
        invalid.emit("unchanged", "store", "thread", "step")
