from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

from ._cute_aux import _cpu_codegen
from helion._compiler.cute.chained_tcgen05 import _layout


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,width,members", [(16, 16, 1), (16, 64, 3), (32, 128, 2)])
def test_actual_native_views_have_identical_vector_and_row_owned_cells(
    dtype, rows, width, members
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    cute_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    with _cpu_codegen(), ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            namespace: dict[str, Any] = {
                "cute": cute,
                "cutlass": cutlass,
                "tcgen05": tcgen05,
                "target_ptr": cute.make_ptr(
                    cute_dtype, 128, cute.AddressSpace.smem, assumed_align=128
                ),
            }
            exec(
                "\n".join(
                    _layout(
                        "target",
                        (members * rows, width),
                        1,
                        "cutlass.BFloat16"
                        if dtype == torch.bfloat16
                        else "cutlass.Float16",
                    )
                ),
                namespace,
            )
            physical = namespace["target_layout"]
            target = namespace["target"]
            columns = width // 8
            vector_rows = 128 // columns
            old_copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cute_dtype, num_bits_per_copy=128
                ),
                cute.make_layout((vector_rows, columns), stride=(columns, 1)),
                cute.make_layout((1, 8)),
            )
            all_bytes = set()
            for member in range(members):
                offset = member * rows
                identity = cute.domain_offset(
                    (offset, 0), cute.make_identity_tensor((members * rows, width))
                )
                previous, current = set(), set()
                for thread in range(128):
                    owned = old_copy.get_slice(thread).partition_D(identity)
                    for step in range((rows + vector_rows - 1) // vector_rows):
                        row = thread // columns + step * vector_rows
                        if row < rows:
                            for element in range(8):
                                coordinate = tuple(
                                    map(int, owned[None, step, 0][element])
                                )
                                assert coordinate == (
                                    offset + row,
                                    thread % columns * 8 + element,
                                )
                                assert coordinate not in previous
                                previous.add(coordinate)
                    for step in range((rows + 3) // 4):
                        row = thread // 32 + step * 4
                        for part in range((width + 31) // 32):
                            column = thread % 32 + part * 32
                            if row < rows and column < width:
                                coordinate = offset + row, column
                                assert coordinate not in current
                                current.add(coordinate)
                assert previous == current and len(current) == rows * width
                for coordinate in current:
                    address = int(physical(coordinate)) * 2
                    assert 0 <= address < members * rows * width * 2
                    assert address not in all_bytes
                    all_bytes.add(address)
                # Lower an actual scalar tensor store through the same recast
                # native pointer, including a nonzero grouped member origin.
                target[offset, 0] = cute_dtype(0)
            assert len(all_bytes) == members * rows * width
        assert module.operation.verify()
