"""CPU/offline shared-A descriptor bridge using the generic tile formulas."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor
import pytest
import torch

from helion._compiler.cute.cute_mma_dag_tcgen05 import Tcgen05Tile
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_conditional_mma import launch_conditional_mma

pytestmark = [pytest.mark.cpu_only, pytest.mark.slow]


@pytest.mark.parametrize("columns", [32, 40])
def test_conditional_shared_a_actual_issue_compiles(tmp_path, columns):
    initialized = torch.cuda.is_initialized()
    tile = Tcgen05Tile(
        key=(),
        tag="a",
        region="fixed",
        operand=None,
        route="shared",
        rows=128,
        cols=128,
        elem_bytes=2,
        atom_bytes=128,
        swizzle="s128b",
        slabs=2,
        offset=0,
    )
    qtile = Tcgen05Tile(
        key=(),
        tag="q",
        region="fixed",
        operand=None,
        route="shared",
        rows=columns,
        cols=128,
        elem_bytes=2,
        atom_bytes=128,
        swizzle="s128b",
        slabs=2,
        offset=0,
    )
    assert tile.leading_byte_offset(True) == qtile.leading_byte_offset(True) == 16
    assert tile.leading_byte_offset(False) == 16384
    assert tile.stride_byte_offset == qtile.stride_byte_offset == 1024
    steps = (
        tuple(tile.k_step_offset(s, True) for s in range(8)),
        tuple(tile.k_step_offset(s, False) for s in range(8)),
        tuple(qtile.k_step_offset(s, True) for s in range(8)),
        tuple(256 * s for s in range(8)),
    )
    # Complete physical panel coverage and exact bit-preserving placement.
    for rows in (128, columns):
        addresses = [
            128 * r + 2 * ((c % 64) ^ ((r % 8) * 8)) + 128 * rows * (c // 64)
            for r in range(rows)
            for c in range(128)
        ]
        assert sorted(addresses) == list(range(0, rows * 128 * 2, 2))
    p_addresses = [
        2048 * (q // 8) + 16 * k + 2 * (q % 8)
        for q in range(columns)
        for k in range(128)
    ]
    assert sorted(p_addresses) == list(range(0, columns * 128 * 2, 2))
    tensors = [
        make_fake_tensor(dtype, shape, stride, assumed_align=16)
        for dtype, shape, stride in (
            (cutlass.BFloat16, (128, 128), (128, 1)),
            (cutlass.BFloat16, (128, 128), (128, 1)),
            (cutlass.BFloat16, (columns, 128), (128, 1)),
            (cutlass.BFloat16, (columns, 128), (128, 1)),
            (cutlass.Float32, (128, columns), (columns, 1)),
            (cutlass.Float32, (128, columns), (columns, 1)),
        )
    ]
    with cpu_target(forbid_native=False):
        compiled = cute.compile(
            launch_conditional_mma,
            *tensors,
            cuda.CUstream(0),
            columns,
            *steps,
            no_jit_engine=True,
            no_cache=True,
        )
    (tmp_path / "compiled.mlir").write_text(str(compiled.ir_module))
    assert torch.cuda.is_initialized() == initialized
