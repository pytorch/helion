"""Actual static CuTe K64 partitions/read sets; no device compilation."""

from __future__ import annotations

from collections import Counter
import importlib
from unittest.mock import patch

import pytest
import torch

from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def _mlir():
    # Optional CuTe must not be imported during default-backend collection.
    # Its native extension exports these classes at runtime, not in its stub.
    return importlib.import_module("cutlass._mlir.ir")


@pytest.mark.parametrize("dtype_name", ("BFloat16", "Float16"))
def test_actual_k_sw128_half_copy_ownership(dtype_name):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = _mlir()
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            layout = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(
                    tcgen05.SmemLayoutAtomKind.K_SW128, dtype
                ),
                (128, 128),
                order=(0, 1),
            )
            target = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(dtype, 0, cute.AddressSpace.smem, assumed_align=128),
                    layout.inner,
                    dtype=dtype,
                ),
                layout.outer,
            )
            assert str(layout.inner) == "S<3,4,3>"
            identity = cute.make_identity_tensor((128, 128))
            copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128
                ),
                cute.make_layout((16, 8), stride=(8, 1)),
                cute.make_layout((1, 8)),
            )
            cells = []
            halves = [set(), set()]
            for half in range(2):
                coordinates = cute.local_tile(identity, (128, 64), (0, half))
                destination = cute.local_tile(target, (128, 64), (0, half))
                for thread in range(128):
                    owner = copy.get_slice(thread)
                    dc = owner.partition_D(coordinates)
                    dp = owner.partition_D(destination)
                    sc = owner.partition_S(coordinates)
                    assert [int(cute.size(dp, mode=[i])) for i in range(3)] == [8, 8, 1]
                    for step in range(8):
                        group = []
                        for element in range(8):
                            expected = (
                                thread // 8 + 16 * step,
                                64 * half + 8 * (thread % 8) + element,
                            )
                            assert (
                                tuple(map(int, dc[element, step, 0]))
                                == tuple(map(int, sc[element, step, 0]))
                                == expected
                            )
                            raw = 2 * int(layout.outer(expected))
                            assert (
                                raw
                                == 128 * expected[0]
                                + 2 * (expected[1] % 64)
                                + 16384 * half
                            )
                            group.append(raw ^ ((raw >> 3) & 112))
                            cells.append(expected)
                        assert (
                            group == list(range(group[0], group[0] + 16, 2))
                            and group[0] % 16 == 0
                        )
                        halves[half].update(range(group[0], group[0] + 16))
                    if thread == 0:
                        registers = cute.make_rmem_tensor(dp[None, 0, 0].shape, dtype)
                        cute.copy(copy, registers, dp[None, 0, 0])
            assert Counter(cells) == Counter(
                (m, k) for m in range(128) for k in range(128)
            )
            assert halves == [set(range(16384)), set(range(16384, 32768))]
            # Include every 128-byte-aligned SW128 phase; no zero-base premise.
            for base in range(0, 1024, 128):
                physical = [
                    {
                        (base + 2 * int(layout.outer((m, k))))
                        ^ (((base + 2 * int(layout.outer((m, k)))) >> 3) & 112)
                        for m in range(128)
                        for k in range(half * 64, (half + 1) * 64)
                    }
                    for half in range(2)
                ]
                assert len(physical[0]) == len(physical[1]) == 8192
                assert not physical[0] & physical[1]
        assert module.operation.verify()
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize("dtype_name", ("BFloat16", "Float16"))
@pytest.mark.parametrize("n", range(32, 257, 32))
@pytest.mark.parametrize("major_b", ("K", "MN"))
def test_actual_mma_eight_k16_read_sets(dtype_name, n, major_b):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = _mlir()
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    mode = {"K": cute.nvgpu.OperandMajorMode.K, "MN": cute.nvgpu.OperandMajorMode.MN}
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = blackwell_helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                mode["K"],
                mode[major_b],
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, n),
                tcgen05.OperandSource.SMEM,
            )
            ac = mma.get_slice(0).partition_A(cute.make_identity_tensor((128, 128)))
            bc = mma.get_slice(0).partition_B(cute.make_identity_tensor((n, 128)))
            assert int(cute.size(ac, mode=[2])) == int(cute.size(bc, mode=[2])) == 8
            for kk in range(8):
                av, bv = ac[None, None, kk], bc[None, None, kk]
                a = {tuple(map(int, av[i])) for i in range(cute.size(av))}
                b = {tuple(map(int, bv[i])) for i in range(cute.size(bv))}
                assert a == {
                    (m, k) for m in range(128) for k in range(kk * 16, (kk + 1) * 16)
                }
                assert b == {
                    (j, k) for j in range(n) for k in range(kk * 16, (kk + 1) * 16)
                }
                assert {k // 64 for _, k in a} == {kk // 4}
        assert module.operation.verify()
