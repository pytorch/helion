"""CPU ownership and allocation proofs for paired column-major TMA outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from test._cute_binding import _forbid_cuda_work

from helion._compiler.cute.pipeline_smem import Tcgen05PipelineSmemFacts
from helion._compiler.cute.pipeline_smem import pipeline_smem_bytes
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
ir = pytest.importorskip("cutlass._mlir").ir
helpers = pytest.importorskip("cutlass.utils.blackwell_helpers")
epilogue = pytest.importorskip("cutlass.utils.gemm.sm100")
LayoutEnum = pytest.importorskip("cutlass.utils.layout").LayoutEnum

pytestmark = skipUnlessBackends(["cute"])

_TILES = ((256, 64, 128), (128, 64, 256))


@pytest.fixture(autouse=True)
def _mlir_module() -> Iterator[None]:
    # Also protects full-suite workers whose CUDA context already exists.
    with _forbid_cuda_work(), ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield


def _mma(tile):
    return helpers.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Float32,
        cute.nvgpu.tcgen05.CtaGroup.TWO,
        tile[:2],
        cute.nvgpu.tcgen05.OperandSource.SMEM,
    )


def _epi_tile(index):
    pm, pn, _pk = _TILES[index]
    return helpers.compute_epilogue_tile_shape(
        (pm // 2 if pm == 128 else pm, pn),
        pm == 128,
        LayoutEnum.COL_MAJOR,
        cutlass.BFloat16,
        layout_c=None if pm == 128 else LayoutEnum.COL_MAJOR,
        elem_ty_c=None if pm == 128 else cutlass.BFloat16,
    )


def _fragment(mma, tile, acc_stages):
    pm, pn, _pk = tile
    return mma.make_fragment_C(cute.append(mma.partition_shape_C((pm, pn)), acc_stages))


def _stage_zero_copies(index, acc_stages):
    """The T2R and R2S copies of accumulator stage zero emitted for ``index``."""
    physical_tile = _TILES[index]
    pm, pn, pk = physical_tile
    mma = _mma(physical_tile)
    tile = _epi_tile(index)
    accumulator = epilogue.transform_partitioned_tensor_layout(
        cute.make_tensor(
            cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16),
            _fragment(mma, physical_tile, acc_stages).layout,
        )
    )
    load = helpers.get_tmem_load_op(
        (pm // 2 if pm == 128 else pm, pn, pk),
        LayoutEnum.COL_MAJOR,
        cutlass.BFloat16,
        cutlass.Float32,
        tile,
        True,
    )
    if index == 0:
        assert str(load) == str(
            cute.make_copy_atom(
                cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x4),
                cutlass.Float32,
            )
        )
    t2r = cute.nvgpu.tcgen05.make_tmem_copy(
        load, cute.flat_divide(accumulator, tile)[None, None, 0, 0, 0]
    )
    store = helpers.get_smem_store_op(
        LayoutEnum.COL_MAJOR, cutlass.BFloat16, cutlass.Float32, t2r
    )
    assert str(store) == str(
        cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4),
            cutlass.BFloat16,
        )
    )
    return mma, tile, t2r, cute.make_tiled_copy_D(store, t2r)


@pytest.mark.parametrize("index,depth,acc2_columns", ((0, 10, 128), (1, 8, 64)))
@pytest.mark.parametrize("acc_stages", (1, 2))
def test_column_major_actual_staged_allocation(index, depth, acc2_columns, acc_stages):
    physical_tile = _TILES[index]
    pm, pn, pk = physical_tile
    mma = _mma(physical_tile)
    fragment = _fragment(mma, physical_tile, acc_stages)
    actual_columns = cutlass.utils.get_num_tmem_alloc_cols(fragment, arch="sm_100")
    assert actual_columns == acc2_columns * acc_stages // 2
    a = helpers.make_smem_layout_a(mma, physical_tile, cutlass.Float8E4M3FN, depth)
    b = helpers.make_smem_layout_b(mma, physical_tile, cutlass.Float8E4M3FN, depth)
    d = helpers.make_smem_layout_epi(
        cutlass.BFloat16, LayoutEnum.COL_MAJOR, _epi_tile(index), 2
    )
    actual_bytes = (
        cute.cosize(a.outer),
        cute.cosize(b.outer),
        2 * cute.cosize(d.outer),
    )
    assert actual_bytes == (pm * pk * depth // 2, pn * pk * depth // 2, 16384)
    assert (
        pipeline_smem_bytes(
            Tcgen05PipelineSmemFacts(1, 2, 232448, output_column_major=True),
            bm=pm,
            bn=pn,
            bk=pk,
            ab_stages=depth,
            c_stages=2,
            acc_stages=acc_stages,
        )
        == sum(actual_bytes) + 2048 + 16 * depth + 16 * acc_stages
    )


@pytest.mark.parametrize("index", (0, 1))
def test_column_major_stage_zero_bf16_copy_ownership(index):
    pm, pn, _pk = _TILES[index]
    tile = _epi_tile(index)
    em, en = cute.size(tile[0]), cute.size(tile[1])
    mma, tile, t2r, r2s = _stage_zero_copies(index, acc_stages=2)
    # ACC1 changes only the trailing TMEM stage mode, so its stage-zero copies
    # (and therefore the ownership proved below) are the ACC2 copies.
    _mma1, _tile1, t2r_acc1, r2s_acc1 = _stage_zero_copies(index, acc_stages=1)
    assert (str(t2r_acc1), str(r2s_acc1)) == (str(t2r), str(r2s))
    threads = cute.size(r2s.thr_id.shape)
    values = cute.size(r2s.layout_src_tv.shape[1])
    # Each thread's register keys and its slots in both D ring stages; they do
    # not depend on the CTA rank or the epilogue subtile.
    slots = []
    for tid in range(128):
        output = r2s.get_slice(tid).partition_D(cute.make_identity_tensor((em, en, 2)))
        stages = [output[None, None, None, stage] for stage in range(2)]
        entries = []
        for repeat in range(cute.size(stages[0].shape) // values):
            for value in range(values):
                i = value + values * repeat
                src_key = (
                    tid // threads,
                    repeat,
                    r2s.layout_src_tv((tid % threads, value)),
                )
                dst_key = (
                    tid // threads,
                    repeat,
                    r2s.layout_dst_tv((tid % threads, value)),
                )
                entries.append((i, src_key, dst_key, [tuple(v[i]) for v in stages]))
        slots.append(entries)
    all_coords = set()
    for rank in range(2):
        coords = epilogue.transform_partitioned_tensor_layout(
            mma.get_slice(rank).partition_C(cute.make_identity_tensor((pm, pn)))
        )
        divided = cute.flat_divide(coords, tile)
        partitions = [t2r.get_slice(tid).partition_D(divided) for tid in range(128)]
        for sub_m in range(pm // 2 // em):
            for sub_n in range(pn // en):
                sources, destinations = {}, [{}, {}]
                for tid, partition in enumerate(partitions):
                    registers = r2s.retile(partition[None, None, None, sub_m, sub_n])
                    assert cute.size(registers.shape) // values * values == len(
                        slots[tid]
                    )
                    for i, src_key, dst_key, stage_slots in slots[tid]:
                        assert src_key not in sources
                        sources[src_key] = tuple(registers[i])
                        for destination, slot in zip(
                            destinations, stage_slots, strict=True
                        ):
                            assert dst_key not in destination
                            destination[dst_key] = slot
                assert len(sources) == len(set(sources.values())) == em * en
                assert not all_coords.intersection(sources.values())
                all_coords.update(sources.values())
                for stage, destination in enumerate(destinations):
                    assert destination.keys() == sources.keys()
                    for key, origin in sources.items():
                        assert destination[key] == (
                            origin[0] - rank * (pm // 2) - sub_m * em,
                            origin[1] - sub_n * en,
                            stage,
                        )
    assert all_coords == {(m, n) for m in range(pm) for n in range(pn)}
