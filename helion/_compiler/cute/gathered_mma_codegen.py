"""Emit an admitted gathered-row pipeline without changing source arithmetic."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .gathered_mma import _clone
from .gathered_mma import _conjuncts
from .gathered_mma import _normalize
from .gathered_mma import _read_names
from .gathered_mma import _replace
from .gathered_mma import _same

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..device_function import DeviceFunction
    from .collective_matmul import CollectiveMmaSite
    from .gathered_mma import GatheredMmaRegion


_PIPELINE = """
tid = cutlass.Int32(cute.arch.thread_idx()[0])
warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
block = cutlass.Int32(cute.arch.block_idx()[0])
expert = cutlass.Int32(_GATHER_GROUP)
n_tiles = cute.ceil_div(_GATHER_N, _GATHER_BN)
n_tile = block // _GATHER_E % n_tiles
m_start = block // (_GATHER_E * n_tiles) * 128
m_grid = cutlass.Int32(cute.arch.grid_dim()[0]) // (_GATHER_E * n_tiles)
m_stride = m_grid * 128
n_start = n_tile * _GATHER_BN
live_rows = cutlass.max(0, _GATHER_ROW_LIMIT)
remaining_rows = cutlass.max(0, live_rows - m_start)
m_tiles = remaining_rows // m_stride + cutlass.Int32(remaining_rows % m_stride != 0)
k_tiles = cute.ceil_div(_GATHER_K, 64)
dtype = out.element_type
ap = cute.arch.alloc_smem(dtype, cute.cosize(al), alignment=1024)
bp = cute.arch.alloc_smem(dtype, cute.cosize(bl), alignment=1024)
sa = cute.make_tensor(cute.recast_ptr(ap, al.inner, dtype=dtype), al.outer)
sb = cute.make_tensor(cute.recast_ptr(bp, bl.inner, dtype=dtype), bl.outer)
rp = cute.arch.alloc_smem(cutlass.Int32, 128 * 2, alignment=16)
sr = cute.make_tensor(rp, cute.make_layout((128, 2), stride=(1, 128)))
cp = cute.arch.alloc_smem(dtype, 128 * 64, alignment=128)
cl = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, cute.make_layout((128, 64), stride=(64, 1)))
sc = cute.make_tensor(cp, cl)
full = cute.arch.alloc_smem(cutlass.Int64, _GATHER_STAGES, alignment=8)
empty = cute.arch.alloc_smem(cutlass.Int64, _GATHER_STAGES, alignment=8)
cfull = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
cempty = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
rfull = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)
rempty = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)
if warp == 0:
    with cute.arch.elect_one():
        for s in cutlass.range_constexpr(_GATHER_STAGES):
            cute.arch.mbarrier_init(full + s, 1)
            cute.arch.mbarrier_init(empty + s, 1)
        cute.arch.mbarrier_init(cfull, 1)
        cute.arch.mbarrier_init(cempty, 128)
        for s in cutlass.range_constexpr(2):
            cute.arch.mbarrier_init(rfull + s, 1)
            cute.arch.mbarrier_init(rempty + s, 128)
        cute.nvgpu.cpasync.prefetch_descriptor(a_atom)
        cute.nvgpu.cpasync.prefetch_descriptor(b_atom)
cute.arch.mbarrier_init_fence()
cute.arch.sync_threads()
holding = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)
alloc_bar = _gather_runtime.pipeline.NamedBarrier(barrier_id=1, num_threads=288)
allocator = _gather_runtime.utils.TmemAllocator(holding, barrier_for_retrieve=alloc_bar)
allocator.allocate(_GATHER_BN)
allocator.wait_for_alloc()
tmem = allocator.retrieve_ptr(cutlass.Float32)
cbase = mma.make_fragment_C(mma.partition_shape_C((128, _GATHER_BN)))
acc = cute.make_tensor(tmem, cbase.layout)
if warp >= 5:
    p_tid = tid - 160
    p_warp = cute.arch.make_warp_uniform(warp - 5)
    producer_bar = _gather_runtime.pipeline.NamedBarrier(barrier_id=2, num_threads=128)
    gb = cute.local_tile(b_tma[None, None, expert], (_GATHER_BN, 64), (n_tile, None))
    sb_part, gb_part = cute.nvgpu.cpasync.tma_partition(b_atom, 0, cute.make_layout(1), cute.group_modes(sb, 0, 2), cute.group_modes(gb, 0, 2))
    stage = cutlass.Int32(0)
    phase = cutlass.Int32(1)
    row_slot = cutlass.Int32(0)
    row_phase = cutlass.Int32(1)
    for mi in cutlass.range(m_tiles, unroll=1):
        cute.arch.mbarrier_wait(rempty + row_slot, row_phase)
        logical_row = cutlass.Int32(m_start + mi * m_stride + p_tid)
        row = _GATHER_ROW(logical_row)
        sr[p_tid, row_slot] = row
        producer_bar.arrive_and_wait()
        if p_warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(rfull + row_slot)
        row_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128), cute.make_layout(4), cute.make_layout(4))
        row_thread = row_copy.get_slice(p_warp)
        shared_ids = row_thread.partition_S(sr[None, row_slot])
        shared_gather = row_thread.partition_S(sa)
        row_ids = cute.make_rmem_tensor(shared_ids.shape, cutlass.Int32)
        cute.autovec_copy(shared_ids, row_ids)
        descriptor = _gather_runtime.tma_descriptor_address(a_atom)
        for ki in cutlass.range(k_tiles, unroll=1):
            cute.arch.mbarrier_wait(empty + stage, phase)
            if p_warp == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, (128 + _GATHER_BN) * 64 * 2)
            producer_bar.arrive_and_wait()
            for gather_slot in cutlass.range_constexpr(cute.size(row_ids, mode=[1])):
                gather_pointer = shared_gather[None, gather_slot, None, stage].iterator
                with cute.arch.elect_one():
                    _gather_runtime.gather_four_rows(descriptor, gather_pointer, full + stage, ki * 64, row_ids[0, gather_slot], row_ids[1, gather_slot], row_ids[2, gather_slot], row_ids[3, gather_slot])
            if p_warp == 0:
                cute.copy(b_atom, gb_part[None, ki], sb_part[None, stage], tma_bar_ptr=full + stage)
            stage += 1
            if stage == _GATHER_STAGES:
                stage = cutlass.Int32(0)
                phase ^= 1
        row_slot ^= 1
        if row_slot == 0:
            row_phase ^= 1
if warp == 4:
    thr = mma.get_slice(0)
    ra = thr.make_fragment_A(thr.partition_A(sa))
    rb = thr.make_fragment_B(thr.partition_B(sb))
    stage = cutlass.Int32(0)
    phase = cutlass.Int32(0)
    cphase = cutlass.Int32(1)
    for _mi in cutlass.range(m_tiles, unroll=1):
        cute.arch.mbarrier_wait(cempty, cphase)
        for ki in cutlass.range(k_tiles, unroll=1):
            cute.arch.mbarrier_wait(full + stage, phase)
            for kk in cutlass.range_constexpr(64 // 16):
                mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, ki != 0 or kk != 0)
                cute.gemm(mma, acc, ra[None, None, kk, stage], rb[None, None, kk, stage], acc)
            with cute.arch.elect_one():
                cute.nvgpu.tcgen05.commit(empty + stage)
            stage += 1
            if stage == _GATHER_STAGES:
                stage = cutlass.Int32(0)
                phase ^= 1
        with cute.arch.elect_one():
            cute.nvgpu.tcgen05.commit(cfull)
        cphase ^= 1
if warp < 4:
    epi_bar = _gather_runtime.pipeline.NamedBarrier(barrier_id=3, num_threads=128)
    acc_mn = _gather_runtime.transform_partitioned_tensor_layout(acc)
    acc_tiles = cute.local_tile(acc_mn, (128, 64), (0, None))
    load_atom = _gather_runtime.blackwell_helpers.get_tmem_load_op((128, _GATHER_BN, 64), _gather_runtime.LayoutEnum.ROW_MAJOR, dtype, cutlass.Float32, (128, 64), False)
    load_copy = cute.nvgpu.tcgen05.make_tmem_copy(load_atom, acc_tiles[None, None, 0])
    load_thr = load_copy.get_slice(tid)
    load_dst = load_thr.partition_D(sc)
    regs = cute.make_rmem_tensor(load_dst.shape, cutlass.Float32)
    narrowed = cute.make_rmem_tensor(load_dst.shape, dtype)
    vector_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128)
    cphase = cutlass.Int32(0)
    row_slot = cutlass.Int32(0)
    row_phase = cutlass.Int32(0)
    for mi in cutlass.range(m_tiles, unroll=1):
        cute.arch.mbarrier_wait(rfull + row_slot, row_phase)
        cute.arch.mbarrier_wait(cfull, cphase)
        for ni in cutlass.range(_GATHER_BN // 64, unroll=1):
            load_src = load_thr.partition_S(acc_tiles[None, None, ni])
            cute.copy(load_copy, load_src, regs)
            narrowed.store(regs.load().to(dtype))
            cute.autovec_copy(narrowed, load_dst)
            epi_bar.arrive_and_wait()
            for vi in cutlass.range_constexpr(128 * 64 // 1024):
                flat = (tid + vi * 128) * 8
                m = flat // 64
                n = flat % 64
                logical_row = cutlass.Int32(m_start + mi * m_stride + m)
                row = sr[m, row_slot]
                col = n_start + ni * 64 + n
                if _GATHER_PRED(logical_row) and col < _GATHER_N:
                    if col + 7 < _GATHER_N and (row * out.stride[0] + col) % 8 == 0:
                        src = cute.make_tensor(sc.iterator + cute.assume(cute.crd2idx((m, n), sc.layout), divby=8), cute.make_layout(8))
                        dst = cute.make_tensor(out.iterator + cute.assume(row * out.stride[0] + col, divby=8), cute.make_layout(8))
                        cute.copy(vector_atom, src, dst)
                    else:
                        for tail in cutlass.range_constexpr(8):
                            if col + tail < out.shape[1]:
                                out[row, col + tail] = sc[m, n + tail]
            epi_bar.arrive_and_wait()
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(cempty)
        cute.arch.mbarrier_arrive(rempty + row_slot)
        cphase ^= 1
        row_slot ^= 1
        if row_slot == 0:
            row_phase ^= 1
cute.arch.sync_threads()
allocator.relinquish_alloc_permit()
allocator.free(tmem)
"""


def _row_limit(
    region: GatheredMmaRegion,
    site: CollectiveMmaSite,
    constants: Mapping[str, int],
    thread_dims: tuple[int, int, int],
) -> ast.expr:
    """Cull complete M tiles using exact integer row bounds from the source.

    A bound is used only if its expression also occurs in the already proved
    integer gather recipe. Arbitrary extra store predicates remain at each
    scatter; a false predicate at one row does not cull other rows.
    """
    result = _clone(region.m_extent)
    for predicate in _conjuncts(region.row_predicate):
        normalized = _normalize(predicate, constants, thread_dims)
        if not (
            isinstance(normalized, ast.Compare)
            and len(normalized.ops) == 1
            and isinstance(normalized.ops[0], ast.Lt)
            and isinstance(normalized.left, ast.Name)
            and normalized.left.id == site.m_index
        ):
            continue
        bound = None
        if isinstance(predicate, ast.Compare) and len(predicate.ops) == 1:
            bound = predicate.comparators[0]
        elif (
            isinstance(predicate, ast.Call)
            and ast.unparse(predicate.func) == "operator.lt"
            and len(predicate.args) == 2
        ):
            bound = predicate.args[1]
        if (
            bound is None
            or site.m_index in _read_names(bound)
            or not any(_same(bound, node) for node in ast.walk(region.row))
        ):
            continue
        result = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="cutlass", ctx=ast.Load()),
                attr="min",
                ctx=ast.Load(),
            ),
            args=[result, _clone(bound)],
            keywords=[],
        )
    return result


def emit_gathered_mma_region(
    region: GatheredMmaRegion,
    site: CollectiveMmaSite,
    df: DeviceFunction,
    *,
    bn: int,
    stages: int,
    constants: Mapping[str, int],
    thread_dims: tuple[int, int, int],
) -> None:
    """Replace only the proved scalar M region and register its launch ABI."""
    assert bn in (128, 256, 512)
    assert stages in (2, 3, 4)
    assert (128 + bn) * 64 * 2 * stages + 18432 <= 232448
    module = ast.parse(_PIPELINE)
    runtime_name = df.new_var("gathered_mma_runtime")
    df.codegen.module_statements.extend(
        ast.parse(
            f"from helion._compiler.cute import gathered_mma_runtime as {runtime_name}"
        ).body
    )
    wrapper_parameters = {
        name: df.new_var(f"gathered_{name}")
        for name in ("a_atom", "b_atom", "b_tma", "mma", "al", "bl")
    }
    local_names = {
        node.id
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    names = {name: df.new_var(f"gathered_{name}") for name in sorted(local_names)}
    names.update(wrapper_parameters)
    names["_gather_runtime"] = runtime_name
    names["out"] = region.output
    grid_cap = min(
        65535,
        ((1 << 31) - 1) // (region.group_count * ((region.n_size + bn - 1) // bn)),
    )
    assert grid_cap >= 1
    values = {
        "_GATHER_GROUP": region.group,
        "_GATHER_N": ast.Constant(region.n_size),
        "_GATHER_K": ast.Constant(region.k_size),
        "_GATHER_E": ast.Constant(region.group_count),
        "_GATHER_BN": ast.Constant(bn),
        "_GATHER_STAGES": ast.Constant(stages),
        "_GATHER_GRID_CAP": ast.Constant(grid_cap),
        "_GATHER_MAX": region.m_extent,
        "_GATHER_ROW_LIMIT": _row_limit(region, site, constants, thread_dims),
    }

    class Instantiate(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id in values:
                return _clone(values[node.id])
            if node.id in names:
                return ast.Name(id=names[node.id], ctx=node.ctx)
            return node

        def visit_Call(self, node: ast.Call) -> ast.expr:
            if isinstance(node.func, ast.Name) and node.func.id in {
                "_GATHER_ROW",
                "_GATHER_PRED",
            }:
                coordinate = self.visit(node.args[0])
                assert isinstance(coordinate, ast.expr)
                expression = (
                    region.row
                    if node.func.id == "_GATHER_ROW"
                    else region.row_predicate
                )
                return _replace(expression, {site.m_index: coordinate})
            result = self.generic_visit(node)
            assert isinstance(result, ast.expr)
            return result

    Instantiate().visit(module)
    ast.fix_missing_locations(module)
    region.m_parent[region.m_position : region.m_position + 1] = module.body
    parameters = list(wrapper_parameters.values())
    plan: dict[str, object] = {
        "kind": "gathered_mma_tma",
        "lhs_name": region.lhs,
        "rhs_name": region.rhs,
        "out_name": region.output,
        "kernel_args": parameters,
        "bn": bn,
        "stages": stages,
        "groups": region.group_count,
        "n_size": region.n_size,
        "grid_cap": grid_cap,
        "source_block": thread_dims,
    }
    extent = region.m_extent
    while (
        isinstance(extent, ast.Call)
        and ast.unparse(extent.func) in {"cutlass.Int32", "cutlass.Int64"}
        and len(extent.args) == 1
        and not extent.keywords
    ):
        extent = extent.args[0]
    if isinstance(extent, ast.Name) and extent.id not in constants:
        plan["m_extent_name"] = extent.id
    else:
        assert isinstance(extent, (ast.Name, ast.Constant))
        value = constants[extent.id] if isinstance(extent, ast.Name) else extent.value
        assert type(value) is int
        plan["m_extent"] = value
    df.wrapper_only_params.extend(parameters)
    df.codegen.cute_wrapper_plans.append(plan)
    df.codegen.cute_uses_matmul = True
