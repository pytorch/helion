"""Trace-time helpers of the tcgen05 split-K partial exchange.

``tcgen05_split_k`` runs each output tile's K loop on ``S`` CTAs. Every CTA
stores its fp32 partial accumulator to a launcher-owned workspace slab, takes a
ticket on the tile's arrival counter, and the last arriver adds the other
partials back into its TMEM accumulator before the unchanged epilogue runs.
The TMEM write-back needs the ``tcgen05.st`` mirror of the epilogue's
``tcgen05.ld`` atom, which this module derives from the load atom at trace
time (the generated code calls it inside the kernel body).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ... import exc
from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import barrier
from ...language import device_print
from ...language import distributed_ops
from ...language import inline_asm_ops
from ...language import memory_ops

if TYPE_CHECKING:
    import cutlass.cute as cute

# ``tcgen05.ld`` op class -> its ``tcgen05.st`` mirror (same datapath shape).
_STORE_OP_NAMES: dict[str, str] = {
    "Ld16x64bOp": "St16x64bOp",
    "Ld16x128bOp": "St16x128bOp",
    "Ld16x256bOp": "St16x256bOp",
    "Ld16x32bx2Op": "St16x32bx2Op",
    "Ld32x32bOp": "St32x32bOp",
}


def tmem_store_atom(load_atom: cute.CopyAtom) -> cute.CopyAtom:
    """The ``tcgen05.st`` copy atom mirroring a ``tcgen05.ld`` epilogue atom
    (same shape op and repetition, so the per-thread register fragment of the
    load partition is exactly the store partition's source)."""
    # Imported here: the renderers below run on hosts without the CuTe DSL.
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    op = load_atom.op
    store_op = getattr(tcgen05, _STORE_OP_NAMES[type(op).__name__])
    return cute.make_copy_atom(store_op(cast("Any", op).repeat), cutlass.Float32)


def split_k_partition(k_tiles: int, splits: int) -> tuple[int, int]:
    """``(quotient, remainder)`` of the balanced K-tile partition: split ``s``
    owns tiles ``[s * q + min(s, r), (s + 1) * q + min(s + 1, r))``, so every
    split gets at least one tile when ``splits <= k_tiles``."""
    return divmod(k_tiles, splits)


def render_split_k_prologue(
    *,
    splits: int,
    k_tiles: int,
    bk: int,
    k_size: int,
    idx: str,
    tile: str,
    tile_begin: str,
    tile_end: str,
    k_begin: str,
    k_end: str,
) -> list[str]:
    """Per-CTA split index and K range of a ``tcgen05_split_k`` launch (the
    flat grid holds ``splits`` consecutive CTAs per output tile)."""
    quotient, remainder = split_k_partition(k_tiles, splits)
    return [
        f"{idx} = cutlass.Int32(cute.arch.block_idx()[0]) % cutlass.Int32({splits})",
        f"{tile} = cutlass.Int32(cute.arch.block_idx()[0]) // cutlass.Int32({splits})",
        f"{tile_begin} = {idx} * cutlass.Int32({quotient}) + cutlass.min({idx}, cutlass.Int32({remainder}))",
        f"{tile_end} = ({idx} + cutlass.Int32(1)) * cutlass.Int32({quotient}) + cutlass.min({idx} + cutlass.Int32(1), cutlass.Int32({remainder}))",
        f"{k_begin} = {tile_begin} * cutlass.Int32({bk})",
        f"{k_end} = cutlass.min({tile_end} * cutlass.Int32({bk}), cutlass.Int32({k_size}))",
    ]


def render_split_k_exchange(
    *,
    splits: int,
    tiles: int,
    bm: int,
    bn: int,
    bk: int,
    epi_threads: int,
    c_layout: str,
    c_dtype: str,
    barrier_id: int,
    workspace: str,
    idx: str,
    tile: str,
    last: str,
    flag: str,
    epi_active: str,
    epi_tidx: str,
    thr_mma: str,
    epi_acc_frag_base: str,
    epi_tile: str,
    epilogue_rest_mode: str,
    acc_pipeline: str,
    acc_consumer_state: str,
) -> str:
    """The partial exchange of a ``tcgen05_split_k`` output tile, emitted
    between the K loop and the epilogue of the flat one-CTA pipeline.

    Every split's epilogue warps copy the accumulator out of TMEM subtile by
    subtile and store it to the CTA's fp32 slab of the launcher-owned
    workspace, laid out per subtile as ``[vector][thread]`` so a warp's
    16-byte stores and loads cover 512 contiguous bytes. After a gpu-scope
    release fence and an epilogue-warp barrier, thread 0 takes a ticket on
    the tile's arrival counter (``atom.add.acq_rel.gpu``). The counter only
    grows across launches: the knob admits power-of-two split counts, so the
    ticket's low bits count this launch's arrivals and every launch starts a
    fresh period without a reset. The last arriver sums every
    split's slab (its own included, so the sum order is fixed) in groups of
    four independent vector loads and writes the total back into its TMEM
    accumulator through the mirrored ``tcgen05.st`` atom; every other split
    releases the accumulator stage and skips the epilogue (the store codegen
    wraps it in ``if <last>``).
    """
    if splits & (splits - 1):
        raise exc.BackendUnsupported(
            "cute", f"tcgen05_split_k={splits} is not a power of two"
        )
    p = "tcgen05_sk"
    tile_elems = bm * bn
    group = 4
    if (bm * bn) % (4 * epi_threads):
        # The slab is exchanged as 16-byte vectors of four fp32 values per
        # epilogue thread; the ``(4, frag // 4)`` views would drop a remainder.
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_split_k exchanges 16-byte vectors: the output tile must "
            "hold a multiple of four fp32 values per epilogue thread",
        )
    lines: list[str] = []
    emit = lines.append
    emit(f"{tile}_elems = cutlass.Int64({tile_elems})")
    emit(f"if {epi_active}:")
    emit(f"    {acc_pipeline}.consumer_wait({acc_consumer_state})")
    emit(
        f"    {p}_tAcc = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout({epi_acc_frag_base})"
    )
    emit(
        f"    {p}_slab = {workspace} + cutlass.Int64({tile} * cutlass.Int32({splits}) + {idx}) * {tile}_elems * cutlass.Int64(4)"
    )
    emit(
        f"    {p}_gC = cute.make_tensor(cute.make_ptr(cutlass.Float32, {p}_slab, cute.AddressSpace.gmem, assumed_align=16), "
        f"cute.make_layout(({bm}, {bn}), stride=({bn}, 1)))"
    )
    emit(
        f"    {p}_tCgC = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout({thr_mma}.partition_C({p}_gC))"
    )
    emit(
        f"    {p}_tCgC_planned = cute.make_tensor({p}_tCgC.iterator, cute.append(cute.append(cute.append({p}_tCgC.layout, {epilogue_rest_mode}), {epilogue_rest_mode}), {epilogue_rest_mode}))"
    )
    # The CUTLASS helper's body, kept inline so the load atom is at hand for
    # the mirrored store atom below.
    emit(
        f"    {p}_ld_atom = cutlass.utils.blackwell_helpers.get_tmem_load_op(({bm}, {bn}, {bk}), {c_layout}, {c_dtype}, cutlass.Float32, {epi_tile}, False)"
    )
    emit(f"    {p}_tAcc_epi = cute.flat_divide({p}_tAcc, {epi_tile})")
    emit(
        f"    {p}_t2r = cute.nvgpu.tcgen05.make_tmem_copy({p}_ld_atom, {p}_tAcc_epi[(None, None, 0, 0, 0)])"
    )
    emit(f"    {p}_thr_t2r = {p}_t2r.get_slice({epi_tidx})")
    emit(f"    {p}_tTR_tAcc_all = {p}_thr_t2r.partition_S({p}_tAcc_epi)")
    emit(f"    {p}_tCgC_epi = cute.flat_divide({p}_tCgC_planned, {epi_tile})")
    emit(f"    {p}_tTR_gC = {p}_thr_t2r.partition_D({p}_tCgC_epi)")
    emit(
        f"    {p}_tTR_rAcc = cute.make_rmem_tensor({p}_tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, cutlass.Float32)"
    )
    emit(
        f"    {p}_tTR_tAcc_stage = {p}_tTR_tAcc_all[None, None, None, None, None, {acc_consumer_state}.index]"
    )
    emit(
        f"    {p}_tTR_tAcc = cute.group_modes({p}_tTR_tAcc_stage, 3, cute.rank({p}_tTR_tAcc_stage))"
    )
    emit(
        f"    {p}_subtiles = cutlass.const_expr(cute.size({p}_tTR_tAcc.shape, mode=[3]))"
    )
    emit(f"    {p}_frag = cutlass.const_expr(cute.size({p}_tTR_rAcc))")
    # Per subtile the slab holds ``epi_threads * frag`` floats as
    # ``[vector j][thread t]`` 16-byte chunks.
    emit(
        f"    {p}_frag_layout = cute.make_layout((4, {p}_frag // 4), stride=(1, {4 * epi_threads}))"
    )
    emit(
        f"    {p}_rfrag = cute.make_tensor({p}_tTR_rAcc.iterator, cute.make_layout((4, {p}_frag // 4)))"
    )
    emit(f"    for _{p}_sub in cutlass.range({p}_subtiles, unroll_full=True):")
    emit(
        f"        cute.copy({p}_t2r, {p}_tTR_tAcc[None, None, None, cutlass.Int32(_{p}_sub)], {p}_tTR_rAcc)"
    )
    emit(
        f"        {p}_off = cutlass.Int64(cutlass.Int32(_{p}_sub) * cutlass.Int32({epi_threads}) * cutlass.Int32({p}_frag) + {epi_tidx} * cutlass.Int32(4)) * cutlass.Int64(4)"
    )
    emit(
        f"        {p}_dst = cute.make_tensor(cute.make_ptr(cutlass.Float32, {p}_slab + {p}_off, cute.AddressSpace.gmem, assumed_align=16), {p}_frag_layout)"
    )
    emit(f"        cute.autovec_copy({p}_rfrag, {p}_dst)")
    emit("    cute.arch.fence_view_async_tmem_load()")
    emit("    cute.arch.fence_acq_rel_gpu()")
    emit(
        f"    {p}_bar = cutlass.pipeline.NamedBarrier(barrier_id={barrier_id}, num_threads={epi_threads})"
    )
    emit(f"    {p}_bar.arrive_and_wait()")
    emit(f"    if {epi_tidx} == cutlass.Int32(0):")
    emit(
        f"        {p}_counter = cute.make_ptr(cutlass.Int32, {workspace} + cutlass.Int64({tiles * splits * tile_elems * 4}) + cutlass.Int64({tile}) * cutlass.Int64(4), cute.AddressSpace.gmem, assumed_align=4)"
    )
    emit(
        f"        {p}_ticket = cute.arch.atomic_add({p}_counter, cutlass.Int32(1), sem='acq_rel', scope='gpu')"
    )
    emit(f"        {p}_arrival = {p}_ticket & cutlass.Int32({splits - 1})")
    emit(f"        {flag}[0] = {p}_arrival")
    emit(f"    {p}_bar.arrive_and_wait()")
    emit(f"    if {flag}[0] == cutlass.Int32({splits - 1}):")
    emit("        cute.arch.fence_acq_rel_gpu()")
    emit(f"        {p}_st_atom = _cute_tmem_store_atom({p}_ld_atom)")
    emit(
        f"        {p}_r2t = cute.nvgpu.tcgen05.make_tmem_copy({p}_st_atom, {p}_tAcc_epi[(None, None, 0, 0, 0)])"
    )
    emit(f"        {p}_thr_r2t = {p}_r2t.get_slice({epi_tidx})")
    emit(f"        {p}_tRT_tAcc_all = {p}_thr_r2t.partition_D({p}_tAcc_epi)")
    emit(
        f"        {p}_tRT_tAcc_stage = {p}_tRT_tAcc_all[None, None, None, None, None, {acc_consumer_state}.index]"
    )
    emit(
        f"        {p}_tRT_tAcc = cute.group_modes({p}_tRT_tAcc_stage, 3, cute.rank({p}_tRT_tAcc_stage))"
    )
    emit(
        f"        {p}_tRT_rAcc = cute.make_rmem_tensor({p}_tTR_rAcc.shape, cutlass.Float32)"
    )
    emit(
        f"        {p}_rsum = cute.make_tensor({p}_tRT_rAcc.iterator, cute.make_layout((4, {p}_frag // 4)))"
    )
    for j in range(min(group, splits)):
        emit(
            f"        {p}_tmp{j} = cute.make_rmem_tensor(cute.make_layout((4, {p}_frag // 4)), cutlass.Float32)"
        )
    emit(f"        for _{p}_sub in cutlass.range({p}_subtiles, unroll_full=True):")
    emit(
        f"            {p}_off = cutlass.Int64(cutlass.Int32(_{p}_sub) * cutlass.Int32({epi_threads}) * cutlass.Int32({p}_frag) + {epi_tidx} * cutlass.Int32(4)) * cutlass.Int64(4)"
    )
    for start in range(0, splits, group):
        count = min(group, splits - start)
        for j in range(count):
            emit(
                f"            {p}_src{j} = cute.make_tensor(cute.make_ptr(cutlass.Float32, {workspace} + cutlass.Int64({tile} * cutlass.Int32({splits}) + cutlass.Int32({start + j})) * {tile}_elems * cutlass.Int64(4) + {p}_off, cute.AddressSpace.gmem, assumed_align=16), {p}_frag_layout)"
            )
        for j in range(count):
            emit(f"            cute.autovec_copy({p}_src{j}, {p}_tmp{j})")
        for j in range(count):
            if start == 0 and j == 0:
                emit(f"            {p}_sum = {p}_tmp0.load()")
            else:
                emit(f"            {p}_sum = {p}_sum + {p}_tmp{j}.load()")
    emit(f"            {p}_rsum.store({p}_sum)")
    emit(
        f"            cute.copy({p}_r2t, {p}_tRT_rAcc, {p}_tRT_tAcc[None, None, None, cutlass.Int32(_{p}_sub)])"
    )
    emit("        cute.arch.fence_view_async_tmem_store()")
    emit("    else:")
    emit("        with cute.arch.elect_one():")
    emit(f"            {acc_pipeline}.consumer_release({acc_consumer_state})")
    emit("cute.arch.sync_threads()")
    emit(f"{last} = {flag}[0] == cutlass.Int32({splits - 1})")
    return "\n".join(lines) + "\n"


def host_function_has_plain_store_body(host_function: object) -> bool:
    """Whether the device program's only side effect is one ``hl.store``.

    ``tcgen05_split_k`` runs each output tile's body on several CTAs and
    guards only the tcgen05 tile store with the last-arriver check; any
    other side effect (an atomic, a second store, a data-dependent branch, a
    barrier, inline asm, a remote copy, a device print) would run once per
    split. The search offers the knob and the pipeline takes it only when
    this holds.
    """
    device_ir = cast("Any", host_function).device_ir
    stores = 0
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            if target is memory_ops.store:
                stores += 1
            elif (
                getattr(target, "__module__", None) in _SIDE_EFFECT_MODULES
                or target is _tracing_ops._if
            ):
                return False
    return stores == 1


# Modules whose ops have effects besides the tcgen05 tile store.
_SIDE_EFFECT_MODULES = frozenset(
    module.__name__
    for module in (atomic_ops, barrier, device_print, distributed_ops, inline_asm_ops)
)
