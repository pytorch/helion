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

import ast
import dataclasses
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
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    import cutlass.cute as cute

    from ..device_function import DeviceFunction
    from .tcgen05_epilogue_rounds import Tcgen05EpilogueRoundNames

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


# The K-split cluster's device helpers (``_split_k_cluster_runtime``), imported
# once per generated module.
SPLIT_K_CLUSTER_RUNTIME = "_helion_skc"
_SPLIT_K_CLUSTER_PREAMBLE = f"import helion._compiler.cute._split_k_cluster_runtime as {SPLIT_K_CLUSTER_RUNTIME}"


def split_k_cluster_runtime(device_function: DeviceFunction) -> str:
    """The alias of the K-split cluster's runtime module, importing it once."""
    statements = device_function.codegen.module_statements
    if not any(ast.unparse(stmt) == _SPLIT_K_CLUSTER_PREAMBLE for stmt in statements):
        statements.append(statement_from_string(_SPLIT_K_CLUSTER_PREAMBLE))
    return SPLIT_K_CLUSTER_RUNTIME


@dataclasses.dataclass(frozen=True)
class Tcgen05SplitKClusterNames:
    """Generated-source names of the K-split cluster's exchange, beside the
    rounds renderer's (``Tcgen05EpilogueRoundNames``): the epilogue thread
    index, the pair's split index (its N-lane cluster coordinate), the fp32
    receive buffer, its transaction mbarrier and the runtime alias."""

    rounds: Tcgen05EpilogueRoundNames
    epi_tidx: str
    split_idx: str
    recv: str
    recv_bar: str
    runtime: str


def render_split_k_cluster_exchange(
    names: Tcgen05SplitKClusterNames,
    *,
    acc_rhs: str,
    acc_prelude: str,
    subtile_count: int,
    device_subtile_count: str,
    c_stage_count: int,
    epi_threads: int,
    half_exchange: str | None = None,
) -> str:
    """The TMA-store epilogue of a K-split cluster's output tile
    (``tcgen05_cluster_split_k``): a reduce-scatter of the two pairs' fp32
    partials, then the plain store of this pair's half of the subtiles.

    Pair ``p`` (``names.split_idx``) owns subtiles ``[p * h, (p + 1) * h)``
    of the ``2h`` epilogue subtiles.  Each epilogue thread loads the other
    pair's subtiles from TMEM and stores its fragment into the partner CTA
    (cluster rank ``rank ^ 2``: the other pair's CTA holding the same 64
    rows, whose thread ``t`` owns the same output elements) with
    ``st.async``, 16 bytes per store, laid out ``[subtile][vector][thread]``
    so a warp's 32 stores fill 512 contiguous bytes (conflict-free); then it
    loads its own subtiles, releases the accumulator, waits for the
    partner's bytes on ``names.recv_bar`` (armed with the expected count
    before the accumulator wait) and adds them before the identity
    conversion and the TMA store.  bf16 1024x512x4096 on 128x128 pairs (two
    64x64 subtiles per CTA, 32 values per thread)::

        if tcgen05_epi_tidx == cutlass.Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(skc_recv_bar, 1 * skc_frag * 128 * 4)
        tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)
        for skc_s in cutlass.range_constexpr(1):       # the partner's subtile
            cute.copy(t2r, tTR_tAcc[None, None, None, skc_theirs + skc_s], tTR_rAcc)
            for skc_j in cutlass.range_constexpr(skc_frag // 4):
                _helion_skc.st_async_f32x4(skc_remote_recv + ((skc_s * 8 + skc_j) * 128 + tidx) * 16,
                                           skc_remote_bar, tTR_rAcc[4 * skc_j], ...)
        ...load own subtile; release the accumulator...
        cute.arch.mbarrier_wait(skc_recv_bar, 0)
        for skc_s in cutlass.range_constexpr(1):       # own subtile: add, store
            tTR_rAcc[4 * j + e] = skc_own[skc_s][4 * j + e] + skc_recv_view[skc_s, j, tidx, e]
            ...convert, R2S into the C ring, TMA store of subtile skc_mine + skc_s...

    ``acc_rhs`` converts the carrier (``names.rounds.trs_racc``) once the
    partials are summed into it, after ``acc_prelude``: the epilogue chain's
    steps over the accumulator alone (``torch.relu(acc)``), empty for the
    identity cast.  ``subtile_count`` is the host's count of the plan's
    subtiles, which the generated code asserts against the layout's
    (``device_subtile_count``, a trace-time constant), so a disagreement
    fails the compile instead of leaving subtiles unwritten.

    ``half_exchange`` ("bf16" or "f16", ``tcgen05_split_k_exchange="output"``)
    sends the partials rounded to the 16-bit output dtype, eight per 16-byte
    store into a receive buffer of that dtype, half the bytes; the receiver
    still adds them to its own fp32 partials::

        for skc_j in cutlass.range_constexpr(skc_frag // 8):
            _helion_skc.st_async_b32x4(
                skc_remote_recv + ((skc_s * 4 + skc_j) * 128 + tidx) * 16,
                skc_remote_bar,
                _helion_skc.pack_bf16x2(tTR_rAcc[8 * skc_j], tTR_rAcc[8 * skc_j + 1]),
                ...,
            )
        ...
        tTR_rAcc[8 * j + e] = skc_own[skc_s][8 * j + e] + cutlass.Float32(
            skc_recv_view[skc_s, j, tidx, e]
        )

    No CTA leaves while the partner may still write into it: each one waits
    for all the bytes it expects, and the grid runs one tile per cluster
    (``Tcgen05PersistentProgramIDs``), so a receive buffer and its phase
    serve one tile and a C buffer is reused only within the tile (subtile
    ``s`` stages in buffer ``s % c_stage_count``, acquired past the first
    ``c_stage_count``).
    """
    r = names.rounds
    # (the carrier the conversion reads: in the identity cast itself, or in
    # the chain's prelude, ``acc_loaded = carrier.load()``)
    assert r.trs_racc in acc_rhs + acc_prelude, (r.trs_racc, acc_rhs, acc_prelude)
    if subtile_count % 2:
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05_split_k=2 on a 4-CTA cluster splits the tile's {subtile_count} "
            "epilogue subtiles between its two pairs: an even subtile count",
        )
    half = subtile_count // 2
    rt = names.runtime
    tidx = names.epi_tidx
    # values per 16-byte store, and their bytes
    per, itemsize = (8, 2) if half_exchange else (4, 4)
    vec_stride = per * epi_threads
    body = [
        f"assert cutlass.const_expr({device_subtile_count} == {subtile_count})",
        f"skc_frag = cute.size({r.ttr_racc})",
        *(
            # (a thread's fragment in whole 8-value stores)
            ["assert cutlass.const_expr(skc_frag % 8 == 0)"] if half_exchange else []
        ),
        (
            f"skc_recv_view = cute.make_tensor({names.recv}, cute.make_layout("
            f"({half}, skc_frag // {per}, {epi_threads}, {per}), "
            f"stride=((skc_frag // {per}) * {vec_stride}, {vec_stride}, {per}, 1)))"
        ),
        f"skc_mine = {names.split_idx} * cutlass.Int32({half})",
        f"skc_theirs = (cutlass.Int32(1) - {names.split_idx}) * cutlass.Int32({half})",
        f"if {tidx} == cutlass.Int32(0):",
        (
            f"    cute.arch.mbarrier_arrive_and_expect_tx({names.recv_bar}, "
            f"{half} * skc_frag * {epi_threads} * {itemsize})"
        ),
        f"{r.acc_pipeline}.consumer_wait({r.acc_consumer_state})",
        (
            "skc_peer = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())"
            " ^ cutlass.Int32(2)"
        ),
        f"skc_remote_recv = {rt}.peer_smem_address({names.recv}, skc_peer)",
        f"skc_remote_bar = {rt}.peer_smem_address({names.recv_bar}, skc_peer)",
        f"for skc_s in cutlass.range_constexpr({half}):",
        f"    {r.ttr_tacc_mn} = {r.ttr_tacc}[(None, None, None, skc_theirs + cutlass.Int32(skc_s))]",
        f"    cute.copy({r.tiled_copy_t2r}, {r.ttr_tacc_mn}, {r.ttr_racc})",
        f"    for skc_j in cutlass.range_constexpr(skc_frag // {per}):",
        (
            f"        {rt}.{'st_async_b32x4' if half_exchange else 'st_async_f32x4'}("
            f"skc_remote_recv + ((cutlass.Int32(skc_s) * "
            f"cutlass.Int32(skc_frag // {per}) + cutlass.Int32(skc_j)) * "
            f"cutlass.Int32({epi_threads}) + {tidx}) * cutlass.Int32(16), "
            "skc_remote_bar, "
            + ", ".join(
                f"{rt}.pack_{half_exchange}x2({r.ttr_racc}[8 * skc_j + {2 * w}], "
                f"{r.ttr_racc}[8 * skc_j + {2 * w + 1}])"
                if half_exchange
                else f"{r.ttr_racc}[4 * skc_j + {w}]"
                for w in range(4)
            )
            + ")"
        ),
        (
            f"skc_own = [cute.make_rmem_tensor({r.ttr_racc}.shape, cutlass.Float32)"
            f" for _ in range({half})]"
        ),
        f"for skc_s in cutlass.range_constexpr({half}):",
        (
            f"    cute.copy({r.tiled_copy_t2r}, {r.ttr_tacc}[(None, None, None, "
            "skc_mine + cutlass.Int32(skc_s))], skc_own[skc_s])"
        ),
        "cute.arch.fence_view_async_tmem_load()",
        "with cute.arch.elect_one():",
        f"    {r.acc_pipeline}.consumer_release({r.acc_consumer_state})",
        f"cute.arch.mbarrier_wait({names.recv_bar}, 0)",
    ]
    for subtile in range(half):
        recv = f"skc_recv_view[{subtile}, skc_j, {tidx}, skc_e]"
        if half_exchange:
            recv = f"cutlass.Float32({recv})"
        if subtile >= c_stage_count:
            body.extend(
                [
                    f"if {r.epi_leader_predicate}:",
                    f"    {r.c_pipeline}.producer_acquire()",
                    f"{r.epilog_sync_barrier}.arrive_and_wait()",
                ]
            )
        body.extend(
            [
                f"for skc_j in cutlass.range_constexpr(skc_frag // {per}):",
                f"    for skc_e in cutlass.range_constexpr({per}):",
                (
                    f"        {r.ttr_racc}[{per} * skc_j + skc_e] = skc_own[{subtile}][{per} * skc_j + skc_e]"
                    f" + {recv}"
                ),
                *(line for line in acc_prelude.splitlines() if line.strip()),
                f"{r.acc_vec} = {acc_rhs}",
                f"{r.trs_rd}.store({r.acc_vec})",
                # (the plain loop's barrier ahead of the R2S copy, which
                # publishes the leader's C-buffer acquire; the measured
                # prototype kept it on every subtile)
                f"{r.epilog_sync_barrier}.arrive_and_wait()",
                f"{r.c_buffer} = cutlass.Int32({subtile % c_stage_count})",
                f"cute.copy({r.tiled_copy_r2s}, {r.trs_rd}, {r.trs_sd}[(None, None, None, {r.c_buffer})])",
                "cute.arch.fence_view_async_shared()",
                f"{r.epilog_sync_barrier}.arrive_and_wait()",
                f"if {r.epi_leader_predicate}:",
                (
                    f"    cute.copy({r.tma_store_atom}, {r.bsg_sd}[(None, {r.c_buffer})], "
                    f"{r.bsg_gd}[(None, skc_mine + cutlass.Int32({subtile}))]{r.tma_desc_arg})"
                ),
                f"    {r.c_pipeline}.producer_commit()",
            ]
        )
    return f"if {r.epi_active}:\n" + "".join(f"    {line}\n" for line in body)
