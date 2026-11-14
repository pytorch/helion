"""Special-purpose CuTe emitter: the tcgen05 store epilogue.

``_codegen_cute_store_tcgen05_tile`` renders the accumulator store of a
tcgen05 GEMM tile (TMEM -> registers -> SMEM/TMA or direct stores) together
with its epilogue variants (unary chain, grouped tail, fragment epilogue,
fanout, rowvec/aux staging).  The registration point stays
``@_decorators.codegen(store, "cute")`` in ``helion/_compiler/cute/memory_ops.py``,
which is the only production caller.  Moved out of
``helion/language/memory_ops.py``; the ``InvalidConfig`` admission checks it
used to carry live in ``tcgen05_config/repair.py``.
"""

from __future__ import annotations

import ast
import dataclasses
import textwrap
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import torch

from .... import exc
from ...ast_extension import expr_from_string
from ...ast_extension import statement_from_string
from ...compile_environment import CompileEnvironment
from ...indexing_strategy import exact_tile_block_ids
from ..cutedsl_compat import emit_pipeline_advance
from ..device_state import CuteTcgen05MatmulPlan
from ..device_state import Tcgen05EpiRowPrefetch
from ..device_state import Tcgen05GroupedDMode
from ..device_state import Tcgen05Orientation
from ..emitters.memory_access import _cute_tensor_dim_size_expr
from ..emitters.memory_access import _cute_tile_begin_expr
from ..emitters.memory_access import _cute_wrap_negative_index
from ..epilogue_fanout import FANOUT_CONFIG_KEY
from ..epilogue_fanout import FanoutIteration
from ..epilogue_fanout import FanoutStore
from ..epilogue_fanout import prove_paired_fanout
from ..epilogue_fanout import render_chain as render_fanout_chain
from ..epilogue_fanout import schedule_supported as fanout_schedule_supported
from ..epilogue_fanout import (
    store_protocol_supported as fanout_store_protocol_supported,
)
from ..state import cute_codegen_state
from ..strategies import tcgen05_explicit_d_store_tile_expr
from ..strategies import tcgen05_is_two_cta_m128
from ..strategies import tcgen05_resolve_epilogue_tile
from ..tcgen05_constants import SM100_SMEM_PER_BLOCK_OPTIN_BYTES
from ..tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP
from ..tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP
from ..tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
from ..tcgen05_constants import TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_BEFORE_STORE
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_DIRECT
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE
from ..tcgen05_constants import TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_ACC_T2R
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_STORE_TAIL
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_NORMAL
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_SPLIT_ACC_T2R_STORE_TAIL
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R
from ..tcgen05_constants import TCGEN05_ROW_LOOKAHEAD_SLOTS
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from ..tcgen05_constants import tcgen05_default_epilogue_tile_size
from ..tcgen05_constants import tcgen05_fixed_smem_overhead_bytes
from ..tcgen05_constants import tcgen05_grouped_worklist_store_shape
from ..tcgen05_epilogue_rounds import Tcgen05EpilogueRoundNames
from ..tcgen05_epilogue_rounds import render_round_subtile_loop

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ...device_function import DeviceFunction
    from ...inductor_lowering import CodegenState
    from ..cute_epilogue import Tcgen05GroupedTailEpilogueMatch
    from ..cute_epilogue import Tcgen05UnaryEpilogueChain
    from ..cute_epilogue import _AuxiliaryTensorLoadExpr
    from ..cute_epilogue import _LeafBindings
    from ..device_state import CuteTcgen05StoreValue
    from ..epilogue_fanout import RenderedChain
    from ..fragment_epilogue import Tcgen05FragmentEpiloguePlan


@dataclasses.dataclass(frozen=True)
class _AuxStepRecord:
    """Per-step splice-side AST locals for one auxiliary chain step.

    Holds the underlying aux tensor name, broadcast/leading-axis metadata, and
    the AST var names allocated for the partition pipeline. ``aux_view2d`` is
    set for broadcast inputs and exact inputs with a leading passthrough axis.
    Used by ``_codegen_cute_store_tcgen05_tile`` to thread per-aux locals through
    the per-output-tile setup and per-subtile load helpers.
    """

    expr: _AuxiliaryTensorLoadExpr
    aux_tensor_name: str
    broadcast_axis: int | None
    has_leading_passthrough: bool
    aux_tile: str
    aux_part_base: str
    aux_xfm: str
    aux_planned: str
    aux_epi: str
    aux_dtype: str
    aux_dtype_bits: int
    aux_extent: int | None
    ttr_aux: str
    ttr_aux_grouped: str
    ttr_aux_subtile: str
    aux_rmem: str
    aux_loaded: str
    aux_view2d: str | None
    # Pre-wait register hoist (bm=128 2-CTA family only): name of the
    # whole-fragment register tensor filled by ``autovec_copy`` BEFORE the
    # accumulator ``consumer_wait`` so the rowvec GMEM latency hides under
    # the MMA wait. ``None`` keeps the per-subtile GMEM load.
    aux_rmem_full: str | None = None
    # Full-tile bm=256 four-CTA path: the per-row value is invariant across
    # every N subtile, so retain it in a scalar register for the whole tile.
    colvec_scalar_full: str | None = None

    @property
    def is_boolean_mask(self) -> bool:
        return self.aux_dtype == "cutlass.Boolean"

    @property
    def edge_rmem_dtype(self) -> str:
        """The register fragment type of element-wise (edge) copies: one
        element of a Boolean global tensor reads as its Int8 storage."""
        return "cutlass.Int8" if self.is_boolean_mask else self.aux_dtype

    def mask_value(self, loaded: str) -> str:
        """The operand value of ``loaded``: a boolean mask loads as its Int8
        storage and is read as ``!= 0``."""
        return f"{loaded} != 0" if self.is_boolean_mask else loaded

    @property
    def edge_rmem_value(self) -> str:
        """The value of the element-wise filled fragment ``aux_rmem``."""
        return self.mask_value(f"{self.aux_rmem}.load()")


def _exact_shape_aux_fragment_hoisted(
    *, requested: bool, broadcast_axis: int | None, aux_ndim: int
) -> bool:
    """Whether ``tcgen05_aux_fragment_hoist`` hoists this auxiliary step.

    Only a rank-2 exact-shape operand (``broadcast_axis is None``) is hoisted;
    row-vector and leading-broadcast operands keep their own family gate. The
    strict normalizer (``aux_fragment_hoist_admissible``) keeps the hoisted
    fragment within ``TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD`` per
    epilogue thread.
    """
    return requested and broadcast_axis is None and aux_ndim == 2


class _RowvecCopyNames(NamedTuple):
    """Locals of one row-vector GMEM-to-SMEM copy (``_rowvec_copy_lines``)."""

    tiled_copy: str
    thr_copy: str
    gmem_tile: str
    gmem_part: str
    smem_part: str
    coord: str
    limit: str
    pred: str


def _rowvec_copy_lines(
    names: _RowvecCopyNames,
    *,
    copy_atom: str,
    dtype: str,
    copy_bits: int,
    threads: int | str,
    copy_elems: int,
    tidx: str,
    aux_tensor_name: str,
    bn: int,
    tile_coord_n: str,
    smem: str,
    predicate: tuple[str, str, int] | None,
) -> list[str]:
    """One N tile of a row operand copied into ``smem`` by ``threads``
    threads of ``copy_elems`` elements each; with ``predicate`` (the N
    size, the tile's N offset and the operand's extent) the columns past the
    row stay uncopied."""
    atom = f"cute.make_copy_atom({copy_atom}, {dtype}, num_bits_per_copy={copy_bits})"
    lines = [
        (
            f"{names.tiled_copy} = cute.make_tiled_copy_tv({atom}, "
            f"cute.make_layout({threads}), cute.make_layout({copy_elems}))"
        ),
        f"{names.thr_copy} = {names.tiled_copy}.get_slice({tidx})",
        (
            f"{names.gmem_tile} = cute.local_tile({aux_tensor_name}, ({bn},), "
            f"({tile_coord_n},))"
        ),
        f"{names.gmem_part} = {names.thr_copy}.partition_S({names.gmem_tile})",
        f"{names.smem_part} = {names.thr_copy}.partition_D({smem})",
    ]
    if predicate is None:
        lines.append(
            f"cute.copy({names.tiled_copy}, {names.gmem_part}, {names.smem_part})"
        )
        return lines
    n_size, n_index, extent = predicate
    lines += [
        (
            f"{names.coord} = {names.thr_copy}.partition_S("
            f"cute.make_identity_tensor({bn}))"
        ),
        (
            f"{names.limit} = min({n_size} - ({n_index}), "
            f"cutlass.Int32({extent}) - ({n_index}), cutlass.Int32({bn}))"
        ),
        (
            f"{names.pred} = cute.make_rmem_tensor((1, cute.size("
            f"{names.smem_part}.shape[1])), cutlass.Boolean)"
        ),
        (
            f"for _rowvec_i in cutlass.range(cute.size({names.smem_part}.shape[1]), "
            "unroll_full=True):"
        ),
        (
            f"    {names.pred}[0, _rowvec_i] = {names.coord}[0, _rowvec_i] < "
            f"{names.limit}"
        ),
        (
            f"cute.copy({names.tiled_copy}, {names.gmem_part}, {names.smem_part}, "
            f"pred={names.pred})"
        ),
    ]
    return lines


def _register_tcgen05_epi_row_prefetch(
    df: DeviceFunction,
    *,
    stage: _RowvecAuxStageRecord,
    aux_tensor_name: str,
    aux_dtype: str,
    bn: int,
    n_size: str,
    n_index: str,
    tile_coord_n: str,
    epi_tidx: str,
) -> None:
    """Hand the role-local loop the asynchronous copy of a tile's row.

    The synchronous copy's predicated form (``_rowvec_copy_lines``) on a
    ``cp.async`` atom, one 16-byte copy per thread of the ``bn``-wide row,
    into ``slot`` of the two-slot stage (``scheduler_pids``'
    ``_tcgen05_row_lookahead_stmts`` places it). Generated for a 256-wide
    bf16 bias row::

        if tcgen05_epi_tidx < cutlass.Int32(32):
            <copy> = cute.make_tiled_copy_tv(cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), cutlass.BFloat16,
                num_bits_per_copy=128), cute.make_layout(32), cute.make_layout(8))
            ...  # partitions of bias at the lookahead tile's N offset
            <dst> = <thr>.partition_D(tcgen05_aux_rowvec_smem_0[slot, None])
            ...  # the predicate: columns below the row's extent
            cute.copy(<copy>, <src>, <dst>, pred=<pred>)

    Threads past the row issue nothing: a predicated-off ``cp.async`` still
    zero-fills its 16 bytes, so on the synchronous copy's layout (every
    epilogue thread, 1024 columns' worth) threads 32-127 zero the other
    slot (4096x30000x128 + bias: max error 4.25).
    """
    threads = _rowvec_copy_threads(bn, stage.copy_elems)

    def issue(translate: Callable[[str], str], slot: str, suffix: str) -> str:
        names = _RowvecCopyNames(
            *(
                df.new_var(f"{stage.smem}_pf_{field}_{suffix}")
                for field in _RowvecCopyNames._fields
            )
        )
        lines = _rowvec_copy_lines(
            names,
            copy_atom="cute.nvgpu.cpasync.CopyG2SOp()",
            dtype=aux_dtype,
            copy_bits=stage.copy_bits,
            threads=threads,
            copy_elems=stage.copy_elems,
            tidx=epi_tidx,
            aux_tensor_name=aux_tensor_name,
            bn=bn,
            tile_coord_n=translate(tile_coord_n),
            smem=f"{stage.smem}[{slot}, None]",
            predicate=(translate(n_size), translate(n_index), stage.aux_extent),
        )
        return f"if {epi_tidx} < cutlass.Int32({threads}):\n" + "\n".join(
            f"    {line}" for line in lines
        )

    df.cute_state.tcgen05.epi_row_prefetches.append(
        Tcgen05EpiRowPrefetch(coordinates=(n_index, tile_coord_n, n_size), issue=issue)
    )


def _rowvec_copy_threads(bn: int, copy_elems: int) -> int:
    """Threads that copy a ``bn``-wide row ``copy_elems`` elements each."""
    return -(-bn // copy_elems)


@dataclasses.dataclass(frozen=True)
class _RowvecAuxStageRecord:
    """Per-tile compact SMEM staging locals for one row-vector aux step."""

    smem_layout: str
    smem_ptr: str
    smem: str
    tiled_copy: str
    thr_copy: str
    gmem_tile: str
    gmem_part: str
    smem_part: str
    coord: str
    limit: str
    pred: str
    copy_bits: int
    copy_elems: int
    aux_extent: int
    warp_private: bool
    # SMEM element type of the stage. It differs from the source only for the
    # promoted 16-bit row vector (``promote``): the row is converted to FP32
    # once, cooperatively, by the epilogue warps (one 128-thread copy plus one
    # epilogue barrier before the accumulator wait), so the per-subtile reads
    # skip the 64 per-thread fp16->fp32 converts the FP32 carrier op would
    # otherwise repeat for every subtile.
    stage_dtype: str = ""
    promote: bool = False
    # Threads of the one cooperative copy filling a promoted stage (a
    # multiple of 32, at most the epilogue warps' threads).
    copy_threads: int = 0
    # The epilogue role loads the row one tile ahead into the other of two
    # SMEM slots (``Tcgen05EpiRowPrefetch``): the per-tile staging only
    # waits for the copy issued while the previous tile ran.
    prefetch: bool = False

    @property
    def copy_names(self) -> _RowvecCopyNames:
        """The copy's locals, as ``_rowvec_copy_lines`` takes them."""
        return _RowvecCopyNames(
            self.tiled_copy,
            self.thr_copy,
            self.gmem_tile,
            self.gmem_part,
            self.smem_part,
            self.coord,
            self.limit,
            self.pred,
        )


def _tcgen05_rowvec_aux_stage_copy_elems(
    aux_dtype_bits: int,
    block_n: int,
    aux_extent: int | None,
    *,
    copy_bits: int = 128,
) -> int | None:
    """Return the vector width when a row-vector aux can be staged safely."""
    from ..cute_epilogue import rowvec_aux_stage_copy_elems

    return rowvec_aux_stage_copy_elems(
        aux_dtype_bits, block_n, aux_extent, copy_bits=copy_bits
    )


def _tcgen05_segment_store_matches_proof(
    state: CodegenState,
    tcgen05_value: CuteTcgen05StoreValue,
) -> bool:
    from ..cute_mma import _rank3_rhs_broadcast_mask_base
    from ..cute_mma import _trace_to_outer_graph_arg

    store_node = tcgen05_value.segment_store_node
    if store_node is None or state.fx_node is not store_node:
        return False
    index = store_node.args[1] if len(store_node.args) > 1 else None
    extra_mask = store_node.args[3] if len(store_node.args) > 3 else None
    if (
        not isinstance(index, (list, tuple))
        or len(index) != 2
        or not isinstance(index[0], torch.fx.Node)
        or not isinstance(extra_mask, torch.fx.Node)
        or tcgen05_value.segment_store_row_index is None
        or tcgen05_value.segment_store_valid_m is None
    ):
        return False
    store_row = _trace_to_outer_graph_arg(state.codegen, index[0])
    if store_row is not tcgen05_value.segment_store_row_index:
        return False
    store_mask = _trace_to_outer_graph_arg(state.codegen, extra_mask)
    store_valid_m = _rank3_rhs_broadcast_mask_base(store_mask, broadcast_dim=1)
    if store_valid_m is None:
        return False
    store_valid_m = _trace_to_outer_graph_arg(state.codegen, store_valid_m)
    return store_valid_m is tcgen05_value.segment_store_valid_m


def _cute_leading_passthrough_view_2d(
    result_name: str,
    tensor_name: str,
    leading_index: str,
) -> str:
    """Select one leading coordinate while preserving trailing matrix strides."""
    return (
        f"{result_name} = cute.make_tensor("
        f"{tensor_name}.iterator + cute.crd2idx("
        f"(cutlass.Int32({leading_index}), cutlass.Int32(0), cutlass.Int32(0)), "
        f"{tensor_name}.layout), "
        f"cute.make_layout(({tensor_name}.shape[1], {tensor_name}.shape[2]), "
        f"stride=({tensor_name}.layout.stride[1], "
        f"{tensor_name}.layout.stride[2])))"
    )


def _destination_gemm_extents(
    env: CompileEnvironment, tensor: torch.Tensor, output_block_ids: Sequence[int]
) -> tuple[int, int] | None:
    """The tile loops' static (M, N) where a rank-2 destination is larger
    than them, else ``None``."""
    if tensor.ndim != 2 or len(output_block_ids) < 2:
        return None
    rows, cols = (env.block_sizes[block_id].size for block_id in output_block_ids[-2:])
    dest_rows, dest_cols = tensor.shape
    if not all(type(v) is int for v in (rows, cols, dest_rows, dest_cols)):
        return None
    rows, cols = cast("int", rows), cast("int", cols)
    if (rows, cols) == (dest_rows, dest_cols) or rows > dest_rows or cols > dest_cols:
        return None
    return rows, cols


def _check_tcgen05_tma_store_destination(
    tensor: torch.Tensor,
    tensor_name: str,
    *,
    transposed_store: bool,
    output_column_major: bool,
    radix_store: bool = False,
) -> None:
    """Refuse a TMA store whose destination is known to break TensorMap rules.

    The pipeline decision proves plain destinations and falls back to the
    SIMT store otherwise; this is the last line for the paths it does not
    gate (grouped and row-union stores, untraced fan-outs).  A static stride
    or storage offset that is not a 16-byte multiple, an argument whose
    bound base residue is not zero, or a plain MN destination whose
    contiguous axis is not the one the staged D layout assumes would be
    stored silently wrong, so fail loudly instead.  Unknown (symbolic)
    facts are not judged here.
    """
    from ..cute_mma import _tcgen05_tma_matrix_major
    from ..memory_ops import _bound_vec_alignment_signature

    env = CompileEnvironment.current()
    element_size = tensor.element_size()
    problems: list[str] = []
    for dim, stride in enumerate(tensor.stride()):
        if type(stride) is int and stride != 1 and (stride * element_size) % 16:
            problems.append(f"stride({dim})={stride} ({stride * element_size} B)")
    offset = tensor.storage_offset()
    if type(offset) is int and (offset * element_size) % 16:
        problems.append(f"storage_offset={offset} ({offset * element_size} B)")
    signature = _bound_vec_alignment_signature(env, env.tensor_input_source(tensor))
    if signature is not None and signature[0] != 0:
        problems.append(f"base address residue {signature[0]} B")
    # A radix store addresses the destination through its proved hierarchical
    # [M, N] view, whatever the tensor's own orientation.
    if not transposed_store and not radix_store and tensor.ndim in (2, 3):
        expected_major = "col" if output_column_major else "row"
        if _tcgen05_tma_matrix_major(tensor) != expected_major:
            problems.append(
                f"{'N' if expected_major == 'row' else 'M'}-contiguous storage expected"
            )
    if problems:
        raise exc.BackendUnsupported(
            "cute",
            f"the tcgen05 TMA store cannot address {tensor_name}: "
            + ", ".join(problems)
            + " (a TensorMap needs a 16-byte base and 16-byte outer strides)",
        )


# The staged SIMT store's SMEM tile at most (a 128x128 bf16 tile with its
# padding is 34 KiB).
_STAGED_SIMT_STORE_MAX_BYTES = 64 * 1024
# Slack for the allocations' alignment padding.
_STAGED_SIMT_STORE_ALIGN_SLACK_BYTES = 1024


def _staged_simt_store_fits(plan: CuteTcgen05MatmulPlan, staged_bytes: int) -> bool:
    """Whether the staged store's SMEM tile fits beside the plan's AB ring and
    fixed reservations (the SIMT store allocates no C ring), pricing the
    operands at four bytes (tf32) whatever their dtype: the plan does not
    record it, and the narrow flat tiles that stage keep shallow rings."""
    if plan.m_subtile_count > 1:
        # M-paired tiles allocate a second A ring
        return False
    used = (
        tcgen05_ab_smem_bytes_per_cta(
            bm=plan.bm,
            bn=plan.bn,
            bk=plan.bk,
            dtype_bytes=4,
            ab_stages=plan.ab_stage_count,
            cluster_m=plan.cluster_m,
        )
        + tcgen05_fixed_smem_overhead_bytes(
            ab_stages=plan.ab_stage_count,
            acc_stages=plan.acc_stage_count,
            cluster_m=plan.cluster_m,
        )
        + staged_bytes
        + _STAGED_SIMT_STORE_ALIGN_SLACK_BYTES
    )
    return used <= SM100_SMEM_PER_BLOCK_OPTIN_BYTES


def _staged_simt_store_lines(
    df: DeviceFunction,
    *,
    tile: tuple[int, int],
    pad: int,
    extents: tuple[int, int],
    tile_coord: tuple[str, str],
    dtype: str,
    thr_mma: str,
    thr_copy_t2r: str,
    epi_tile: str,
    epi_tidx: str,
    epi_active: str,
    epi_warps: int,
    barrier_id: int,
    registers: str,
    gmem_tile: str,
) -> tuple[list[str], list[str], str, str]:
    """The staged SIMT store of a flat kernel's one tile: (setup, coordinate
    setup, per-subtile write, store). Each subtile's T2R fragment lands in a padded SMEM
    tile at its tile coordinates; after the last subtile the epilogue warps
    store the tile row by row, lane ``l`` of each warp writing column
    ``32 * c + l`` of its rows. With rows of 1001 bf16 (2002 bytes) the
    direct copy's warp instruction covers 32 rows, one 2-byte element each::

        cute.copy(simt_atom, tTR_rD, tTR_gC_subtile)  # 32 sectors / instr

    while each staged instruction covers 32 consecutive elements of one
    row::

        tile[coord[i]] = tTR_rD[i]  # SMEM, padded rows
        barrier.arrive_and_wait()
        gC_tile[row, 32 * c + lane] = tile[row, 32 * c + lane]

    The row padding (16 bytes) spreads a warp's column writes over the SMEM
    banks: without it the rows of a 256-byte tile row alias the same banks
    (64 x 1000x1000x64 into 1001-element rows: 114 us unpadded, 96 us
    padded)."""
    bm, bn = tile
    m_size, n_size = extents
    staged_ptr = df.new_var("tcgen05_staged_ptr")
    staged_tile = df.new_var("tcgen05_staged_tile")
    staged_barrier = df.new_var("tcgen05_staged_barrier")
    coord = df.new_var("tcgen05_staged_coord")
    coord_part = df.new_var("tcgen05_staged_coord_part")
    coord_grouped = df.new_var("tcgen05_staged_coord_grouped")
    coord_subtile = df.new_var("tcgen05_staged_coord_subtile")
    bounds = df.new_var("tcgen05_staged_bounds")
    rows_per_warp = bm // epi_warps
    setup = [
        (
            f"{staged_ptr} = cute.arch.alloc_smem({dtype}, "
            f"cutlass.Int32({bm * (bn + pad)}), alignment=16)"
        ),
        (
            f"{staged_tile} = cute.make_tensor({staged_ptr}, "
            f"cute.make_layout(({bm}, {bn}), stride=({bn + pad}, 1)))"
        ),
        (
            f"{staged_barrier} = cutlass.pipeline.NamedBarrier("
            f"barrier_id=cutlass.Int32({barrier_id}), "
            f"num_threads=cutlass.Int32({epi_warps * 32}))"
        ),
    ]
    # The fragment's tile coordinates: the output partition's chain applied
    # to the tile's identity tensor.
    tile_setup = [
        f"{coord} = {thr_mma}.partition_C(cute.make_identity_tensor(({bm}, {bn})))",
        (
            f"{coord_part} = {thr_copy_t2r}.partition_D(cute.flat_divide("
            "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{coord}), {epi_tile}))"
        ),
        f"{coord_grouped} = cute.group_modes({coord_part}, 3, cute.rank({coord_part}))",
        (
            f"{bounds} = cute.local_tile(cute.make_identity_tensor("
            f"({m_size}, {n_size})), ({bm}, {bn}), "
            f"({tile_coord[0]}, {tile_coord[1]}))"
        ),
    ]
    write = (
        f"        {coord_subtile} = {coord_grouped}["
        "(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
        f"        for _staged_i in range(cute.size({registers}.shape)):\n"
        f"            {staged_tile}[{coord_subtile}[_staged_i]] = "
        f"{registers}[_staged_i]\n"
    )
    store = (
        f"if {epi_active}:\n"
        f"    {staged_barrier}.arrive_and_wait()\n"
        f"    for _staged_r in range({rows_per_warp}):\n"
        f"        _staged_row = {epi_tidx} // cutlass.Int32(32) * "
        f"cutlass.Int32({rows_per_warp}) + cutlass.Int32(_staged_r)\n"
        f"        for _staged_c in range({bn // 32}):\n"
        f"            _staged_col = cutlass.Int32(_staged_c * 32) + "
        f"{epi_tidx} % cutlass.Int32(32)\n"
        f"            if cute.elem_less({bounds}[_staged_row, _staged_col], "
        f"({m_size}, {n_size})):\n"
        f"                {gmem_tile}[_staged_row, _staged_col] = "
        f"{staged_tile}[_staged_row, _staged_col]\n"
        # the next tile's fragments overwrite the SMEM tile
        f"    {staged_barrier}.arrive_and_wait()\n"
    )
    return setup, tile_setup, write, store


def _codegen_cute_store_tcgen05_tile(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_name: str,
    epilogue_chain: Tcgen05UnaryEpilogueChain | None = None,
    grouped_tail_epilogue: Tcgen05GroupedTailEpilogueMatch | None = None,
    fragment_epilogue: Tcgen05FragmentEpiloguePlan | None = None,
) -> list[ast.AST] | ast.AST | None:
    df = state.device_function
    candidate_names = df.variable_aliases(value_name)
    tcgen05_value = df.cute_state.get_tcgen05_store_value(candidate_names)
    if tcgen05_value is None:
        return None
    row_union = tcgen05_value.row_union
    proved_segment_store = bool(
        tcgen05_value.segment_store_m_offset
        and tcgen05_value.segment_store_start
        and tcgen05_value.segment_store_actual_m
    )
    if proved_segment_store and (
        extra_mask is None
        or not _tcgen05_segment_store_matches_proof(state, tcgen05_value)
    ):
        return None
    # Both routes consume the same exact store/offset proof. The dense route
    # replaces only the segment-local address with its allocation-wide tile.
    segment_store = proved_segment_store and row_union is None
    if row_union is not None and not proved_segment_store:
        raise exc.BackendUnsupported("cute", "row-union store lost its segment proof")
    if extra_mask is not None:
        grouped_tail_mask = (
            grouped_tail_epilogue is not None
            and grouped_tail_epilogue.store_node is state.fx_node
            and grouped_tail_epilogue.store_mask is not None
            and len(grouped_tail_epilogue.store_node.args) == 4
            and grouped_tail_epilogue.store_node.args[3]
            is grouped_tail_epilogue.store_mask
        )
        if not proved_segment_store and not grouped_tail_mask:
            return None
    # ``radix_store`` imports ``helion.language``, which imports this module.
    from ..radix_store import proven_radix_store_layout

    env = CompileEnvironment.current()
    store_node = state.fx_node
    # A fused view/permute store into a hierarchical [M, N] destination
    # (``radix_store``; [M, N, L] for a batched matmul's): the tile store,
    # addressed through the proved layout by the TMA store.
    radix_layout = (
        proven_radix_store_layout(store_node) if store_node is not None else None
    )
    if tensor.ndim not in (2, 3) and radix_layout is None:
        return None
    # A batched store's TMA tile sits at the batch coordinate of its L mode,
    # the destination's own or the proved layout's.
    leading_passthrough_output = (
        tensor.ndim == 3 if radix_layout is None else bool(radix_layout.batch)
    )
    store_subscripts = store_node.args[1] if store_node is not None else None
    actual_block_ids = None
    if isinstance(store_subscripts, (list, tuple)):
        if proved_segment_store:
            # The registered segment proof ties the dynamic row index to the
            # work and M axes. The remaining N index must still be the exact
            # zero-offset output tile, just as it is for ordinary stores.
            actual_block_ids = exact_tile_block_ids(env, store_subscripts[1:])
            expected_block_ids = tcgen05_value.output_block_ids[-1:]
        else:
            actual_block_ids = exact_tile_block_ids(env, store_subscripts)
            expected_block_ids = tcgen05_value.output_block_ids
    else:
        expected_block_ids = tcgen05_value.output_block_ids
    if (
        fragment_epilogue is not None
        and fragment_epilogue.store_node is store_node
        and fragment_epilogue.changes_shape
    ):
        # The committed planner exhaustively proved the derived indices are
        # the exact compact destination tile. ``exact_tile_block_ids`` cannot
        # see through the reshape/split/floor-divide address expression.
        actual_block_ids = expected_block_ids
    if radix_layout is not None:
        # ``radix_store`` proved the subscript addresses the output tile
        # through the hierarchical layout. Only the TMA store addresses the
        # destination through it: the SIMT fringe store and the direct store
        # would index the destination's own (higher-rank) layout, so every
        # tile must be a full TMA-store tile, and each explicit epilogue box
        # must sit inside one innermost run of the layout.
        actual_block_ids = expected_block_ids
        if (
            not tcgen05_value.use_tma_store_epilogue
            or tcgen05_value.tma_store_full_tiles_only
            or not tcgen05_value.has_explicit_epilogue_tile
            or grouped_tail_epilogue is not None
            or fragment_epilogue is not None
            or row_union is not None
            or segment_store
            or tcgen05_value.orientation is Tcgen05Orientation.NM
            or len(expected_block_ids) != (3 if radix_layout.batch else 2)
            or tcgen05_value.output_column_major != radix_layout.column_major
            or not radix_layout.tma_store_supported(
                rows_per_store=cast("int", tcgen05_value.explicit_epi_tile_m),
                cols_per_store=cast("int", tcgen05_value.explicit_epi_tile_n),
                element_bytes=tensor.element_size(),
            )
            or not radix_layout.tiles(
                tcgen05_value.source_tile_m, tcgen05_value.source_tile_n
            )
        ):
            raise exc.BackendUnsupported(
                "cute",
                "a hierarchical (view/permute) store destination requires the "
                "full-tile TMA-store epilogue, an explicit epilogue "
                "tile inside the layout's innermost runs and CTA tiles that "
                "partition the layout",
            )
    if actual_block_ids != expected_block_ids:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 matmul store requires the exact zero-offset output tile axes",
        )
    if row_union is not None and (
        tensor.ndim != 2
        or epilogue_chain is not None
        or grouped_tail_epilogue is not None
        or fragment_epilogue is not None
        or (tcgen05_value.use_tma_store_epilogue and row_union.schedule is None)
    ):
        raise exc.BackendUnsupported(
            "cute", "row-union requires the proved rank-2 identity SIMT store"
        )
    if tcgen05_value.orientation is Tcgen05Orientation.NM and (
        epilogue_chain is not None
        or grouped_tail_epilogue is not None
        or fragment_epilogue is not None
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 N,M-oriented worklist path supports only an identity store",
        )
    assert (
        sum(
            value is not None
            for value in (epilogue_chain, grouped_tail_epilogue, fragment_epilogue)
        )
        <= 1
    )
    # When one matmul accumulator fans out to multiple output stores (e.g.
    # aux = pre-activation and out = gelu(pre)), the per-matmul TMA-store
    # atom/tensor kernel-arg names allocated in cute_mma are shared by every
    # store site. Emitting them verbatim at each site produces duplicate kernel
    # parameters (SyntaxError) and binds both device epilogues to the same TMA
    # descriptor. The secondary store gets fresh per-store descriptor names so
    # each store threads its own TMA descriptor; the first store keeps the
    # original names. The secondary store also reuses the accumulator the first
    # store already waited for. The final store releases/advances the pipeline
    # only after every store has read TMEM, preventing the next persistent
    # tile's producer from overwriting the stage during fanout. The matmul
    # drain / TMEM-free teardown is emitted once.
    is_secondary_store, is_final_store = df.cute_state.claim_tcgen05_store_site(
        tcgen05_value, store_node
    )
    has_store_fanout = is_secondary_store or not is_final_store
    fanout_plan = None
    fanout_chain: RenderedChain | None = None
    fanout_iteration: FanoutIteration | None = None
    fanout_advance: str | None = None
    if has_store_fanout and df.config.get(FANOUT_CONFIG_KEY, "off") == "shared":
        assert tcgen05_value.output_stores is not None
        fanout_plan = prove_paired_fanout(
            env,
            state.codegen.host_function,
            state.codegen.codegen_graphs,
            tcgen05_value.output_stores,
            df.cute_state.tcgen05.matmul_fx_nodes,
        )
        if fanout_plan is not None and not (
            fanout_schedule_supported(df.config.config)
            and fanout_store_protocol_supported(tcgen05_value)
            and fragment_epilogue is None
            and grouped_tail_epilogue is None
            and epilogue_chain is not None
        ):
            raise exc.BackendUnsupported(
                "cute", "shared fanout requires the standard full-tile TMA protocol"
            )
    if has_store_fanout and fragment_epilogue is not None:
        raise exc.BackendUnsupported(
            "cute", "tcgen05 fragment epilogues require one output store"
        )
    if is_secondary_store and tcgen05_value.use_tma_store_epilogue:
        tcgen05_value = dataclasses.replace(
            tcgen05_value,
            tma_store_atom=df.new_var("tcgen05_tma_store_atom"),
            tma_store_tensor=df.new_var("tcgen05_tma_store_tensor"),
        )
    tcgen05_lifecycle = tcgen05_value.lifecycle_context

    # Every output reads the unchanged live stage. Only the final store
    # advances it, so no snapshot can be hoisted outside a persistent tile loop.
    tcgen05_acc_stage_index_expr = f"{tcgen05_lifecycle.acc_consumer_state}.index"

    # Backstop for callers that bypass Config.normalize() validation;
    # see _tcgen05_epi_warp_count docstring and cute_plan.md.
    if tcgen05_value.epi_warp_count != 4:
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 SIMT-store epilogue requires "
            f"tcgen05_num_epi_warps=4 (got {tcgen05_value.epi_warp_count}). "
            "CUTLASS tmem_warp_shape_mn=(4,1) hard-codes a 4-warp t2r "
            "partition for the supported tcgen05 path; per-warp "
            "tcgen05.ld semantics make the partition uncoverable by "
            "fewer warps. Lifts when the c_pipeline-driven multi-warp "
            "epilogue lands (see cute_plan.md).",
        )

    backend = env.backend
    tensor_name = df.tensor_arg(tensor).name
    target_dtype = backend.dtype_str(tensor.dtype)
    # Every store of the accumulator walks the matmul plan's subtiles
    # (`epi_elem_dtype`, the widest output store's; the tile expression is
    # shown at ``cute_mma._tcgen05_epilogue_tile_dtype``) and stages and
    # stores them in its own `target_dtype`: beside an FP32
    # `out0[tile] = acc` on a 128x64 tile, a BF16
    # `out2[tile] = (acc + aux).to(bf16)` stages the FP32-sized (128, 16)
    # subtiles as
    #
    #   tcgen05_sD_layout = make_smem_layout_epi(cutlass.BFloat16,
    #       ROW_MAJOR, tcgen05_store_epi_tile, 2)
    #
    # and the wrapper builds its TMA store atom from the same tile
    # (``epi_tile_raw_expr`` below). Sized by `target_dtype` instead, out2
    # walks (128, 32) subtiles while its epilogue-issued aux ring
    # (``tcgen05_aux_load_mode="epi_tma"``) holds the plan's (128, 16) ones,
    # and the copy out of the ring fails to build (16 staged values against
    # 32 per thread). A store wider than the plan's dtype (its
    # forward trace failed and the plan fell back to the input dtype) would
    # outgrow the C ring the SMEM model prices, so it is refused.
    epi_tile_torch_dtype = tcgen05_value.epi_elem_dtype or tensor.dtype
    epi_tile_dtype = backend.dtype_str(epi_tile_torch_dtype)
    epi_tile_bits = epi_tile_torch_dtype.itemsize * 8
    if tensor.dtype.itemsize > epi_tile_torch_dtype.itemsize:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 epilogue element-type mismatch: matmul plan was set "
            f"up with the epilogue dtype {epi_tile_dtype!r} but the store "
            f"target tensor dtype is the wider {target_dtype!r}.",
        )
    if tcgen05_value.orientation is Tcgen05Orientation.NM and target_dtype not in (
        "cutlass.Float16",
        "cutlass.BFloat16",
        "cutlass.Float32",
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 N,M-oriented worklist path requires an FP16/BF16/FP32 store",
        )
    tcgen05_d_store_layout = tcgen05_value.d_store_layout
    tcgen05_nm_store = tcgen05_value.orientation is Tcgen05Orientation.NM
    tcgen05_nm_store_shape = (
        (row_union.schedule.epi_m, row_union.schedule.epi_n, row_union.schedule.epi_n)
        if row_union is not None and row_union.schedule is not None
        else tcgen05_grouped_worklist_store_shape(tcgen05_value.bm, tcgen05_value.bn)
    )
    if (
        tcgen05_nm_store
        and (
            tcgen05_value.explicit_epi_tile_m,
            tcgen05_value.explicit_epi_tile_n,
            tcgen05_value.explicit_d_store_box_n,
        )
        != tcgen05_nm_store_shape
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 N,M-oriented store requires explicit "
            f"epi_tile={tcgen05_nm_store_shape[:2]} and "
            f"d_store_box_n={tcgen05_nm_store_shape[2]}",
        )
    if radix_layout is None and len(subscript) != (
        3 if leading_passthrough_output else 2
    ):
        return None
    if fragment_epilogue is not None and fragment_epilogue.changes_shape:
        source_base_indices = [
            _cute_tile_begin_expr(state, size)
            for size in fragment_epilogue.store_tile_sizes
        ]
        column_ratio = (
            fragment_epilogue.source_shape[-1]
            // fragment_epilogue.destination_shape[-1]
        )
        base_indices = [
            *source_base_indices[:-1],
            f"({source_base_indices[-1]}) // cutlass.Int32({column_ratio})",
        ]
    elif row_union is not None:
        base_indices = [
            tcgen05_value.segment_store_m_offset,
            _cute_tile_begin_expr(state, subscript[1]),
        ]
    elif segment_store:
        base_indices = [
            "",
            _cute_tile_begin_expr(state, subscript[1]),
        ]
    elif radix_layout is not None:
        base_indices = [
            _cute_tile_begin_expr(state, env.block_sizes[block_id].var)
            for block_id in expected_block_ids
        ]
    else:
        base_indices = [
            _cute_tile_begin_expr(
                state, _cute_wrap_negative_index(idx, tensor.shape[dim])
            )
            for dim, idx in enumerate(subscript)
        ]
    leading_index = base_indices[0] if leading_passthrough_output else None
    m_index, n_index = base_indices[-2:]
    if radix_layout is not None:
        m_size, n_size = (str(extent) for extent in radix_layout.extents)
    else:
        m_size = _cute_tensor_dim_size_expr(state, tensor, tensor.ndim - 2)
        n_size = _cute_tensor_dim_size_expr(state, tensor, tensor.ndim - 1)
    if row_union is not None:
        m_size, n_size = str(row_union.m), str(row_union.n)
    # A destination larger than the GEMM (an ``out`` wider than N, written
    # over the tile loops' N) has its rows and columns past the loops written
    # by others: the edge predicates and the TMA store's TensorMap view stop
    # at the loops' extents, not the destination's (the TMA store's edge
    # tiles zeroed columns 576-639 of a 640-wide output).
    gemm_extents = (
        _destination_gemm_extents(env, tensor, tcgen05_value.output_block_ids)
        if radix_layout is None
        and row_union is None
        and not segment_store
        and fragment_epilogue is None
        and grouped_tail_epilogue is None
        and not leading_passthrough_output
        and tcgen05_value.orientation is not Tcgen05Orientation.NM
        else None
    )
    if gemm_extents is not None:
        m_size, n_size = (str(extent) for extent in gemm_extents)
    segment_store_local_m = tcgen05_value.segment_store_m_offset
    segment_store_base_m = (
        f"cutlass.Int32({tcgen05_value.segment_store_start}) + {segment_store_local_m}"
        if segment_store
        else ""
    )
    matmul_plan = df.cute_state.tcgen05.matmul_plan
    segment_grouped = (
        matmul_plan.grouped if segment_store and matmul_plan is not None else None
    )
    grouped_fixed_d_tensormap = (
        segment_store
        and segment_grouped is not None
        and segment_grouped.fixed_tensormaps
    )
    grouped_bounded_output_shift = bool(
        segment_grouped is not None and segment_grouped.bounded_output_shift
    )
    tcgen05_source_bm = tcgen05_value.source_tile_m
    tcgen05_source_bn = tcgen05_value.source_tile_n
    tile_coord_m = (
        "cutlass.Int32(0)"
        if segment_store
        else f"({m_index}) // cutlass.Int32({tcgen05_source_bm})"
    )
    compact_fragment_epilogue = (
        fragment_epilogue
        if fragment_epilogue is not None and fragment_epilogue.changes_shape
        else None
    )
    compact_fragment_store = compact_fragment_epilogue is not None
    tcgen05_destination_bm = (
        compact_fragment_epilogue.destination_shape[-2]
        if compact_fragment_epilogue is not None
        else tcgen05_source_bm
    )
    tcgen05_destination_bn = (
        compact_fragment_epilogue.destination_shape[-1]
        if compact_fragment_epilogue is not None
        else tcgen05_source_bn
    )
    # NM grouped stores expose transposed logical source dimensions, while the
    # wrapper already applies the NM orientation when constructing the D
    # TensorMap. Keep the descriptor and device local tile in the physical MMA
    # orientation; MN compact-fragment stores still use their proven
    # destination shape.
    tcgen05_tma_store_bm = (
        tcgen05_value.bm if tcgen05_nm_store else tcgen05_destination_bm
    )
    tcgen05_tma_store_bn = (
        tcgen05_value.bn if tcgen05_nm_store else tcgen05_destination_bn
    )
    tile_coord_n = f"({n_index}) // cutlass.Int32({tcgen05_destination_bn})"
    static_tile_coord_m = tile_coord_m
    static_tile_coord_n = tile_coord_n
    # M-paired tiles: the epilogue body is wrapped in a two-iteration
    # ``_tcgen05_msub`` loop (one per 256-row subtile); each iteration drains
    # its own acc stage at the subtile's M tile coordinate.
    tcgen05_m_subtile_count = (
        matmul_plan.m_subtile_count if matmul_plan is not None else 1
    )
    if has_store_fanout and tcgen05_m_subtile_count > 1:
        raise exc.BackendUnsupported(
            "cute", "tcgen05 output fanout does not support paired M subtiles"
        )
    if (
        tcgen05_m_subtile_count > 1
        and matmul_plan is not None
        and (not matmul_plan.is_two_cta)
    ):
        # The cp.async-A flat kernel's 128-row subtiles (``cute_mma``'s
        # ``tcgen05_cp_async_a``) can run past M, so the subtile's rows go into
        # the M index itself: the full-tile predicate, the auxiliary tiles and
        # the edge coordinates all read the drained subtile's rows, e.g.
        # ``tcgen05_full_tile = (tile_offset_0 + cutlass.Int32(_tcgen05_msub)
        # * cutlass.Int32(128)) + cutlass.Int32(128) <= 30522``.
        m_index = (
            f"({m_index} + cutlass.Int32(_tcgen05_msub) * "
            f"cutlass.Int32({tcgen05_source_bm}))"
        )
        tile_coord_m = f"({m_index}) // cutlass.Int32({tcgen05_source_bm})"
        static_tile_coord_m = tile_coord_m
    elif tcgen05_m_subtile_count > 1:
        static_tile_coord_m = f"({tile_coord_m} + cutlass.Int32(_tcgen05_msub))"
    full_tile = df.new_var("tcgen05_full_tile")

    gmem_tile = df.new_var("tcgen05_gC")
    coord_tile = df.new_var("tcgen05_cC")
    tcgc_base = df.new_var("tcgen05_tCgC_base")
    tccc_base = df.new_var("tcgen05_tCcC_base")
    tcgc = df.new_var("tcgen05_tCgC")
    tcgc_planned = df.new_var("tcgen05_tCgC_planned")
    tccc = df.new_var("tcgen05_tCcC")
    tacc = df.new_var("tcgen05_tAcc")
    tacc_epi = df.new_var("tcgen05_tAcc_epi")
    epi_tile = df.new_var("tcgen05_store_epi_tile")
    tiled_copy_t2r = df.new_var("tcgen05_tiled_copy_t2r")
    thr_copy_t2r = df.new_var("tcgen05_thr_copy_t2r")
    ttr_tacc_base = df.new_var("tcgen05_tTR_tAcc_base")
    tcgc_epi = df.new_var("tcgen05_tCgC_epi")
    tccc_epi = df.new_var("tcgen05_tCcC_epi")
    ttr_gc = df.new_var("tcgen05_tTR_gC")
    ttr_cc = df.new_var("tcgen05_tTR_cC")
    ttr_racc = df.new_var("tcgen05_tTR_rAcc")
    ttr_rd = df.new_var("tcgen05_tTR_rD")
    ttr_tacc_stage = df.new_var("tcgen05_tTR_tAcc_stage")
    ttr_tacc = df.new_var("tcgen05_tTR_tAcc")
    ttr_gc_grouped = df.new_var("tcgen05_tTR_gC_grouped")
    ttr_cc_grouped = df.new_var("tcgen05_tTR_cC_grouped")
    compact_gmem_2d = df.new_var("tcgen05_compact_gD2d")
    compact_gmem_epi = df.new_var("tcgen05_compact_gD_epi")
    compact_ttr_gd = df.new_var("tcgen05_compact_tTR_gD")
    compact_ttr_gd_grouped = df.new_var("tcgen05_compact_tTR_gD_grouped")
    compact_gd_subtile = df.new_var("tcgen05_compact_tTR_gD_subtile")
    compact_coord_epi = df.new_var("tcgen05_compact_cD_epi")
    compact_ttr_coord = df.new_var("tcgen05_compact_tTR_cD")
    compact_ttr_coord_grouped = df.new_var("tcgen05_compact_tTR_cD_grouped")
    compact_coord_subtile = df.new_var("tcgen05_compact_tTR_cD_subtile")
    ttr_tacc_mn = df.new_var(
        "tcgen05_tTR_tAcc_nm" if tcgen05_nm_store else "tcgen05_tTR_tAcc_mn"
    )
    ttr_gc_subtile = df.new_var("tcgen05_tTR_gC_subtile")
    ttr_cc_subtile = df.new_var("tcgen05_tTR_cC_subtile")
    trs_cc = df.new_var("tcgen05_tRS_cC")
    trs_cc_grouped = df.new_var("tcgen05_tRS_cC_grouped")
    trs_cc_subtile = df.new_var("tcgen05_tRS_cC_subtile")
    acc_vec = df.new_var("tcgen05_acc_vec")
    kernel_desc = df.new_var("tcgen05_kernel_desc")
    mcld = df.new_var("tcgen05_mcld")
    num_bits = df.new_var("tcgen05_num_bits")
    simt_atom = df.new_var("tcgen05_simt_atom")
    smem_d_layout = df.new_var("tcgen05_sD_layout")
    smem_d_ptr = df.new_var("tcgen05_sD_ptr")
    smem_d = df.new_var("tcgen05_sD")
    tiled_copy_r2s = df.new_var("tcgen05_tiled_copy_r2s")
    copy_atom_r2s = df.new_var("tcgen05_selected_nm_stsm_atom")
    thr_copy_r2s = df.new_var("tcgen05_thr_copy_r2s")
    trs_rd = df.new_var("tcgen05_tRS_rD")
    trs_racc = df.new_var("tcgen05_tRS_rAcc")
    trs_sd = df.new_var("tcgen05_tRS_sD")
    bsg_sd = df.new_var("tcgen05_bSG_sD")
    bsg_gd_partitioned = df.new_var("tcgen05_bSG_gD_partitioned")
    bsg_gd = df.new_var("tcgen05_bSG_gD")
    grouped_d_tensormap_manager = df.new_var("tcgen05_grouped_d_tensormap_manager")
    grouped_d_tensormap_grid_dim = df.new_var("tcgen05_grouped_d_tensormap_grid_dim")
    grouped_d_tensormap_workspace_idx = df.new_var(
        "tcgen05_grouped_d_tensormap_workspace_idx"
    )
    grouped_d_tensormap_ptr = df.new_var("tcgen05_grouped_d_tensormap_ptr")
    grouped_d_tensormap_desc_ptr = df.new_var("tcgen05_grouped_d_tensormap_desc_ptr")
    grouped_d_tensormap_smem_ptr = df.new_var("tcgen05_grouped_d_tensormap_smem_ptr")
    grouped_d_tensormap_last_group = df.new_var(
        "tcgen05_grouped_d_tensormap_last_group"
    )
    grouped_d_tensormap_group_changed = df.new_var(
        "tcgen05_grouped_d_tensormap_group_changed"
    )
    grouped_d_tensormap_base = df.new_var("tcgen05_grouped_d_tensormap_base")
    grouped_d_tensormap_addr = df.new_var("tcgen05_grouped_d_tensormap_addr")
    grouped_d_tensormap_stride_m = df.new_var("tcgen05_grouped_d_tensormap_stride_m")
    grouped_d_tensormap_stride_n = df.new_var("tcgen05_grouped_d_tensormap_stride_n")
    grouped_d_tensormap_real_d = df.new_var(
        "tcgen05_grouped_d_tensormap_d_nm"
        if tcgen05_nm_store
        else "tcgen05_grouped_d_tensormap_real_d"
    )
    c_buffer = df.new_var("tcgen05_c_buffer")
    epilog_sync_barrier = df.new_var("tcgen05_epilog_sync_barrier")
    c_pipeline_producer_group = df.new_var("tcgen05_c_pipeline_producer_group")
    c_pipeline = df.new_var("tcgen05_c_pipeline")
    subtile_count = df.new_var("tcgen05_subtile_count")
    # Workstream A Stage 4 (cycle 93, Path B): the C-store producer->consumer
    # edge over the C-ring SMEM (``tRS_sD``, depth ``c_stage_count``). Producer
    # = the 4 epi warps (arrive after R2S + ``fence_view_async_shared``);
    # consumer = the single store warp (waits, issues the TMA-D, releases the
    # SMEM stage). Replaces the second ``epilog_sync_barrier`` (R2S-visible)
    # CTA-wide barrier with a cheaper cross-warp pipeline edge that lets the
    # epi warps proceed to the next subtile while the store warp drains.
    c_store_edge_barriers = df.new_var("tcgen05_c_store_edge_barriers")
    c_store_edge_producer_group = df.new_var("tcgen05_c_store_edge_producer_group")
    c_store_edge_consumer_group = df.new_var("tcgen05_c_store_edge_consumer_group")
    c_store_edge = df.new_var("tcgen05_c_store_edge")
    c_store_edge_producer_state = df.new_var("tcgen05_c_store_edge_producer_state")
    c_store_edge_consumer_state = df.new_var("tcgen05_c_store_edge_consumer_state")
    # Separate consumer state for the LAGGED release. The store warp's TMA-D is
    # an async bulk copy that reads the C-ring SMEM stage; the stage may not be
    # reused (epi R2S overwrite) until that read completes. ``c_pipeline``
    # (PipelineTmaStore) tracks store completion via ``cp_async_bulk_wait_group``
    # (read=True), which after committing store i and waiting drains every store
    # except the ``c_stages - 1`` most recent. So the store warp releases the
    # C-ring stage from ``c_stages - 1`` subtiles ago (provably drained), lagging
    # the consumer-wait by ``c_stages - 1``. This leaves exactly one free stage
    # (edge depth ``c_stages``), giving the ~1-subtile store/T2R overlap the
    # acc_stages=2 bound permits. The first ``c_stages - 1`` releases are
    # suppressed (no drained stage yet); the trailing stages release naturally
    # in subsequent tiles as the global subtile index advances.
    c_store_edge_release_state = df.new_var("tcgen05_c_store_edge_release_state")
    epi_warp_ids = ", ".join(
        f"cutlass.Int32({i})" for i in range(tcgen05_value.epi_warp_count)
    )
    if tcgen05_value.epi_warp_count == 1:
        epi_warp_ids += ","

    # Per-aux-step plumbing: per-thread auxiliary tensor reads at
    # the splice site. For each ``_AuxiliaryTensorLoadExpr`` in the
    # chain we register the auxiliary tensor as a kernel arg,
    # allocate fresh AST var names for the partitioning chain, and
    # later (inside each per-thread splice site) emit per-subtile
    # ``aux_loaded = ...`` lines that the chain renderer references.
    # Static-full TMA-store tiles use the historical direct
    # ``ttr_aux_subtile.load()`` form. SIMT-store edge tiles use a
    # predicated GMEM-to-register copy first, so the aux read observes
    # the same runtime predicate as the output store.
    aux_steps_in_chain: tuple[_AuxiliaryTensorLoadExpr, ...] = (
        epilogue_chain.auxiliary_tensor_loads if epilogue_chain is not None else ()
    )
    # A store of epilogue inputs alone (``out1[tile] = in2[tile].to(f32)``,
    # ``analyze_tcgen05_aux_only_store_chain``) reads no accumulator, so its
    # subtiles read only the auxiliary operand:
    #
    #   cute.autovec_copy(tcgen05_tTR_gAux_subtile, tcgen05_aux_rmem)
    #   tcgen05_aux_loaded = tcgen05_aux_rmem.load()
    #   tcgen05_acc_vec = tcgen05_aux_loaded.to(cutlass.Float32)
    #
    # instead of also issuing ``cute.copy(tcgen05_tiled_copy_t2r,
    # tcgen05_tTR_tAcc_mn, tcgen05_tTR_rAcc)`` and
    # ``tcgen05_acc_loaded = tcgen05_tRS_rAcc.load()`` into registers
    # nothing reads. It keeps the fanout protocol: as the first store it
    # waits for the accumulator, as the final one it releases it after the
    # other stores' TMEM loads.
    chain_reads_accumulator = epilogue_chain is None or epilogue_chain.reads_accumulator
    # Rank-0 device scalars of the chain (``acc * scale[()]``) bind once per
    # output tile in the per-tile setup, ahead of the accumulator wait, instead
    # of re-reading global memory inside every subtile iteration.
    runtime_scalar_prelude, runtime_scalar_locals = (
        epilogue_chain.render_runtime_scalar_bindings(df.new_var, "")
        if epilogue_chain is not None and epilogue_chain.steps
        else ("", {})
    )
    tcgen05_aux_load_placement = df.config.get(
        TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
        TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT,
    )
    # The strict normalizer (``pre_acc_wait_admissible``) places the reads
    # ahead of the accumulator wait only where every store chain loads an
    # auxiliary operand, or the shared fanout merges the chains with the
    # pair's reads proved safe.
    if (
        tcgen05_aux_load_placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
        and tcgen05_value.partial_output_tma_store
    ):
        raise exc.InvalidConfig(
            f"invalid {TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY}="
            f"{TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT!r}: per-subtile "
            "auxiliary loads cannot precede the accumulator wait for a "
            "partial-output TMA-store epilogue"
        )
    # Whole-fragment register hoist of exact-shape auxiliary tiles
    # (``tcgen05_aux_fragment_hoist``): the tile's auxiliary fragment is read
    # once before the accumulator wait (``_aux_tile_setup_lines``) and sliced
    # per subtile from registers. The strict normalizer
    # (``aux_fragment_hoist_admissible``) judges the knob's placement
    # (``pre_acc_wait``), the kernel-wide exact-shape operand and the
    # per-thread register budget from the config. What it cannot see stays
    # codegen's proof and fails closed here: the plan's TMA-store epilogue
    # (``use_tma_store_epilogue`` also needs the output's TMA legality; an
    # unproved destination takes the SIMT store body) and *this* store's
    # chain, which must read the exact-shape operand itself (a sibling store
    # of the kernel may read only a row vector).
    tcgen05_aux_fragment_hoist = (
        df.config.get(TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY, False) is True
    )
    exact_shape_aux_in_chain = any(
        step.broadcast_axis is None
        for step in (
            aux_steps_in_chain
            if fanout_plan is None
            else tuple(
                step
                for chain in fanout_plan.chains
                for step in chain.auxiliary_tensor_loads
            )
        )
    )
    if tcgen05_aux_fragment_hoist and not (
        tcgen05_value.use_tma_store_epilogue
        and not tcgen05_value.partial_output_tma_store
        and exact_shape_aux_in_chain
    ):
        raise exc.InvalidConfig(
            f"{TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY}=True requires "
            f"{TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY}="
            f"{TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT!r}, the static-full "
            "TMA-store epilogue and an exact-shape auxiliary load in the chain"
        )
    # Epilogue warp groups (``tcgen05_epi_warp_groups``, decided by the matmul
    # plan: ``epi_warp_groups`` groups of four epilogue warps) and the register
    # prefetch ring of the exact-shape auxiliary fragment
    # (``tcgen05_aux_prefetch_subtiles``: the fragment of subtile ``i + d`` is
    # read at the top of subtile ``i``, the first ``d`` ahead of the
    # accumulator wait). Both rewrite the subtile loop into a trace-time
    # (``cutlass.range_constexpr``) loop; with several groups the loop walks
    # every G-th subtile, waits for the accumulator ahead of the loop and
    # releases it after, and each group keeps its own named barrier, C stages
    # and bulk-store groups. Legal on the standard full-tile TMA-store body
    # only (the strict normalizer's ``aux_prefetch_refusal`` judges the ring's
    # placement, the superseding hoist and the depth; the plan's TMA-store
    # epilogue and this store's own exact-shape read are codegen's proof, as
    # for the hoist above; the store-mode flags below the rest).
    tcgen05_epi_groups = tcgen05_value.epi_warp_groups
    grouped_epilogue = tcgen05_epi_groups > 1
    tcgen05_aux_prefetch_depth = cast(
        "int", df.config.get(TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY, 0)
    )
    if tcgen05_aux_prefetch_depth > 0 and (
        not tcgen05_value.use_tma_store_epilogue
        or tcgen05_value.partial_output_tma_store
        or not exact_shape_aux_in_chain
    ):
        raise exc.InvalidConfig(
            f"{TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY}="
            f"{tcgen05_aux_prefetch_depth} requires "
            f"{TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY}="
            f"{TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT!r}, the static-full "
            "TMA-store epilogue, an exact-shape auxiliary load in the chain and "
            f"no {TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY}"
        )
    static_subtile_loop = grouped_epilogue or tcgen05_aux_prefetch_depth > 0
    if static_subtile_loop and not (
        tcgen05_value.use_role_local_epi
        and tcgen05_value.use_tma_store_epilogue
        and not tcgen05_value.tma_store_full_tiles_only
        and not tcgen05_value.partial_output_tma_store
        and tcgen05_m_subtile_count == 1
        and not compact_fragment_store
        and fragment_epilogue is None
        and row_union is None
        and tcgen05_value.orientation is Tcgen05Orientation.MN
        and not tcgen05_value.output_column_major
        and not tcgen05_value.segment_store_m_offset
        and not leading_passthrough_output
    ):
        raise exc.InvalidConfig(
            "tcgen05 epilogue warp groups and the auxiliary prefetch ring need "
            "the standard static-full role-local TMA-store epilogue"
        )
    # The warp that issues the TMA stores, the first C acquire and the final
    # drain of its group's bulk-store groups: warp 0 of the epilogue, or each
    # group's leader (``epi_leader``: warp 0 of every four).
    epi_leader_predicate = (
        tcgen05_value.epi_leader
        if grouped_epilogue
        else f"{tcgen05_value.warp_idx} == cutlass.Int32(0)"
    )
    # The trace-time subtile index of the static loop: the group-local index
    # with several groups (the subtile itself is then a dynamic value), the
    # subtile with one.
    static_subtile_index = "_tcgen05_local" if grouped_epilogue else "_tcgen05_subtile"

    _TCGEN05_TMEM_DATAPATH_M = 128
    if tcgen05_value.has_explicit_epilogue_tile:
        assert tcgen05_value.explicit_epi_tile_m is not None
        tcgen05_epi_tile_m = tcgen05_value.explicit_epi_tile_m
    elif tcgen05_is_two_cta_m128(
        is_two_cta=tcgen05_lifecycle.is_two_cta, bm=tcgen05_value.bm
    ):
        tcgen05_epi_tile_m = tcgen05_value.bm // 2
    else:
        tcgen05_epi_tile_m = tcgen05_value.bm
    # The static loop needs the subtile count at codegen time (its ring
    # buffers and group split are Python-level): per-CTA rows over the plan's
    # subtile. The subtile width is the explicit tile's or CuTe's with-source
    # rule at the plan's epilogue dtype width (the plan's
    # ``compute_epilogue_tile_shape``); the generated loop asserts the
    # rendered count matches.
    tcgen05_static_subtile_count = 0
    tcgen05_static_epi_tile_m = 0
    tcgen05_static_epi_tile_n = 0
    tcgen05_local_subtile_count = 0
    tcgen05_ragged_groups = False
    if static_subtile_loop:
        # The subtile the plan renders: the explicit tile, or CuTe's
        # with-source rule at the plan's epilogue dtype width
        # (``tcgen05_epi_tile_m`` above is the per-thread row extent, the
        # whole 256-row pair tile on the legacy two-CTA path; the rule's M is
        # the 128-lane datapath tile).
        if tcgen05_value.has_explicit_epilogue_tile:
            assert tcgen05_value.explicit_epi_tile_m is not None
            assert tcgen05_value.explicit_epi_tile_n is not None
            tcgen05_static_epi_tile_m = tcgen05_value.explicit_epi_tile_m
            tcgen05_static_epi_tile_n = tcgen05_value.explicit_epi_tile_n
        else:
            tcgen05_static_epi_tile_m, tcgen05_static_epi_tile_n = (
                tcgen05_default_epilogue_tile_size(
                    tcgen05_value.bm,
                    tcgen05_value.bn,
                    elem_width_d=epi_tile_bits,
                    elem_width_c=epi_tile_bits,
                )
            )
        tcgen05_static_rows = (
            tcgen05_value.bm // 2 if tcgen05_lifecycle.is_two_cta else tcgen05_value.bm
        )
        tcgen05_static_subtile_count = (
            tcgen05_static_rows // tcgen05_static_epi_tile_m
        ) * (tcgen05_value.bn // tcgen05_static_epi_tile_n)
        tcgen05_local_subtile_count = -(
            -tcgen05_static_subtile_count // tcgen05_epi_groups
        )
        tcgen05_ragged_groups = tcgen05_static_subtile_count % tcgen05_epi_groups != 0

    aux_matmul_plan = df.cute_state.tcgen05.matmul_plan

    def can_stage_rowvec_per_warp(
        rec: _AuxStepRecord, *, has_full_register_hoist: bool
    ) -> bool:
        """Whether this rowvec uses a warp-private full-tile FP32 SMEM stage.

        The site conditions here; the shape half (a 32-bit N-broadcast row
        whose width splits into whole 32-lane copies -- the warp-private
        stage is one ``make_tiled_copy_tv`` over 32 lanes x ``copy_elems``
        values, so a shorter row would let the upper lanes copy past it --
        on the B200 sweep's profitable widths) is
        ``rowvec_aux_takes_per_warp_f32_stage``, shared with the fan-out
        config gate.  16-bit rows take the promoted shared stage instead
        (``rowvec_stage_promotes_to_f32``).
        """
        from ..cute_epilogue import rowvec_aux_takes_per_warp_f32_stage

        return (
            tcgen05_aux_load_placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
            and tcgen05_value.use_tma_store_epilogue
            and not tcgen05_value.partial_output_tma_store
            and not tcgen05_value.output_column_major
            and tcgen05_epi_tile_m >= _TCGEN05_TMEM_DATAPATH_M
            and not has_full_register_hoist
            and rowvec_aux_takes_per_warp_f32_stage(
                aux_dtype_bits=rec.aux_dtype_bits,
                broadcast_axis=rec.broadcast_axis,
                epi_warp_count=tcgen05_value.epi_warp_count,
                bn=tcgen05_value.bn,
                aux_extent=rec.aux_extent,
            )
        )

    def rowvec_stage_promotes_to_f32(rec: _AuxStepRecord) -> bool:
        """16-bit row vector consumed only by the FP32 carrier op.

        The chain's root ``acc <op> aux`` reads the 16-bit operand as FP32
        (the result dtype is FP32, so ``_round_epilogue_expression`` left the
        template unrounded) and the leaf carries no cast wrapper but an
        explicit FP32 conversion, so staging the row already converted is
        bit-identical: fp16/bf16 -> fp32 is exact. Any other use (a rounded
        16-bit sub-expression, another wrapped leaf, a multi-step chain)
        takes no stage: the row is read from GMEM per subtile in its source
        dtype. The predicate
        (``aux_leaf_takes_promoted_f32_stage``) is shared with the matmul
        plan, which picks the (128, 32) subtile for these rows.
        """
        from ..cute_epilogue import aux_leaf_takes_promoted_f32_stage

        return epilogue_chain is not None and aux_leaf_takes_promoted_f32_stage(
            epilogue_chain,
            rec.expr,
            aux_dtype_bits=rec.aux_dtype_bits,
            epi_warp_count=tcgen05_value.epi_warp_count,
            bn=tcgen05_value.bn,
            aux_extent=rec.aux_extent,
        )

    def shared_rowvec_stage_copy_shape(
        rec: _AuxStepRecord,
    ) -> tuple[int, int] | None:
        """``(threads, elems)`` of the one cooperative copy filling the stage."""
        from ..cute_epilogue import tcgen05_rowvec_stage_copy_shape

        return tcgen05_rowvec_stage_copy_shape(
            epi_warp_count=tcgen05_value.epi_warp_count,
            bn=tcgen05_value.bn,
            aux_extent=rec.aux_extent,
        )

    def can_stage_rowvec_shared_f32(
        rec: _AuxStepRecord, copy_elems: int | None
    ) -> bool:
        """Whether this 16-bit rowvec uses the promoted CTA-shared FP32 stage."""

        return (
            tcgen05_aux_load_placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
            and tcgen05_value.use_tma_store_epilogue
            and not tcgen05_value.partial_output_tma_store
            and not tcgen05_value.output_column_major
            and tcgen05_epi_tile_m >= _TCGEN05_TMEM_DATAPATH_M
            and copy_elems is not None
            and rec.aux_rmem_full is None
            and rowvec_stage_promotes_to_f32(rec)
        )

    use_full_tile_bm256_broadcast_aux = (
        tcgen05_aux_load_placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
        and aux_matmul_plan is not None
        and aux_matmul_plan.cluster_n == 2
        and tcgen05_lifecycle.is_two_cta
        and tcgen05_value.bm == 256
        and tcgen05_value.bn == 128
        and tcgen05_value.use_tma_store_epilogue
        and not tcgen05_value.partial_output_tma_store
        and not tcgen05_value.output_column_major
    )

    aux_step_records: list[_AuxStepRecord] = []
    for aux_idx, aux_step in enumerate(aux_steps_in_chain):
        aux_tensor_node = aux_step.load_node.args[0]
        assert isinstance(aux_tensor_node, torch.fx.Node)
        aux_torch_tensor = aux_tensor_node.meta.get("val")
        assert isinstance(aux_torch_tensor, torch.Tensor)
        aux_tensor_name = df.tensor_arg(aux_torch_tensor).name
        aux_dtype = backend.dtype_str(aux_torch_tensor.dtype)
        aux_dtype_bits = aux_torch_tensor.dtype.itemsize * 8
        # Aux tensors must be passed through to the device function as
        # placeholder args so the wrapper plumbs them into the cute
        # kernel signature (the role-local persistent path otherwise
        # treats unreferenced tensors as captures, which doesn't work
        # for tensors only read inside a per-subtile loop body).
        df.placeholder_args.add(aux_tensor_name)
        # Broadcast axes 0/1 build a stride-0 2-D view of a rank-1 tensor.
        # An exact rank-3 input uses a view of its selected leading coordinate.
        # The colvec form (2) already carries an (M, N) stride-(1, 0) view.
        aux_view2d = (
            df.new_var(f"tcgen05_aux_view2d_{aux_idx}")
            if aux_step.broadcast_axis in (0, 1) or aux_torch_tensor.ndim == 3
            else None
        )
        aux_step_records.append(
            _AuxStepRecord(
                expr=aux_step,
                aux_tensor_name=aux_tensor_name,
                broadcast_axis=aux_step.broadcast_axis,
                has_leading_passthrough=aux_torch_tensor.ndim == 3,
                aux_tile=df.new_var(f"tcgen05_aux_tile_{aux_idx}"),
                aux_part_base=df.new_var(f"tcgen05_tCgAux_base_{aux_idx}"),
                aux_xfm=df.new_var(f"tcgen05_tCgAux_xfm_{aux_idx}"),
                aux_planned=df.new_var(f"tcgen05_tCgAux_planned_{aux_idx}"),
                aux_epi=df.new_var(f"tcgen05_tCgAux_epi_{aux_idx}"),
                aux_dtype=aux_dtype,
                aux_dtype_bits=aux_dtype_bits,
                aux_extent=(
                    aux_torch_tensor.shape[0]
                    if (
                        aux_step.broadcast_axis == 1
                        and isinstance(aux_torch_tensor.shape[0], int)
                    )
                    else None
                ),
                ttr_aux=df.new_var(f"tcgen05_tTR_gAux_{aux_idx}"),
                ttr_aux_grouped=df.new_var(f"tcgen05_tTR_gAux_grouped_{aux_idx}"),
                ttr_aux_subtile=df.new_var(f"tcgen05_tTR_gAux_subtile_{aux_idx}"),
                aux_rmem=df.new_var(f"tcgen05_aux_rmem_{aux_idx}"),
                aux_loaded=df.new_var(f"tcgen05_aux_loaded_{aux_idx}"),
                aux_view2d=aux_view2d,
                # Pre-wait whole-fragment register hoist of N-broadcast
                # (rowvec) aux on the bm=128 2-CTA full-tile TMA-store path.
                # The fragment there is small (2 subtiles x epi-tile N of 64
                # at bn=128 = a handful of fp32 registers per thread), so the
                # whole-fragment LDG fits without spills and hides its GMEM
                # latency under the MMA wait (standalone CUTLASS does the
                # same; ~2% on the 512x6144x2048 fp8 scaled_mm shape). It is
                # deliberately NOT applied to the bm=256 family: its larger
                # whole-tile fragment regressed via register spills (see the
                # fp8_gap_v2 history of the rowvec hoist removal at bn=128/
                # epi-32 -- 409k LDL/STL on the 4096^3 shape).
                aux_rmem_full=(
                    df.new_var(f"tcgen05_aux_rmem_full_{aux_idx}")
                    if (
                        aux_step.broadcast_axis in (0, 1)
                        and tcgen05_aux_load_placement
                        == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
                        and tcgen05_is_two_cta_m128(
                            is_two_cta=tcgen05_lifecycle.is_two_cta,
                            bm=tcgen05_value.bm,
                        )
                        and tcgen05_value.use_tma_store_epilogue
                    )
                    or _exact_shape_aux_fragment_hoisted(
                        # With several epilogue warp groups the per-group
                        # fragment is hoisted through the subtile ring below.
                        requested=tcgen05_aux_fragment_hoist and not grouped_epilogue,
                        broadcast_axis=aux_step.broadcast_axis,
                        aux_ndim=aux_torch_tensor.ndim,
                    )
                    else None
                ),
                colvec_scalar_full=(
                    df.new_var(f"tcgen05_colvec_scalar_full_{aux_idx}")
                    if (
                        use_full_tile_bm256_broadcast_aux
                        and aux_step.broadcast_axis == 2
                    )
                    else None
                ),
            )
        )

    # Pyrefly does not preserve the non-None ``tcgen05_value`` narrowing
    # inside the nested source-formatter closures, so keep local
    # string aliases for attributes the closures read.
    tcgen05_aux_bm = tcgen05_value.bm
    tcgen05_aux_bn = tcgen05_value.bn
    tcgen05_aux_thr_mma = tcgen05_value.thr_mma
    tcgen05_aux_epi_tidx = tcgen05_value.epi_tidx
    tcgen05_aux_epi_active = tcgen05_lifecycle.epi_active
    tcgen05_aux_epi_warp_count = tcgen05_value.epi_warp_count
    tcgen05_aux_epilogue_rest_mode = tcgen05_value.epilogue_rest_mode
    tcgen05_aux_use_tma_store_epilogue = tcgen05_value.use_tma_store_epilogue
    tcgen05_aux_tmem_load_atom = tcgen05_value.tmem_load_atom
    tcgen05_aux_tma_store_tensor = tcgen05_value.tma_store_tensor
    tcgen05_aux_tail_tma_store_tensor = tcgen05_value.tail_tma_store_tensor
    tcgen05_aux_segment_store_actual_m = tcgen05_value.segment_store_actual_m
    tcgen05_explicit_store_tile_expr: str | None = None
    if tcgen05_value.has_explicit_epilogue_tile:
        assert tcgen05_value.explicit_epi_tile_m is not None
        assert tcgen05_value.explicit_d_store_box_n is not None
        tcgen05_explicit_store_tile_expr = tcgen05_explicit_d_store_tile_expr(
            tcgen05_value.explicit_epi_tile_m,
            tcgen05_value.explicit_d_store_box_n,
        )
    # The bm=128 CtaGroup.TWO family stores through the per-CTA epilogue tile
    # (m of 64, ``use_2cta=True``, no-source) so the store-side
    # ``tcgen05_store_epi_tile``, the ``kernel_desc.cta_tile_shape_mnk``, and
    # the host TMA-store atom all match the device-side N-mode-permuted tile.
    # Resolved through the shared helper so this ``(store_tile_m, epi_tile_expr)``
    # pair stays identical to the layout-plan side in ``cute_mma.py``.
    tcgen05_store_tile_m, tcgen05_store_epi_tile_expr = tcgen05_resolve_epilogue_tile(
        bm=tcgen05_value.bm,
        bn=tcgen05_value.bn,
        is_two_cta=tcgen05_lifecycle.is_two_cta,
        elem_dtype=epi_tile_dtype,
        c_layout=tcgen05_d_store_layout,
        explicit_expr=tcgen05_explicit_store_tile_expr,
    )

    # Per-thread epilogue M extent. The tcgen05 TMEM->register (T2R) copy
    # distributes the epilogue tile's M dimension across the 128-lane TMEM
    # datapath: when ``epi_tile_m >= 128`` every lane owns >= 1 full output
    # row, so each thread's per-subtile fragment stays within a SINGLE M row;
    # when ``epi_tile_m < 128`` (e.g. block_m=64) a lane's fragment spans
    # multiple M rows. This decides whether the per-row colvec scalar read is
    # valid (see the ``broadcast_axis == 2`` branch below). Mirrors the
    # ``epi_tile_m`` computation in ``tcgen05_resolve_epilogue_tile`` /
    # ``cute_mma``: ``bm`` for the default tile, ``bm // 2`` for the 2-CTA
    # bm=128 family, or the user-supplied explicit tile M.
    #
    # Correctness of the colvec scalar fast-path rests on the invariant that
    # the T2R atom FILLS the 128-lane M datapath before packing multiple rows
    # per lane (a property of CUTLASS's epilogue_tmem_copy_and_partition, not
    # enforced here). Verified geometrically by partitioning an identity
    # coordinate tensor through the same T2R copy and reading the distinct M
    # coordinates a lane holds: per-CTA per-thread M-extent is 1 at
    # epi_tile_m=128 (block_m=128) and 1 at epi_tile_m=128 (block_m=256 2-CTA),
    # but 2 at epi_tile_m=64 (block_m=64) -- matching the >= 128 threshold
    # exactly. If a future CUTLASS changes the lane distribution only the
    # threshold needs revisiting: the materialize fallback below is always
    # correct, so the worst case is the fast-path mis-firing (caught by the
    # row-dependent colvec tests).
    tcgen05_colvec_fragment_single_m_row = (
        tcgen05_epi_tile_m >= _TCGEN05_TMEM_DATAPATH_M
    )

    # C-input warp productive-body gate (``cute_plan.md`` §7.5.3.2
    # cycle 2b producer + consumer flip). When the matmul plan has
    # ``has_c_input_warp`` AND a non-empty ``aux_tensor_descriptors``
    # tuple AND the aux pipeline plan was registered by
    # ``cute_mma._codegen_cute_mma``, the consumer-side per-thread
    # GMEM aux LDG flips to an SMEM read from the
    # ``c_pipeline_aux``-staged ring populated by the C-input warp's
    # cooperative copy. The producer body in
    # ``program_id._build_c_input_warp_role_local_while`` writes
    # ONE ``epi_tile`` subtile of the per-CTA aux region
    # (``(bm_per_cta, bn)`` under 2cta; ``(bm, bn)`` otherwise) per
    # stage per subtile iteration under ``producer_acquire`` /
    # ``producer_commit`` framing; the consumer issues one
    # ``consumer_wait`` / mode-specific ``consumer_release`` pair
    # per subtile and feeds the SMEM stage into Quack's
    # ``tiled_copy_s2r`` flow (``make_tiled_copy_D`` against
    # ``tiled_copy_t2r`` →  ``partition_S(sC_ring)`` → per-
    # subtile ``cute.copy(s2r, sC[..., stage], rmem)`` →
    # ``rmem.load()``). Gate-closed configs keep the historical
    # GMEM path byte-identical.
    aux_pipeline_plan_obj = df.cute_state.tcgen05.aux_pipeline_plan
    # Workstream A Stage 4 (cycle 93, Path B): when the plan carries a store
    # warp, the per-subtile R2S->TMA-D tail is split by warp role and the
    # second epilogue barrier is replaced by the C-store pipeline edge. The
    # store warp drains the TMA-D so the 4 epi warps proceed to the next
    # subtile's T2R. ``store_warps=0`` keeps the original fused tail unchanged
    # (the production path; byte-identical codegen).
    has_store_warp = aux_matmul_plan is not None and aux_matmul_plan.has_store_warp
    store_warp_predicate = (
        f"{tcgen05_value.warp_idx} == cutlass.Int32({aux_matmul_plan.store_warp_id})"
        if aux_matmul_plan is not None and has_store_warp
        else ""
    )
    # Match each store-side record to its descriptor by
    # ``load_node`` FX-node identity rather than positional
    # index. The descriptor walker dedups by ``store_value_node``
    # at MMA-codegen time, so a single-store kernel's
    # descriptors and records share the same ``load_node``
    # values in some permutation. The matmul plan's
    # ``aux_single_store_value`` gate (in ``cute_mma`` and the
    # ``program_id`` role-local-while admission) only allocates
    # the producer-side pipeline when every descriptor shares
    # one ``store_value_node``, so the multi-store fan-out
    # wedge (producer commits to rings the per-store consumer
    # never releases) cannot occur — the productive body
    # closes its gate at MMA-codegen time and the consumer
    # path here falls back to GMEM. Broadcast row-vector aux loads are
    # deliberately not staged by the C-input producer, so the per-record lookup
    # below allows a mixed chain: matched exact-shape records read from SMEM,
    # unmatched records keep the direct GMEM path.
    aux_step_load_nodes: tuple = (
        tuple(rec_step.load_node for rec_step in aux_steps_in_chain)
        if aux_step_records
        else ()
    )
    aux_ring_index_by_step: list[int | None] = []
    aux_descriptor_load_nodes: tuple = (
        tuple(d.load_node for d in aux_matmul_plan.c_input_aux_tensor_descriptors)
        if aux_matmul_plan is not None
        else ()
    )
    for step_load_node in aux_step_load_nodes:
        try:
            aux_ring_index_by_step.append(
                aux_descriptor_load_nodes.index(step_load_node)
            )
        except ValueError:
            aux_ring_index_by_step.append(None)
    aux_has_staged_steps = any(
        ring_idx is not None for ring_idx in aux_ring_index_by_step
    )
    # Workstream A Stage 5 (cycle 94, the merge): the aux SMEM ring producer is
    # the C-input warp normally (SIMT or TMA), or the store warp under the merge
    # — but the store warp is TMA-ONLY (there is no SIMT store-warp producer;
    # ``store_warps=1 + SIMT aux`` falls back to direct-GMEM aux). The epi-warp
    # consumer reads the staged ring whenever a producer is present. The
    # ``aux_pipeline_plan_obj is not None`` term already closes this gate for
    # ``store_warps=1 + SIMT`` (``cute_mma`` never allocates the plan there);
    # the explicit ``use_tma_load`` term on the store-warp branch makes the
    # TMA-only requirement local and defensive.
    aux_producer_warp_present = aux_matmul_plan is not None and (
        aux_matmul_plan.has_c_input_warp
        or (
            aux_matmul_plan.has_store_warp
            and aux_pipeline_plan_obj is not None
            and aux_pipeline_plan_obj.use_tma_load
        )
    )
    # ``tcgen05_aux_load_mode="epi_tma"``: the same per-subtile SMEM rings,
    # filled by the epilogue warps themselves (``_epi_tma_aux_tile_lines`` /
    # ``_epi_tma_aux_refill_source`` below) instead of a producer warp.
    aux_epilogue_issued = (
        aux_pipeline_plan_obj is not None and aux_pipeline_plan_obj.epilogue_issued
    )
    use_aux_smem_source = (
        aux_step_records
        and aux_matmul_plan is not None
        and (aux_producer_warp_present or aux_epilogue_issued)
        and bool(aux_matmul_plan.c_input_aux_tensor_descriptors)
        and aux_pipeline_plan_obj is not None
        and aux_has_staged_steps
        # Multi-store fan-out gate (same predicate as the
        # producer-side allocator + role-local-while
        # admission). Without this guard the producer fires
        # ``producer_commit`` on rings whose only matching
        # consumer-store is a different per-store-codegen
        # invocation — the per-store splice site here only
        # releases its own subset, leaving the unmatched rings
        # uncommitted and deadlocking the producer once a CTA
        # wraps the pipeline depth.
        and len(
            {d.store_value_node for d in aux_matmul_plan.c_input_aux_tensor_descriptors}
        )
        <= 1
    )
    if use_aux_smem_source:
        assert aux_pipeline_plan_obj is not None
        aux_pipeline_name = aux_pipeline_plan_obj.pipeline
        aux_consumer_state_name = aux_pipeline_plan_obj.consumer_state
        aux_pipeline_uses_tma_load = aux_pipeline_plan_obj.use_tma_load
        all_rings = aux_pipeline_plan_obj.rings
        aux_ring_smem_names: tuple[str | None, ...] = tuple(
            all_rings[ring_idx].smem if ring_idx is not None else None
            for ring_idx in aux_ring_index_by_step
        )
    else:
        aux_pipeline_name = ""
        aux_consumer_state_name = ""
        aux_pipeline_uses_tma_load = False
        aux_ring_smem_names = tuple(None for _ in aux_step_records)
    # Epilogue-issued aux TMA (``epi_tma``): per-ring names of the tile's TMA
    # partitions, allocated once so the per-tile setup and the per-subtile
    # refill agree, and the prefetch depth (the ring depth).
    epi_tma_rings: list[tuple[str, str, str, str, str, str]] = []
    epi_tma_barrier_var = ""
    if use_aux_smem_source and aux_epilogue_issued:
        assert aux_pipeline_plan_obj is not None
        assert aux_pipeline_plan_obj.use_tma_load
        assert tcgen05_value.use_tma_store_epilogue, (
            "epi_tma aux staging renders only on the TMA-store epilogue"
        )
        epi_tma_barrier_var = df.new_var("tcgen05_aux_tma_barrier")
        for ring_idx, ring in enumerate(aux_pipeline_plan_obj.rings):
            assert ring.tma_atom is not None and ring.tma_tensor is not None
            epi_tma_rings.append(
                (
                    ring.tma_atom,
                    ring.tma_tensor,
                    ring.smem,
                    df.new_var(f"tcgen05_aux_gmem_tile_{ring_idx}"),
                    df.new_var(f"tcgen05_aux_tma_smem_part_{ring_idx}"),
                    df.new_var(f"tcgen05_aux_tma_gmem_part_{ring_idx}"),
                )
            )

    def _epi_tma_aux_issue_lines(indent: str, subtile_expr: str) -> list[str]:
        """Warp 0 acquires one ring stage, issues every ring's bulk load of
        ``subtile_expr`` into it and advances the producer state. Runs under
        an outer ``if warp_idx == 0`` the caller emits."""
        assert aux_pipeline_plan_obj is not None
        producer_state = aux_pipeline_plan_obj.producer_state
        lines = [
            f"{indent}{aux_pipeline_name}.producer_acquire({producer_state})\n",
            (
                f"{indent}{epi_tma_barrier_var} = "
                f"{aux_pipeline_name}.producer_get_barrier({producer_state})\n"
            ),
        ]
        for tma_atom, _tma_tensor, _smem, _gtile, smem_part, gmem_part in epi_tma_rings:
            lines.append(
                f"{indent}cute.copy({tma_atom}, "
                f"{gmem_part}[None, {subtile_expr}], "
                f"{smem_part}[None, {producer_state}.index], "
                f"tma_bar_ptr={epi_tma_barrier_var})\n"
            )
        lines.append(emit_pipeline_advance(producer_state, indent=indent) + "\n")
        return lines

    def _epi_tma_aux_tile_lines() -> list[str]:
        """Per-output-tile setup of the epilogue-issued aux rings.

        Partitions each staged operand's per-CTA ``(bm_per_cta, bn)`` tile
        into ``epi_tile`` subtiles against its TMA atom and SMEM ring (the
        same flat subtile axis the consumer indexes), then warp 0 issues the
        first ``stage_count`` subtiles' loads. Emitted before the
        accumulator wait, so the loads land under the tile's MMA; the
        remaining subtiles are refilled one subtile ahead of their read by
        ``_epi_tma_aux_refill_source``. A CTA pair's epilogue owns the
        per-CTA half of the pair tile's rows (``peer_m`` of the cluster).
        """
        if not epi_tma_rings:
            return []
        assert aux_pipeline_plan_obj is not None
        assert aux_matmul_plan is not None
        cluster_m = aux_matmul_plan.cluster_m
        if tcgen05_lifecycle.is_two_cta:
            bm_per_cta = tcgen05_aux_bm // cluster_m
            peer_m = (
                "(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) "
                f"% cutlass.Int32({cluster_m}))"
            )
            aux_tile_m = f"({tile_coord_m}) * cutlass.Int32({cluster_m}) + {peer_m}"
        else:
            bm_per_cta = tcgen05_aux_bm
            aux_tile_m = tile_coord_m
        epi_tile_var = aux_pipeline_plan_obj.epi_tile_var
        lines: list[str] = []
        for tma_atom, tma_tensor, smem, gtile, smem_part, gmem_part in epi_tma_rings:
            gsub = f"{gtile}_subtiles"
            lines.extend(
                [
                    (
                        f"{gtile} = cute.local_tile({tma_tensor}, "
                        f"({bm_per_cta}, {tcgen05_aux_bn}), "
                        f"({aux_tile_m}, {tile_coord_n}))"
                    ),
                    f"{gsub} = cute.flat_divide({gtile}, {epi_tile_var})",
                    f"{gsub} = cute.group_modes({gsub}, 2, cute.rank({gsub}))",
                    (
                        f"{smem_part}, {gmem_part} = cute.nvgpu.cpasync.tma_partition("
                        f"{tma_atom}, 0, cute.make_layout(1), "
                        f"cute.group_modes({smem}, 0, cute.rank({smem}) - 1), "
                        f"cute.group_modes({gsub}, 0, cute.rank({gsub}) - 1))"
                    ),
                ]
            )
        stage_count = aux_pipeline_plan_obj.stage_count
        issue = "".join(
            _epi_tma_aux_issue_lines("        ", "cutlass.Int32(_tcgen05_aux_prefetch)")
        )
        lines.append(
            f"if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"    for _tcgen05_aux_prefetch in cutlass.range("
            f"min({stage_count}, {subtile_count}), unroll_full=True):\n"
            f"{issue}"
        )
        return lines

    def _epi_tma_aux_refill_source(indent: str) -> str:
        """Refill the stage the subtile just released with the subtile
        ``stage_count`` ahead (nothing past the tile's last subtile)."""
        if not epi_tma_rings:
            return ""
        assert aux_pipeline_plan_obj is not None
        stage_count = aux_pipeline_plan_obj.stage_count
        issue = "".join(
            _epi_tma_aux_issue_lines(
                indent + "        ",
                f"cutlass.Int32(_tcgen05_subtile + {stage_count})",
            )
        )
        return (
            f"{indent}if _tcgen05_subtile + {stage_count} < {subtile_count}:\n"
            f"{indent}    if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"{issue}"
        )

    # Row-vector aux (``bias[n]`` / rowwise ``scale_b[n]``) reads stay
    # per-subtile (the generic ``ttr_aux_subtile.load()`` path below, placed
    # after the c_pipeline acquire / acc ``consumer_wait`` / T2R prefix per the
    # cycle-69 placement).
    rowvec_aux_stage_records: list[_RowvecAuxStageRecord | None] = []
    for aux_idx, rec in enumerate(aux_step_records):
        copy_bits = 128
        copy_elems = _tcgen05_rowvec_aux_stage_copy_elems(
            rec.aux_dtype_bits,
            tcgen05_aux_bn,
            rec.aux_extent,
            copy_bits=copy_bits,
        )
        stage_partial_tma_rowvec = (
            tcgen05_value.partial_output_tma_store
            and tcgen05_value.use_tma_store_epilogue
            and rec.broadcast_axis == 1
            and copy_elems is not None
            # a ``cutlass.Boolean`` stage and copy atom are one bit wide
            # against the mask's byte storage: a mask reads it directly
            and not rec.is_boolean_mask
        )
        stage_rowvec_per_warp = can_stage_rowvec_per_warp(
            rec, has_full_register_hoist=rec.aux_rmem_full is not None
        )
        shared_copy_shape = shared_rowvec_stage_copy_shape(rec)
        shared_copy_threads, shared_copy_elems = (
            shared_copy_shape if shared_copy_shape is not None else (0, None)
        )
        stage_rowvec_shared_f32 = (
            not stage_partial_tma_rowvec
            and not stage_rowvec_per_warp
            and can_stage_rowvec_shared_f32(rec, shared_copy_elems)
        )
        if stage_rowvec_shared_f32:
            assert shared_copy_elems is not None
            copy_elems = shared_copy_elems
            copy_bits = rec.aux_dtype_bits * shared_copy_elems
        # ``cute_mma`` decides the walk (``partial_output_row_lookahead``);
        # the row must also be one copy pass of the epilogue warps, from a
        # base its ``cp.async`` vectors are aligned to: the copy atom needs
        # ``copy_bits`` alignment where the synchronous copy compiles at the
        # pointer's own (``bias = buf[1:1 + n]``: 2 bytes, so a 128-bit
        # ``CopyG2SOp`` failed IR verification).
        # local: ``..memory_ops`` imports this module
        from ..memory_ops import cute_tensor_base_is_aligned

        prefetch_row = (
            tcgen05_value.partial_output_row_lookahead
            and stage_partial_tma_rowvec
            and copy_elems is not None
            and _rowvec_copy_threads(tcgen05_aux_bn, copy_elems)
            <= tcgen05_value.epi_warp_count * 32
        )
        if prefetch_row:
            aux_row_tensor = rec.expr.load_node.args[0]
            assert isinstance(aux_row_tensor, torch.fx.Node)
            prefetch_row = cute_tensor_base_is_aligned(
                env, aux_row_tensor.meta["val"], copy_bits // 8
            )
        if stage_partial_tma_rowvec or stage_rowvec_per_warp or stage_rowvec_shared_f32:
            assert rec.aux_extent is not None
            assert copy_elems is not None
            stage_record = _RowvecAuxStageRecord(
                smem_layout=df.new_var(f"tcgen05_aux_rowvec_smem_layout_{aux_idx}"),
                smem_ptr=df.new_var(f"tcgen05_aux_rowvec_smem_ptr_{aux_idx}"),
                smem=df.new_var(f"tcgen05_aux_rowvec_smem_{aux_idx}"),
                tiled_copy=df.new_var(f"tcgen05_aux_rowvec_tiled_copy_{aux_idx}"),
                thr_copy=df.new_var(f"tcgen05_aux_rowvec_thr_copy_{aux_idx}"),
                gmem_tile=df.new_var(f"tcgen05_aux_rowvec_gmem_tile_{aux_idx}"),
                gmem_part=df.new_var(f"tcgen05_aux_rowvec_gmem_part_{aux_idx}"),
                smem_part=df.new_var(f"tcgen05_aux_rowvec_smem_part_{aux_idx}"),
                coord=df.new_var(f"tcgen05_aux_rowvec_coord_{aux_idx}"),
                limit=df.new_var(f"tcgen05_aux_rowvec_limit_{aux_idx}"),
                pred=df.new_var(f"tcgen05_aux_rowvec_pred_{aux_idx}"),
                copy_bits=copy_bits,
                copy_elems=copy_elems,
                aux_extent=rec.aux_extent,
                warp_private=stage_rowvec_per_warp,
                stage_dtype=(
                    "cutlass.Float32" if stage_rowvec_shared_f32 else rec.aux_dtype
                ),
                promote=stage_rowvec_shared_f32,
                copy_threads=shared_copy_threads if stage_rowvec_shared_f32 else 0,
                prefetch=prefetch_row,
            )
            rowvec_aux_stage_records.append(stage_record)
            if prefetch_row:
                _register_tcgen05_epi_row_prefetch(
                    df,
                    stage=stage_record,
                    aux_tensor_name=rec.aux_tensor_name,
                    aux_dtype=rec.aux_dtype,
                    bn=tcgen05_aux_bn,
                    n_size=str(n_size),
                    n_index=n_index,
                    tile_coord_n=tile_coord_n,
                    epi_tidx=tcgen05_aux_epi_tidx,
                )
        else:
            rowvec_aux_stage_records.append(None)
    if fanout_plan is not None and any(
        stage is not None for stage in rowvec_aux_stage_records
    ):
        # The shared two-output traversal concatenates both stores' per-tile
        # setups; a staged rowvec's SMEM stage, cooperative copy and barrier
        # are per store and are not composed.
        raise exc.BackendUnsupported(
            "cute", "shared fanout does not compose with SMEM-staged rowvec aux"
        )
    # Exact-shape auxiliary records read through the register ring: ``d``
    # subtile fragments ahead (``tcgen05_aux_prefetch_subtiles``), or -- with
    # several epilogue warp groups -- the group's whole fragment hoisted
    # before the accumulator wait (``tcgen05_aux_fragment_hoist``: the
    # one-group hoist keeps its single whole-fragment copy, ``aux_rmem_full``).
    # Each buffer is laid out as the accumulator carrier, like the per-subtile
    # read's register tile, so the copy re-lays a multi-row fragment out too.
    aux_ring_records: dict[int, tuple[str, ...]] = {}
    aux_ring_names: dict[int, str] = {}
    aux_ring_depth = 0
    if static_subtile_loop and (
        tcgen05_aux_prefetch_depth > 0
        or (grouped_epilogue and tcgen05_aux_fragment_hoist)
    ):
        aux_ring_hoists_group = grouped_epilogue and tcgen05_aux_fragment_hoist
        aux_ring_depth = (
            tcgen05_local_subtile_count
            if aux_ring_hoists_group
            else min(tcgen05_aux_prefetch_depth, tcgen05_local_subtile_count - 1)
        )
        aux_ring_size = (
            tcgen05_local_subtile_count if aux_ring_hoists_group else aux_ring_depth + 1
        )
        for aux_idx, rec in enumerate(aux_step_records):
            if (
                rec.broadcast_axis is not None
                or rec.has_leading_passthrough
                or rowvec_aux_stage_records[aux_idx] is not None
            ):
                continue
            aux_ring_records[aux_idx] = tuple(
                df.new_var(f"tcgen05_aux_ring_{aux_idx}_{j}")
                for j in range(aux_ring_size)
            )
            aux_ring_names[aux_idx] = df.new_var(f"tcgen05_aux_ring_{aux_idx}")

    def _aux_ring_subtile_expr(local_offset: int, *, in_loop: bool) -> str:
        """Subtile index of the group's ``local_offset``-th subtile.

        Ahead of the loop (``in_loop`` False) the offset is absolute; inside
        it, relative to the current subtile.
        """
        if grouped_epilogue:
            if in_loop:
                return f"_tcgen05_subtile + cutlass.Int32({tcgen05_epi_groups * local_offset})"
            return (
                f"cutlass.Int32({tcgen05_epi_groups * local_offset}) + "
                f"{tcgen05_value.epi_group}"
            )
        if in_loop:
            return f"cutlass.Int32(_tcgen05_subtile + {local_offset})"
        return f"cutlass.Int32({local_offset})"

    def _aux_ring_pre_loop_source() -> str:
        """Ring buffers and the first ``d`` fragment reads, ahead of the wait."""
        lines: list[str] = []
        for aux_idx, buffers in aux_ring_records.items():
            rec = aux_step_records[aux_idx]
            for buffer in buffers:
                lines.append(
                    f"{buffer} = cute.make_rmem_tensor(cute.make_layout({trs_racc}.shape), "
                    f"{rec.aux_dtype})\n"
                )
            lines.append(f"{aux_ring_names[aux_idx]} = ({', '.join(buffers)},)\n")
            for j in range(min(aux_ring_depth, len(buffers))):
                sub = _aux_ring_subtile_expr(j, in_loop=False)
                copy = (
                    f"cute.autovec_copy({rec.ttr_aux_grouped}[(None, None, None, {sub})], "
                    f"{buffers[j]})\n"
                )
                if (
                    tcgen05_ragged_groups
                    and tcgen05_epi_groups * j + tcgen05_epi_groups - 1
                    >= tcgen05_static_subtile_count
                ):
                    # A group whose share is one subtile short skips the read
                    # of the subtile it does not own.
                    lines.append(
                        f"if {sub} < cutlass.Int32({subtile_count}):\n    {copy}"
                    )
                else:
                    lines.append(copy)
        return "".join(lines)

    def _aux_ring_subtile_load_source(prelude_indent: str, aux_idx: int) -> str:
        """Prefetch subtile ``i + d`` into the ring and read subtile ``i`` from it."""
        rec = aux_step_records[aux_idx]
        buffers = aux_ring_records[aux_idx]
        ring = aux_ring_names[aux_idx]
        lines: list[str] = []
        if aux_ring_depth < tcgen05_local_subtile_count:
            sub = _aux_ring_subtile_expr(aux_ring_depth, in_loop=True)
            copy = (
                f"cute.autovec_copy({rec.ttr_aux_grouped}[(None, None, None, {sub})], "
                f"{ring}[({static_subtile_index} + {aux_ring_depth}) % {len(buffers)}])\n"
            )
            lines.append(
                f"{prelude_indent}if cutlass.const_expr({static_subtile_index} + "
                f"{aux_ring_depth} < {tcgen05_local_subtile_count}):\n"
            )
            if tcgen05_ragged_groups:
                lines.append(
                    f"{prelude_indent}    if {sub} < cutlass.Int32({subtile_count}):\n"
                    f"{prelude_indent}        {copy}"
                )
            else:
                lines.append(f"{prelude_indent}    {copy}")
        lines.append(
            f"{prelude_indent}{rec.aux_loaded} = "
            f"{ring}[{static_subtile_index} % {len(buffers)}].load()\n"
        )
        return "".join(lines)

    partial_tma_needs_full_tile_guard = tcgen05_value.partial_output_tma_store and any(
        # ``aux_ring_smem_names`` and ``rowvec_aux_stage_records`` are both
        # positionally aligned with ``aux_step_records``.
        name is None and rowvec_aux_stage_records[aux_idx] is None
        for aux_idx, name in enumerate(aux_ring_smem_names)
    )

    def _rowvec_aux_smem_setup_lines() -> list[str]:
        """Emit compact per-tile SMEM allocation for staged row-vector aux."""

        lines: list[str] = []
        for stage in rowvec_aux_stage_records:
            if stage is None:
                continue
            lines.extend(
                [
                    (
                        f"{stage.smem_layout} = cute.make_layout("
                        + (
                            f"({tcgen05_aux_epi_warp_count}, {tcgen05_aux_bn}), "
                            f"stride=({tcgen05_aux_bn}, 1))"
                            if stage.warp_private
                            else f"({TCGEN05_ROW_LOOKAHEAD_SLOTS}, {tcgen05_aux_bn}), "
                            f"stride=({tcgen05_aux_bn}, 1))"
                            if stage.prefetch
                            else f"({tcgen05_aux_bn},), stride=(1,))"
                        )
                    ),
                    (
                        f"{stage.smem_ptr} = cute.arch.alloc_smem("
                        f"{stage.stage_dtype}, cute.cosize({stage.smem_layout}), "
                        "alignment=128)"
                    ),
                    (
                        f"{stage.smem} = cute.make_tensor("
                        f"{stage.smem_ptr}, {stage.smem_layout})"
                    ),
                ]
            )
        return lines

    def _rowvec_aux_copy_lines() -> list[str]:
        """Emit the predicated GMEM-to-SMEM copy for staged row-vector aux;
        for rows the previous tile prefetched, the wait for their ``cp.async``
        group and the barrier publishing them."""

        lines: list[str] = []
        prefetched = False
        for aux_idx, rec in enumerate(aux_step_records):
            stage = rowvec_aux_stage_records[aux_idx]
            if stage is None:
                continue
            if stage.promote:
                # One cooperative copy by the epilogue warps: each thread loads
                # its ``copy_elems`` source elements, converts them once, and
                # stores FP32; the epilogue barrier publishes the row before
                # the accumulator wait.  The per-subtile reads below then load
                # FP32 straight into the carrier op.
                # Rows narrower than the epilogue warps' thread count are
                # copied by whole warps (one element per thread); the other
                # warps skip straight to the barrier.
                threads = stage.copy_threads or tcgen05_aux_epi_warp_count * 32
                copy_indent = "    "
                copy_guard = ""
                if threads < tcgen05_aux_epi_warp_count * 32:
                    copy_guard = (
                        f"    if {tcgen05_aux_epi_tidx} < cutlass.Int32({threads}):\n"
                    )
                    copy_indent = "        "
                r16 = f"{stage.smem_part}_src"
                r32 = f"{stage.smem_part}_f32"
                copy_body = (
                    f"{stage.tiled_copy} = cute.make_tiled_copy_tv("
                    f"cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), "
                    f"{rec.aux_dtype}, num_bits_per_copy={stage.copy_bits}), "
                    f"cute.make_layout({threads}), "
                    f"cute.make_layout({stage.copy_elems}))\n"
                    f"{stage.thr_copy} = {stage.tiled_copy}.get_slice("
                    f"{tcgen05_aux_epi_tidx})\n"
                    f"{stage.gmem_tile} = cute.local_tile("
                    f"{rec.aux_tensor_name}, ({tcgen05_aux_bn},), "
                    f"({tile_coord_n},))\n"
                    f"{stage.gmem_part} = {stage.thr_copy}.partition_S("
                    f"{stage.gmem_tile})\n"
                    f"{stage.smem_part} = cute.make_tiled_copy_tv("
                    f"cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), "
                    f"{stage.stage_dtype}, num_bits_per_copy={32 * stage.copy_elems}), "
                    f"cute.make_layout({threads}), "
                    f"cute.make_layout({stage.copy_elems})).get_slice("
                    f"{tcgen05_aux_epi_tidx}).partition_D({stage.smem})\n"
                    f"{r16} = cute.make_rmem_tensor("
                    f"{stage.gmem_part}.shape, {rec.aux_dtype})\n"
                    f"cute.autovec_copy({stage.gmem_part}, {r16})\n"
                    f"{r32} = cute.make_rmem_tensor("
                    f"{stage.gmem_part}.shape, {stage.stage_dtype})\n"
                    f"{r32}.store({r16}.load().to({stage.stage_dtype}))\n"
                    f"cute.autovec_copy({r32}, {stage.smem_part})"
                )
                lines.append(
                    f"if {tcgen05_aux_epi_active}:\n"
                    + copy_guard
                    + "".join(
                        f"{copy_indent}{line}\n" for line in copy_body.split("\n")
                    )
                    + f"    {epilog_sync_barrier}.arrive_and_wait()"
                )
                continue
            if stage.prefetch:
                # copied one tile ahead (``_register_tcgen05_epi_row_prefetch``)
                prefetched = True
                continue
            names = stage.copy_names
            if stage.warp_private:
                copy_lines = _rowvec_copy_lines(
                    names,
                    copy_atom="cute.nvgpu.CopyUniversalOp()",
                    dtype=rec.aux_dtype,
                    copy_bits=stage.copy_bits,
                    threads=32,
                    copy_elems=stage.copy_elems,
                    tidx=f"{tcgen05_aux_epi_tidx} % cutlass.Int32(32)",
                    aux_tensor_name=rec.aux_tensor_name,
                    bn=tcgen05_aux_bn,
                    tile_coord_n=tile_coord_n,
                    smem=(
                        f"{stage.smem}[{tcgen05_aux_epi_tidx} // cutlass.Int32(32), "
                        "None]"
                    ),
                    predicate=None,
                )
            else:
                copy_lines = [
                    *_rowvec_copy_lines(
                        names,
                        copy_atom="cute.nvgpu.CopyUniversalOp()",
                        dtype=rec.aux_dtype,
                        copy_bits=stage.copy_bits,
                        threads=tcgen05_aux_epi_warp_count * 32,
                        copy_elems=stage.copy_elems,
                        tidx=tcgen05_aux_epi_tidx,
                        aux_tensor_name=rec.aux_tensor_name,
                        bn=tcgen05_aux_bn,
                        tile_coord_n=tile_coord_n,
                        smem=stage.smem,
                        predicate=(str(n_size), n_index, stage.aux_extent),
                    ),
                    "cute.arch.fence_acq_rel_cta()",
                    f"{epilog_sync_barrier}.arrive_and_wait()",
                ]
            lines.append(
                f"if {tcgen05_aux_epi_active}:\n"
                + "\n".join(f"    {line}" for line in copy_lines)
            )
        if prefetched:
            # This tile's rows were copied while the previous tile ran (one
            # ``cp.async`` group per scheduler step, padding steps committing
            # empty ones, ``_register_tcgen05_epi_row_prefetch``)
            # and only the next tile's group was committed after them (the
            # lookahead walks one tile ahead, whatever the slot count), so the
            # wait completes all but that newest group; the epilogue barrier
            # then publishes every thread's part:
            #     if tcgen05_epi_active:
            #         cute.arch.cp_async_wait_group(1)
            #         tcgen05_epilog_sync_barrier.arrive_and_wait()
            # The named barrier orders the epilogue warps' shared-memory
            # accesses, so this drops the synchronous copy's
            # ``cute.arch.fence_acq_rel_cta()`` ahead of it, which held each
            # tile about 0.45 us of its 2.3 us (4096x30000x128 + bias: 0.95
            # of the ATen GEMM with the fence, 1.04 without).
            lines.append(
                f"if {tcgen05_aux_epi_active}:\n"
                "    cute.arch.cp_async_wait_group(1)\n"
                f"    {epilog_sync_barrier}.arrive_and_wait()"
            )
        return lines

    def _simt_edge_coord_subtile_source(indent: str) -> str:
        coord_m_size, coord_n_size = m_size, n_size
        coord_m, coord_n = tile_coord_m, tile_coord_n
        if row_union is not None and row_union.schedule is not None:
            coord_m_size, coord_n_size = n_size, m_size
            coord_m, coord_n = tile_coord_n, tile_coord_m
        if segment_store:
            return (
                f"{indent}{coord_tile} = cute.make_identity_tensor("
                f"({tcgen05_aux_bm}, {tcgen05_aux_bn}))\n"
                f"{indent}{tccc_base} = {tcgen05_aux_thr_mma}.partition_C("
                f"{coord_tile})\n"
                f"{indent}{tccc} = "
                "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
                f"{tccc_base})\n"
                f"{indent}{tccc_epi} = cute.flat_divide({tccc}, {epi_tile})\n"
                f"{indent}{ttr_cc} = {thr_copy_t2r}.partition_D({tccc_epi})\n"
                f"{indent}{ttr_cc_grouped} = cute.group_modes("
                f"{ttr_cc}, 3, cute.rank({ttr_cc}))\n"
                f"{indent}{ttr_cc_subtile} = {ttr_cc_grouped}["
                f"(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            )
        return (
            f"{indent}{coord_tile} = cute.local_tile("
            f"cute.make_identity_tensor(({coord_m_size}, {coord_n_size})), "
            f"({tcgen05_aux_bm}, {tcgen05_aux_bn}), "
            f"({coord_m}, {coord_n}))\n"
            f"{indent}{tccc_base} = {tcgen05_aux_thr_mma}.partition_C("
            f"{coord_tile})\n"
            f"{indent}{tccc} = "
            "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tccc_base})\n"
            f"{indent}{tccc_epi} = cute.flat_divide({tccc}, {epi_tile})\n"
            f"{indent}{ttr_cc} = {thr_copy_t2r}.partition_D({tccc_epi})\n"
            f"{indent}{ttr_cc_grouped} = cute.group_modes({ttr_cc}, 3, "
            f"cute.rank({ttr_cc}))\n"
            f"{indent}{ttr_cc_subtile} = {ttr_cc_grouped}[(None, None, None, "
            f"cutlass.Int32(_tcgen05_subtile))]\n"
        )

    def _tma_r2s_coord_subtile_source(indent: str) -> str:
        coord_m_size, coord_n_size = m_size, n_size
        coord_m, coord_n = tile_coord_m, tile_coord_n
        if row_union is not None and row_union.schedule is not None:
            coord_m_size, coord_n_size = n_size, m_size
            coord_m, coord_n = tile_coord_n, tile_coord_m
        if segment_store:
            return (
                f"{indent}{coord_tile} = cute.make_identity_tensor("
                f"({tcgen05_aux_bm}, {tcgen05_aux_bn}))\n"
                f"{indent}{tccc_base} = {tcgen05_aux_thr_mma}.partition_C("
                f"{coord_tile})\n"
                f"{indent}{tccc} = "
                "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
                f"{tccc_base})\n"
                f"{indent}{tccc_epi} = cute.flat_divide({tccc}, {epi_tile})\n"
                f"{indent}{ttr_cc} = {thr_copy_t2r}.partition_D({tccc_epi})\n"
                f"{indent}{trs_cc} = {tiled_copy_r2s}.retile({ttr_cc})\n"
                f"{indent}{trs_cc_grouped} = cute.group_modes("
                f"{trs_cc}, 3, cute.rank({trs_cc}))\n"
                f"{indent}{trs_cc_subtile} = {trs_cc_grouped}["
                f"(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            )
        return (
            f"{indent}{coord_tile} = cute.local_tile("
            f"cute.make_identity_tensor(({coord_m_size}, {coord_n_size})), "
            f"({tcgen05_aux_bm}, {tcgen05_aux_bn}), "
            f"({coord_m}, {coord_n}))\n"
            f"{indent}{tccc_base} = {tcgen05_aux_thr_mma}.partition_C("
            f"{coord_tile})\n"
            f"{indent}{tccc} = "
            "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tccc_base})\n"
            f"{indent}{tccc_epi} = cute.flat_divide({tccc}, {epi_tile})\n"
            f"{indent}{ttr_cc} = {thr_copy_t2r}.partition_D({tccc_epi})\n"
            f"{indent}{trs_cc} = {tiled_copy_r2s}.retile({ttr_cc})\n"
            f"{indent}{trs_cc_grouped} = cute.group_modes({trs_cc}, 3, "
            f"cute.rank({trs_cc}))\n"
            f"{indent}{trs_cc_subtile} = {trs_cc_grouped}[(None, None, None, "
            f"cutlass.Int32(_tcgen05_subtile))]\n"
        )

    def _coord_subtile_source(
        indent: str, coord_layout: str, *, include_setup: bool
    ) -> tuple[str, str]:
        if coord_layout == "ttr":
            return (
                _simt_edge_coord_subtile_source(indent) if include_setup else "",
                ttr_cc_subtile,
            )
        assert coord_layout == "trs"
        return (
            _tma_r2s_coord_subtile_source(indent) if include_setup else "",
            trs_cc_subtile,
        )

    worklist_nm_segment_valid_m_bound = (
        tcgen05_value.segment_store_valid_m_bound
        if tcgen05_value.orientation is Tcgen05Orientation.NM
        and segment_store
        and not grouped_bounded_output_shift
        else ""
    )

    def _worklist_nm_segment_valid_m_prelude_and_expr(
        prelude_indent: str,
        carrier_name: str,
        coord_subtile_name: str,
    ) -> tuple[str, str]:
        valid_row_mask = df.new_var("tcgen05_selected_valid_row_mask")
        valid_row_coord = df.new_var("tcgen05_selected_valid_row_coord")
        if grouped_tail_store:
            local_m = f"{valid_row_coord}[1]"
        else:
            local_m = f"({segment_store_local_m}) + {valid_row_coord}[1]"
        prelude = (
            f"{prelude_indent}{valid_row_mask} = cute.make_rmem_tensor("
            f"cute.make_layout({carrier_name}.shape), cutlass.Boolean)\n"
            f"{prelude_indent}for _selected_valid_i in range("
            f"cute.size({valid_row_mask}.shape)):\n"
            f"{prelude_indent}    {valid_row_coord} = "
            f"{coord_subtile_name}[_selected_valid_i]\n"
            f"{prelude_indent}    {valid_row_mask}[_selected_valid_i] = "
            f"{local_m} < cutlass.Int32({worklist_nm_segment_valid_m_bound})\n"
        )
        return prelude, f"{valid_row_mask}.load()"

    def _simt_edge_scalar_copy_source(
        indent: str, src: str, dst: str, *, include_coord_setup: bool = True
    ) -> str:
        # General SIMT edge copies keep the scalar loop unless the call site
        # retile below can build a predicate with one lane per logical element.
        return (
            (_simt_edge_coord_subtile_source(indent) if include_coord_setup else "")
            + f"{indent}for _edge_i in range(cute.size({src}.shape)):\n"
            f"{indent}    _coord = {ttr_cc_subtile}[_edge_i]\n"
            f"{indent}    if {_edge_predicate_expr('_coord')}:\n"
            f"{indent}        {dst}[_edge_i] = {src}[_edge_i]\n"
        )

    def _simt_edge_vector_copy_source(
        indent: str, src: str, dst: str, *, n_size_static: int
    ) -> str:
        # Fringe tiles of a ragged destination with a proven static layout:
        # store whole per-thread chunks (the ``num_bits`` vector the interior
        # tiles use) under one predicate per chunk, and only the chunks that
        # straddle the N edge element by element. A per-element loop
        # scattered every warp store over 32 rows (one sector each): bf16
        # 1000^3 spent 45 us in 15 fringe tiles.
        vector = f"({num_bits} // {target_dtype}.width)"
        edge_src = df.new_var("tcgen05_edge_src")
        edge_dst = df.new_var("tcgen05_edge_dst")
        edge_coord = df.new_var("tcgen05_edge_coord")
        edge_pred = df.new_var("tcgen05_edge_pred")
        tail_src = df.new_var("tcgen05_edge_tail_src")
        tail_dst = df.new_var("tcgen05_edge_tail_dst")
        tail_coord = df.new_var("tcgen05_edge_tail_coord")
        tail_pred = df.new_var("tcgen05_edge_tail_pred")
        tail_atom = df.new_var("tcgen05_edge_tail_atom")
        return (
            _simt_edge_coord_subtile_source(indent)
            + f"{indent}{edge_src} = cute.logical_divide({src}, cute.make_layout({vector}))\n"
            f"{indent}{edge_dst} = cute.logical_divide({dst}, cute.make_layout({vector}))\n"
            f"{indent}{edge_coord} = cute.logical_divide({ttr_cc_subtile}, cute.make_layout({vector}))\n"
            # ``cute.copy`` wants one predicate per element even under a
            # vector atom: every element of a chunk takes the chunk's.
            f"{indent}{edge_pred} = cute.make_rmem_tensor({edge_src}.shape, cutlass.Boolean)\n"
            f"{indent}for _edge_i in range(cute.size({edge_src}.shape[1])):\n"
            f"{indent}    _coord = {edge_coord}[0, _edge_i]\n"
            f"{indent}    _chunk_pred = {_edge_vector_predicate_expr('_coord', vector)}\n"
            f"{indent}    for _edge_j in range({vector}):\n"
            f"{indent}        {edge_pred}[_edge_j, _edge_i] = _chunk_pred\n"
            f"{indent}cute.copy({simt_atom}, {edge_src}, {edge_dst}, pred={edge_pred})\n"
            f"{indent}if cutlass.const_expr({n_size_static} % {vector} != 0):\n"
            f"{indent}    {tail_atom} = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), {target_dtype}, "
            f"num_bits_per_copy={target_dtype}.width, "
            f"l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)\n"
            f"{indent}    {tail_src} = cute.logical_divide({src}, cute.make_layout(1))\n"
            f"{indent}    {tail_dst} = cute.logical_divide({dst}, cute.make_layout(1))\n"
            f"{indent}    {tail_coord} = cute.logical_divide({ttr_cc_subtile}, cute.make_layout(1))\n"
            f"{indent}    {tail_pred} = cute.make_rmem_tensor((1, {tail_src}.shape[1]), cutlass.Boolean)\n"
            f"{indent}    for _edge_i in range(cute.size({tail_src}.shape[1])):\n"
            f"{indent}        _coord = {tail_coord}[0, _edge_i]\n"
            f"{indent}        {tail_pred}[0, _edge_i] = {_edge_tail_predicate_expr('_coord', vector)}\n"
            f"{indent}    cute.copy({tail_atom}, {tail_src}, {tail_dst}, pred={tail_pred})\n"
        )

    def _simt_edge_logical_divide_copy_source(
        indent: str,
        src: str,
        dst: str,
        *,
        include_coord_setup: bool = True,
        var_prefix: str = "tcgen05_edge",
        copy_atom: str | None = None,
    ) -> str:
        # The make_layout(1) retile gives cute.copy a per-element predicate.
        # Its atom must transfer one element too: a wider atom can cross the
        # mask boundary or require alignment that this retile does not prove.
        # The same shape drives D stores and exact-aux G2R register loads.
        copy_atom = copy_atom or simt_atom
        edge_src = df.new_var(f"{var_prefix}_src")
        edge_dst = df.new_var(f"{var_prefix}_dst")
        edge_coord = df.new_var(f"{var_prefix}_coord")
        edge_pred = df.new_var(f"{var_prefix}_pred")
        return (
            (_simt_edge_coord_subtile_source(indent) if include_coord_setup else "")
            + f"{indent}{edge_src} = cute.logical_divide({src}, cute.make_layout(1))\n"
            f"{indent}{edge_dst} = cute.logical_divide({dst}, cute.make_layout(1))\n"
            f"{indent}{edge_coord} = cute.logical_divide({ttr_cc_subtile}, cute.make_layout(1))\n"
            f"{indent}{edge_pred} = cute.make_rmem_tensor((1, {edge_src}.shape[1]), cutlass.Boolean)\n"
            f"{indent}for _edge_i in range(cute.size({edge_src}.shape[1])):\n"
            f"{indent}    _coord = {edge_coord}[0, _edge_i]\n"
            f"{indent}    {edge_pred}[0, _edge_i] = {_edge_predicate_expr('_coord')}\n"
            f"{indent}cute.copy({copy_atom}, {edge_src}, {edge_dst}, pred={edge_pred})\n"
        )

    def _aux_tile_setup_lines(
        *,
        thr_copy_t2r_var: str,
        define_thr_copy_t2r: bool,
        force_gmem_aux: bool = False,
        retile_for_r2s: bool = False,
    ) -> list[str]:
        """Emit the per-output-tile aux partitioning lines.

        Each line goes once per output tile, before the per-subtile
        loop. Mirrors the existing ``tcgc -> tcgc_planned -> tcgc_epi
        -> ttr_gc -> ttr_gc_grouped`` pipeline used for the result D
        tensor, but partitions a separate auxiliary GMEM tensor per
        chain step. Calls ``thr_mma.partition_C`` and
        ``thr_copy_t2r.partition_D`` against the aux tile so the
        per-thread layout matches D's layout exactly — both the
        exact-shape (``residual[tile_m, tile_n]``) and rank-1
        broadcast (``bias[tile_n]`` / ``bias[tile_m]``) forms feed
        the same downstream pipeline.

        For the broadcast form the helper first builds a 2-D view
        of the underlying rank-1 tensor with stride 0 on the
        orthogonal axis (see :class:`_AuxiliaryTensorLoadExpr` for the
        canonical contract).

        When ``define_thr_copy_t2r`` is True the helper emits the
        ``thr_copy_t2r = tiled_copy_t2r.get_slice(...)`` line first
        (the TMA-store path does not otherwise create
        ``thr_copy_t2r``); the SIMT path passes False because it
        already creates the slice as part of its existing partition
        pipeline. ``retile_for_r2s`` mirrors Quack's SM100 epilogue
        visitor layout: TMA-store chains read aux operands in the
        R2S-retiled layout so the chain carrier can be ``tRS_rAcc`` /
        ``tRS_rD`` instead of the raw T2R fragment layout.
        ``force_gmem_aux`` is used by the hybrid edge-only
        SIMT path: C-input staging is only safe for full tiles because
        the producer-side bulk copy is not predicated for M/N fringes, and
        only the full-tile TMA body fills a ``pre_acc_wait`` row stage.
        """
        lines: list[str] = [
            line for line in runtime_scalar_prelude.splitlines() if line.strip()
        ]
        if not aux_step_records:
            return lines
        if define_thr_copy_t2r:
            lines.append(
                f"{thr_copy_t2r_var} = "
                f"{tiled_copy_t2r}.get_slice({tcgen05_aux_epi_tidx})"
            )
        for aux_idx, rec in enumerate(aux_step_records):
            staged_ring_name = aux_ring_smem_names[aux_idx]
            rowvec_stage = rowvec_aux_stage_records[aux_idx]
            if (
                use_aux_smem_source
                and staged_ring_name is not None
                and not force_gmem_aux
            ):
                # C-input warp productive-body gate is open for this exact-shape
                # descriptor: build the Quack-style SMEM->register path. Rowvec
                # broadcast records are not staged and fall through to the GMEM
                # partition setup below.
                assert aux_matmul_plan is not None
                ring_idx = aux_ring_index_by_step[aux_idx]
                assert ring_idx is not None
                aux_dtype_str = backend.dtype_str(
                    aux_matmul_plan.c_input_aux_tensor_descriptors[
                        ring_idx
                    ].host_tensor_val.dtype
                )
                tiled_copy_s2r_var = f"{rec.aux_tile}_tiled_copy_s2r"
                thr_copy_s2r_var = f"{rec.aux_tile}_thr_copy_s2r"
                tsr_sc_var = f"{rec.aux_tile}_tSR_sC"
                trs_rc_var = f"{rec.aux_tile}_tRS_rC"
                tsr_rc_var = f"{rec.aux_tile}_tSR_rC"
                rmem_shape_expr = (
                    f"{trs_rd}.layout" if retile_for_r2s else f"{ttr_racc}.shape"
                )
                lines.extend(
                    [
                        (
                            f"{tiled_copy_s2r_var} = "
                            f"cute.make_tiled_copy_D("
                            f"cute.make_copy_atom("
                            f"cute.nvgpu.CopyUniversalOp(), "
                            f"{aux_dtype_str}), "
                            f"{tiled_copy_t2r})"
                        ),
                        (
                            f"{thr_copy_s2r_var} = "
                            f"{tiled_copy_s2r_var}.get_slice("
                            f"{tcgen05_aux_epi_tidx})"
                        ),
                        (
                            f"{tsr_sc_var} = "
                            f"{thr_copy_s2r_var}.partition_S("
                            f"{staged_ring_name})"
                        ),
                        (
                            f"{trs_rc_var} = cute.make_rmem_tensor("
                            f"{rmem_shape_expr}, {aux_dtype_str})"
                        ),
                        (f"{tsr_rc_var} = {tiled_copy_s2r_var}.retile({trs_rc_var})"),
                    ]
                )
                continue

            if rec.broadcast_axis is None or rec.broadcast_axis == 2:
                # Exact-shape aux (or the colvec form) uses the trailing matrix
                # directly. For a rank-3 residual, first select the current
                # leading-passthrough slice without changing its M/N strides.
                if rec.has_leading_passthrough:
                    assert leading_index is not None
                    assert rec.aux_view2d is not None
                    lines.append(
                        _cute_leading_passthrough_view_2d(
                            rec.aux_view2d,
                            rec.aux_tensor_name,
                            leading_index,
                        )
                    )
                    source_for_local_tile = rec.aux_view2d
                else:
                    source_for_local_tile = rec.aux_tensor_name
                aux_tile_is_local = False
            # Only the full-tile TMA body fills a row stage, so the SIMT edge
            # body of a hybrid TMA-store epilogue (``force_gmem_aux``) reads
            # the row from global memory (the broadcast view below).
            elif rowvec_stage is not None and not force_gmem_aux:
                assert rec.broadcast_axis == 1
                assert rec.aux_view2d is not None
                # The compact SMEM rowvec holds one output tile's row (a
                # prefetched row: this tile's slot, filled one tile ahead), so
                # its 2-D broadcast view is already tile-sized. The
                # full-tile bm256 path gives every epilogue warp a private copy,
                # avoiding a CTA barrier between the copy and its first read.
                rowvec_smem_iterator = (
                    f"{rowvec_stage.smem}[{tcgen05_aux_epi_tidx} // "
                    "cutlass.Int32(32), None].iterator"
                    if rowvec_stage.warp_private
                    else f"{rowvec_stage.smem}[{tcgen05_value.role_local_tile_counter} "
                    f"% cutlass.Int32({TCGEN05_ROW_LOOKAHEAD_SLOTS}), None].iterator"
                    if rowvec_stage.prefetch
                    else f"{rowvec_stage.smem}.iterator"
                )
                lines.append(
                    f"{rec.aux_view2d} = cute.make_tensor("
                    f"{rowvec_smem_iterator}, "
                    f"cute.make_layout(({tcgen05_bm}, {tcgen05_bn}), "
                    f"stride=(0, 1)))"
                )
                source_for_local_tile = rec.aux_view2d
                aux_tile_is_local = True
            else:
                # M-axis (row) broadcast aux: build a 2-D logical view
                # over the underlying tensor's ``.iterator`` with
                # stride 0 on the leading (M) axis and stride 1 on the
                # trailing (N) axis. Stride 0 on M causes every lane
                # "owning" output ``(m, n)`` to read the same source
                # element regardless of m, which is the broadcast
                # semantic shared by two accepted forms:
                #   * ``broadcast_axis == 1`` — a bare rank-1 tensor
                #     ``bias[tile_n]`` with shape ``(N,)`` (rank-1 RHS
                #     aligns to the trailing axis under PyTorch
                #     broadcasting).
                #   * ``broadcast_axis == 0`` — an explicit ``(1, N)``
                #     tensor ``bias[tile_m, tile_n]`` (row 0 broadcasts
                #     over M).
                # Both have the same contiguous N-major memory layout
                # (element ``(0, n)`` at offset ``n``), so the
                # stride-(0, 1) view over ``.iterator`` is identical
                # and feeds the same ``partition_C → flat_divide →
                # partition_D`` pipeline used by exact-shape aux.
                # Mirrors Quack's ``RowVecLoad`` epilogue
                # (``quack/quack/epi_ops.py``). The classifier
                # (``aux_tensor_load_kind``) admits only these two
                # broadcast shapes; everything else drops to the
                # loud-failure backstop.
                assert rec.broadcast_axis in (0, 1)
                assert rec.aux_view2d is not None
                lines.append(
                    f"{rec.aux_view2d} = cute.make_tensor("
                    f"{rec.aux_tensor_name}.iterator, "
                    f"cute.make_layout(({m_size}, {n_size}), "
                    f"stride=(0, 1)))"
                )
                source_for_local_tile = rec.aux_view2d
                aux_tile_is_local = False
            if aux_tile_is_local:
                lines.append(f"{rec.aux_tile} = {source_for_local_tile}")
            else:
                lines.append(
                    f"{rec.aux_tile} = cute.local_tile("
                    f"{source_for_local_tile}, ({tcgen05_bm}, {tcgen05_bn}), "
                    f"({tile_coord_m}, {tile_coord_n}))"
                )
            lines.extend(
                [
                    (
                        f"{rec.aux_part_base} = "
                        f"{tcgen05_thr_mma}.partition_C({rec.aux_tile})"
                    ),
                    (
                        f"{rec.aux_xfm} = "
                        "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
                        f"{rec.aux_part_base})"
                    ),
                    (
                        f"{rec.aux_planned} = cute.make_tensor("
                        f"{rec.aux_xfm}.iterator, "
                        f"cute.append(cute.append(cute.append({rec.aux_xfm}.layout, "
                        f"{tcgen05_aux_epilogue_rest_mode}), "
                        f"{tcgen05_aux_epilogue_rest_mode}), "
                        f"{tcgen05_aux_epilogue_rest_mode}))"
                    ),
                    (
                        f"{rec.aux_epi} = cute.flat_divide("
                        f"{rec.aux_planned}, {epi_tile})"
                    ),
                    (f"{rec.ttr_aux} = {thr_copy_t2r_var}.partition_D({rec.aux_epi})"),
                    *(
                        [f"{rec.ttr_aux} = {tiled_copy_r2s}.retile({rec.ttr_aux})"]
                        if retile_for_r2s
                        else []
                    ),
                    (
                        f"{rec.ttr_aux_grouped} = cute.group_modes("
                        f"{rec.ttr_aux}, 3, cute.rank({rec.ttr_aux}))"
                    ),
                    # Pre-wait hoist: one cooperative LDG of the whole rowvec
                    # fragment, issued here (before the accumulator
                    # consumer_wait downstream) so the GMEM latency overlaps
                    # the MMA wait. Per-subtile reads then come from
                    # registers. See the ``aux_rmem_full`` field docs for the
                    # family gate.
                    *(
                        [
                            (
                                f"{rec.aux_rmem_full} = cute.make_rmem_tensor("
                                f"{rec.ttr_aux_grouped}.shape, {rec.aux_dtype})"
                            ),
                            (
                                f"cute.autovec_copy({rec.ttr_aux_grouped}, "
                                f"{rec.aux_rmem_full})"
                            ),
                        ]
                        if rec.aux_rmem_full is not None and not force_gmem_aux
                        else []
                    ),
                    *(
                        [
                            (
                                f"{rec.colvec_scalar_full} = "
                                f"{rec.ttr_aux_grouped}[(0, 0, 0, 0)]"
                            )
                        ]
                        if rec.colvec_scalar_full is not None and not force_gmem_aux
                        else []
                    ),
                ]
            )
        return lines

    def _materialize_broadcast_aux_source(
        indent: str, rec: object, carrier_name: str
    ) -> str:
        """Emit a per-subtile aux load that matches the accumulator carrier's
        register profile.

        Example output (``indent='    '``, ``carrier_name='tcgen05_tRS_rAcc'``)::

            tcgen05_aux_rmem_0 = cute.make_rmem_tensor(
                cute.make_layout(tcgen05_tRS_rAcc.shape), cutlass.Float32
            )
            cute.autovec_copy(tcgen05_tTR_gAux_subtile_0, tcgen05_aux_rmem_0)
            tcgen05_aux_loaded_0 = tcgen05_aux_rmem_0.load()
        """
        return (
            f"{indent}{rec.aux_rmem} = "  # type: ignore[attr-defined]
            f"cute.make_rmem_tensor(cute.make_layout({carrier_name}.shape), "
            f"{rec.edge_rmem_dtype})\n"  # type: ignore[attr-defined]
            f"{indent}cute.autovec_copy("
            f"{rec.ttr_aux_subtile}, {rec.aux_rmem})\n"  # type: ignore[attr-defined]
            f"{indent}{rec.aux_loaded} = "  # type: ignore[attr-defined]
            f"{rec.edge_rmem_value}\n"  # type: ignore[attr-defined]
        )

    def _aux_subtile_load_source(
        prelude_indent: str,
        carrier_name: str,
        *,
        force_simt_edge_aux: bool = False,
        safe_direct_aux_with_full_tile: bool = False,
        unconditional_full_tile: bool = False,
    ) -> str:
        """Per-subtile aux GMEM-load source lines (one per aux step).

        ``unconditional_full_tile``: every tile is statically full (the
        register-direct store on a divisible output), so the direct GMEM
        form loads without the runtime full-tile guard or its edge loop.

        Each step emits the per-thread GMEM subtile slice of
        ``tTR_gAux_grouped_<idx>`` followed by a ``.load()`` call
        into the per-subtile ``tcgen05_aux_loaded_*`` local. Goes
        inside the per-subtile loop body. The slice depends on
        ``_tcgen05_subtile`` so it cannot be hoisted out of the
        loop entirely. Splice sites choose where to place this
        block: the default TMA-store path keeps it after the
        c_pipeline acquire, acc ``consumer_wait``, and t2r
        async TMEM→reg copy so residual and bias fragments are
        not live through the store-prefix waits. SIMT fallback
        concatenates it with the chain prelude because it does not
        use the TMA aux-pipeline shape; diagnostic helper paths keep
        the same flat prelude order for unary chains and reject aux
        chains at validation time.

        Cycle 39 (GPU 6) replan note: an alternative form that
        pre-loads all subtile aux into a per-thread register
        tensor outside the per-subtile loop (``cute.autovec_copy``
        from ``tTR_gAux_grouped_<idx>`` into a fresh
        ``tTR_rAux_<idx>``) was tested. The single cooperative
        LDG fired before the per-subtile loop, but the multi-
        subtile register tensor pushed local-memory spills from
        356k to 1.17M and grew kernel duration from 308 µs to
        332 µs. The per-subtile GMEM load form below pays one
        LDG per chain-add but the compiler IR / SASS scheduler
        already lifts the LDG ahead of the chain-add given the
        independent dependency graph.

        Cycle 69 found a related spill tradeoff inside the default
        TMA-store body: placing the per-subtile aux LDG after the
        acquire/T2R prefix removes most local-memory spill traffic,
        so that path no longer uses the older top-of-loop hoist.
        """
        if not aux_step_records:
            return ""
        lines: list[str] = []
        force_simt_edge_coord_emitted = False
        if use_aux_smem_source and not force_simt_edge_aux:
            # C-input warp productive-body gate is open: per-subtile
            # SMEM ring staging. Each subtile iteration waits on
            # ``c_pipeline_aux`` for the producer warp to fill the
            # active stage, then issues one filtered
            # ``cute.copy(tiled_copy_s2r, tSR_sC[..., stage], tSR_rC)``
            # per descriptor to load the active stage into the
            # per-thread register tensor (Quack's
            # ``epilog_smem_load_and_partition`` flow from
            # ``quack/gemm_sm100.py``: ``tiled_copy_s2r`` is built via
            # ``make_tiled_copy_D`` against ``tiled_copy_t2r``;
            # ``tSR_sC = thr_copy_s2r.partition_S(sC_ring)`` selects
            # the SMEM source; ``tSR_rC`` is a re-layout view of the
            # same register memory as ``tRS_rC``). The chain reads
            # ``tRS_rC.load()`` (== ``aux_loaded``). Drain the shared
            # reads before release; assigning their register values does
            # not force outstanding LDS completion. SIMT pipelines count
            # every reader's arrival. TMA pipelines retain their existing
            # elected-warp signaling protocol.
            # The release plus state advance run in the
            # same per-subtile iteration so the producer can refill
            # the same stage on the very next persistent tile
            # (matches the mode-specific consumer arrive count set by
            # ``_emit_tcgen05_aux_pipeline_setup``).
            #
            # Note: ``partition_D(smem_stage).load()`` on
            # ``thr_copy_t2r`` (an earlier prior-subagent variant)
            # produced a deadlocking SMEM read — TMEM→reg-shaped
            # partition_D applied to a SMEM tensor does not
            # compose with the producer's
            # ``make_tiled_copy_tv`` cooperative copy in a way the
            # mbarrier handshake recognizes. The Quack-style
            # ``tiled_copy_s2r`` flow is the canonical CUTLASS-DSL
            # pattern.
            lines.append(
                f"{prelude_indent}{aux_pipeline_name}.consumer_wait("
                f"{aux_consumer_state_name})\n"
            )
            if aux_pipeline_uses_tma_load:
                # TMA producer writes arrive through the async proxy; after the
                # pipeline wait, fence that view before generic SMEM reads.
                # The warp sync mirrors CUTLASS/Quack's TMA-load consumer
                # sequence so every lane observes the fenced view before the
                # per-lane SMEM->register copy below.
                lines.extend(
                    [
                        f"{prelude_indent}cute.arch.fence_view_async_shared()\n",
                        f"{prelude_indent}cute.arch.sync_warp()\n",
                    ]
                )
            for aux_idx, rec in enumerate(aux_step_records):
                if aux_ring_smem_names[aux_idx] is None:
                    continue
                tiled_copy_s2r_var = f"{rec.aux_tile}_tiled_copy_s2r"
                tsr_sc_var = f"{rec.aux_tile}_tSR_sC"
                trs_rc_var = f"{rec.aux_tile}_tRS_rC"
                tsr_rc_var = f"{rec.aux_tile}_tSR_rC"
                lines.extend(
                    [
                        (
                            # The S2R visitor layout can carry zero/unused lanes;
                            # filtering keeps the residual SMEM read footprint
                            # aligned with the lanes that feed the R2S fragment.
                            f"{prelude_indent}cute.copy("
                            f"{tiled_copy_s2r_var}, "
                            f"cute.filter_zeros({tsr_sc_var}[None, None, None, "
                            f"{aux_consumer_state_name}.index]), "
                            f"cute.filter_zeros({tsr_rc_var}))\n"
                        ),
                        (f"{prelude_indent}{rec.aux_loaded} = {trs_rc_var}.load()\n"),
                    ]
                )
            lines.extend(
                [
                    # The ring may be refilled immediately after release.
                    # A warp sync or mbarrier arrival alone need not drain
                    # the outstanding per-thread shared loads in SASS.
                    f"{prelude_indent}cute.arch.fence_acq_rel_cta()\n",
                    (
                        f"{prelude_indent}with cute.arch.elect_one():\n"
                        f"{prelude_indent}    {aux_pipeline_name}.consumer_release("
                        f"{aux_consumer_state_name})\n"
                        if aux_pipeline_uses_tma_load
                        # Each SIMT reader orders its own completed reads
                        # before its arrival. Election alone does not hand
                        # other lanes' read completion to the elected lane.
                        else (
                            f"{prelude_indent}{aux_pipeline_name}.consumer_release("
                            f"{aux_consumer_state_name})\n"
                        )
                    ),
                    emit_pipeline_advance(
                        aux_consumer_state_name, indent=prelude_indent
                    )
                    + "\n",
                ]
            )
            if aux_epilogue_issued:
                lines.append(_epi_tma_aux_refill_source(prelude_indent))
        # Broadcast auxiliaries are partitioned against the same accumulator
        # carrier, so their fragments share an index domain, coordinates, and
        # validity predicate. Load all of them in one scalar edge loop instead
        # of traversing that coordinate fragment once per auxiliary tensor.
        if (
            force_simt_edge_aux
            and len(aux_step_records) >= 2
            and all(rec.broadcast_axis is not None for rec in aux_step_records)
        ):
            for rec in aux_step_records:
                lines.extend(
                    [
                        (
                            f"{prelude_indent}{rec.ttr_aux_subtile} = "
                            f"{rec.ttr_aux_grouped}[(None, None, None, "
                            "cutlass.Int32(_tcgen05_subtile))]\n"
                        ),
                        (
                            f"{prelude_indent}{rec.aux_rmem} = "
                            f"cute.make_rmem_tensor({rec.ttr_aux_subtile}.shape, "
                            f"{rec.edge_rmem_dtype})\n"
                        ),
                        f"{prelude_indent}{rec.aux_rmem}.fill(0)\n",
                    ]
                )
            first_rec = aux_step_records[0]
            lines.extend(
                [
                    _simt_edge_coord_subtile_source(prelude_indent),
                    (
                        f"{prelude_indent}for _edge_i in range(cute.size("
                        f"{first_rec.ttr_aux_subtile}.shape)):\n"
                    ),
                    f"{prelude_indent}    _coord = {ttr_cc_subtile}[_edge_i]\n",
                    (
                        f"{prelude_indent}    if cute.elem_less("
                        f"_coord, ({m_size}, {n_size})):\n"
                    ),
                ]
            )
            for rec in aux_step_records:
                lines.append(
                    f"{prelude_indent}        {rec.aux_rmem}[_edge_i] = "
                    f"{rec.ttr_aux_subtile}[_edge_i]\n"
                )
            for rec in aux_step_records:
                lines.append(
                    f"{prelude_indent}{rec.aux_loaded} = {rec.edge_rmem_value}\n"
                )
            return "".join(lines)
        for aux_idx, rec in enumerate(aux_step_records):
            rowvec_stage = rowvec_aux_stage_records[aux_idx]
            if (
                use_aux_smem_source
                and not force_simt_edge_aux
                and aux_ring_smem_names[aux_idx] is not None
            ):
                continue
            if aux_idx in aux_ring_records and not force_simt_edge_aux:
                lines.append(_aux_ring_subtile_load_source(prelude_indent, aux_idx))
                continue
            if force_simt_edge_aux:
                include_coord_setup = not force_simt_edge_coord_emitted
                force_simt_edge_coord_emitted = True
                if rec.broadcast_axis is None and not rec.is_boolean_mask:
                    edge_aux_copy_source = _simt_edge_logical_divide_copy_source(
                        prelude_indent,
                        rec.ttr_aux_subtile,
                        rec.aux_rmem,
                        include_coord_setup=include_coord_setup,
                        var_prefix=f"{rec.aux_rmem}_edge",
                        copy_atom=simt_edge_aux_atoms[aux_idx],
                    )
                else:
                    # Rowvec broadcast stayed scalar in the cycle-74 ablation:
                    # vectorizing it did not reduce stack pressure or runtime.
                    # A boolean mask copies element-wise too: a Boolean copy
                    # atom is narrower than the mask's byte storage.
                    edge_aux_copy_source = _simt_edge_scalar_copy_source(
                        prelude_indent,
                        rec.ttr_aux_subtile,
                        rec.aux_rmem,
                        include_coord_setup=include_coord_setup,
                    )
                lines.append(
                    f"{prelude_indent}{rec.ttr_aux_subtile} = "
                    f"{rec.ttr_aux_grouped}"
                    f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                    f"{prelude_indent}{rec.aux_rmem} = "
                    f"cute.make_rmem_tensor({rec.ttr_aux_subtile}.shape, "
                    f"{rec.edge_rmem_dtype})\n"
                    f"{prelude_indent}{rec.aux_rmem}.fill(0)\n"
                    + edge_aux_copy_source
                    + f"{prelude_indent}{rec.aux_loaded} = "
                    f"{rec.edge_rmem_value}\n"
                )
                continue
            if tcgen05_value.output_column_major and unconditional_full_tile:
                # Column-major broadcasts may retain a nested R2S profile while
                # the accumulator carrier is flat. Materialize the broadcast
                # into the carrier profile, preserving its coordinates and the
                # original order of the epilogue's FP32 operations.
                lines.extend(
                    (
                        (
                            f"{prelude_indent}{rec.ttr_aux_subtile} = "
                            f"{rec.ttr_aux_grouped}"
                            f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                        ),
                        _materialize_broadcast_aux_source(
                            prelude_indent, rec, carrier_name
                        ),
                    )
                )
                continue
            if rowvec_stage is None and unconditional_full_tile:
                lines.append(
                    f"{prelude_indent}{rec.ttr_aux_subtile} = "
                    f"{rec.ttr_aux_grouped}"
                    f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                    f"{prelude_indent}{rec.aux_loaded} = "
                    f"{rec.ttr_aux_subtile}.load()\n"
                )
                continue
            if rowvec_stage is None and (
                safe_direct_aux_with_full_tile or not tcgen05_aux_use_tma_store_epilogue
            ):
                if rec.is_boolean_mask:
                    # A boolean mask on partial tiles. The DSL rejects
                    # rebinding a Boolean TensorSSA across the full/edge
                    # branches (CONTAINER_STRUCTURE_CHANGED): both branches
                    # fill one Int8 fragment (``edge_rmem_dtype``), and the
                    # mask is read after them.
                    lines.append(
                        f"{prelude_indent}{rec.ttr_aux_subtile} = "
                        f"{rec.ttr_aux_grouped}"
                        f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                        f"{prelude_indent}{rec.aux_rmem} = "
                        f"cute.make_rmem_tensor({rec.ttr_aux_subtile}.shape, "
                        f"{rec.edge_rmem_dtype})\n"
                        f"{prelude_indent}{rec.aux_rmem}.fill(0)\n"
                        f"{prelude_indent}if {full_tile}:\n"
                        f"{prelude_indent}    {rec.aux_rmem}.store("
                        f"{rec.ttr_aux_subtile}.load().to(cutlass.Int8))\n"
                        f"{prelude_indent}else:\n"
                        f"{_simt_edge_scalar_copy_source(prelude_indent + '    ', rec.ttr_aux_subtile, rec.aux_rmem)}"
                        f"{prelude_indent}{rec.aux_loaded} = {rec.edge_rmem_value}\n"
                    )
                    continue
                lines.append(
                    f"{prelude_indent}{rec.ttr_aux_subtile} = "
                    f"{rec.ttr_aux_grouped}"
                    f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                    f"{prelude_indent}{rec.aux_loaded} = cute.full("
                    f"{rec.ttr_aux_subtile}.shape, 0, {rec.aux_dtype})\n"
                    f"{prelude_indent}if {full_tile}:\n"
                    f"{prelude_indent}    {rec.aux_loaded} = "
                    f"{rec.ttr_aux_subtile}.load()\n"
                    f"{prelude_indent}else:\n"
                    f"{prelude_indent}    {rec.aux_rmem} = "
                    f"cute.make_rmem_tensor({rec.ttr_aux_subtile}.shape, "
                    f"{rec.aux_dtype})\n"
                    f"{prelude_indent}    {rec.aux_rmem}.fill(0)\n"
                    f"{_simt_edge_scalar_copy_source(prelude_indent + '    ', rec.ttr_aux_subtile, rec.aux_rmem)}"
                    f"{prelude_indent}    {rec.aux_loaded} = "
                    f"{rec.aux_rmem}.load()\n"
                )
                continue
            if rowvec_stage is not None and not force_simt_edge_aux:
                # Row-vector staging broadcasts through a stride-0 M mode; filter
                # that layout so the SMEM read does not reload duplicate lanes.
                # A promoted stage already holds FP32.
                lines.append(
                    f"{prelude_indent}{rec.ttr_aux_subtile} = "
                    f"{rec.ttr_aux_grouped}"
                    "[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                    f"{prelude_indent}{rec.aux_rmem} = "
                    f"cute.make_rmem_tensor({rec.ttr_aux_subtile}.layout, "
                    f"{rowvec_stage.stage_dtype})\n"
                    f"{prelude_indent}cute.autovec_copy("
                    f"cute.filter_zeros({rec.ttr_aux_subtile}), "
                    f"cute.filter_zeros({rec.aux_rmem}))\n"
                    f"{prelude_indent}{rec.aux_loaded} = {rec.aux_rmem}.load()\n"
                )
                continue
            if rec.broadcast_axis == 2:
                # Column-vector (per-row) aux: a rank-2 ``(m, n)`` operand
                # broadcast over N (its value depends only on the row m). When
                # a thread's fragment is within a single M row the per-row value
                # is uniform across it, so the cheap scalar read
                # ``tTR_gAux[(0, 0, 0, subtile)]`` is exact (PR #2742). When it
                # spans multiple M rows the scalar applies row 0's value to
                # every row, so materialize per element instead.
                #
                # Decide this at codegen time via
                # ``tcgen05_colvec_fragment_single_m_row`` (epi_tile_m vs the
                # 128-lane TMEM datapath). A runtime layout test cannot: after
                # ``partition_D`` + ``group_modes`` the per-thread strides are
                # all dynamic (nothing for ``cute.filter`` to drop, and mode 0
                # conflates M and N), so such tests silently degrade to
                # always-materialize.
                lines.append(
                    f"{prelude_indent}{rec.ttr_aux_subtile} = "
                    f"{rec.ttr_aux_grouped}"
                    f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                )
                if tcgen05_colvec_fragment_single_m_row:
                    # A boolean mask's element reads as its byte storage,
                    # which torch reads as ``!= 0``.
                    scalar = (
                        rec.colvec_scalar_full
                        if rec.colvec_scalar_full is not None
                        else f"{rec.ttr_aux_grouped}"
                        "[(0, 0, 0, cutlass.Int32(_tcgen05_subtile))]"
                    )
                    lines.append(
                        f"{prelude_indent}{rec.aux_loaded} = {rec.mask_value(scalar)}\n"
                    )
                else:
                    lines.append(
                        _materialize_broadcast_aux_source(
                            prelude_indent, rec, carrier_name
                        )
                    )
                continue
            if rec.aux_rmem_full is not None:
                # Pre-wait hoisted fragment (rowvec or exact-shape): the whole
                # tile is already in registers (loaded before the accumulator
                # consumer_wait by ``_aux_tile_setup_lines``); slice the active
                # subtile. An exact-shape fragment whose thread spans several
                # M rows (``epi_tile_m`` below the 128-lane TMEM datapath, e.g.
                # block_m=64) keeps the nested T2R profile the flat carrier
                # cannot combine with, so its slice is re-laid out onto the
                # carrier's layout exactly as the per-subtile global read is
                # (a register-to-register copy); a single-row fragment and
                # the rowvec hoist load the slice directly.
                if (
                    rec.broadcast_axis is None
                    and not tcgen05_colvec_fragment_single_m_row
                ):
                    lines.append(
                        f"{prelude_indent}{rec.ttr_aux_subtile} = "
                        f"{rec.aux_rmem_full}"
                        f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                        + _materialize_broadcast_aux_source(
                            prelude_indent, rec, carrier_name
                        )
                    )
                else:
                    lines.append(
                        f"{prelude_indent}{rec.aux_loaded} = "
                        f"{rec.aux_rmem_full}"
                        f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))].load()\n"
                    )
                continue
            # Both remaining cases -- rowvec / leading-broadcast aux
            # (``broadcast_axis in (0, 1)``: a per-column operand broadcast
            # over M) and exact-shape aux (``broadcast_axis is None``: a full
            # ``(m, n)`` operand) -- always materialize. Neither has a cheaper
            # scalar form (the operand is a full per-thread vector either way),
            # and both otherwise produce a nested profile that cannot combine
            # with the flat carrier (rowvec from its stride-0 mode, exact-shape
            # once the fragment spans multiple M rows). The materialize copy is
            # a no-op reshape when the profile already matches (block_m >= the
            # atom M).
            lines.append(
                f"{prelude_indent}{rec.ttr_aux_subtile} = "
                f"{rec.ttr_aux_grouped}"
                f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                + _materialize_broadcast_aux_source(prelude_indent, rec, carrier_name)
            )
        return "".join(lines)

    # Render the per-thread carrier expression for the accumulator
    # vector. The identity epilogue (no chain or empty chain) emits
    # the original `rAcc.load().to(target_dtype)` line. When a
    # chain is present, hoist `rAcc.load()` to a local TensorSSA so
    # the chain reads the loaded vector once; for chains with
    # auxiliary-tensor steps, also emit per-subtile aux-load lines
    # that bind the aux locals the chain references. Each splice
    # site below uses the appropriate carrier name (`ttr_racc` for
    # the SIMT path, `trs_racc` for the TMA path, and
    # `tcgen05_tRS_rAcc` for the @cute.jit module helper). The
    # returned snippet is a sequence of zero-or-more prelude
    # statements (each newline-terminated, indented with
    # `prelude_indent`) plus the assignment expression for
    # `tcgen05_acc_vec`.
    def _splice_acc_vec(
        carrier_name: str,
        prelude_indent: str,
        *,
        force_simt_edge_aux: bool = False,
        safe_direct_aux_with_full_tile: bool = False,
        coord_layout: str = "ttr",
        unconditional_full_tile: bool = False,
        bind_runtime_scalars: bool = True,
    ) -> tuple[str, str, str]:
        """Return ``(early_aux_prelude, late_prelude, assignment_rhs)``.

        ``early_aux_prelude`` is the per-subtile auxiliary-tensor LDG
        block (``ttr_aux_subtile = ...``; ``aux_loaded = .load()``) and
        is empty when the chain has no aux steps. ``late_prelude``
        holds the ``acc_loaded = carrier.load()`` and the chain-step
        renderings. ``assignment_rhs`` is the right-hand side of
        ``acc_vec = ...`` (without leading whitespace or the trailing
        newline). Both preludes are empty for the identity epilogue
        (no chain) — in that case ``assignment_rhs`` is the original
        ``carrier.load().to(target_dtype)`` expression.

        Each chain step renders into a fresh ``tcgen05_chain_step*``
        local so chain composition stays linear in source size — the
        relu template duplicates ``{inner}`` 5 times, so without per-
        step binding a 3-deep relu chain would emit 125x duplication
        and pessimize parse / IR-build time. Per-step locals keep
        the rendered source O(N) in chain depth and CuTe CSEs the
        loads at compile.

        Auxiliary-tensor chain steps additionally emit per-aux-step
        ``ttr_aux_subtile = ...`` slice + ``aux_loaded = ...`` lines
        (the per-tile aux setup runs once per output tile and is
        emitted by the splice site's surrounding scaffolding via
        ``_aux_tile_setup_lines()``). Splitting the aux LDG out of
        the chain prelude lets each splice site place the GMEM load
        where it best fits its live ranges. The default TMA-store
        splice now inserts it after the c_pipeline acquire, acc
        ``consumer_wait``, and t2r async TMEM→reg copy so residual
        and bias fragments are not live through those prefix waits.
        SIMT-store edge tiles use the same aux prelude, but route
        the aux load through a predicated copy before rendering the
        chain.
        """
        nonlocal fanout_chain
        load_expr = f"{carrier_name}.load()"
        if fragment_epilogue is not None:
            coordinate_setup, coordinate_name = _coord_subtile_source(
                prelude_indent, coord_layout, include_setup=True
            )
            from ..fragment_epilogue import render_tcgen05_fragment_epilogue

            fragment_prelude, fragment_expression = render_tcgen05_fragment_epilogue(
                state,
                fragment_epilogue,
                carrier_name=carrier_name,
                coordinate_name=coordinate_name,
                target_dtype=target_dtype,
                indent=prelude_indent,
            )
            return "", coordinate_setup + fragment_prelude, fragment_expression
        if epilogue_chain is None or not epilogue_chain.steps:
            rhs = load_expr
            late_prelude = ""
            if worklist_nm_segment_valid_m_bound:
                coord_prelude, coord_subtile = _coord_subtile_source(
                    prelude_indent,
                    coord_layout,
                    include_setup=True,
                )
                worklist_valid_prelude, worklist_valid_expr = (
                    _worklist_nm_segment_valid_m_prelude_and_expr(
                        prelude_indent,
                        carrier_name,
                        coord_subtile,
                    )
                )
                rhs = f"cute.where({worklist_valid_expr}, {rhs}, 0.0)"
                late_prelude = coord_prelude + worklist_valid_prelude
            return "", late_prelude, f"({rhs}).to({target_dtype})"
        loaded = df.new_var("tcgen05_acc_loaded")
        prelude_load = (
            f"{prelude_indent}{loaded} = {load_expr}\n"
            if chain_reads_accumulator
            else ""
        )
        early_aux_prelude = _aux_subtile_load_source(
            prelude_indent,
            carrier_name,
            force_simt_edge_aux=force_simt_edge_aux,
            safe_direct_aux_with_full_tile=safe_direct_aux_with_full_tile,
            unconditional_full_tile=unconditional_full_tile,
        )
        aux_locals_by_expr: _LeafBindings = {
            rec.expr: rec.aux_loaded for rec in aux_step_records
        }
        assert len(aux_locals_by_expr) == len(aux_step_records)
        if bind_runtime_scalars:
            # Bound once per tile by ``_aux_tile_setup_lines``; the diagnostic
            # module-helper layout renders into a separate function and keeps
            # the inline reads.
            aux_locals_by_expr.update(runtime_scalar_locals)
        if (
            len(epilogue_chain.steps) == 1
            and epilogue_chain.steps[0].hoistable_aux_expr is not None
        ):
            # The auxiliary expression does not depend on the accumulator.
            # Form it with the auxiliary-load prelude so pre-wait placement can
            # overlap its loads and elementwise operations with the MMA tail.
            expr_step = epilogue_chain.steps[0]
            aux_expr_prelude, aux_expr = (
                expr_step.render_hoistable_aux_prelude_and_expr(
                    aux_locals_by_expr,
                    df.new_var,
                    prelude_indent,
                )
            )
            final_expr = df.new_var("tcgen05_chain_step")
            rendered_step = expr_step.render_with_hoisted_aux(loaded, aux_expr)
            return (
                early_aux_prelude + aux_expr_prelude,
                prelude_load + f"{prelude_indent}{final_expr} = {rendered_step}\n",
                f"({final_expr}).to({target_dtype})",
            )
        if fanout_plan is not None and coord_layout == "trs":
            fanout_chain = render_fanout_chain(
                epilogue_chain, loaded, df.new_var, prelude_indent, aux_locals_by_expr
            )
            chain_prelude, final_expr = fanout_chain.source, fanout_chain.result
        else:
            chain_prelude, final_expr = epilogue_chain.render_prelude_and_expr(
                loaded,
                df.new_var,
                prelude_indent,
                aux_locals_by_expr=aux_locals_by_expr or None,
            )
        return (
            early_aux_prelude,
            prelude_load + chain_prelude,
            f"({final_expr}).to({target_dtype})",
        )

    grouped_tma_plan = matmul_plan if grouped_tail_epilogue or segment_store else None
    grouped_tma = grouped_tma_plan.grouped if grouped_tma_plan is not None else None
    d_tma_uses_rank3_mnl_tensor = (
        (tcgen05_value.output_column_major and not leading_passthrough_output)
        or (row_union is not None and row_union.schedule is not None)
        or grouped_fixed_d_tensormap
        or grouped_bounded_output_shift
        or (
            grouped_tma is not None
            and grouped_tma.d_mode is not Tcgen05GroupedDMode.NONE
            and grouped_tma.d_tensormap is not None
            and grouped_tma.d_mode is not Tcgen05GroupedDMode.EDGE_ONLY
        )
    )
    d_tma_uses_tail_rank3_mnl_tensor = (
        grouped_tma is not None
        and grouped_tma.d_mode is not Tcgen05GroupedDMode.NONE
        and grouped_tma.d_tensormap is not None
        and grouped_tma.d_mode is Tcgen05GroupedDMode.EDGE_ONLY
        and bool(tcgen05_value.tail_tma_store_atom)
        and bool(tcgen05_value.tail_tma_store_tensor)
    )
    if tcgen05_value.use_tma_store_epilogue:
        _check_tcgen05_tma_store_destination(
            tensor,
            tensor_name,
            transposed_store=tcgen05_nm_store or row_union is not None,
            output_column_major=tcgen05_value.output_column_major,
            radix_store=radix_layout is not None,
        )
        df.placeholder_args.add(tensor_name)
        df.wrapper_only_params.extend(
            [
                tcgen05_value.tma_store_atom,
                tcgen05_value.tma_store_tensor,
                *(
                    [
                        tcgen05_value.tail_tma_store_atom,
                        tcgen05_value.tail_tma_store_tensor,
                    ]
                    if d_tma_uses_tail_rank3_mnl_tensor
                    else []
                ),
            ]
        )
        if tcgen05_value.use_role_local_epi and tcgen05_value.role_local_tile_counter:
            df.cute_state.register_tcgen05_epi_role_tile_counter(
                tcgen05_value.role_local_tile_counter,
                increment_per_tile=not tcgen05_value.tma_store_full_tiles_only,
            )
        # The bm=128 CtaGroup.TWO family's epilogue tile is N-mode permuted
        # (see ``tcgen05_two_cta_m128_epilogue_tile_expr``); the host TMA-store
        # atom must be built from this *exact* device-side expression, not from
        # the plain ``epi_tile_m/n`` integer keys (which build an unpermuted
        # ``(m, n)`` tile and silently scramble the output -- the correctness
        # bug §7 in FINDINGS_512_SHAPE.md). ``epi_tile_raw_expr`` carries the
        # verbatim device expression to the wrapper.
        d_two_cta_m128 = tcgen05_is_two_cta_m128(
            is_two_cta=tcgen05_lifecycle.is_two_cta, bm=tcgen05_value.bm
        )
        # The wrapper sizes the TMA store box from the output dtype unless
        # the plan names the tile: a BF16 store beside an FP32 one passes the
        # FP32-sized tile its device loop stages (the expression shown at
        # ``cute_mma._tcgen05_epilogue_tile_dtype``). Without it the host
        # builds a (128, 32) BF16 box over the (128, 16) SMEM subtiles and the
        # stored values come back wrong.
        d_epi_tile_raw_expr = (
            tcgen05_store_epi_tile_expr
            if (d_two_cta_m128 or epi_tile_dtype != target_dtype)
            and not tcgen05_value.has_explicit_epilogue_tile
            else None
        )
        d_tma_plan: dict[str, object] = {
            "kind": "tcgen05_d_tma",
            "d_name": tensor_name,
            "bm": tcgen05_tma_store_bm,
            "bn": tcgen05_tma_store_bn,
            "c_stage_count": tcgen05_value.c_stage_count,
            "output_dtype": target_dtype,
            "kernel_args": [
                tcgen05_value.tma_store_atom,
                tcgen05_value.tma_store_tensor,
            ],
            **({"orientation": "nm"} if tcgen05_nm_store else {}),
            **({"fixed_tensormap": True} if grouped_fixed_d_tensormap else {}),
            **({"rank3_mnl_tensor": True} if d_tma_uses_rank3_mnl_tensor else {}),
            **({"bounded_output_shift": True} if grouped_bounded_output_shift else {}),
            **(
                {"epi_tile_raw_expr": d_epi_tile_raw_expr}
                if d_epi_tile_raw_expr is not None
                else {}
            ),
            **(
                {
                    "epi_tile_m": tcgen05_value.explicit_epi_tile_m,
                    "epi_tile_n": tcgen05_value.explicit_epi_tile_n,
                    "d_store_box_n": tcgen05_value.explicit_d_store_box_n,
                }
                if tcgen05_value.has_explicit_epilogue_tile
                else {}
            ),
        }
        if leading_passthrough_output:
            d_tma_plan["d_leading_passthrough"] = True
        if gemm_extents is not None:
            d_tma_plan["d_extents"] = list(gemm_extents)
        if tcgen05_value.output_column_major:
            d_tma_plan["d_column_major"] = True
        if radix_layout is not None:
            d_tma_plan["d_hier_layout"] = list(radix_layout.cute_layout_args())
        wrapper_plans = cute_codegen_state(state.codegen).wrapper_plans
        wrapper_plans.append(d_tma_plan)
        if d_tma_uses_tail_rank3_mnl_tensor:
            tail_d_tma_plan = {
                **d_tma_plan,
                "kernel_args": [
                    tcgen05_value.tail_tma_store_atom,
                    tcgen05_value.tail_tma_store_tensor,
                ],
                "rank3_mnl_tensor": True,
            }
            wrapper_plans.append(tail_d_tma_plan)

    tcgen05_bm = tcgen05_value.bm
    tcgen05_bn = tcgen05_value.bn
    tcgen05_bk = tcgen05_value.bk
    tcgen05_epilog_sync_barrier_id = tcgen05_value.epilog_sync_barrier_id
    # Stages of the C ring each epilogue warp group cycles through (the SMEM
    # layout, the kernel descriptor and the wrapper's TMA store box take the
    # whole ``tcgen05_value.c_stage_count`` ring).
    tcgen05_c_stage_count = tcgen05_value.c_stages_per_group
    # ``tcgen05_epi_round_subtiles``: subtiles staged and stored per round
    # (``tcgen05_epilogue_rounds``), a choice dividing ``tcgen05_c_stages``
    # (the strict normalizer's ``epi_round_subtiles_refusal``).  The
    # bulk-group pipeline counts rounds, so it has a stage per round of C
    # buffers; the ring itself keeps ``c_stage_count`` subtile buffers.
    tcgen05_epi_round_subtiles = cast(
        "int",
        state.device_function.config.get(TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY, 1),
    )
    tcgen05_c_pipeline_stage_count = tcgen05_c_stage_count // tcgen05_epi_round_subtiles
    tcgen05_is_two_cta = tcgen05_lifecycle.is_two_cta
    tcgen05_thr_mma = tcgen05_value.thr_mma
    full_tile_expr = (
        (
            f"{segment_store_local_m} + cutlass.Int32({tcgen05_source_bm}) "
            f"<= cutlass.Int32({tcgen05_value.segment_store_actual_m}) "
            f"and {segment_store_base_m} + cutlass.Int32({tcgen05_source_bm}) "
            f"<= {m_size} "
        )
        if segment_store
        else f"({m_index}) + cutlass.Int32({tcgen05_source_bm}) <= {m_size} "
    ) + f"and ({n_index}) + cutlass.Int32({tcgen05_source_bn}) <= {n_size}"
    if row_union is not None and row_union.schedule is not None:
        full_tile_expr = (
            row_union.all_rows
            if row_union.linear_record_clc
            else f"{row_union.all_rows} and ({full_tile_expr})"
        )
    grouped_tail = grouped_tma
    grouped_full_coverage_predicate = (
        grouped_tail.full_coverage.predicate
        if grouped_tail is not None and grouped_tail.full_coverage is not None
        else None
    )
    grouped_tail_store = (
        grouped_tail is not None
        and bool(grouped_tail.global_m_start)
        and bool(grouped_tail.problem_m)
        and bool(grouped_tail.problem_n)
    )
    grouped_tail_global_m_start = ""
    grouped_tail_problem_m = ""
    grouped_tail_problem_n = ""
    grouped_dynamic_d_tensormap = False
    grouped_tail_metadata_idx = ""
    grouped_tail_cta_tile_idx_m = ""
    grouped_tail_cta_tile_idx_n = ""
    grouped_tail_d_tensormap = ""
    grouped_tail_d_tensormap_slot = 0
    grouped_direct_pointer_metadata = False
    grouped_direct_pointers = ""
    grouped_direct_strides = ""
    grouped_dynamic_d_tensormap_edge_only = False
    grouped_tail_store_problem_m = ""
    grouped_tail_d_tensormap_problem_m = ""
    if grouped_tail_store:
        assert grouped_tail is not None
        grouped_tail_global_m_start = grouped_tail.global_m_start
        grouped_tail_problem_m = grouped_tail.problem_m
        grouped_tail_problem_n = grouped_tail.problem_n
        grouped_tail_store_problem_m = (
            f"cutlass.Int32({tcgen05_value.segment_store_actual_m})"
            if segment_store and tcgen05_value.orientation is Tcgen05Orientation.NM
            else grouped_tail_problem_m
        )
        grouped_tail_d_tensormap_problem_m = (
            grouped_tail_problem_m if tcgen05_nm_store else grouped_tail_store_problem_m
        )
        grouped_tail_metadata_idx = grouped_tail.metadata_idx
        grouped_tail_cta_tile_idx_m = grouped_tail.cta_tile_idx_m
        grouped_tail_cta_tile_idx_n = grouped_tail.cta_tile_idx_n
        grouped_tail_d_tensormap = grouped_tail.d_tensormap or ""
        grouped_tail_d_tensormap_slot = (
            2 if grouped_tail.d_mode is Tcgen05GroupedDMode.ALL_TILES else 0
        )
        grouped_direct_pointer_metadata = grouped_tail.direct_pointers is not None
        grouped_direct_pointers = grouped_tail.direct_pointers or ""
        grouped_direct_strides = grouped_tail.direct_strides or ""
        grouped_dynamic_d_tensormap = (
            grouped_tail.d_mode is not Tcgen05GroupedDMode.NONE
            and bool(grouped_tail_d_tensormap)
        )
        grouped_dynamic_d_tensormap_edge_only = (
            grouped_dynamic_d_tensormap
            and grouped_tail.d_mode is Tcgen05GroupedDMode.EDGE_ONLY
            and tcgen05_value.tma_store_full_tiles_only
        )
        if segment_store:
            segment_grouped_local_m = (
                "cutlass.Int32(0)"
                if tcgen05_nm_store
                else f"(({segment_store_local_m}) - {grouped_tail_global_m_start})"
            )
            full_tile_expr = (
                f"{segment_grouped_local_m} + cutlass.Int32({tcgen05_source_bm}) "
                f"<= {grouped_tail_store_problem_m} "
                f"and ({segment_store_base_m}) + cutlass.Int32({tcgen05_source_bm}) "
                f"<= {m_size} "
                f"and ({n_index}) + cutlass.Int32({tcgen05_source_bn}) "
                f"<= {grouped_tail_problem_n} "
                f"and ({n_index}) + cutlass.Int32({tcgen05_source_bn}) "
                f"<= {n_size}"
            )
        else:
            full_tile_expr = (
                f"{full_tile_expr} and "
                f"(({m_index}) - {grouped_tail_global_m_start}) "
                f"+ cutlass.Int32({tcgen05_source_bm}) <= {grouped_tail_store_problem_m} "
                f"and ({n_index}) + cutlass.Int32({tcgen05_source_bn}) "
                f"<= {grouped_tail_problem_n}"
            )
    grouped_dynamic_d_tensormap_all_tiles = (
        grouped_dynamic_d_tensormap and not grouped_dynamic_d_tensormap_edge_only
    )

    def _edge_predicate_expr(coord: str) -> str:
        if segment_store:
            if grouped_tail_store:
                local_m = (
                    f"{coord}[0]"
                    if tcgen05_nm_store
                    else (
                        f"(({segment_store_local_m}) - {grouped_tail_global_m_start}) "
                        f"+ {coord}[0]"
                    )
                )
                actual_m = grouped_tail_store_problem_m
            else:
                local_m = f"({segment_store_local_m}) + {coord}[0]"
                actual_m = f"cutlass.Int32({tcgen05_aux_segment_store_actual_m})"
            global_m = f"({segment_store_base_m}) + {coord}[0]"
            global_n = f"({n_index}) + {coord}[1]"
            return (
                f"{local_m} < {actual_m} "
                f"and cutlass.Int32(0) <= {global_m} "
                f"and {global_m} < {m_size} "
                f"and {global_n} < {n_size}"
            )
        global_pred = f"cute.elem_less({coord}, ({m_size}, {n_size}))"
        if not grouped_tail_store:
            return global_pred
        return (
            f"{global_pred} and "
            f"({coord}[0] - {grouped_tail_global_m_start}) >= cutlass.Int32(0) "
            f"and ({coord}[0] - {grouped_tail_global_m_start}) "
            f"< {grouped_tail_store_problem_m} "
            f"and {coord}[1] < {grouped_tail_problem_n}"
        )

    def _edge_vector_predicate_expr(coord: str, vector: str) -> str:
        # A whole per-thread chunk lies inside the destination: its row is
        # valid and its last column is inside N (chunk starts are multiples
        # of the vector, as are the tile and subtile column offsets).
        return f"{coord}[0] < {m_size} and {coord}[1] + {vector} <= {n_size}"

    def _edge_tail_predicate_expr(coord: str, vector: str) -> str:
        # Elements of the one chunk per row that straddles the N edge.
        return (
            f"cute.elem_less({coord}, ({m_size}, {n_size})) and "
            f"({coord}[1] // {vector}) * {vector} + {vector} > {n_size}"
        )

    def store_common_setup(
        gmem_tensor: str,
        *,
        include_full_tile: bool,
        tma_store: bool = False,
        dynamic_d_tensormap: bool = False,
        rank3_mnl_tensor: bool | None = None,
        static_strides: tuple[int, ...] | None = None,
    ) -> tuple[list[str], list[str]]:
        if rank3_mnl_tensor is None:
            rank3_mnl_tensor = d_tma_uses_rank3_mnl_tensor
        store_bm = tcgen05_tma_store_bm if tma_store else tcgen05_bm
        store_bn = tcgen05_tma_store_bn if tma_store else tcgen05_bn
        epi_tile_expr = tcgen05_store_epi_tile_expr
        static_setup = [
            (
                f"{kernel_desc} = type('Tcgen05KernelDesc', (), {{"
                f"'cta_tile_shape_mnk': ({tcgen05_store_tile_m}, {tcgen05_bn}, {tcgen05_bk}), "
                f"'c_layout': {tcgen05_d_store_layout}, "
                f"'c_dtype': {target_dtype}, "
                "'acc_dtype': cutlass.Float32, "
                f"'epilog_sync_bar_id': cutlass.Int32({tcgen05_epilog_sync_barrier_id}), "
                f"'epilogue_warp_id': ({epi_warp_ids}), "
                f"'num_c_stage': cutlass.Int32({tcgen05_value.c_stage_count}), "
                f"'use_2cta_instrs': {tcgen05_is_two_cta!s}"
                "})()"
            ),
            (
                # The fallback helper must receive the plan's epilogue dtype
                # (the widest output store's, maybe wider than this store's)
                # through ``layout_c=`` / ``elem_ty_c=`` so it selects the same
                # with-source branch and tile as the matmul-plan
                # ``tcgen05_epi_tile``.
                # The explicit path instead uses the D-store box field directly.
                # Keep both forms in lockstep with the wrapper-side TMA atom.
                f"{epi_tile} = {epi_tile_expr}"
            ),
        ]
        tile_setup: list[str] = []
        if include_full_tile:
            tile_setup.append(f"{full_tile} = {full_tile_expr}")
        local_gmem_tensor = gmem_tensor
        static_view_built = False
        if row_union is not None and not tma_store:
            # The proof establishes a fresh contiguous torch.empty result.
            # Preserve the original pointer/ABI while exposing that allocation
            # layout to the vector-copy partitioner.
            local_gmem_tensor = df.new_var("tcgen05_row_union_output")
            layout = (
                f"({row_union.n}, {row_union.m}), stride=(1, {row_union.n})"
                if row_union.schedule is not None
                else f"({row_union.m}, {row_union.n}), stride=({row_union.n}, 1)"
            )
            static_setup.append(
                f"{local_gmem_tensor} = cute.make_tensor({gmem_tensor}.iterator, "
                f"cute.make_layout({layout}))"
            )
        elif static_strides is not None and not tma_store:
            # Mixed SIMT store over a ragged destination: the launcher keeps
            # partial-tile kernels on runtime tensor layouts, so the DSL sees
            # dynamic strides and ``max_common_layout`` admits only element
            # copies (bf16 1024x1024x1000 stored 1M two-byte scalars: 56 us
            # against 8.6 us). Re-view the destination with its compile-time
            # layout; the interior tiles then store 128-bit vectors and the
            # fringe predicates still come from the identity tensor.
            local_gmem_tensor = df.new_var("tcgen05_static_output")
            static_view_built = True
            static_layout = (
                f"cute.make_layout(({tensor.shape[-2]}, {tensor.shape[-1]}), "
                f"stride=({static_strides[-2]}, {static_strides[-1]}))"
            )
            if leading_passthrough_output:
                assert leading_index is not None
                index_dtype = CompileEnvironment.current().index_type()
                tile_setup.append(
                    f"{local_gmem_tensor} = cute.make_tensor("
                    f"{gmem_tensor}.iterator + {index_dtype}({leading_index}) * "
                    f"{index_dtype}({static_strides[0]}), {static_layout})"
                )
            else:
                static_setup.append(
                    f"{local_gmem_tensor} = cute.make_tensor("
                    f"{gmem_tensor}.iterator, {static_layout})"
                )
        if leading_passthrough_output and tma_store:
            assert leading_index is not None
            gmem_tile_3d = df.new_var("tcgen05_gC3d")
            tile_setup.extend(
                [
                    (
                        f"{gmem_tile_3d} = cute.local_tile("
                        f"{gmem_tensor}, ({store_bm}, {store_bn}, 1), "
                        f"({tile_coord_m}, {tile_coord_n}, "
                        f"cutlass.Int32({leading_index})))"
                    ),
                    f"{gmem_tile} = {gmem_tile_3d}[(None, None, 0)]",
                    f"{tcgc_base} = {tcgen05_thr_mma}.partition_C({gmem_tile})",
                ]
            )
            return static_setup, tile_setup
        if leading_passthrough_output and not static_view_built:
            assert leading_index is not None
            local_gmem_tensor = df.new_var("tcgen05_gmem2d")
            tile_setup.append(
                _cute_leading_passthrough_view_2d(
                    local_gmem_tensor,
                    gmem_tensor,
                    leading_index,
                )
            )
        if row_union is not None and row_union.schedule is not None and tma_store:
            local_gmem_tensor = f"{gmem_tensor}[None, None, 0]"
            tile_coord = f"({static_tile_coord_n}, {static_tile_coord_m})"
        elif (
            grouped_bounded_output_shift
            and gmem_tensor
            in (tcgen05_aux_tma_store_tensor, tcgen05_aux_tail_tma_store_tensor)
            and rank3_mnl_tensor
        ):
            assert grouped_tma is not None and grouped_tma.m_size is not None
            row_delta = f"cutlass.Int32({grouped_tma.m_size}) - {grouped_tma.problem_m}"
            row_end = f"{grouped_tma.global_m_start} + {grouped_tma.problem_m}"
            offset = (
                f"(cutlass.Int32(0), {row_delta}, {row_end})"
                if tcgen05_nm_store
                else f"({row_delta}, cutlass.Int32(0), {row_end})"
            )
            local_gmem_tensor = df.new_var("tcgen05_bounded_output")
            tile_setup.append(
                f"{local_gmem_tensor} = cute.domain_offset({offset}, {gmem_tensor})"
            )
            coord_m, coord_n = grouped_tma.cta_tile_idx_m, grouped_tma.cta_tile_idx_n
            if tcgen05_nm_store:
                coord_m, coord_n = coord_n, coord_m
            tile_coord = f"({coord_m}, {coord_n}, 0)"
        elif (
            gmem_tensor
            in (tcgen05_aux_tma_store_tensor, tcgen05_aux_tail_tma_store_tensor)
            and rank3_mnl_tensor
        ):
            if grouped_fixed_d_tensormap:
                source_coord_m = (
                    f"{grouped_tail_global_m_start} // "
                    f"cutlass.Int32({tcgen05_source_bm}) + "
                    f"{grouped_tail_cta_tile_idx_m}"
                )
                source_coord_n = grouped_tail_cta_tile_idx_n
            else:
                source_coord_m = (
                    grouped_tail_cta_tile_idx_m
                    if dynamic_d_tensormap
                    else static_tile_coord_m
                )
                source_coord_n = (
                    grouped_tail_cta_tile_idx_n
                    if dynamic_d_tensormap
                    else static_tile_coord_n
                )
            if tcgen05_nm_store:
                coord_m = source_coord_n
                coord_n = source_coord_m
            else:
                coord_m = source_coord_m
                coord_n = source_coord_n
            tile_coord = f"({coord_m}, {coord_n}, 0)"
        elif row_union is not None and row_union.schedule is not None:
            tile_coord = f"({static_tile_coord_n}, {static_tile_coord_m})"
        else:
            tile_coord = f"({static_tile_coord_m}, {static_tile_coord_n})"
        if segment_store and gmem_tensor == tensor_name:
            index_dtype = CompileEnvironment.current().index_type()
            tile_setup.extend(
                [
                    (
                        f"{gmem_tile} = cute.make_tensor("
                        f"{gmem_tensor}.iterator + "
                        f"{index_dtype}({segment_store_base_m}) * "
                        f"{index_dtype}({gmem_tensor}.layout.stride[0]) + "
                        f"{index_dtype}({n_index}) * "
                        f"{index_dtype}({gmem_tensor}.layout.stride[1]), "
                        "cute.make_layout("
                        f"({store_bm}, {store_bn}), "
                        f"stride=({gmem_tensor}.layout.stride[0], "
                        f"{gmem_tensor}.layout.stride[1])))"
                    ),
                    f"{coord_tile} = cute.make_identity_tensor(({store_bm}, {store_bn}))",
                    f"{tcgc_base} = {tcgen05_thr_mma}.partition_C({gmem_tile})",
                ]
            )
        else:
            tile_setup.extend(
                [
                    (
                        f"{gmem_tile} = cute.local_tile("
                        f"{local_gmem_tensor}, ({store_bm}, {store_bn}), "
                        f"{tile_coord})"
                    ),
                    f"{tcgc_base} = {tcgen05_thr_mma}.partition_C({gmem_tile})",
                ]
            )
        return static_setup, tile_setup

    # A hybrid TMA/SIMT epilogue reaches SIMT only for edge tiles. A static
    # output smaller than its selected tile also cannot produce a full tile,
    # independent of whether its storage is row-major or column-major.
    simt_edge_only = (
        tcgen05_value.tma_store_full_tiles_only
        and not grouped_dynamic_d_tensormap_edge_only
    )
    # A destination larger than the GEMM (``gemm_extents``) is bounded at
    # the GEMM's (M, N), like the edge predicates above.
    static_m_size, static_n_size = (
        radix_layout.extents
        if radix_layout is not None
        else gemm_extents
        if gemm_extents is not None
        else tensor.shape[-2:]
    )
    simt_edge_only = simt_edge_only or (
        isinstance(static_m_size, int)
        and isinstance(static_n_size, int)
        and (static_m_size < tcgen05_value.bm or static_n_size < tcgen05_value.bn)
    )
    simt_edge_aux_atoms: dict[int, str] = {}
    simt_edge_aux_atom_setup: list[str] = []
    if simt_edge_only:
        for aux_idx, rec in enumerate(aux_step_records):
            if rec.broadcast_axis is None and not rec.is_boolean_mask:
                edge_aux_atom = df.new_var(f"{rec.aux_rmem}_edge_atom")
                simt_edge_aux_atoms[aux_idx] = edge_aux_atom
                # Use a per-aux atom typed to the aux dtype. Reusing the
                # output SIMT atom here was spill-free but slower on the
                # measured Target8 edge path.
                simt_edge_aux_atom_setup.append(
                    f"{edge_aux_atom} = "
                    f"cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), "
                    f"{rec.aux_dtype})"
                )
    # ``tcgen05_c_store_mode="direct"`` on a statically full-tiled output:
    # every tile of this store is a whole tile (the output extents divide the
    # tile; no segment, grouped-tail or row-union remap), so the SIMT body
    # writes each subtile with one vectorized ``CopyR2GOp`` copy and carries
    # neither the runtime full-tile predicate nor the scalar edge loop.
    direct_full_tile_store = (
        state.device_function.config.get(
            TCGEN05_C_STORE_MODE_CONFIG_KEY, TCGEN05_C_STORE_MODE_NORMAL
        )
        == TCGEN05_C_STORE_MODE_DIRECT
        and not tcgen05_value.use_tma_store_epilogue
        and not simt_edge_only
        and row_union is None
        and not segment_store
        and not grouped_tail_store
        and isinstance(static_m_size, int)
        and isinstance(static_n_size, int)
        and tcgen05_value.bm > 0
        and tcgen05_value.bn > 0
        and static_m_size % tcgen05_value.bm == 0
        and static_n_size % tcgen05_value.bn == 0
    )
    # The mixed SIMT epilogue: a vector copy on interior tiles, predicated
    # element copies on fringe tiles (ragged flat tcgen05 kernels).
    simt_mixed_store = (
        not simt_edge_only
        and not direct_full_tile_store
        and not tcgen05_value.use_tma_store_epilogue
    )
    simt_static_strides: tuple[int, ...] | None = None
    if (
        simt_mixed_store
        and tensor.ndim in (2, 3)
        and not segment_store
        and not grouped_tail_store
        and row_union is None
    ):
        strides = tuple(tensor.stride())
        if (
            all(isinstance(value, int) for value in (*strides, *tensor.shape))
            and strides[-1] == 1
            and not tcgen05_value.output_column_major
        ):
            simt_static_strides = strides
    simt_static_store_setup, simt_tile_store_setup = store_common_setup(
        tensor_name,
        include_full_tile=not simt_edge_only and not direct_full_tile_store,
        static_strides=simt_static_strides,
    )
    # Vector width of the direct store.  The DSL only knows the kernel
    # argument's pointer alignment; the tile / thread offsets it adds are
    # dynamic, so it would fall back to element-wide stores.  With static
    # strides the alignment of every per-thread chunk is provable at codegen
    # time: the row (and batch) strides are whole 16-byte multiples, each
    # thread's chunk starts at a column that is a whole multiple of its own
    # width (the T2R partition tiles the subtile by the thread's chunk), and
    # the tile / subtile column offsets are multiples of the chunk too.  The
    # generated code then re-asserts that alignment on the per-subtile pointer
    # (``make_ptr(..., assumed_align=...)`` on the same address: no rounding,
    # so a wrong claim faults loudly instead of corrupting the output) and the
    # copy atom widens to 128-bit stores.
    # The mixed SIMT epilogue proves the same alignment: its chunk addresses
    # are identical, and a fringe tile only narrows the predicate.
    direct_store_align_bytes: int | None = None
    if direct_full_tile_store or simt_mixed_store:
        elem_bytes = tensor.element_size()
        strides = tuple(tensor.stride())
        if (
            all(isinstance(stride, int) for stride in strides)
            and strides[-1] == 1
            and not tcgen05_value.output_column_major
            and all((stride * elem_bytes) % 16 == 0 for stride in strides[:-1])
            and (tcgen05_value.bn * elem_bytes) % 16 == 0
        ):
            direct_store_align_bytes = 16
    # A destination whose rows are not 16-byte multiples (attention scores
    # stored in 1001-element rows, window attention's 49-element ones): each
    # thread's T2R fragment is one row's columns, so a warp's store
    # instruction writes 32 rows two bytes each, 32 sectors for 64 useful
    # bytes. The staged store instead writes the tile's fragments into a
    # padded SMEM tile, and the epilogue warps then store it row by row, each
    # warp instruction 32 consecutive elements of one row
    # (``_staged_simt_store_lines``), on interior and edge tiles alike. B200,
    # 64 x 1000x1000x64 into 1001-element rows on the 64x128 tile: 197 ->
    # 135 us at two AB stages, 96 us at one.
    staged_store_pad = 16 // tensor.element_size()
    staged_store_bytes = (
        tcgen05_value.bm * (tcgen05_value.bn + staged_store_pad) * tensor.element_size()
    )
    output_strides = tuple(tensor.stride())
    unaligned_rows = (
        tensor.ndim in (2, 3)
        and all(isinstance(value, int) for value in (*output_strides, *tensor.shape))
        and output_strides[-1] == 1
        and not tcgen05_value.output_column_major
        and any(
            (cast("int", stride) * tensor.element_size()) % 16
            for stride in output_strides[:-1]
        )
    )
    staged_plan = df.cute_state.tcgen05.matmul_plan
    staged_store = (
        (simt_mixed_store or simt_edge_only)
        and not tcgen05_value.use_tma_store_epilogue
        # a flat kernel's one tile per CTA (the setup sits in the per-tile
        # body) and one store (each store site allocates its own tile)
        and not tcgen05_value.use_role_local_epi
        and not has_store_fanout
        and staged_plan is not None
        and unaligned_rows
        and not segment_store
        and not grouped_tail_store
        and row_union is None
        and not grouped_epilogue
        and not tcgen05_lifecycle.is_two_cta
        and (fragment_epilogue is None or not fragment_epilogue.changes_shape)
        and tcgen05_value.bn % 32 == 0
        and tcgen05_value.bm % tcgen05_value.epi_warp_count == 0
        and staged_store_bytes <= _STAGED_SIMT_STORE_MAX_BYTES
        and _staged_simt_store_fits(staged_plan, staged_store_bytes)
    )
    if fragment_epilogue is not None and fragment_epilogue.changes_shape:
        simt_early_aux = simt_late_prelude = ""
        simt_acc_vec_rhs = ""
    else:
        simt_early_aux, simt_late_prelude, simt_acc_vec_rhs = _splice_acc_vec(
            ttr_racc,
            "        ",
            force_simt_edge_aux=simt_edge_only,
            unconditional_full_tile=direct_full_tile_store,
        )
    simt_acc_vec_prelude = simt_early_aux + simt_late_prelude
    # SIMT auxiliary loads are independent of the accumulator. Edge-only
    # stores always issue them early; full-tile SIMT stores do so when the
    # placement config explicitly requests it. In either case their GMEM
    # latency overlaps the MMA tail, and TMEM is copied only after the wait.
    prefetch_simt_aux = (
        (
            simt_edge_only
            or (
                tcgen05_aux_load_placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
                and not tcgen05_value.use_tma_store_epilogue
            )
        )
        and bool(aux_steps_in_chain)
        and not is_secondary_store
    )
    simt_acc_wait = (
        "        if _tcgen05_subtile == 0:\n"
        f"            {tcgen05_lifecycle.acc_pipeline}.consumer_wait("
        f"{tcgen05_lifecycle.acc_consumer_state})\n"
    )
    if tcgen05_value.use_tma_store_epilogue:
        tma_static_store_setup, tma_tile_store_setup = store_common_setup(
            tcgen05_value.tma_store_tensor,
            include_full_tile=partial_tma_needs_full_tile_guard,
            tma_store=True,
            dynamic_d_tensormap=grouped_dynamic_d_tensormap_all_tiles,
        )
        dynamic_d_tma_tile_store_setup = (
            store_common_setup(
                tcgen05_value.tail_tma_store_tensor,
                include_full_tile=False,
                tma_store=True,
                dynamic_d_tensormap=True,
                rank3_mnl_tensor=True,
            )[1]
            if grouped_dynamic_d_tensormap_edge_only
            else []
        )
    else:
        tma_static_store_setup, tma_tile_store_setup = [], []
        dynamic_d_tma_tile_store_setup = []
    grouped_d_tensormap_atom = (
        tcgen05_value.tail_tma_store_atom
        if grouped_dynamic_d_tensormap_edge_only
        else tcgen05_value.tma_store_atom
    )
    grouped_d_tensormap_shape_expr = (
        f"({grouped_tail_problem_n}, {grouped_tail_d_tensormap_problem_m}, cutlass.Int32(1))"
        if tcgen05_nm_store
        else f"({grouped_tail_d_tensormap_problem_m}, {grouped_tail_problem_n}, cutlass.Int32(1))"
    )
    grouped_d_tensormap_stride_expr = (
        (
            f"({grouped_d_tensormap_stride_n}, {grouped_d_tensormap_stride_m}, "
            "cutlass.Int32(0))"
        )
        if grouped_direct_pointer_metadata and tcgen05_nm_store
        else (
            f"({grouped_d_tensormap_stride_m}, {grouped_d_tensormap_stride_n}, "
            "cutlass.Int32(0))"
        )
        if grouped_direct_pointer_metadata
        else (
            f"({tensor_name}.layout.stride[1], {tensor_name}.layout.stride[0], "
            "cutlass.Int32(0))"
            if tcgen05_nm_store
            else f"({tensor_name}.layout.stride[0], {tensor_name}.layout.stride[1], "
            "cutlass.Int32(0))"
        )
    )
    grouped_d_tensormap_setup: list[str] = []
    grouped_d_tensormap_update: list[str] = []

    def dynamic_d_initializer(source: str) -> str:
        if grouped_full_coverage_predicate is None:
            return source
        return f"if not {grouped_full_coverage_predicate}:\n    {source}"

    if grouped_dynamic_d_tensormap:
        index_dtype = CompileEnvironment.current().index_type()

        def _grouped_direct_metadata_load(
            tensor_name: str,
            *indices: str | int,
        ) -> str:
            offset_terms = [
                f"{index_dtype}({index}) * {index_dtype}("
                f"{tensor_name}.layout.stride[{dim}])"
                for dim, index in enumerate(indices)
            ]
            return f"({tensor_name}.iterator + {' + '.join(offset_terms)}).load()"

        grouped_d_tensormap_base_expr = (
            (
                f"cute.make_ptr({target_dtype}, "
                f"cutlass.Int64({grouped_d_tensormap_addr}), "
                "cute.AddressSpace.gmem)"
            )
            if grouped_direct_pointer_metadata
            else (
                f"{tensor_name}.iterator + "
                f"{index_dtype}({grouped_tail_global_m_start}) * "
                f"{index_dtype}({tensor_name}.layout.stride[0])"
            )
        )
        grouped_d_tensormap_direct_loads = (
            (
                f"    {grouped_d_tensormap_addr} = "
                f"{_grouped_direct_metadata_load(grouped_direct_pointers, grouped_tail_metadata_idx, 2)}\n"
                f"    {grouped_d_tensormap_stride_m} = "
                f"{_grouped_direct_metadata_load(grouped_direct_strides, grouped_tail_metadata_idx, 2, 0)}\n"
                f"    {grouped_d_tensormap_stride_n} = "
                f"{_grouped_direct_metadata_load(grouped_direct_strides, grouped_tail_metadata_idx, 2, 1)}\n"
            )
            if grouped_direct_pointer_metadata
            else ""
        )
        grouped_d_tensormap_setup = [
            (
                f"{grouped_d_tensormap_manager} = "
                "cutlass.utils.TensorMapManager("
                "cutlass.utils.TensorMapUpdateMode.SMEM, 128)"
            ),
            f"{grouped_d_tensormap_grid_dim} = cute.arch.grid_dim()",
            (
                f"{grouped_d_tensormap_workspace_idx} = ("
                f"cute.arch.block_idx()[2] * "
                f"{grouped_d_tensormap_grid_dim}[1] * "
                f"{grouped_d_tensormap_grid_dim}[0] + "
                f"cute.arch.block_idx()[1] * "
                f"{grouped_d_tensormap_grid_dim}[0] + "
                "cute.arch.block_idx()[0])"
            ),
            (
                f"{grouped_d_tensormap_ptr} = "
                f"{grouped_d_tensormap_manager}.get_tensormap_ptr("
                f"{grouped_tail_d_tensormap}"
                f"[({grouped_d_tensormap_workspace_idx}, "
                f"{grouped_tail_d_tensormap_slot}, None)].iterator)"
            ),
            (
                f"{grouped_d_tensormap_desc_ptr} = "
                f"{grouped_d_tensormap_manager}.get_tensormap_ptr("
                f"{grouped_d_tensormap_ptr}, cute.AddressSpace.generic)"
            ),
            (
                f"{grouped_d_tensormap_smem_ptr} = "
                "cute.arch.alloc_smem(cutlass.Int64, cutlass.Int32(16), "
                "alignment=128)"
            ),
            dynamic_d_initializer(
                f"{grouped_d_tensormap_manager}.init_tensormap_from_atom("
                f"{grouped_d_tensormap_atom}, {grouped_d_tensormap_smem_ptr}, "
                "0)"
            ),
            dynamic_d_initializer(
                f"{grouped_d_tensormap_manager}.fence_tensormap_initialization()"
            ),
            f"{grouped_d_tensormap_last_group} = cutlass.Int32(-1)",
        ]
        grouped_d_tensormap_update = [
            (
                f"{grouped_d_tensormap_group_changed} = "
                + (
                    f"not {grouped_full_coverage_predicate} and "
                    if grouped_full_coverage_predicate is not None
                    else ""
                )
                + f"{grouped_tail_metadata_idx} != {grouped_d_tensormap_last_group}"
            ),
            (
                f"if {grouped_d_tensormap_group_changed}:\n"
                f"{grouped_d_tensormap_direct_loads}"
                f"    {grouped_d_tensormap_base} = "
                f"{grouped_d_tensormap_base_expr}\n"
                f"    {grouped_d_tensormap_real_d} = cute.make_tensor("
                f"{grouped_d_tensormap_base}, "
                "cute.make_layout("
                f"{grouped_d_tensormap_shape_expr}, "
                f"stride={grouped_d_tensormap_stride_expr}))\n"
                f"    {grouped_d_tensormap_manager}.update_tensormap("
                f"({grouped_d_tensormap_real_d},), "
                f"({grouped_d_tensormap_atom},), "
                f"({grouped_d_tensormap_ptr},), 0, "
                f"({grouped_d_tensormap_smem_ptr},))\n"
                f"    if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
                f"        {grouped_d_tensormap_manager}.fence_tensormap_update("
                f"{grouped_d_tensormap_ptr})\n"
                f"    {grouped_d_tensormap_last_group} = {grouped_tail_metadata_idx}"
            ),
        ]
    # Role-local TMA stores reuse one C pipeline across work tiles. Static-full
    # kernels increment this counter once per role-local tile; hybrid
    # output-edge kernels increment it only in the full-tile branch so SIMT
    # fallback edge tiles do not perturb the C-pipeline SMEM stage sequence.
    tma_c_buffer_expr = "cutlass.Int32(_tcgen05_subtile)"
    # Each epilogue warp group owns ``c_stages_per_group`` consecutive stages
    # of the ring and cycles them with its own subtile counter (``_tcgen05_local``:
    # the group's subtiles per tile), continuous across tiles. The per-tile
    # stride has to be the number of subtiles the group itself commits:
    # ``PipelineTmaStore.producer_acquire`` only drains the store issued
    # ``c_stages`` commits earlier, which covers a stage's reuse only when
    # consecutive commits cycle the stages. On an even split every group
    # commits ``local_count`` subtiles; on a ragged one (``count % G != 0``)
    # the short groups skip their last local index, so their stride is the
    # group-uniform ``(count - group + G - 1) // G`` (``local_count`` would
    # leave a stage skipped per tile: G 3, two stages, eight subtiles walks
    # 0, 1 | 1, 0 | 0, 1 ... and the reuse of stage 1 across the tile boundary
    # is not covered by the one-deep drain).
    tcgen05_local_count = df.new_var("tcgen05_local_count") if grouped_epilogue else ""
    c_buffer_group_offset = (
        f"{tcgen05_value.epi_group} * cutlass.Int32({tcgen05_c_stage_count}) + "
        if grouped_epilogue
        else ""
    )
    if grouped_epilogue:
        tcgen05_group_owned_count = (
            f"((cutlass.Int32({tcgen05_static_subtile_count + tcgen05_epi_groups - 1}) - "
            f"{tcgen05_value.epi_group}) // cutlass.Int32({tcgen05_epi_groups}))"
            if tcgen05_ragged_groups
            else f"cutlass.Int32({tcgen05_local_subtile_count})"
        )
        tma_c_buffer_expr = (
            f"{tcgen05_value.role_local_tile_counter} * {tcgen05_group_owned_count} + "
            if tcgen05_value.role_local_tile_counter
            else ""
        ) + "cutlass.Int32(_tcgen05_local)"
    elif tcgen05_value.role_local_tile_counter:
        if tcgen05_m_subtile_count > 1:
            # The role-local tile counter advances once per WORK tile; fold
            # the M-subtile index in so the C-ring parity stays continuous
            # across the two drained subtiles regardless of subtile_count
            # parity.
            tma_c_buffer_expr = (
                f"{tcgen05_value.role_local_tile_counter} * "
                f"cutlass.Int32({tcgen05_m_subtile_count}) * "
                f"cutlass.Int32({subtile_count}) + "
                f"cutlass.Int32(_tcgen05_msub) * cutlass.Int32({subtile_count})"
                " + cutlass.Int32(_tcgen05_subtile)"
            )
        else:
            tma_c_buffer_expr = (
                f"{tcgen05_value.role_local_tile_counter} * "
                f"cutlass.Int32({subtile_count}) + cutlass.Int32(_tcgen05_subtile)"
            )
    simt_store_edge_coord_preloaded = simt_edge_only and bool(aux_steps_in_chain)
    staged_store_setup: list[str] = []
    staged_store_tile_setup: list[str] = []
    staged_store_phase = ""
    if staged_store:
        (
            staged_store_setup,
            staged_store_tile_setup,
            simt_store_copy_source,
            staged_store_phase,
        ) = _staged_simt_store_lines(
            df,
            tile=(tcgen05_value.bm, tcgen05_value.bn),
            pad=staged_store_pad,
            extents=(static_m_size, static_n_size),
            tile_coord=(tile_coord_m, tile_coord_n),
            dtype=target_dtype,
            thr_mma=tcgen05_value.thr_mma,
            thr_copy_t2r=thr_copy_t2r,
            epi_tile=epi_tile,
            epi_tidx=tcgen05_value.epi_tidx,
            epi_active=tcgen05_lifecycle.epi_active,
            epi_warps=tcgen05_value.epi_warp_count,
            barrier_id=tcgen05_value.epilog_sync_barrier_id,
            registers=ttr_rd,
            gmem_tile=gmem_tile,
        )
    elif row_union is not None:
        simt_store_copy_source = _simt_edge_coord_subtile_source("        ")
        simt_store_copy_source += textwrap.indent(
            row_union.masked_copy(
                source=ttr_rd,
                destination=ttr_gc_subtile,
                coordinates=ttr_cc_subtile,
                bits=num_bits,
                atom=simt_atom,
            ),
            "        ",
        )
    elif simt_edge_only:
        simt_store_copy_source = _simt_edge_logical_divide_copy_source(
            "        ",
            ttr_rd,
            ttr_gc_subtile,
            include_coord_setup=not simt_store_edge_coord_preloaded,
        )
    elif direct_full_tile_store:
        simt_store_copy_source = (
            f"        cute.copy({simt_atom}, {ttr_rd}, {ttr_gc_subtile})\n"
        )
    elif simt_static_strides is not None and isinstance(static_n_size, int):
        simt_store_copy_source = (
            f"        if {full_tile}:\n"
            f"            cute.copy({simt_atom}, {ttr_rd}, {ttr_gc_subtile})\n"
            f"        else:\n"
            f"{_simt_edge_vector_copy_source('            ', ttr_rd, ttr_gc_subtile, n_size_static=static_n_size)}"
        )
    else:
        simt_store_copy_source = (
            f"        if {full_tile}:\n"
            f"            cute.copy({simt_atom}, {ttr_rd}, {ttr_gc_subtile})\n"
            f"        else:\n"
            f"{_simt_edge_scalar_copy_source('            ', ttr_rd, ttr_gc_subtile)}"
        )
    simt_store_body_core = [
        *simt_static_store_setup,
        *staged_store_setup,
        *simt_tile_store_setup,
        (
            f"{tcgc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgc_base})"
        ),
        (
            f"{tcgc_planned} = cute.make_tensor("
            f"{tcgc}.iterator, "
            f"cute.append(cute.append(cute.append({tcgc}.layout, {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}))"
        ),
        (
            f"{tacc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgen05_value.epi_acc_frag_base})"
        ),
        (
            f"{tiled_copy_t2r}, {ttr_tacc_base}, {ttr_racc} = "
            "cutlass.utils.gemm.sm100.epilogue_tmem_copy_and_partition("
            f"{kernel_desc}, {tcgen05_value.epi_tidx}, {tacc}, {tcgc_planned}, {epi_tile}, {tcgen05_lifecycle.is_two_cta!s})"
        ),
        f"{thr_copy_t2r} = {tiled_copy_t2r}.get_slice({tcgen05_value.epi_tidx})",
        f"{tcgc_epi} = cute.flat_divide({tcgc_planned}, {epi_tile})",
        f"{ttr_gc} = {thr_copy_t2r}.partition_D({tcgc_epi})",
        *staged_store_tile_setup,
        (
            f"{ttr_tacc_stage} = {ttr_tacc_base}["
            f"(None, None, None, None, None, {tcgen05_acc_stage_index_expr})]"
        ),
        # Per-aux-step partitioning lines (one chain per auxiliary
        # tensor) and the per-tile scalar bindings: address math and
        # accumulator-independent reads, issued ahead of the accumulator
        # wait as on the TMA-store path.  No-op when the chain has no aux
        # steps; generated source is byte-identical to the unary-chain
        # shape for unary chains and to the identity-store golden for
        # identity stores.
        *_aux_tile_setup_lines(
            thr_copy_t2r_var=thr_copy_t2r,
            define_thr_copy_t2r=False,
            force_gmem_aux=simt_edge_only,
        ),
        *(
            [
                (
                    f"if {tcgen05_lifecycle.epi_active}:\n"
                    f"    {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})"
                )
            ]
            if not (is_secondary_store or prefetch_simt_aux)
            else []
        ),
        f"{ttr_tacc} = cute.group_modes({ttr_tacc_stage}, 3, cute.rank({ttr_tacc_stage}))",
        f"{ttr_gc_grouped} = cute.group_modes({ttr_gc}, 3, cute.rank({ttr_gc}))",
        (
            f"{ttr_racc} = cute.make_rmem_tensor("
            f"{ttr_gc_grouped}[(None, None, None, 0)].shape, cutlass.Float32)"
        ),
        f"{ttr_rd} = cute.make_rmem_tensor({ttr_racc}.shape, {target_dtype})",
        (
            f"{mcld} = cute.max_common_layout("
            f"{ttr_rd}.layout, {ttr_gc_grouped}[(None, None, None, 0)].layout)"
        ),
        (
            f"{num_bits} = {target_dtype}.width"
            if simt_edge_only
            else (
                f"{num_bits} = min("
                f"min({tensor_name}.iterator.alignment, {direct_store_align_bytes}) * 8, "
                f"cute.size({mcld}) * {target_dtype}.width)"
                if direct_store_align_bytes is not None
                else f"{num_bits} = min("
                f"{ttr_gc_grouped}.iterator.alignment * 8, "
                f"cute.size({mcld}) * {target_dtype}.width, 256)"
            )
        ),
        (
            f"{simt_atom} = cute.make_copy_atom("
            f"cute.nvgpu.CopyR2GOp(), {target_dtype}, "
            f"num_bits_per_copy={num_bits}, "
            f"l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)"
        ),
        *simt_edge_aux_atom_setup,
        f"{subtile_count} = cutlass.const_expr(cute.size({ttr_tacc}.shape, mode=[3]))",
        (
            # Per-subtile loop: TMEM->reg (t2r) first, then reg->GMEM (SIMT
            # store). On the last subtile we release the acc consumer slot
            # *before* the GMEM store so the next mainloop tile's MMA can
            # producer_acquire the TMEM stage and begin issuing UMMAs while
            # this tile's epilogue is still draining to GMEM. This mirrors the
            # release-acc-inside-the-subtile-loop pattern in Quack's sm100
            # gemm epilogue. Without c_pipeline SMEM staging we can only
            # release after the final t2r (not per-subtile), but even one
            # tile of overlap measurably improves the wide tcgen05 path on
            # B200. `cutlass.range(..., unroll_full=True)` keeps the loop
            # statically unrolled so `tiled_copy_t2r` (a TiledCopy that wraps
            # a tcgen05 tmem_load atom) is not captured as an scf.for iter_arg
            # — the cute-to-nvvm pass cannot legalize that conversion through
            # iter_args and aborts during compile.
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"    if {tcgen05_lifecycle.epi_active}:\n"
            f"        {ttr_tacc_mn} = {ttr_tacc}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            f"        {ttr_gc_subtile} = {ttr_gc_grouped}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            + (
                f"        {ttr_gc_subtile} = cute.make_tensor(cute.make_ptr({target_dtype}, "
                f"{ttr_gc_subtile}.iterator.toint(), cute.AddressSpace.gmem, "
                f"assumed_align=min({tensor_name}.iterator.alignment, {direct_store_align_bytes})), "
                f"{ttr_gc_subtile}.layout)\n"
                if direct_store_align_bytes is not None
                else ""
            )
            + (f"{simt_early_aux}{simt_acc_wait}" if prefetch_simt_aux else "")
            + (
                f"        cute.copy({tiled_copy_t2r}, {ttr_tacc_mn}, {ttr_racc})\n"
                if chain_reads_accumulator
                else ""
            )
            + (
                f"{simt_late_prelude}"
                if prefetch_simt_aux
                else f"{simt_acc_vec_prelude}"
            )
            + f"        {acc_vec} = {simt_acc_vec_rhs}\n"
            f"        {ttr_rd}.store({acc_vec})\n"
            # Keep the accumulator live until the final output has read it.
            + (
                ""
                if not is_final_store
                else (
                    f"        if _tcgen05_subtile == {subtile_count} - 1:\n"
                    # `cute.copy(t2r, ...)` issues async TMEM->reg loads.
                    # Releasing the acc consumer slot lets the MMA producer
                    # re-acquire the TMEM stage and issue UMMAs that overwrite
                    # TMEM, so we must fence the in-flight async TMEM loads
                    # first to avoid a race on the last subtile's `ttr_racc` /
                    # `ttr_rd` data. This matches Quack's sm100 gemm
                    # fence-before-release pattern.
                    f"            cute.arch.fence_view_async_tmem_load()\n"
                    f"            with cute.arch.elect_one():\n"
                    f"                {tcgen05_lifecycle.acc_pipeline}.consumer_release({tcgen05_lifecycle.acc_consumer_state})\n"
                )
            )
            + f"{simt_store_copy_source}"
            + staged_store_phase
            # Advance is a per-thread local state update, so it intentionally
            # stays outside elect_one; only the mbarrier release is elected.
            + (
                ""
                if not is_final_store
                else (
                    f"if {tcgen05_lifecycle.epi_active}:\n"
                    + emit_pipeline_advance(
                        tcgen05_lifecycle.acc_consumer_state, indent="    "
                    )
                )
            )
        ),
    ]
    compact_tma_per_tile_setup: list[str] | None = None
    if fragment_epilogue is not None and fragment_epilogue.changes_shape:
        from ..fragment_epilogue import render_tcgen05_fragment_epilogue_group

        destination_bm = fragment_epilogue.destination_shape[-2]
        destination_bn = fragment_epilogue.destination_shape[-1]
        destination_subtile_count = len(fragment_epilogue.programs)
        compact_tma_store = tcgen05_value.use_tma_store_epilogue
        assert leading_index is not None

        # The normal epilogue_tmem_copy_and_partition helper couples the full
        # accumulator partition to a destination derived from transformed
        # thr_mma.partition_C(C). Compact fragments instead partition the
        # destination from a separately proven logical compact tile, whose
        # subtile count can differ from the accumulator.
        def _compact_fragment_partition_setup(
            *,
            before_t2r_setup: list[str],
            before_destination_partition: list[str],
        ) -> list[str]:
            return [
                *before_t2r_setup,
                f"{tacc_epi} = cute.flat_divide({tacc}, {epi_tile})",
                (
                    f"{tiled_copy_t2r} = cute.nvgpu.tcgen05.make_tmem_copy("
                    f"{tcgen05_value.tmem_load_atom}, "
                    f"{tacc_epi}[(None, None, 0, 0, 0)])"
                ),
                f"{thr_copy_t2r} = {tiled_copy_t2r}.get_slice({tcgen05_value.epi_tidx})",
                f"{ttr_tacc_base} = {thr_copy_t2r}.partition_S({tacc_epi})",
                *before_destination_partition,
                f"{compact_ttr_gd} = {thr_copy_t2r}.partition_D({compact_gmem_epi})",
                (
                    f"{compact_ttr_gd_grouped} = cute.group_modes("
                    f"{compact_ttr_gd}, 3, cute.rank({compact_ttr_gd}))"
                ),
                (
                    f"{coord_tile} = cute.local_tile("
                    f"cute.make_identity_tensor(({m_size}, {n_size})), "
                    f"({destination_bm}, {destination_bn}), "
                    f"({tile_coord_m}, {tile_coord_n}))"
                ),
                f"{compact_coord_epi} = cute.flat_divide({coord_tile}, {epi_tile})",
                f"{compact_ttr_coord} = {thr_copy_t2r}.partition_D({compact_coord_epi})",
                (
                    f"{compact_ttr_coord_grouped} = cute.group_modes("
                    f"{compact_ttr_coord}, 3, cute.rank({compact_ttr_coord}))"
                ),
                (
                    f"{ttr_tacc_stage} = {ttr_tacc_base}["
                    f"(None, None, None, None, None, "
                    f"{tcgen05_acc_stage_index_expr})]"
                ),
                (
                    f"if {tcgen05_lifecycle.epi_active}:\n"
                    f"    {tcgen05_lifecycle.acc_pipeline}.consumer_wait("
                    f"{tcgen05_lifecycle.acc_consumer_state})"
                ),
                (
                    f"{ttr_tacc} = cute.group_modes({ttr_tacc_stage}, 3, "
                    f"cute.rank({ttr_tacc_stage}))"
                ),
                (
                    f"{ttr_racc} = cute.make_rmem_tensor("
                    f"{compact_ttr_gd_grouped}["
                    f"(None, None, None, cutlass.Int32(0))].shape, cutlass.Float32)"
                ),
                f"{ttr_rd} = cute.make_rmem_tensor({ttr_racc}.shape, {target_dtype})",
            ]

        # Traverse the committed fragment program exactly once. The program is
        # independent of the drain: SIMT writes registers directly to GMEM,
        # while TMA stages the same destination registers through SMEM.
        scheduled_source = f"if {tcgen05_lifecycle.epi_active}:\n"
        for destination_subtile, destination_program in enumerate(
            fragment_epilogue.programs
        ):
            if compact_tma_store and destination_subtile:
                scheduled_source += (
                    f"    if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
                    f"        {c_pipeline}.producer_acquire()\n"
                )
            if not compact_tma_store:
                scheduled_source += (
                    f"    {compact_gd_subtile} = {compact_ttr_gd_grouped}["
                    f"(None, None, None, cutlass.Int32({destination_subtile}))]\n"
                )
            scheduled_source += (
                f"    {compact_coord_subtile} = {compact_ttr_coord_grouped}["
                f"(None, None, None, cutlass.Int32({destination_subtile}))]\n"
            )
            for program in destination_program.groups:
                scheduled_source += (
                    f"    {ttr_tacc_mn} = {ttr_tacc}["
                    f"(None, None, None, cutlass.Int32({program.source_subtile}))]\n"
                    f"    cute.copy({tiled_copy_t2r}, {ttr_tacc_mn}, {ttr_racc})\n"
                )
                scheduled_source += render_tcgen05_fragment_epilogue_group(
                    state,
                    fragment_epilogue,
                    program,
                    carrier_name=ttr_racc,
                    destination_name=ttr_rd,
                    coordinate_name=compact_coord_subtile,
                    target_dtype=target_dtype,
                    indent="    ",
                )
            if destination_subtile == destination_subtile_count - 1:
                scheduled_source += (
                    "    cute.arch.fence_view_async_tmem_load()\n"
                    "    with cute.arch.elect_one():\n"
                    f"        {tcgen05_lifecycle.acc_pipeline}.consumer_release("
                    f"{tcgen05_lifecycle.acc_consumer_state})\n"
                )
            if compact_tma_store:
                c_buffer_index = (
                    f"{tcgen05_value.role_local_tile_counter} * "
                    f"cutlass.Int32({destination_subtile_count}) + "
                    f"cutlass.Int32({destination_subtile})"
                    if tcgen05_value.role_local_tile_counter
                    else f"cutlass.Int32({destination_subtile})"
                )
                scheduled_source += (
                    f"    {epilog_sync_barrier}.arrive_and_wait()\n"
                    f"    {c_buffer} = ({c_buffer_index}) % "
                    f"cutlass.Int32({tcgen05_value.c_stage_count})\n"
                    f"    cute.copy({tiled_copy_r2s}, {trs_rd}, "
                    f"{trs_sd}[(None, None, None, {c_buffer})])\n"
                    "    cute.arch.fence_view_async_shared()\n"
                    f"    {epilog_sync_barrier}.arrive_and_wait()\n"
                    f"    if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
                    f"        cute.copy({tcgen05_value.tma_store_atom}, "
                    f"{bsg_sd}[(None, {c_buffer})], "
                    f"{bsg_gd}[(None, cutlass.Int32({destination_subtile}))])\n"
                    f"        {c_pipeline}.producer_commit()\n"
                )
            else:
                scheduled_source += (
                    f"    cute.copy({simt_atom}, {ttr_rd}, {compact_gd_subtile})\n"
                )
        scheduled_source += emit_pipeline_advance(
            tcgen05_lifecycle.acc_consumer_state,
            indent="    ",
        )

        if compact_tma_store:
            compact_tma_per_tile_setup = [
                (
                    f"if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
                    f"    {c_pipeline}.producer_acquire()"
                ),
                *tma_tile_store_setup,
                (
                    f"{tcgc} = cutlass.utils.gemm.sm100."
                    f"transform_partitioned_tensor_layout({tcgc_base})"
                ),
                (
                    f"{tcgc_planned} = cute.make_tensor("
                    f"{tcgc}.iterator, cute.append(cute.append(cute.append("
                    f"{tcgc}.layout, {tcgen05_aux_epilogue_rest_mode}), "
                    f"{tcgen05_aux_epilogue_rest_mode}), "
                    f"{tcgen05_aux_epilogue_rest_mode}))"
                ),
                *_compact_fragment_partition_setup(
                    before_t2r_setup=[],
                    before_destination_partition=[
                        f"{compact_gmem_epi} = cute.flat_divide({gmem_tile}, {epi_tile})"
                    ],
                ),
                (
                    f"{tiled_copy_r2s}, {trs_rd}, {trs_sd} = "
                    "cutlass.utils.gemm.sm100.epilogue_smem_copy_and_partition("
                    f"{kernel_desc}, {tiled_copy_t2r}, {ttr_rd}, "
                    f"{tcgen05_aux_epi_tidx}, {smem_d})"
                ),
                f"{tcgc_epi} = cute.flat_divide({tcgc_planned}, {epi_tile})",
                (
                    f"{bsg_sd}, {bsg_gd_partitioned} = "
                    "cute.nvgpu.cpasync.tma_partition("
                    f"{tcgen05_value.tma_store_atom}, 0, cute.make_layout(1), "
                    f"cute.group_modes({smem_d}, 0, 2), "
                    f"cute.group_modes({tcgc_epi}, 0, 2))"
                ),
                (
                    f"{bsg_gd} = {bsg_gd_partitioned}["
                    "(None, None, None, cutlass.Int32(0), "
                    "cutlass.Int32(0), cutlass.Int32(0))]"
                ),
                f"{bsg_gd} = cute.group_modes({bsg_gd}, 1, cute.rank({bsg_gd}))",
                scheduled_source,
            ]
        else:
            simt_store_body_core = [
                *simt_static_store_setup,
                _cute_leading_passthrough_view_2d(
                    compact_gmem_2d,
                    tensor_name,
                    leading_index,
                ),
                (
                    f"{gmem_tile} = cute.local_tile({compact_gmem_2d}, "
                    f"({destination_bm}, {destination_bn}), "
                    f"({tile_coord_m}, {tile_coord_n}))"
                ),
                *_compact_fragment_partition_setup(
                    before_t2r_setup=[
                        f"{compact_gmem_epi} = cute.flat_divide({gmem_tile}, {epi_tile})",
                        (
                            f"{tacc} = cutlass.utils.gemm.sm100."
                            f"transform_partitioned_tensor_layout("
                            f"{tcgen05_value.epi_acc_frag_base})"
                        ),
                    ],
                    before_destination_partition=[],
                ),
                (
                    f"{mcld} = cute.max_common_layout({ttr_rd}.layout, "
                    f"{compact_ttr_gd_grouped}["
                    f"(None, None, None, cutlass.Int32(0))].layout)"
                ),
                (
                    f"{num_bits} = min({compact_ttr_gd_grouped}.iterator.alignment "
                    f"* 8, cute.size({mcld}) * {target_dtype}.width, 256)"
                ),
                (
                    f"{simt_atom} = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), "
                    f"{target_dtype}, num_bits_per_copy={num_bits}, "
                    "l1c_evict_priority="
                    "cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)"
                ),
                scheduled_source,
            ]
    # Workstream A Stage 4 (cycle 93, Path B): C-store producer->consumer edge.
    # Mirrors ``_emit_tcgen05_aux_pipeline_setup``'s SIMT PipelineAsync shape.
    # producer_arrive_count = ``epi_warp_count`` (per-warp: each of the 4 epi
    # warps arrives once via ``elect_one`` after R2S + fence); consumer_arrive
    # _count = 1 (the single store warp); num_stages = ``c_stage_count`` so the
    # store can lag up to ``c_stages`` subtiles behind the epi warps' T2R/R2S.
    # Producer (epi ``producer_commit``) AND consumer (store ``consumer_wait`` /
    # ``consumer_release``) BOTH land in this commit so the ring is never a
    # one-sided handshake that wedges only after wrapping the depth (the
    # cycle-2a partial-handshake lesson).
    c_store_edge_setup = (
        [
            (
                f"{c_store_edge_barriers} = cute.arch.alloc_smem("
                f"cutlass.Int64, cutlass.Int32({tcgen05_value.c_stage_count * 2}))"
            ),
            (
                f"{c_store_edge_producer_group} = cutlass.pipeline.CooperativeGroup("
                f"cutlass.pipeline.Agent.Thread, "
                f"cutlass.Int32({tcgen05_value.epi_warp_count}))"
            ),
            (
                f"{c_store_edge_consumer_group} = cutlass.pipeline.CooperativeGroup("
                "cutlass.pipeline.Agent.Thread, cutlass.Int32(1))"
            ),
            (
                f"{c_store_edge} = cutlass.pipeline.PipelineAsync.create("
                f"num_stages={tcgen05_value.c_stage_count}, "
                f"producer_group={c_store_edge_producer_group}, "
                f"consumer_group={c_store_edge_consumer_group}, "
                f"barrier_storage={c_store_edge_barriers})"
            ),
            (
                f"{c_store_edge_producer_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Producer, "
                f"{tcgen05_value.c_stage_count})"
            ),
            (
                f"{c_store_edge_consumer_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Consumer, "
                f"{tcgen05_value.c_stage_count})"
            ),
            (
                f"{c_store_edge_release_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Consumer, "
                f"{tcgen05_value.c_stage_count})"
            ),
        ]
        if has_store_warp
        else []
    )
    tma_store_pipeline_setup = [
        (
            f"{epilog_sync_barrier} = cutlass.pipeline.NamedBarrier("
            f"barrier_id=cutlass.Int32({tcgen05_value.epilog_sync_barrier_id})"
            + (f" + {tcgen05_value.epi_group}" if grouped_epilogue else "")
            + f", num_threads=cutlass.Int32({tcgen05_value.epi_warp_count * 32}))"
        ),
        *c_store_edge_setup,
        (
            f"{c_pipeline_producer_group} = cutlass.pipeline.CooperativeGroup("
            f"cutlass.pipeline.Agent.Thread, cutlass.Int32({tcgen05_value.epi_warp_count * 32}))"
        ),
        (
            f"{c_pipeline} = cutlass.pipeline.PipelineTmaStore.create("
            f"num_stages={tcgen05_c_pipeline_stage_count}, "
            f"producer_group={c_pipeline_producer_group})"
        ),
    ]
    c_acquire_placement = state.device_function.config.get(
        TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
        TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
    )
    acc_wait_placement = state.device_function.config.get(
        TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
        TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
    )
    c_store_mode = state.device_function.config.get(
        TCGEN05_C_STORE_MODE_CONFIG_KEY,
        TCGEN05_C_STORE_MODE_NORMAL,
    )
    epilogue_layout = state.device_function.config.get(
        TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
        TCGEN05_EPILOGUE_LAYOUT_NORMAL,
    )
    paired_row_protocol = row_union.paired_protocol if row_union is not None else None
    paired_row_late_c = (
        paired_row_protocol is not None and paired_row_protocol.late_c_acquire
    )
    paired_row_issuers = (
        " or ".join(
            f"{tcgen05_value.warp_idx} == cutlass.Int32({warp})"
            for warp in paired_row_protocol.output_issuer_warps
        )
        if paired_row_protocol is not None
        else ""
    )
    fanout_late_c_acquire = (
        c_acquire_placement == TCGEN05_C_ACQUIRE_PLACEMENT_BEFORE_STORE
    )
    if fanout_late_c_acquire and fanout_plan is None:
        raise exc.InvalidConfig("before_store requires a proved shared fanout epilogue")
    diagnose_first_c_acquire_in_loop = (
        c_acquire_placement == TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP
    )
    diagnose_later_c_acquire_before_barrier = (
        c_acquire_placement == TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER
    )
    diagnose_acc_wait_before_subtile_loop = (
        acc_wait_placement == TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP
    )
    diagnose_skip_epilogue_store = (
        c_store_mode == TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE
    )
    diagnose_split_first_t2r = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R
    )
    diagnose_split_acc_t2r_store_tail = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_SPLIT_ACC_T2R_STORE_TAIL
    )
    diagnose_module_helper_acc_t2r = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_ACC_T2R
    )
    diagnose_module_helper_store_tail = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_STORE_TAIL
    )
    diagnose_split_epilogue_layout = (
        diagnose_split_first_t2r
        or diagnose_split_acc_t2r_store_tail
        or diagnose_module_helper_acc_t2r
        or diagnose_module_helper_store_tail
    )
    if has_store_fanout and (
        diagnose_split_epilogue_layout or diagnose_skip_epilogue_store
    ):
        raise exc.BackendUnsupported(
            "cute", "tcgen05 output fanout requires the standard epilogue body"
        )
    # ``cute_mma`` lets the one-tile epilogue free TMEM only when the
    # accumulator reaches a single store through the normal TMA body; every
    # variant below that drops, splits or duplicates the t2r region is
    # excluded there, so the free is emitted exactly once and the post-loop
    # teardown (``render_store_post_loop_lines``) skips its own.
    assert not tcgen05_lifecycle.free_tmem_in_epilogue or not (
        has_store_fanout
        or has_store_warp
        or compact_fragment_store
        or diagnose_skip_epilogue_store
        or diagnose_split_epilogue_layout
    )
    elide_role_local_epi_active = (
        tcgen05_value.use_role_local_epi
        and tcgen05_value.use_tma_store_epilogue
        and not has_store_warp
        and not diagnose_split_epilogue_layout
        and not diagnose_skip_epilogue_store
    )
    if static_subtile_loop and (
        has_store_warp
        or paired_row_protocol is not None
        or use_aux_smem_source
        or aux_epilogue_issued
        or diagnose_split_epilogue_layout
        or diagnose_skip_epilogue_store
        or diagnose_first_c_acquire_in_loop
        or diagnose_later_c_acquire_before_barrier
        or (grouped_epilogue and diagnose_acc_wait_before_subtile_loop)
        or grouped_dynamic_d_tensormap
        or not elide_role_local_epi_active
        or (
            grouped_epilogue
            and any(stage is not None for stage in rowvec_aux_stage_records)
        )
        or bool(worklist_nm_segment_valid_m_bound)
        or simt_edge_only
    ):
        raise exc.InvalidConfig(
            "tcgen05 epilogue warp groups and the auxiliary prefetch ring need "
            "the standard role-local TMA-store subtile loop (no store or producer "
            "warps, staged rows, diagnostic placements or edge tiles)"
        )
    # The subtile loop: CuTe's unrolled dynamic loop, or -- for the ring and
    # the warp groups -- a trace-time loop whose index selects ring buffers;
    # with several groups the body walks ``_tcgen05_subtile = G * local +
    # group`` under a guard where the subtile count does not split evenly.
    if grouped_epilogue:
        tma_store_subtile_loop_header = (
            f"{tcgen05_local_count} = ({subtile_count} + {tcgen05_epi_groups - 1}) // "
            f"{tcgen05_epi_groups}\n"
            f"for _tcgen05_local in cutlass.range_constexpr({tcgen05_local_count}):\n"
            f"    _tcgen05_subtile = cutlass.Int32({tcgen05_epi_groups} * _tcgen05_local) + "
            f"{tcgen05_value.epi_group}\n"
            + (
                f"    if _tcgen05_subtile < cutlass.Int32({subtile_count}):\n"
                if tcgen05_ragged_groups
                else ""
            )
        )
        tma_store_subtile_body_extra_indent = "    " if tcgen05_ragged_groups else ""
    elif static_subtile_loop:
        tma_store_subtile_loop_header = (
            f"for _tcgen05_subtile in cutlass.range_constexpr({subtile_count}):\n"
        )
        tma_store_subtile_body_extra_indent = ""
    else:
        tma_store_subtile_loop_header = f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
        tma_store_subtile_body_extra_indent = ""
    tma_store_subtile_loop_aux_prefix = (
        f"assert cutlass.const_expr({subtile_count} == {tcgen05_static_subtile_count})\n"
        if static_subtile_loop
        else ""
    ) + _aux_ring_pre_loop_source()
    # The non-default split/helper epilogue layouts route the per-subtile tail
    # through helpers that emit ONLY the ``if epi_active`` half under
    # ``has_store_warp`` (and ``module_helper_store_tail`` keeps the OLD
    # two-barrier warp-0 ``c_pipeline`` tail while the main path suppressed the
    # matching acquires) — so the C-store edge would have no consumer and wedge
    # once the ring wraps, or the ``c_pipeline`` commit/acquire counts mismatch.
    # Reject the combination loudly. ``split_first_t2r`` routes through
    # ``tma_store_subtile_body`` and IS handled by the Stage-4 split, so it is
    # intentionally excluded.
    if has_store_warp and (
        diagnose_split_acc_t2r_store_tail
        or diagnose_module_helper_acc_t2r
        or diagnose_module_helper_store_tail
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} does not "
            "support tcgen05_warp_spec_store_warps>0 (the split/helper "
            "epilogue layouts do not emit the store-warp tail half of the "
            "Workstream A Stage 4 split; use the default layout)",
        )
    # Workstream A Stage 4 (cycle 93, Path B): the ``c_pipeline``
    # (PipelineTmaStore) producer lifecycle is per-warp — its
    # ``producer_acquire`` is a ``cp_async_bulk_wait_group`` and its
    # ``producer_commit`` a ``cp_async_bulk_commit_group``, both scoped to
    # the warp that ISSUES the TMA-D bulk copy. So when a store warp owns
    # the TMA-D, the entire ``c_pipeline`` lifecycle (acquire + commit +
    # tail) moves onto the store warp: its ``wait_group`` reuse guard lives
    # in the store-warp tail (after the TMA-D + commit, gating the lagged
    # release), the epi warps' historical store-prefix acquire lines are
    # dropped (the C-ring is gated by the cross-warp C-store edge instead),
    # and ``producer_tail`` (final ``wait_group(0)``) stays on the store warp.
    c_pipeline_owner_predicate = (
        store_warp_predicate
        if has_store_warp
        else paired_row_issuers or epi_leader_predicate
    )
    first_acquire_role_gate = (
        epi_leader_predicate
        if elide_role_local_epi_active
        else f"{tcgen05_lifecycle.epi_active} and {epi_leader_predicate}"
    )
    tma_store_pipeline_tail = (
        f"if {c_pipeline_owner_predicate}:\n    {c_pipeline}.producer_tail()"
    )
    tma_store_first_subtile_acquire = (
        []
        if (
            diagnose_first_c_acquire_in_loop
            or has_store_warp
            or fanout_late_c_acquire
            or paired_row_late_c
        )
        else [(f"if {first_acquire_role_gate}:\n    {c_pipeline}.producer_acquire()")]
    )
    tma_store_loop_first_subtile_acquire = (
        (
            f"        if _tcgen05_subtile == 0 and "
            f"{tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
        if (diagnose_first_c_acquire_in_loop and not has_store_warp)
        else ""
    )
    tma_store_loop_later_subtile_acquire = (
        ""
        if (
            diagnose_later_c_acquire_before_barrier
            or has_store_warp
            or fanout_late_c_acquire
            or paired_row_late_c
        )
        else (
            f"        if cutlass.const_expr({static_subtile_index} != 0) and "
            f"{epi_leader_predicate}:\n"
            f"            {c_pipeline}.producer_acquire()\n"
            if static_subtile_loop
            else f"        if _tcgen05_subtile != 0 and "
            f"{tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
    )
    tma_store_loop_late_later_subtile_acquire = (
        (
            f"        if _tcgen05_subtile != 0 and "
            f"{tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
        if (diagnose_later_c_acquire_before_barrier and not has_store_warp)
        else ""
    )
    if diagnose_split_epilogue_layout:
        if not (
            tcgen05_value.use_role_local_epi and tcgen05_value.use_tma_store_epilogue
        ):
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} "
                "requires the "
                "role-local TMA-store tcgen05 epilogue",
            )
        if not tcgen05_lifecycle.is_two_cta:
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} requires "
                "CtaGroup.TWO",
            )
        # Conservative proxy for the validated static-full CtaGroup.TWO
        # two-or-more-subtile envelope; the exact subtile count is only
        # available after the CUTLASS epilogue partitioning below.
        if tcgen05_value.bn < TCGEN05_TWO_CTA_BLOCK_N:
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} is only "
                f"validated for CtaGroup.TWO block_n >= {TCGEN05_TWO_CTA_BLOCK_N}",
            )
        # The split/helper epilogue layouts emit the per-thread chain into
        # separate ``@cute.jit`` helpers or split source boundaries; the
        # auxiliary-tensor splice site needs per-tile aux setup that is not
        # currently plumbed into those helper signatures. Reject the combination
        # loudly so a user does not silently get a kernel that drops the aux
        # read.
        if (
            diagnose_module_helper_acc_t2r
            or diagnose_module_helper_store_tail
            or diagnose_split_first_t2r
            or diagnose_split_acc_t2r_store_tail
        ) and (
            aux_steps_in_chain
            or (epilogue_chain is not None and epilogue_chain.runtime_scalars)
        ):
            raise exc.BackendUnsupported(
                "cute",
                "auxiliary-tensor or runtime-scalar epilogue (e.g. "
                "`out[tile] = (acc + residual[tile]).to(dtype)`) is "
                f"not plumbed through {TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}="
                f"{epilogue_layout!r}. Drop the layout config to use the "
                "default production layout.",
            )
    tma_store_split_first_subtile_acquire = (
        (
            f"        if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
        if diagnose_first_c_acquire_in_loop
        else ""
    )
    tma_store_pre_loop_acc_wait = (
        [
            (
                f"if {tcgen05_lifecycle.epi_active}:\n"
                f"    {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})"
            )
        ]
        if diagnose_acc_wait_before_subtile_loop and not is_secondary_store
        else []
    )
    tma_store_loop_acc_wait = (
        ""
        if (
            diagnose_acc_wait_before_subtile_loop
            or is_secondary_store
            # Several epilogue warp groups wait ahead of their own loop
            # (``tma_store_grouped_pre_loop_wait``).
            or grouped_epilogue
        )
        else (
            f"        if cutlass.const_expr(_tcgen05_subtile == 0):\n"
            f"            {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})\n"
            if static_subtile_loop
            else f"        if _tcgen05_subtile == 0:\n"
            f"            {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})\n"
        )
    )
    tma_store_grouped_pre_loop_wait = (
        f"{tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})\n"
        if grouped_epilogue and not is_secondary_store
        else ""
    )
    # The final store of a grouped epilogue releases the accumulator stage
    # after the loop: every group's warps arrive once (the consumer group
    # counts ``4G`` warps per CTA), after their last TMEM read.
    tma_store_grouped_post_loop_release = (
        "cute.arch.fence_view_async_tmem_load()\n"
        "with cute.arch.elect_one():\n"
        f"    {tcgen05_lifecycle.acc_pipeline}.consumer_release({tcgen05_lifecycle.acc_consumer_state})\n"
        if grouped_epilogue and is_final_store
        else ""
    )
    tma_store_subtile_loop_prefix = (
        tma_store_subtile_loop_aux_prefix + tma_store_grouped_pre_loop_wait
    )
    tma_store_split_first_acc_wait = (
        ""
        if diagnose_acc_wait_before_subtile_loop
        else (
            f"        {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})\n"
        )
    )
    tma_store_split_tail_later_subtile_acquire = (
        ""
        if diagnose_later_c_acquire_before_barrier
        else (
            f"        if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
    )
    tma_store_split_tail_late_later_subtile_acquire = (
        (
            f"        if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
        if diagnose_later_c_acquire_before_barrier
        else ""
    )
    # Pyrefly does not preserve the non-None tcgen05_value narrowing inside
    # the nested source formatter, so keep local string aliases for attributes
    # read only by that closure.
    tcgen05_epi_active = tcgen05_lifecycle.epi_active
    tcgen05_acc_pipeline = tcgen05_lifecycle.acc_pipeline
    tcgen05_acc_consumer_state = tcgen05_lifecycle.acc_consumer_state
    tcgen05_warp_idx = tcgen05_value.warp_idx
    tcgen05_tma_store_atom = tcgen05_value.tma_store_atom
    grouped_d_tensormap_desc_arg = (
        f", tma_desc_ptr={grouped_d_tensormap_desc_ptr}"
        if grouped_dynamic_d_tensormap
        else ""
    )
    tma_store_desc_arg = (
        grouped_d_tensormap_desc_arg if grouped_dynamic_d_tensormap_all_tiles else ""
    )
    # Locals for the store-warp tail closure (Pyrefly drops the non-None
    # tcgen05_value narrowing inside nested source formatters; see above).
    tcgen05_role_local_tile_counter = tcgen05_value.role_local_tile_counter
    tma_store_d_subtile_expr = "cutlass.Int32(_tcgen05_subtile)"

    def tma_store_acc_t2r_region_body(
        *, acc_wait: str, allow_aux_chain: bool = False
    ) -> str:
        """Return the t2r/math/store-source region.

        The default path renders the aux prelude after the TMEM→register
        copy to keep large residual fragments out of the store prefix. The
        compact M64 and two-CTA M128 scaled-FP8 paths prefetch their small
        scale fragments before the accumulator wait to hide the LDG latency.
        """
        nonlocal fanout_iteration
        assert allow_aux_chain or not aux_steps_in_chain, (
            "split/helper epilogue layouts reject aux-tensor chains at validate "
            "time; use allow_aux_chain=True only for the default TMA store body "
            "that threads the aux LDG through the main T2R body."
        )
        carrier = trs_racc
        store_target = trs_rd
        worklist_nm_identity_store = bool(
            worklist_nm_segment_valid_m_bound
            and (epilogue_chain is None or not epilogue_chain.steps)
        )
        early_aux_prelude, late_prelude, rhs = _splice_acc_vec(
            carrier,
            "            " if worklist_nm_identity_store else "        ",
            safe_direct_aux_with_full_tile=partial_tma_needs_full_tile_guard,
            coord_layout="trs",
        )
        # The final fanout store owns release, after every output's TMEM read.
        # The one-tile epilogue's TMEM free follows the subtile loop
        # (``epilogue_tmem_free_src``), not this release.
        acc_release = (
            ""
            if not is_final_store or grouped_epilogue
            else (
                f"        if cutlass.const_expr(_tcgen05_subtile == {subtile_count} - 1):\n"
                f"            cute.arch.fence_view_async_tmem_load()\n"
                f"            with cute.arch.elect_one():\n"
                f"                {tcgen05_acc_pipeline}.consumer_release({tcgen05_acc_consumer_state})\n"
                if static_subtile_loop
                else f"        if _tcgen05_subtile == {subtile_count} - 1:\n"
                f"            cute.arch.fence_view_async_tmem_load()\n"
                f"            with cute.arch.elect_one():\n"
                f"                {tcgen05_acc_pipeline}.consumer_release({tcgen05_acc_consumer_state})\n"
            )
        )
        pre_wait_aux = (
            tcgen05_aux_load_placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
        )
        if worklist_nm_identity_store:
            # Identity-store means the epilogue chain is empty, so
            # ``_splice_acc_vec`` returns no auxiliary-load prelude by
            # construction.
            if early_aux_prelude:
                raise RuntimeError(
                    "worklist identity store unexpectedly produced an aux prelude"
                )
            # The generated worklist clamps valid_m to source_tile_m.  Almost
            # every scheduled tile therefore needs no padding fixup at all.
            # Keep the existing coordinate-derived predicate only in the
            # uniform tail branch; this removes identity-tensor partitioning,
            # predicate construction, and cute.where from full output tiles.
            full_acc_vec = df.new_var("tcgen05_full_acc_vec")
            tail_acc_vec = df.new_var("tcgen05_tail_acc_vec")
            branch_acc_release = textwrap.indent(acc_release, "    ")
            store_source = (
                f"        if {worklist_nm_segment_valid_m_bound} == "
                f"cutlass.Int32({tcgen05_source_bm}):\n"
                f"            {full_acc_vec} = "
                f"({carrier}.load()).to({target_dtype})\n"
                f"{branch_acc_release}"
                f"            {store_target}.store({full_acc_vec})\n"
                f"        else:\n"
                f"{late_prelude}"
                f"            {tail_acc_vec} = {rhs}\n"
                f"{branch_acc_release}"
                f"            {store_target}.store({tail_acc_vec})\n"
            )
        else:
            store_source = (
                f"{'' if pre_wait_aux else early_aux_prelude}"
                f"{late_prelude}"
                f"        {acc_vec} = {rhs}\n"
                f"{acc_release}"
                f"        {store_target}.store({acc_vec})\n"
            )
        if fanout_plan is not None:
            assert fanout_chain is not None
            assert not worklist_nm_identity_store
            fanout_iteration = FanoutIteration(
                tmem_read=(
                    f"        {ttr_tacc_mn} = {ttr_tacc}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                    f"        cute.copy({tiled_copy_t2r}, {ttr_tacc_mn}, {ttr_racc})\n"
                ),
                auxiliary_loads=early_aux_prelude,
                arithmetic=late_prelude,
                assignment=f"        {acc_vec} = {rhs}\n",
                value_name=acc_vec,
                target_dtype=target_dtype,
                release=acc_release,
                register_store=f"        {store_target}.store({acc_vec})\n",
                chain=fanout_chain,
                # The shared suffix renders this store's chain against its
                # aux-load locals and the per-tile scalar bindings of its
                # setup (``_aux_tile_setup_lines``), as the single-output
                # splice does; without the bindings a rank-0 leaf would be
                # re-read from global memory inside the subtile loop.
                auxiliary_names={rec.expr: rec.aux_loaded for rec in aux_step_records}
                | runtime_scalar_locals,
                pre_wait_aux=pre_wait_aux,
            )
        tmem_read = (
            f"        {ttr_tacc_mn} = {ttr_tacc}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            f"        cute.copy({tiled_copy_t2r}, {ttr_tacc_mn}, {ttr_racc})\n"
            if chain_reads_accumulator
            else ""
        )
        return (
            f"{early_aux_prelude if pre_wait_aux else ''}"
            f"{acc_wait}"
            f"{tmem_read}"
            f"{store_source}"
        )

    def tma_store_tail_region(
        *,
        late_later_subtile_acquire: str,
        tma_desc_arg: str | None = None,
        tma_store_atom: str | None = None,
    ) -> str:
        if tma_desc_arg is None:
            tma_desc_arg = tma_store_desc_arg
        if tma_store_atom is None:
            tma_store_atom = tcgen05_tma_store_atom
        if has_store_warp:
            # Path B epi-warp tail (inside ``if epi_active:``): acquire the
            # C-store edge stage (wait until the store warp released it, i.e.
            # the prior TMA-D reading this physical C-ring slot completed),
            # barrier-1 (intra-epi convergence), R2S, fence, then a C-store-edge
            # PRODUCER commit in place of the second CTA barrier. The TMA-D +
            # ``c_pipeline`` lifecycle move to the store warp's tail
            # (``tma_store_store_warp_tail_region``). The epi warps drop
            # straight into the next subtile's T2R after committing — that is
            # the store/T2R overlap. The producer cooperative group is per-warp
            # (count ``epi_warp_count``), so ``producer_acquire`` is a full-warp
            # wait on every epi warp and ``producer_commit`` arrives once per
            # warp via ``elect_one``.
            return (
                f"        {c_store_edge}.producer_acquire({c_store_edge_producer_state})\n"
                f"        {epilog_sync_barrier}.arrive_and_wait()\n"
                f"        {c_buffer} = ({tma_c_buffer_expr}) % cutlass.Int32({tcgen05_c_stage_count})\n"
                f"        cute.copy({tiled_copy_r2s}, {trs_rd}, {trs_sd}[(None, None, None, {c_buffer})])\n"
                f"        cute.arch.fence_view_async_shared()\n"
                f"        with cute.arch.elect_one():\n"
                f"            {c_store_edge}.producer_commit({c_store_edge_producer_state})\n"
                f"        {c_store_edge_producer_state}.advance()\n"
            )
        if paired_row_protocol is not None:
            assert row_union is not None and row_union.schedule is not None
            issuer_half = f"{tcgen05_warp_idx} // cutlass.Int32(2)"
            return (
                f"        if {paired_row_issuers}:\n"
                f"            {c_pipeline}.producer_acquire()\n"
                f"        {epilog_sync_barrier}.arrive_and_wait()\n"
                f"        {c_buffer} = ({tma_c_buffer_expr}) % cutlass.Int32({tcgen05_c_stage_count})\n"
                f"        cute.copy({tiled_copy_r2s}, {trs_rd}, {trs_sd}[(None, None, None, {c_buffer})])\n"
                f"        cute.arch.fence_view_async_shared()\n"
                f"        {epilog_sync_barrier}.arrive_and_wait()\n"
                f"        if {paired_row_issuers}:\n"
                f"            if ({m_index}) + cutlass.Int32(_tcgen05_subtile) * cutlass.Int32({row_union.schedule.epi_n}) < cutlass.Int32({row_union.m}):\n"
                f"                cute.copy({tma_store_atom}, {bsg_sd}[(None, {issuer_half}), {c_buffer}], {bsg_gd}[(None, {issuer_half}), {tma_store_d_subtile_expr}])\n"
                f"            {c_pipeline}.producer_commit()\n"
            )
        tma_copy_line = (
            f"            cute.copy({tma_store_atom}, "
            f"{bsg_sd}[(None, {c_buffer})], "
            f"{bsg_gd}[(None, {tma_store_d_subtile_expr})]{tma_desc_arg})\n"
        )
        if grouped_full_coverage_predicate is not None:
            assert grouped_dynamic_d_tensormap_all_tiles and tma_desc_arg
            tma_copy_line = (
                f"            if {grouped_full_coverage_predicate}:\n"
                f"                cute.copy({tma_store_atom}, "
                f"{bsg_sd}[(None, {c_buffer})], "
                f"{bsg_gd}[(None, {tma_store_d_subtile_expr})])\n"
                "            else:\n"
                f"    {tma_copy_line}"
            )
        return (
            f"{late_later_subtile_acquire}"
            f"        {epilog_sync_barrier}.arrive_and_wait()\n"
            f"        {c_buffer} = {c_buffer_group_offset}({tma_c_buffer_expr}) % cutlass.Int32({tcgen05_c_stage_count})\n"
            f"        cute.copy({tiled_copy_r2s}, {trs_rd}, {trs_sd}[(None, None, None, {c_buffer})])\n"
            f"        cute.arch.fence_view_async_shared()\n"
            f"        {epilog_sync_barrier}.arrive_and_wait()\n"
            f"        if {epi_leader_predicate}:\n"
            f"{tma_copy_line}"
            f"            {c_pipeline}.producer_commit()\n"
        )

    def tma_store_store_warp_tail_region(
        *, tma_desc_arg: str | None = None, tma_store_atom: str | None = None
    ) -> str:
        if tma_desc_arg is None:
            tma_desc_arg = tma_store_desc_arg
        if tma_store_atom is None:
            tma_store_atom = tcgen05_tma_store_atom
        # Path B store-warp tail (inside ``if store_warp_predicate:``): consume
        # the C-store edge, issue the TMA-D, and recycle the C-ring SMEM stage
        # with a ``c_stages - 1`` lagged release so a stage is only freed for
        # the epi producer AFTER its TMA-D read has provably completed.
        #
        # Ordering (per subtile, ``S`` = ``c_buffer``):
        #  1. ``consumer_wait``: the epi warps' R2S of stage ``S`` has landed.
        #  2. TMA-D ``S`` -> GMEM + ``c_pipeline.producer_commit`` (commit_group).
        #  3. ``c_pipeline.producer_acquire`` = ``cp_async_bulk_wait_group(
        #     c_stages - 1, read=True)``: after committing store i this drains
        #     every store except the ``c_stages - 1`` most recent, i.e. proves
        #     store ``i - (c_stages - 1)`` finished reading its SMEM stage.
        #  4. release that proven-drained stage (the ``release_state``, which
        #     lags the wait ``consumer_state`` by ``c_stages - 1``). Suppressed
        #     for the first ``c_stages - 1`` global subtiles (nothing drained
        #     yet); the trailing stages release naturally as later tiles' global
        #     subtile index advances, and the final unreleased stores drain via
        #     ``c_pipeline.producer_tail`` after the loop.
        lag = tcgen05_c_stage_count - 1
        global_subtile = (
            f"({tcgen05_role_local_tile_counter} * "
            f"cutlass.Int32({subtile_count}) + cutlass.Int32(_tcgen05_subtile))"
            if tcgen05_role_local_tile_counter
            else "cutlass.Int32(_tcgen05_subtile)"
        )
        tma_copy_line = (
            f"        cute.copy({tma_store_atom}, "
            f"{bsg_sd}[(None, {c_buffer})], "
            f"{bsg_gd}[(None, {tma_store_d_subtile_expr})]{tma_desc_arg})\n"
        )
        return (
            f"        {c_store_edge}.consumer_wait({c_store_edge_consumer_state})\n"
            f"        {c_store_edge_consumer_state}.advance()\n"
            f"        {c_buffer} = ({tma_c_buffer_expr}) % cutlass.Int32({tcgen05_c_stage_count})\n"
            f"{tma_copy_line}"
            f"        {c_pipeline}.producer_commit()\n"
            f"        {c_pipeline}.producer_acquire()\n"
            f"        if {global_subtile} >= cutlass.Int32({lag}):\n"
            f"            with cute.arch.elect_one():\n"
            f"                {c_store_edge}.consumer_release({c_store_edge_release_state})\n"
            f"            {c_store_edge_release_state}.advance()\n"
        )

    def tma_store_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        acc_wait: str,
        late_later_subtile_acquire: str,
        tma_desc_arg: str | None = None,
        tma_store_atom: str | None = None,
        gate_epi_active: bool = True,
    ) -> str:
        if tma_desc_arg is None:
            tma_desc_arg = tma_store_desc_arg
        if tma_store_atom is None:
            tma_store_atom = tcgen05_tma_store_atom
        # The aux LDG depends on ``_tcgen05_subtile`` and stays inside
        # the per-subtile T2R body. It intentionally runs after the
        # c_pipeline acquire and TMEM→register copy so the residual/bias
        # fragments are not live through the store-prefix waits.
        t2r_body = tma_store_acc_t2r_region_body(
            acc_wait=acc_wait,
            allow_aux_chain=True,
        )
        if has_store_warp:
            # Path B: the epi warps own T2R/R2S + the C-store producer commit;
            # the store warp (a SEPARATE ``if``, NOT under ``epi_active``) owns
            # the TMA-D + ``c_pipeline`` commit/acquire (its ``cp_async_bulk
            # _wait_group`` reuse guard) + the lagged edge release. The C-ring
            # acquire/commit move WHOLLY onto the store warp (PipelineTmaStore
            # is per-warp commit-group state), so the epi warps never touch
            # ``c_pipeline``; their store-prefix acquire lines are dropped.
            return (
                f"    if {tcgen05_epi_active}:\n"
                f"{t2r_body}"
                f"{tma_store_tail_region(late_later_subtile_acquire='', tma_desc_arg=tma_desc_arg, tma_store_atom=tma_store_atom)}"
                f"    if {store_warp_predicate}:\n"
                f"{tma_store_store_warp_tail_region(tma_desc_arg=tma_desc_arg, tma_store_atom=tma_store_atom)}"
            )
        epi_body = (
            f"{first_subtile_acquire}"
            f"{later_subtile_acquire}"
            f"{t2r_body}"
            f"{tma_store_tail_region(late_later_subtile_acquire=late_later_subtile_acquire, tma_desc_arg=tma_desc_arg, tma_store_atom=tma_store_atom)}"
        )
        if gate_epi_active:
            return f"    if {tcgen05_epi_active}:\n{epi_body}"
        return textwrap.indent(textwrap.dedent(epi_body), "    ")

    def indented_diagnostic_region(source: str) -> str:
        if not source:
            return "            pass\n"
        return "".join(f"    {line}" for line in source.splitlines(keepends=True))

    def tma_store_helper_boundary_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        acc_wait: str,
        late_later_subtile_acquire: str,
    ) -> str:
        acquire_region = f"{first_subtile_acquire}{later_subtile_acquire}"
        acc_region = tma_store_acc_t2r_region_body(acc_wait=acc_wait)
        tail_region = tma_store_tail_region(
            late_later_subtile_acquire=late_later_subtile_acquire
        )
        # These constant-true blocks are diagnostic source boundaries. The
        # generated-code AST round trip preserves them, while emitted comments
        # are not reliable line-info anchors.
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"        if True:\n"
            f"{indented_diagnostic_region(acquire_region)}"
            f"        if True:\n"
            f"{indented_diagnostic_region(acc_region)}"
            f"        if True:\n"
            f"{indented_diagnostic_region(tail_region)}"
        )

    module_acc_t2r_helper_name = (
        df.unique_name("tcgen05_acc_t2r_region")
        if diagnose_module_helper_acc_t2r
        else ""
    )
    module_store_tail_helper_name = (
        df.unique_name("tcgen05_store_tail_region")
        if diagnose_module_helper_store_tail
        else ""
    )

    def tma_store_module_acc_t2r_helper_source(*, acc_wait: str) -> str:
        # Aux-tensor chains are rejected for the diagnostic module-helper
        # layouts (see the ``BackendUnsupported`` raise above), so
        # ``module_early_aux`` is always empty here. Concatenating it
        # with ``module_late_prelude`` preserves the prior flat-prelude
        # source order for unary chains and identity stores in this
        # diagnostic layout.
        module_early_aux, module_late_prelude, rhs = _splice_acc_vec(
            "tcgen05_tRS_rAcc", "    ", bind_runtime_scalars=False
        )
        prelude = module_early_aux + module_late_prelude
        return (
            "@cute.jit\n"
            f"def {module_acc_t2r_helper_name}("
            "_tcgen05_subtile, "
            "tcgen05_acc_pipeline, "
            "tcgen05_acc_consumer_state, "
            "tcgen05_tTR_tAcc, "
            "tcgen05_tiled_copy_t2r, "
            "tcgen05_tTR_rAcc, "
            "tcgen05_tRS_rAcc, "
            "tcgen05_tRS_rD, "
            "tcgen05_subtile_count"
            "):\n"
            f"{acc_wait}"
            "    tcgen05_tTR_tAcc_mn = tcgen05_tTR_tAcc[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            "    cute.copy(tcgen05_tiled_copy_t2r, tcgen05_tTR_tAcc_mn, tcgen05_tTR_rAcc)\n"
            f"{prelude}"
            f"    tcgen05_acc_vec = {rhs}\n"
            "    if _tcgen05_subtile == tcgen05_subtile_count - 1:\n"
            "        cute.arch.fence_view_async_tmem_load()\n"
            "        with cute.arch.elect_one():\n"
            "            tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)\n"
            "    tcgen05_tRS_rD.store(tcgen05_acc_vec)"
        )

    def tma_store_module_acc_t2r_helper_call() -> str:
        return (
            f"        {module_acc_t2r_helper_name}("
            f"_tcgen05_subtile, "
            f"{tcgen05_acc_pipeline}, "
            f"{tcgen05_acc_consumer_state}, "
            f"{ttr_tacc}, "
            f"{tiled_copy_t2r}, "
            f"{ttr_racc}, "
            f"{trs_racc}, "
            f"{trs_rd}, "
            f"{subtile_count})\n"
        )

    def tma_store_module_helper_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        late_later_subtile_acquire: str,
    ) -> str:
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"{first_subtile_acquire}"
            f"{later_subtile_acquire}"
            f"{tma_store_module_acc_t2r_helper_call()}"
            f"{tma_store_tail_region(late_later_subtile_acquire=late_later_subtile_acquire)}"
        )

    def tma_store_module_tail_helper_source(*, late_later_subtile_acquire: str) -> str:
        return (
            "@cute.jit\n"
            f"def {module_store_tail_helper_name}("
            "_tcgen05_subtile, "
            "tcgen05_tma_c_buffer_index, "
            "tcgen05_epilog_sync_barrier, "
            "tcgen05_tiled_copy_r2s, "
            "tcgen05_tRS_rD, "
            "tcgen05_tRS_sD, "
            "tcgen05_tma_store_atom, "
            "tcgen05_bSG_sD, "
            "tcgen05_bSG_gD, "
            "tcgen05_c_pipeline, "
            "tcgen05_warp_idx"
            "):\n"
            f"{late_later_subtile_acquire}"
            "    tcgen05_epilog_sync_barrier.arrive_and_wait()\n"
            f"    tcgen05_c_buffer = tcgen05_tma_c_buffer_index % cutlass.Int32({tcgen05_c_stage_count})\n"
            "    cute.copy(tcgen05_tiled_copy_r2s, tcgen05_tRS_rD, tcgen05_tRS_sD[(None, None, None, tcgen05_c_buffer)])\n"
            "    cute.arch.fence_view_async_shared()\n"
            "    tcgen05_epilog_sync_barrier.arrive_and_wait()\n"
            "    if tcgen05_warp_idx == cutlass.Int32(0):\n"
            "        cute.copy(tcgen05_tma_store_atom, tcgen05_bSG_sD[(None, tcgen05_c_buffer)], tcgen05_bSG_gD[(None, cutlass.Int32(_tcgen05_subtile))])\n"
            "        tcgen05_c_pipeline.producer_commit()"
        )

    def tma_store_module_tail_helper_call() -> str:
        return (
            f"        {module_store_tail_helper_name}("
            f"_tcgen05_subtile, "
            f"{tma_c_buffer_expr}, "
            f"{epilog_sync_barrier}, "
            f"{tiled_copy_r2s}, "
            f"{trs_rd}, "
            f"{trs_sd}, "
            f"{tcgen05_tma_store_atom}, "
            f"{bsg_sd}, "
            f"{bsg_gd}, "
            f"{c_pipeline}, "
            f"{tcgen05_warp_idx})\n"
        )

    def tma_store_module_tail_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        acc_wait: str,
    ) -> str:
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"{first_subtile_acquire}"
            f"{later_subtile_acquire}"
            f"{tma_store_acc_t2r_region_body(acc_wait=acc_wait)}"
            f"{tma_store_module_tail_helper_call()}"
        )

    def _round_subtile_loop() -> str:
        """The TMA-store epilogue in rounds of ``tcgen05_epi_round_subtiles``
        subtiles (``tcgen05_epilogue_rounds``): the plain single-output,
        full-tile, role-local store with the identity epilogue and a tile
        counter; every other shape of the store refuses the knob."""
        rounds = tcgen05_epi_round_subtiles
        refused = (
            diagnose_split_first_t2r
            or diagnose_split_acc_t2r_store_tail
            or diagnose_module_helper_acc_t2r
            or diagnose_module_helper_store_tail
            or diagnose_acc_wait_before_subtile_loop
            or diagnose_first_c_acquire_in_loop
            or diagnose_later_c_acquire_before_barrier
            or grouped_epilogue
            or static_subtile_loop
            or has_store_warp
            or paired_row_protocol is not None
            or fanout_plan is not None
            or bool(aux_steps_in_chain)
            or tcgen05_m_subtile_count > 1
            or grouped_dynamic_d_tensormap_edge_only
            or grouped_full_coverage_predicate is not None
            or bool(worklist_nm_segment_valid_m_bound)
            or not is_final_store
            or is_secondary_store
            or fanout_late_c_acquire
            or paired_row_late_c
            or not tcgen05_value.use_tma_store_epilogue
            or not tcgen05_value.role_local_tile_counter
        )
        if refused:
            raise exc.InvalidConfig(
                f"{TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY}={rounds} needs the plain "
                "single-output role-local TMA-store epilogue (no fan-out, auxiliary "
                "chain, store warp, paired tail, epilogue groups or diagnostic "
                "placement) with a tile counter"
            )
        if tcgen05_value.has_explicit_epilogue_tile:
            assert tcgen05_value.explicit_epi_tile_m is not None
            assert tcgen05_value.explicit_epi_tile_n is not None
            round_epi_tile_m = tcgen05_value.explicit_epi_tile_m
            round_epi_tile_n = tcgen05_value.explicit_epi_tile_n
        else:
            round_epi_tile_m, round_epi_tile_n = tcgen05_default_epilogue_tile_size(
                tcgen05_value.bm,
                tcgen05_value.bn,
                elem_width_d=epi_tile_bits,
                elem_width_c=epi_tile_bits,
            )
        round_rows = tcgen05_value.bm // 2 if tcgen05_is_two_cta else tcgen05_value.bm
        if round_rows % round_epi_tile_m or tcgen05_value.bn % round_epi_tile_n:
            raise exc.InvalidConfig(
                f"{TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY}={rounds} needs whole "
                "epilogue subtiles of the CTA tile"
            )
        round_subtile_count = (round_rows // round_epi_tile_m) * (
            tcgen05_value.bn // round_epi_tile_n
        )
        if rounds > round_subtile_count:
            raise exc.InvalidConfig(
                f"{TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY}={rounds} exceeds the "
                f"{round_subtile_count} epilogue subtiles of the tile"
            )
        early_aux_prelude, late_prelude, rhs = _splice_acc_vec(
            trs_racc,
            "        ",
            safe_direct_aux_with_full_tile=partial_tma_needs_full_tile_guard,
            coord_layout="trs",
        )
        if early_aux_prelude or late_prelude:
            raise exc.InvalidConfig(
                f"{TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY}={rounds} needs the "
                "identity epilogue (the accumulator cast to the output dtype)"
            )
        assert tcgen05_value.role_local_tile_counter is not None
        return render_round_subtile_loop(
            Tcgen05EpilogueRoundNames(
                epi_active=tcgen05_epi_active,
                epi_leader_predicate=epi_leader_predicate,
                acc_pipeline=tcgen05_acc_pipeline,
                acc_consumer_state=tcgen05_acc_consumer_state,
                tiled_copy_t2r=tiled_copy_t2r,
                tiled_copy_r2s=tiled_copy_r2s,
                ttr_tacc=ttr_tacc,
                ttr_tacc_mn=ttr_tacc_mn,
                ttr_racc=ttr_racc,
                trs_racc=trs_racc,
                ttr_rd=ttr_rd,
                trs_rd=trs_rd,
                trs_sd=trs_sd,
                acc_vec=acc_vec,
                c_buffer=c_buffer,
                epilog_sync_barrier=epilog_sync_barrier,
                tma_store_atom=tcgen05_tma_store_atom,
                bsg_sd=bsg_sd,
                bsg_gd=bsg_gd,
                c_pipeline=c_pipeline,
                target_dtype=target_dtype,
                tile_counter=tcgen05_value.role_local_tile_counter,
                tma_desc_arg=tma_store_desc_arg,
            ),
            subtile_count=round_subtile_count,
            round_subtiles=rounds,
            c_stage_count=tcgen05_c_stage_count,
            acc_rhs=rhs,
            acc_wait=True,
            acc_release=True,
        )

    def default_tma_store_subtile_loop(
        tma_desc_arg: str | None = None,
        tma_store_atom: str | None = None,
    ) -> str:
        tma_store_default_subtile_body = tma_store_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            acc_wait=tma_store_loop_acc_wait,
            late_later_subtile_acquire=tma_store_loop_late_later_subtile_acquire,
            tma_desc_arg=tma_desc_arg,
            tma_store_atom=tma_store_atom,
            gate_epi_active=not elide_role_local_epi_active,
        )
        return (
            f"{tma_store_subtile_loop_header}"
            f"{textwrap.indent(tma_store_default_subtile_body, tma_store_subtile_body_extra_indent)}"
        )

    if fragment_epilogue is not None and fragment_epilogue.changes_shape:
        # The compact SIMT schedule above owns T2R and never instantiates the
        # same-shape TMA renderer. Avoid eagerly formatting that unused body.
        tma_store_subtile_loop = ""
    elif tcgen05_epi_round_subtiles > 1:
        tma_store_subtile_loop = _round_subtile_loop()
    elif diagnose_split_first_t2r:
        tma_store_split_first_subtile_body = tma_store_subtile_body(
            first_subtile_acquire=tma_store_split_first_subtile_acquire,
            later_subtile_acquire="",
            acc_wait=tma_store_split_first_acc_wait,
            late_later_subtile_acquire="",
        )
        tma_store_split_tail_subtile_body = tma_store_subtile_body(
            first_subtile_acquire="",
            later_subtile_acquire=tma_store_split_tail_later_subtile_acquire,
            acc_wait="",
            late_later_subtile_acquire=(
                tma_store_split_tail_late_later_subtile_acquire
            ),
        )
        # Diagnostic-only scaffolding: reuse the one-indent subtile formatter
        # for a static first subtile without changing production source layout.
        # The tail loop maps split-loop indices back to logical subtile ids 1..N-1;
        # unroll_full=True keeps those subtile values compile-time constants.
        tma_store_subtile_loop = (
            "if True:\n"
            f"    _tcgen05_subtile = 0\n"
            f"{tma_store_split_first_subtile_body}"
            f"for _tcgen05_split_subtile in cutlass.range({subtile_count} - 1, unroll_full=True):\n"
            f"    _tcgen05_subtile = _tcgen05_split_subtile + 1\n"
            f"{tma_store_split_tail_subtile_body}"
        )
    elif diagnose_split_acc_t2r_store_tail:
        tma_store_helper_boundary_body = tma_store_helper_boundary_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            acc_wait=tma_store_loop_acc_wait,
            late_later_subtile_acquire=tma_store_loop_late_later_subtile_acquire,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_helper_boundary_body}"
        )
    elif diagnose_module_helper_acc_t2r:
        module_helper_acc_wait = (
            ""
            if diagnose_acc_wait_before_subtile_loop
            else (
                "    if _tcgen05_subtile == 0:\n"
                "        tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)\n"
            )
        )
        state.codegen.module_statements.append(
            statement_from_string(
                tma_store_module_acc_t2r_helper_source(acc_wait=module_helper_acc_wait)
            )
        )
        tma_store_module_helper_body = tma_store_module_helper_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            late_later_subtile_acquire=tma_store_loop_late_later_subtile_acquire,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_module_helper_body}"
        )
    elif diagnose_module_helper_store_tail:
        module_tail_late_later_subtile_acquire = (
            (
                "    if _tcgen05_subtile != 0 and "
                "tcgen05_warp_idx == cutlass.Int32(0):\n"
                "        tcgen05_c_pipeline.producer_acquire()\n"
            )
            if diagnose_later_c_acquire_before_barrier
            else ""
        )
        state.codegen.module_statements.append(
            statement_from_string(
                tma_store_module_tail_helper_source(
                    late_later_subtile_acquire=module_tail_late_later_subtile_acquire
                )
            )
        )
        tma_store_module_tail_body = tma_store_module_tail_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            acc_wait=tma_store_loop_acc_wait,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_module_tail_body}"
        )
    else:
        tma_store_subtile_loop = default_tma_store_subtile_loop()
    dynamic_d_tma_store_subtile_loop = (
        default_tma_store_subtile_loop(
            grouped_d_tensormap_desc_arg,
            tcgen05_value.tail_tma_store_atom,
        )
        if grouped_dynamic_d_tensormap_edge_only
        else ""
    )
    tma_store_smem_setup = [
        # Must match the wrapper-side `tcgen05_d_tma` TMA atom layout in
        # `helion/runtime/cute/host_wrappers/tcgen05.py`; both describe one D
        # SMEM stage.
        (
            f"{smem_d_layout} = cutlass.utils.blackwell_helpers.make_smem_layout_epi("
            f"{target_dtype}, {tcgen05_d_store_layout}, "
            f"{epi_tile}, {tcgen05_value.c_stage_count})"
        ),
        (
            f"{smem_d_ptr} = cute.arch.alloc_smem("
            f"{target_dtype}, cute.cosize({smem_d_layout}.outer), alignment=1024)"
        ),
        (
            f"{smem_d} = cute.make_tensor("
            f"cute.recast_ptr({smem_d_ptr}, {smem_d_layout}.inner, dtype={target_dtype}), "
            f"{smem_d_layout}.outer)"
        ),
        *_rowvec_aux_smem_setup_lines(),
    ]
    tma_store_acc_layout_setup = [
        (
            f"{tacc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgen05_value.epi_acc_frag_base})"
        ),
    ]
    tma_store_role_invariant_setup = [
        *tma_static_store_setup,
        *tma_store_smem_setup,
        *tma_store_acc_layout_setup,
    ]
    suppressed_store_body_core = [
        (
            # Diagnostic-only invalid-output mode. Keep the accumulator
            # pipeline draining so persistent kernels do not deadlock, but
            # suppress C-pipeline acquire/commit, R2S/SMEM work, and TMA D
            # stores to bound whether hot waits are tied to the C-store path.
            f"if {tcgen05_lifecycle.epi_active}:\n"
            f"    {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})\n"
            f"    with cute.arch.elect_one():\n"
            f"        {tcgen05_lifecycle.acc_pipeline}.consumer_release({tcgen05_lifecycle.acc_consumer_state})\n"
            + emit_pipeline_advance(
                tcgen05_lifecycle.acc_consumer_state,
                indent="    ",
            )
        )
    ]
    # C-input warp aux pipeline consumer-wait + mode-specific
    # consumer-release framing (``cute_plan.md`` §7.5.3.2 cycle 2b).
    # Gate-closed configs (default ``c_input_warps=0`` or no aux
    # residual) keep the historical GMEM aux path. When the gate
    # fires, the wait/release pair runs once per *subtile* of the
    # per-output-tile aux region: per-subtile staging keeps the
    # SMEM ring footprint at one ``epi_tile`` chunk per stage
    # rather than one ``(bm, bn)`` chunk, which is essential to
    # fit cluster_m=2 + ``tcgen05_ab_stages=3`` in the 228 KB
    # B200 SMEM cap. The wait begins the aux-load block emitted by
    # ``_aux_subtile_load_source`` (before any ``.load()`` from the
    # SMEM ring); the default TMA-store path now splices that block
    # after the c_pipeline acquire and T2R copy to keep aux fragments
    # out of the store-prefix live range. The release + ``advance``
    # follow a CTA-scoped read-completion fence, before the chain consumes
    # ``aux_loaded``. SIMT releases from every reader with a per-thread
    # arrival count; TMA retains elected releases and its per-warp count.

    # Static-full role-local stores have no dynamic full-tile branch, so all
    # C-store invariant setup can be hoisted once. Scheduler-backed hybrid
    # output-edge stores split full and fringe tiles into separate role-local
    # scheduler phases, which gives the full-tile phase the same hoist shape.
    # The monolithic hybrid path still keeps descriptor/SMEM layout setup
    # inside its dynamic full-tile branch.
    split_hybrid_tma_store_role = (
        tcgen05_value.use_role_local_epi
        and tcgen05_value.use_tma_store_epilogue
        and tcgen05_value.tma_store_full_tiles_only
        and aux_matmul_plan is not None
        and aux_matmul_plan.has_scheduler_warp
        # CLC publishes a single hardware-scheduled stream today. The
        # full/edge split below requires the scheduler warp to publish two
        # static streams with a sentinel between them.
        and not aux_matmul_plan.is_clc_persistent
        and not diagnose_skip_epilogue_store
    )
    hoist_tma_store_resources = (
        tcgen05_value.use_role_local_epi
        and tcgen05_value.use_tma_store_epilogue
        and (not tcgen05_value.tma_store_full_tiles_only or split_hybrid_tma_store_role)
        and not diagnose_skip_epilogue_store
    )
    hoist_hybrid_tma_store_pipeline = (
        tcgen05_value.use_role_local_epi
        and tcgen05_value.use_tma_store_epilogue
        and tcgen05_value.tma_store_full_tiles_only
        and not split_hybrid_tma_store_role
        and not diagnose_skip_epilogue_store
    )

    def worklist_nm_explicit_store_wave_setup_lines() -> list[str]:
        # The transposed stmatrix atom packs pairs of 16-bit output elements.
        # FP32 identity outputs retain the TMEM partition and copy its original
        # scalar coordinates to the transposed shared-memory destination.
        copy_op = (
            "cute.nvgpu.CopyUniversalOp()"
            if tensor.dtype is torch.float32
            else "cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4)"
        )
        return [
            f"{tacc_epi} = cute.flat_divide({tacc}, {epi_tile})",
            (
                f"{tiled_copy_t2r} = cute.nvgpu.tcgen05.make_tmem_copy("
                f"{tcgen05_aux_tmem_load_atom}, "
                f"{tacc_epi}[(None, None, 0, 0, 0)])"
            ),
            f"{thr_copy_t2r} = {tiled_copy_t2r}.get_slice({tcgen05_aux_epi_tidx})",
            f"{ttr_tacc_base} = {thr_copy_t2r}.partition_S({tacc_epi})",
            f"{tcgc_epi} = cute.flat_divide({tcgc_planned}, {epi_tile})",
            f"{ttr_gc} = {thr_copy_t2r}.partition_D({tcgc_epi})",
            (
                f"{ttr_racc} = cute.make_rmem_tensor("
                f"{ttr_gc}[(None, None, None, 0, 0, 0, 0, 0)].shape, "
                "cutlass.Float32)"
            ),
            f"{ttr_rd} = cute.make_rmem_tensor({ttr_racc}.shape, {target_dtype})",
            (f"{copy_atom_r2s} = cute.make_copy_atom({copy_op}, {target_dtype})"),
            (
                f"{tiled_copy_r2s} = cute.make_tiled_copy_D("
                f"{copy_atom_r2s}, {tiled_copy_t2r})"
            ),
            f"{thr_copy_r2s} = {tiled_copy_r2s}.get_slice({tcgen05_aux_epi_tidx})",
            f"{trs_sd} = {thr_copy_r2s}.partition_D({smem_d})",
            f"{trs_rd} = {tiled_copy_r2s}.retile({ttr_rd})",
            f"{trs_racc} = {tiled_copy_r2s}.retile({ttr_racc})",
        ]

    def default_store_wave_setup_lines() -> list[str]:
        return [
            (
                f"{tiled_copy_t2r}, {ttr_tacc_base}, {ttr_racc} = "
                "cutlass.utils.gemm.sm100.epilogue_tmem_copy_and_partition("
                f"{kernel_desc}, {tcgen05_aux_epi_tidx}, {tacc}, "
                f"{tcgc_planned}, {epi_tile}, {tcgen05_lifecycle.is_two_cta!s})"
            ),
            *(
                [f"{thr_copy_t2r} = {tiled_copy_t2r}.get_slice({tcgen05_aux_epi_tidx})"]
                if fragment_epilogue is not None
                else []
            ),
            f"{ttr_rd} = cute.make_rmem_tensor({ttr_racc}.shape, {target_dtype})",
            (
                f"{tiled_copy_r2s}, {trs_rd}, {trs_sd} = "
                "cutlass.utils.gemm.sm100.epilogue_smem_copy_and_partition("
                f"{kernel_desc}, {tiled_copy_t2r}, {ttr_rd}, "
                f"{tcgen05_aux_epi_tidx}, {smem_d})"
            ),
            f"{trs_racc} = {tiled_copy_r2s}.retile({ttr_racc})",
            f"{tcgc_epi} = cute.flat_divide({tcgc_planned}, {epi_tile})",
        ]

    def build_tma_store_body_setup_core(
        *,
        tma_store_atom: str,
        tile_store_setup: list[str],
        dynamic_d_setup: list[str],
        dynamic_d_update: list[str],
        include_acquire: bool = True,
    ) -> list[str]:
        return [
            *(tma_static_store_setup if not hoist_tma_store_resources else []),
            *(
                tma_store_pipeline_setup
                if not (hoist_tma_store_resources or hoist_hybrid_tma_store_pipeline)
                else []
            ),
            *(tma_store_smem_setup if not hoist_tma_store_resources else []),
            *_rowvec_aux_copy_lines(),
            *(tma_store_first_subtile_acquire if include_acquire else []),
            *dynamic_d_setup,
            *dynamic_d_update,
            *tile_store_setup,
            (
                f"{tcgc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
                f"{tcgc_base})"
            ),
            (
                f"{tcgc_planned} = cute.make_tensor("
                f"{tcgc}.iterator, "
                f"cute.append(cute.append(cute.append({tcgc}.layout, {tcgen05_aux_epilogue_rest_mode}), {tcgen05_aux_epilogue_rest_mode}), {tcgen05_aux_epilogue_rest_mode}))"
            ),
            *(tma_store_acc_layout_setup if not hoist_tma_store_resources else []),
            *(
                worklist_nm_explicit_store_wave_setup_lines()
                if tcgen05_nm_store and row_union is None
                else default_store_wave_setup_lines()
            ),
            # Per-aux-step partitioning lines (one chain per auxiliary tensor).
            # No-op when the chain has no aux steps; the TMA path requires an
            # explicit ``thr_copy_t2r`` slice because it consumes the partition
            # through SMEM-staged stores rather than via partition_D.
            *_aux_tile_setup_lines(
                thr_copy_t2r_var=thr_copy_t2r,
                define_thr_copy_t2r=not tcgen05_nm_store or row_union is not None,
                retile_for_r2s=True,
            ),
            (
                f"{bsg_sd}, {bsg_gd_partitioned} = cute.nvgpu.cpasync.tma_partition("
                f"{tma_store_atom}, 0, cute.make_layout(1), "
                f"cute.group_modes({smem_d}, 0, 2), "
                f"cute.group_modes({tcgc_epi}, 0, 2))"
            ),
            (
                f"{bsg_gd} = {bsg_gd_partitioned}["
                f"(None, None, None, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0))]"
            ),
            f"{bsg_gd} = cute.group_modes({bsg_gd}, 1, cute.rank({bsg_gd}))",
            (
                f"{ttr_tacc_stage} = {ttr_tacc_base}["
                f"(None, None, None, None, None, {tcgen05_acc_stage_index_expr})]"
            ),
            f"{ttr_tacc} = cute.group_modes({ttr_tacc_stage}, 3, cute.rank({ttr_tacc_stage}))",
            f"{subtile_count} = cutlass.const_expr(cute.size({ttr_tacc}.shape, mode=[3]))",
            # Epilogue-issued aux TMA: partition the tile's staged operands and
            # prefetch the first ring stages ahead of the accumulator wait.
            *_epi_tma_aux_tile_lines(),
            *tma_store_pre_loop_acc_wait,
        ]

    tma_store_body_setup_core = build_tma_store_body_setup_core(
        tma_store_atom=tcgen05_value.tma_store_atom,
        tile_store_setup=tma_tile_store_setup,
        dynamic_d_setup=[],
        dynamic_d_update=(
            [] if grouped_dynamic_d_tensormap_edge_only else grouped_d_tensormap_update
        ),
    )
    dynamic_d_tma_store_body_setup_core = (
        build_tma_store_body_setup_core(
            tma_store_atom=tcgen05_value.tail_tma_store_atom,
            tile_store_setup=dynamic_d_tma_tile_store_setup,
            dynamic_d_setup=grouped_d_tensormap_setup,
            dynamic_d_update=grouped_d_tensormap_update,
        )
        if grouped_dynamic_d_tensormap_edge_only
        else []
    )
    # Warp 0 pre-acquires the first TMA-store SMEM stage before per-tile
    # C-store setup. The subtile loop acquires only later stages, so C-stage
    # waits can overlap setup, the first acc-pipeline wait, and the other epi
    # warps' TMEM load/conversion work on later subtile iterations. Most
    # alternate placements are diagnostics, but the edge+K-tail production seed
    # uses the measured first_in_loop / before_subtile_loop pair.
    # tcgen05_c_acquire_placement=first_in_loop moves only that first acquire
    # into the subtile loop; later acquires and the accumulator wait keep their
    # default order. The diagnostic later_before_barrier placement keeps the
    # first acquire in production position and moves only later-subtile
    # acquires just before the first epilogue barrier.
    # tcgen05_acc_wait_placement=before_subtile_loop keeps both C acquire sites
    # in production position and moves only the accumulator consumer wait
    # before the subtile loop. A CTA-scoped named barrier ensures all epi warps
    # have observed warp 0's acquire before they write SMEM; a second barrier
    # ensures the SMEM writes and Quack-style async-shared fence are visible
    # before warp 0 issues and commits the TMA operation. Compute the SMEM ring
    # index after the first barrier so the acquire/barrier/index order stays
    # aligned with Quack's TMA-store epilogue.
    # The accumulator consumer state advances after the loop, matching Quack's
    # call-site ordering while preserving the early release. After warp 0
    # commits the TMA store, the next subtile's producer_acquire plus the first
    # named barrier are enough to keep all epi warps from writing a reused SMEM
    # stage too early. Avoiding a post-commit barrier matches Quack's epilogue
    # loop. The split_first_t2r diagnostic emits the first static subtile as a
    # standalone source block, then loops over later subtile work. It is a
    # layout discriminator for the hot acc-wait/T2R SASS row; the default
    # production source shape remains the single loop.
    # Advance is a per-thread local state update, so it intentionally stays
    # outside elect_one; only the mbarrier release is elected.
    tma_store_pipeline_tail_lines = (
        [tma_store_pipeline_tail]
        if not (hoist_tma_store_resources or hoist_hybrid_tma_store_pipeline)
        else []
    )
    dynamic_d_tma_store_body_core: list[str] = []
    # The final output store advances the shared accumulator transaction.
    tma_store_acc_advance = (
        ""
        if not is_final_store
        else (
            f"if {tcgen05_lifecycle.epi_active}:\n"
            + emit_pipeline_advance(
                tcgen05_lifecycle.acc_consumer_state,
                indent="    ",
            )
        )
    )
    fanout_advance = tma_store_acc_advance
    # One tile per CTA and a single store (asserted above): free TMEM once
    # the last subtile's TMA store is issued, so the dealloc handshake with
    # the MMA warp (the ``tmem_alloc_barrier`` meet; the MMA warp arrives
    # after its last commit, ahead of its own ``producer_tail``) overlaps
    # the store drain instead of holding the last subtile's SMEM staging
    # behind it. Measured under flushed graph
    # replay (device time): bmm 8x256x256x512 one-CTA 4.58 -> 4.45 us,
    # fp8 1024^3 one-CTA 128x64x128 4.96 -> 4.83 us; freeing before the
    # staging (the previous placement) or issuing the next subtile's TMEM
    # load early measured neutral.
    epilogue_tmem_free_src = (
        ""
        if not (tcgen05_lifecycle.free_tmem_in_epilogue and is_final_store)
        else (
            f"if {tcgen05_lifecycle.epi_active}:\n"
            + "".join(
                f"    {line}\n"
                for line in tcgen05_lifecycle.render_epilogue_tmem_free_lines()
            )
        )
    )

    def with_epilogue_tmem_free(loop_src: str) -> str:
        if not epilogue_tmem_free_src:
            return loop_src
        return loop_src.rstrip("\n") + "\n" + epilogue_tmem_free_src

    tma_store_body_core = [
        *tma_store_body_setup_core,
        with_epilogue_tmem_free(
            tma_store_subtile_loop_prefix
            + tma_store_subtile_loop
            + tma_store_grouped_post_loop_release
            + tma_store_acc_advance
        ),
        *tma_store_pipeline_tail_lines,
    ]
    if grouped_dynamic_d_tensormap_edge_only:
        dynamic_d_tma_store_body_core = [
            *dynamic_d_tma_store_body_setup_core,
            with_epilogue_tmem_free(
                dynamic_d_tma_store_subtile_loop + tma_store_acc_advance
            ),
            *tma_store_pipeline_tail_lines,
        ]
    if compact_fragment_store and tcgen05_value.use_tma_store_epilogue:
        assert compact_tma_per_tile_setup is not None
        tma_store_body_core = [
            *(tma_static_store_setup if not hoist_tma_store_resources else []),
            *(tma_store_pipeline_setup if not hoist_tma_store_resources else []),
            *(tma_store_smem_setup if not hoist_tma_store_resources else []),
            *(tma_store_acc_layout_setup if not hoist_tma_store_resources else []),
            *compact_tma_per_tile_setup,
            *tma_store_pipeline_tail_lines,
        ]
    tma_store_full_tile_body_core = list(tma_store_body_core)
    if (
        tcgen05_value.tma_store_full_tiles_only
        and tcgen05_value.role_local_tile_counter
    ):
        tma_store_full_tile_body_core.append(
            f"{tcgen05_value.role_local_tile_counter} = "
            f"{tcgen05_value.role_local_tile_counter} + cutlass.Int32(1)"
        )
        if grouped_dynamic_d_tensormap_edge_only:
            dynamic_d_tma_store_body_core.append(
                f"{tcgen05_value.role_local_tile_counter} = "
                f"{tcgen05_value.role_local_tile_counter} + cutlass.Int32(1)"
            )
    tma_store_body_source = "\n".join(tma_store_full_tile_body_core)
    simt_store_body_source = "\n".join(simt_store_body_core)
    edge_store_body_source = (
        "\n".join(dynamic_d_tma_store_body_core)
        if grouped_dynamic_d_tensormap_edge_only and dynamic_d_tma_store_body_core
        else simt_store_body_source
    )
    hybrid_tma_store_body_core = [
        f"{full_tile} = {full_tile_expr}",
        (
            f"if {full_tile}:\n"
            f"{textwrap.indent(tma_store_body_source, '    ')}\n"
            "else:\n"
            f"{textwrap.indent(edge_store_body_source, '    ')}"
        ),
    ]
    if diagnose_skip_epilogue_store:
        store_body_core = suppressed_store_body_core
    elif tcgen05_value.tma_store_full_tiles_only:
        store_body_core = hybrid_tma_store_body_core
    elif tcgen05_value.use_tma_store_epilogue:
        store_body_core = tma_store_body_core
    else:
        store_body_core = simt_store_body_core
    main_stmts: list[ast.AST]
    main_stmt: ast.AST | None = None
    if tcgen05_value.use_role_local_epi:
        # These setup statements intentionally remain virtual-pid-independent.
        # The persistent splitter hoists pipeline state before the role-local
        # scheduler loops. Scheduler-backed hybrid stores keep descriptor and
        # layout Python objects inside the epilogue role prelude so they do
        # not leak across unrelated dynamic warp-role ``if`` regions.
        tma_store_pipeline_hoisted_stmts = (
            [statement_from_string(line) for line in tma_store_pipeline_setup]
            if (hoist_tma_store_resources or hoist_hybrid_tma_store_pipeline)
            else []
        )
        grouped_d_tensormap_hoisted_stmts = (
            [statement_from_string(line) for line in grouped_d_tensormap_setup]
            if grouped_d_tensormap_setup
            and (
                hoist_tma_store_resources
                or tcgen05_value.orientation is Tcgen05Orientation.NM
            )
            else []
        )
        tma_store_role_invariant_stmts = (
            [statement_from_string(line) for line in tma_store_role_invariant_setup]
            if hoist_tma_store_resources
            else []
        )
        if split_hybrid_tma_store_role:
            tma_store_hoisted_stmts = tma_store_pipeline_hoisted_stmts
        elif hoist_tma_store_resources or hoist_hybrid_tma_store_pipeline:
            tma_store_hoisted_stmts = [
                *tma_store_pipeline_hoisted_stmts,
                *grouped_d_tensormap_hoisted_stmts,
                *tma_store_role_invariant_stmts,
            ]
        else:
            tma_store_hoisted_stmts = grouped_d_tensormap_hoisted_stmts
        if split_hybrid_tma_store_role:
            sync_before_stmt = statement_from_string("cute.arch.sync_threads()")
            sync_after_stmt = statement_from_string("cute.arch.sync_threads()")
            full_main_stmt = statement_from_string(
                "if True:\n"
                + textwrap.indent("\n".join(tma_store_full_tile_body_core), "    ")
            )
            edge_main_stmt = statement_from_string(
                "if True:\n" + textwrap.indent(edge_store_body_source, "    ")
            )
            df.cute_state.register_tcgen05_per_tile_stmts(
                [sync_before_stmt, full_main_stmt, edge_main_stmt, sync_after_stmt]
            )
            df.cute_state.register_tcgen05_epi_role_full_edge_stmts(
                full_tile_stmts=[full_main_stmt],
                edge_tile_stmts=[edge_main_stmt],
            )
            # `cute.arch.alloc_smem` is a CuTe DSL static allocation even
            # though it is represented as a statement. Keeping the descriptor,
            # layout, and allocation statements in the epi-role prelude scopes
            # CuTe Python objects away from unrelated warp-role branches
            # without making the shared-memory reservation data-dependent on
            # the runtime epi-warp predicate.
            df.cute_state.register_tcgen05_epi_role_prelude_stmts(
                [
                    *grouped_d_tensormap_hoisted_stmts,
                    *tma_store_role_invariant_stmts,
                ]
            )
            main_stmts = [
                *tma_store_hoisted_stmts,
                *tma_store_role_invariant_stmts,
                sync_before_stmt,
                full_main_stmt,
                edge_main_stmt,
                sync_after_stmt,
            ]
        else:
            sync_before_stmt = statement_from_string("cute.arch.sync_threads()")
            sync_after_stmt = statement_from_string("cute.arch.sync_threads()")
            if tcgen05_m_subtile_count > 1:
                # M-paired tiles: drain both 256-row subtiles per work tile
                # (their acc stages committed together after the shared-B K
                # loop). ``unroll_full`` keeps ``_tcgen05_msub`` a trace-time
                # Python int so tile coordinates stay static expressions.
                main_stmt = statement_from_string(
                    f"for _tcgen05_msub in cutlass.range("
                    f"{tcgen05_m_subtile_count}, unroll_full=True):\n"
                    + textwrap.indent("\n".join(store_body_core), "    ")
                )
            else:
                main_stmt = statement_from_string(
                    "if True:\n" + textwrap.indent("\n".join(store_body_core), "    ")
                )
            df.cute_state.register_tcgen05_per_tile_stmts(
                [sync_before_stmt, main_stmt, sync_after_stmt]
            )
            df.cute_state.register_tcgen05_epi_role_stmts([main_stmt])
            main_stmts = [
                *tma_store_hoisted_stmts,
                sync_before_stmt,
                main_stmt,
                sync_after_stmt,
            ]
    else:
        if tcgen05_m_subtile_count > 1:
            # The cp.async-A flat kernel's M-paired tile drains its two
            # 128-row subtiles' acc stages in turn.
            store_body_core = [
                f"for _tcgen05_msub in cutlass.range("
                f"{tcgen05_m_subtile_count}, unroll_full=True):\n"
                + textwrap.indent("\n".join(store_body_core), "    ")
            ]
        store_body = [
            "cute.arch.sync_threads()",
            *store_body_core,
            "cute.arch.sync_threads()",
        ]
        main_stmt = statement_from_string(
            "if True:\n" + textwrap.indent("\n".join(store_body), "    ")
        )
        main_stmts = [main_stmt]
    # Pipeline drain + TMEM dealloc are one-shot cleanup. They must run
    # AFTER all tiles have been processed (in the persistent path) and
    # naturally land at the end of the kernel in the non-persistent path.
    # Keep them as separate statements so the persistent splitter can
    # extract them via the post-loop registration below.
    tma_store_post_loop_tail = ""
    if hoist_tma_store_resources or hoist_hybrid_tma_store_pipeline:
        # Role-local persistent epilogues reuse the C-store pipeline across
        # scheduler-recycled work tiles. Draining it inside each tile would
        # serialize the next tile's epilogue against this tile's TMA stores.
        # The lifecycle teardown places the tail against the TMEM dealloc.
        tma_store_post_loop_tail = tma_store_pipeline_tail
    if is_secondary_store:
        # The matmul drain + TMEM-free teardown is one-shot and owned by the
        # primary store. Each separately hoisted output pipeline still needs
        # its own drain before the kernel returns.
        post_loop_stmts = (
            [statement_from_string(tma_store_post_loop_tail)]
            if tma_store_post_loop_tail
            else []
        )
        df.cute_state.register_tcgen05_post_loop_stmts(post_loop_stmts)
    else:
        post_loop_lines = tcgen05_lifecycle.render_store_post_loop_lines(
            tma_store_pipeline_tail=tma_store_post_loop_tail
        )
        post_loop_stmts = [statement_from_string(line) for line in post_loop_lines]
        df.cute_state.register_tcgen05_post_loop_stmts(post_loop_stmts)
    if fanout_plan is not None:
        assert fanout_iteration is not None and isinstance(main_stmt, ast.If)
        assert fanout_advance is not None
        assert store_node is not None and hoist_tma_store_resources
        assert not split_hybrid_tma_store_role and not has_store_warp
        assert not tma_store_loop_first_subtile_acquire
        assert not tma_store_loop_late_later_subtile_acquire
        assert not tma_store_desc_arg and not tma_store_pipeline_tail_lines
        store = FanoutStore(
            plan=fanout_plan,
            site=store_node,
            main=main_stmt,
            setup=tuple(tma_store_body_setup_core),
            setup_without_acquire=tuple(
                build_tma_store_body_setup_core(
                    tma_store_atom=tcgen05_value.tma_store_atom,
                    tile_store_setup=tma_tile_store_setup,
                    dynamic_d_setup=[],
                    dynamic_d_update=[],
                    include_acquire=False,
                )
            ),
            iteration=fanout_iteration,
            subtile_count=subtile_count,
            acquire=tma_store_loop_later_subtile_acquire,
            wait=tma_store_loop_acc_wait,
            barrier=epilog_sync_barrier,
            buffer=c_buffer,
            buffer_expr=tma_c_buffer_expr,
            c_stages=tcgen05_c_stage_count,
            r2s=f"        cute.copy({tiled_copy_r2s}, {trs_rd}, {trs_sd}[(None, None, None, {c_buffer})])\n",
            tma=(
                (
                    f"        if {epi_leader_predicate}:\n"
                    if not is_secondary_store
                    else ""
                )
                + f"            cute.copy({tcgen05_tma_store_atom}, {bsg_sd}[(None, {c_buffer})], {bsg_gd}[(None, {tma_store_d_subtile_expr})])\n"
            ),
            commit=f"            {c_pipeline}.producer_commit()\n",
            advance=fanout_advance,
            late_acquire=(
                f"        if {epi_leader_predicate}:\n"
                f"            {c_pipeline}.producer_acquire()\n"
                if fanout_late_c_acquire
                else ""
            ),
            buffer_prefix=c_buffer_group_offset,
            pre_loop=tma_store_subtile_loop_aux_prefix,
            pre_loop_wait=tma_store_grouped_pre_loop_wait,
            loop_header=tma_store_subtile_loop_header,
            body_indent=tma_store_subtile_body_extra_indent,
            post_loop=tma_store_grouped_post_loop_release,
        )
        if df.cute_state.register_paired_fanout_store(store, post_loop_stmts):
            main_stmts.remove(main_stmt)
            # The first pipeline now owns both copies in each commit group.
            # Its existing wait.read0 precedes the original TMEM teardown.
            post_loop_stmts = []
    if tcgen05_value.split_k_last:
        # ``tcgen05_split_k``: the pipeline's exchange reduced the other
        # splits' partials into this CTA's TMEM accumulator when it arrived
        # last; every other split released the accumulator and skips the
        # output stores (the predicate is CTA-uniform).
        # The post-loop pipeline tails and the TMEM teardown run on every CTA.
        return [
            ast.If(
                test=cast("ast.expr", expr_from_string(tcgen05_value.split_k_last)),
                body=cast("list[ast.stmt]", main_stmts),
                orelse=[],
            ),
            *post_loop_stmts,
        ]
    return [*main_stmts, *post_loop_stmts]
