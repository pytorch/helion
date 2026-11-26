"""Few-row matmuls on 8x8 simdgroup-matrix fragments built in registers.

``metal_matmul="fragment"`` lowers each cooperative group of
:mod:`mpp_matmul` on Apple's 8x8 matrix atom instead of on ``matmul2d``:
lane ``l`` of the SIMD group holds elements ``(fm, fn)`` and ``(fm, fn + 1)``
of every 8x8 fragment (``fm = ((l / 4) & 4) + ((l / 2) % 4)``, ``fn = ((l /
4) & 2) * 2 + (l % 2) * 2``), and ``simdgroup_multiply_accumulate(D, A, B,
C)`` is ``D = A @ B + C``.  This is CuTe's ``partition_fragment`` and TiledMMA
for that atom.  A K tile's op is its 8-deep MMA steps, unrolled; in each,
every lane computes the two elements it holds of each A and B fragment in
registers and the SIMD group multiplies them.  No operand goes through
threadgroup memory and the op needs no barrier, which is what a computed
(dequantized) operand costs ``matmul2d`` at few rows.

The planner, the accumulator's storage and the element loops are
:mod:`mpp_matmul`'s; three things change:

- the accumulator (``_helion_mpp_acc(..., fragment=(rows, cols,
  transposed))``) is ``rows * cols`` float fragments, and an element loop
  visits the elements a lane holds of them (:func:`fragment_elements`);
- the op (``_helion_mpp_run(..., fragment=[...])``, :func:`codegen_fragment_run`)
  carries each operand as a device view (loads at the element, masked or
  clamped at the view's extents: :func:`_dim_mode`) or as the name of its
  value, cast to the operand's dtype as staging would; and
- :func:`fragment_run` (``elementize``) replaces the op with its loops: an
  operand element of a computed operand re-runs the Scalar statements that
  compute it, with the thread and lanes that hold it in the Scalar layout
  bound, by the element loops' rules, now over the operand's K as well.

The accumulator is transposed (``C^T = B^T A^T``) when only the rhs is
computed, so the computed operand is the MMA's A, whose lane elements lie
in one row; over a tile's steps a lane's are one run of K positions
(:func:`emit_fragment_run`), a few words and scales of a quantized weight.
The planner admits a group when the SIMD group can recompute each computed
operand element from loads, iotas and element-wise ops alone, or build it
from another group's result fragments (a chained operand,
:func:`_chain_refusal`) (:func:`check_fragment_group`); a tile laid out over
one SIMD group's 32 threads (``check_mpp_launch``) and multiples of 8
(:func:`fragment_grid`).

A few-row matmul whose activation the planner reads as the 16-bit view it
widens (``mpp_matmul._widen``) is transposed, the weight the MMA's A, and
its runs (the accumulator's ``by_fragment``) order the MMAs as MLX's
few-row kernel (``affine_qmv_mma``) does: the tile's B fragments first,
then each A fragment computed and multiplied at each step in turn
(:func:`_emit_by_fragment`).

``num_warps`` SIMD groups split each K loop (the plan's ``split``): each
runs the tile's 32-thread layout over every ``num_warps``-th K tile, from
its own, and after the loop the first adds the others' accumulators to
its own (:func:`emit_split_combine`) and alone writes memory.  The others
run the same code beside it, so every barrier is reached by all of them.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from fractions import Fraction
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

import sympy
import torch

from ...language.creation_ops import full
from ...language.matmul_ops import dot
from ...language.memory_ops import load
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..compile_environment import _to_sympy
from ..inductor_lowering import ReductionLowering
from ._constants import MAX_THREADGROUP_MEMORY_BYTES
from ._constants import SIMD_WIDTH
from .affine_fold import Affine
from .affine_fold import Fact
from .affine_fold import evaluate
from .affine_fold import render
from .backend import static_extent
from .free_arange import live_graphs
from .lane_fold import load_mask_other
from .lane_fold import tl_load
from .mpp_matmul import ACC_SEED
from .mpp_matmul import FRAG_A
from .mpp_matmul import FRAG_B
from .mpp_matmul import FRAG_HOIST
from .mpp_matmul import FRAG_RUN
from .mpp_matmul import PUT
from .mpp_matmul import RUN
from .mpp_matmul import WRITES
from .mpp_matmul import Dim
from .mpp_matmul import Graphs
from .mpp_matmul import MppGroup
from .mpp_matmul import MppPlan
from .mpp_matmul import View
from .mpp_matmul import assigned_names
from .mpp_matmul import call_name
from .mpp_matmul import current_plan
from .mpp_matmul import group_sides
from .mpp_matmul import is_tile_product
from .mpp_matmul import matmul_operands
from .mpp_matmul import metal_unsupported
from .mpp_matmul import names_read
from .mpp_matmul import op_dims
from .mpp_matmul import run_outer
from .mpp_matmul import static_tile
from .mpp_matmul import storage_key
from .mpp_matmul import view_args
from .mpp_matmul import virtual_thread
from .operand_elements import SIMDGROUP
from .operand_elements import SPLIT_BUFFER
from .operand_elements import ElementBody
from .operand_elements import OperandRecomputable
from .operand_elements import clamp_past_end
from .operand_elements import element_body
from .operand_elements import emit_split_adds
from .operand_elements import keyword
from .operand_elements import offset_fact
from .operand_elements import parse_expr
from .operand_elements import src
from .operand_elements import static_int
from .operand_elements import static_loop
from .operand_elements import tile_mask
from .operand_elements import tiles_in_range
from .reduction import alloc_threadgroup_buffer
from .search_surface import fragment_stages
from .search_surface import metal_tabulate_max_entries

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState
    from .msl_ast_walker import EmitState


#: Preamble of a kernel with fragment groups: the lane's row and first
#: column in every 8x8 fragment, from its index in the SIMD group (the
#: element loops shadow ``tid``).
LANE = "_frag_lane"
LANE_PREAMBLE = (
    f"const short _fm = (({LANE} / 4) & 4) + (({LANE} / 2) % 4);",
    f"const short _fn = (({LANE} / 4) & 2) * 2 + ({LANE} % 2) * 2;",
)
#: What the fold (``affine_fold``) knows of ``_fm`` and ``_fn``.
_LANE_FACTS = {"_fm": Fact(1, 0, 7), "_fn": Fact(2, 0, 6)}


def lane_facts(state: EmitState) -> dict[str, Fact]:
    """What the fold knows of a lane's names: ``_fm`` and ``_fn``, and its
    SIMD group's index when SIMD groups split the fragment rows."""
    facts = dict(_LANE_FACTS)
    if state.row_split > 1:
        facts[SIMDGROUP] = Fact(1, 0, state.row_split - 1)
    return facts


def check_fragment_group(group: MppGroup, plan: MppPlan, graphs: Graphs) -> None:
    """Raise ``BackendUnsupported`` unless the fragment run lowers the group:
    a matmul whose K is a block, accumulated across its K loop, a per-K-tile
    product, or one run over a ``:`` K whose extent fits a fragment K tile
    (:func:`k_tile_refusal`; a config can't change it, so the search offers
    fragments only when it fits), whose result is read only by element
    loops (stores and in-place updates, no Scalar read and no staged
    vector), whose computed operands a lane recomputes at any element: from
    loads, iotas, tile indices, constants, 0-d values and element-wise ops,
    none of them reading the accumulator or another matmul's result but
    through a chain (an operand built from that result's fragments,
    :func:`_chain_refusal`), and
    none a load of a tensor the kernel writes, which a lane would re-run at
    the op, after a write the Scalar load preceded.  A batched matmul runs
    on its batch tile's one index (``_check_batch``), which its views read
    as an outer index and its computed operands' elements as a uniform
    value."""

    def refuse(what: str) -> NoReturn:
        raise metal_unsupported(
            f"metal_matmul='fragment' can't lower {what}; use metal_matmul='mpp'"
        )

    if not group.lhs.dtype.is_floating_point:
        # MSL's is_simdgroup_matrix_element: half, bfloat and float only.
        refuse(f"{group.lhs.dtype} operands, which simdgroup_matrix can't hold")
    if group.k_factor != 1:
        refuse("a matmul over a K held as per-thread parts")
    if not group.levels and not is_tile_product(group):
        k = _whole_k_tile(group.k_block)
        if k is not None and (why := k_tile_refusal(k)) is not None:
            refuse(f"a matmul over a ':' K: {why}")
    if group.mats:
        refuse("a matmul whose result is read in the Scalar layout")
    if any(g is group for g in plan.by_vec.values()):
        refuse("a matmul whose element loop reads a staged row or column vector")
    if group.stage_uses and transposed(group):
        refuse(
            "a matmul transposed for its computed rhs whose result another "
            "matmul's operand is computed from"
        )
    _, lhs, rhs = matmul_operands(group.matmul)
    for side, (operand, node, block) in enumerate(
        ((group.lhs, lhs, group.m_block), (group.rhs, rhs, group.n_block))
    ):
        if operand.elements is not None:
            if (why := _chain_refusal(group, side, plan)) is not None:
                refuse(f"an operand computed from another matmul's result: {why}")
            continue
        if not operand.staged:
            continue
        blocks = {block, group.k_block}
        if group.batch_block is not None:
            # One index, the same at every element (``_check_batch``).
            blocks.add(group.batch_block)
        recomputable = OperandRecomputable(group, graphs, blocks)
        if not recomputable(node):
            refuse(
                f"the operand {node.name}, which a lane can't recompute at each "
                "of its elements from loads, iotas and element-wise ops"
            )
        if any(
            ld.target is load and storage_key(ld) in plan.written
            for ld in recomputable.nodes(node)
        ):
            refuse(
                f"the operand {node.name}, computed from a load of a tensor the "
                "kernel writes"
            )


def chain_source(group: MppGroup, side: int, plan: MppPlan) -> MppGroup:
    """The group whose result the operand on ``side`` (0 lhs, 1 rhs) is
    computed from (``mpp_matmul._link_element_staging``)."""
    operand = group.lhs if side == 0 else group.rhs
    (source,) = [h for h in plan.by_matmul.values() if h.gid == operand.elements]
    return source


def _chain_refusal(group: MppGroup, side: int, plan: MppPlan) -> str | None:
    """Why the operand on ``side``, computed element-wise from another
    group's result, can't be built from that result's fragments, or None.
    A, B and C fragments share one lane mapping, so the operand's fragment
    at (row, col) is the source's at the same (row, col) when its two dims
    are the source's M and N blocks, in order, in an accumulator that is
    not transposed (:func:`transposed`); the consumer then takes K in its
    natural order (:func:`_run_setup`).  The side reads the registers of
    the source's op, so the op runs first, in the consumer's graph: a
    consumer that reads the source's loop carry before the op (or from
    another graph) is refused."""
    source = chain_source(group, side, plan)
    nodes = list(group.matmul.graph.nodes)
    if source.matmul not in nodes or nodes.index(source.matmul) > nodes.index(
        group.matmul
    ):
        return "it reads the source's result before the source's op runs"
    if transposed(source):
        return "the source is transposed for its computed rhs"
    if transposed(group):
        return "the consumer is transposed for its computed rhs"
    dims = (
        (group.m_block, group.k_block) if side == 0 else (group.k_block, group.n_block)
    )
    if dims != (source.m_block, source.n_block):
        return "its dims are not the source's M and N blocks, in order"
    return None


def row_split_refusal(plan: MppPlan, graphs: list[GraphInfo]) -> str | None:
    """Why the SIMD groups of a threadgroup can't split the grid rows of
    the plan's fragment groups (``metal_fragment_row_split``), or None when
    they can.  The SIMD groups stack along z as a K split's do, each
    running the kernel's one-SIMD-group code, on its own rows of every
    fragment grid (``EmitState.row_base``): its element loops visit, and
    its row reductions fold, those rows only, and it re-runs every Scalar
    statement, which is why the kernel may write only from those loops.  So
    the groups' grid rows lie over one block (M, or N for a group
    transposed for its computed rhs), and no operand is built from another
    group's rows (a chained rhs takes the source's rows as its K); the
    kernel writes memory only from element loops, at their rows, and reads
    no tensor it writes; and nothing passes through threadgroup memory,
    which every SIMD group would write: no reduction but the row
    reductions of fragment results, no ``hl.dot`` the runs don't take, no
    seed tile or update vector.  SIMD groups splitting the
    K loops (``split_refusal``) are a different launch."""
    groups = list(plan.by_matmul.values())
    if len({g.n_block if transposed(g) else g.m_block for g in groups}) > 1:
        return "the fragment groups' rows lie over different blocks"
    if any(g.rhs.elements is not None and not transposed(g) for g in groups):
        return "an operand is built from another matmul's rows"
    # A fragment group's ``stage_uses`` are chains (``check_fragment_group``),
    # which read registers.
    if any(g.vecs for g in groups) or any(
        g.seed is not None and g.seed.target is not full for g in groups
    ):
        return "a fragment group stages values in threadgroup memory"
    live = live_graphs(graphs)
    nodes = [node for info in live for node in info.graph.nodes]
    if any(
        isinstance(node.meta.get("lowering"), ReductionLowering)
        and node not in plan.line_reds
        for node in nodes
    ):
        return "the kernel has a reduction its fragment rows don't hold"
    if any(node.target is dot and node not in plan.by_matmul for node in nodes):
        return (
            "the kernel has an hl.dot the fragment runs don't take, whose "
            "contraction reduces across threads"
        )
    element_stores = plan.line_stores.union(*(g.stores for g in groups))
    if any(node.target in WRITES and node not in element_stores for node in nodes):
        return "the kernel writes memory outside a fragment element loop"
    if any(node.target is load and storage_key(node) in plan.written for node in nodes):
        return "the kernel reads a tensor it writes"
    return None


def row_block(plan: MppPlan) -> int:
    """The block the plan's fragment grid rows lie over, which
    ``row_split_refusal`` admits only when it is one."""
    (block,) = {
        g.n_block if transposed(g) else g.m_block for g in plan.by_matmul.values()
    }
    return block


def transposed(group: MppGroup) -> bool:
    """Whether the group computes ``C^T = B^T A^T``: only its rhs is computed
    by each lane at its elements, which then takes the MMA's A, whose lane
    elements lie in one row.  A rhs built from another group's fragments
    (a chain, :func:`_chain_refusal`) keeps the orientation."""
    return group.rhs.staged and group.rhs.elements is None and not group.lhs.staged


#: Most MMA steps in a K tile (a K block of 128).  The steps are unrolled
#: (:func:`emit_fragment_run`), so the kernel's source and compile time grow
#: with them: a dequantized operand's run at 8 steps is thousands of lines.
#: A bound on the unrolled code (codegen and compile time), not a hardware
#: or performance limit.  Crossing it is graceful: codegen raises
#: ``BackendUnsupported`` before any MSL compile, and the autotuner skips
#: that config.
MAX_STEPS = 16


def k_tile_refusal(k: int) -> str | None:
    """Why a fragment run can't take a K tile of ``k``, or None when it can:
    whole 8-deep MMA steps, at most :data:`MAX_STEPS` of them."""
    if k % 8:
        return f"a fragment matmul K tile of {k}: K must be a multiple of 8"
    if k // 8 > MAX_STEPS:
        return (
            f"a fragment matmul K tile of {k}: its {k // 8} MMA steps are "
            f"unrolled (max {MAX_STEPS})"
        )
    return None


def _whole_k_tile(block: int) -> int | None:
    """The K tile of a run over the ``:`` K of ``block`` when no config
    changes it: the extent a persistent reduction dim takes
    (``ReductionLoopBlockSizeSource``), or None when ``reduction_loops`` can
    roll the dim or it takes a tiled axis's size."""
    env = CompileEnvironment.current()
    if (
        block in env.config_spec.reduction_loops.valid_block_ids()
        or env.registered_extent_owner(block) is not None
    ):
        return None
    return env.backend.static_rdim_size(max(1, env.block_sizes[block].size_hint()))


def fragment_grid(group: MppGroup, m: int, n: int, k: int) -> tuple[int, int, int]:
    """The accumulator's ``(rows, cols, transposed)``: its 8x8 fragments
    over the op's M and N (N and M when transposed), for an op of ``m`` x
    ``n`` over a K tile of ``k``, the rows those of one SIMD group's share
    when SIMD groups split them (``MppPlan.row_split``).  A wide tile keeps
    many fragments live in each lane; that is slower, not wrong, and the
    autotuner times it."""
    if m % 8 or n % 8:
        raise metal_unsupported(
            f"a fragment matmul tile {m}x{n}x{k}: M and N must be multiples of 8"
        )
    if (why := k_tile_refusal(k)) is not None:
        raise metal_unsupported(f"{why}; use a smaller K block")
    rows, cols, flipped = (
        (n // 8, m // 8, 1) if transposed(group) else (m // 8, n // 8, 0)
    )
    split = current_plan().row_split
    if rows % split:
        raise metal_unsupported(
            f"a fragment matmul tile of {rows * 8} rows split over {split} SIMD "
            "groups (metal_fragment_row_split): each takes whole 8-row fragments"
        )
    return rows // split, cols, flipped


#: How a view's element load treats a dim (:func:`_dim_mode`).
_UNMASKED, _MASKED, _BROADCAST, _CLAMPED = 0, 1, 2, 3


def _dim_mode(
    cg: GenerateAST, view: View, d: int, dim: Dim, op: int, k_block: int
) -> int:
    """How a view's element loads index dim ``d`` (``op`` positions from the
    view's origin): a tile over a dim of size 1 broadcasts it (index 0, K
    masked at the loop's end only); otherwise the dim's extent masks it
    unless a tile dim's positions stay inside it at every tile, or a ``:``
    dim's inside its size.  An M or N tile whose loop's end is the dim's
    size clamps it instead of masking it (:func:`_view_element`): a
    position past the size is past the loop's end, where the element loops
    store nothing."""
    size = static_extent(view.tensor.size(view.positions[d]))
    if dim.full:
        return _MASKED if size is None or op > size else _UNMASKED
    if dim.iota is not None or dim.base is not None or dim.factor != 1:
        return _MASKED
    if size == 1:
        return _BROADCAST
    if dim.block_id == k_block:
        # The view's K extent is the loop's end (view_args).
        states = cg.active_device_loops[k_block]
        size = static_int(states[-1].block_id_to_info[k_block].end_expr)
    if tiles_in_range(cg, dim.block_id, op, size):
        return _UNMASKED
    if dim.block_id != k_block:
        info = cg.active_device_loops[dim.block_id][-1].block_id_to_info.get(
            dim.block_id
        )
        end = None if info is None else info.end_expr
        extent = _to_sympy(view.tensor.size(view.positions[d]))
        # The same size, not two sizes equal at these inputs only: keying the
        # bound kernel on their equality (is_end_matching_keyed) would rebind
        # it for inputs where they differ, to save a select.
        if end is not None and _specialized(end) == _specialized(extent):
            return _CLAMPED
    return _MASKED


def _max_offset(tensor: torch.Tensor) -> int | None:
    """The largest element offset from ``tensor``'s data pointer that its
    indices reach, ``sum((size - 1) * |stride|)``: what decides whether its
    offsets fit an ``int`` (a strided view's can pass 2**31 at few
    elements).  None when a size or stride isn't known at codegen."""
    reach = 0
    for size, stride in zip(tensor.size(), tensor.stride(), strict=True):
        n, step = static_extent(size), static_extent(stride)
        if n is None or step is None:
            return None
        reach += max(n - 1, 0) * abs(step)
    return reach


def _specialized(end: object) -> object:
    """A loop end with the sizes ``hl.specialize`` pins substituted."""
    if isinstance(end, sympy.Expr):
        return CompileEnvironment.current().specialize_expr(end)
    return end


def loads_ahead(begin: object, end: object) -> bool:
    """Whether a K loop from ``begin`` to ``end`` can load its fragment
    runs' operands a tile ahead (``range_num_stages=2``): both are known,
    the begin isn't negative and the end is past it, so the loop runs and
    its first tile's loads can go before it.  ``register_search_surface``
    offers the loop a stages slot only where its call's bounds pass, and
    codegen checks its loop's again under the slot, refusing the config
    (``BackendUnsupported``) where they don't."""
    first, last = static_int(begin), static_int(end)
    return first is not None and first >= 0 and last is not None and last > first


def codegen_fragment_run(
    cg: GenerateAST, group: MppGroup, acc: ast.Name, values: tuple[object, object]
) -> ast.AST:
    """``_helion_mpp_run(acc, gid, lhs..., rhs..., ..., fragment=[lhs, rhs])``
    of a fragment group: each operand as a view's nine run arguments
    (``view_args``) or the name of its value cast to the operand's dtype,
    and per operand ``(computed, view dims masked, bindings, wide, past,
    chain, hoisted, tile)``, where the bindings make a lane act as the
    thread holding the element at
    ``(_e_m or _e_n, _e_k)`` of a computed operand, ``wide`` marks a view
    whose indices may pass 2**31 and ``past`` names a computed operand's M
    or N tile's mask, index and offset when the mask is live
    (:func:`tile_mask`), for its loads past the tile's loop end
    (:func:`clamp_past_end`); ``chain`` names the result a chained operand
    is built from (:func:`_chain_name`), ``hoisted`` the fragments a hoisted
    one was built into (:func:`codegen_fragment_hoist`) and ``tile`` the
    layout of a tile side's copy (:data:`TILE_COPY`).  ``steps`` is the K tile's MMA steps,
    ``k_live`` its live length, which masks a computed operand's K tail
    (None when every tile is whole), ``facts`` the tile offsets' ``(offset,
    multiple, least, most)`` (:func:`offset_fact`), over whose multiple and
    least the run folds its elements' index arithmetic, ``tables`` the
    config's
    ``metal_tabulate_max_entries``, ``k_step`` the K loop's offset and
    step when it loads its operands a tile ahead (``range_num_stages=2``),
    else None, and ``split`` the K loop's offset and the SIMD groups that
    split it (:func:`_split`), None for one."""
    backend = CompileEnvironment.current().backend
    m, n, k = static_tile(cg, (group.m_block, group.n_block, group.k_block))
    op = dict(zip((group.m_block, group.n_block), op_dims(group, m, n), strict=True))
    op[group.k_block] = k
    parts: list[str] = []
    anchors: list[tuple[int, str | None]] = []
    outers: list[str] = []
    sides: list[str] = []
    plan = current_plan()
    for side, ((operand, blocks, k_pos), value, e) in enumerate(
        zip(group_sides(group), values, ("_e_m", "_e_n"), strict=True)
    ):
        view = operand.view
        if view is None and tile_side(group, side):
            # A widened view: copied into its tile like a view, as its 16-bit
            # elements, which the fragment loads widen (``_tile_loads``).
            view = operand.widened
        chain = None
        if operand.elements is not None:
            chain = _chain_name(cg, group, side, plan, op, k)
        hoisted = None if operand.hoist is None else hoist_name(group, side)
        tile = None
        if view is None:
            assert isinstance(value, ast.AST)
            name = cg.lift(backend.cast_ast(value, operand.dtype), prefix="_mpp_fv")
            op_parts = [name.id]
            masks: list[int] = []
            wide = 0
            binds = virtual_thread(
                cg, ((blocks[1 - k_pos], e), (group.k_block, "_e_k"))
            )
            outer = ""
            past = tile_mask(cg, blocks[1 - k_pos])
        else:
            op_parts, op_anchors, _, view_outer = view_args(
                cg, view, group.k_block, value
            )
            anchors += [(2 + len(parts) + i, a) for i, a in op_anchors]
            masks = [
                _dim_mode(cg, view, d, dim, op[dim.block_id], group.k_block)
                for d, dim in enumerate(view.dims)
            ]
            reach = _max_offset(view.tensor)
            wide = int(reach is None or reach >= 2**31)
            binds = []
            outer = "".join(f"({x}, {s}, {st}, {a}), " for x, s, st, a in view_outer)
            past = None
            if tile_side(group, side):
                shape = (op[group.m_block], k) if side == 0 else (k, op[group.n_block])
                tile_outer, pitch, vector = _tile_layout(view, shape, masks)
                buf = alloc_threadgroup_buffer(
                    backend.dtype_str(view.tensor.dtype),
                    shape[tile_outer] * pitch,
                    prefix=TILE_PREFIX,
                )
                size = _TILE_TYPES[view.tensor.dtype]
                # A widened view's tile holds the view's type; each lane
                # widens its fragment elements as it loads them.
                widen = (
                    backend.dtype_str(view.tensor.dtype)
                    if view.tensor.dtype != operand.dtype
                    else None
                )
                tile = (buf, shape, tile_outer, pitch, vector, size, widen)
        parts += op_parts
        outers.append(f"[{outer}]")
        sides.append(
            f"({int(operand.staged)}, {masks}, [{', '.join(binds)}], {wide}, "
            f"{past!r}, {chain!r}, {hoisted!r}, {tile!r})"
        )
    k_live = _k_live(cg, group, k) if group.lhs.staged or group.rhs.staged else "None"
    anchor_list = ", ".join(f"({p}, {a})" for p, a in anchors)
    facts = [
        fact
        for block in (group.m_block, group.n_block, group.k_block)
        if (fact := offset_fact(cg, block)) is not None
    ]
    config = cg.device_function.config
    stages = fragment_stages(
        CompileEnvironment.current().config_spec, config, group.k_block
    )
    # Loading a tile ahead needs the K offset's first value, the least of
    # its fact, which the first tile's loads before the loop read, and a
    # loop that runs.  The search offers 2 stages only on a K loop whose
    # call's bounds loads_ahead admits (``register_search_surface``); this
    # checks the loop codegen sees.
    k_step = None
    k_tiles = None
    # Tile sides load a tile ahead through their copies, between any bounds
    # (``pipeline_tile_copies``); the other sides' element loads need known
    # ones, unless no side loads per lane (chained or hoisted sides).
    tiles = [tile_side(group, side) for side in (0, 1)]
    pipe = stages >= 2 and any(tiles)
    per_lane = any(
        not tiles[side] and operand.elements is None and operand.hoist is None
        for side, operand in enumerate((group.lhs, group.rhs))
    )
    if stages >= 2 and (per_lane or not pipe):
        k_fact = offset_fact(cg, group.k_block)
        k_loop = static_loop(cg, group.k_block)
        end = None if k_loop is None else static_int(_specialized(k_loop[0].end_expr))
        if k_fact is None or k_loop is None or not loads_ahead(k_fact[2], end):
            raise metal_unsupported(
                "range_num_stages=2 on a K loop without known bounds it runs between"
            )
        assert end is not None  # loads_ahead admits known bounds only
        k_step = (k_fact[0], k_loop[2])
        k_tiles = -(-(end - k_loop[1]) // k_loop[2])
    natural, element = run_form(group)
    return expr_from_string(
        f"{RUN}({acc.id}, {group.gid}, {', '.join(parts)}, "
        f"anchors=[{anchor_list}], outer=[{', '.join(outers)}], steps={k // 8}, "
        f"k_live={k_live}, facts={facts!r}, "
        f"tables={metal_tabulate_max_entries(cg.device_function.config)}, "
        f"k_step={k_step!r}, split={_split(cg, group, k_tiles)!r}, "
        f"natural={int(natural)}, element={element!r}, pipe={int(pipe)}, "
        f"fragment=[{', '.join(sides)}])"
    )


def run_form(group: MppGroup) -> tuple[bool, str]:
    """Whether the group's run takes K in its natural order, and its
    operand fragments' element type.  A chained side reads its source's
    fragments a K step at a time, in order, and a tile side loads them so
    (:func:`tile_side`); such a run's operand fragments hold the operands'
    own type (half or bfloat); any other run keeps the order
    :func:`emit_fragment_run` describes and float fragments."""
    natural = any(o.elements is not None for o in (group.lhs, group.rhs)) or any(
        tile_side(group, side) for side in (0, 1)
    )
    element = "float"
    if natural and group.lhs.dtype in (torch.float16, torch.bfloat16):
        element = CompileEnvironment.current().backend.dtype_str(group.lhs.dtype)
    return natural, element


def hoist_name(group: MppGroup, side: int) -> str:
    """The fragments a hoisted operand is built into before its loop."""
    return f"_frag_h{group.gid}_{side}"


#: The element types a tile side copies as raw bits: their bytes.
_TILE_TYPES = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}
#: The threadgroup tiles of tile sides (:func:`tile_side`).
TILE_PREFIX = "_frag_tile"
#: A tile side's copy, ``_helion_frag_tile_copy(tile, (shape, outer, pitch,
#: vector, element bytes, widen), view..., masks=, wide=, outer=)``: the third argument of its
#: side's loop marker (:func:`fragment_run`), emitted by
#: :func:`emit_tile_copy`.
TILE_COPY = "_helion_frag_tile_copy"


def tile_view(group: MppGroup, side: int) -> View | None:
    """The device view the operand on ``side`` would copy into a tile: its
    own, or the half or bfloat view a float32 operand converts from
    (``mpp_matmul._Operand.widened``), copied as its 16-bit elements and
    widened at the fragment loads."""
    operand = (group.lhs, group.rhs)[side]
    return operand.view if operand.view is not None else operand.widened


def can_stage(group: MppGroup, side: int) -> bool:
    """Whether the operand on ``side`` can load its fragments from a
    threadgroup tile (``metal_fragment_stage``, :func:`tile_side`): a device
    view (:func:`tile_view`), not hoisted, the tile holding the run's
    operand type (both operands half, bfloat or float), over two dims of
    more than one element (a size-1 dim broadcasts, which a tile row
    can't).  In a transposed group (:func:`transposed`) the lhs view's
    fragments are the MMA's B side (:func:`_tile_loads`).  Not a few-row
    activation (``_Operand.few_rows``), which a transposed group reads as
    its 16-bit view, widened at each element in the order of
    :func:`_emit_by_fragment`, and whose tile was never checked against
    the GPU."""
    operand = (group.lhs, group.rhs)[side]
    view = tile_view(group, side)
    if view is None or operand.hoist is not None or operand.few_rows:
        return False
    dtype = operand.dtype
    if dtype not in _TILE_TYPES or {group.lhs.dtype, group.rhs.dtype} != {dtype}:
        return False
    return all(static_extent(view.tensor.size(p)) != 1 for p in view.positions)


def tile_side(group: MppGroup, side: int) -> bool:
    """Whether the run loads the operand on ``side`` from a threadgroup tile
    that every thread of the threadgroup copies from its view each time the
    run's loop body runs (``metal_fragment_stage``; :func:`emit_tile_copy`),
    with ``simdgroup_load``: one copy shared by the SIMD groups that split
    the rows, coalesced, in place of each lane's element loads."""
    return current_plan().stage and can_stage(group, side)


def _tile_layout(
    view: View, shape: tuple[int, int], modes: list[int]
) -> tuple[int, int, int]:
    """A tile side's ``(outer, pitch, vector)``: the op dim its tile rows
    run over (the other, its tile columns, is the view's unit-stride dim
    when it has one), its row pitch in elements, padded by 16 bytes so the
    rows of an 8 x 8 load start in different banks, and the elements each
    thread copies at a time: 8 bytes' worth where the columns are unit
    stride, unmasked and a multiple of it, else 1."""
    inner = 1 if view.strides[1] == 1 or view.strides[0] != 1 else 0
    outer = 1 - inner
    size = _TILE_TYPES[view.tensor.dtype]
    vector = 8 // size
    if view.strides[inner] != 1 or modes[inner] != _UNMASKED or shape[inner] % vector:
        vector = 1
    return outer, shape[inner] + 16 // size, vector


def _k_live(cg: GenerateAST, group: MppGroup, k: int) -> str:
    """The live length of the group's K tile, ``end - offset``, where its
    last tile may pass the K extent (:func:`tiles_in_range`), else
    ``"None"``: the bound past which a computed operand's elements are 0."""
    info = cg.active_device_loops[group.k_block][-1].block_id_to_info[group.k_block]
    if tiles_in_range(cg, group.k_block, k, static_int(info.end_expr)):
        return "None"
    return f"{info.end_var_name} - {cg.offset_var(group.k_block)}"


def codegen_fragment_hoist(state: CodegenState, group: MppGroup, side: int) -> ast.AST:
    """``_helion_mpp_frag_hoist(value, name, role, position, steps, count,
    natural, element, k_live, binds)`` before the loop of a hoisted operand
    (``mpp_matmul._hoist_invariant``, under ``metal_fragment_hoist``): the
    operand's fragments for every MMA step of its op, built once into
    ``name``, ``steps * count`` of them, the MMA's A fragments (``role`` 0)
    or B fragments (1), over M (``position`` ``_e_m``, the lhs) or N, in
    the order and type of its run
    (:func:`run_form`); ``k_live`` is the live length of a K tile past
    the operand's extent, whose elements there are 0, as a run's are
    (:func:`fragment_run`), or None; ``binds`` make a lane act as the thread
    holding an element.  Its run reads them (:func:`emit_fragment_run`)."""
    cg = state.codegen
    backend = CompileEnvironment.current().backend
    m, n, k = static_tile(cg, (group.m_block, group.n_block, group.k_block))
    rows, cols, flipped = fragment_grid(group, *op_dims(group, m, n), k)
    role = side ^ flipped
    operand, blocks, k_pos = group_sides(group)[side]
    position = "_e_m" if side == 0 else "_e_n"
    binds = virtual_thread(cg, ((blocks[1 - k_pos], position), (group.k_block, "_e_k")))
    value = cg.lift(backend.cast_ast(state.ast_arg(0), operand.dtype), prefix="_mpp_fv")
    natural, element = run_form(group)
    k_live = _k_live(cg, group, k)
    return cg.lift(
        expr_from_string(
            f"{FRAG_HOIST}({value.id}, {hoist_name(group, side)!r}, {role}, "
            f"{position!r}, {k // 8}, {cols if role else rows}, {int(natural)}, "
            f"{element!r}, {k_live}, [{', '.join(binds)}])"
        ),
        prefix="_frag_hv",
    )


def _chain_name(
    cg: GenerateAST,
    group: MppGroup,
    side: int,
    plan: MppPlan,
    op: Mapping[int, int],
    k: int,
) -> str:
    """The name of the result a chained side is built from, after checking
    that the source's op covers the side's fragments: its M and N op dims
    are the side's (the consumer's M or N, and its K tile)."""
    source = chain_source(group, side, plan)
    sm, sn, _ = static_tile(cg, (source.m_block, source.n_block, source.k_block))
    sm, sn = op_dims(source, sm, sn)
    if side == 0:
        want = (op[group.m_block], k)
    else:
        want = (k, op[group.n_block])
    if (sm, sn) != want:
        raise metal_unsupported(
            f"a fragment operand built from another op's {sm}x{sn} result over "
            f"a {want[0]}x{want[1]} tile"
        )
    # The source's op, ``_helion_mpp_run(acc, ...)``: its accumulator is the
    # cooperative root the side reads.
    code = source.matmul.meta["codegen"]
    if isinstance(code, ast.Name):
        return code.id
    assert isinstance(code, ast.Call) and isinstance(code.args[0], ast.Name)
    return code.args[0].id


def _loop_fragments(cg: GenerateAST, group: MppGroup) -> int:
    """The accumulator fragments of the fragment groups of ``group``'s K
    loop, its own included: what a split's combine adds."""
    loop = group.levels[0].owner
    total = 0
    for other in current_plan().by_matmul.values():
        if other.levels[0].owner is loop:
            m, n, _ = static_tile(cg, (other.m_block, other.n_block, other.k_block))
            m, n = op_dims(other, m, n)
            total += (m // 8) * (n // 8)
    return total


def _split(
    cg: GenerateAST, group: MppGroup, tiles: int | None
) -> tuple[str, int] | None:
    """A fragment run's K loop's offset and the SIMD groups that split the
    loop (the plan's ``split``), or None when one SIMD group runs it.
    Refuses (``BackendUnsupported``) a split whose combine needs more
    threadgroup memory than there is, and one into more SIMD groups than
    the loop's ``tiles`` when it loads a tile ahead (``tiles`` given): a
    SIMD group without a tile would load past the loop's end before it."""
    split = current_plan().split
    if split == 1:
        return None
    fragments = _loop_fragments(cg, group)
    combine = (split - 1) * fragments * SIMD_WIDTH * 8
    if combine > MAX_THREADGROUP_MEMORY_BYTES:
        raise metal_unsupported(
            f"a fragment K loop split over {split} SIMD groups (num_warps): adding "
            f"their {fragments} accumulator fragments takes {combine} bytes of "
            f"threadgroup memory (max {MAX_THREADGROUP_MEMORY_BYTES}); lower num_warps"
        )
    if tiles is not None and split > tiles:
        raise metal_unsupported(
            f"range_num_stages=2 on a K loop of {tiles} tiles split over {split} "
            "SIMD groups (num_warps): a SIMD group without a tile would load "
            "past the loop's end"
        )
    return cg.offset_var(group.k_block), split


def is_fragment_acc(call: ast.Call) -> bool:
    """Whether a ``_helion_mpp_acc`` call declares a fragment group's
    accumulator (:func:`emit_fragment_acc`)."""
    return keyword(call, "fragment") is not None


def fragment_run_binds(call: ast.Call) -> list[list[ast.Tuple]]:
    """A ``_helion_mpp_run``'s per-operand bindings when it is a fragment
    run; ``[]`` for an MPP run."""
    sides = keyword(call, "fragment")
    if sides is None:
        return []
    assert isinstance(sides, ast.List)
    out: list[list[ast.Tuple]] = []
    for side in sides.elts:
        assert isinstance(side, ast.Tuple)
        binds = side.elts[2]
        assert isinstance(binds, ast.List)
        out.append(cast("list[ast.Tuple]", binds.elts))
    return out


def _view_element(
    args: list[ast.expr],
    masks: list[int],
    outer: list[list[ast.expr]],
    index: tuple[str, str],
    wide: int,
    element: str = "float",
) -> ast.expr:
    """A view's element at ``index`` (the op's two positions from its
    origin) as an ``element`` (the run's fragment type): a load, masked to 0 past each masked dim's extent
    and where an outer index lies outside its dim; a broadcast dim reads
    index 0, and a clamped dim its origin past its extent
    (:func:`_dim_mode`).  A ``wide`` view, one that may hold 2**31 elements
    or more, computes its offset in ``long``."""
    tensor, *dims = args
    offset: list[str] = []
    terms: list[str] = []
    for d, i in enumerate(index):
        o, e, size, stride = dims[4 * d : 4 * d + 4]
        at = f"({src(o)} + {i})"
        if masks[d] == _BROADCAST:
            # K past the loop's end (the view's K extent) reads 0; rows and
            # columns past M's and N's are never stored.
            if i == "_e_k":
                terms.append(f"{at} < {src(e)}")
            continue
        if masks[d] == _CLAMPED:
            # A row or column past the extent, the tile's loop end, is never
            # stored, so its value is free: it reads the tile's first, which
            # lies inside the extent at every tile the loop runs.
            at = f"({at} if {at} < {src(e)} else {src(o)})"
        offset.append(f"{f'static_cast(long, {at})' if wide else at} * {src(stride)}")
        if masks[d] == _MASKED:
            terms.append(f"{at} < {src(e)}")
            if ast.dump(e) != ast.dump(size):
                terms.append(f"{at} < {src(size)}")
    for x, size, stride in outer:
        offset.append(
            f"{f'static_cast(long, {src(x)})' if wide else src(x)} * {src(stride)}"
        )
        terms.append(f"0 <= {src(x)} and {src(x)} < {src(size)}")
    mask = " and ".join(terms) if terms else "None"
    other = ", other=0" if terms else ""
    return parse_expr(
        f"static_cast({element}, tl.load({src(tensor)} + "
        f"({' + '.join(offset) or '0'}), {mask}{other}))"
    )


def fragment_run(
    call: ast.Call,
    acc_call: ast.Call,
    rerun: Callable[[Mapping[str, ast.expr], set[str], list[ast.AST]], list[ast.stmt]],
) -> ast.For:
    """The loop that replaces a fragment run ``call`` (after
    ``_Copies.rewrite_run``) of the group whose accumulator ``acc_call``
    declares: over the K tile's MMA steps, the loop over the elements each
    lane holds of the MMA's A fragments, then of its B fragments, each
    writing the operand's value at the element (``_helion_mpp_put``).  A
    computed operand's element is its value's name with the Scalar
    statements it reads re-run by ``rerun(binds, reads, checked)``, under
    its bindings, their loads past the M or N tile's loop end read at the
    tile's first position (:func:`clamp_past_end`), and is 0 past the K
    tile's live length; a view's is a load masked or clamped at the view's
    extents (:func:`_view_element`)."""
    grid = keyword(acc_call, "fragment")
    assert grid is not None
    steps, k_live, facts, tables, k_step, split, natural, element_arg, pipe, sides = (
        keyword(call, name)
        for name in (
            "steps",
            "k_live",
            "facts",
            "tables",
            "k_step",
            "split",
            "natural",
            "element",
            "pipe",
            "fragment",
        )
    )
    assert grid is not None and steps is not None and facts is not None
    assert tables is not None and k_step is not None and split is not None
    assert natural is not None and element_arg is not None and pipe is not None
    assert isinstance(sides, ast.List)
    element = ast.literal_eval(element_arg)
    _, _, flipped, _ = ast.literal_eval(grid)
    outer = run_outer(call)
    pos = 2
    loops: list[list[ast.stmt]] = []
    chains: list[str | None] = []
    hoists: list[str | None] = []
    tiles: list[ast.expr] = []
    for i, (side, binds_list) in enumerate(
        zip(sides.elts, fragment_run_binds(call), strict=True)
    ):
        assert isinstance(side, ast.Tuple)
        staged, masks, wide, past, chain, hoisted = (
            ast.literal_eval(side.elts[j]) for j in (0, 1, 3, 4, 5, 6)
        )
        chains.append(chain)
        hoists.append(hoisted)
        tile = ast.literal_eval(side.elts[7])
        if tile is not None:
            # Copied into its tile (:func:`emit_tile_copy`) and loaded from
            # it, with no element loop.
            buf, *layout = tile
            tiles.append(
                ast.Call(
                    ast.Name(TILE_COPY, ast.Load()),
                    [
                        ast.Name(buf, ast.Load()),
                        _literal(tuple(layout)),
                        *call.args[pos : pos + 9],
                    ],
                    [
                        ast.keyword("masks", _literal(tuple(masks))),
                        ast.keyword("wide", ast.Constant(wide)),
                        ast.keyword(
                            "outer",
                            ast.List(
                                [ast.Tuple(list(o), ast.Load()) for o in outer[i]],
                                ast.Load(),
                            ),
                        ),
                    ],
                )
            )
            pos += 9
            loops.append([])
            continue
        tiles.append(ast.Constant(None))
        index = ("_e_m", "_e_k") if i == 0 else ("_e_k", "_e_n")
        if staged:
            value = parse_expr(f"static_cast({element}, {src(call.args[pos])})")
            pos += 1
            if not (isinstance(k_live, ast.Constant) and k_live.value is None):
                assert k_live is not None
                # 0 of the fragments' type: a half or bfloat select of a
                # float 0 would be float.
                zero = "0.0" if element == "float" else f"static_cast({element}, 0.0)"
                value = parse_expr(
                    f"({src(value)} if _e_k < {src(k_live)} else {zero})"
                )
        else:
            value = _view_element(
                call.args[pos : pos + 9], masks, outer[i], index, wide, element
            )
            pos += 9
        put = ast.Expr(ast.Call(ast.Name(PUT, ast.Load()), [value], []))
        binds: dict[str, ast.expr] = {}
        for b in binds_list:
            name, expr = b.elts
            binds[cast("str", ast.literal_eval(name))] = expr
        stmts = rerun(binds, names_read(put), [value])
        if past is not None:
            mask, tile_index, offset = past
            position = index[0] if i == 0 else index[1]
            stmts = clamp_past_end(stmts, mask, tile_index, offset, binds, position)
        # A hoisted side's fragments were built before the loop.
        loops.append([] if hoisted is not None else [*stmts, put])
    assert pos == len(call.args)
    a_side = 1 if flipped else 0
    body: list[ast.stmt] = [
        ast.For(
            ast.Name("_it", ast.Store()),
            ast.Call(
                ast.Name(marker, ast.Load()),
                [ast.Constant(chains[side]), ast.Constant(hoists[side]), tiles[side]],
                [],
            ),
            loops[side],
            [],
        )
        for marker, side in ((FRAG_A, a_side), (FRAG_B, 1 - a_side))
    ]
    loop = ast.For(
        ast.Name("_it", ast.Store()),
        ast.Call(
            ast.Name(FRAG_RUN, ast.Load()),
            [
                call.args[0],
                steps,
                facts,
                tables,
                k_step,
                split,
                natural,
                element_arg,
                pipe,
            ],
            [],
        ),
        body,
        [],
    )
    ast.fix_missing_locations(loop)
    return loop


# MSL emission (msl_ast_walker).


def _grid(state: EmitState, root: str) -> tuple[int, int, bool]:
    rows, cols, flipped = state.fragments[root]
    return rows, cols, bool(flipped)


def _element_names(flipped: bool) -> tuple[str, str]:
    """The names of an accumulator element's position along a fragment's
    rows and along its columns."""
    return ("_e_n", "_e_m") if flipped else ("_e_m", "_e_n")


def _copies(pad: str, count: int, lines: list[str]) -> list[str]:
    """``lines``, written for an index ``_fi``, once per ``_fi < count``, each
    in a block of its own with ``_fi`` a constant, so every fragment index is
    one and the fragments stay in registers."""
    out: list[str] = []
    for i in range(count):
        out += [f"{pad}{{", f"{pad}    constexpr short _fi = {i};", *lines, f"{pad}}}"]
    return out


#: The element ``_fi`` of a lane's fragments, after an array of them: the
#: lane holds two elements of each 8x8 fragment.
FRAGMENT_SLOT = "[_fi >> 1].thread_elements()[_fi & 1]"


def fragment_elements(
    coop: str,
    root: str,
    pad: str,
    state: EmitState,
    body: Callable[[int], list[str]],
    read: bool = True,
) -> list[str]:
    """A loop over the accumulator elements a lane holds, inside its block,
    unrolled (:func:`_copies`): each element's row and column in ``_e_m`` and
    ``_e_n``, its value in ``_fv`` (unless the body doesn't ``read`` it) and
    its lvalue in ``state.put`` around the lines ``body(indent)`` emits."""
    rows, cols, flipped = _grid(state, root)
    row, col = _element_names(flipped)
    block_m, block_n = state.coop_tile[root]
    inner = pad + "    "
    element = f"{coop}{FRAGMENT_SLOT}"
    lines = [
        f"{inner}int {row} = 8 * ((_fi >> 1) / {cols}) + _fm{state.row_base};",
        f"{inner}int {col} = 8 * ((_fi >> 1) % {cols}) + _fn + (_fi & 1);",
    ]
    saved = state.put
    state.put = element
    value = [f"float _fv = {element};"] if read else []
    if root in state.coop_rounded:
        lines += [
            f"{inner}if (_e_m < {block_m} && _e_n < {block_n}) {{",
            *(f"{inner}    {line}" for line in value),
            *body(len(inner) + 4),
            f"{inner}}}",
        ]
    else:
        lines += [*(f"{inner}{line}" for line in value), *body(len(inner))]
    state.put = saved
    return _copies(pad, 2 * rows * cols, lines)


def emit_fragment_acc(
    name: str, call: ast.Call, parts: list[str], pad: str, state: EmitState
) -> None:
    """Declare a fragment group's accumulator, its ``rows * cols`` float
    fragments, filled with the seed value or from the staged seed tile.
    When SIMD groups split the K loops, only the first starts from the
    seed value; the others add their tiles to 0 (:func:`emit_split_combine`)."""
    grid = keyword(call, "fragment")
    assert grid is not None
    rows, cols, flipped, split = ast.literal_eval(grid)
    # This SIMD group's rows of the tile, set before the kernel's first
    # statement (``metal_jit._fragment_row_split``).
    assert state.row_split == split, (state.row_split, split)
    assert state.row_base == (f" + {SIMDGROUP} * {8 * rows}" if split > 1 else "")
    fill = ast.literal_eval(call.args[7])
    block_m, block_n = (ast.literal_eval(a) for a in call.args[11:ACC_SEED])
    tm, tn = (ast.literal_eval(a) for a in call.args[1:3])
    seed = ast.literal_eval(call.args[ACC_SEED])
    state.declare(name)
    state.coop[name] = name
    state.coop_group[name] = (
        ast.literal_eval(call.args[0]),
        ast.literal_eval(call.args[4]),
    )
    if keyword(call, "fold") is not None:
        # A fold's update reads its product's element at the same slot: both
        # are float fragments of one grid, whose lane layout is fixed.
        state.coop_fold.add(name)
    state.coop_tile[name] = (block_m, block_n)
    state.fragments[name] = (rows, cols, flipped)
    if keyword(call, "by_fragment") is not None:
        state.by_fragment.add(name)
    # The fragments cover the op's M and N, which are cut to the loops'
    # extents and so lie inside the blocks unless a block of 1 grew to 8.
    if tm > block_m or tn > block_n:
        state.coop_rounded.add(name)
    value = float(fill if seed is None else 0.0)
    start = f"{value!r}f"
    if value and state.split > 1:
        start = f"{SIMDGROUP} == 0 ? {start} : 0.0f"
    parts.append(f"{pad}simdgroup_float8x8 {name}[{rows * cols}];")
    parts += [
        f"{pad}{name}[{i}] = make_filled_simdgroup_matrix<float, 8, 8>({start});"
        for i in range(rows * cols)
    ]
    if seed is not None:
        # As ``_emit_coop_acc``'s staged seed: between barriers.
        parts += [
            f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);",
            *fragment_elements(
                name,
                name,
                pad,
                state,
                lambda indent: [
                    f"{' ' * indent}{state.put} = {seed}[_e_m * {block_n} + _e_n];"
                ],
                read=False,
            ),
            f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]


def _run_setup(
    stmt: ast.For, state: EmitState, root: str
) -> tuple[int, list[tuple[ast.For, str, int, tuple[str, str]]], dict[str, Fact], int]:
    """A fragment run's step count, its sides ``(loop, fragment array,
    fragments, position declarations)``, the facts its elements fold
    over and its table cap (``metal_tabulate_max_entries``); ``root`` is its
    accumulator's.  A run in natural K order (a chained side,
    :func:`_chain_refusal`) takes K positions ``8 s`` to ``8 s + 7`` at step
    ``s``; any other the order :func:`emit_fragment_run` describes."""
    assert isinstance(stmt.iter, ast.Call)
    _, steps, facts_arg, tables, _, _, natural, _, _ = stmt.iter.args
    rows, cols, flipped = _grid(state, root)
    row, col = _element_names(flipped)
    a_loop, b_loop = stmt.body
    assert isinstance(a_loop, ast.For) and isinstance(b_loop, ast.For)
    facts = lane_facts(state)
    for name, multiple, least, _ in ast.literal_eval(facts_arg):
        facts[name] = Fact(multiple, least)
    n_steps = ast.literal_eval(steps)
    if ast.literal_eval(natural):
        a_k = "_e_k = 8 * _fs + _fn + (_fi & 1)"
        b_k = "_e_k = 8 * _fs + _fm"
    else:
        a_k = f"_e_k = {n_steps} * (_fn + (_fi & 1)) + _fs"
        b_k = f"_e_k = {n_steps} * _fm + _fs"
    sides = [
        (
            a_loop,
            "_frag_a",
            rows,
            (f"{row} = 8 * (_fi >> 1) + _fm{state.row_base}", a_k),
        ),
        (b_loop, "_frag_b", cols, (b_k, f"{col} = 8 * (_fi >> 1) + _fn + (_fi & 1)")),
    ]
    return n_steps, sides, facts, ast.literal_eval(tables)


def emit_fragment_run(
    stmt: ast.For,
    parts: list[str],
    indent: int,
    state: EmitState,
    emit_body: Callable[
        [list[ast.stmt], list[str], int, EmitState, dict[str, str]], None
    ],
    expr: Callable[[ast.expr], str],
) -> None:
    """A fragment run's loop (:func:`fragment_run`): per MMA step, the A and
    B fragments of every lane's operand elements, or of a tile side's tile
    (:func:`tile_side`, copied first unless its statement list's copies
    were, :func:`tile_copy_groups`), then each accumulator fragment's
    ``simdgroup_multiply_accumulate``.  Lane ``l``'s elements of
    an A fragment are row ``fm``, columns ``j = fn`` and ``fn + 1``; of a B
    fragment, row ``j = fm`` and columns ``fn`` and ``fn + 1``.  Column (row)
    ``j`` of step ``s`` is K position ``steps * j + s`` of the tile: any
    bijection works, because the A and B fragments use the same one, and this
    one gives each lane one run of K positions over the steps (``2 * steps``
    adjacent ones on the A side, ``steps`` on the B side), so a lane reads a
    few words of packed codes and their scales per tile, and adjacent
    elements of an operand row over its steps.

    The steps and the operand loops are unrolled, each in a block of its own
    with its step ``_fs`` and element ``_fi`` constants, and each element's
    index arithmetic is folded (``affine_fold``) over its position, the
    lane's facts, the tile offsets' (codegen's ``facts``) and the kernel's
    integer constants: elements that read one word of packed codes, or one
    scale, then read it through one expression, whose loads the compiler
    shares.  Under a table cap (``metal_tabulate_max_entries``) each folded
    element's chains from a low-bit field read their tables
    (``tabulate``).  In a K loop that loads a tile ahead
    (``range_num_stages=2``), the run reuses the elements the prefetch
    planner built (:func:`prefetch_loads`, through ``state.prefetch``), and
    their early loads read the registers loaded the iteration before.  A
    few-row group's runs (``_Operand.few_rows``, the accumulator's
    ``by_fragment``) order their MMAs fragment by fragment
    (:func:`_emit_by_fragment`) instead of step by step
    (:func:`_emit_steps`)."""
    assert isinstance(stmt.iter, ast.Call)
    acc = stmt.iter.args[0]
    assert isinstance(acc, ast.Name)
    split = _run_split(stmt)
    # Its K loop, which it lies in, splits it (``split_range``).
    assert split is None or split[0] in state.split_loops, split
    rows, cols, flipped = _grid(state, state.coop[acc.id])
    n_steps, sides, facts, tables = _run_setup(stmt, state, state.coop[acc.id])
    copies = [_side_tile(loop) for loop, _, _, _ in sides]
    pending = [
        c
        for c in copies
        if c is not None and cast("ast.Name", c.args[0]).id not in state.tiles_copied
    ]
    if pending:
        parts += emit_tile_copies(pending, " " * indent, state, expr)
    elem_type = ast.literal_eval(stmt.iter.args[7])
    frag_type = (
        "simdgroup_float8x8"
        if elem_type == "float"
        else f"simdgroup_matrix<{elem_type}, 8, 8>"
    )
    k, early, elements = state.prefetch.get(stmt, ("", {}, {}))
    elements = dict(elements)
    for step in range(n_steps):
        for side, (loop, _, count, decls) in enumerate(sides):
            if not loop.body:
                # Hoisted, or loaded from its tile.
                continue
            for i in range(2 * count):
                if (side, step, i) not in elements:
                    elements[side, step, i] = element_body(
                        loop, decls, step, i, facts, tables, state
                    )
    subs = [_chain_subs(loop, side, state) for side, (loop, *_) in enumerate(sides)]

    def element_lines(side: int, step: int, i: int, at: str) -> list[str]:
        """The block, at indent ``at``, that writes lane element ``i`` of a
        ``side`` fragment at ``step``: its constant and declarations, then
        its statements, which in a K loop that loads a tile ahead read the
        registers loaded the iteration before."""
        decls = sides[side][3]
        body, env, element_facts = elements[side, step, i]
        if early:
            body = _read_early(body, env, element_facts, k, early)
        lines = [
            f"{at}{{",
            f"{at}    constexpr short _fi = {i};",
            *(f"{at}    int {decl};" for decl in decls),
        ]
        with state.isolated_scope():
            emit_body(body, lines, len(at) + 4, state, subs[side])
        return [*lines, f"{at}}}"]

    # Each side's fragment ``i`` of step ``s``: built at the step, or
    # hoisted, built before the loop (``codegen_fragment_hoist``).
    names: list[Callable[[int, int], str]] = []
    for loop, array, count, _ in sides:
        hoisted = _hoisted(loop)
        if hoisted is None:
            names.append(lambda s, i, a=array: f"{a}[{i}]")
        else:
            names.append(lambda s, i, h=hoisted, c=count: f"{h}[{s * c + i}]")
    run = _Run(
        acc.id,
        frag_type,
        rows,
        cols,
        n_steps,
        sides,
        copies,
        names,
        int(flipped),
        element_lines,
    )
    saved = state.put
    if state.coop[acc.id] in state.by_fragment and not any(
        c is not None or _hoisted(loop) is not None
        for c, (loop, *_) in zip(copies, sides, strict=True)
    ):
        assert not early, "the planner keeps a prefetching K loop's x computed"
        _emit_by_fragment(run, parts, indent, state)
    else:
        _emit_steps(run, parts, indent, state)
    state.put = saved


def _hoisted(loop: ast.For) -> str | None:
    """The array a hoisted run side's fragments were built in before the
    loop (``codegen_fragment_hoist``), None for a side built at its steps."""
    assert isinstance(loop.iter, ast.Call)
    return ast.literal_eval(loop.iter.args[1])


@dataclasses.dataclass
class _Run:
    """A fragment run as :func:`emit_fragment_run` emits it: its
    accumulator, the fragments' ``simdgroup_matrix`` type, the grid's rows
    and columns, its MMA steps, its sides (:func:`_run_setup`), each side's
    tile copy or None (:func:`_side_tile`), each side's fragment ``i`` of
    step ``s`` (``names``), the side the A fragments come from
    (``flipped``), and the block that writes a lane's element of a fragment
    at a side, step, element and indent (``element_lines``)."""

    acc: str
    frag_type: str
    rows: int
    cols: int
    n_steps: int
    sides: list[tuple[ast.For, str, int, tuple[str, str]]]
    copies: list[ast.Call | None]
    names: list[Callable[[int, int], str]]
    flipped: int
    element_lines: Callable[[int, int, int, str], list[str]]


def _emit_steps(run: _Run, parts: list[str], indent: int, state: EmitState) -> None:
    """A run's steps in turn: each declares the fragments its sides build at
    the step, loads a tile side's from its tile (:func:`_tile_loads`) and
    computes the others', then runs its MMAs (:func:`emit_fragment_run`)."""
    pad = " " * indent
    inner = pad + "    "
    for step in range(run.n_steps):
        parts += [f"{pad}{{", f"{inner}constexpr short _fs = {step};"]
        for loop, array, count, _ in run.sides:
            if _hoisted(loop) is None:
                parts.append(f"{inner}{run.frag_type} {array}[{count}];")
        # ``run.sides`` hold the A fragments, then the B fragments; the A
        # fragments come from the rhs in a transposed run (``flipped``).
        for slot, (copy_call, (_, array, count, _)) in enumerate(
            zip(run.copies, run.sides, strict=True)
        ):
            if copy_call is not None:
                side = slot ^ run.flipped
                parts += [
                    f"{inner}{line}"
                    for line in _tile_loads(
                        copy_call, side, run.flipped, step, array, count, state
                    )
                ]
        for side, (loop, array, count, _) in enumerate(run.sides):
            if not loop.body:
                # Hoisted, or loaded from its tile.
                continue
            state.put = f"{array}{FRAGMENT_SLOT}"
            for i in range(2 * count):
                parts += run.element_lines(side, step, i, inner)
        parts += [
            f"{inner}simdgroup_multiply_accumulate({run.acc}[{a * run.cols + b}], "
            f"{run.names[0](step, a)}, {run.names[1](step, b)}, "
            f"{run.acc}[{a * run.cols + b}]);"
            for a in range(run.rows)
            for b in range(run.cols)
        ]
        parts.append(f"{pad}}}")


def _emit_by_fragment(
    run: _Run, parts: list[str], indent: int, state: EmitState
) -> None:
    """A few-row run's MMAs fragment by fragment, as MLX's few-row kernel
    orders them: first the B fragments (the activation's) of every step,
    then per A fragment (a weight row's) that fragment at each step,
    computed and followed by its MMAs, so the compiler overlaps one
    fragment's decode with the MMAs before it.  Each accumulator adds its
    steps in the order :func:`_emit_steps` does.  A run with a hoisted or
    tile side takes the step order."""
    acc, frag_type, rows, cols = run.acc, run.frag_type, run.rows, run.cols
    element_lines = run.element_lines
    pad = " " * indent
    inner = pad + "    "
    parts.append(f"{pad}{{")
    for step in range(run.n_steps):
        parts += [
            f"{inner}{frag_type} _frag_b{step}[{cols}];",
            f"{inner}{{",
            f"{inner}    constexpr short _fs = {step};",
        ]
        state.put = f"_frag_b{step}{FRAGMENT_SLOT}"
        for i in range(2 * cols):
            parts += element_lines(1, step, i, inner + "    ")
        parts.append(f"{inner}}}")
    for a in range(rows):
        for step in range(run.n_steps):
            parts += [
                f"{inner}{{",
                f"{inner}    constexpr short _fs = {step};",
                f"{inner}    {frag_type} _frag_a[{rows}];",
            ]
            state.put = f"_frag_a{FRAGMENT_SLOT}"
            for i in (2 * a, 2 * a + 1):
                parts += element_lines(0, step, i, inner + "    ")
            parts += [
                f"{inner}    simdgroup_multiply_accumulate({acc}[{a * cols + b}], "
                f"_frag_a[{a}], _frag_b{step}[{b}], {acc}[{a * cols + b}]);"
                for b in range(cols)
            ]
            parts.append(f"{inner}}}")
    parts.append(f"{pad}}}")


def _side_tile(loop: ast.For) -> ast.Call | None:
    """A run side's tile copy (:func:`fragment_run`), None for a side that
    builds its fragments otherwise."""
    assert isinstance(loop.iter, ast.Call)
    copy_call = loop.iter.args[2]
    if isinstance(copy_call, ast.Constant):
        return None
    assert isinstance(copy_call, ast.Call)
    return copy_call


def _tile_loads(
    call: ast.Call,
    side: int,
    flipped: int,
    step: int,
    array: str,
    count: int,
    state: EmitState,
) -> list[str]:
    """A tile side's ``simdgroup_load`` of each of its ``count`` fragments of
    MMA step ``step`` (K positions ``8 step`` to ``8 step + 7``, the run's
    natural order) into ``array``, for the operand on ``side`` (0 the lhs,
    1 the rhs).  The operand's two dims are the op's (M, K) on the lhs and
    (K, N) on the rhs; its A fragments (the side
    ``flipped`` names) run over the grid rows, this SIMD group's share
    (``EmitState.row_base``), its B fragments over the grid columns.  A
    fragment whose rows are the tile's columns loads transposed."""
    buf = cast("ast.Name", call.args[0]).id
    _, outer, pitch, _, _, widen = ast.literal_eval(call.args[1])
    k_dim = 1 - side
    role_a = side == flipped
    lines = []
    for i in range(count):
        at = [""] * 2
        at[k_dim] = f"{8 * step}"
        at[1 - k_dim] = f"{8 * i}{state.row_base if role_a else ''}"
        rows = 1 - k_dim if role_a else k_dim
        first, second = (rows, 1 - rows) if rows == outer else (1 - rows, rows)
        offset = f"({at[first]}) * {pitch} + {at[second]}"
        transpose = "false" if rows == outer else "true"
        load = f"{buf} + {offset}, {pitch}, ulong2(0, 0), {transpose});"
        if widen is None:
            lines.append(f"simdgroup_load({array}[{i}], {load}")
            continue
        # A widened view's tile (``tile_view``): loaded in its own type, each
        # lane's two elements widened (all fragment types share the lanes).
        lines += [
            "{",
            f"    simdgroup_matrix<{widen}, 8, 8> _frag_w;",
            f"    simdgroup_load(_frag_w, {load}",
            f"    {array}[{i}].thread_elements()[0] = _frag_w.thread_elements()[0];",
            f"    {array}[{i}].thread_elements()[1] = _frag_w.thread_elements()[1];",
            "}",
        ]
    return lines


def tile_copy_groups(stmts: list[ast.stmt]) -> dict[int, list[ast.Call]]:
    """The tile copies (:func:`tile_side`) of the fragment runs among
    ``stmts``, grouped at the first run of each group: a run's copy joins
    the group before it unless it reads a name a statement since that
    group's first run assigns.  Each group's copies run together between
    one barrier pair (:func:`emit_tile_copies`), each into its own tile, so
    a loop body whose runs read the same tensors' tiles (attention's K and
    V) waits once per iteration.  A view reads no tensor the kernel writes
    (``mpp_matmul._order_memory``), so moving its copy up reads the same
    values."""
    groups: dict[int, list[ast.Call]] = {}
    first: int | None = None
    assigned: set[str] = set()
    for index, stmt in enumerate(stmts):
        copies: list[ast.Call] = []
        if isinstance(stmt, ast.For) and call_name(stmt.iter) == FRAG_RUN:
            for loop in stmt.body:
                assert isinstance(loop, ast.For)
                if (copy_call := _side_tile(loop)) is not None:
                    copies.append(copy_call)
        if copies:
            reads = {
                n.id for c in copies for n in ast.walk(c) if isinstance(n, ast.Name)
            }
            if first is None or reads & assigned:
                first, assigned = index, set()
            groups.setdefault(first, []).extend(copies)
        if first is not None:
            assigned |= {
                n.id
                for n in ast.walk(stmt)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
    return groups


#: The raw-bits type a tile copy moves ``(element bytes, vector)`` as, and
#: its zero.
_TILE_BITS = {
    (2, 1): ("ushort", "0"),
    (2, 4): ("packed_ushort4", "packed_ushort4(0, 0, 0, 0)"),
    (4, 1): ("uint", "0u"),
    (4, 2): ("packed_uint2", "packed_uint2(0u, 0u)"),
}


def _literal(value: tuple[object, ...]) -> ast.expr:
    """``value`` as the expression ``ast.literal_eval`` reads it back from."""
    return ast.parse(repr(value), mode="eval").body


def emit_tile_copies(
    calls: list[ast.Call], pad: str, state: EmitState, expr: Callable[[ast.expr], str]
) -> list[str]:
    """Tile copies (:func:`emit_tile_copy`) between a barrier that keeps
    their writes after every SIMD group's reads of the tiles' previous
    contents and one that keeps the runs' loads after the writes."""
    barrier = f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);"
    lines = [barrier]
    for call in calls:
        lines += emit_tile_copy(call, pad, state, expr)
        state.tiles_copied.add(cast("ast.Name", call.args[0]).id)
    return [*lines, barrier]


def emit_tile_copy(
    call: ast.Call, pad: str, state: EmitState, expr: Callable[[ast.expr], str]
) -> list[str]:
    """Every thread of the threadgroup copies a tile side's view (the op's
    two dims from their origins, ``mpp_matmul.view_args``) into its tile as
    raw bits, ``vector`` elements at a time, in tile-row order, so
    consecutive threads read consecutive addresses; an element past a
    masked or clamped dim's extent (a clamped one's value is free:
    :func:`_dim_mode`), or where an outer index lies outside its dim, is
    written as 0.  A pipelined tile (:func:`pipeline_tile_copies`) is
    written from the registers its copy loaded the iteration before."""
    copy = _TileCopy(call, state, expr)
    regs = state.tile_regs.get(copy.buf)
    if regs is not None:
        return copy.unrolled(pad, [f"{copy.dst} = {reg};" for reg in regs])
    inner = pad + "    "
    lines = [
        f"{pad}for (uint _c = {copy.thread}; _c < {copy.total}; _c += {copy.threads}) {{",
        *copy.position(inner),
        *copy.assign(inner, copy.dst),
    ]
    return [*lines, f"{pad}}}"]


class _TileCopy:
    """A tile copy's pieces (:func:`emit_tile_copy`): the tile, its chunks
    (``vector`` elements each, ``total`` in all, the threads taking chunks
    ``thread``, ``thread + threads``, ...), each chunk's tile and device
    addresses at ``_tr`` and ``_tc`` and the terms that keep it in bounds."""

    def __init__(
        self, call: ast.Call, state: EmitState, expr: Callable[[ast.expr], str]
    ) -> None:
        self.buf = cast("ast.Name", call.args[0]).id
        shape, outer, pitch, vector, element_bytes, _ = ast.literal_eval(call.args[1])
        tensor, *dims = call.args[2:11]
        keywords = {k.arg: k.value for k in call.keywords}
        masks = ast.literal_eval(keywords["masks"])
        wide = ast.literal_eval(keywords["wide"])
        outer_dims = cast("ast.List", keywords["outer"]).elts
        inner = 1 - outer
        self.chunks = shape[inner] // vector
        self.vector = vector
        self.total = shape[outer] * self.chunks
        position = {outer: "_tr", inner: "_tc"}
        offset: list[str] = []
        self.terms: list[str] = []
        for d in range(2):
            o, e, size, stride = (expr(x) for x in dims[4 * d : 4 * d + 4])
            at = f"({o} + {position[d]})"
            offset.append(f"{f'static_cast<long>({at})' if wide else at} * {stride}")
            if masks[d] != _UNMASKED:
                self.terms.append(f"{at} < {e}")
                if e != size:
                    self.terms.append(f"{at} < {size}")
        for item in outer_dims:
            x, size, stride = (expr(v) for v in cast("ast.Tuple", item).elts)
            offset.append(f"{f'static_cast<long>({x})' if wide else x} * {stride}")
            self.terms.append(f"0 <= {x} && {x} < {size}")
        self.threads = SIMD_WIDTH * state.row_split
        self.thread = (
            LANE if state.row_split == 1 else f"{SIMDGROUP} * {SIMD_WIDTH} + {LANE}"
        )
        self.bits, self.zero = _TILE_BITS[element_bytes, vector]
        self.dst = (
            f"*reinterpret_cast<threadgroup {self.bits}*>"
            f"({self.buf} + _tr * {pitch} + _tc)"
        )
        self.src = (
            f"*reinterpret_cast<device const {self.bits}*>({expr(tensor)} + "
            f"{' + '.join(offset)})"
        )

    def position(self, pad: str) -> list[str]:
        """The chunk ``_c``'s tile row and column."""
        return [
            f"{pad}int _tr = int(_c / {self.chunks});",
            f"{pad}int _tc = int(_c % {self.chunks}) * {self.vector};",
        ]

    def assign(self, pad: str, target: str) -> list[str]:
        """``target`` set to the chunk's device bits, or 0 out of bounds."""
        if not self.terms:
            return [f"{pad}{target} = {self.src};"]
        return [
            f"{pad}if ({' && '.join(self.terms)}) {{",
            f"{pad}    {target} = {self.src};",
            f"{pad}}} else {{",
            f"{pad}    {target} = {self.zero};",
            f"{pad}}}",
        ]

    @property
    def per_thread(self) -> int:
        """The chunks a thread takes at most."""
        return -(-self.total // self.threads)

    def unrolled(self, pad: str, bodies: list[list[str]] | list[str]) -> list[str]:
        """Each of a thread's chunks in turn, its ``_c``, ``_tr`` and ``_tc``
        bound around its body (a line, or lines); a chunk past the tile's
        last runs nothing."""
        lines: list[str] = []
        inner = pad + "    "
        for j, body in enumerate(bodies):
            body_lines = [body] if isinstance(body, str) else body
            lines += [
                f"{pad}{{",
                f"{inner}uint _c = {self.thread} + {j * self.threads};",
            ]
            guard = self.total % self.threads != 0
            at = inner + ("    " if guard else "")
            if guard:
                lines.append(f"{inner}if (_c < {self.total}) {{")
            lines += self.position(at) + [f"{at}{line}" for line in body_lines]
            if guard:
                lines.append(f"{inner}}}")
            lines.append(f"{pad}}}")
        return lines


def pipeline_tile_copies(
    body: list[ast.stmt],
    loop_var: str,
    start: str,
    end: str,
    step: str,
    pad: str,
    state: EmitState,
    expr: Callable[[ast.expr], str],
) -> tuple[list[str], list[str]]:
    """Software pipelining of a K loop's tile copies (a fragment run's
    ``pipe``, ``range_num_stages=2`` on a run with tile sides): the lines
    to emit before the loop and at the top of its body.  Each thread loads
    its chunks of the first tile into registers before the loop (when the
    loop runs), and at the top of each iteration copies them out and loads
    the next tile's, so those loads are in flight across this tile's
    barriers and multiplies; the copy writes the tile from the registers
    (:func:`emit_tile_copy`).  A copy that reads a name the loop body
    assigns (a view origin computed in the loop) stays unpipelined.  The
    loop pipelines when a run whose K it is takes ``pipe``."""
    assigned = {
        n.id
        for s in body
        for n in ast.walk(s)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    before: list[str] = []
    top: list[str] = []
    ahead: list[str] = []
    runs = [
        stmt
        for stmt in body
        if isinstance(stmt, ast.For) and call_name(stmt.iter) == FRAG_RUN
    ]
    # The loop's stages (its slot) reach the runs whose K it is; a run over
    # another K (a ``:`` K, or one this loop is N of) copies its tiles in
    # the same loop body, so the loop pipelines every copy in it.
    if not any(ast.literal_eval(cast("ast.Call", r.iter).args[8]) for r in runs):
        return before, top
    for stmt in runs:
        for loop in stmt.body:
            assert isinstance(loop, ast.For)
            call = _side_tile(loop)
            if call is None:
                continue
            buf = cast("ast.Name", call.args[0]).id
            reads = {
                n.id
                for a in [*call.args[2:], *(k.value for k in call.keywords)]
                for n in ast.walk(a)
                if isinstance(n, ast.Name)
            }
            if buf in state.tile_regs or (reads & assigned) - {loop_var}:
                continue
            copy = _TileCopy(call, state, expr)
            regs = [f"_pipe{j}{buf}" for j in range(copy.per_thread)]
            state.tile_regs[buf] = regs
            loads = copy.unrolled(
                "    ", [copy.assign("", f"{reg}_next") for reg in regs]
            )
            before += [f"{pad}{copy.bits} {reg}_next = {copy.zero};" for reg in regs]
            before += [
                f"{pad}if ({start} < {end}) {{",
                f"{pad}    const int {loop_var} = {start};",
            ]
            before += [f"{pad}{line}" for line in loads] + [f"{pad}}}"]
            top += [f"{pad}    {copy.bits} {reg} = {reg}_next;" for reg in regs]
            ahead += [f"{pad}    {line}" for line in loads]
    if ahead:
        top += [
            f"{pad}    if ({loop_var} + {step} < {end}) {{",
            f"{pad}        const int _pipe_next = {loop_var} + {step};",
            f"{pad}        {{",
            f"{pad}            const int {loop_var} = _pipe_next;",
            *(f"{pad}        {line}" for line in ahead),
            f"{pad}        }}",
            f"{pad}    }}",
        ]
    return before, top


def _chain_subs(loop: ast.For, side: int, state: EmitState) -> dict[str, str]:
    """The names a chained side's elements read in place (an A side's
    element ``_fi`` of step ``_fs`` is its source's element at row fragment
    ``_fi >> 1`` and column fragment ``_fs``, a B side's at row fragment
    ``_fs`` and column fragment ``_fi >> 1``): the source result's names,
    its row values (``fragment_lines``) at their row slot and its held
    values at the element's slot; none for another side."""
    assert isinstance(loop.iter, ast.Call)
    chain = ast.literal_eval(loop.iter.args[0])
    if chain is None:
        return {}
    root = state.coop[chain]
    _, cols, flipped = _grid(state, root)
    assert not flipped
    row, col = ("(_fi >> 1)", "_fs") if side == 0 else ("_fs", "(_fi >> 1)")
    slot = f"({row} * {cols} + {col})"
    subs = {
        name: f"{root}[{slot}].thread_elements()[_fi & 1]"
        for name, r in state.coop.items()
        if r == root
    }
    subs |= {name: f"{name}[{row}]" for name in state.lines}
    subs |= {
        name: f"{name}[2 * {slot} + (_fi & 1)]"
        for name, r in state.held.items()
        if r == root
    }
    return subs


def emit_fragment_hoist(
    stmt: ast.For,
    parts: list[str],
    indent: int,
    state: EmitState,
    emit_body: Callable[
        [list[ast.stmt], list[str], int, EmitState, dict[str, str]], None
    ],
) -> None:
    """A hoisted operand's loop (``mpp_matmul.elementize`` of
    :func:`codegen_fragment_hoist`'s marker): its fragments for every MMA
    step, declared before the loop that runs the op, each element computed
    by the loop's body at its position, as its run would compute it at the
    step, with the same position declarations (:func:`_run_setup`)."""
    pad = " " * indent
    inner = pad + "    "
    assert isinstance(stmt.iter, ast.Call)
    name, role, position, n_steps, count, natural, element = (
        ast.literal_eval(a) for a in stmt.iter.args
    )
    frag_type = (
        "simdgroup_float8x8"
        if element == "float"
        else f"simdgroup_matrix<{element}, 8, 8>"
    )
    if role == 0:
        k_decl = (
            "_e_k = 8 * _fs + _fn + (_fi & 1)"
            if natural
            else f"_e_k = {n_steps} * (_fn + (_fi & 1)) + _fs"
        )
        decls = (f"{position} = 8 * (_fi >> 1) + _fm{state.row_base}", k_decl)
    else:
        k_decl = "_e_k = 8 * _fs + _fm" if natural else f"_e_k = {n_steps} * _fm + _fs"
        decls = (k_decl, f"{position} = 8 * (_fi >> 1) + _fn + (_fi & 1)")
    parts.append(f"{pad}{frag_type} {name}[{n_steps * count}];")
    facts = lane_facts(state)
    saved = state.put
    for step in range(n_steps):
        for i in range(2 * count):
            state.put = f"{name}[{step * count + (i >> 1)}].thread_elements()[{i & 1}]"
            lines = [
                f"{pad}{{",
                f"{inner}constexpr short _fs = {step};",
                f"{inner}constexpr short _fi = {i};",
                *(f"{inner}int {decl};" for decl in decls),
            ]
            body, _, _ = element_body(stmt, decls, step, i, facts, 0, state)
            with state.isolated_scope():
                emit_body(body, lines, indent + 4, state, {})
            parts += [*lines, f"{pad}}}"]
    state.put = saved


#: A load a fragment run reads a K tile early: its tensor, its index without
#: the K offset's term, and that term's coefficient.
EarlyLoad = tuple[str, Affine, Fraction]


def _assigned_from(body: list[ast.stmt]) -> list[set[str]]:
    """For each statement of ``body``, the names it or a later statement
    assigns (or binds).  ``env`` holds a name's last value, which a load in
    such a statement may not read."""
    out: list[set[str]] = []
    names: set[str] = set()
    for stmt in reversed(body):
        names = names.union(*(assigned_names(n) for n in ast.walk(stmt)))
        out.append(names)
    return out[::-1]


def _early_load(
    node: ast.AST,
    env: Mapping[str, object],
    facts: Mapping[str, Fact],
    k: str,
    unstable: set[str],
) -> EarlyLoad | None:
    """``node``'s key when it is an unmasked load whose index is affine in
    the K offset ``k`` and names the K loop doesn't change: the lane's,
    the other tiles' offsets and floor terms of those.  It reads none of
    the names ``unstable`` (:func:`_assigned_from`), whose values in ``env``
    may not be the ones it reads."""
    load = tl_load(node)
    if not (
        load is not None
        and len(load.args) >= 1
        and load_mask_other(load) is None
        and isinstance(ptr := load.args[0], ast.BinOp)
        and isinstance(ptr.op, ast.Add)
        and isinstance(ptr.left, ast.Name)
    ):
        return None
    if any(isinstance(n, ast.Name) and n.id in unstable for n in ast.walk(ptr.right)):
        return None
    facts = dict(facts)
    index = evaluate(ptr.right, env, facts)
    if index is None:
        return None
    terms = dict(index.terms)
    coeff = terms.pop(k, Fraction(0))
    # Every term is a fact's name: a lane coordinate, a tile offset or a
    # floor term of those.  Of these, only the K offset and floor terms
    # over it change in the K loop.
    if not coeff or any(k in facts[name].depends for name in terms):
        return None
    return ptr.left.id, Affine(index.const, tuple(sorted(terms.items()))), coeff


def _read_early(
    body: list[ast.stmt],
    env: Mapping[str, object],
    facts: Mapping[str, Fact],
    k: str,
    early: Mapping[EarlyLoad, str],
) -> list[ast.stmt]:
    """``body`` with each load the K loop issued a tile early
    (:func:`prefetch_loads`) read from the register that holds it."""

    class Rewrite(ast.NodeTransformer):
        def __init__(self, unstable: set[str]) -> None:
            self.unstable = unstable

        def visit_Call(self, node: ast.Call) -> ast.AST:
            key = _early_load(node, env, facts, k, self.unstable)
            name = None if key is None else early.get(key)
            if name is not None:
                return ast.Name(name, ast.Load())
            return self.generic_visit(node)

    return [
        Rewrite(unstable).visit(copy.deepcopy(s))
        for s, unstable in zip(body, _assigned_from(body), strict=True)
    ]


def prefetch_loads(
    body: list[ast.stmt], loop_var: str, end: str, pad: str, state: EmitState
) -> tuple[list[str], list[str]]:
    """Software pipelining of the fragment runs in a K loop's ``body``: the
    lines to emit before the loop and at the top of its body.  A run's
    element loads that the loop may issue a tile early (:func:`_early_load`)
    are each loaded for the first tile before the loop into a register,
    and at the top of each iteration copied out and loaded for the next
    tile, so the next tile's loads are in flight while this tile decodes and
    multiplies.  The run's elements read the copies (:func:`_read_early`).
    No such load reads a tensor the kernel writes: the planner stages a view
    of one (``mpp_matmul._order_memory``), and ``check_fragment_group``
    refuses a staged operand's loads of one."""
    before: list[str] = []
    top: list[str] = []
    ahead: list[str] = []
    # The loop's test for a next tile, once a run loads ahead.
    next_tile: str | None = None
    # The loop body's references to an accumulator (``acc_copy = acc``),
    # which it declares before its runs.
    alias = {
        s.targets[0].id: s.value.id
        for s in body
        if isinstance(s, ast.Assign)
        and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Name)
        and isinstance(s.value, ast.Name)
    }
    for stmt in body:
        if not (isinstance(stmt, ast.For) and call_name(stmt.iter) == FRAG_RUN):
            continue
        assert isinstance(stmt.iter, ast.Call)
        k_step = ast.literal_eval(stmt.iter.args[4])
        if k_step is None or k_step[0] != loop_var:
            continue
        k, step = k_step
        split = _run_split(stmt)
        # SIMD groups that split the loop each start at their own tile and
        # step over the others'.
        simdgroups = 1 if split is None else split[1]
        acc = stmt.iter.args[0]
        assert isinstance(acc, ast.Name)
        root = acc.id
        while root not in state.coop and root in alias:
            root = alias[root]
        if root not in state.coop:
            # An accumulator the loop body declares: a run per tile.
            continue
        n_steps, sides, facts, tables = _run_setup(stmt, state, state.coop[root])
        begin = facts[k].lo
        assert begin is not None
        early: dict[EarlyLoad, str] = {}
        # The elements built here, which the run's emission reuses.
        elements: dict[tuple[int, int, int], ElementBody] = {}
        for s in range(n_steps):
            for side, (loop, _, count, decls) in enumerate(sides):
                for i in range(2 * count):
                    element = element_body(loop, decls, s, i, facts, tables, state)
                    elements[side, s, i] = element
                    stmts, env, element_facts = element
                    for line, unstable in zip(
                        stmts, _assigned_from(stmts), strict=True
                    ):
                        for node in ast.walk(line):
                            key = _early_load(node, env, element_facts, k, unstable)
                            if key is not None and key not in early:
                                early[key] = f"_early_{state.early_count + len(early)}"
        # The emission reuses the elements whether or not any load moves.
        state.prefetch[stmt] = (k, early, elements)
        if not early:
            continue
        next_tile = f"{loop_var} + {step * simdgroups} < {end}"
        state.early_count += len(early)
        for (ptr, rest, coeff), name in early.items():
            first = rest + Affine.of(int(coeff * begin))
            if simdgroups > 1:
                # The index is an integer at every tile, so ``coeff * step``
                # is one.
                first += Affine.term(SIMDGROUP, int(coeff * step))
            nxt = (
                rest
                + Affine.term(k, 1).scale(coeff)
                + Affine.of(int(coeff * step * simdgroups))
            )
            before.append(
                f"{pad}auto {name}_next = {ptr}[{ast.unparse(render(first))}];"
            )
            top.append(f"{pad}    auto {name} = {name}_next;")
            ahead.append(
                f"{pad}        {name}_next = {ptr}[{ast.unparse(render(nxt))}];"
            )
    if next_tile is not None:
        top += [f"{pad}    if ({next_tile}) {{", *ahead, f"{pad}    }}"]
    return before, top


def _run_split(stmt: ast.For) -> tuple[str, int] | None:
    """A fragment run loop's ``split`` (:func:`fragment_run`): its K loop's
    offset and the SIMD groups that split the loop, None for one."""
    assert isinstance(stmt.iter, ast.Call)
    return ast.literal_eval(stmt.iter.args[5])


def split_runs(body: list[ast.stmt], loop_var: str) -> list[ast.For]:
    """The fragment runs in a K loop's ``body`` that SIMD groups split over
    the loop (``loop_var``): their ``split`` names its offset."""
    runs: list[ast.For] = []
    for stmt in body:
        if isinstance(stmt, ast.For) and call_name(stmt.iter) == FRAG_RUN:
            assert isinstance(stmt.iter, ast.Call)
            split = _run_split(stmt)
            if split is not None and split[0] == loop_var:
                runs.append(stmt)
    return runs


def emit_split_combine(
    runs: list[ast.For], parts: list[str], pad: str, state: EmitState
) -> None:
    """After a K loop the SIMD groups split, add their accumulators (those
    the loop's ``runs`` update) into the first's (``emit_split_adds``): a
    lane's two elements of each fragment, each fragment then refilled with
    0."""
    fragments: list[str] = []
    for run in runs:
        assert isinstance(run.iter, ast.Call)
        acc = run.iter.args[0]
        assert isinstance(acc, ast.Name)
        root = state.coop[acc.id]
        rows, cols, _ = _grid(state, root)
        held = [f"{root}[{i}]" for i in range(rows * cols)]
        if held[0] not in fragments:
            fragments += held
    count = len(fragments)
    lanes = 2 * SIMD_WIDTH
    state.split_slots = max(state.split_slots, (state.split - 1) * count * lanes)

    def slot(simdgroup: str, i: int) -> str:
        f, e = divmod(i, 2)
        return f"{SPLIT_BUFFER}[({simdgroup} * {count} + {f}) * {lanes} + 2 * {LANE} + {e}]"

    emit_split_adds(
        [
            f"{fragment}.thread_elements()[{e}]"
            for fragment in fragments
            for e in range(2)
        ],
        [
            f"{fragment} = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
            for fragment in fragments
        ],
        slot,
        state.split,
        parts,
        pad,
    )


def fragment_acc_keywords(
    cg: GenerateAST, group: MppGroup, m: int, n: int, k: int
) -> str:
    """The keywords of a fragment group's ``_helion_mpp_acc``: its grid
    (:func:`fragment_grid`) and the SIMD groups splitting its rows, and ``by_fragment=True`` for a few-row group
    (``_Operand.few_rows``)."""
    grid = fragment_grid(group, m, n, k)
    # A few-row group's runs order their MMAs fragment by fragment
    # (:func:`_emit_by_fragment`): the case that order was measured in.  A
    # transposed group whose activation is already a float32 view keeps the
    # step order, though the overlap the order buys doesn't depend on it.
    by_fragment = ", by_fragment=True" if group.lhs.few_rows else ""
    rows, cols, flipped = grid
    return (
        f", fragment={(rows, cols, flipped, current_plan().row_split)!r}{by_fragment}"
    )


def plan_fragment_lines(plan: MppPlan, graphs: Graphs) -> None:
    """``fragment_lines.plan_lines``: the plan's row values."""
    # Imported here: fragment_lines builds on this module.
    from .fragment_lines import plan_lines

    plan_lines(plan, graphs)


def plan_fragment_holds(plan: MppPlan, graphs: Graphs) -> None:
    """``fragment_lines.plan_holds``: the element values to compute once."""
    from .fragment_lines import plan_holds

    plan_holds(plan, graphs)


@dataclasses.dataclass(frozen=True)
class FragmentLowering:
    """What :mod:`mpp_matmul` calls to lower a fragment group: the planner's
    row values (``fragment_lines``) and check, the accumulator's keywords,
    the op's codegen and its loops in ``elementize``.
    ``MetalBackend.pre_codegen`` gives it to the planner and ``metal_jit``
    to ``elementize``, so :mod:`mpp_matmul`, which this module builds on,
    doesn't import it.  ``hoist`` is the config's ``metal_fragment_hoist``:
    the planner hoists loop-invariant operands
    (:func:`codegen_fragment_hoist`)."""

    hoist: bool = False

    codegen_hoist = staticmethod(codegen_fragment_hoist)
    plan_lines = staticmethod(plan_fragment_lines)
    plan_holds = staticmethod(plan_fragment_holds)
    check_group = staticmethod(check_fragment_group)
    acc_keywords = staticmethod(fragment_acc_keywords)
    codegen_run = staticmethod(codegen_fragment_run)
    run = staticmethod(fragment_run)
    run_binds = staticmethod(fragment_run_binds)


FRAGMENT_LOWERING = FragmentLowering()
