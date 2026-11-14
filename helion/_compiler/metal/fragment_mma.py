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
  carries each operand as a device view (masked loads at the element) or as
  the name of its value, cast to the operand's dtype as staging would; and
- :func:`fragment_run` (``elementize``) replaces the op with its loops: an
  operand element of a computed operand re-runs the Scalar statements that
  compute it, with the thread and lanes that hold it in the Scalar layout
  bound, by the element loops' rules, now over the operand's K as well.

The accumulator is transposed (``C^T = B^T A^T``) when only the rhs is
computed, so the computed operand is the MMA's A, whose lane elements lie
in one row; over a tile's steps a lane's are one run of K positions
(:func:`emit_fragment_run`), a few words and scales of a quantized weight.
The planner admits a group when the SIMD group can recompute each computed
operand element from loads, iotas and element-wise ops alone
(:func:`check_fragment_group`); one SIMD group per threadgroup
(``check_mpp_launch``) and multiples of 8 (:func:`fragment_grid`).
"""

from __future__ import annotations

import ast
import collections
import copy
from fractions import Fraction
import math
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

import sympy
import torch

from ...language import _tracing_ops
from ...language.memory_ops import load
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from .affine_fold import Affine
from .affine_fold import Fact
from .affine_fold import evaluate
from .affine_fold import fold
from .affine_fold import render
from .backend import static_extent
from .lane_fold import load_mask_other
from .lane_fold import tl_load
from .mpp_matmul import ACC_SEED
from .mpp_matmul import FRAG_A
from .mpp_matmul import FRAG_B
from .mpp_matmul import FRAG_RUN
from .mpp_matmul import PUT
from .mpp_matmul import RUN
from .mpp_matmul import Dim
from .mpp_matmul import Graphs
from .mpp_matmul import MppGroup
from .mpp_matmul import MppPlan
from .mpp_matmul import Recomputable
from .mpp_matmul import View
from .mpp_matmul import assigned_names
from .mpp_matmul import block_id_of
from .mpp_matmul import call_name
from .mpp_matmul import current_plan
from .mpp_matmul import group_sides
from .mpp_matmul import matmul_operands
from .mpp_matmul import metal_unsupported
from .mpp_matmul import names_read
from .mpp_matmul import op_dims
from .mpp_matmul import run_outer
from .mpp_matmul import static_tile
from .mpp_matmul import storage_key
from .mpp_matmul import view_args
from .mpp_matmul import virtual_thread
from .search_surface import fragment_stages
from .search_surface import metal_tabulate_max_entries
from .tabulate import tabulate

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from torch.fx.node import Node

    from ..generate_ast import GenerateAST
    from ..tile_strategy import LoopDimInfo
    from .msl_ast_walker import EmitState

#: Ops a computed operand's element re-runs as they are, besides the
#: element-wise ones: a permute, an alias and a same-width bitcast keep each
#: element on the thread that holds it, and a mask fill is a select on the
#: recomputed tile masks.
_ELEMENT_MOVES = (
    torch.ops.aten.permute.default,
    torch.ops.aten.alias.default,
    torch.ops.aten.view.dtype,
    _tracing_ops._mask_to,
)

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


class _OperandRecomputable(Recomputable):
    """Whether a computed operand's value can be computed at any element of
    its tile, over its M or N block and K, by any lane."""

    def __init__(self, group: MppGroup, graphs: Graphs, blocks: set[int]) -> None:
        super().__init__(group, graphs)
        self._operand_blocks = blocks

    def blocks(self) -> set[int]:
        return self._operand_blocks

    def _compute(self, node: Node, val: object) -> bool:
        if node.target not in _ELEMENT_MOVES:
            return super()._compute(node, val)
        return all(map(self, node.all_input_nodes)) and (
            not isinstance(val, torch.Tensor)
            or all(s == 1 or block_id_of(s) in self.blocks() for s in val.size())
        )


def check_fragment_group(group: MppGroup, plan: MppPlan, graphs: Graphs) -> None:
    """Raise ``BackendUnsupported`` unless the fragment run lowers the group:
    a K loop's matmul whose K is a block, with no batch, whose result is
    read only by element loops (stores and in-place updates, no Scalar read
    and no staged vector), whose computed operands a lane recomputes at any
    element: from loads, iotas, tile indices, constants, 0-d values and
    element-wise ops, none of them reading the accumulator or another
    matmul's result, and none a load of a tensor the kernel writes, which a
    lane would re-run at the op, after a write the Scalar load preceded."""

    def refuse(what: str) -> NoReturn:
        raise metal_unsupported(
            f"metal_matmul='fragment' can't lower {what}; use metal_matmul='mpp'"
        )

    if not group.levels:
        refuse("a matmul over a ':' K, outside a K loop")
    if group.k_factor != 1:
        refuse("a matmul over a K held as per-thread parts")
    if group.batch_block is not None:
        refuse("a batched matmul")
    if group.mats:
        refuse("a matmul whose result is read in the Scalar layout")
    if group.vecs or any(g is group for g in plan.by_vec.values()):
        refuse("a matmul whose element loop reads a staged row or column vector")
    if group.stage_uses:
        refuse("a matmul whose result another matmul's operand is computed from")
    _, lhs, rhs = matmul_operands(group.matmul)
    for operand, node, block in (
        (group.lhs, lhs, group.m_block),
        (group.rhs, rhs, group.n_block),
    ):
        if operand.elements is not None:
            refuse("an operand computed from another matmul's result")
        if not operand.staged:
            continue
        recomputable = _OperandRecomputable(group, graphs, {block, group.k_block})
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


def transposed(group: MppGroup) -> bool:
    """Whether the group computes ``C^T = B^T A^T``: only its rhs is computed,
    which then takes the MMA's A, whose lane elements lie in one row."""
    return group.rhs.staged and not group.lhs.staged


#: Most fragments a lane keeps live in an MMA step: the accumulator's rows x
#: cols, the step's rows A fragments and its cols B fragments.  A speed
#: cutoff measured on an M4 Pro, not a correctness limit: 9 live 8x8 float
#: fragments run from registers there; 17 (a tile of twice the columns) give
#: correct results but ran up to 1.5x slower, the fragments spilling to
#: memory.  A refusal at codegen rather than a repair of the config, so the
#: autotuner skips fragment configs past it before any compile instead of
#: timing tiles that spill.  A GPU with more registers per lane may want a
#: higher cutoff.
MAX_FRAGMENTS = 9

#: Most MMA steps in a K tile (a K block of 128).  The steps are unrolled
#: (:func:`emit_fragment_run`), so the kernel's source and compile time grow
#: with them: a dequantized operand's run at 8 steps is thousands of lines.
#: A bound on the unrolled code (codegen and compile time), not a hardware
#: or performance limit.  Crossing it is graceful: codegen raises
#: ``BackendUnsupported`` before any MSL compile, and the autotuner skips
#: that config as it does one over :data:`MAX_FRAGMENTS`.
MAX_STEPS = 16


def _loop_accumulators(cg: GenerateAST, group: MppGroup) -> int:
    """The fragments the K loop's other fragment groups keep live through
    it: their accumulators."""
    loop = group.levels[0].owner
    total = 0
    for other in current_plan().by_matmul.values():
        if other is group or not other.levels or other.levels[0].owner is not loop:
            continue
        m, n, _ = static_tile(cg, (other.m_block, other.n_block, other.k_block))
        m, n = op_dims(other, m, n)
        total += -(-m // 8) * -(-n // 8)
    return total


def fragment_grid(
    group: MppGroup, m: int, n: int, k: int, others: int = 0
) -> tuple[int, int, int]:
    """The accumulator's ``(rows, cols, transposed)``: its 8x8 fragments
    over the op's M and N (N and M when transposed), for an op of ``m`` x
    ``n`` over a K tile of ``k`` in a K loop whose other groups keep
    ``others`` fragments live."""
    if m % 8 or n % 8 or k % 8:
        raise metal_unsupported(
            f"a fragment matmul tile {m}x{n}x{k}: M, N and K must be multiples of 8"
        )
    if k // 8 > MAX_STEPS:
        raise metal_unsupported(
            f"a fragment matmul K tile of {k}: its {k // 8} MMA steps are "
            f"unrolled (max {MAX_STEPS}); use a smaller K block"
        )
    rows, cols = m // 8, n // 8
    live = others + rows * cols + rows + cols
    if live > MAX_FRAGMENTS:
        held = f", {others} of them the K loop's other accumulators" if others else ""
        raise metal_unsupported(
            f"a fragment matmul tile {m}x{n}: a lane would keep {live} 8x8 "
            f"fragments live (max {MAX_FRAGMENTS}{held}); use smaller M and N blocks"
        )
    if transposed(group):
        return n // 8, m // 8, 1
    return m // 8, n // 8, 0


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


def _static_int(expr: object) -> int | None:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, sympy.Integer):
        return int(expr)
    return None


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
    first, last = _static_int(begin), _static_int(end)
    return first is not None and first >= 0 and last is not None and last > first


def _static_loop(cg: GenerateAST, block: int) -> tuple[LoopDimInfo, int, int] | None:
    """The active loop over ``block``: its info, begin and block size, or
    None when no loop over it is active or its begin isn't known at
    codegen."""
    states = cg.active_device_loops.get(block)
    info = states[-1].block_id_to_info.get(block) if states else None
    if info is None:
        return None
    begin = (
        0
        if info.begin_expr is None and info.begin_var_name is None
        else _static_int(info.begin_expr)
    )
    if begin is None:
        return None
    step = cg.device_function.resolved_block_size(block)
    assert isinstance(step, int)
    return info, begin, step


def _tiles_in_range(cg: GenerateAST, block: int, op: int, extent: int | None) -> bool:
    """Whether ``offset + i < extent`` for every ``i < op`` at every tile
    offset the loop over ``block`` takes: its begin, end and ``extent`` are
    known at codegen and the last tile's ``op`` positions end by ``extent``."""
    loop = _static_loop(cg, block)
    if loop is None or extent is None:
        return False
    info, begin, step = loop
    end = _static_int(info.end_expr)
    if end is None or end <= begin:
        return False
    last = begin + (end - begin - 1) // step * step
    return last + op <= extent


#: How a view's element load treats a dim (:func:`_dim_mode`).
_UNMASKED, _MASKED, _BROADCAST = 0, 1, 2


def _dim_mode(
    cg: GenerateAST, view: View, d: int, dim: Dim, op: int, k_block: int
) -> int:
    """How a view's element loads index dim ``d`` (``op`` positions from the
    view's origin): a tile over a dim of size 1 broadcasts it (index 0, K
    masked at the loop's end only); otherwise the dim's extent masks it
    unless a tile dim's positions stay inside it at every tile, or a ``:``
    dim's inside its size."""
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
        size = _static_int(states[-1].block_id_to_info[k_block].end_expr)
    return _UNMASKED if _tiles_in_range(cg, dim.block_id, op, size) else _MASKED


def codegen_fragment_run(
    cg: GenerateAST, group: MppGroup, acc: ast.Name, values: tuple[object, object]
) -> ast.AST:
    """``_helion_mpp_run(acc, gid, lhs..., rhs..., ..., fragment=[lhs, rhs])``
    of a fragment group: each operand as a view's nine run arguments
    (``view_args``) or the name of its value cast to the operand's dtype,
    and per operand ``(computed, view dims masked, bindings, wide)``, where
    the bindings make a lane act as the thread holding the element at
    ``(_e_m or _e_n, _e_k)`` of a computed operand and ``wide`` marks a view
    whose indices may pass 2**31.  ``steps`` is the K tile's MMA steps,
    ``k_live`` its live length, which masks a computed operand's K tail
    (None when every tile is whole), ``facts`` the tile offsets' ``(offset,
    multiple, least)`` (:func:`_offset_fact`), over which the run folds its
    elements' index arithmetic, ``tables`` the config's
    ``metal_tabulate_max_entries``, and ``k_step`` the K loop's offset and
    step when it loads its operands a tile ahead (``range_num_stages=2``),
    else None."""
    backend = CompileEnvironment.current().backend
    m, n, k = static_tile(cg, (group.m_block, group.n_block, group.k_block))
    op = dict(zip((group.m_block, group.n_block), op_dims(group, m, n), strict=True))
    op[group.k_block] = k
    parts: list[str] = []
    anchors: list[tuple[int, str | None]] = []
    outers: list[str] = []
    sides: list[str] = []
    for (operand, blocks, k_pos), value, e in zip(
        group_sides(group), values, ("_e_m", "_e_n"), strict=True
    ):
        view = operand.view
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
        parts += op_parts
        outers.append(f"[{outer}]")
        sides.append(f"({int(operand.staged)}, {masks}, [{', '.join(binds)}], {wide})")
    info = cg.active_device_loops[group.k_block][-1].block_id_to_info[group.k_block]
    k_live = "None"
    if (group.lhs.staged or group.rhs.staged) and not _tiles_in_range(
        cg, group.k_block, k, _static_int(info.end_expr)
    ):
        k_live = f"{info.end_var_name} - {cg.offset_var(group.k_block)}"
    anchor_list = ", ".join(f"({p}, {a})" for p, a in anchors)
    facts = [
        fact
        for block in (group.m_block, group.n_block, group.k_block)
        if (fact := _offset_fact(cg, block)) is not None
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
    if stages >= 2:
        k_fact = _offset_fact(cg, group.k_block)
        k_loop = _static_loop(cg, group.k_block)
        if (
            k_fact is None
            or k_loop is None
            or not loads_ahead(k_fact[2], _specialized(k_loop[0].end_expr))
        ):
            raise metal_unsupported(
                "range_num_stages=2 on a K loop without known bounds it runs between"
            )
        k_step = (k_fact[0], k_loop[2])
    return expr_from_string(
        f"{RUN}({acc.id}, {group.gid}, {', '.join(parts)}, "
        f"anchors=[{anchor_list}], outer=[{', '.join(outers)}], steps={k // 8}, "
        f"k_live={k_live}, facts={facts!r}, "
        f"tables={metal_tabulate_max_entries(cg.device_function.config)}, "
        f"k_step={k_step!r}, "
        f"fragment=[{', '.join(sides)}])"
    )


def _offset_fact(cg: GenerateAST, block: int) -> tuple[str, int, int] | None:
    """``(offset, multiple, least)`` of the loop over ``block`` when its
    begin is known: each tile's offset is the begin plus a multiple of the
    block size."""
    loop = _static_loop(cg, block)
    if loop is None or loop[1] < 0:
        return None
    _, begin, step = loop
    return cg.offset_var(block), math.gcd(step, begin), begin


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def is_fragment_acc(call: ast.Call) -> bool:
    """Whether a ``_helion_mpp_acc`` call declares a fragment group's
    accumulator (:func:`emit_fragment_acc`)."""
    return _keyword(call, "fragment") is not None


def fragment_run_binds(call: ast.Call) -> list[list[ast.Tuple]]:
    """A ``_helion_mpp_run``'s per-operand bindings when it is a fragment
    run; ``[]`` for an MPP run."""
    sides = _keyword(call, "fragment")
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


def _parse(text: str) -> ast.expr:
    """A plain ``ast`` expression: MSL assembly works on the kernel's parsed
    source (``metal_jit``), whose nodes it deep-copies."""
    return cast("ast.Expr", ast.parse(text).body[0]).value


def _src(node: ast.AST) -> str:
    return f"({ast.unparse(node)})"


def _view_element(
    args: list[ast.expr],
    masks: list[int],
    outer: list[list[ast.expr]],
    index: tuple[str, str],
    wide: int,
) -> ast.expr:
    """A view's element at ``index`` (the op's two positions from its
    origin) as a float: a load, masked to 0 past each masked dim's extent
    and where an outer index lies outside its dim; a broadcast dim reads
    index 0 (:func:`_dim_mode`).  A ``wide`` view, one that may hold 2**31
    elements or more, computes its offset in ``long``."""
    tensor, *dims = args
    offset: list[str] = []
    terms: list[str] = []
    for d, i in enumerate(index):
        o, e, size, stride = dims[4 * d : 4 * d + 4]
        at = f"({_src(o)} + {i})"
        if masks[d] == _BROADCAST:
            # K past the loop's end (the view's K extent) reads 0; rows and
            # columns past M's and N's are never stored.
            if i == "_e_k":
                terms.append(f"{at} < {_src(e)}")
            continue
        offset.append(f"{f'static_cast(long, {at})' if wide else at} * {_src(stride)}")
        if masks[d] == _MASKED:
            terms.append(f"{at} < {_src(e)}")
            if ast.dump(e) != ast.dump(size):
                terms.append(f"{at} < {_src(size)}")
    for x, size, stride in outer:
        offset.append(
            f"{f'static_cast(long, {_src(x)})' if wide else _src(x)} * {_src(stride)}"
        )
        terms.append(f"0 <= {_src(x)} and {_src(x)} < {_src(size)}")
    mask = " and ".join(terms) if terms else "None"
    other = ", other=0" if terms else ""
    return _parse(
        f"static_cast(float, tl.load({_src(tensor)} + ({' + '.join(offset) or '0'}), "
        f"{mask}{other}))"
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
    its bindings, and is 0 past the K tile's live length; a view's is a
    masked load (:func:`_view_element`)."""
    grid = _keyword(acc_call, "fragment")
    steps, k_live, facts, tables, k_step, sides = (
        _keyword(call, name)
        for name in ("steps", "k_live", "facts", "tables", "k_step", "fragment")
    )
    assert grid is not None and steps is not None and facts is not None
    assert tables is not None and k_step is not None
    assert isinstance(sides, ast.List)
    _, _, flipped = ast.literal_eval(grid)
    outer = run_outer(call)
    pos = 2
    loops: list[list[ast.stmt]] = []
    for i, (side, binds_list) in enumerate(
        zip(sides.elts, fragment_run_binds(call), strict=True)
    ):
        assert isinstance(side, ast.Tuple)
        staged, masks, wide = (ast.literal_eval(side.elts[j]) for j in (0, 1, 3))
        index = ("_e_m", "_e_k") if i == 0 else ("_e_k", "_e_n")
        if staged:
            value = _parse(f"static_cast(float, {_src(call.args[pos])})")
            pos += 1
            if not (isinstance(k_live, ast.Constant) and k_live.value is None):
                assert k_live is not None
                value = _parse(f"({_src(value)} if _e_k < {_src(k_live)} else 0.0)")
        else:
            value = _view_element(
                call.args[pos : pos + 9], masks, outer[i], index, wide
            )
            pos += 9
        put = ast.Expr(ast.Call(ast.Name(PUT, ast.Load()), [value], []))
        binds: dict[str, ast.expr] = {}
        for b in binds_list:
            name, expr = b.elts
            binds[cast("str", ast.literal_eval(name))] = expr
        loops.append([*rerun(binds, names_read(put), [value]), put])
    assert pos == len(call.args)
    a_side = 1 if flipped else 0
    body: list[ast.stmt] = [
        ast.For(
            ast.Name("_it", ast.Store()),
            ast.Call(ast.Name(marker, ast.Load()), [], []),
            loops[side],
            [],
        )
        for marker, side in ((FRAG_A, a_side), (FRAG_B, 1 - a_side))
    ]
    loop = ast.For(
        ast.Name("_it", ast.Store()),
        ast.Call(
            ast.Name(FRAG_RUN, ast.Load()),
            [call.args[0], steps, facts, tables, k_step],
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
    element = f"{coop}[_fi >> 1].thread_elements()[_fi & 1]"
    lines = [
        f"{inner}int {row} = 8 * ((_fi >> 1) / {cols}) + _fm;",
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
    fragments, filled with the seed value or from the staged seed tile."""
    grid = _keyword(call, "fragment")
    assert grid is not None
    rows, cols, flipped = ast.literal_eval(grid)
    fill = ast.literal_eval(call.args[7])
    block_m, block_n = (ast.literal_eval(a) for a in call.args[11:ACC_SEED])
    tm, tn = (ast.literal_eval(a) for a in call.args[1:3])
    seed = ast.literal_eval(call.args[ACC_SEED])
    state.declare(name)
    state.coop[name] = name
    state.coop_tile[name] = (block_m, block_n)
    state.fragments[name] = (rows, cols, flipped)
    # The fragments cover the op's M and N, which are cut to the loops'
    # extents and so lie inside the blocks unless a block of 1 grew to 8.
    if tm > block_m or tn > block_n:
        state.coop_rounded.add(name)
    value = float(fill if seed is None else 0.0)
    parts.append(f"{pad}simdgroup_float8x8 {name}[{rows * cols}];")
    parts += [
        f"{pad}{name}[{i}] = make_filled_simdgroup_matrix<float, 8, 8>({value!r}f);"
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
    accumulator's."""
    assert isinstance(stmt.iter, ast.Call)
    _, steps, facts_arg, tables, _ = stmt.iter.args
    rows, cols, flipped = _grid(state, root)
    row, col = _element_names(flipped)
    a_loop, b_loop = stmt.body
    assert isinstance(a_loop, ast.For) and isinstance(b_loop, ast.For)
    facts = dict(_LANE_FACTS)
    for name, multiple, least in ast.literal_eval(facts_arg):
        facts[name] = Fact(multiple, least)
    n_steps = ast.literal_eval(steps)
    sides = [
        (
            a_loop,
            "_frag_a",
            rows,
            (
                f"{row} = 8 * (_fi >> 1) + _fm",
                f"_e_k = {n_steps} * (_fn + (_fi & 1)) + _fs",
            ),
        ),
        (
            b_loop,
            "_frag_b",
            cols,
            (
                f"_e_k = {n_steps} * _fm + _fs",
                f"{col} = 8 * (_fi >> 1) + _fn + (_fi & 1)",
            ),
        ),
    ]
    return n_steps, sides, facts, ast.literal_eval(tables)


#: A fragment element's statements, its names' affine values and the facts
#: they fold over (:func:`_element_body`).
FragmentElement = tuple[list[ast.stmt], dict[str, object], dict[str, Fact]]


def _element_body(
    loop: ast.For,
    decls: tuple[str, str],
    step: int,
    i: int,
    facts: dict[str, Fact],
    tables: int,
    state: EmitState,
) -> FragmentElement:
    """Element ``i`` of step ``step``'s statements, its position bound and
    folded (``affine_fold``) over the run's facts and the kernel's integer
    constants, and its decodes tabulated (``tabulate``) under the cap
    ``tables``; the affine values of its names; and the facts they fold
    over, the run's and its floor terms'."""
    env: dict[str, object] = {
        name: Affine.of(value)
        for name, value in state.constants.items()
        if isinstance(value, int)
    }
    env |= {"_fs": Affine.of(step), "_fi": Affine.of(i)}
    facts = dict(facts)
    positions = ast.parse("\n".join(decls)).body
    body = fold([*positions, *loop.body], env, facts)[len(positions) :]
    return tabulate(body, state.constants, state.tables, tables), env, facts


def emit_fragment_run(
    stmt: ast.For,
    parts: list[str],
    indent: int,
    state: EmitState,
    emit_body: Callable[[list[ast.stmt], list[str], int, EmitState], None],
) -> None:
    """A fragment run's loop (:func:`fragment_run`): per MMA step, the A and
    B fragments of every lane's operand elements, then each accumulator
    fragment's ``simdgroup_multiply_accumulate``.  Lane ``l``'s elements of
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
    their early loads read the registers loaded the iteration before."""
    pad = " " * indent
    inner = pad + "    "
    assert isinstance(stmt.iter, ast.Call)
    acc = stmt.iter.args[0]
    assert isinstance(acc, ast.Name)
    rows, cols, _ = _grid(state, state.coop[acc.id])
    n_steps, sides, facts, tables = _run_setup(stmt, state, state.coop[acc.id])
    k, early, elements = state.prefetch.get(id(stmt), ("", {}, {}))
    saved = state.put
    for step in range(n_steps):
        parts += [
            f"{pad}{{",
            f"{inner}constexpr short _fs = {step};",
            f"{inner}simdgroup_float8x8 _frag_a[{rows}];",
            f"{inner}simdgroup_float8x8 _frag_b[{cols}];",
        ]
        for side, (loop, array, count, decls) in enumerate(sides):
            state.put = f"{array}[_fi >> 1].thread_elements()[_fi & 1]"
            for i in range(2 * count):
                lines = [
                    f"{inner}{{",
                    f"{inner}    constexpr short _fi = {i};",
                    *(f"{inner}    int {decl};" for decl in decls),
                ]
                element = elements.get((side, step, i))
                if element is None:
                    element = _element_body(loop, decls, step, i, facts, tables, state)
                body, env, element_facts = element
                if early:
                    body = _read_early(body, env, element_facts, k, early)
                with state.isolated_scope():
                    emit_body(body, lines, indent + 8, state)
                parts += [*lines, f"{inner}}}"]
        parts += [
            f"{inner}simdgroup_multiply_accumulate({acc.id}[{a * cols + b}], "
            f"_frag_a[{a}], _frag_b[{b}], {acc.id}[{a * cols + b}]);"
            for a in range(rows)
            for b in range(cols)
        ]
        parts.append(f"{pad}}}")
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
        elements: dict[tuple[int, int, int], FragmentElement] = {}
        for s in range(n_steps):
            for side, (loop, _, count, decls) in enumerate(sides):
                for i in range(2 * count):
                    element = _element_body(loop, decls, s, i, facts, tables, state)
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
        state.prefetch[id(stmt)] = (k, early, elements)
        if not early:
            continue
        next_tile = f"{loop_var} + {step} < {end}"
        state.early_count += len(early)
        for (ptr, rest, coeff), name in early.items():
            first = rest + Affine.of(int(coeff * begin))
            nxt = rest + Affine.term(k, 1).scale(coeff) + Affine.of(int(coeff * step))
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


def kernel_constants(body: list[ast.stmt]) -> dict[str, int | float]:
    """The kernel's constants a fragment run folds over: names assigned
    once in the whole kernel, an integer or float literal.  A name assigned
    anywhere else too (``mpp_matmul.assigned_names``) is not one, whatever
    the control flow; a fragment run's own statements are folded per element
    instead."""
    counts: collections.Counter[str] = collections.Counter()
    literals: dict[str, int | float] = {}
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.For) and call_name(node.iter) == FRAG_RUN:
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
            and not isinstance(node.value.value, bool)
        ):
            literals[node.targets[0].id] = node.value.value
        counts.update(assigned_names(node))
        stack.extend(ast.iter_child_nodes(node))
    return {name: v for name, v in literals.items() if counts[name] == 1}


def fragment_acc_keywords(
    cg: GenerateAST, group: MppGroup, m: int, n: int, k: int
) -> str:
    """The keywords of a fragment group's ``_helion_mpp_acc``: its grid
    (:func:`fragment_grid`)."""
    grid = fragment_grid(group, m, n, k, _loop_accumulators(cg, group))
    return f", fragment={grid!r}"


class FragmentLowering:
    """What :mod:`mpp_matmul` calls to lower a fragment group: the planner's
    check, the accumulator's keywords, the op's codegen and its loops in
    ``elementize``.  ``MetalBackend.pre_codegen`` gives it to the planner and
    ``metal_jit`` to ``elementize``, so :mod:`mpp_matmul`, which this module
    builds on, doesn't import it."""

    check_group = staticmethod(check_fragment_group)
    acc_keywords = staticmethod(fragment_acc_keywords)
    codegen_run = staticmethod(codegen_fragment_run)
    run = staticmethod(fragment_run)
    run_binds = staticmethod(fragment_run_binds)


FRAGMENT_LOWERING = FragmentLowering()
