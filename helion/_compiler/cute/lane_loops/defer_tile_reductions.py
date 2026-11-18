"""Combine a carried tile sum across threads once, after its device loop.

Role: device-body pass (``cute_defer_tile_reduce``), right after the
two-pass lane split and before the grid vector-loop sink.

A kernel that accumulates a tile reduction through a serial device loop::

    for tile_s, tile_c in hl.tile([splits, channels]):
        acc = hl.zeros([tile_s, tile_c])
        for tile_r in hl.tile(rows):
            acc = acc + x[tile_r, tile_c].sum(0)

leaves the lane split with one cross-thread combine per row tile::

    for tile_offset in range(0, rows, BLOCK):
        lane_acc = 0  # reset
        for lane in range(ROWS):
            lane_acc += ...  # this thread's rows
        tile_sum = grouped_reduce(lane_acc, "sum")  # warp shuffles or smem
        acc = acc + tile_sum  # carried add

The combine costs a shuffle tree (or a shared-memory round with barriers when
the row threads span warps) per tile and carried sum, times the vector lanes
once ``cute_vloop_sink`` interchanges the V-loop.  This pass keeps the
per-thread accumulators live across the whole device loop and combines once
after it, which is how TorchInductor's looped split reductions accumulate::

    lane_acc = 0
    for tile_offset in range(0, rows, BLOCK):
        for lane in range(ROWS):
            lane_acc += ...
    tile_sum = grouped_reduce(lane_acc, "sum")
    acc = acc + tile_sum

The addends are the same and so is every thread's share of them; the order
changes: each thread sums its rows of every tile before the threads combine,
where the loop summed the threads tile by tile into the carry.  The result
differs by the rounding of another summation order.  No bound independent of
the length holds for that: the rounding error of a sum of ``n`` addends, in
either order, is at most ``n - 1`` unit roundoffs times the sum of the
addends' magnitudes (to first order), so the two results can differ by up to
twice that, more the more elements are summed and by far more than the last
bits where the addends cancel.  So the knob is off by default.  Only fp32 and
fp64 sums are deferred: integer and half-precision sums reach the carry
through casts, which the pass declines, and stay per tile.

The pass reads what the lane split recorded on the accumulate loop it built
(``LaneSplitRecord``: the markers, their accumulators, resets and finalizers)
and declines unless

- the device loop has no ``else``, ``break``, ``continue`` or ``return`` and
  holds exactly one split lane loop, whose resets and finalizers are its own
  statements;
- every marker is a plain sum whose result keeps the accumulator's dtype;
- a finalizer combines threads: with one thread on the tile each sum is
  finalized as that thread's own accumulator, and moving it would change the
  summation order without saving a combine;
- every result is read only as one addend of ``carry = carry + result`` in
  the loop body, the carry read directly or through plain copies made
  before, with casts to the accumulator's dtype only (the rename groups say
  which name the loop carries);
- the accumulators are used by their resets, their self-adds in the lane
  loop (``accumulator = accumulator + value``, the value reading no
  accumulator, so that the partial sums can run on from one tile to the
  next) and the finalizers alone, the results and copies by the moved
  statements alone, anywhere in the kernel, and the carries by nothing else
  in the loop;
- the moved statements read nothing else that the loop binds (its index, a
  lane-loop local).

A max or a min would combine exactly, but only when the carry's combine
treats NaN as the lane fold does, which the forms a carry can take
(``_lane_carry_combine_operands``) do not all share; they stay per tile.

The search offers the knob where the device IR shows such a loop
(``carried_tile_sum_blocks``): every reduction over the loop's tile is an
fp32 or fp64 sum read only by ``carried + sum``, in the carry's dtype, and
the loop reads the carry nowhere else.  That is the part of the conditions
above that the kernel fixes; the rest (several threads on the tile, each
with several of its elements, so that the split this pass reads folds them
in a lane loop) is the config's, and where it fails the pass leaves the code
as it is.  A loop that runs one tile (a tile that covers a static extent) is
rewritten like any other: its statements run once, in the order they had.

A kernel with a matmul is not offered the knob and rejects it
(``register_search_surface``), the rule ``cute_tail_peel`` and
``cute_bool_byte_packets`` follow.  The pass would decline a matmul's own
sums in any case: the split lowers a certified running sum without the
record read here, and a matmul contribution is no plain sum (``_plain_sum``).
"""

from __future__ import annotations

import ast
import collections
import operator
from typing import TYPE_CHECKING

import torch

from ....language._tracing_ops import _for_loop
from ....language._tracing_ops import _new_var
from ....language._tracing_ops import _phi
from ...ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ...compile_environment import CompileEnvironment
from ...compile_environment import _symint_expr
from ...device_ir import ForLoopGraphInfo
from ...inductor_lowering import ReductionLowering
from .._ast_pass_utils import _self_addend
from .._ast_pass_utils import name_assignment
from .markers import HELION_FINALIZED
from .markers import HELION_LANE_SPLIT_ATTR
from .markers import LaneSplitRecord
from .markers import _dtype_ctor_from_identity
from .markers import _is_constexpr_lane_iter

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    import sympy

    from ...device_ir import DeviceIR
    from .markers import _LaneReduceMarker

# The dtypes in which a sum's lane accumulator, its combined result and a
# loop-carried value are one type in the generated code, so the carried add
# is the plain ``carry = carry + result`` the pass moves.  A half-precision
# sum accumulates in fp32 and is cast back, and a half-precision or integer
# carry is cast to its initial type at the end of the loop body
# (``carried_type_joins``); the pass declines both.
_CARRIED_SUM_DTYPES = frozenset({torch.float32, torch.float64})


def carried_tile_sum_blocks(device_ir: DeviceIR) -> frozenset[int]:
    """The blocks of the one-axis ``hl.tile`` device loops whose reductions
    over their own tile are all carried sums the pass can move
    (``acc = acc + x[tile].sum(0)``, see ``_carried_sum``): where
    ``cute_defer_tile_reduce`` can apply (``register_search_surface`` takes
    them for kernels without a matmul only).  Structure and dtypes only:
    whether the sum's threads combine at all (several threads on the tile)
    and the nest takes the split this pass reads are facts of the config, and
    where they fail the knob leaves the code as it is.  A dim is matched by
    its block's size symbol, so this needs no current host function."""
    env = CompileEnvironment.current()
    blocks: set[int] = set()
    for info in device_ir.graphs:
        for node in info.graph.find_nodes(op="call_function", target=_for_loop):
            loop = device_ir.graphs[node.args[0]]
            # A rolled reduction's loop is a ``ReductionLoopGraphInfo``.
            if type(loop) is not ForLoopGraphInfo or len(loop.block_ids) != 1:
                continue
            (block_id,) = loop.block_ids
            if _adds_tile_sums_to_carries(
                loop, node, env.block_sizes[block_id].symbol()
            ):
                blocks.add(block_id)
    return frozenset(blocks)


def _has_dim(value: object, symbol: sympy.Symbol) -> bool:
    return isinstance(value, torch.Tensor) and any(
        isinstance(size, torch.SymInt) and _symint_expr(size) == symbol
        for size in value.shape
    )


def _adds_tile_sums_to_carries(
    loop: ForLoopGraphInfo, for_node: torch.fx.Node, symbol: sympy.Symbol
) -> bool:
    """Whether ``loop``'s graph reduces over the tile dim ``symbol`` and
    every such reduction is a carried sum (``_carried_sum``).  The pass moves
    the combines of a loop's tile reductions together or not at all, so one
    that has to stay (a max, a sum the loop reads) keeps the others too."""
    (output,) = loop.graph.find_nodes(op="output")
    outputs = output.args[0]
    if not isinstance(outputs, (list, tuple)):
        return False
    # Output slot -> the outer value the loop's result replaces.
    joined: dict[object, object] = {}
    for item in for_node.users:
        if item.target is operator.getitem:
            for phi in item.users:
                if phi.target is _phi and phi.args[1] is item:
                    joined[item.args[1]] = phi.args[0]
    reductions = [
        node
        for node in loop.graph.nodes
        if isinstance(node.meta.get("lowering"), ReductionLowering)
        and isinstance(source := node.args[0], torch.fx.Node)
        and _has_dim(source.meta.get("val"), symbol)
        and not _has_dim(node.meta.get("val"), symbol)
    ]
    return bool(reductions) and all(
        _carried_sum(node, loop, outputs, joined) for node in reductions
    )


def _carried_sum(
    node: torch.fx.Node,
    loop: ForLoopGraphInfo,
    outputs: Sequence[object],
    joined: Mapping[object, object],
) -> bool:
    """Whether the reduction ``node`` of ``loop``'s graph is a sum whose only
    use is one addend of ``carried + sum``, the loop's next value of
    ``carried`` (the outer graph joins that output with the value passed in)
    and the only read of either in the loop, all three in one dtype that the
    lane accumulator and the carry keep (``_CARRIED_SUM_DTYPES``)."""
    if node.meta["lowering"].reduction_type != "sum" or len(node.users) != 1:
        return False
    (add,) = node.users
    if (
        add.target is not torch.ops.aten.add.Tensor
        or len(add.args) != 2
        # The loop's output alone: the body reads the updated carry nowhere.
        or len(add.users) != 1
    ):
        return False
    (carried,) = [arg for arg in add.args if arg is not node] or [None]
    if not (
        isinstance(carried, torch.fx.Node)
        and carried.target is _new_var
        # The add alone: the body reads the carry it was passed nowhere else.
        and len(carried.users) == 1
        and isinstance(placeholder := carried.args[0], torch.fx.Node)
        and placeholder.op == "placeholder"
    ):
        return False
    dtypes = {
        value.dtype if isinstance(value := item.meta.get("val"), torch.Tensor) else None
        for item in (node, carried, add)
    }
    if len(dtypes) != 1 or not dtypes <= _CARRIED_SUM_DTYPES:
        return False
    outer = loop.placeholder_to_outer_arg(placeholder)
    return any(
        result is add and joined.get(slot) is outer
        for slot, result in enumerate(outputs)
    )


def _uses(nodes: Iterable[ast.AST], rename_groups: Mapping[str, str]) -> dict[str, int]:
    """How often each name occurs in ``nodes``, read or written, under its
    rename group's first spelling."""
    counts: collections.Counter[str] = collections.Counter()
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                counts[rename_groups.get(node.id, node.id)] += 1
    return counts


def _names_bound_in(loop: ast.For, rename_groups: Mapping[str, str]) -> set[str]:
    """The names ``loop`` may bind: its own and its inner loops' variables
    and every name in an assignment target.  For a subscript or attribute
    target that is its root and also the names its index reads, more than
    the statement binds: the caller declines on a bound name, so the surplus
    only adds declines."""
    bound: set[str] = set()

    def bind(target: ast.AST) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                bound.add(rename_groups.get(node.id, node.id))

    for node in ast.walk(loop):
        if isinstance(node, ast.For):
            bind(node.target)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            bind(node.target)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bind(node)
    return bound


def _below_casts(node: ast.AST, ctor: str | None) -> ast.AST:
    """``node`` below any casts to the accumulator dtype ``ctor``."""
    while (
        ctor is not None
        and isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) == ctor
    ):
        node = node.args[0]
    return node


def _plain_sum(marker: _LaneReduceMarker) -> bool:
    """A sum whose finalized value is the combined accumulator itself, under
    no more than a cast to the accumulator's own dtype."""
    if (
        marker.reduction_type != "sum"
        or marker.matmul_contribution
        or marker.group_cluster_n > 1
    ):
        return False
    template = ast.parse(marker.wrap_template, mode="eval").body
    inner = _below_casts(template, _dtype_ctor_from_identity(marker.identity_expr))
    return isinstance(inner, ast.Name) and inner.id == HELION_FINALIZED


def _combines_threads(record: LaneSplitRecord) -> bool:
    """Whether a finalizer of ``record`` combines accumulators across threads.
    For a tile with one thread the split finalizes each sum as that thread's
    own accumulator, ``result = accumulator`` under casts to its dtype
    (``_finalize_lane_reduce_marker``): nothing to combine."""
    ctors = {
        marker.result_var: _dtype_ctor_from_identity(marker.identity_expr)
        for marker in record.markers
    }
    for stmt in record.finalize:
        assignment = name_assignment(stmt)
        if assignment is None or assignment[0] not in ctors:
            return True
        value = _below_casts(assignment[1], ctors[assignment[0]])
        if not (isinstance(value, ast.Name) and value.id in record.accumulators):
            return True
    return False


def _sum_operands(value: ast.AST, ctor: str | None) -> tuple[str, str] | None:
    """The two names ``value`` adds, each under casts to ``ctor`` only."""
    value = _below_casts(value, ctor)
    if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)):
        return None
    left = _below_casts(value.left, ctor)
    right = _below_casts(value.right, ctor)
    if isinstance(left, ast.Name) and isinstance(right, ast.Name):
        return left.id, right.id
    return None


def _is_serial_device_loop(loop: ast.For) -> bool:
    """Not a lane loop and not a constexpr vector lane."""
    return getattr(
        loop, HELION_LANE_LOOP_VAR_ATTR, None
    ) is None and not _is_constexpr_lane_iter(loop)


def _deferred(
    loop: ast.For, kernel_uses: Mapping[str, int], rename_groups: Mapping[str, str]
) -> list[ast.AST] | None:
    """``[resets..., loop, combine and carried adds...]`` for a device loop
    whose split lane loop's sums can combine after it, else ``None``."""
    if loop.orelse or any(
        isinstance(node, (ast.Break, ast.Continue, ast.Return))
        for node in ast.walk(loop)
    ):
        return None
    split_loops = [
        stmt
        for stmt in loop.body
        if getattr(stmt, HELION_LANE_SPLIT_ATTR, None) is not None
    ]
    if len(split_loops) != 1:
        return None
    (lane_loop,) = split_loops
    assert isinstance(lane_loop, ast.For)
    record: LaneSplitRecord = getattr(lane_loop, HELION_LANE_SPLIT_ATTR)
    position = {id(stmt): index for index, stmt in enumerate(loop.body)}
    at = position[id(lane_loop)]
    if not (
        all(position.get(id(stmt), at) < at for stmt in record.resets)
        and all(position.get(id(stmt), at) > at for stmt in record.finalize)
        and all(_plain_sum(marker) for marker in record.markers)
        # With one thread on the tile nothing combines: moving the sums would
        # change their order and save nothing.
        and _combines_threads(record)
    ):
        return None

    def canonical(name: str) -> str:
        return rename_groups.get(name, name)

    finalize_ids = {id(stmt) for stmt in record.finalize}
    results = {marker.result_var: marker for marker in record.markers}
    # Plain copies in the loop's tail by target (``carry_copy = carry``) and
    # the carried adds of the results.  A result that no carried add takes is
    # read elsewhere, which the use counts below decline, or dead.
    copies: dict[str, tuple[str, ast.AST]] = {}
    moved_ids = set(finalize_ids)
    carries: set[str] = set()
    for stmt in loop.body[at + 1 :]:
        if id(stmt) in finalize_ids:
            continue
        assignment = name_assignment(stmt)
        if assignment is None:
            continue
        target, value = assignment
        if isinstance(value, ast.Name):
            copies[target] = (value.id, stmt)
            continue
        for result, marker in results.items():
            operands = _sum_operands(
                value, _dtype_ctor_from_identity(marker.identity_expr)
            )
            if operands is None or result not in operands:
                continue
            carried = operands[1] if operands[0] == result else operands[0]
            # Each copy is made before the statement that reads it.
            chain: list[ast.AST] = []
            reader = stmt
            while carried in copies:
                carried, copy = copies[carried]
                if position[id(copy)] >= position[id(reader)]:
                    return None
                chain.append(copy)
                reader = copy
            if canonical(carried) != canonical(target):
                return None
            carries.add(canonical(target))
            moved_ids.update(id(copy) for copy in chain)
            moved_ids.add(id(stmt))
            break

    moved = [stmt for stmt in loop.body if id(stmt) in moved_ids]
    accumulators = {canonical(name) for name in record.accumulators}
    # Every accumulator is written by its reset and its self-add alone, and
    # the self-add is ``accumulator = accumulator + value`` with a value that
    # reads no accumulator: an accumulator the lane loop assigned or scaled
    # would not hold the sum of the tiles once its reset has left the loop.
    writes = [
        (stmt, assignment[0])
        for stmt in lane_loop.body
        if (assignment := name_assignment(stmt)) is not None
        and canonical(assignment[0]) in accumulators
    ]
    if len(writes) != len(accumulators) or any(
        (addend := _self_addend(stmt, target)) is None
        or accumulators & _uses([addend], rename_groups).keys()
        for stmt, target in writes
    ):
        return None
    self_adds = [stmt for stmt, _ in writes]
    moved_uses = _uses(moved, rename_groups)
    own_uses = _uses([*record.resets, *self_adds, *moved], rename_groups)
    private = {
        canonical(assignment[0])
        for stmt in moved
        if (assignment := name_assignment(stmt)) is not None
    } - carries
    if any(name_assignment(stmt) is None for stmt in moved):
        return None
    if any(kernel_uses[name] != own_uses[name] for name in accumulators):
        return None
    if any(kernel_uses[name] != moved_uses[name] for name in private - accumulators):
        return None
    loop_uses = _uses([loop], rename_groups)
    if any(loop_uses[name] != moved_uses[name] for name in carries):
        return None
    bound = _names_bound_in(loop, rename_groups)
    if any(
        name in bound and name not in accumulators | private | carries
        for name in moved_uses
    ):
        return None

    reset_ids = {id(stmt) for stmt in record.resets}
    loop.body = [stmt for stmt in loop.body if id(stmt) not in reset_ids | moved_ids]
    return [*record.resets, loop, *moved]


def defer_tile_reductions(
    body: Sequence[ast.AST], rename_groups: Mapping[str, str]
) -> list[ast.AST]:
    """``body`` with the cross-thread combine of every carried tile sum that
    qualifies (see the module docstring) moved after its device loop."""
    kernel_uses = _uses(body, rename_groups)

    def visit(statements: Sequence[ast.AST]) -> list[ast.AST]:
        result: list[ast.AST] = []
        for stmt in statements:
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(stmt, field, None)
                if isinstance(inner, list) and all(
                    isinstance(child, ast.stmt) for child in inner
                ):
                    setattr(stmt, field, visit(inner))
            replaced = (
                _deferred(stmt, kernel_uses, rename_groups)
                if isinstance(stmt, ast.For) and _is_serial_device_loop(stmt)
                else None
            )
            if replaced is None:
                result.append(stmt)
            else:
                result.extend(replaced)
        return result

    return visit(body)
