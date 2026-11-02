"""Sink a grid constexpr vector loop into the serial loops it encloses.

A grid tile axis with ``cute_vector_widths[block] = V > 1`` is partitioned
into an outer lane loop and a constexpr V-loop that wraps the WHOLE grid body
(``DeviceGridState.wrap_body``).  When that body reduces another axis with a
serial device loop and a per-thread lane loop (a column sum: ``for tile_m:
acc += sum(x[tile_m, tile_n], dim=0)``), the emitted nest is::

    for vec_lane in cutlass.range_constexpr(V):  # grid axis element
        acc = 0
        for tile_offset in range(0, m, BLOCK):  # serial device loop
            lane_acc = 0
            for lane in range(ROWS):  # per-thread rows
                lane_acc += x[row(lane), base + vec_lane]  # 2-byte load
            acc += combine_across_threads(lane_acc)
        store(acc)

so every thread walks its row strip V times with scalar loads and one load
in flight, and pays V cross-thread combines per row tile.  This pass
distributes the V-loop over the body and interchanges it with the loops::

    acc = make_rmem_tensor(V)
    for vec_lane in range_constexpr(V): acc[vec_lane] = 0
    for tile_offset in range(0, m, BLOCK):
        lane_acc = make_rmem_tensor(V)
        for vec_lane in range_constexpr(V): lane_acc[vec_lane] = 0
        ptr = &x[row(0), base]                       # strength-reduced address
        for lane in range(ROWS):                     # optionally unrolled
            vec = cute.arch.load(ptr + lane * row_stride, V)   # ONE vector load
            for vec_lane in range_constexpr(V):
                lane_acc[vec_lane] += vec[vec_lane]
        reduce_fragment(lane_acc -> reduced)         # one grouped combine
        for vec_lane in range_constexpr(V): acc[vec_lane] += reduced[vec_lane]
    for vec_lane in range_constexpr(V): store(acc[vec_lane])

Per-V-lane scalars that live across a loop boundary are expanded into
V-element register fragments (or recomputed where they are cheap functions
of the lane), V-lane-invariant pure statements run once outside the sunk
V-loops, and everything else keeps its original per-lane order inside a
constexpr V-loop, so the observable effects (stores, barriers) run in the
same order as before within each segment.

The pass is driven by facts the strategy and the load lowering record
(``CuteVloopWrapperFact`` / ``CuteVloopLoadFact`` / ``CuteVloopStoreFact``,
for grid and device-loop V-loops alike); it fires only for the constexpr
loop of a recorded wrapper, only when at least one scalar load nested in an
inner loop becomes a vector load (or a store a packet), and it leaves the
body untouched whenever any statement or value is outside the proven forms
(a load under a per-lane condition the facts do not cover, a hoist whose
operands only the V-loop defines, ...).  When the knob shaped the thread
layout but nothing was sunk, ``generate_ast`` regenerates the knob-off code.

A masked load becomes one predicated packet load that leaves zeros
(``cute/masked_packet_load.py``); the per-lane selects that only repeat the
zero fill are dropped, so ptxas keeps a strip's loads in flight instead of
consuming each one at once.  A load may move above a segment's stores into
tensors proven disjoint from the loaded one, and the per-lane stores of such
a segment (a flat gather's ``out[row + lane]``) flush as one packet after the
V-loop.  A scalar folded across the lanes (``acc = acc + e``, through its
loop-carry aliases) stays scalar and accumulates in lane order inside its
segment's V-loop.  A chunk straddling a jagged tile's extent loads under the
extent mask evaluated at the chunk base when the load fact proves every
aligned chunk inside the tensor, keeping the per-lane selects.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast

from ...autotuner.config_spec import _CUTE_LANE_UNROLL_CHOICES
from ...language.memory_ops import _CUTE_VECTOR_UNROLL_CARRIER
from ...language.memory_ops import _CUTE_VECTOR_UNROLL_DTYPES
from ...language.memory_ops import _cute_unroll_vec_elem_type
from ...language.memory_ops import _cute_unroll_vec_load_expr
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import ReadWrites
from ..ast_read_writes import ast_rename
from ..compile_environment import CompileEnvironment
from .masked_packet_load_names import MASKED_PACKET_LOAD_HELPERS
from .persistent_branch_vec import _LOAD_MARKER
from .persistent_branch_vec import _STORE_MARKER
from .persistent_branch_vec import _marker_call
from .vector_reduction_packets import _clone

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    import torch

    from .device_state import CuteVloopLoadFact
    from .device_state import CuteVloopStoreFact
    from .device_state import CuteVloopWrapperFact

# Marks vector-load statements this pass emitted (an attribute on the node).
_VSINK_VECTOR_LOAD_ATTR = "_helion_vsink_vector_load"
# A select whose lane-invariant conjunct was moved onto the fold it feeds
# (``_hoist_fold_masks``): the widened load drops it.
_VSINK_FOLD_MASKED_ATTR = "_helion_vsink_fold_masked"
_TWO_STAGE_REDUCE = "_cute_grouped_reduce_shared_two_stage"
_TWO_STAGE_REDUCE_FRAGMENT = "_cute_grouped_reduce_shared_two_stage_fragment"
# Calls whose result carries the dtype of their first argument.
_DTYPE_PRESERVING_CALLS = (
    "cute.arch.warp_reduction",
    "_cute_grouped_reduce_",
    "cute.arch.fmax",
    "cute.arch.fmin",
    "cute.math.",
    "math.",
)
_UNKNOWN = "__helion_unknown__"
_CONFLICT = "__conflict__"


def _names_read(node: ast.AST) -> set[str]:
    return set(ReadWrites.from_ast(node).reads)


def _names_written(node: ast.AST) -> set[str]:
    return set(ReadWrites.from_ast(node).writes)


def _names_written_list(stmts: Sequence[ast.AST]) -> set[str]:
    return set().union(*(_names_written(s) for s in stmts)) if stmts else set()


def _plain_target(stmt: ast.AST) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    return None


def _is_constexpr_loop(stmt: ast.AST) -> tuple[str, int] | None:
    """``for X in cutlass.range_constexpr(V)`` -> ``(X, V)``."""
    if not isinstance(stmt, ast.For) or not isinstance(stmt.target, ast.Name):
        return None
    call = stmt.iter
    if (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) == "cutlass.range_constexpr"
        and len(call.args) == 1
        and not call.keywords
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, int)
        and not stmt.orelse
    ):
        return stmt.target.id, call.args[0].value
    return None


def _lane_loop_extent(loop: ast.For) -> int | None:
    """Extent of an ascending ``range(N)`` lane loop, else None."""
    call = loop.iter
    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and len(call.args) == 1
        and not call.keywords
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, int)
    ):
        return call.args[0].value
    return None


def _is_dtype_ctor(name: str) -> bool:
    member = name.rsplit(".", 1)[-1]
    return name.startswith("cutlass.") and bool(member) and member[0].isupper()


def _is_relocatable_statement(
    stmt: ast.AST, *, allow_load: bool, tensor_names: Collection[str]
) -> bool:
    """A plain assignment whose calls are all proven effect-free and that
    reads memory only if ``allow_load``.

    A load is a ``.load()`` / ``cute.arch.load`` call or a bare subscript of
    a kernel tensor argument (``tensor_names``): the gather lowering emits
    ``t[i] if mask else 0`` with no load call, and moving that read past a
    store or atomic to ``t`` would change the value it sees.
    """
    from ..tile_strategy import _is_proven_relocatable_call

    if not isinstance(stmt, ast.Assign) or _plain_target(stmt) is None:
        return False

    def relocatable(node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            return _is_proven_relocatable_call(node, allow_load=allow_load)
        if isinstance(node, ast.Subscript):
            return allow_load or not (
                isinstance(node.value, ast.Name) and node.value.id in tensor_names
            )
        return True

    return all(relocatable(node) for node in ast.walk(stmt))


def _constexpr_loop(var: str, width: int, body: list[ast.stmt]) -> ast.For:
    loop = create(
        ast.For,
        target=create(ast.Name, id=var, ctx=ast.Store()),
        iter=expr_from_string(f"cutlass.range_constexpr({width})"),
        body=body,
        orelse=[],
        type_comment=None,
    )
    # A sunk V-loop is a synthetic lane loop: when dead-code elimination
    # leaves its body with nothing that reads the lane (a segment that only
    # defined the per-element index), ``dead_lane_loop_elimination`` splices
    # the body out instead of leaving an empty ``for``.
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, var)
    return loop


class _Unsupported(Exception):
    """The body is outside the proven forms; leave it untouched."""


@dataclasses.dataclass
class _Analysis:
    varying: set[str]
    expanded: dict[str, str]  # name -> fragment name
    fragment_dtypes: dict[str, str]  # fragment name -> dtype constructor
    # Cheap per-lane definitions recomputed where a segment needs them.
    rematerialized: dict[str, ast.Assign]
    definition_counts: dict[str, int]


class _Sinker:
    def __init__(
        self,
        vloop: ast.For,
        fact: CuteVloopWrapperFact,
        loads: Mapping[str, CuteVloopLoadFact],
        rename_groups: dict[str, str],
        new_var: Callable[..., str],
        lane_unroll: int,
        tensor_dtypes: Mapping[str, torch.dtype],
        proven_disjoint_tensor_pairs: Collection[frozenset[str]] = (),
        stores: Mapping[str, CuteVloopStoreFact] | None = None,
    ) -> None:
        self.vec_var = fact.vec_lane_var
        self.vec_width = fact.vec_width
        self.fact = fact
        self.loads = loads
        self.stores: Mapping[str, CuteVloopStoreFact] = stores or {}
        # Kernel tensor arguments by name: a subscript of one is a load.
        self.tensor_dtypes = tensor_dtypes
        # Tensor argument pairs whose storages cannot overlap
        # (``DeviceFunction.proven_disjoint_tensor_pairs``): a vector load
        # may move above a segment's stores into such a tensor.
        self.proven_disjoint_tensor_pairs = frozenset(proven_disjoint_tensor_pairs)
        self.new_var = new_var
        self.lane_unroll = lane_unroll
        self.fragment_allocs: list[ast.stmt] = []
        self.fragment_names: set[str] = set()
        # Scalars folded across the V lanes (``acc = acc + e``): kept scalar,
        # updated inside the V-loop of the segment holding the update.
        self.fold_names: set[str] = set()
        self.vector_loads_emitted = 0
        # Persistent-reduction vector markers of this wrapper's block nested
        # in the loops the V-loop encloses (``cute/persistent_branch_vec.py``
        # widens them once each sunk V-loop holds them directly).
        self.persistent_markers_nested = 0
        # Lane loops this pass unrolled itself (``_unroll_lane_loop``).
        self.lane_loops_unrolled = 0
        # Per segment: guard text -> names whose value that guard's
        # predicated packet load zero-filled (see ``_vectorize_loads``).
        self._zero_filled: dict[str, set[str]] = {}
        # Canonicalize loop-carried aliases (``v_1`` -> ``col_acc``) on a
        # private copy so the dataflow sees one name per carried value.
        body: list[ast.stmt] = []
        for stmt in vloop.body:
            if isinstance(stmt, ast.Pass):
                continue
            cloned = _clone(cast("ast.stmt", stmt))
            ast_rename(cloned, rename_groups)
            body.append(cloned)
        self.body = body

    # ------------------------------------------------------------------ analysis

    def run(self) -> list[ast.stmt] | None:
        body = self.body
        self._check_statement_forms(body)
        self._uniformize_vector_mask(body)
        self.persistent_markers_nested = self._nested_persistent_markers(body)
        self._define_chunk_mask(body)
        self._hoist_fold_masks(body)
        if not self._has_sinkable_nested_load(body) and not (
            self.persistent_markers_nested and self._index_definition_ok(body)
        ):
            return None
        varying = self._varying_names(body)
        exposed = self._exposed_reads(body, set())
        folds = exposed & _names_written_list(body)
        if folds:
            # A scalar carried across the V lanes (read before it is written
            # in the body) has no per-lane meaning after distribution, except
            # as a fold: ``acc = acc + e`` run once per lane in lane order
            # accumulates the lanes' values whichever segment it sits in,
            # provided nothing else reads the running value (``_lane_fold``).
            if not all(self._lane_fold(body, name) for name in folds):
                return None
            self.fold_names = folds
        varying = varying - folds
        # Once the V-loop is distributed, a name a segment reads before
        # writing it is carried across that segment's own V-loop.  Per-lane
        # values get fragments below; a V-invariant carried scalar (a
        # row-only accumulator ``wsum += w[row]``) would instead repeat its
        # update V times, so it becomes per-lane as well.
        carried = self._segment_carried_names(body) - varying - folds
        if carried:
            varying = self._varying_names(body, carried) - folds
        definitions = self._definitions(body)
        definition_counts = self._definition_counts(body)
        rematerialized = self._rematerializable(body, varying, definition_counts)
        expanded: dict[str, str] = {}
        fragment_dtypes: dict[str, str] = {}
        for name in sorted(self._segment_exposed_names(body, varying)):
            if name in rematerialized:
                continue
            dtype = self._infer_dtype(name, definitions, {}, set())
            if dtype is None:
                return None
            fragment = self.new_var(f"{name}_frag", dce=False)
            expanded[name] = fragment
            fragment_dtypes[fragment] = dtype
            self.fragment_names.add(fragment)
            self.fragment_allocs.append(
                statement_from_string(
                    f"{fragment} = cute.make_rmem_tensor({self.vec_width}, {dtype})"
                )
            )
        self.analysis = _Analysis(
            varying, expanded, fragment_dtypes, rematerialized, definition_counts
        )
        emitted = self._emit(body)
        if self.vector_loads_emitted == 0 and not self.persistent_markers_nested:
            return None
        return [*self.fragment_allocs, *emitted]

    def _hoist_fold_masks(self, body: list[ast.stmt]) -> None:
        """Move a lane-invariant select conjunct off a fold's loaded operand.

        ``t = load(...) if row_mask and lane_mask else identity`` inside a row
        lane loop, with ``lane_mask`` invariant across the rows and ``t`` read
        by exactly one statement, the fold ``acc = acc + t`` (or ``max`` /
        ``min`` with the matching identity), accumulates ``identity`` for the
        rows where ``lane_mask`` is false; the fold's result is therefore its
        entry value where ``lane_mask`` is false and the unmasked fold where it
        is true.  The select is marked for the widened load to drop
        (``_VSINK_FOLD_MASKED_ATTR``: the packet's lanes past ``lane_mask``
        then hold the loaded value, not ``identity``), the entry value is
        saved before the loop and the fold is re-masked after each update
        (``acc = acc if lane_mask else entry``, one select on the accumulator
        per row; see below for why not once after the loop): exact for any
        entry value, and the strip's loads no longer have a select behind each
        of them (which made ptxas reuse one destination register and serialize
        the rows).  The row mask stays on the select for the chunk
        guard to read; the load's own predicate already zero-fills those rows.
        """
        all_statements = self._walk_statements(body)
        for loop in all_statements:
            if (
                not isinstance(loop, ast.For)
                or getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None) is None
                or not isinstance(loop.target, ast.Name)
            ):
                continue
            loop_body = cast("list[ast.stmt]", loop.body)
            loop_writes = _names_written_list(loop_body) | {loop.target.id}
            for stmt in list(loop_body):
                target = _plain_target(stmt)
                if (
                    target is None
                    or not isinstance(stmt, ast.Assign)
                    or not isinstance(stmt.value, ast.IfExp)
                ):
                    continue
                select = stmt.value
                test = select.test
                if not (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)):
                    continue
                if not any(
                    self._sinkable_load(child) is not None
                    for child in ast.walk(select.body)
                ):
                    continue
                invariant = [
                    conjunct
                    for conjunct in test.values
                    if not _names_read(conjunct) & loop_writes
                ]
                if not invariant or len(invariant) == len(test.values):
                    continue
                fold = self._single_fold_consumer(
                    target, loop_body, all_statements, select.orelse
                )
                if fold is None:
                    continue
                acc, update = fold
                # Inside the loop only the update may touch the accumulator:
                # another write would not be a fold, another read (a store of
                # the running value, an ``if`` on it) would observe the
                # unmasked partial fold.
                touching = [
                    inner
                    for inner in self._walk_statements(loop_body)
                    if acc in _names_written(inner)
                    or acc in _statement_reads_outside_bodies(inner)
                ]
                if touching != [update]:
                    continue
                parent = self._statement_list_containing(body, loop)
                if parent is None:
                    continue
                entry = self.new_var(f"{acc}_entry", dce=False)
                mask = (
                    invariant[0]
                    if len(invariant) == 1
                    else ast.BoolOp(op=ast.And(), values=[_clone(c) for c in invariant])
                )
                index = parent.index(loop)
                parent.insert(index, statement_from_string(f"{entry} = {acc}"))
                # Re-masked after every update rather than once after the
                # loop: a per-lane write of the fold between the loop and
                # its cross-thread combine would keep that combine from being
                # hoisted out of the V-loop (``_hoist_fragment_reductions``).
                loop_body.insert(
                    loop_body.index(update) + 1,
                    statement_from_string(
                        f"{acc} = {acc} if {{mask}} else {entry}", mask=_clone(mask)
                    ),
                )
                setattr(select, _VSINK_FOLD_MASKED_ATTR, True)

    def _single_fold_consumer(
        self,
        name: str,
        loop_body: list[ast.stmt],
        all_statements: list[ast.stmt],
        identity: ast.expr,
    ) -> tuple[str, ast.stmt] | None:
        """``(acc, update)`` when ``name`` is read by exactly one statement of
        the sink body, a top-level statement of ``loop_body`` of the form
        ``acc = acc + name`` / ``acc = name + acc`` with ``identity`` zero; the
        operand may be wrapped in a dtype constructor.

        Only the zero identity is admitted: once the select is dropped, the
        rows past the row mask contribute the predicated packet load's zero
        fill, which is the identity of a sum but not of a max or min (an
        extremum fold's ``-inf`` / ``inf`` reaches the fold through a separate
        ``_mask_to`` select, which stays).  A read of ``name`` in an ``if``
        test or a loop header counts as a consumer.
        """
        readers = [
            stmt
            for stmt in all_statements
            if name in _statement_reads_outside_bodies(stmt)
        ]
        if len(readers) != 1 or readers[0] not in loop_body:
            return None
        update = readers[0]
        acc = _plain_target(update)
        if acc is None or not isinstance(update, ast.Assign) or acc == name:
            return None
        value = update.value
        if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)):
            return None
        if not _is_zero_literal(identity):
            return None
        operands = [value.left, value.right]
        carried = [
            operand
            for operand in operands
            if isinstance(operand, ast.Name) and operand.id == acc
        ]
        if len(carried) != 1:
            return None
        other = operands[1 - operands.index(carried[0])]
        if _ctor_operand_name(other) != name:
            return None
        return acc, update

    def _statement_list_containing(
        self, body: list[ast.stmt], stmt: ast.stmt
    ) -> list[ast.stmt] | None:
        if stmt in body:
            return body
        for outer in body:
            for field in ("body", "orelse"):
                inner = getattr(outer, field, None)
                if isinstance(inner, list):
                    found = self._statement_list_containing(inner, stmt)
                    if found is not None:
                        return found
        return None

    def _walk_statements(self, stmts: Sequence[ast.AST]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for stmt in stmts:
            result.append(cast("ast.stmt", stmt))
            for field in ("body", "orelse"):
                inner = getattr(stmt, field, None)
                if isinstance(inner, list):
                    result.extend(self._walk_statements(inner))
        return result

    def _definition_counts(self, stmts: Sequence[ast.AST]) -> dict[str, int]:
        """How many simple statements of ``stmts`` (nested ``if`` bodies and
        loop bodies included) write each name."""
        counts: dict[str, int] = {}
        for stmt in self._walk_statements(stmts):
            if isinstance(stmt, (ast.For, ast.If)):
                continue
            for name in _names_written(stmt):
                counts[name] = counts.get(name, 0) + 1
        return counts

    def _check_statement_forms(self, stmts: Sequence[ast.AST]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                if stmt.orelse or not isinstance(stmt.target, ast.Name):
                    raise _Unsupported
                self._check_statement_forms(stmt.body)
            elif isinstance(stmt, ast.If):
                if any(
                    isinstance(node, (ast.For, ast.While))
                    for node in ast.walk(stmt)
                    if node is not stmt
                ):
                    raise _Unsupported
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(
                    stmt.targets[0], (ast.Name, ast.Subscript)
                ):
                    raise _Unsupported
            elif isinstance(stmt, ast.AugAssign):
                if not isinstance(stmt.target, (ast.Name, ast.Subscript)):
                    raise _Unsupported
            elif not isinstance(stmt, (ast.Expr, ast.Pass)):
                raise _Unsupported
            if any(
                isinstance(node, (ast.NamedExpr, ast.Lambda, ast.ListComp, ast.Yield))
                for node in ast.walk(stmt)
            ):
                raise _Unsupported

    def _uniformize_vector_mask(self, body: list[ast.stmt]) -> None:
        """Evaluate the vectorized axis' bounds mask at the chunk base.

        With the tile extent a multiple of V and the chunk base V-aligned,
        ``base + v < extent`` holds for every ``v`` in ``[0, V)`` exactly when
        ``base < extent`` does, so the per-element mask is uniform across the
        V lanes.  Rewriting its definition makes it lane-invariant: it is then
        computed once and can guard whole vector loads.
        """
        fact = self.fact
        if fact.mask_var is None or not fact.uniform_vector_mask:
            return
        definitions = [
            stmt
            for stmt in self._walk_statements(body)
            if fact.mask_var in _names_written(stmt)
        ]
        if len(definitions) != 1 or definitions[0] not in body:
            return
        definition = definitions[0]
        if not isinstance(definition, ast.Assign) or _plain_target(definition) is None:
            return
        _RenameLoads({fact.index_var: fact.base_index_var}).visit(definition)

    def _lane_fold(self, body: list[ast.stmt], name: str) -> bool:
        """Whether ``name`` is only folded across the lanes of the body.

        Every statement of the body (nested loop and ``if`` bodies included)
        that reads or writes ``name`` or one of its copies (``alias = name``,
        ``alias_2 = alias``: the loop-carry renames) must be such a copy or a
        top-level ``name = carried + e`` / ``name = e + carried`` with
        ``carried`` the name or a copy and ``e`` reading none of them.  The
        copies are dropped and the update reads ``name`` itself, so the lanes'
        contributions join in lane order wherever the update's segment runs
        its V-loop, and no statement observes a partial fold.
        """
        aliases = {name}
        copies: list[ast.stmt] = []
        walked = self._walk_statements(body)
        statements = [
            stmt for stmt in walked if not isinstance(stmt, (ast.For, ast.If))
        ]
        # A read in an ``if`` test or a loop header observes the running value.
        if any(
            isinstance(stmt, (ast.For, ast.If))
            and name in _statement_reads_outside_bodies(stmt)
            for stmt in walked
        ):
            return False
        changed = True
        while changed:
            changed = False
            for stmt in statements:
                target = _plain_target(stmt)
                if (
                    isinstance(stmt, ast.Assign)
                    and target is not None
                    and isinstance(stmt.value, ast.Name)
                    and stmt.value.id in aliases
                    and stmt not in copies
                ):
                    copies.append(stmt)
                    if target not in aliases:
                        aliases.add(target)
                        changed = True
        updates: list[tuple[ast.stmt, ast.Name]] = []
        for stmt in statements:
            if stmt in copies:
                continue
            reads = _names_read(stmt)
            writes = _names_written(stmt)
            if not aliases & (reads | writes):
                continue
            if stmt not in body or not isinstance(stmt, ast.Assign):
                return False
            if _plain_target(stmt) != name:
                return False
            value = stmt.value
            if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)):
                return False
            operands = [value.left, value.right]
            carried = [
                operand
                for operand in operands
                if isinstance(operand, ast.Name) and operand.id in aliases
            ]
            if len(carried) != 1 or aliases & _names_read(
                operands[1 - operands.index(carried[0])]
            ):
                return False
            updates.append((stmt, carried[0]))
        if not updates:
            return False
        for _stmt, carried_name in updates:
            carried_name.id = name
        _remove_statements(body, set(map(id, copies)))
        return True

    def _define_chunk_mask(self, body: list[ast.stmt]) -> None:
        """Evaluate a non-uniform vectorized-axis mask at the chunk base too.

        A jagged tile's extent is not a multiple of V, so its per-element
        mask ``index < extent`` stays per lane (``_uniformize_vector_mask``
        leaves it).  The same bound at the chunk base, ``base < extent``, is
        lane-invariant and holds exactly when the chunk's first lane is in
        bounds; a load the program issues for that lane proves the aligned
        chunk lies inside its tensor (``whole_chunks_in_tensor``), so the
        packet load of such a chunk may be guarded by this mask while the
        per-lane selects keep masking the lanes past the extent.  Defined
        right after the mask when the mask is ``index_var < e`` (or
        ``e > index_var``) with ``e`` not reading the index.
        """
        fact = self.fact
        self.chunk_mask_var: str | None = None
        if fact.mask_var is None or fact.uniform_vector_mask:
            return
        definitions = [
            stmt
            for stmt in self._walk_statements(body)
            if fact.mask_var in _names_written(stmt)
        ]
        if len(definitions) != 1 or definitions[0] not in body:
            return
        definition = definitions[0]
        if not isinstance(definition, ast.Assign) or _plain_target(definition) is None:
            return
        test = definition.value
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and len(test.comparators) == 1
        ):
            return
        left, right = test.left, test.comparators[0]
        if isinstance(test.ops[0], ast.Lt):
            index, bound = left, right
        elif isinstance(test.ops[0], ast.Gt):
            index, bound = right, left
        else:
            return
        if (
            not (isinstance(index, ast.Name) and index.id == fact.index_var)
            or fact.index_var in _names_read(bound)
            or self.vec_var in _names_read(bound)
        ):
            return
        chunk_mask = self.new_var("_vsink_chunk_mask", dce=False)
        base_test = _clone(test)
        _RenameLoads({fact.index_var: fact.base_index_var}).visit(base_test)
        body.insert(
            body.index(definition) + 1,
            statement_from_string(f"{chunk_mask} = {{test}}", test=base_test),
        )
        self.chunk_mask_var = chunk_mask

    def _chunk_guard(self, test: ast.expr) -> ast.expr | None:
        """``test`` with the per-element mask replaced by the chunk mask when
        the mask is its only lane-varying term, else None."""
        chunk_mask = getattr(self, "chunk_mask_var", None)
        mask_var = self.fact.mask_var
        if chunk_mask is None or mask_var is None:
            return None
        varying = self.analysis.varying
        terms = (
            test.values
            if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
            else [test]
        )
        for term in terms:
            reads = _names_read(term)
            if isinstance(term, ast.Name) and term.id == mask_var:
                continue
            if self.vec_var in reads or reads & varying or mask_var in reads:
                return None
        guard = _clone(test)
        _RenameLoads({mask_var: chunk_mask}).visit(guard)
        return guard

    def _index_definition_ok(self, body: list[ast.stmt]) -> bool:
        """The per-element index is ``base + cutlass.Int32(vec_lane)``."""
        fact = self.fact
        definitions = [
            stmt
            for stmt in self._walk_statements(body)
            if fact.index_var in _names_written(stmt)
        ]
        if len(definitions) != 1 or definitions[0] not in body:
            return False
        definition = definitions[0]
        if not isinstance(definition, ast.Assign):
            return False
        expected = f"{fact.base_index_var} + cutlass.Int32({fact.vec_lane_var})"
        return ast.unparse(definition.value) == expected

    def _sinkable_load(self, node: ast.AST) -> CuteVloopLoadFact | None:
        """The load fact when ``node`` is a scalar load this pass may widen."""
        if not isinstance(node, ast.Call):
            return None
        pointer = _scalar_load_pointer(node)
        if pointer is None:
            return None
        fact = self.loads.get(ast.unparse(pointer))
        if fact is None or fact.vec_lane_var != self.vec_var:
            return None
        if self.fact.index_var not in _names_read(pointer):
            return None
        return fact

    def _sinkable_store(
        self, stmt: ast.AST
    ) -> tuple[ast.Call, ast.expr, CuteVloopStoreFact] | None:
        """``(call, pointer, fact)`` when ``stmt`` is ``(ptr).store(value)`` of
        a recorded per-lane store this pass may pack."""
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "store"
            and len(stmt.value.args) == 1
            and not stmt.value.keywords
        ):
            return None
        call = stmt.value
        pointer = cast("ast.Attribute", call.func).value
        fact = self.stores.get(ast.unparse(pointer))
        if fact is None or fact.vec_lane_var != self.vec_var:
            return None
        if self.fact.index_var not in _names_read(pointer):
            return None
        return call, pointer, fact

    def _has_sinkable_nested_load(self, body: list[ast.stmt]) -> bool:
        if not self._index_definition_ok(body):
            return False
        return any(
            isinstance(stmt, ast.For)
            and any(self._sinkable_load(node) is not None for node in ast.walk(stmt))
            for stmt in body
        )

    def _nested_persistent_markers(self, body: list[ast.stmt]) -> int:
        """Persistent vector markers of this wrapper's block inside the loops
        the V-loop encloses.

        A persistent one-vector lane marks its exact-fragment loads and
        stores for the late branch-local vectorizer instead of hoisting them
        here; that pass places a packet beside the constexpr lane loop whose
        body holds the marker directly, so a marker inside a nested lane loop
        (a full-slice head axis) only becomes a packet once this pass has
        interchanged the V-loop into that loop.  This pass widens none of
        them itself (``_sinkable_load`` matches plain scalar loads only).
        """
        block_id = self.fact.block_id
        count = 0
        for stmt in body:
            for node in ast.walk(stmt):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id.startswith("_helion_")
                    and node.func.id not in (_LOAD_MARKER, _STORE_MARKER)
                ):
                    # A lane reduction or another structural placeholder: the
                    # pass expanding it owns this nest.
                    return 0
        for stmt in body:
            if not isinstance(stmt, ast.For):
                continue
            for node in ast.walk(stmt):
                marker = _marker_call(node, _LOAD_MARKER, 6) or _marker_call(
                    node, _STORE_MARKER, 6
                )
                if (
                    marker is not None
                    and isinstance(marker.args[0], ast.Constant)
                    and marker.args[0].value == block_id
                ):
                    count += 1
        return count

    def _definitions(self, body: list[ast.stmt]) -> dict[str, list[ast.expr]]:
        definitions: dict[str, list[ast.expr]] = {}
        for stmt in self._walk_statements(body):
            if isinstance(stmt, (ast.For, ast.If)):
                # Compound statements only aggregate their nested writes.
                continue
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                definitions.setdefault(stmt.targets[0].id, []).append(stmt.value)
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                definitions.setdefault(stmt.target.id, []).append(
                    ast.BinOp(
                        left=ast.Name(id=stmt.target.id, ctx=ast.Load()),
                        op=stmt.op,
                        right=stmt.value,
                    )
                )
            else:
                for name in _names_written(stmt):
                    # A structured write (tensor element, tuple) has no
                    # scalar definition; record it as unknown.
                    definitions.setdefault(name, []).append(
                        ast.Name(id=_UNKNOWN, ctx=ast.Load())
                    )
        return definitions

    def _varying_names(
        self, body: list[ast.stmt], seeds: set[str] | None = None
    ) -> set[str]:
        """Names whose value may differ between V lanes: the fixpoint from
        the V-lane variable and ``seeds``."""
        units: list[tuple[set[str], set[str]]] = []

        def collect(stmts: Sequence[ast.AST], control: set[str]) -> None:
            for stmt in stmts:
                if isinstance(stmt, ast.For):
                    units.append((_names_read(stmt.iter) | control, {"__loop_iter__"}))
                    collect(stmt.body, control)
                elif isinstance(stmt, ast.If):
                    test_reads = _names_read(stmt.test) | control
                    collect(stmt.body, test_reads)
                    collect(stmt.orelse, test_reads)
                else:
                    units.append((_names_read(stmt) | control, _names_written(stmt)))

        collect(body, set())
        varying = {self.vec_var, *(seeds or ())}
        changed = True
        while changed:
            changed = False
            for reads, writes in units:
                if reads & varying:
                    if "__loop_iter__" in writes:
                        # A loop bound that depends on the V lane cannot be
                        # interchanged with the V-loop.
                        raise _Unsupported
                    new = writes - varying
                    if new:
                        varying |= new
                        changed = True
        varying.discard(self.vec_var)
        return varying

    def _exposed_reads(self, stmts: Sequence[ast.AST], defined: set[str]) -> set[str]:
        """Names read in ``stmts`` before ``stmts`` (or ``defined``) write them."""
        exposed: set[str] = set()
        defined = set(defined)
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                exposed |= _names_read(stmt.iter) - defined
                target = cast("ast.Name", stmt.target).id
                exposed |= self._exposed_reads(stmt.body, defined | {target}) - defined
                defined |= _names_written(stmt)
            elif isinstance(stmt, ast.If):
                exposed |= _names_read(stmt.test) - defined
                exposed |= self._exposed_reads(stmt.body, defined) - defined
                exposed |= self._exposed_reads(stmt.orelse, defined) - defined
                if stmt.orelse:
                    defined |= _names_written_list(stmt.body) & _names_written_list(
                        stmt.orelse
                    )
            else:
                exposed |= _names_read(stmt) - defined
                defined |= _names_written(stmt)
        return exposed

    def _segments(self, stmts: Sequence[ast.AST]) -> list[list[ast.stmt] | ast.For]:
        groups: list[list[ast.stmt] | ast.For] = []
        current: list[ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                if current:
                    groups.append(current)
                    current = []
                groups.append(stmt)
            elif not isinstance(stmt, ast.Pass):
                current.append(cast("ast.stmt", stmt))
        if current:
            groups.append(current)
        return groups

    def _segment_exposed_names(
        self, body: list[ast.stmt], varying: set[str]
    ) -> set[str]:
        """V-lane-varying names that flow between segments.

        Each segment becomes its own constexpr V-loop, so a varying name that
        a segment reads before writing it (loop-carried, or produced by an
        earlier segment) must be recomputed or live in a per-lane fragment.
        """
        exposed: set[str] = set()

        def visit(stmts: Sequence[ast.AST], loop_targets: set[str]) -> None:
            for group in self._segments(stmts):
                if isinstance(group, ast.For):
                    target = cast("ast.Name", group.target).id
                    visit(group.body, loop_targets | {target})
                else:
                    exposed.update(self._exposed_reads(group, loop_targets))

        visit(body, set())
        return exposed & varying

    def _segment_carried_names(self, body: list[ast.stmt]) -> set[str]:
        """Names some segment both reads before writing and writes."""
        carried: set[str] = set()

        def visit(stmts: Sequence[ast.AST], loop_targets: set[str]) -> None:
            for group in self._segments(stmts):
                if isinstance(group, ast.For):
                    target = cast("ast.Name", group.target).id
                    visit(group.body, loop_targets | {target})
                else:
                    carried.update(
                        self._exposed_reads(group, loop_targets)
                        & _names_written_list(group)
                    )

        visit(body, set())
        return carried

    def _rematerializable(
        self, body: list[ast.stmt], varying: set[str], counts: dict[str, int]
    ) -> dict[str, ast.Assign]:
        """Varying names with one pure definition over lane-external inputs.

        Such a definition (the per-element index ``base + vec_lane``, a mask
        derived from it) can be recomputed at the start of any segment that
        reads it instead of occupying a register fragment.
        """
        written = _names_written_list(body)
        candidates: dict[str, ast.Assign] = {}
        for stmt in body:
            name = _plain_target(stmt)
            if (
                name is not None
                and isinstance(stmt, ast.Assign)
                and name in varying
                and counts.get(name) == 1
                and _is_relocatable_statement(
                    stmt, allow_load=False, tensor_names=self.tensor_dtypes
                )
            ):
                candidates[name] = stmt
        result: dict[str, ast.Assign] = {}
        changed = True
        while changed:
            changed = False
            for name, stmt in candidates.items():
                if name in result:
                    continue
                reads = _names_read(stmt) - {self.vec_var}
                if all(read not in written or read in result for read in reads):
                    result[name] = stmt
                    changed = True
        return result

    def _infer_dtype(
        self,
        name: str,
        definitions: dict[str, list[ast.expr]],
        cache: dict[str, str | None],
        visiting: set[str],
    ) -> str | None:
        if name in cache:
            return cache[name]
        if name in visiting or name not in definitions:
            return None
        visiting.add(name)
        result: str | None = None
        for value in definitions[name]:
            dtype = self._expr_dtype(value, definitions, cache, visiting)
            if dtype == _CONFLICT or (
                dtype is not None and result is not None and dtype != result
            ):
                result = _CONFLICT
                break
            if dtype is not None:
                result = dtype
        visiting.discard(name)
        if result == _CONFLICT:
            result = None
        cache[name] = result
        return result

    def _expr_dtype(
        self,
        node: ast.expr,
        definitions: dict[str, list[ast.expr]],
        cache: dict[str, str | None],
        visiting: set[str],
    ) -> str | None:
        def combine(*dtypes: str | None) -> str | None:
            known = {dtype for dtype in dtypes if dtype is not None}
            if _CONFLICT in known or len(known) > 1:
                return _CONFLICT
            return next(iter(known), None)

        if isinstance(node, ast.Name):
            if node.id == _UNKNOWN:
                return _CONFLICT
            return self._infer_dtype(node.id, definitions, cache, visiting)
        if isinstance(node, ast.Call):
            func = ast.unparse(node.func)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "bitcast":
                return ast.unparse(node.args[0]) if node.args else _CONFLICT
            if (fact := self._sinkable_load(node)) is not None:
                # A scalar load this pass widens: the tensor's element type
                # (the per-lane value is the bitcast of one vector element).
                return CompileEnvironment.current().backend.dtype_str(fact.dtype)
            if (
                func == "_helion_persistent_branch_vec_load"
                and len(node.args) >= 3
                and isinstance(node.args[2], ast.Constant)
                and isinstance(node.args[2].value, str)
            ):
                # The persistent branch vectorizer's load marker names its
                # element type (``persistent_branch_vec.py``).
                return node.args[2].value
            if func == "_cute_inline_asm_elementwise":
                # The explicitly rounded products of ``scalar_recipe_rounding``
                # declare their result type.
                declared = next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "dtype"
                    ),
                    None,
                )
                if declared is not None and _is_dtype_ctor(ast.unparse(declared)):
                    return ast.unparse(declared)
                return _CONFLICT
            if _is_dtype_ctor(func):
                return func
            if func.startswith(_DTYPE_PRESERVING_CALLS) and node.args:
                return self._expr_dtype(node.args[0], definitions, cache, visiting)
            if func in ("max", "min"):
                return combine(
                    *(
                        self._expr_dtype(arg, definitions, cache, visiting)
                        for arg in node.args
                    )
                )
            return _CONFLICT
        if isinstance(node, ast.BinOp):
            return combine(
                self._expr_dtype(node.left, definitions, cache, visiting),
                self._expr_dtype(node.right, definitions, cache, visiting),
            )
        if isinstance(node, ast.IfExp):
            return combine(
                self._expr_dtype(node.body, definitions, cache, visiting),
                self._expr_dtype(node.orelse, definitions, cache, visiting),
            )
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return "cutlass.Boolean"
            return self._expr_dtype(node.operand, definitions, cache, visiting)
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return "cutlass.Boolean"
        if isinstance(node, ast.Constant):
            return None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            # An element of a kernel tensor argument (``t[i]``) has the
            # tensor's element type.
            tensor_dtype = self.tensor_dtypes.get(node.value.id)
            if tensor_dtype is not None:
                return CompileEnvironment.current().backend.dtype_str(tensor_dtype)
        return _CONFLICT

    # ------------------------------------------------------------------ emission

    def _emit(self, stmts: Sequence[ast.AST]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for group in self._segments(stmts):
            if isinstance(group, ast.For):
                result.extend(self._emit_loop(group))
            else:
                result.extend(self._emit_segment(group))
        return result

    def _emit_loop(self, loop: ast.For) -> list[ast.stmt]:
        inner = self._emit(loop.body)
        lane_var = getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None)
        target = cast("ast.Name", loop.target).id
        pre: list[ast.stmt] = []
        vector_loads = [
            stmt for stmt in inner if getattr(stmt, _VSINK_VECTOR_LOAD_ATTR, False)
        ]
        if vector_loads:
            pre.extend(self._strength_reduce(inner, target, vector_loads))
        new_loop = create(
            ast.For,
            target=create(ast.Name, id=target, ctx=ast.Store()),
            iter=_clone(loop.iter),
            body=inner,
            orelse=[],
            type_comment=None,
        )
        if lane_var is not None:
            setattr(new_loop, HELION_LANE_LOOP_VAR_ATTR, lane_var)
            extent = _lane_loop_extent(loop)
            if vector_loads and extent is not None:
                new_loop = self._unroll_lane_loop(new_loop, extent)
        return [*pre, new_loop]

    def _emit_segment(self, segment: list[ast.stmt]) -> list[ast.stmt]:
        analysis = self.analysis
        vec_var = self.vec_var
        pre: list[ast.stmt] = []
        inside: list[ast.stmt] = []
        vector_cache: dict[str, str] = {}
        self._zero_filled = {}
        # Distribution turns the per-lane execution of a hoisted statement
        # into one execution before this segment's V-loop.  For that to be
        # exact, the statement must be the only definition of the names it
        # writes in the segment (nested ``if`` bodies included): a second
        # definition kept inside, before or after it, would reach the later
        # lanes.  It may only read names defined outside the segment or by
        # statements hoisted ahead of it, and it must not redefine a name a
        # statement kept inside has read.  The same holds for the operands of
        # a widened load.
        definition_counts = self._definition_counts(segment)
        segment_writes = set(definition_counts)
        single_definitions = frozenset(
            name for name, count in definition_counts.items() if count == 1
        )
        # Names the hoisted statements have defined so far.
        pre_writes: set[str] = set()
        # Names the statements kept inside so far have read.
        inside_reads: set[str] = set()
        # In the original order every effect of the segment in lane ``v``
        # precedes the loads of lanes ``v + 1, ...``, so a load moved before
        # the V-loop crosses the segment's effects in both directions: loads
        # only move out of a segment without any memory effect.
        has_effects = any(self._has_memory_effect(stmt) for stmt in segment)
        stored_tensors = (
            self._segment_stored_tensors(segment) if has_effects else frozenset()
        )
        # Recompute cheap per-lane values this segment reads before defining.
        needed = self._exposed_reads(segment, set()) & set(analysis.rematerialized)
        for name in self._rematerialization_order(needed):
            recomputed = _clone(analysis.rematerialized[name])
            inside.append(recomputed)
            inside_reads |= _names_read(recomputed)
            segment_writes |= _names_written(recomputed)
        post: list[ast.stmt] = []
        packable_stores = self._packable_store_statements(
            segment, stored_tensors, has_effects
        )
        for stmt in segment:
            reads = _names_read(stmt)
            writes = _names_written(stmt)
            rewritten = _clone(stmt)
            if stmt in packable_stores:
                packed = self._pack_store(
                    rewritten,
                    packable_stores[stmt],
                    pre,
                    post,
                    segment_writes - pre_writes,
                )
                if packed is not None:
                    inside.append(packed)
                    inside_reads |= reads
                    continue
            if (
                vec_var not in reads
                and not reads & analysis.varying
                and not writes & analysis.varying
            ):
                # Lane-invariant: run once before the V-loop when its calls
                # are relocatable and nothing kept inside is ordered against
                # it; otherwise keep its per-lane execution and order.
                if (
                    _is_relocatable_statement(
                        rewritten,
                        allow_load=not has_effects,
                        tensor_names=self.tensor_dtypes,
                    )
                    and all(definition_counts.get(name) == 1 for name in writes)
                    and not reads & (segment_writes - pre_writes)
                    and not writes & inside_reads
                ):
                    pre.append(rewritten)
                    pre_writes |= writes
                    continue
            else:
                self._vectorize_loads(
                    rewritten,
                    vector_cache,
                    pre,
                    segment_writes - pre_writes,
                    has_effects,
                    stored_tensors,
                    single_definitions,
                )
                _ExpandNames(analysis.expanded, vec_var).visit(rewritten)
            inside.append(rewritten)
            inside_reads |= reads
        inside = self._hoist_fragment_reductions(inside, pre)
        result = list(pre)
        if inside:
            result.append(_constexpr_loop(vec_var, self.vec_width, inside))
        result.extend(post)
        return result

    def _packable_store_statements(
        self,
        segment: list[ast.stmt],
        stored_tensors: frozenset[str] | None,
        has_effects: bool,
    ) -> dict[ast.stmt, tuple[ast.Call, ast.expr, CuteVloopStoreFact, ast.expr | None]]:
        """The segment's per-lane stores whose V lanes may flush as one packet.

        A store moved after the V-loop crosses every later lane's statements
        of the segment, so it is packed only when nothing else in the segment
        can observe it: every other memory effect of the segment is a plain
        store (``stored_tensors`` is known) into a tensor proven disjoint from
        the stored one, no other store site writes the stored tensor, and
        every load of the segment addresses tensors proven disjoint from it
        (the vector loads this pass hoists included).  The guard of a masked
        store must be V-lane-invariant: the flush is predicated by it.
        """
        from .fuse_two_pass_loads import _tensor_arg_roots

        result: dict[
            ast.stmt, tuple[ast.Call, ast.expr, CuteVloopStoreFact, ast.expr | None]
        ] = {}
        if not has_effects or stored_tensors is None:
            return result
        varying = self.analysis.varying
        tensor_names = set(self.tensor_dtypes)
        candidates: list[
            tuple[ast.stmt, ast.Call, ast.expr, CuteVloopStoreFact, ast.expr | None]
        ] = []
        for stmt in segment:
            match = self._sinkable_store(stmt)
            guard: ast.expr | None = None
            if match is None and (
                isinstance(stmt, ast.If)
                and len(stmt.body) == 1
                and all(isinstance(inner, ast.Pass) for inner in stmt.orelse)
                and not any(isinstance(node, ast.Call) for node in ast.walk(stmt.test))
            ):
                match = self._sinkable_store(stmt.body[0])
                guard = stmt.test
                if match is not None:
                    reads = _names_read(guard)
                    if self.vec_var in reads or reads & varying:
                        match = None
            if match is not None:
                candidates.append((stmt, *match, guard))
        if not candidates:
            return result
        # Tensors the segment's loads address (None: an unresolved pointer).
        loaded: set[str] = set()
        for stmt in segment:
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Call):
                    continue
                pointer = _scalar_load_pointer(node)
                if pointer is None:
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "load"
                    ) or ast.unparse(node.func) == "cute.arch.load":
                        return result
                    continue
                roots = _tensor_arg_roots(pointer, tensor_names)
                if roots is None:
                    return result
                loaded |= roots
        store_sites: dict[str, int] = {}
        for _stmt, _call, _pointer, fact, _guard in candidates:
            store_sites[fact.tensor_name] = store_sites.get(fact.tensor_name, 0) + 1
        for stmt, call, pointer, fact, guard in candidates:
            tensor = fact.tensor_name
            others = (stored_tensors - {tensor}) | loaded
            if store_sites[tensor] != 1 or any(
                frozenset((tensor, other)) not in self.proven_disjoint_tensor_pairs
                for other in others
            ):
                continue
            result[stmt] = (call, pointer, fact, guard)
        return result

    def _pack_store(
        self,
        stmt: ast.stmt,
        match: tuple[ast.Call, ast.expr, CuteVloopStoreFact, ast.expr | None],
        pre: list[ast.stmt],
        post: list[ast.stmt],
        unavailable: set[str],
    ) -> ast.stmt | None:
        """Turn a per-lane store into an append and queue its packet flush.

        Inside the V-loop the lane's value is appended to a trace-time list;
        after the loop one ``_cute_store_u32_vec`` / ``_cute_store_u16_vec``
        writes the V lanes at the chunk base, under the store's own
        V-invariant guard.  None when the address needs a per-lane value.
        """
        call, pointer, fact, guard = match
        vector_pointer = _clone(pointer)
        _RenameLoads({self.fact.index_var: self.fact.base_index_var}).visit(
            vector_pointer
        )
        needed = _names_read(vector_pointer)
        if guard is not None:
            needed = needed | _names_read(guard)
        if (
            self.vec_var in needed
            or needed & self.analysis.varying
            or needed & unavailable
        ):
            return None
        list_var = self.new_var("_vsink_store_vals", dce=False)
        pre.append(statement_from_string(f"{list_var} = []"))
        carrier = _CUTE_VECTOR_UNROLL_CARRIER[fact.dtype]
        helper = (
            "_cute_store_u32_vec" if fact.dtype.itemsize == 4 else "_cute_store_u16_vec"
        )
        flush = f"{helper}({ast.unparse(vector_pointer)}, {list_var})"
        if guard is not None:
            post.append(statement_from_string(f"if {ast.unparse(guard)}:\n    {flush}"))
        else:
            post.append(statement_from_string(flush))
        value = _clone(call.args[0])
        _ExpandNames(self.analysis.expanded, self.vec_var).visit(value)
        append = statement_from_string(
            f"{list_var}.append(({{value}}).bitcast({carrier}))", value=value
        )
        self.vector_loads_emitted += 1
        return append

    def _has_memory_effect(self, stmt: ast.stmt) -> bool:
        """Whether ``stmt`` stores, synchronizes or otherwise acts on memory;
        writes into this pass's register fragments do not count.

        An assignment counts when it writes a tensor in place or calls
        anything outside the proven pure / load set: an atomic whose result
        is bound to a name (``old = cute.arch.atomic_add(...)``) or a
        shared-memory reduce is such a call.  Every other statement form
        (``if`` blocks, bare expressions) counts.
        """
        from ..tile_strategy import _has_side_effect
        from ..tile_strategy import _is_proven_relocatable_call

        if isinstance(stmt, ast.Assign):
            inplace = set(ReadWrites.from_ast(stmt).inplace_writes)
            if inplace - self.fragment_names:
                return True
            return not all(
                _is_proven_relocatable_call(node, allow_load=True)
                for node in ast.walk(stmt)
                if isinstance(node, ast.Call)
            )
        return _has_side_effect(stmt)

    def _rematerialization_order(self, names: set[str]) -> list[str]:
        """``names`` plus their rematerialized inputs, producers first."""
        table = self.analysis.rematerialized
        ordered: list[str] = []
        seen: set[str] = set()

        def visit(name: str) -> None:
            if name in seen or name not in table:
                return
            seen.add(name)
            for read in sorted(_names_read(table[name])):
                visit(read)
            ordered.append(name)

        for name in sorted(names):
            visit(name)
        return ordered

    def _vectorize_loads(
        self,
        stmt: ast.stmt,
        vector_cache: dict[str, str],
        pre: list[ast.stmt],
        unavailable: set[str],
        has_effects: bool,
        stored_tensors: frozenset[str] | None = frozenset(),
        single_definitions: frozenset[str] = frozenset(),
    ) -> None:
        """Replace sinkable scalar loads in ``stmt`` by vector-lane extracts.

        The vector load itself is V-lane-invariant (it addresses the chunk
        base) and is appended to ``pre``.  A load is only widened when every
        V lane executes it: unconditionally, or under a V-invariant guard,
        which then selects a safe anchor pointer for the transaction while
        the per-lane select keeps its meaning.  A load under a per-lane
        condition (a per-element bounds mask) has no whole-chunk equivalent,
        so the body is left untouched.  ``unavailable`` names the segment's
        definitions that are not hoisted ahead of ``stmt``; ``has_effects``
        says whether the segment has any memory effect, and
        ``stored_tensors`` which tensors those effects write (None when an
        effect is not a plain store): a load of a tensor proven disjoint from
        every one of them may still move above them, any other load of an
        effectful segment keeps its scalar form in lane order.
        """
        sinker = self
        varying = self.analysis.varying
        # Names bound to a value the predicated packet load zero-filled under
        # a guard (``x = vec[v]`` after its ``if guard else 0`` select was
        # dropped): a later ``x if guard else 0`` repeats the fill.
        zero_filled = self._zero_filled
        # The name a dropped select's zero fill is recorded under: only the
        # standalone form ``t = load if guard else 0`` of a name the segment
        # defines once (a select nested in a larger expression, or a ``t``
        # redefined later, binds ``t`` to something else where the guard is
        # false).
        target = (
            stmt.targets[0].id
            if isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.IfExp)
            and stmt.targets[0].id in single_definitions
            else None
        )
        standalone_select = stmt.value if isinstance(stmt, ast.Assign) else None

        def contains_load(node: ast.AST) -> bool:
            return any(
                sinker._sinkable_load(child) is not None for child in ast.walk(node)
            )

        def repeats_zero_fill(body: ast.expr, guard: str) -> bool:
            name = _ctor_operand_name(body)
            return name is not None and name in zero_filled.get(guard, set())

        def invariant(node: ast.expr) -> bool:
            reads = _names_read(node)
            return sinker.vec_var not in reads and not reads & varying

        class Rewrite(ast.NodeTransformer):
            def __init__(self) -> None:
                self.guard: ast.expr | None = None

            def visit_If(self, node: ast.If) -> ast.AST:
                if contains_load(node):
                    raise _Unsupported
                return node

            def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
                # Operands after the first are evaluated conditionally.
                if any(contains_load(value) for value in node.values[1:]):
                    raise _Unsupported
                node.values[0] = self.visit(node.values[0])
                return node

            def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
                if contains_load(node.test) or contains_load(node.orelse):
                    raise _Unsupported
                if not invariant(node.test):
                    if contains_load(node.body):
                        guard = sinker._chunk_guard(node.test)
                        if (
                            guard is None
                            or self.guard is not None
                            or any(
                                not fact.whole_chunks_in_tensor
                                for fact in (
                                    sinker._sinkable_load(child)
                                    for child in ast.walk(node.body)
                                )
                                if fact is not None
                            )
                        ):
                            raise _Unsupported
                        # The chunk straddles the tile's extent: load it
                        # when its first lane is in bounds and keep the
                        # per-lane select for the lanes past the extent.
                        self.guard = guard
                        node.body = self.visit(node.body)
                        self.guard = None
                        if getattr(
                            node, _VSINK_FOLD_MASKED_ATTR, False
                        ) and _is_lane_extract(node.body):
                            # The packet was widened under the chunk guard
                            # (rows past the row mask are zero-filled) and its
                            # lanes past the extent reach only a fold re-masked
                            # after each update (``_hoist_fold_masks``).  A load
                            # kept scalar (an effectful segment) keeps its
                            # select: it would otherwise read rows past the
                            # row mask unconditionally.
                            return node.body
                        return node
                    node.test = self.visit(node.test)
                    return node
                outer = self.guard
                self.guard = (
                    node.test
                    if outer is None
                    else ast.BoolOp(op=ast.And(), values=[outer, node.test])
                )
                widened = contains_load(node.body)
                node.body = self.visit(node.body)
                self.guard = outer
                guard_text = ast.unparse(node.test)
                if (
                    outer is None
                    and _is_zero_literal(node.orelse)
                    and (
                        (widened and _is_lane_extract(node.body))
                        or repeats_zero_fill(node.body, guard_text)
                    )
                ):
                    # The predicated packet load already left zeros in the
                    # lanes the guard masks off; the select would only make
                    # the loaded register's first use immediate.
                    if target is not None and node is standalone_select:
                        zero_filled.setdefault(guard_text, set()).add(target)
                    return node.body
                return node

            def visit_Call(self, node: ast.Call) -> ast.AST:
                load_fact = sinker._sinkable_load(node)
                if load_fact is None:
                    return self.generic_visit(node)
                if has_effects and not sinker._load_crosses_stores(
                    load_fact.tensor_name, stored_tensors
                ):
                    # The transaction would move across an effect of this
                    # segment; the per-lane scalar load keeps its order.
                    return node
                return sinker._emit_vector_load(
                    node,
                    load_fact,
                    self.guard,
                    vector_cache,
                    pre,
                    unavailable,
                )

        Rewrite().visit(stmt)

    def _load_crosses_stores(
        self, tensor_name: str, stored_tensors: frozenset[str] | None
    ) -> bool:
        """Whether a load of ``tensor_name`` may move above the segment's effects.

        Only when every effect is a plain store into a tensor proven disjoint
        from the loaded one: the hoisted transaction then reads bytes no
        store of the segment can change, in any order.
        """
        return stored_tensors is not None and all(
            frozenset((tensor_name, stored)) in self.proven_disjoint_tensor_pairs
            for stored in stored_tensors
        )

    def _segment_stored_tensors(
        self, segment: Sequence[ast.stmt]
    ) -> frozenset[str] | None:
        """The tensors the segment's memory effects write, or None.

        An effect is accepted when it is a ``ptr.store(value)`` call on a
        tensor argument's pointer, bare or as the only body of an ``if``
        (a masked store), whose operands carry no other effect.  Any other
        effect (an atomic, a barrier, a shared-memory reduce, a store through
        an unresolved pointer) makes the set unknown.
        """
        from ..tile_strategy import _is_proven_relocatable_call
        from .fuse_two_pass_loads import _is_store_call
        from .fuse_two_pass_loads import _store_tensor_roots

        tensor_names = set(self.tensor_dtypes)
        stored: set[str] = set()

        def store_statement(stmt: ast.stmt) -> bool:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                return False
            call = stmt.value
            if not (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "store"
                and _is_store_call(call)
            ):
                return False
            roots = _store_tensor_roots(call, tensor_names)
            if roots is None or any(
                isinstance(node, ast.Call)
                and node is not call
                and not _is_proven_relocatable_call(node, allow_load=False)
                for node in ast.walk(call)
            ):
                return False
            stored.update(roots)
            return True

        for stmt in segment:
            if not self._has_memory_effect(stmt):
                continue
            if store_statement(stmt):
                continue
            if (
                isinstance(stmt, ast.If)
                and stmt.body
                and all(
                    store_statement(inner) or isinstance(inner, ast.Pass)
                    for inner in stmt.body
                )
                and all(isinstance(inner, ast.Pass) for inner in stmt.orelse)
                and not any(isinstance(node, ast.Call) for node in ast.walk(stmt.test))
            ):
                continue
            return None
        return frozenset(stored)

    def _emit_vector_load(
        self,
        call: ast.Call,
        load_fact: CuteVloopLoadFact,
        mask: ast.expr | None,
        vector_cache: dict[str, str],
        pre: list[ast.stmt],
        unavailable: set[str],
    ) -> ast.expr:
        pointer = _scalar_load_pointer(call)
        assert pointer is not None
        vector_pointer = _clone(pointer)
        _RenameLoads({self.fact.index_var: self.fact.base_index_var}).visit(
            vector_pointer
        )
        needed = _names_read(vector_pointer)
        if mask is not None:
            needed = needed | _names_read(mask)
        if (
            self.vec_var in needed
            or needed & self.analysis.varying
            or needed & unavailable
        ):
            # The address or guard differs per lane, or needs a value this
            # segment defines inside its V-loop (before or after the load).
            raise _Unsupported
        pointer_text = ast.unparse(vector_pointer)
        guard = ast.unparse(mask) if mask is not None else ""
        key = f"{pointer_text}|{guard}"
        vector_var = vector_cache.get(key)
        if vector_var is None:
            vector_var = self.new_var("_vsink_vec", dce=False)
            if mask is not None:
                # One predicated ``@p ld.global`` that leaves zeros when the
                # guard is false (``cute/masked_packet_load.py``): the packet
                # is written once and has no consumer until the arithmetic
                # needs it, so ptxas keeps every row's load in flight.  The
                # per-lane selects on the value stay correct and are dropped
                # where they only repeat the zero fill (``visit_IfExp``).
                load_expr = _masked_packet_load_expr(
                    pointer_text,
                    guard,
                    load_fact.dtype,
                    load_fact.vec_width,
                    load_fact.eviction_suffix,
                )
            else:
                load_expr = _cute_unroll_vec_load_expr(
                    pointer_text,
                    load_fact.dtype,
                    load_fact.vec_width,
                    load_fact.eviction_suffix,
                )
            load_stmt = statement_from_string(f"{vector_var} = {load_expr}")
            setattr(load_stmt, _VSINK_VECTOR_LOAD_ATTR, True)
            pre.append(load_stmt)
            vector_cache[key] = vector_var
            self.vector_loads_emitted += 1
        carrier = _CUTE_VECTOR_UNROLL_CARRIER[load_fact.dtype]
        element = _CUTE_VECTOR_UNROLL_DTYPES[load_fact.dtype]
        return cast(
            "ast.expr",
            expr_from_string(
                f"{carrier}({vector_var}[{self.vec_var}]).bitcast({element})"
            ),
        )

    def _hoist_fragment_reductions(
        self, inside: list[ast.stmt], pre: list[ast.stmt]
    ) -> list[ast.stmt]:
        """Combine all V accumulators of a segment with one grouped reduce.

        ``R = wrap(_cute_grouped_reduce_shared_two_stage(acc_frag[v], ...))``
        inside the V-loop pays the two-barrier two-stage reduce V times.
        When the input is an expanded fragment and no other statement of the
        segment has a memory effect, the fragment form reduces all V values
        with one pair of barriers before the V-loop and the loop reads
        ``R = wrap(R_frag[v])``.
        """
        analysis = self.analysis
        result: list[ast.stmt] = []
        # The hoisted reduce runs before the V-loop: it cannot read a name
        # any statement kept inside defines (before or after it: the later
        # lanes would have seen that definition), and it does not move
        # across a memory effect of the segment.
        inside_writes = _names_written_list(inside)
        if any(
            self._has_memory_effect(stmt)
            for stmt in inside
            if _fragment_reduce_match(stmt, self.vec_var, self.fragment_names) is None
        ):
            return inside
        for stmt in inside:
            match = _fragment_reduce_match(stmt, self.vec_var, self.fragment_names)
            if match is not None:
                call, source_fragment, pre_count, group_span, group_count = match
                if (
                    pre_count > 0
                    and not pre_count & (pre_count - 1)
                    and pre_count <= 32
                    and group_span % 32 == 0
                    and pre_count * self.vec_width <= group_span
                    and not any(
                        self.vec_var in _names_read(arg)
                        or _names_read(arg) & analysis.varying
                        for arg in call.args[1:]
                    )
                    and not (_names_read(call) - {self.vec_var}) & inside_writes
                ):
                    target = cast("str", _plain_target(stmt))
                    result_fragment = self.new_var(f"{target}_frag", dce=False)
                    dtype = analysis.fragment_dtypes[source_fragment]
                    self.fragment_allocs.append(
                        statement_from_string(
                            f"{result_fragment} = cute.make_rmem_tensor("
                            f"{self.vec_width}, {dtype})"
                        )
                    )
                    args = ", ".join(ast.unparse(arg) for arg in call.args[1:])
                    pre.append(
                        statement_from_string(
                            f"{_TWO_STAGE_REDUCE_FRAGMENT}({source_fragment}, "
                            f"{result_fragment}, {args}, count={self.vec_width}, "
                            f"pre={pre_count}, group_span={group_span}, "
                            f"group_count={group_count})"
                        )
                    )
                    replaced = _clone(stmt)
                    _ReplaceCall(call, f"{result_fragment}[{self.vec_var}]").visit(
                        replaced
                    )
                    result.append(replaced)
                    continue
            result.append(stmt)
        return result

    def _strength_reduce(
        self, body: list[ast.stmt], lane_var: str, vector_loads: list[ast.stmt]
    ) -> list[ast.stmt]:
        """Address the loop's vector loads from a hoisted base pointer.

        ``T.iterator + Int32(A + Int32(lane) * C) * Int32(s_r) + rest`` with
        ``A``, ``C`` and ``rest`` lane-invariant becomes ``base + Int32(lane) *
        step`` with ``base = T.iterator + Int32(A) * Int32(s_r) + rest`` and
        ``step = Int32(C) * Int32(s_r)`` computed once before the loop, which
        drops the per-row index arithmetic to one multiply-add.
        """
        pre: list[ast.stmt] = []
        body_writes = _names_written_list(body)
        definitions = {
            name: stmt.value
            for stmt in body
            if isinstance(stmt, ast.Assign)
            and (name := _plain_target(stmt)) is not None
        }
        for load_stmt in vector_loads:
            assert isinstance(load_stmt, ast.Assign)
            for pointer in _maximal_pointer_sums(load_stmt.value):
                reduced = _reduce_pointer(
                    pointer, lane_var, definitions, body_writes, self.new_var
                )
                if reduced is None:
                    continue
                base_stmt, step_stmt, replacement = reduced
                pre.extend((base_stmt, step_stmt))
                _ReplaceNode(pointer, replacement).visit(load_stmt)
        return pre

    def _unroll_lane_loop(self, loop: ast.For, extent: int) -> ast.For:
        """Unroll by the ``cute_lane_unroll`` factor, issuing every copy's
        vector loads before any copy's compute."""
        unroll = 1
        for choice in _CUTE_LANE_UNROLL_CHOICES:
            if choice <= self.lane_unroll and extent % choice == 0:
                unroll = choice
        if unroll <= 1:
            return loop
        lane_var = cast("ast.Name", loop.target).id
        body = cast("list[ast.stmt]", loop.body)
        writes = _names_written_list(body)
        carried = self._exposed_reads(body, {lane_var}) & writes
        if carried - self.fragment_names:
            # A scalar carried across lane iterations cannot be renamed per
            # copy; keep the rolled loop.
            return loop
        self.lane_loops_unrolled += 1
        renamable = sorted(writes - self.fragment_names)
        heads: list[list[ast.stmt]] = []
        tails: list[list[ast.stmt]] = []
        for copy_index in range(unroll):
            renames = {
                name: self.new_var(f"{name}_u{copy_index}", dce=True)
                for name in renamable
            }
            copy: list[ast.stmt] = []
            for stmt in body:
                cloned = _clone(stmt)
                ast_rename(cloned, renames)
                _SubstituteLane(lane_var, unroll, copy_index).visit(cloned)
                copy.append(cloned)
            head_indices = {
                index
                for index, stmt in enumerate(copy)
                if getattr(stmt, _VSINK_VECTOR_LOAD_ATTR, False)
            }
            needed: set[str] = set()
            for index in head_indices:
                needed |= _names_read(copy[index])
            for index in range(len(copy) - 1, -1, -1):
                if index in head_indices:
                    continue
                stmt_writes = _names_written(copy[index])
                if stmt_writes & needed and not stmt_writes & self.fragment_names:
                    head_indices.add(index)
                    needed |= _names_read(copy[index])
            heads.append([copy[i] for i in range(len(copy)) if i in head_indices])
            tails.append([copy[i] for i in range(len(copy)) if i not in head_indices])
        new_body: list[ast.stmt] = []
        for head in heads:
            new_body.extend(head)
        for tail in tails:
            new_body.extend(tail)
        new_loop = create(
            ast.For,
            target=create(ast.Name, id=lane_var, ctx=ast.Store()),
            iter=expr_from_string(f"range({extent // unroll})"),
            body=new_body,
            orelse=[],
            type_comment=None,
        )
        setattr(
            new_loop,
            HELION_LANE_LOOP_VAR_ATTR,
            getattr(loop, HELION_LANE_LOOP_VAR_ATTR),
        )
        return new_loop


_ZERO_LITERAL = re.compile(r"^(cutlass\.\w+\()?(0|0\.0)\)?$")
_LANE_EXTRACT = re.compile(
    r"^(cutlass\.\w+\()?cutlass\.U?[Ii]nt\d+\(_vsink_vec\w*\[\w+\]\)"
    r"\.bitcast\(cutlass\.\w+\)\)?$"
)
# ``level1_eviction_priority='...'`` of a load site -> PTX L1 qualifier name.
_L1_PRIORITY = re.compile(r"level1_eviction_priority='(\w+)'")


def _ctor_operand_name(node: ast.expr) -> str | None:
    """``x`` for ``x`` or ``cutlass.T(x)``."""
    if (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func).startswith("cutlass.")
    ):
        node = node.args[0]
    return node.id if isinstance(node, ast.Name) else None


def _is_zero_literal(node: ast.expr) -> bool:
    return _ZERO_LITERAL.match(ast.unparse(node)) is not None


def _is_lane_extract(node: ast.expr) -> bool:
    """``Uint32(_vsink_vec[v]).bitcast(T)``, optionally in a dtype ctor."""
    return _LANE_EXTRACT.match(ast.unparse(node)) is not None


def _masked_packet_load_expr(
    pointer: str, guard: str, dtype: torch.dtype, vec_width: int, eviction_suffix: str
) -> str:
    """``_cute_load_v16b_if(ptr, guard, vec_type[, priority])`` for the packet.

    The L1 eviction priority of the site is kept (``'evict_first'`` /
    ``'evict_last'``, or the streaming ``cop='cs'``); the inline-PTX L2
    policy helpers have no predicated form, so their marker is dropped.
    """
    helper = MASKED_PACKET_LOAD_HELPERS[vec_width * dtype.itemsize]
    vec_type = f"ir.VectorType.get([{vec_width}], {_cute_unroll_vec_elem_type(dtype)}.mlir_type)"
    priority = ""
    if (match := _L1_PRIORITY.search(eviction_suffix)) is not None:
        priority = match.group(1)
    elif "cop='cs'" in eviction_suffix:
        priority = "cs"
    arguments = f"{pointer}, {guard}, {vec_type}"
    if priority:
        arguments += f", {priority!r}"
    return f"{helper}({arguments})"


def _statement_reads_outside_bodies(stmt: ast.AST) -> set[str]:
    """Names a statement reads itself: a simple statement's whole expression,
    a compound statement's header only (``if`` test, ``for`` iterable), not
    the statements nested in its bodies."""
    if isinstance(stmt, ast.If):
        return _names_read(stmt.test)
    if isinstance(stmt, ast.For):
        return _names_read(stmt.iter)
    if isinstance(stmt, ast.While):
        return _names_read(stmt.test)
    return _names_read(stmt)


def _written_tensors(body: Sequence[ast.AST], tensor_names: set[str]) -> set[str]:
    """Tensor arguments any store or atomic of ``body`` may write (every
    tensor when a write's destination cannot be resolved)."""
    from .fuse_two_pass_loads import _is_store_call
    from .fuse_two_pass_loads import _store_tensor_roots

    written: set[str] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and _is_store_call(node):
                roots = _store_tensor_roots(node, tensor_names)
                if roots is None:
                    return set(tensor_names)
                written |= roots
    return written


def _packet_load_parts(stmt: ast.AST) -> tuple[str, str, int, str] | None:
    """``(target, tensor, V, carrier)`` of a vector-load statement this pass
    emitted (``_VSINK_VECTOR_LOAD_ATTR``), reading its ``ir.VectorType``."""
    if not (
        getattr(stmt, _VSINK_VECTOR_LOAD_ATTR, False)
        and isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
    ):
        return None
    target = _plain_target(stmt)
    root = next(
        (
            node.value.id
            for node in ast.walk(stmt.value)
            if isinstance(node, ast.Attribute)
            and node.attr == "iterator"
            and isinstance(node.value, ast.Name)
        ),
        None,
    )
    vector_type = next(
        (
            node
            for node in ast.walk(stmt.value)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "ir.VectorType.get"
            and len(node.args) == 2
        ),
        None,
    )
    if (
        target is None
        or root is None
        or vector_type is None
        or not isinstance(vector_type.args[0], ast.List)
        or len(vector_type.args[0].elts) != 1
        or not isinstance(vector_type.args[0].elts[0], ast.Constant)
        or not isinstance(vector_type.args[1], ast.Attribute)
        or vector_type.args[1].attr != "mlir_type"
    ):
        return None
    width = vector_type.args[0].elts[0].value
    if type(width) is not int:
        return None
    return target, root, width, ast.unparse(vector_type.args[1].value)


def _reuse_sibling_packets(
    stmts: list[ast.stmt], cacheable: frozenset[str], new_var: Callable[..., str]
) -> None:
    """Serve a later sibling lane loop's packets from the earlier loop's.

    Two sweeps of one body over the same rows (softmax's max sweep and sum
    sweep) emit the same lane loop twice; after sinking, each loads the same
    packets (identical load statements: same pointer, same guard).  When no
    write of the kernel can reach the loaded tensor's storage (``cacheable``:
    neither written nor possibly aliasing a written tensor), the earlier loop
    keeps its packets in a register fragment and the later loop reads the
    fragment instead of re-issuing the loads.  Loops are matched by their
    lane variable and trip count, loads by position and text.
    """
    for container in _statement_lists(stmts):
        lanes: dict[tuple[str, str], tuple[ast.For, list[ast.Assign]]] = {}
        for stmt in list(container):
            if not isinstance(stmt, ast.For) or not isinstance(stmt.target, ast.Name):
                continue
            lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
            trips = _lane_loop_extent(stmt)
            if lane_var is None or trips is None:
                continue
            loads = [
                cast("ast.Assign", inner)
                for inner in stmt.body
                if _packet_load_parts(inner) is not None
            ]
            key = (lane_var, ast.unparse(stmt.iter))
            earlier = lanes.get(key)
            if earlier is None:
                if loads:
                    lanes[key] = (stmt, loads)
                continue
            first_loop, first_loads = earlier
            first_definitions = _pure_loop_definitions(first_loop)
            first_texts = {
                _canonical_text(load.value, first_definitions): (position, load)
                for position, load in enumerate(first_loads)
            }
            definitions = _pure_loop_definitions(stmt)
            renames: dict[str, str] = {}
            for load in loads:
                match = first_texts.get(_canonical_text(load.value, definitions))
                parts = _packet_load_parts(load)
                if match is None or parts is None:
                    continue
                position, first_load = match
                target, root, width, carrier = parts
                if root not in cacheable:
                    continue
                first_parts = _packet_load_parts(first_load)
                assert first_parts is not None
                if first_parts[2:] != (width, carrier):
                    continue
                # One fragment per loop pair holds every packet of every
                # trip: slot ``(lane * loads + position) * V + v``.
                cache = getattr(first_loop, _VSINK_PACKET_CACHE_ATTR, None)
                if cache is None:
                    cache = new_var("_vsink_cache", dce=False)
                    setattr(first_loop, _VSINK_PACKET_CACHE_ATTR, cache)
                    container.insert(
                        container.index(first_loop),
                        statement_from_string(
                            f"{cache} = cute.make_rmem_tensor("
                            f"{trips * len(first_loads) * width}, {carrier})"
                        ),
                    )
                slot = f"({lane_var} * {len(first_loads)} + {position}) * {width}"
                if not getattr(first_load, _VSINK_PACKET_CACHE_ATTR, False):
                    setattr(first_load, _VSINK_PACKET_CACHE_ATTR, True)
                    lane = new_var("_vsink_cache_lane", dce=False)
                    first_loop.body.insert(
                        first_loop.body.index(first_load) + 1,
                        statement_from_string(
                            f"for {lane} in cutlass.range_constexpr({width}):\n"
                            f"    {cache}[{slot} + {lane}] = "
                            f"{carrier}({first_parts[0]}[{lane}])"
                        ),
                    )
                renames[target] = f"{cache}[{slot} + {{lane}}]"
            if renames:
                stmt.body = [
                    inner for inner in stmt.body if _plain_target(inner) not in renames
                ]
                _RewritePacketReads(renames).visit(stmt)


def _pure_loop_definitions(loop: ast.For) -> dict[str, ast.expr]:
    """Single pure definitions of the loop body (per-row index arithmetic),
    by name: the unrolled copies of sibling loops rename them differently,
    so loads are compared with these inlined."""
    counts: dict[str, int] = {}
    values: dict[str, ast.expr] = {}
    for inner in loop.body:
        if isinstance(inner, ast.Assign) and (name := _plain_target(inner)) is not None:
            counts[name] = counts.get(name, 0) + 1
            if _is_relocatable_statement(inner, allow_load=False, tensor_names={}):
                values[name] = inner.value
    return {name: value for name, value in values.items() if counts[name] == 1}


def _canonical_text(expr: ast.expr, definitions: dict[str, ast.expr]) -> str:
    """``expr`` with the loop-local definitions inlined (bounded depth)."""

    class Inline(ast.NodeTransformer):
        def __init__(self) -> None:
            self.depth = 0

        def visit_Name(self, node: ast.Name) -> ast.AST:
            value = definitions.get(node.id)
            if value is None or self.depth > 24 or not isinstance(node.ctx, ast.Load):
                return node
            self.depth += 1
            try:
                return self.visit(_clone(value))
            finally:
                self.depth -= 1

    return ast.unparse(Inline().visit(_clone(expr)))


class _RewritePacketReads(ast.NodeTransformer):
    """``vec[v]`` of a replaced packet -> its cache slot ``cache[... + v]``."""

    def __init__(self, renames: dict[str, str]) -> None:
        super().__init__()
        self.renames = renames

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.renames
            and isinstance(node.ctx, ast.Load)
        ):
            template = self.renames[node.value.id]
            return expr_from_string(template.format(lane=ast.unparse(node.slice)))
        return node


def _statement_lists(stmts: list[ast.stmt]) -> list[list[ast.stmt]]:
    """``stmts`` and every nested statement list, outermost first."""
    result = [stmts]
    for stmt in stmts:
        for field in ("body", "orelse"):
            inner = getattr(stmt, field, None)
            if isinstance(inner, list):
                result.extend(_statement_lists(inner))
    return result


_VSINK_PACKET_CACHE_ATTR = "_helion_vsink_packet_cache"


def _remove_statements(stmts: list[ast.stmt], targets: set[int]) -> None:
    """Drop the statements whose ``id`` is in ``targets`` from ``stmts`` and
    every nested statement list."""
    stmts[:] = [stmt for stmt in stmts if id(stmt) not in targets]
    for stmt in stmts:
        for field in ("body", "orelse"):
            inner = getattr(stmt, field, None)
            if isinstance(inner, list):
                _remove_statements(inner, targets)


def _scalar_load_pointer(call: ast.Call) -> ast.expr | None:
    """Pointer expression of ``(ptr).load()`` / ``cute.arch.load(ptr, dtype)``."""
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "load"
        and not call.args
        and not call.keywords
    ):
        return call.func.value
    if (
        ast.unparse(call.func) == "cute.arch.load"
        and len(call.args) == 2
        and isinstance(call.args[1], (ast.Name, ast.Attribute))
    ):
        return call.args[0]
    return None


def _pointer_root(node: ast.expr) -> str | None:
    """Tensor name of a ``T.iterator + ...`` pointer sum."""
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        node = node.left
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    ):
        return node.value.id
    return None


def _maximal_pointer_sums(node: ast.AST) -> list[ast.BinOp]:
    """Outermost ``T.iterator + ...`` sums in ``node``."""
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and _pointer_root(node) is not None
    ):
        return [node]
    result: list[ast.BinOp] = []
    for child in ast.iter_child_nodes(node):
        result.extend(_maximal_pointer_sums(child))
    return result


def _additive_terms(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_additive_terms(node.left), *_additive_terms(node.right)]
    return [node]


def _split_affine(
    node: ast.expr, lane_var: str
) -> tuple[list[ast.expr], ast.expr | None] | None:
    """Split ``A + Int32(lane) * C`` (any term order) into ``([A terms], C)``.

    ``C`` is None for a bare ``cutlass.Int32(lane)`` term.  Returns None
    when the lane variable appears other than in exactly one such term.
    """
    invariant: list[ast.expr] = []
    scale: ast.expr | None = None
    found = False
    lane_call = f"cutlass.Int32({lane_var})"
    for term in _additive_terms(node):
        if lane_var not in _names_read(term):
            invariant.append(term)
            continue
        if found:
            return None
        found = True
        if ast.unparse(term) == lane_call:
            scale = None
        elif (
            isinstance(term, ast.BinOp)
            and isinstance(term.op, ast.Mult)
            and ast.unparse(term.left) == lane_call
            and lane_var not in _names_read(term.right)
        ):
            scale = term.right
        elif (
            isinstance(term, ast.BinOp)
            and isinstance(term.op, ast.Mult)
            and ast.unparse(term.right) == lane_call
            and lane_var not in _names_read(term.left)
        ):
            scale = term.left
        else:
            return None
    if not found:
        return None
    return invariant, scale


def _reduce_pointer(
    pointer: ast.BinOp,
    lane_var: str,
    definitions: dict[str, ast.expr],
    body_writes: set[str],
    new_var: Callable[..., str],
) -> tuple[ast.stmt, ast.stmt, ast.expr] | None:
    terms = _additive_terms(pointer)
    root = terms[0]
    lane_term: tuple[int, list[ast.expr], ast.expr | None, ast.expr] | None = None
    for index, term in enumerate(terms[1:], start=1):
        # ``cutlass.Int32(idx) * cutlass.Int32(T.layout.stride[d])``
        if not (
            isinstance(term, ast.BinOp)
            and isinstance(term.op, ast.Mult)
            and isinstance(term.left, ast.Call)
            and ast.unparse(term.left.func) == "cutlass.Int32"
            and len(term.left.args) == 1
        ):
            return None
        reads = _names_read(term)
        if lane_var in reads:
            return None
        if not reads & body_writes:
            continue
        index_node = term.left.args[0]
        if not isinstance(index_node, ast.Name) or index_node.id not in definitions:
            return None
        definition = definitions[index_node.id]
        split = _split_affine(definition, lane_var)
        if (
            split is None
            or (_names_read(definition) - {lane_var}) & body_writes
            or _names_read(term.right) & body_writes
            or lane_term is not None
        ):
            return None
        lane_term = (index, split[0], split[1], term.right)
    if lane_term is None:
        return None
    index, invariant_terms, scale, stride = lane_term
    base_index = " + ".join(ast.unparse(t) for t in invariant_terms) or "0"
    stride_text = ast.unparse(stride)
    step_text = (
        f"cutlass.Int32({ast.unparse(scale)}) * {stride_text}"
        if scale is not None
        else stride_text
    )
    other_terms = [
        ast.unparse(term) for i, term in enumerate(terms) if i not in (0, index)
    ]
    base_var = new_var("_vsink_ptr", dce=False)
    step_var = new_var("_vsink_step", dce=False)
    base_text = " + ".join(
        [
            ast.unparse(root),
            f"cutlass.Int32({base_index}) * {stride_text}",
            *other_terms,
        ]
    )
    base_stmt = statement_from_string(f"{base_var} = {base_text}")
    step_stmt = statement_from_string(f"{step_var} = {step_text}")
    replacement = expr_from_string(
        f"{base_var} + cutlass.Int32({lane_var}) * {step_var}"
    )
    return base_stmt, step_stmt, cast("ast.expr", replacement)


def _fragment_reduce_match(
    stmt: ast.stmt, vec_var: str, fragments: set[str]
) -> tuple[ast.Call, str, int, int, int] | None:
    """``R = wrap(_cute_grouped_reduce_shared_two_stage(frag[vec], ...))``."""
    if _plain_target(stmt) is None or not isinstance(stmt, ast.Assign):
        return None
    calls = [
        node
        for node in ast.walk(stmt.value)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _TWO_STAGE_REDUCE
    ]
    if len(calls) != 1:
        return None
    call = calls[0]
    if len(call.args) != 6 or len(call.keywords) != 3:
        return None
    source = call.args[0]
    if not (
        isinstance(source, ast.Subscript)
        and isinstance(source.value, ast.Name)
        and source.value.id in fragments
        and isinstance(source.slice, ast.Name)
        and source.slice.id == vec_var
    ):
        return None
    keywords = {kw.arg: kw.value for kw in call.keywords}
    values: dict[str, int] = {}
    for key in ("pre", "group_span", "group_count"):
        node = keywords.get(key)
        if not (isinstance(node, ast.Constant) and isinstance(node.value, int)):
            return None
        values[key] = node.value
    return (
        call,
        source.value.id,
        values["pre"],
        values["group_span"],
        values["group_count"],
    )


class _RenameLoads(ast.NodeTransformer):
    """Rename ``ast.Name`` loads (not stores) by a mapping."""

    def __init__(self, renames: dict[str, str]) -> None:
        self.renames = renames

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.renames:
            node.id = self.renames[node.id]
        return node


class _ExpandNames(ast.NodeTransformer):
    """Replace expanded scalars by their fragment element ``frag[vec]``."""

    def __init__(self, expanded: dict[str, str], vec_var: str) -> None:
        self.expanded = expanded
        self.vec_var = vec_var

    def visit_Name(self, node: ast.Name) -> ast.AST:
        fragment = self.expanded.get(node.id)
        if fragment is None:
            return node
        return ast.copy_location(
            ast.Subscript(
                value=ast.Name(id=fragment, ctx=ast.Load()),
                slice=ast.Name(id=self.vec_var, ctx=ast.Load()),
                ctx=node.ctx,
            ),
            node,
        )


class _SubstituteLane(ast.NodeTransformer):
    """``lane`` -> ``lane * unroll + copy`` in load contexts."""

    def __init__(self, lane_var: str, unroll: int, copy_index: int) -> None:
        self.lane_var = lane_var
        self.unroll = unroll
        self.copy_index = copy_index

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id == self.lane_var:
            return ast.copy_location(
                cast(
                    "ast.expr",
                    expr_from_string(
                        f"{self.lane_var} * {self.unroll} + {self.copy_index}"
                    ),
                ),
                node,
            )
        return node


class _ReplaceCall(ast.NodeTransformer):
    """Replace one call node (matched by source text) with an expression."""

    def __init__(self, call: ast.Call, replacement: str) -> None:
        self.text = ast.unparse(call)
        self.replacement = replacement

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if ast.unparse(node) == self.text:
            return cast("ast.AST", expr_from_string(self.replacement))
        return self.generic_visit(node)


class _ReplaceNode(ast.NodeTransformer):
    def __init__(self, target: ast.AST, replacement: ast.expr) -> None:
        self.target = target
        self.replacement = replacement

    def visit(self, node: ast.AST) -> ast.AST:
        if node is self.target:
            return self.replacement
        return super().visit(node)


def sink_grid_vector_loops(
    body: list[ast.AST],
    *,
    wrappers: Mapping[str, CuteVloopWrapperFact],
    loads: Mapping[str, CuteVloopLoadFact],
    rename_groups: dict[str, str],
    lane_unroll: int,
    new_var: Callable[..., str],
    tensor_dtypes: Mapping[str, torch.dtype],
    proven_disjoint_tensor_pairs: Collection[frozenset[str]] = (),
    stores: Mapping[str, CuteVloopStoreFact] | None = None,
) -> tuple[list[ast.AST], bool, bool]:
    """Sink every recorded grid V-loop that encloses a vectorizable serial
    reduction nest; bodies outside the proven forms are left untouched.

    ``tensor_dtypes`` maps every kernel tensor argument to its dtype: a
    subscript of one of these names is a memory read.
    ``proven_disjoint_tensor_pairs`` are the argument pairs whose storages
    cannot overlap; a vector load may move above a segment's stores into
    such a tensor.

    Returns the body, whether any V-loop was sunk and whether this pass
    unrolled a lane loop itself (``lane_unroll``; a sunk nest whose loads are
    persistent markers leaves its lane loops rolled for
    ``cute/unroll_lane_loads.py`` to take once the markers are packets)."""
    if not wrappers:
        return body, False, False
    fired = False
    unrolled = False
    # Packets of a tensor may be served to a sibling lane loop from the first
    # loop's registers only when no write of the kernel can reach that
    # tensor's storage: the tensor is not written itself and is proven
    # disjoint from every written tensor (two arguments may share storage:
    # ``x.view(-1)`` next to ``x``, or one tensor passed as input and out).
    written = _written_tensors(body, set(tensor_dtypes))
    cacheable = frozenset(
        name
        for name in tensor_dtypes
        if name not in written
        and all(
            frozenset((name, other)) in proven_disjoint_tensor_pairs
            for other in written
        )
    )

    def visit(stmts: list[ast.AST]) -> list[ast.AST]:
        nonlocal fired, unrolled
        result: list[ast.AST] = []
        for stmt in stmts:
            constexpr = _is_constexpr_loop(stmt)
            fact = wrappers.get(constexpr[0]) if constexpr is not None else None
            if (
                fact is not None
                and constexpr is not None
                and constexpr[1] == fact.vec_width
                and isinstance(stmt, ast.For)
            ):
                sinker = _Sinker(
                    stmt,
                    fact,
                    loads,
                    rename_groups,
                    new_var,
                    lane_unroll,
                    tensor_dtypes,
                    proven_disjoint_tensor_pairs,
                    stores,
                )
                try:
                    replacement = sinker.run()
                except _Unsupported:
                    replacement = None
                if replacement is not None:
                    _reuse_sibling_packets(replacement, cacheable, new_var)
                    result.extend(replacement)
                    fired = True
                    unrolled = unrolled or sinker.lane_loops_unrolled > 0
                    continue
            for field in ("body", "orelse"):
                inner = getattr(stmt, field, None)
                if isinstance(inner, list) and inner:
                    setattr(stmt, field, visit(inner))
            result.append(stmt)
        return result

    return visit(body), fired, unrolled
