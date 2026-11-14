"""Interleave a grid lane loop's lanes into the loops of its body.

A grid tile axis with fewer threads than elements gives each thread several
of its elements as a lane loop around the whole tile body
(``PerThreadNDTileStrategy``; ``DeviceGridState.wrap_body`` nests it around
the body).  When the body walks another axis with device loops, every lane
runs the whole walk on its own.  T5's relative-position softmax tiles the
eight heads of one query row per CTA with no thread on the head axis, so each
thread runs the heads one after the other::

    for lane_1 in range(8):                               # the eight heads
        indices_1 = tile_offset_1 + cutlass.Int32(lane_1)
        row_max = -inf
        row_sum = 0.0
        for tile_offset_3 in range(0, 1024, 1024):        # online max and sum
            packet = cute.arch.load(scores[b, indices_1, q, k:k + 8])
            for vec_lane_3 in cutlass.range_constexpr(8):
                ... (bias[b, indices_1, q, k + vec_lane_3]).load() ...
            amax = _cute_grouped_reduce_shared_two_stage(acc, 'max', ...)
            ...
        for tile_offset_4 in range(0, 1024, 1024):        # normalize, store
            ...

The bias is laid out heads-innermost: the eight heads of one key are adjacent
in memory, but each lane gathers its own head at a 16-byte stride, two bytes
per load, and the kernel is bound by L1 wavefronts.  ``jam_grid_lanes``
unrolls the lane loop and jams the copies into the loops, statement by
statement, so the lanes' copies of a statement sit side by side::

    for tile_offset_3 in range(0, 1024, 1024):
        packet_l0 = ...
        ...
        packet_l7 = ...
        for vec_lane_3 in cutlass.range_constexpr(8):
            vec = cute.arch.load(bias[b, tile_offset_1, q, k + vec_lane_3], 8)
            load_1_l0 = vec[0]
            ...
            load_1_l7 = vec[7]  # one 16-byte load
            ...
        _cute_grouped_reduce_shared_two_stage_fragment(acc_frag, amax_frag, ...)
        ...

Then the lanes' copies of a scalar load whose lane index addresses its
tensor's stride-1 dimension become one vector load (``_vectorize_loads``),
and the lanes' copies of a shared-memory grouped reduce become one fragment
reduce (``_batched_reduce``), which combines every value in the order the
scalar reduce it replaces does.  Each lane computes the same values in the
same order as in the rolled loop; only the order of work belonging to
different lanes changes, so the results are bitwise the rolled loop's.
When the tile spans the row, each device loop runs once and a two-pass
kernel's second loop reloads what its first loaded: such loops become
straight-line code and a load of an address an earlier load read takes that
load's value (``_LoadReuse``).

The one exception to bitwise results are the adds into a privatized atomic
target's shared-memory copy (``cute_privatize_atomics``), whose order the
knob leaves unspecified.  T5's bias-gradient scatter adds each key's
batch-summed gradient into the cell of its relative-position bucket, and
neighbouring keys mostly share a bucket; the copy's fp32 adds are
compare-and-swap loops that serialize on a contended cell.  The lanes'
copies of such an add become one add per cell the lanes address
(``_combined_atomic``): a thread sums its lanes' values for a cell in lane
order and adds the sum once.

That reordering is exact when the lanes share nothing the jam reorders, which
the pass proves on the lane loop's body and fails closed otherwise (the body
is left as it is):

- Registers: no name is carried from one lane to the next (read in the body
  before the body writes it on every path, while the body also writes it),
  and no name the body writes is read after the loop.  No name the body
  updates in place is bound to another name's object or element (``a = y``,
  ``row = x[i]``), whose update the proof would not see, and no name is
  bound to a tensor (``_check_aliases``).  Each lane's names
  become names of their own; the statements that read and write only
  lane-invariant names run once, at their place, unless they call something
  other than a proven pure computation or load (a reduce, an append), which
  every lane runs on names of its own.  A two-pass register cache
  (``fuse_two_pass_loads``) whose slots the body writes before it reads
  them is each lane's own: each lane gets a copy of it, within a budget for
  the copies together, and the copies replace the cache.
- Memory: the tensors the body stores to are proven disjoint from one
  another (two views of one buffer may share elements, which the jam would
  store in another order) and from every other tensor the body loads, and
  every access of a stored tensor, a load too, addresses one common
  dimension of it (a non-overlapping tensor) by the lane's own index: each
  lane stores to its own elements and reads back only those (a workspace
  row the first sweep stores and the second reads).  Loads, stores, the
  shared-memory grouped reduces (each traced call takes a buffer of its own
  and synchronizes the CTA around it), barriers and the adds into a
  privatized atomic target's shared-memory copy are the only effects; any
  other call that is not a proven pure computation rejects the body.  Such
  an add is admitted when the body names the copy nowhere else, so no lane
  reads what another adds, and it is the copy's one add in the body, outside
  every loop of it and every branch that runs per lane, the shape
  ``_combined_atomic`` handles (``_private_atomics``).
- Structure: the lane loop has a literal extent of at most ``_MAX_LANES``,
  its copies trace at most ``_MAX_UNROLLED_STATEMENTS`` statements (the
  loads-first lane unroll's budget), its lane index is the tile offset plus
  the lane, bound once and unmasked, the body never rebinds the lane
  variable, and a loop or branch of the body whose header or test is the
  same in every lane is jammed; one that differs per lane is copied whole
  per lane.

The copies multiply the per-lane register budgets that earlier passes sized
for one lane: each lane keeps its own stash list (``stash_lane_loads`` caps
one lane's at ``_MAX_STASHED_REGISTERS``) and its own unroll packets, so a
kernel whose stash already sits at that cap may spill under the knob, which
autotuning weighs.  The jam's own limit is the traced-statement budget.
"""

from __future__ import annotations

import ast
import dataclasses
import logging
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import ReadWrites
from ..ast_read_writes import ast_rename
from ._ast_pass_utils import EXACT_DIVISION_HELPERS
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import deep_clone
from ._ast_pass_utils import exposed_reads
from ._ast_pass_utils import flatten_addition
from ._ast_pass_utils import is_cta_reduce_call
from ._ast_pass_utils import is_name_assign
from ._ast_pass_utils import loop_extent
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_written
from ._ast_pass_utils import replace_node
from ._ast_pass_utils import substitute_names
from .emitters.lane_packets import _CUTE_VECTOR_MAX_BYTES
from .emitters.lane_packets import _CUTE_VECTOR_UNROLL_CARRIER
from .emitters.lane_packets import _CUTE_VECTOR_UNROLL_DTYPES
from .emitters.lane_packets import _cute_unroll_vec_load_expr
from .fuse_two_pass_loads import AUTO_REGISTER_CACHE_ELEMENTS
from .fuse_two_pass_loads import FUSE_CACHE_PREFIX
from .fuse_two_pass_loads import _range_bounds
from .fuse_two_pass_loads import _static_int
from .fuse_two_pass_loads import _trip_count_for
from .memory_ops import cute_reduction_vector_layout_aligned
from .packet_access import PacketAccess
from .packet_access import match_packet_access
from .reduce_call import GROUPED_REDUCE_HELPERS
from .reduce_call import GROUPED_SHARED_REDUCES
from .reduce_call import ReduceScope
from .stash_lane_loads import load_registers
from .unroll_lane_loads import _MAX_UNROLLED_STATEMENTS

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    import torch

    from ..compile_environment import CompileEnvironment

log = logging.getLogger(__name__)

# Lanes a jammed loop may have: every statement of the body that differs per
# lane is copied once per lane.
_MAX_LANES = 16
# 32-bit registers ``_LoadReuse`` may keep live to serve later loads: the
# eight scores packets and eight bias packets of T5's eight head lanes, 64.
# The budget is on top of ``stash_lane_loads``' (``_MAX_STASHED_REGISTERS``,
# also 64), whose stashes T5 keeps live beside these values.
_MAX_REUSED_REGISTERS = 64
# Elements the lanes' copies of the private register caches may hold
# together: one thread's cache under ``"auto"`` reloads, which T5's eight
# head lanes fill with eight each.
MAX_PRIVATE_CACHE_ELEMENTS = AUTO_REGISTER_CACHE_ELEMENTS
# The lane copies of one statement of the body: ``(template id, lane)``.
_JAM_GROUP_ATTR = "_helion_jam_group"
# A region whose names are its own (``_Jam.region_names``): its key.
_JAM_REGION_ATTR = "_helion_jam_region"
_LOAD_KINDS = frozenset(
    {"load", "arch_load", "packet", "hinted_packet", "masked_packet"}
)
_STORE_KINDS = frozenset({"store", "flush"})
_LOOP_ITERATORS = frozenset({"range", "cutlass.range", "cutlass.range_constexpr"})
_BARRIER = "cute.arch.sync_threads"
_TWO_STAGE = ReduceScope.SHARED_TWO_STAGE.value
_TWO_STAGE_FRAGMENT = ReduceScope.SHARED_TWO_STAGE_FRAGMENT.value


class _Decline(Exception):
    """The lane loop is outside the forms this pass proves; keep it."""


def jam_grid_lanes(
    body: list[ast.stmt],
    *,
    rename_groups: Mapping[str, str],
    tensors: Mapping[str, torch.Tensor],
    proven_disjoint_tensor_pairs: Collection[frozenset[str]],
    constexpr_values: Mapping[str, int],
    env: CompileEnvironment,
    new_var: Callable[[str], str],
) -> list[ast.stmt]:
    """Jam every top-level grid lane loop of ``body`` that wraps a device loop.

    ``rename_groups`` is ``ast_rename``'s map of each name to the name it is
    merged into (loop-carried values are spelled through aliases until the
    kernel-def stage renames them); ``tensors`` maps each tensor argument to
    its fake tensor.  A lane loop the pass cannot prove is left unchanged.
    """
    result: list[ast.stmt] = []
    for index, stmt in enumerate(body):
        if isinstance(stmt, ast.For) and _is_grid_lane_loop(stmt):
            jam = _Jam(
                stmt,
                preceding=result,
                following=body[index + 1 :],
                rename_groups=rename_groups,
                tensors=tensors,
                proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                constexpr_values=constexpr_values,
                env=env,
                new_var=new_var,
            )
            try:
                jammed = jam.run()
            except _Decline as reason:
                log.debug(
                    "grid lane loop %s kept: %s",
                    cast("ast.Name", stmt.target).id,
                    reason,
                )
                result.append(stmt)
                continue
            # The lanes' copies of a private cache replace its allocation.
            dead = {id(allocation) for allocation in jam.dead_allocations}
            result[:] = [kept for kept in result if id(kept) not in dead]
            result.extend(jammed)
            continue
        result.append(stmt)
    return result


def _is_grid_lane_loop(loop: ast.For) -> bool:
    """A lane loop of literal extent that runs a device loop of the tile program."""
    if getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None) is None:
        return False
    extent = loop_extent(loop, kinds=("range", "constexpr"), minimum=2)
    if extent is None or extent > _MAX_LANES:
        return False
    return any(
        isinstance(node, ast.For)
        and node is not loop
        and getattr(node, HELION_LANE_LOOP_VAR_ATTR, None) is None
        and loop_extent(node, kinds=("constexpr",), minimum=None) is None
        for node in ast.walk(loop)
    )


@dataclasses.dataclass
class _Pointer:
    """``T.iterator + Int32(i0) * Int32(T.layout.stride[0]) + ...``."""

    tensor: str
    indices: dict[int, ast.expr]

    def lane_dims(self, lane_index: str) -> set[int]:
        """The dimensions the pointer indexes by the name ``lane_index``."""
        return {
            dim
            for dim, index in self.indices.items()
            if isinstance(index, ast.Name) and index.id == lane_index
        }


def _parse_pointer(pointer: ast.expr) -> _Pointer | None:
    terms = flatten_addition(pointer)
    head = terms[0]
    if not (
        isinstance(head, ast.Attribute)
        and head.attr == "iterator"
        and isinstance(head.value, ast.Name)
    ):
        return None
    tensor = head.value.id
    indices: dict[int, ast.expr] = {}
    for term in terms[1:]:
        if not (isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult)):
            return None
        index = _int32_operand(term.left)
        stride = _int32_operand(term.right)
        if index is None or stride is None:
            return None
        if not (
            isinstance(stride, ast.Subscript)
            and ast.unparse(stride.value) == f"{tensor}.layout.stride"
            and isinstance(stride.slice, ast.Constant)
            and type(stride.slice.value) is int
            and stride.slice.value not in indices
        ):
            return None
        indices[stride.slice.value] = index
    return _Pointer(tensor, indices)


def _traced_statements(stmts: Sequence[ast.stmt]) -> int:
    """The statements one trace of ``stmts`` emits: a constexpr loop's body
    once per trip, any other loop's body and both sides of a branch once."""
    size = 0
    for stmt in stmts:
        size += 1
        if isinstance(stmt, ast.For):
            trips = loop_extent(stmt, kinds=("constexpr",), minimum=None) or 1
            size += trips * _traced_statements(stmt.body)
        elif isinstance(stmt, ast.If):
            size += _traced_statements(stmt.body) + _traced_statements(stmt.orelse)
    return size


def _strip_int32(node: ast.expr) -> ast.expr:
    """``x`` of ``cutlass.Int32(cutlass.Int32(x))``."""
    while (operand := _int32_operand(node)) is not None:
        node = operand
    return node


class _ElideResidueBounds(ast.NodeTransformer):
    """``value if residue < c else other`` -> ``value`` when ``residue`` is
    ``index % m`` (``_cute_python_mod``, never negative) and ``c >= m``, in
    statements the residue's binding reaches (``_Jam._fold_mod_lane_bounds``)."""

    def __init__(
        self,
        residues: Mapping[str, int],
        constant: Callable[[ast.expr], int | None],
    ) -> None:
        self.residues = residues
        self.constant = constant

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Lt)
            and isinstance(left := _strip_int32(test.left), ast.Name)
            and left.id in self.residues
            and (bound := self.constant(test.comparators[0])) is not None
            and bound >= self.residues[left.id]
        ):
            return node.body
        return node


def _bindings(body: Sequence[ast.stmt], name: str) -> int:
    """How many times ``body`` binds ``name``."""
    return sum(
        1
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Store)
    )


def _update_reads(stmt: ast.stmt) -> set[str]:
    """The names ``stmt`` reads, where a store ``x[i] = v`` reads ``i`` and
    ``v`` but not ``x``: ``x[i] += v`` and ``x[i] = x[i] + v`` read ``x``."""
    receivers = {
        id(node.value)
        for target in (stmt.targets if isinstance(stmt, ast.Assign) else [])
        for node in (target.elts if isinstance(target, ast.Tuple) else [target])
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
    }
    return {
        node.id
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and id(node) not in receivers
    }


def _definitely_written(stmt: ast.AST) -> set[str]:
    """The names ``stmt`` writes on every path through it, for
    ``exposed_reads``: a branch writes what both of its arms write, a loop
    nothing (it may run no trip), any other statement what it writes."""
    if isinstance(stmt, ast.If):
        body, orelse = (
            {name for inner in arm for name in _definitely_written(inner)}
            for arm in (stmt.body, stmt.orelse)
        )
        return body & orelse
    if isinstance(stmt, (ast.For, ast.While)):
        return set()
    return set(ReadWrites.from_ast(stmt).writes)


def _mutated_receivers(stmt: ast.stmt) -> set[str]:
    """Names a statement updates in place: ``x[i] = v``, ``x.append(v)``."""
    result: set[str] = set()
    targets = (
        stmt.targets
        if isinstance(stmt, ast.Assign)
        else [stmt.target]
        if isinstance(stmt, ast.AugAssign)
        else []
    )
    for target in targets:
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            result.add(target.value.id)
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "append"
        and isinstance(stmt.value.func.value, ast.Name)
    ):
        result.add(stmt.value.func.value.id)
    return result


def _cache_accesses(stmt: ast.stmt, cache: str) -> tuple[bool, bool]:
    """Whether ``stmt``, or a statement in it, writes the register cache
    ``cache`` and whether one reads it: a store ``cache[i] = v`` writes it
    and reads it only where ``i`` or ``v`` names it, and a loop or branch
    header reads what it names."""
    if isinstance(stmt, (ast.For, ast.While, ast.If)):
        header = stmt.iter if isinstance(stmt, ast.For) else stmt.test
        writes, reads = False, cache in names_read(header)
        for inner in (*stmt.body, *stmt.orelse):
            inner_writes, inner_reads = _cache_accesses(inner, cache)
            writes |= inner_writes
            reads |= inner_reads
        return writes, reads
    return cache in _mutated_receivers(stmt), cache in _update_reads(stmt)


def _written_before_read(
    stmts: Sequence[ast.stmt], cache: str, *, whole: bool = True
) -> bool:
    """Whether running ``stmts`` writes the register cache ``cache`` before
    it reads it, as the statements' order shows: the first statement that
    accesses the cache writes it without reading it, a branch does so in
    both arms, and a loop in each trip.  A branch that holds every access
    in one arm is checked within that arm, where ``stmts`` hold every
    access of the cache (``whole``): a lane that skips it does not access
    the cache at all.  A loop's trips are checked as a branch's arms are
    (a trip whose branch skips the write leaves its slots to the next
    lane).  A loop that writes the cache counts although it may
    run no trip: the later sweeps that read it run the same trips
    (``fuse_two_pass_loads`` matches their headers), and which slots they
    read is that pass's invariant too."""
    touching = [
        (stmt, *accesses)
        for stmt in stmts
        if any(accesses := _cache_accesses(stmt, cache))
    ]
    if not touching:
        return False
    first, writes, reads = touching[0]
    if (
        isinstance(first, ast.If)
        and whole
        and len(touching) == 1
        and cache not in names_read(first.test)
    ):
        arms = [
            arm
            for arm in (first.body, first.orelse)
            if any(any(_cache_accesses(inner, cache)) for inner in arm)
        ]
        if len(arms) == 1:
            return _written_before_read(arms[0], cache)
    if reads or not writes:
        return False
    if isinstance(first, ast.If):
        return all(
            _written_before_read(arm, cache, whole=False)
            for arm in (first.body, first.orelse)
        )
    if isinstance(first, (ast.For, ast.While)):
        return _written_before_read(first.body, cache, whole=False)
    return True


def _may_update_arguments(call: ast.Call) -> bool:
    """Whether ``call`` may update a tensor or list it is handed in place (a
    batched reduce fills its results): any call but a proven pure
    computation, a load or store, a loop header or an ``append``."""
    access = match_packet_access(call)
    if access is not None and access.kind in _LOAD_KINDS | _STORE_KINDS:
        return False
    if call_path_str(call.func) in _LOOP_ITERATORS or (
        isinstance(call.func, ast.Attribute) and call.func.attr == "append"
    ):
        return False
    return not _is_proven_relocatable_call(call, allow_load=True)


def _load_pointer(access: PacketAccess | None) -> ast.expr | None:
    """The pointer a load call reads, from its parsed ``access``."""
    if access is not None and access.kind in _LOAD_KINDS:
        return access.pointer
    return None


def _private_atomic_site(call: ast.Call) -> tuple[str, str] | None:
    """The tensor a relaxed CTA-scope scalar ``cute.arch.atomic_add`` adds
    into, through ``(copy.iterator + cute.crd2idx(coord, copy.layout))
    .llvm_ptr`` (the privatized atomics' spelling), and the keyword of its
    addend, the one besides ``sem`` and ``scope`` (``val``, or the name the
    installed CUTLASS gives it: ``_resolve_cute_atomic_kwargs``), or None."""
    addends = [
        keyword.arg for keyword in call.keywords if keyword.arg not in ("sem", "scope")
    ]
    if not (
        call_path_str(call.func) == "cute.arch.atomic_add"
        and len(call.args) == 1
        and len(addends) == 1
        and addends[0] is not None
        and {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
            if keyword.arg in ("sem", "scope")
        }
        == {"sem": "'relaxed'", "scope": "'cta'"}
    ):
        return None
    pointer = call.args[0]
    if not (
        isinstance(pointer, ast.Attribute)
        and pointer.attr == "llvm_ptr"
        and isinstance(pointer.value, ast.BinOp)
        and isinstance(pointer.value.op, ast.Add)
        and isinstance(iterator := pointer.value.left, ast.Attribute)
        and iterator.attr == "iterator"
        and isinstance(iterator.value, ast.Name)
        and isinstance(index := pointer.value.right, ast.Call)
        and call_path_str(index.func) == "cute.crd2idx"
        and len(index.args) == 2
        and ast.unparse(index.args[1]) == f"{iterator.value.id}.layout"
    ):
        return None
    return iterator.value.id, addends[0]


def _compared_names(compare: ast.Compare) -> set[str]:
    """The names a comparison reads outside the memory accesses in it: a
    loaded boolean (``keep[...].load() != 0``) addresses its element by the
    lane index without comparing the index."""
    names: set[str] = set()
    stack: list[ast.AST] = [compare]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call) and match_packet_access(node) is not None:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return names


def _tensor_roots(pointer: ast.expr) -> set[str]:
    """The tensors whose ``.iterator`` a pointer expression names."""
    return {
        node.value.id
        for node in ast.walk(pointer)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }


def _updated_arguments(node: ast.AST) -> set[str]:
    """The names ``node``, when a call ``_may_update_arguments`` accepts,
    hands to it: those it may fill in place."""
    if not (isinstance(node, ast.Call) and _may_update_arguments(node)):
        return set()
    return {
        argument.id
        for argument in (*node.args, *(kw.value for kw in node.keywords))
        if isinstance(argument, ast.Name)
    }


def _alias_sources(value: ast.expr) -> set[str]:
    """Names whose object, or an element of it, ``value`` may evaluate to:
    a name, either arm of a conditional or boolean operator, a subscript or
    ``operator.getitem`` of one, and a display packing one.  Any other
    expression (a call, arithmetic, a constant) makes a new object."""
    if isinstance(value, ast.Name):
        return {value.id}
    if isinstance(value, ast.IfExp):
        return _alias_sources(value.body) | _alias_sources(value.orelse)
    if isinstance(value, ast.BoolOp):
        return set().union(*map(_alias_sources, value.values))
    if isinstance(value, (ast.Subscript, ast.Starred, ast.NamedExpr)):
        return _alias_sources(value.value)
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return set().union(*map(_alias_sources, value.elts))
    if (
        isinstance(value, ast.Call)
        and call_path_str(value.func) == "operator.getitem"
        and value.args
    ):
        return _alias_sources(value.args[0])
    return set()


def _whole_objects(value: ast.expr) -> set[str]:
    """Names whose whole object ``value`` evaluates to or packs: a name,
    either arm of a conditional or boolean operator, and the elements of a
    display (``_alias_sources`` without the subscripts, which read an
    element)."""
    if isinstance(value, ast.Name):
        return {value.id}
    if isinstance(value, ast.IfExp):
        return _whole_objects(value.body) | _whole_objects(value.orelse)
    if isinstance(value, ast.BoolOp):
        return set().union(*map(_whole_objects, value.values))
    if isinstance(value, (ast.Starred, ast.NamedExpr)):
        return _whole_objects(value.value)
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return set().union(*map(_whole_objects, value.elts))
    return set()


def _int32_operand(node: ast.expr) -> ast.expr | None:
    """``x`` of ``cutlass.Int32(x)``."""
    if (
        isinstance(node, ast.Call)
        and call_path_str(node.func) == "cutlass.Int32"
        and len(node.args) == 1
        and not node.keywords
    ):
        return node.args[0]
    return None


def _non_overlapping(tensor: torch.Tensor) -> bool:
    """No two in-bounds index tuples of ``tensor`` address one element."""
    dims = [
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        if not (isinstance(size, int) and size == 1)
    ]
    if not all(isinstance(s, int) and isinstance(n, int) for s, n in dims):
        return False
    span = 1
    for stride, size in sorted(cast("list[tuple[int, int]]", dims)):
        if stride < span:
            return False
        span = stride * size
    return True


class _Jam:
    def __init__(
        self,
        loop: ast.For,
        *,
        preceding: Sequence[ast.stmt],
        following: Sequence[ast.stmt],
        rename_groups: Mapping[str, str],
        tensors: Mapping[str, torch.Tensor],
        proven_disjoint_tensor_pairs: Collection[frozenset[str]],
        constexpr_values: Mapping[str, int],
        env: CompileEnvironment,
        new_var: Callable[[str], str],
    ) -> None:
        self.loop = loop
        self.preceding = preceding
        self.following = following
        self.rename_groups = dict(rename_groups)
        self.tensors = tensors
        self.proven_disjoint_tensor_pairs = frozenset(proven_disjoint_tensor_pairs)
        self.constexpr_values = constexpr_values
        self.env = env
        self.new_var = new_var
        self.lane_var = cast("ast.Name", loop.target).id
        self.extent = cast("int", loop_extent(loop, kinds=("range", "constexpr")))
        self.varying: set[str] = set()
        # The regions whose names are their own: ``id(region)`` -> its names.
        self.region_names: dict[int, set[str]] = {}
        # Statements to run once before the jammed body (fragment buffers).
        self.allocations: list[ast.stmt] = []
        self.lane_index = ""
        self.lane_base: ast.expr | None = None
        # Per lane: each varying name -> the lane's own name.
        self.lane_names: list[dict[str, str]] = []
        # Names whose per-lane values are consecutive in lane order: the lane
        # index and its residues ``_cute_python_mod(index, m)``, m a multiple
        # of the lane count.
        self.consecutive: set[str] = set()
        # Register caches each lane gets a copy of: name -> its allocation.
        self.private_caches: dict[str, ast.Assign] = {}
        # Their allocations no other statement before the loop names, which
        # the lanes' copies replace.
        self.dead_allocations: list[ast.Assign] = []
        # The tensors the body stores to (``_check_memory``).
        self.stored: set[str] = set()
        # The body's adds into privatized atomic targets the jam reorders
        # (``_private_atomics``), by ``id``: the copy each adds into and the
        # keyword of its addend (``_private_atomic_site``).
        self.private_atomics: dict[int, tuple[str, str]] = {}

    # ------------------------------------------------------------- analysis

    def run(self) -> list[ast.stmt]:
        body = [deep_clone(stmt) for stmt in self.loop.body]
        for stmt in body:
            # Loop-carried values are written through aliases until the
            # kernel-def stage renames them; analyze and emit the merged names.
            ast_rename(stmt, self.rename_groups)
        self._check_forms(body)
        self._check_aliases(body)
        if _traced_statements(body) * self.extent > _MAX_UNROLLED_STATEMENTS:
            raise _Decline("the lanes' copies would trace too many statements")
        self._find_lane_index(body)
        self._fold_mod_lane_bounds(body)
        self._find_private_caches(body)
        self.varying = self._varying_names(body)
        self._check_registers(body)
        self._check_memory(body)
        copies = [self._lane_copy(body, lane) for lane in range(self.extent)]
        jammed = self._merge(body, copies)
        self._vectorize_loads(jammed)
        jammed = self._rewrite_groups(jammed, self._batched_reduce)
        if self.private_atomics:
            jammed = self._rewrite_groups(jammed, self._combined_atomic)
        jammed = _LoadReuse(
            self.constexpr_values, self.new_var, self.tensors, frozenset(self.stored)
        ).run(jammed)
        return [*self.allocations, *jammed]

    def _check_forms(self, stmts: Sequence[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                if stmt.orelse or not isinstance(stmt.target, ast.Name):
                    raise _Decline("a loop with an else clause or a structured target")
                self._check_forms(stmt.body)
            elif isinstance(stmt, ast.If):
                self._check_forms(stmt.body)
                self._check_forms(stmt.orelse)
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1:
                    raise _Decline("an assignment to several targets")
                self._check_target(stmt.targets[0])
            elif isinstance(stmt, ast.AugAssign):
                self._check_target(stmt.target)
            elif not isinstance(stmt, (ast.Expr, ast.Pass)):
                raise _Decline("a statement form the jam does not copy")
            if any(
                isinstance(node, (ast.NamedExpr, ast.Lambda, ast.Yield, ast.Await))
                for node in ast.walk(stmt)
            ):
                raise _Decline("an expression the jam does not copy")

    @staticmethod
    def _check_target(target: ast.expr) -> None:
        """A name, a tuple of names or ``x[...]`` of a name ``x``: the forms
        whose updates in place ``_mutated_receivers`` records
        (``x[i][j] = v`` or ``x.attr += v`` update ``x`` too)."""
        if isinstance(target, ast.Tuple):
            if all(isinstance(element, ast.Name) for element in target.elts):
                return
        elif isinstance(target, ast.Name) or (
            isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
        ):
            return
        raise _Decline("an assignment target the jam does not follow")

    def _check_aliases(self, body: Sequence[ast.stmt]) -> None:
        """Decline names that may share an object, failing closed.

        The register proof marks a name the body updates in place (``a[0] =
        v``, ``a.append(v)``, ``a += v``) varying, and nothing else: a
        lane-invariant name holding the same object would run once, and
        every lane's update would land in its one object (a later ``w =
        y[0]`` would read the last lane's).  So each binding of such a name
        in the body must make a new object (no ``_alias_sources``: a list
        display, ``cute.make_rmem_tensor(...)``, arithmetic, a call), never
        a tuple unpacking or a loop target.  Nor may a container the body
        updates (one it stores into or appends to, or augments with a
        display) be bound or stored whole into another name (``b = a``,
        ``b, c = a, 0``, ``b = a if p else c``, ``b.append(a)``); an
        augmented scalar may (``total = acc``).  No binding or stored value
        may take a tensor at all (``t = x``, ``t, u = x, y``, ``t = [x][0]``):
        the memory proof follows a tensor by its own name only."""
        statements = self._all_statements(body)
        updated: set[str] = set()
        augmented: set[str] = set()
        for stmt in statements:
            updated |= _mutated_receivers(stmt)
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                augmented.add(stmt.target.id)
        # The receivers of stores and appends, and names augmented with a
        # display, are containers; other augmented names are scalars.
        containers = updated | {
            stmt.target.id
            for stmt in statements
            if isinstance(stmt, ast.AugAssign)
            and isinstance(stmt.target, ast.Name)
            and isinstance(stmt.value, (ast.List, ast.ListComp))
        }
        updated |= augmented
        for stmt in statements:
            stored: list[ast.expr] = []
            if isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name) and stmt.target.id in updated:
                    raise _Decline("an update in place through an alias")
            elif isinstance(stmt, ast.Assign):
                target = stmt.targets[0]
                if isinstance(target, ast.Tuple):
                    if any(
                        isinstance(element, ast.Name) and element.id in updated
                        for element in target.elts
                    ):
                        raise _Decline("an update in place through an alias")
                elif (
                    isinstance(target, ast.Name)
                    and target.id in updated
                    and _alias_sources(stmt.value)
                ):
                    raise _Decline("an update in place through an alias")
                stored.append(stmt.value)
            elif isinstance(stmt, ast.AugAssign):
                stored.append(stmt.value)
            elif (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "append"
            ):
                stored.extend(stmt.value.args)
            for value in stored:
                if _whole_objects(value) & containers:
                    raise _Decline("an update in place through an alias")
                if _alias_sources(value) & self.tensors.keys():
                    raise _Decline("a tensor bound to another name")

    def _find_lane_index(self, body: Sequence[ast.stmt]) -> None:
        """The lane's tile index, ``index = base + cutlass.Int32(lane)``, with a
        lane-invariant base and no mask.  No other statement, nested ones
        included, reads the lane variable: the store proof takes ``base +
        lane`` for the only part of an address that differs per lane."""
        for stmt in body:
            if self.lane_var not in names_read(stmt):
                continue
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(value := stmt.value, ast.BinOp)
                and isinstance(value.op, ast.Add)
                and ast.unparse(value.right) == f"cutlass.Int32({self.lane_var})"
                and self.lane_var not in names_read(value.left)
                and not self.lane_index
            ):
                self.lane_index = stmt.targets[0].id
                self.lane_base = value.left
                continue
            raise _Decline("another statement reads the lane variable")
        if not self.lane_index:
            raise _Decline("no lane index of the form base + lane")
        if _bindings(body, self.lane_index) != 1:
            raise _Decline("the lane index is rebound")
        if _bindings(body, self.lane_var):
            raise _Decline("the body rebinds the lane variable")
        lane_reads = {
            name
            for stmt in body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Compare)
            for name in _compared_names(node)
        }
        if self.lane_index in lane_reads:
            # A mask over the lane's index: a masked-off lane's accesses may
            # point at a shared fallback element.
            raise _Decline("the lane index is masked")

    def _fold_mod_lane_bounds(self, body: list[ast.stmt]) -> None:
        """Record the lane index and its residues ``index % m`` the body
        defines, ``m`` a multiple of the lane count, and fold their bounds
        tests in the body (a clone the jam discards on a decline): the
        residues of a packet's lanes, whose base ``_vector_load`` requires to
        be a multiple of the packet width, are consecutive (a head
        ``bh % heads`` of consecutive ``bh``).  A residue's bounds test
        ``residue < c`` with ``c >= m`` always holds where the residue's
        binding reaches, the statements after it, and the select it guards
        there becomes its value, so the guarded loads can join a packet.  A
        read before the binding is the previous lane's residue, which
        ``_check_registers`` rejects as carried."""
        self.consecutive = {self.lane_index}
        bound: dict[str, int] = {}
        for stmt in body:
            if bound:
                _ElideResidueBounds(bound, self._constant_value).visit(stmt)
            if not (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call)
                and call_path_str(stmt.value.func) == "_cute_python_mod"
                and len(stmt.value.args) == 2
            ):
                continue
            index = _strip_int32(stmt.value.args[0])
            m = self._constant_value(stmt.value.args[1])
            if (
                isinstance(index, ast.Name)
                and index.id == self.lane_index
                and m is not None
                and m > 0
                and m % self.extent == 0
                and _bindings(body, stmt.targets[0].id) == 1
            ):
                self.consecutive.add(stmt.targets[0].id)
                bound[stmt.targets[0].id] = m

    def _constant_value(self, node: ast.expr) -> int | None:
        """An integer literal, a constexpr, or a name holding one literal in
        the loop (``_definition``); None for a ``cutlass.Int32`` of one past
        the type's range, which the device wraps."""
        stripped = _strip_int32(node)
        value: int | None = None
        if isinstance(stripped, ast.Constant) and type(stripped.value) is int:
            value = stripped.value
        elif isinstance(stripped, ast.Name):
            if stripped.id in self.constexpr_values:
                value = self.constexpr_values[stripped.id]
            else:
                definition = self._definition(stripped.id)
                if (
                    isinstance(definition, ast.Constant)
                    and type(definition.value) is int
                ):
                    value = definition.value
        if value is None or (stripped is not node and not -(2**31) <= value < 2**31):
            return None
        return value

    def _find_private_caches(self, body: Sequence[ast.stmt]) -> None:
        """The two-pass register caches (``fuse_two_pass_loads``) the body
        updates in place, which each lane's copy may hold in a cache of its
        own (``_lane_copy``).  A sweep of the tile writes each slot a later
        sweep of the same tile reads: the pass emits caches
        (``FUSE_CACHE_PREFIX``) only so, which is what proves that a lane
        reads only what it wrote.  ``_written_before_read`` only rejects a
        body that reads a cache before writing it, by the statements' order
        (``cache[k] = cache[k] + x``, a read in a branch's other arm or in
        the loop that writes).  The cache must be a register tensor of a
        known size the code before the loop allocates, the body never
        rebinds and the code after the loop never reads, and the lanes'
        copies hold at most ``MAX_PRIVATE_CACHE_ELEMENTS`` elements
        together."""
        allocations = {
            stmt.targets[0].id: stmt
            for stmt in self.preceding
            if isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id.startswith(FUSE_CACHE_PREFIX)
            and isinstance(stmt.value, ast.Call)
            and call_path_str(stmt.value.func) == "cute.make_rmem_tensor"
            and stmt.value.args
        }
        rebound = {name for stmt in body for name in names_written(stmt)}
        later = {name for stmt in self.following for name in names_read(stmt)}
        updated = {
            name
            for stmt in self._all_statements(body)
            for name in _mutated_receivers(stmt)
        }
        sizes = {
            name: self._constant_value(cast("ast.Call", stmt.value).args[0])
            for name, stmt in allocations.items()
        }
        self.private_caches = {
            name: stmt
            for name, stmt in allocations.items()
            if name in updated
            and name not in rebound
            and name not in later
            and _written_before_read(body, name)
            and sizes[name] is not None
        }
        elements = sum(cast("int", sizes[name]) for name in self.private_caches)
        if elements * self.extent > MAX_PRIVATE_CACHE_ELEMENTS:
            raise _Decline("the lanes' copies of the register caches would not fit")
        caches = {id(stmt) for stmt in self.private_caches.values()}
        named_before = {
            name
            for stmt in self.preceding
            if id(stmt) not in caches
            for name in names_read(stmt) | names_written(stmt)
        }
        self.dead_allocations = [
            stmt
            for name, stmt in self.private_caches.items()
            if name not in named_before
        ]

    def _varying_names(self, body: Sequence[ast.stmt]) -> set[str]:
        """Names whose value differs between lanes: the lane variable, every
        name a statement depending on a varying name writes (an in-place
        update or ``append`` makes its receiver vary), and what a statement
        every lane runs even on lane-invariant operands writes (``_movable``:
        an ``append``, an augmented assignment, a reduce), so a list of
        lane-invariant values still gets one per lane and a reduce that reads
        its own result updates each lane's own value.

        A loop or branch whose header, test or variable varies (a region) runs
        per lane.  When the region reads none of the names it writes before
        writing them and nothing after it may read them, its names are its
        own (``self.region_names``): each lane's copy of the region renames
        them, and their definitions elsewhere stay lane-invariant (the exact
        division's redo branch redefines the second pass's indices).
        Otherwise every name the region writes varies."""
        varying = {self.lane_var, *self.private_caches}
        while True:
            before = len(varying)
            self.region_names = {}
            self._propagate(body, varying, frozenset())
            if len(varying) == before:
                return varying

    @staticmethod
    def _per_lane(stmt: ast.For | ast.If, varying: set[str]) -> bool:
        """Whether the loop or branch ``stmt`` differs per lane: its header or
        test reads a varying name, or the loop binds a varying variable (each
        lane's copy of the loop then binds a name of its own, and so must what
        its body updates)."""
        if isinstance(stmt, ast.For):
            return bool(names_read(stmt.iter) & varying) or (
                cast("ast.Name", stmt.target).id in varying
            )
        return bool(names_read(stmt.test) & varying)

    def _propagate(
        self, stmts: Sequence[ast.stmt], varying: set[str], after: frozenset[str]
    ) -> None:
        """``after`` holds the names read anywhere the lane may run after
        ``stmts``: the statements following them and the bodies of the loops
        around them."""
        # What the statements after each one read, from the end.
        reads_after: list[frozenset[str]] = []
        reads = after
        for stmt in reversed(stmts):
            reads_after.append(reads)
            reads |= names_read(stmt)
        for stmt, later in zip(stmts, reversed(reads_after), strict=True):
            if isinstance(stmt, (ast.For, ast.If)):
                if self._per_lane(stmt, varying):
                    self._region(stmt, varying, later)
                elif isinstance(stmt, ast.For):
                    self._propagate(
                        stmt.body, varying, later | frozenset(names_read(stmt))
                    )
                else:
                    self._propagate(stmt.body, varying, later)
                    self._propagate(stmt.orelse, varying, later)
                continue
            if names_read(stmt) & varying or not self._movable(stmt):
                # A statement that never runs once (``_invariant``) runs in
                # every lane, so what it updates is each lane's own even when
                # its operands are lane-invariant: its copies, one after
                # another on one name, would apply a reduce of that name's
                # value once per lane.
                varying |= names_written(stmt)
                varying |= _mutated_receivers(stmt)

    def _region(
        self, region: ast.For | ast.If, varying: set[str], later: frozenset[str]
    ) -> None:
        written = self._region_writes(region)
        if not (
            exposed_reads([region], defines=_definitely_written) & written
        ) and not (written & later):
            self.region_names[id(region)] = written
            return
        varying |= written

    def _region_writes(self, region: ast.stmt) -> set[str]:
        written = names_written(region)
        for node in ast.walk(region):
            if isinstance(node, ast.stmt):
                written |= _mutated_receivers(node)
        return written

    def _check_registers(self, body: Sequence[ast.stmt]) -> None:
        if names_read(cast("ast.expr", self.lane_base)) & self.varying:
            # The lanes' indices ``base + lane`` differ only while every lane
            # reads one base.
            raise _Decline("the lane index's base differs per lane")
        written = {name for stmt in body for name in self._region_writes(stmt)}
        written.discard(self.lane_var)
        written -= self.private_caches.keys()
        if (
            exposed_reads(body, defined={self.lane_var}, defines=_definitely_written)
            & written
        ):
            # A value carried from one lane to the next.
            raise _Decline("a value carried from one lane to the next")
        later_reads: set[str] = set()
        for stmt in self.following:
            later_reads |= {self.rename_groups.get(n, n) for n in names_read(stmt)}
        if later_reads & (written | {self.lane_var}):
            # The jammed lanes leave no lane variable behind either.
            raise _Decline("a name the loop writes is read after it")

    def _check_memory(self, body: Sequence[ast.stmt]) -> None:
        loads: dict[str, list[_Pointer | None]] = {}
        stores: list[_Pointer] = []
        lists = {
            cast("ast.Name", stmt.targets[0]).id
            for stmt in self._all_statements(body)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.List)
            and not stmt.value.elts
        }
        private_atomics = self._private_atomics(body)
        for node in (n for stmt in body for n in ast.walk(stmt)):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in self.tensors
            ):
                # ``x[i, j]`` (CuTe's ``torch.gather``) reads ``x`` without a
                # pointer the checks below parse.
                raise _Decline("a tensor accessed by subscript")
            if not isinstance(node, ast.Call) or id(node) in private_atomics:
                continue
            access = match_packet_access(node)
            loaded_pointer = _load_pointer(access)
            if loaded_pointer is not None:
                # A masked access may point at a fallback element of the
                # same tensor: its tensor is known, its elements are not.
                roots = _tensor_roots(loaded_pointer)
                if len(roots) != 1 or not roots <= self.tensors.keys():
                    raise _Decline("a load through a pointer the jam does not parse")
                (root,) = roots
                loads.setdefault(root, []).append(_parse_pointer(loaded_pointer))
                continue
            if access is not None and access.kind in _STORE_KINDS:
                pointer = _parse_pointer(access.pointer)
                if pointer is None or pointer.tensor not in self.tensors:
                    raise _Decline("a store through a pointer the jam does not parse")
                if any(
                    frozenset((pointer.tensor, earlier.tensor))
                    not in self.proven_disjoint_tensor_pairs
                    for earlier in stores
                    if earlier.tensor != pointer.tensor
                ):
                    raise _Decline("stores into two tensors that may overlap")
                stores.append(pointer)
                self.stored.add(pointer.tensor)
                continue
            name = call_path_str(node.func)
            if (
                name in _LOOP_ITERATORS
                or name == _BARRIER
                # The fragment form fills a buffer it is handed, an update in
                # place the register checks do not see.
                or (name in GROUPED_SHARED_REDUCES and name != _TWO_STAGE_FRAGMENT)
                or is_cta_reduce_call(node)
                or name in EXACT_DIVISION_HELPERS
                # A load the matcher does not parse would escape the memory
                # checks: it declines.
                or _is_proven_relocatable_call(node, allow_load=False)
            ):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in lists
            ):
                continue
            raise _Decline(
                "a call that is not a load, store, reduce, barrier or pure "
                f"computation: {ast.unparse(node)}"
            )
        for tensor in dict.fromkeys(pointer.tensor for pointer in stores):
            # The lanes touch disjoint elements of a stored tensor when every
            # access of it, a load too, names the lane index in one common
            # dimension: a workspace row the first sweep stores and the
            # second reads back is each lane's own.  ``out[i, j]`` beside
            # ``out[j, i]`` or ``out[0, j]`` reaches another lane's elements,
            # in another order once jammed.
            accesses = [p for p in stores if p.tensor == tensor]
            shared = self._shared_by_lanes(tensor, [*accesses, *loads.get(tensor, [])])
            if shared is not None:
                raise _Decline(f"a store into a tensor {shared}")
            if any(
                frozenset((tensor, other)) not in self.proven_disjoint_tensor_pairs
                for other in loads
                if other != tensor
            ):
                raise _Decline("a store into a tensor that may alias a loaded one")

    def _private_atomics(self, body: Sequence[ast.stmt]) -> set[int]:
        """The calls of the body's adds into privatized atomic targets that
        the jam may reorder, with the pointer arithmetic in them (the add's
        value and coordinates are checked like any other operand).

        Such an add is a relaxed CTA-scope ``cute.arch.atomic_add``
        statement into a shared-memory copy the code before the loop
        allocates (``atomic_privatization.py``): the body names the copy,
        or a view of it, nowhere else, so no lane reads what another lane
        adds.  It is also the copy's only add in the body and sits in no
        loop of it nor in a branch that runs per lane (``_jams``), whose
        lanes' copies would not sit side by side: the shape
        ``_combined_atomic`` was written and measured for (T5's scatter).
        Combining the lanes' copies of each of several adds on its own, or
        those in a jammed loop, would be as sound, as the privatized adds
        leave their order unspecified, but such bodies keep the lane loop, as
        does an add the lanes would issue uncombined.  The statements are
        recorded in ``self.private_atomics``."""
        allocated = {
            cast("ast.Name", stmt.targets[0]).id
            for stmt in self.preceding
            if isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and call_path_str(stmt.value.func) == "cute.make_tensor"
            and stmt.value.args
            and isinstance(stmt.value.args[0], ast.Call)
            and call_path_str(stmt.value.args[0].func) == "cute.arch.alloc_smem"
        }
        sites: dict[str, list[tuple[ast.Expr, str]]] = {}
        for stmt in self._unlooped_statements(body):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                site = _private_atomic_site(stmt.value)
                if site is not None:
                    copy, addend = site
                    sites.setdefault(copy, []).append((stmt, addend))
        referenced = [
            node.id
            for stmt in body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Name)
        ]
        admitted: set[int] = set()
        for copy, statements in sites.items():
            views = {copy}
            for stmt in self.preceding:
                if names_read(stmt) & views:
                    views |= names_written(stmt)
            if (
                copy not in allocated
                or copy in self.tensors
                or len(statements) != 1
                or sum(name in views for name in referenced) != 2
            ):
                continue
            ((stmt, addend),) = statements
            self.private_atomics[id(stmt)] = (copy, addend)
            call = cast("ast.Call", stmt.value)
            pointer = cast("ast.Attribute", call.args[0])
            admitted |= {id(call), id(cast("ast.BinOp", pointer.value).right)}
        return admitted

    def _unlooped_statements(self, stmts: Sequence[ast.stmt]) -> list[ast.stmt]:
        """``stmts`` and the statements of their branches that every lane
        runs alike (``_jams``), outside every loop."""
        result: list[ast.stmt] = []
        for stmt in stmts:
            result.append(stmt)
            if isinstance(stmt, ast.If) and self._jams(stmt):
                result.extend(self._unlooped_statements(stmt.body))
                result.extend(self._unlooped_statements(stmt.orelse))
        return result

    def _shared_by_lanes(
        self, tensor: str, accesses: Sequence[_Pointer | None]
    ) -> str | None:
        """Why the lanes' ``accesses`` may touch one another's elements of
        ``tensor``, or None when each lane's accesses touch only its own
        elements: no two index
        tuples of the tensor address one element, and every access indexes
        one common dimension (of nonzero stride) by the lane index."""
        layout = self.tensors[tensor]
        if not _non_overlapping(layout):
            return "two of whose index tuples may address one element"
        common: set[int] | None = None
        for pointer in accesses:
            if pointer is None:
                return "with an access whose indices the jam does not parse"
            dims = {
                dim
                for dim in pointer.lane_dims(self.lane_index)
                if layout.stride(dim) != 0
            }
            common = dims if common is None else common & dims
        if not common:
            return (
                "whose accesses do not all address one dimension of it by the "
                "lane index"
            )
        return None

    @staticmethod
    def _all_statements(stmts: Sequence[ast.stmt]) -> list[ast.stmt]:
        """``stmts`` and every statement nested in them."""
        return [
            node
            for stmt in stmts
            for node in ast.walk(stmt)
            if isinstance(node, ast.stmt)
        ]

    # ------------------------------------------------------------- the jam

    def _lane_copy(self, body: Sequence[ast.stmt], lane: int) -> list[ast.stmt]:
        renames = {
            name: self.new_var(f"{name}_lane{lane}")
            for name in sorted(self.varying - {self.lane_var})
        }
        self.lane_names.append(renames)
        for name, allocation in self.private_caches.items():
            self.allocations.append(
                create(
                    ast.Assign,
                    targets=[create(ast.Name, id=renames[name], ctx=ast.Store())],
                    value=deep_clone(allocation.value),
                )
            )
        regions = {
            id(node): node
            for stmt in body
            for node in ast.walk(stmt)
            if id(node) in self.region_names
        }
        for key, node in regions.items():
            setattr(node, _JAM_REGION_ATTR, key)
        copy = [deep_clone(stmt) for stmt in body]
        for stmt in copy:
            for node in ast.walk(stmt):
                key = getattr(node, _JAM_REGION_ATTR, None)
                if key is not None:
                    # The region's own names, before the lane's: each lane's
                    # copy of the region keeps them apart.
                    ast_rename(
                        node,
                        {
                            name: self.new_var(f"{name}_lane{lane}")
                            for name in sorted(self.region_names[key])
                        },
                    )
            ast_rename(stmt, renames)
        return [
            substitute_names(
                stmt, {self.lane_var: ast.Constant(value=lane)}, clone=deep_clone
            )
            for stmt in copy
        ]

    def _jams(self, stmt: ast.stmt) -> bool:
        """Whether ``stmt``'s lane copies merge into one statement: a loop or
        branch that every lane runs alike (not ``_per_lane``)."""
        return isinstance(stmt, (ast.For, ast.If)) and not self._per_lane(
            stmt, self.varying
        )

    def _invariant(self, stmt: ast.stmt) -> bool:
        """A statement every lane runs alike, which runs once: it reads and
        writes no varying name and is ``_movable``, or it is a barrier, whose
        copies would follow each other."""
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and call_path_str(stmt.value.func) == _BARRIER
        ):
            return True
        if (names_read(stmt) | names_written(stmt)) & self.varying:
            return False
        if _mutated_receivers(stmt) & self.varying:
            return False
        return self._movable(stmt)

    @staticmethod
    def _movable(stmt: ast.stmt) -> bool:
        """Whether ``stmt`` may run once for every lane when its operands are
        lane-invariant: an assignment that calls only pure computations and
        loads.  Such a load reads a tensor the body never stores to: every
        access of a stored tensor names the varying lane index
        (``_check_memory``)."""
        return isinstance(stmt, ast.Assign) and all(
            _is_proven_relocatable_call(node, allow_load=True)
            for node in ast.walk(stmt)
            if isinstance(node, ast.Call)
        )

    def _merge(
        self, template: Sequence[ast.stmt], copies: Sequence[Sequence[ast.stmt]]
    ) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for position, stmt in enumerate(template):
            parts = [copy[position] for copy in copies]
            if isinstance(stmt, (ast.For, ast.If)) and self._jams(stmt):
                merged = deep_clone(parts[0])
                assert isinstance(merged, (ast.For, ast.If))
                merged.body = self._merge(
                    stmt.body, [cast("ast.For | ast.If", p).body for p in parts]
                )
                merged.orelse = self._merge(
                    stmt.orelse, [cast("ast.For | ast.If", p).orelse for p in parts]
                )
                result.append(merged)
            elif self._invariant(stmt):
                result.append(parts[0])
            else:
                for lane, part in enumerate(parts):
                    setattr(part, _JAM_GROUP_ATTR, (id(stmt), lane))
                    result.append(part)
        return result

    # ------------------------------------------------------- vector loads

    def _vectorize_loads(self, stmts: list[ast.stmt]) -> None:
        """Replace each run of the lanes' copies of a scalar load that reads
        the lanes' consecutive elements of a stride-1 dimension (indexed by the
        lane index or a residue of it) by packet loads ahead of the run, one
        per run of lanes a packet holds."""
        position = 0
        while position < len(stmts):
            stmt = stmts[position]
            for field in ("body", "orelse"):
                inner = getattr(stmt, field, None)
                if isinstance(inner, list):
                    self._vectorize_loads(inner)
            group = self._group_at(stmts, position)
            if group is None:
                position += 1
                continue
            loads = [self._scalar_loads(part) for part in group]
            vector_loads: list[ast.stmt] = []
            for site in range(len(loads[0])):
                calls = [lane_loads[site] for lane_loads in loads]
                vector = self._vector_load(calls)
                if vector is None:
                    continue
                vector_vars, statements, dtype = vector
                vector_loads.extend(statements)
                width = self.extent // len(vector_vars)
                carrier = _CUTE_VECTOR_UNROLL_CARRIER[dtype]
                element = _CUTE_VECTOR_UNROLL_DTYPES[dtype]
                for lane, (part, call) in enumerate(zip(group, calls, strict=True)):
                    vector_var = vector_vars[lane // width]
                    extract = expr_from_string(
                        f"{carrier}({vector_var}[{lane % width}]).bitcast({element})"
                    )
                    replace_node(part, call, extract)
            stmts[position:position] = vector_loads
            position += len(vector_loads) + len(group)

    def _group_at(
        self, stmts: Sequence[ast.stmt], position: int
    ) -> list[ast.stmt] | None:
        """The lanes' copies of one statement starting at ``position``, in
        lane order, or None."""
        first = getattr(stmts[position], _JAM_GROUP_ATTR, None)
        if first is None or first[1] != 0:
            return None
        group = list(stmts[position : position + self.extent])
        if len(group) != self.extent or any(
            getattr(part, _JAM_GROUP_ATTR, None) != (first[0], lane)
            for lane, part in enumerate(group)
        ):
            return None
        return group

    @staticmethod
    def _scalar_loads(stmt: ast.stmt) -> list[ast.Call]:
        """The scalar loads (``(ptr).load()``, ``cute.arch.load(ptr, T)``) of
        ``stmt`` that run whenever it runs and before anything it stores, in
        a fixed walk order: none in a conditional operand, and none in the
        body of a loop or branch, which a lane's copy may not run at all (its
        guard may keep an invalid address from being loaded) or may run after
        storing what it then loads (the packet goes ahead of all the
        copies)."""
        conditional: set[int] = set()
        for node in ast.walk(stmt):
            if isinstance(node, ast.IfExp):
                conditional |= {id(child) for child in ast.walk(node)}
            elif isinstance(node, ast.BoolOp):
                for value in node.values[1:]:
                    conditional |= {id(child) for child in ast.walk(value)}
            elif isinstance(node, (ast.For, ast.If)):
                for inner in (*node.body, *node.orelse):
                    conditional |= {id(child) for child in ast.walk(inner)}
        return [
            node
            for node in ast.walk(stmt)
            if isinstance(node, ast.Call)
            and id(node) not in conditional
            and (access := match_packet_access(node)) is not None
            and access.kind in ("load", "arch_load")
        ]

    def _vector_load(
        self, calls: Sequence[ast.Call]
    ) -> tuple[list[str], list[ast.stmt], torch.dtype] | None:
        """Packet loads of the lanes' consecutive elements, one per run of
        lanes a packet holds (two four-element packets for eight fp32
        lanes), or None."""
        pointers = []
        for call in calls:
            loaded = _load_pointer(match_packet_access(call))
            assert loaded is not None
            pointer = _parse_pointer(loaded)
            if pointer is None:
                return None
            pointers.append(pointer)
        tensor_name = pointers[0].tensor
        tensor = self.tensors.get(tensor_name)
        if (
            tensor is None
            or tensor.dtype not in _CUTE_VECTOR_UNROLL_CARRIER
            or self.extent & (self.extent - 1)
        ):
            return None
        width = min(self.extent, _CUTE_VECTOR_MAX_BYTES // tensor.dtype.itemsize)
        if width < 2:
            return None
        lane_dims = [
            (dim, name)
            for dim, index in pointers[0].indices.items()
            for name in self.consecutive
            if ast.unparse(_strip_int32(index)) == self._lane_name(name, 0)
        ]
        if len(lane_dims) != 1:
            return None
        ((dim, name),) = lane_dims
        if tensor.stride(dim) != 1:
            return None
        for lane, pointer in enumerate(pointers):
            if (
                pointer.tensor != tensor_name
                or pointer.indices.keys() != pointers[0].indices.keys()
            ):
                return None
            for other, index in pointer.indices.items():
                if other == dim:
                    if ast.unparse(_strip_int32(index)) != self._lane_name(name, lane):
                        return None
                elif ast.unparse(index) != ast.unparse(pointers[0].indices[other]):
                    return None
        if not (
            self._multiple_of(self.lane_base, width)
            and cute_reduction_vector_layout_aligned(self.env, tensor, dim, width)
        ):
            return None
        first = calls[0]
        first_access = match_packet_access(first)
        # The scalar loads' cache hint carries over to the packet.
        suffix = ""
        if first_access is not None and first_access.kind == "arch_load":
            # ``cute.arch.load(ptr, cutlass.T, cop=...)``: one element of the
            # tensor's type (not a byte-packed word).
            policy = first_access.cache_policy()
            if (
                policy is None
                or first_access.carrier != _CUTE_VECTOR_UNROLL_DTYPES[tensor.dtype]
            ):
                return None
            suffix = policy.suffix
        vector_vars: list[str] = []
        statements: list[ast.stmt] = []
        for start in range(0, self.extent, width):
            loaded = _load_pointer(match_packet_access(calls[start]))
            assert loaded is not None
            vector_var = self.new_var("_jam_vec")
            load = _cute_unroll_vec_load_expr(
                ast.unparse(loaded), tensor.dtype, width, suffix
            )
            vector_vars.append(vector_var)
            statements.append(statement_from_string(f"{vector_var} = {load}"))
        return vector_vars, statements, tensor.dtype

    def _lane_name(self, name: str, lane: int) -> str:
        return self.lane_names[lane][name]

    def _multiple_of(self, node: ast.expr | None, factor: int, depth: int = 0) -> bool:
        """Whether the lane-invariant ``node`` is a multiple of ``factor``: a
        literal, a constexpr, a product with such a factor, a sum of such
        terms, or a name the code before the loop defines as one."""
        if node is None or depth > 16:
            return False
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value % factor == 0
        if isinstance(node, ast.Name):
            if node.id in self.constexpr_values:
                return self.constexpr_values[node.id] % factor == 0
            return self._multiple_of(self._definition(node.id), factor, depth + 1)
        if (operand := _int32_operand(node)) is not None:
            return self._multiple_of(operand, factor, depth + 1)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Mult):
                return self._multiple_of(
                    node.left, factor, depth + 1
                ) or self._multiple_of(node.right, factor, depth + 1)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return self._multiple_of(
                    node.left, factor, depth + 1
                ) and self._multiple_of(node.right, factor, depth + 1)
        return False

    def _definition(self, name: str) -> ast.expr | None:
        """What ``name`` holds in the loop: the value of the one statement
        before the loop that binds it, a plain assignment, when no statement
        of the body rebinds it; else None.  Names compare through
        ``rename_groups``, as the body's are merged (``run``): another
        member of ``name``'s group binds it too."""
        merged = self.rename_groups.get(name, name)

        def binds(stmt: ast.stmt) -> bool:
            return merged in {self.rename_groups.get(n, n) for n in names_written(stmt)}

        binders = [stmt for stmt in self.preceding if binds(stmt)]
        if len(binders) != 1 or any(binds(stmt) for stmt in self.loop.body):
            return None
        (stmt,) = binders
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            return stmt.value
        return None

    # ------------------------------------------------- rewritten lane copies

    def _rewrite_groups(
        self,
        stmts: list[ast.stmt],
        rewrite: Callable[[Sequence[ast.stmt]], list[ast.stmt] | None],
    ) -> list[ast.stmt]:
        """Replace each run of the lanes' copies of a statement
        (``_group_at``), at any depth, by what ``rewrite`` makes of it: one
        fragment reduce of all lanes' values for a shared-memory two-stage
        reduce (``_batched_reduce``), one add per cell for a privatized
        atomic add (``_combined_atomic``); None keeps the copies."""
        result: list[ast.stmt] = []
        position = 0
        while position < len(stmts):
            stmt = stmts[position]
            for field in ("body", "orelse"):
                inner = getattr(stmt, field, None)
                if isinstance(inner, list):
                    setattr(stmt, field, self._rewrite_groups(inner, rewrite))
            group = self._group_at(stmts, position)
            rewritten = rewrite(group) if group is not None else None
            if rewritten is None:
                result.append(stmt)
                position += 1
                continue
            result.extend(rewritten)
            position += self.extent
        return result

    def _batched_reduce(self, group: Sequence[ast.stmt]) -> list[ast.stmt] | None:
        calls = []
        for part in group:
            if not (
                isinstance(part, ast.Assign) and isinstance(part.targets[0], ast.Name)
            ):
                return None
            found = [
                node
                for node in ast.walk(part.value)
                if isinstance(node, ast.Call)
                and call_path_str(node.func) in GROUPED_REDUCE_HELPERS
            ]
            if len(found) != 1 or call_path_str(found[0].func) != _TWO_STAGE:
                return None
            calls.append(found[0])
        first = calls[0]
        keywords = {kw.arg: kw.value for kw in first.keywords}
        if (
            len(first.args) != 6
            or set(keywords) != {"pre", "group_span", "group_count"}
            or not all(
                isinstance(value, ast.Constant) and type(value.value) is int
                for value in keywords.values()
            )
        ):
            return None
        pre, span, count = (
            cast("int", cast("ast.Constant", keywords[key]).value)
            for key in ("pre", "group_span", "group_count")
        )
        if not ReduceScope.fragment_fits(pre, span, self.extent):
            return None
        shared = [ast.unparse(arg) for arg in first.args[1:]]
        shared_keywords = sorted(
            (kw.arg, ast.unparse(kw.value)) for kw in first.keywords
        )
        if any(
            len(call.args) != 6
            or not isinstance(call.args[0], ast.Name)
            or [ast.unparse(arg) for arg in call.args[1:]] != shared
            or sorted((kw.arg, ast.unparse(kw.value)) for kw in call.keywords)
            != shared_keywords
            for call in calls
        ):
            return None
        identity = first.args[2]
        if not (
            isinstance(identity, ast.Call)
            and (dtype := call_path_str(identity.func)) is not None
            and dtype.startswith("cutlass.")
        ):
            return None
        values = self.new_var("_jam_reduce_in")
        results = self.new_var("_jam_reduce_out")
        self.allocations.extend(
            statement_from_string(
                f"{name} = cute.make_rmem_tensor({self.extent}, {dtype})"
            )
            for name in (values, results)
        )
        batched: list[ast.stmt] = [
            statement_from_string(f"{values}[{lane}] = {ast.unparse(call.args[0])}")
            for lane, call in enumerate(calls)
        ]
        batched.append(
            statement_from_string(
                f"{_TWO_STAGE_FRAGMENT}({values}, {results}, {', '.join(shared)}, "
                f"count={self.extent}, pre={pre}, group_span={span}, "
                f"group_count={count})"
            )
        )
        for lane, (part, call) in enumerate(zip(group, calls, strict=True)):
            replace_node(part, call, expr_from_string(f"{results}[{lane}]"))
            batched.append(part)
        return batched

    # ------------------------------------------------------ combined adds

    def _combined_atomic(self, group: Sequence[ast.stmt]) -> list[ast.stmt] | None:
        """The lanes' copies of a privatized atomic add
        (``_private_atomics``), summed per cell into one add each.

        A thread's lanes often add into one cell (keys that share a bucket),
        and the copy's fp32 adds are compare-and-swap loops that serialize
        on a contended cell.  Each lane's cell is its linear index into the
        copy and its value is bound to a name, both once; a lane adds the
        values of its own and every later lane that addresses its cell, in
        lane order, and only the first lane of a cell adds into the copy.
        That reorders the adds into a cell, which the privatized adds leave
        in no particular order already (``atomic_privatization.py``); int32
        adds are exact either way."""
        template = getattr(group[0], _JAM_GROUP_ATTR)[0]
        if template not in self.private_atomics:
            return None
        copy, addend = self.private_atomics[template]
        calls = [cast("ast.Call", cast("ast.Expr", part).value) for part in group]
        values = [
            next(kw.value for kw in call.keywords if kw.arg == addend) for call in calls
        ]
        cells = [self.new_var("_jam_cell") for _ in group]
        vals = [self.new_var("_jam_val") for _ in group]
        sums = [self.new_var("_jam_add") for _ in group]
        combined: list[ast.stmt] = []
        for cell, val, call, value in zip(cells, vals, calls, values, strict=True):
            pointer = cast("ast.BinOp", cast("ast.Attribute", call.args[0]).value)
            combined.extend(
                (
                    statement_from_string(f"{cell} = {ast.unparse(pointer.right)}"),
                    statement_from_string(f"{val} = {ast.unparse(value)}"),
                )
            )
        for lane, (cell, total) in enumerate(zip(cells, sums, strict=True)):
            combined.append(statement_from_string(f"{total} = {vals[lane]}"))
            combined.extend(
                statement_from_string(
                    f"{total} = ({total} + {vals[later]}) "
                    f"if {cells[later]} == {cell} else {total}"
                )
                for later in range(lane + 1, len(group))
            )
        for lane, (cell, total, call) in enumerate(
            zip(cells, sums, calls, strict=True)
        ):
            add = deep_clone(call)
            add.args[0] = cast(
                "ast.expr", expr_from_string(f"({copy}.iterator + {cell}).llvm_ptr")
            )
            for keyword in add.keywords:
                if keyword.arg == addend:
                    keyword.value = cast("ast.expr", expr_from_string(total))
            stmt: ast.stmt = create(ast.Expr, value=add)
            # Nested tests rather than a chain of ``and``: the DSL expands
            # a boolean operator's operands once per later operand.
            for earlier in reversed(cells[:lane]):
                test = statement_from_string(f"if {cell} != {earlier}:\n    pass")
                assert isinstance(test, ast.If)
                test.body = [stmt]
                stmt = test
            combined.append(stmt)
        return combined


class _LoadReuse:
    """Serve a load of the jammed body from an earlier load of the same
    elements.

    A two-pass row kernel walks its row in two device loops, the second
    reloading what the first loaded (the scores and the heads' bias packets
    of T5's softmax).  When the tile spans the whole row each loop runs once:
    the loops (and the one-lane loops inside them) become straight-line code,
    their variables bound to the loop's start, and a load whose address
    evaluates to an earlier load's reads that load's value: directly, or from
    a trace-time list the earlier loop's iterations fill when both sit in
    constexpr loops of the same extent.  Only loads of tensors the body never
    stores to are served (``stored``: each lane's own workspace rows, which a
    later sweep reads back after a store), so the earlier value is what the
    load would read.
    Addresses are compared as expressions over the definitions reaching them;
    a name some statement defines in a way the comparison does not follow
    (a branch, a rolled loop, a loaded value, a call no proof covers that is
    handed the name) gets a value of its own, and so does a name that may
    hold a container the body creates or updates in place, wherever it is
    read (``_container_names``).  A constexpr loop whose body rebinds its
    target is not walked.  The served values keep at most
    ``_MAX_REUSED_REGISTERS`` registers live, a plain scalar counted by its
    tensor's element type.
    """

    def __init__(
        self,
        constexpr_values: Mapping[str, int],
        new_var: Callable[[str], str],
        tensors: Mapping[str, torch.Tensor],
        stored: frozenset[str],
    ) -> None:
        self.constexpr_values = dict(constexpr_values)
        self.new_var = new_var
        # The kernel's tensors, for the element type of a plain scalar load.
        self.tensors = tensors
        self.stored = stored
        # Name -> the canonical text of its current value.
        self.values: dict[str, str] = {}
        # Canonical address -> the earlier load: its name, the constexpr
        # loop it sits in (None at the top), the value it read and the
        # trace-time list serving it once a later loop reads it.
        self.loads: dict[str, _EarlierLoad] = {}
        self.tokens = 0
        # Canonical text -> the short id standing for it: a value's text
        # names its operands' ids, where spelling them out would double a
        # chain of squarings at every step.
        self.ids: dict[str, str] = {}
        # Registers the served loads keep live (``_MAX_REUSED_REGISTERS``).
        self.registers = 0
        # Names that may hold a container (``_container_names``).
        self.containers: set[str] = set()

    def run(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        flat = self._flatten(stmts)
        self.containers = self._container_names(flat)
        self._walk(flat, None)
        return flat

    # Loops that run once become straight-line code.

    def _flatten(self, stmts: Sequence[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for stmt in stmts:
            start = self._single_trip_start(stmt)
            if start is not None:
                loop = cast("ast.For", stmt)
                target = cast("ast.Name", loop.target).id
                result.append(
                    statement_from_string(f"{target} = cutlass.Int32({start})")
                )
                result.extend(self._flatten(loop.body))
                continue
            if isinstance(stmt, ast.For) and loop_extent(stmt, kinds=("constexpr",)):
                stmt.body = self._flatten(stmt.body)
            result.append(stmt)
        return result

    def _single_trip_start(self, stmt: ast.stmt) -> int | None:
        """The start of a ``range`` loop with static bounds that runs once."""
        if not (
            isinstance(stmt, ast.For)
            and not stmt.orelse
            and isinstance(stmt.iter, ast.Call)
            and call_path_str(stmt.iter.func) == "range"
            and (bounds := _range_bounds(stmt.iter)) is not None
        ):
            return None
        if _trip_count_for(*bounds, self.constexpr_values) != 1:
            return None
        return _static_int(bounds[0], self.constexpr_values)

    # Value numbering over the straight-line body and its constexpr loops.

    def _fresh(self, name: str) -> str:
        self.tokens += 1
        return f"<{name}#{self.tokens}>"

    @staticmethod
    def _container_names(stmts: Sequence[ast.stmt]) -> set[str]:
        """Names that may hold a container the body creates or updates in
        place: one bound to a value with a display or comprehension in it
        (``a = []``), one updated in place (``a[i] = v``, ``a.append(v)``)
        or handed to a call that may fill it, then every name bound to a
        value read from such a name (``b = a``, ``n = len(a)``), and every
        name such a name is bound to the object or an element of (``b = a``
        or ``b = a[i]``, then ``b.append(v)``, updates ``a``)."""
        displays = (
            ast.List,
            ast.ListComp,
            ast.Dict,
            ast.DictComp,
            ast.Set,
            ast.SetComp,
        )
        result: set[str] = set()
        bindings: list[tuple[str, set[str], set[str]]] = []
        for node in ast.walk(ast.Module(body=list(stmts), type_ignores=[])):
            if isinstance(node, ast.stmt):
                result |= _mutated_receivers(node)
            result |= _updated_arguments(node)
            if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if any(isinstance(inner, displays) for inner in ast.walk(value)):
                    result.add(target.id)
                bindings.append((target.id, names_read(value), _alias_sources(value)))
        changed = True
        while changed:
            changed = False
            for target, reads, bound in bindings:
                if target not in result and reads & result:
                    result.add(target)
                    changed = True
                if target in result and not bound <= result:
                    result |= bound
                    changed = True
        return result

    def _canonical(self, node: ast.AST) -> str:
        """The id of ``node``'s value: its text with each name replaced by
        the value it holds.  An element of a container (``b[i]``) is a value
        of its own each time, and so is a name that may hold a container
        (``containers``): an update in place (``a[i] = v``, ``a.append(v)``,
        a call that fills ``a``) changes what every name holding the
        container reads (``b = a``, then ``b[0]`` or ``len(b)``), which the
        numbering does not follow."""
        replaced = deep_clone(node)
        for child in ast.walk(replaced):
            if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
                child.value.id = self._fresh(child.value.id)
            elif isinstance(child, ast.Name) and child.id in self.containers:
                child.id = self._fresh(child.id)
            elif isinstance(child, ast.Name) and child.id in self.values:
                child.id = self.values[child.id]
        return self.ids.setdefault(ast.unparse(replaced), f"<v{len(self.ids)}>")

    def _walk(self, stmts: list[ast.stmt], loop: _ConstexprLoop | None) -> None:
        position = 0
        while position < len(stmts):
            stmt = stmts[position]
            if isinstance(stmt, ast.For) and (
                extent := loop_extent(stmt, kinds=("constexpr",))
            ):
                # The body's names first get values of the loop's own: one
                # the body reads before it writes it carries from one
                # iteration to the next.  A nested constexpr loop is not
                # walked: no load in it is served or serves.  Nor is one
                # whose body rebinds its target, which then no longer
                # numbers the iteration a stash is read by.
                self._forget(stmt)
                target = cast("ast.Name", stmt.target).id
                if loop is None and not any(
                    target in names_written(inner) for inner in stmt.body
                ):
                    self.values[target] = f"<lane{extent}>"
                    self._walk(stmt.body, _ConstexprLoop(stmt, target, extent, stmts))
                    self._forget(stmt)
                # ``_serve`` may have put a stash's ``[]`` before an earlier
                # loop of this list, moving this one down.
                position = stmts.index(stmt) + 1
                continue
            if isinstance(stmt, (ast.For, ast.If)):
                self._forget(stmt)
                position += 1
                continue
            if is_name_assign(stmt):
                self._assign(stmt, cast("ast.Name", stmt.targets[0]).id, loop)
            else:
                self._forget(stmt)
            position += 1

    def _forget(self, stmt: ast.stmt) -> None:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.values[node.id] = self._fresh(node.id)
            elif isinstance(node, ast.stmt):
                for receiver in _mutated_receivers(node):
                    self.values[receiver] = self._fresh(receiver)
            for argument in _updated_arguments(node):
                self.values[argument] = self._fresh(argument)

    def _assign(self, stmt: ast.Assign, name: str, loop: _ConstexprLoop | None) -> None:
        access = match_packet_access(stmt.value)
        if (
            access is not None
            and access.kind in _LOAD_KINDS - {"masked_packet"}
            and not _tensor_roots(access.pointer) & self.stored
        ):
            key = f"{self._canonical(access.pointer)}|{access.width}|{access.carrier}"
            earlier = self.loads.get(key)
            if access.is_vector and None in (access.width, access.carrier):
                # A vector type the key does not spell: another load of the
                # address may read another type.
                self.values[name] = self._fresh(name)
                return
            if earlier is not None and self._serve(stmt, earlier, loop):
                # The value the earlier load read: its name may hold another
                # by now.
                self.values[name] = earlier.token
                return
            self.values[name] = self._fresh(name)
            registers = load_registers(access, self._element_bytes(access.pointer))
            self.loads[key] = _EarlierLoad(
                name, self.values[name], stmt, loop, registers
            )
            return
        if isinstance(stmt.value, ast.Name):
            # A copy holds the value it copies (``q = p``).
            self.values[name] = self.values.get(stmt.value.id, stmt.value.id)
            return
        if all(
            _is_proven_relocatable_call(node, allow_load=False)
            for node in ast.walk(stmt.value)
            if isinstance(node, ast.Call)
        ):
            self.values[name] = self._canonical(stmt.value)
        else:
            self._forget(stmt)

    def _element_bytes(self, pointer: ast.expr) -> int:
        """The bytes of an element of the tensor ``pointer`` addresses, for
        a plain ``.load()``, which spells no type: eight when the tensor is
        unknown."""
        roots = _tensor_roots(pointer)
        tensor = self.tensors.get(next(iter(roots))) if len(roots) == 1 else None
        return tensor.dtype.itemsize if tensor is not None else 8

    def _serve(
        self, stmt: ast.Assign, earlier: _EarlierLoad, loop: _ConstexprLoop | None
    ) -> bool:
        """Rewrite ``stmt`` to read ``earlier``'s value, if it reaches and
        keeping it live fits ``_MAX_REUSED_REGISTERS``."""
        if earlier.loop is None or (
            loop is not None and earlier.loop.node is loop.node
        ):
            # Same scope: the earlier name still holds the value.
            if self.values.get(earlier.name) != earlier.token:
                return False
            if not earlier.kept:
                if not self._keep(earlier.registers):
                    return False
                earlier.kept = True
            stmt.value = cast("ast.expr", expr_from_string(earlier.name))
            return True
        if loop is None or loop.extent != earlier.loop.extent:
            return False
        if earlier.stash is None:
            if not self._keep(earlier.registers * earlier.loop.extent):
                return False
            stash = self.new_var("_jam_stash")
            outer = earlier.loop
            outer.stmts.insert(
                outer.stmts.index(outer.node), statement_from_string(f"{stash} = []")
            )
            body = outer.node.body
            body.insert(
                body.index(earlier.stmt) + 1,
                statement_from_string(f"{stash}.append({earlier.name})"),
            )
            earlier.stash = stash
        stmt.value = cast(
            "ast.expr", expr_from_string(f"{earlier.stash}[{loop.target}]")
        )
        return True

    def _keep(self, registers: int) -> bool:
        """Charge ``registers`` more live registers, if they fit."""
        if self.registers + registers > _MAX_REUSED_REGISTERS:
            return False
        self.registers += registers
        return True


@dataclasses.dataclass
class _ConstexprLoop:
    node: ast.For
    target: str
    extent: int
    # The statement list holding the loop.
    stmts: list[ast.stmt]


@dataclasses.dataclass
class _EarlierLoad:
    name: str
    token: str
    stmt: ast.Assign
    loop: _ConstexprLoop | None
    registers: int
    # Whether the registers are charged for a read in the same scope.
    kept: bool = False
    stash: str | None = None
