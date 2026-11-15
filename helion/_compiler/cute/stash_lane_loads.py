"""Reuse an unrolled lane loop's loads in a later pass over the same lanes.

Role: device-body pass (``passes/pipeline.py``, after the lane unroll and
the cross-lane fold).

The lane split runs a chain of dependent reductions over a lane loop as one
pass per reduction (``lane_loops/split.py``,
``_split_dependent_lane_reductions``), every pass recomputing its inputs
from their producers, loads included: an online softmax over a tile walks
the thread's lanes once for the block's max and again for the sum of
``exp(x - max)``, loading every element twice.  Once the vectorized lane
loops of the reduced block are unrolled at trace time (the forced full
unroll of ``unroll_lane_loads``, for the cross-lane fold), each pass is a
``cutlass.range_constexpr`` loop, and the later pass's load of an element
the earlier pass loaded reads the same value.  This pass keeps the earlier
pass's loaded values in a trace-time list, which only names SSA values, and
reads the later pass's from it: one load per element instead of one per
pass.

A load of a later loop reuses one of an earlier loop when

- both loops are ``for X in cutlass.range_constexpr(N)`` with the same ``X``
  and ``N`` in one statement list, neither of them ends an iteration early
  (a ``break``, ``continue`` or ``return`` at any depth: the stash is filled
  by appends and read by lane number), and the load sits at the same path of
  nested constexpr loops (the same targets and extents) in both, with no
  statement in a loop's body assigning that loop's variable (the stash is
  indexed by them);
- the two load statements are the same text, and so are the statements on
  their paths that (transitively) define the names they read (the slice),
  and every call in those statements is a proven pure computation or a
  load, so the slice computes the same address and predicate from the same
  inputs;
- those inputs, the names the slice and the load read before assigning
  them (in program order), are written by no statement from the earlier
  loop through the later one (nor changed in place, through any name that
  may share their object: a trace-time list's ``reverse()``, a subscript
  or attribute store, an augmented assignment, a register fragment a call
  stores into), no loop before the load on its path rebinds one of the
  names the slice reads (other than the path's variables), and no loop on
  either path carries a name the slice reads or writes from one iteration
  into the next (a statement after the load that updates it for the next
  element);
- nothing that runs between the two loads (either loop, headers included,
  and every statement between them) may write memory the load reads: no
  store, atomic or packet flush (two kernel tensors may share memory), no
  subscript store except into a register tensor created in the span that
  the load does not read, no call that may write memory it does not name
  (impure inline asm, an unknown function: an address it is handed may
  have been loaded from memory), and no call that is neither a load nor a
  proven pure computation handed a kernel tensor, a tensor the kernel reads
  through ``.iterator`` or a name bound to a value that may point into one
  (a layout read, as ``x.layout.stride[1]``, does not), or that such a value
  was computed from (the pointer a tensor was built on, which another
  tensor may be built on too, or the tensor a view was taken of).

Names compare through the device function's rename groups: a value a loop
carries out is written under its loop-output name, which only the final
rename folds into the name the loads read.

``fuse_two_pass_loads`` caches repeated loads across sibling sweeps too, but
it runs before the lane split and the forced unroll create these passes, and
its caches are register or shared-memory fragments indexed per iteration,
sized and placed by the reduction-reload policy (``cute_reduction_reloads``),
with the later sweeps' values and barriers derived from that placement.
Between trace-time loops over the same lanes the reuse needs none of that: a
trace-time list holds the earlier pass's SSA values and the later pass reads
them by its constant lane index.  The proof above follows the same rules as
that pass (rename groups, alias-blind memory effects, proven pure address
producers).

``sink_vector_loops._reuse_sibling_packets`` serves a later sweep from an
earlier sweep's packets as well, but only inside the vector-loop sink's
output for a persistent row: the lane loops it builds from one sunk vector
loop, matched by lane variable and trip count, their vector packet loads at
the loops' top level, and only tensors no write of the kernel may reach.
The passes here come from the lane split of a looped reduction, after the
forced unroll: loads that sit below the pass loop in nested constexpr loops
(per element where a row ends mid-vector), whose addresses the enclosing
loops' bodies compute, so neither its matching nor its kernel-wide cacheable
set applies; the span proof above lets a tensor the kernel writes after the
later pass keep the reuse too.

All the stashes keep at most ``_MAX_STASHED_REGISTERS`` 32-bit registers
live per thread together, a statement list's inside a lane loop and those
around it alike.  A stash holds its assignment's result: per element, the
widest of the type the load names (a 16-byte packet is four registers) and a
type the right-hand side converts it to.  A plain ``.load()`` names none and
counts as one register, so the stash of an 8-byte one holds twice the
registers it counts.  The pass runs only after the lane unroll
(``passes/pipeline.py``).
The values and the order of every computation are unchanged, so results are
bitwise the same.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from typing import TYPE_CHECKING

from ..ast_extension import statement_from_string
from ..ast_read_writes import ReadWrites
from ._ast_pass_utils import _backward_slice
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import dealias_shared_nodes
from ._ast_pass_utils import exposed_reads
from ._ast_pass_utils import loop_extent
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_written
from ._ast_pass_utils import statement_lists
from .lane_loops.owner_guards import _memory_write_calls
from .packet_access import call_write_roots
from .packet_access import fragment_writes
from .packet_access import match_packet_access
from .smem_footprint import _CUTLASS_TYPE_BYTES
from .unroll_lane_loads import _index_text
from .unroll_lane_loads import _is_load_call

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .packet_access import PacketAccess

_LIST_PREFIX = "_lane_stash_loads_"
# 32-bit registers all the stashes keep live per thread.
_MAX_STASHED_REGISTERS = 64
# Bytes per element of the scalar types a packet load spells.
_ELEMENT_BYTES = {**_CUTLASS_TYPE_BYTES, "cutlass.Boolean": 1}
# Attributes of a CuTe tensor that read its layout, not the memory it points
# to (``x.layout.stride[1]``).
_LAYOUT_ATTRIBUTES = frozenset({"layout", "shape", "stride"})


def load_registers(access: PacketAccess, element_bytes: int | None = None) -> int:
    """32-bit registers a thread holds the value ``access`` loads in: its
    width (one for a scalar) times the bytes of its carrier, else of
    ``element_bytes`` (a plain ``.load()`` spells no type), else four."""
    size = _ELEMENT_BYTES.get(access.carrier or "") or element_bytes or 4
    return math.ceil((access.width or 1) * size / 4)


def _header(node: ast.AST) -> tuple[str, int] | None:
    """``(target, extent)`` of ``for target in cutlass.range_constexpr(extent)``."""
    extent = loop_extent(node, kinds=("constexpr",))
    if extent is None:
        return None
    assert isinstance(node, ast.For) and isinstance(node.target, ast.Name)
    return node.target.id, extent


def _ends_iterations_early(loop: ast.For) -> bool:
    """Whether a ``break``, ``continue`` or ``return`` sits anywhere under
    ``loop``: the stash is filled by appends and read by lane number, so an
    element skipped in any nested loop would shift the later reads."""
    return any(
        isinstance(node, (ast.Break, ast.Continue, ast.Return))
        for node in ast.walk(loop)
    )


def _is_load_assignment(stmt: ast.AST) -> bool:
    """``NAME = <expression>`` loading memory and otherwise pure."""
    if assigned_name(stmt) is None:
        return False
    assert isinstance(stmt, ast.Assign)
    calls = [node for node in ast.walk(stmt.value) if isinstance(node, ast.Call)]
    return any(_is_load_call(call) for call in calls) and all(
        _is_proven_relocatable_call(call, allow_load=True) for call in calls
    )


def _iterator_bases(nodes: Iterable[ast.AST]) -> set[str]:
    """The names ``nodes`` read a tensor's ``iterator`` of."""
    return {
        node.value.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }


# Methods that update a trace-time list, dict or set in place.
_MUTATING_METHODS = frozenset(
    {
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "reverse",
        "sort",
        "update",
        "add",
        "discard",
        "setdefault",
        "popitem",
    }
)


def _root_name(node: ast.AST) -> str | None:
    """The name ``node`` subscripts or reads an attribute of, at any depth."""
    while isinstance(node, (ast.Subscript, ast.Attribute, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _mutated_in_place(statements: Iterable[ast.AST]) -> set[str]:
    """The names whose objects ``statements`` may change in place: the
    receiver of a mutating method (``offs.reverse()``, ``rows[0].append(v)``),
    the object of a subscript or attribute store at any depth
    (``offs[0] = v``, ``st.off = v``, ``v[None, 0][1] = v``), the target of
    an augmented assignment (``alias += [v]`` extends the list it names) and
    the register fragments a call stores into (``fragment_writes``: the
    fragment grouped reduce's ``results``)."""
    mutated: set[str] = set()
    for stmt in statements:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                mutated |= fragment_writes(node)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _MUTATING_METHODS
            ):
                receiver = node.func.value
            elif isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(
                node.ctx, ast.Store
            ):
                receiver = node.value
            elif isinstance(node, ast.AugAssign):
                receiver = node.target
            else:
                continue
            if (name := _root_name(receiver)) is not None:
                mutated.add(name)
    return mutated


def _container_aliases(
    body: list[ast.stmt], names: _Names
) -> dict[str, frozenset[str]]:
    """Each name ``body`` binds to another's object or holds another in,
    canonical (``names``): the names that may share a mutable object with it
    (``alias = offs``, ``rows = [offs]``, ``row = rows[0]``,
    ``rows.append(offs)``, transitively)."""
    parent: dict[str, str] = {}

    def find(name: str) -> str:
        while parent.get(name, name) != name:
            name = parent[name]
        return name

    def held(value: ast.AST) -> set[str]:
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return {name for element in value.elts for name in held(element)}
        name = _root_name(value)
        return set(names([name])) if name is not None else set()

    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Assign):
            pairs = [(target, node.value) for target in node.targets]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("append", "extend", "insert")
        ):
            pairs = [(node.func.value, arg) for arg in node.args]
        else:
            continue
        for target, value in pairs:
            for bound in held(target):
                for source in held(value):
                    parent[find(bound)] = find(source)
    groups: dict[str, set[str]] = {}
    for name in parent:
        groups.setdefault(find(name), set()).add(name)
    return {
        name: frozenset(groups[find(name)] | {find(name)})
        for root in groups
        for name in groups[root] | {root}
    }


class _Names:
    """Canonical names through the device function's rename groups."""

    def __init__(self, renames: Mapping[str, str]) -> None:
        self.renames = renames

    def __call__(self, names: Iterable[str]) -> frozenset[str]:
        return frozenset(self.renames.get(name, name) for name in names)

    def written(self, statements: Iterable[ast.AST]) -> frozenset[str]:
        return self(name for stmt in statements for name in names_written(stmt))

    def carried(self, loop: ast.For) -> frozenset[str]:
        """The names one iteration of ``loop`` reads before assigning them
        and also writes: the values it carries into the next iteration."""
        return exposed_reads(loop.body, renames=self.renames) & self.written(loop.body)


@dataclasses.dataclass(frozen=True)
class _Slice:
    """What a load's address and predicate depend on along its path."""

    # The defining statements on the path, as text.
    texts: tuple[str, ...]
    # Names the slice and the load read before assigning them.
    inputs: frozenset[str]
    # Names the slice and the load read, and the names the slice writes.
    names: frozenset[str]
    # Names read through a tensor's ``iterator``: the tensors read.
    tensors: frozenset[str]


@dataclasses.dataclass
class _Site:
    """A load statement of a constexpr loop: its path and the statements before it."""

    # The constexpr loops on its path, outermost (the pass itself) first.
    loops: tuple[ast.For, ...]
    # The statements that run before it on its path, outermost body first.
    prefix: list[ast.stmt]
    stmt: ast.Assign
    body: list[ast.stmt]

    @property
    def path(self) -> tuple[tuple[str, int], ...]:
        return tuple(header for loop in self.loops if (header := _header(loop)))

    @property
    def text(self) -> str:
        return ast.unparse(self.stmt)

    def slice(
        self, names: _Names, aliases: Mapping[str, frozenset[str]]
    ) -> _Slice | None:
        """The slice of the load, or None when it cannot be proven to compute
        the same address in another pass: two path loops share a target, a
        path variable is assigned in its loop, a loop before the load rebinds
        a name it reads, a call in it is not proven pure, the pass changes a
        container it reads in place (``aliases``: the names that may share an
        object), or a loop on the path carries one of its names or a path
        variable into the next iteration (a path variable read before its
        loop starts holds what an earlier loop left: the loop around both
        carries it)."""
        path_vars = [var for var, _extent in self.path]
        # The stash is indexed by every path variable.
        if len(set(path_vars)) != len(path_vars):
            return None
        loop_vars = names(path_vars)
        # The stash is indexed by the path variables, so none may be assigned
        # in its loop, whatever the load reads: before the load the index
        # changes, after it an inner loop's next iterations read the new one.
        for loop in self.loops:
            assert isinstance(loop.target, ast.Name)
            if names([loop.target.id]) & names.written(loop.body):
                return None
        indices, written = _backward_slice(
            [*self.prefix], names_read(self.stmt) - loop_vars, names.renames
        )
        if names(written) & loop_vars:
            return None
        statements = [self.prefix[index] for index in indices]
        nodes: list[ast.AST] = [*statements, self.stmt]
        if not all(
            _is_proven_relocatable_call(call, allow_load=True)
            for node in nodes
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        ):
            return None
        read = names(name for node in nodes for name in names_read(node))
        # A loop target keeps the loop's last value, which the backward slice
        # (name stores only) does not see.
        targets = names(
            name
            for stmt in self.prefix
            for node in ast.walk(stmt)
            if isinstance(node, ast.For)
            for name in names_written(node.target)
        )
        if (read - loop_vars) & targets:
            return None
        slice_names = (read | names(written)) - loop_vars
        if any((slice_names | loop_vars) & names.carried(loop) for loop in self.loops):
            return None
        # An append or subscript store anywhere in the pass changes what a
        # container the slice reads holds, which the slice's text does not
        # show.
        changed = {
            other
            for name in names(_mutated_in_place([self.loops[0]]))
            for other in aliases.get(name, {name})
        }
        if names(changed) & slice_names:
            return None
        return _Slice(
            tuple(map(ast.unparse, statements)),
            exposed_reads(nodes, renames=names.renames) - loop_vars,
            slice_names,
            frozenset(_iterator_bases(nodes)),
        )

    def registers(self) -> int:
        """32-bit registers the stash of this load keeps live per thread.  It
        holds the assignment's result: per element, the width of its load
        times the widest of the type the load names (four bytes for a plain
        ``.load()``) and a type the right-hand side converts it to
        (``cutlass.Float64(x.load())`` takes two registers; a ``bitcast``
        keeps the bits it reinterprets)."""
        value = self.stmt.value
        loads = [
            access
            for node in ast.walk(value)
            if (access := match_packet_access(node)) is not None and access.is_load
        ]
        skipped = {
            id(node)
            for root in ast.walk(value)
            if isinstance(root, ast.Call)
            and (
                (isinstance(root.func, ast.Attribute) and root.func.attr == "bitcast")
                or any(access.call is root for access in loads)
            )
            for node in ast.walk(root)
        }
        converted = max(
            (
                _ELEMENT_BYTES.get(ast.unparse(node), 0)
                for node in ast.walk(value)
                if isinstance(node, ast.Attribute) and id(node) not in skipped
            ),
            default=0,
        )
        per_element = max(
            (
                max(
                    load_registers(access),
                    math.ceil((access.width or 1) * converted / 4),
                )
                for access in loads
            ),
            default=1,
        )
        elements = math.prod(extent for _var, extent in self.path)
        return elements * per_element


def _load_sites(loop: ast.For) -> list[_Site]:
    assert _header(loop) is not None
    sites: list[_Site] = []

    def walk(
        body: list[ast.stmt], loops: tuple[ast.For, ...], prefix: list[ast.stmt]
    ) -> None:
        for index, stmt in enumerate(body):
            if _header(stmt) is not None:
                assert isinstance(stmt, ast.For)
                walk(stmt.body, (*loops, stmt), [*prefix, *body[:index]])
            elif _is_load_assignment(stmt):
                assert isinstance(stmt, ast.Assign)
                sites.append(_Site(loops, [*prefix, *body[:index]], stmt, body))

    walk(loop.body, (loop,), [])
    return sites


def _register_buffers(statements: Sequence[ast.stmt]) -> set[str]:
    """Names the statements bind to a thread's register tensor
    (``cute.make_rmem_tensor``)."""
    return {
        name
        for stmt in statements
        for node in ast.walk(stmt)
        if (name := assigned_name(node)) is not None
        and isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
    }


def _pointers(
    body: list[ast.stmt], kernel_tensors: Collection[str], names: _Names
) -> frozenset[str]:
    """The kernel tensors, the tensors ``body`` reads through ``.iterator``
    (shared-memory and register tensors too) and the names ``body`` binds to
    a value that may point into one of them, canonical: a value that names
    one of them outside the loads it makes (a loaded value points nowhere).
    The other way round, the names a binding of one of them computes it from
    may point into it too, as another tensor built on the same pointer may:
    the names its value reads as values (the ``p`` of
    ``cute.make_tensor(p, layout)``, the ``t`` of a view ``t[None, 0]``; not
    the integer offset of ``x.iterator + index``), and
    those ``body`` binds that it reads as the object of an attribute (the
    ``p`` of ``p.align(16)``; the module ``cute`` of ``cute.make_tensor`` is
    not bound).

    Every binding counts: assignments to names, tuples and subscripts of a
    name, augmented and annotated assignments, ``:=``, loop and
    comprehension targets, ``with ... as`` and an ``append``, ``extend`` or
    ``insert`` into a name."""

    def outside_loads(
        node: ast.AST, *, plain: bool = False, offsets: bool = True
    ) -> set[str]:
        """The names ``node`` reads outside its loads and layouts (the ``x``
        of ``x.layout.stride[1]`` is not read as a pointer); with ``plain``,
        only those it reads as values, not as the object of an attribute (the
        ``cute`` of ``cute.make_tensor``); without ``offsets``, not the
        integer offset added to a pointer."""
        found: set[str] = set()
        pending = [node]
        while pending:
            child = pending.pop()
            if isinstance(child, ast.Call) and _is_load_call(child):
                continue
            if isinstance(child, ast.Attribute) and child.attr in _LAYOUT_ATTRIBUTES:
                continue
            if (
                not offsets
                and isinstance(child, ast.BinOp)
                and isinstance(child.op, (ast.Add, ast.Sub))
            ):
                # ``x.iterator + index``: the pointer is the side reading a
                # tensor's iterator; the integer offset points nowhere.
                sides = [
                    side
                    for side in (child.left, child.right)
                    if _iterator_bases([side])
                ]
                if len(sides) == 1:
                    pending.append(sides[0])
                    continue
            if isinstance(child, ast.Name):
                found.add(child.id)
            if plain and isinstance(child, ast.Attribute):
                child = child.value
                while isinstance(child, ast.Attribute):
                    child = child.value
                if isinstance(child, ast.Name):
                    continue
            pending.extend(ast.iter_child_nodes(child))
        return found

    def bound(target: ast.AST) -> set[str]:
        """The names a store into ``target`` binds or writes into."""
        if isinstance(target, (ast.Tuple, ast.List)):
            return {name for element in target.elts for name in bound(element)}
        while isinstance(target, (ast.Subscript, ast.Attribute, ast.Starred)):
            target = target.value
        return {target.id} if isinstance(target, ast.Name) else set()

    tree = ast.Module(body=body, type_ignores=[])
    pointers = set(names(kernel_tensors)) | names(_iterator_bases([tree]))
    bindings: list[
        tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]
    ] = []
    for node in ast.walk(tree):
        pairs: list[tuple[ast.AST, ast.AST]] = []
        if isinstance(node, ast.Assign):
            pairs = [(target, node.value) for target in node.targets]
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            if node.value is not None:
                pairs = [(node.target, node.value)]
        elif isinstance(node, (ast.For, ast.comprehension)):
            pairs = [(node.target, node.iter)]
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            pairs = [(node.optional_vars, node.context_expr)]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("append", "extend", "insert")
        ):
            pairs = [(node.func.value, arg) for arg in [*node.args, *node.keywords]]
        bindings.extend(
            (
                names(bound(target)),
                names(outside_loads(value)),
                names(outside_loads(value, offsets=False)),
                names(outside_loads(value, plain=True, offsets=False)),
            )
            for target, value in pairs
        )
    local = frozenset().union(*(binding[0] for binding in bindings))
    flows = [
        (targets, reads, (based & local) | plain)
        for targets, reads, based, plain in bindings
    ]
    changed = True
    while changed:
        changed = False
        for targets, reads, sources in flows:
            if reads & pointers and not targets <= pointers:
                pointers |= targets
                changed = True
            if targets & pointers and not sources <= pointers:
                pointers |= sources
                changed = True
    return frozenset(pointers)


def _may_write(
    statements: Sequence[ast.stmt],
    buffers: Collection[str],
    pointers: Collection[str],
    names: _Names,
) -> bool:
    """Whether running ``statements`` may write memory a load reads.

    Any store, atomic or packet flush counts (two kernel tensors may share
    memory), as does a subscript store into anything but the register tensors
    ``buffers`` (at any depth: ``v[None, 0][1] = acc`` writes ``v``), a call
    that may write memory it does not name
    (``call_write_roots``: impure inline asm, an unknown function), whatever
    it is handed -- an address it gets may come from memory (a loaded
    value), which ``_pointers`` does not follow -- and a call that is
    neither a load nor proven pure and is handed one of ``pointers``
    (``_pointers``: every tensor a load reads, and whatever may alias one).
    """
    for stmt in statements:
        if _memory_write_calls(stmt):
            return True
        if set(ReadWrites.from_ast(stmt).inplace_writes) - set(buffers):
            return True
        # ``inplace_writes`` sees ``NAME[...] = v`` only.
        if any(
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and _root_name(node.value) not in buffers
            for node in ast.walk(stmt)
        ):
            return True
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and (
                call_write_roots(node, lambda address: None) is None
                or (
                    not _is_proven_relocatable_call(node, allow_load=True)
                    and names(names_read(node)) & set(pointers)
                )
            ):
                return True
    return False


def stash_sibling_lane_loads(
    body: list[ast.stmt],
    new_var: Callable[[str], str],
    *,
    rename_groups: Mapping[str, str],
    kernel_tensors: Collection[str],
) -> list[ast.stmt]:
    """Rewrite every statement list reachable from ``body`` (in place) and
    return it.  ``rename_groups`` maps a name to the name the final rename
    gives it; ``kernel_tensors`` are the kernel's tensor arguments."""

    def has_siblings(statements: list[ast.stmt]) -> bool:
        headers = [
            header
            for stmt in statements
            if isinstance(stmt, ast.For) and (header := _header(stmt)) is not None
        ]
        return len(set(headers)) < len(headers)

    if not any(has_siblings(statements) for statements in statement_lists(body)):
        return body
    # The lane split may place one statement object in two passes; each
    # position gets its own before the rewrites below touch one of them.
    dealias_shared_nodes(body)
    names = _Names(rename_groups)
    pointers = _pointers(body, kernel_tensors, names)
    aliases = _container_aliases(body, names)
    # One budget for every list: a lane loop's inner stashes and those
    # around it are live together.
    budget = [_MAX_STASHED_REGISTERS]
    for statements in statement_lists(body):
        _stash_list(statements, new_var, names, pointers, aliases, budget)
    return body


def _stash_list(
    statements: list[ast.stmt],
    new_var: Callable[[str], str],
    names: _Names,
    pointers: frozenset[str],
    aliases: Mapping[str, frozenset[str]],
    budget: list[int],
) -> None:
    loops = [
        (index, stmt)
        for index, stmt in enumerate(statements)
        if isinstance(stmt, ast.For)
        and _header(stmt) is not None
        and not _ends_iterations_early(stmt)
    ]
    sites = {index: _load_sites(loop) for index, loop in loops}
    slices: dict[int, _Slice | None] = {}

    def slice_of(site: _Site) -> _Slice | None:
        key = id(site.stmt)
        if key not in slices:
            slices[key] = site.slice(names, aliases)
        return slices[key]

    # id(earlier load) -> (stash name, earlier site, earlier loop index)
    stashes: dict[int, tuple[str, _Site, int]] = {}
    reuses: list[tuple[_Site, str]] = []
    rewritten: set[int] = set()
    for position, (later_index, later) in enumerate(loops):
        if not sites[later_index]:
            continue
        for earlier_index, earlier in loops[:position]:
            if _header(earlier) != _header(later):
                continue
            earlier_sites: dict[tuple[object, str], list[_Site]] = {}
            for site in sites[earlier_index]:
                earlier_sites.setdefault((site.path, site.text), []).append(site)
            candidates = [
                (site, match)
                for site in sites[later_index]
                for match in earlier_sites.get((site.path, site.text), [])
                # A load already reading a stash is neither rewritten again
                # nor a stash of its own.
                if id(site.stmt) not in rewritten and id(match.stmt) not in rewritten
            ]
            if not candidates:
                continue
            span = statements[earlier_index : later_index + 1]
            span_writes = names.written(span) | names(
                other
                for name in names(_mutated_in_place(span))
                for other in aliases.get(name, {name})
            )
            # Between an element's load in one pass and its load in a later
            # one run the first pass's remaining iterations and the later
            # pass's earlier ones: both loops count whole, headers included.
            effects = [earlier, *statements[earlier_index + 1 : later_index], later]
            buffers = _register_buffers(span)
            for site, match in candidates:
                if id(site.stmt) in rewritten:
                    # An earlier candidate of this load matched.
                    continue
                mine, theirs = slice_of(site), slice_of(match)
                if (
                    mine is None
                    or theirs is None
                    or mine.texts != theirs.texts
                    or (mine.inputs | theirs.inputs) & span_writes
                    or not mine.tensors
                    or _may_write(effects, buffers - mine.tensors, pointers, names)
                ):
                    continue
                key = id(match.stmt)
                if key not in stashes:
                    cost = match.registers()
                    if cost > budget[0]:
                        continue
                    budget[0] -= cost
                    stashes[key] = (new_var(_LIST_PREFIX), match, earlier_index)
                reuses.append((site, stashes[key][0]))
                rewritten.add(id(site.stmt))
    if not reuses:
        return
    for site, name in reuses:
        target = assigned_name(site.stmt)
        index = site.body.index(site.stmt)
        site.body[index] = statement_from_string(
            f"{target} = {name}[{_index_text(list(site.path))}]"
        )
    inits: dict[int, list[str]] = {}
    for name, site, loop_index in stashes.values():
        target = assigned_name(site.stmt)
        index = site.body.index(site.stmt)
        site.body.insert(index + 1, statement_from_string(f"{name}.append({target})"))
        inits.setdefault(loop_index, []).append(name)
    for loop_index in sorted(inits, reverse=True):
        for name in reversed(inits[loop_index]):
            statements.insert(loop_index, statement_from_string(f"{name} = []"))
