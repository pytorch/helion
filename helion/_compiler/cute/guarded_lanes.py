"""Run a lane's loads and arithmetic under the guard of the stores they feed.

Role: kernel-definition pass (``cute_guarded_lanes``).

A tile that does not divide its extent, or whose block launches more threads
than the extent has elements, guards its stores with the tile masks, and
everything ahead of the store runs on every thread::

    P_k = cute.arch.load(ADDR_k if C_k else ANCHOR, <packet>)   # per load
    vals = []
    for v in cutlass.range_constexpr(V):
        l_k = <lane v of P_k> if M_k else <zero>
        ...lane arithmetic...
        vals.append(...)
    if S:
        _cute_store_u16_vec(ptr, vals)

(the scalar form is ``l_k = ptr_k.load() if M_k else <zero>`` ... ``if S:
ptr.store(value)`` in a lane loop's body).  A thread ``S`` turns off loads
every packet from its anchor, selects every lane and computes every lane,
for a value nothing observes.  The pass moves the statements ahead of the
guarded stores into their ``if S:`` and drops from the selects and packet
predicates inside what ``S`` implies: a thread the tile masks turn off skips
the body, and where the store's masks are the load's the lanes are plain
reads of the packet, which lets ptxas fold widened half-precision lanes into
packed half2 instructions.

What it proves, per statement list (it declines the region otherwise, and
leaves the code as it was):

- The stores of the region are the ``if`` statements that end it, each
  holding only plain stores or packet flushes under a pure test (the run
  moves behind the test's evaluation, which may read memory and may do
  nothing else).  Several of them are one region when their guards are
  one expression (compared structurally, see below) that reads no memory:
  one that does is evaluated again after the stores ahead of it.  Stores
  under different guards are regions of their own, and a value that
  reaches a store under another guard keeps the region where it is (next
  item).
- The region is the run of statements directly ahead of those stores that
  are assignments of proven pure expressions (``_is_proven_relocatable_
  assignment``: loads, casts and arithmetic, no writes, atomics, barriers or
  unknown calls), appends to a compile-time list, and constexpr loops,
  over pure bounds, of such statements.  The run stops at the last
  statement that assigns a name whose value could be observed when ``S``
  is false: a name read outside the
  region (the guard itself and every other statement of the kernel count),
  one that is also assigned outside the region, and one the region reads
  before it assigns it (a value carried in from an earlier iteration).  An
  append assigns its list, so a list created ahead of the run, or flushed
  or read outside the region, keeps its appends where they are.  So the
  names of the region are temporaries of one execution of it, and skipping
  the region changes nothing a thread turned off could tell.
- A definition inside the region's constexpr loop that the guard reads (the
  lane mask of a vectorized block, evaluated once per packet) moves ahead of
  the region when it is the name's only assignment, pure, and reads nothing
  the region assigns (a mask computed from the loop's index stays, and the
  loop with it).
- A conjunct of a select or of a packet predicate inside the region is
  dropped when the guard implies it by value (a predicated packet of the
  lane-packet emitter whose whole predicate goes becomes the unpredicated
  load the emitter recorded for it; any other keeps its predicate): it is
  structurally one of the guard's conjuncts, or a name whose single
  definition expands to such conjuncts (and the other way around).  Equal
  structure is equal value here: only conjuncts that read no memory are
  compared (``_reads_memory``), and no name the guard reads is assigned
  inside the region (above), so between the guard's test and the conjunct
  nothing it reads changes.  A conjunct that reads memory (a load, an
  element of a tensor or a fragment), in the guard or in a definition, is
  implied by nothing but the name that holds it: a store may run between
  two evaluations of one read (the store into a mask tensor between two
  loads of it; a store of the guard's own body ahead of a select in a
  later one), so two names defined by one load are two values, and a
  select on such a conjunct is kept.  A
  name is expanded through its definition only when that is the name's only
  assignment in the kernel, the assignment reaches the guard (ahead of it
  in program order and inside no branch or loop the guard is outside of),
  and every name the definition reads is bound at most once, by a binding
  that reaches the definition the same way: the definition's conjuncts
  that read no memory then still hold where it is used.  A select whose
  test the guard implies entirely becomes its taken branch.

Bounds are not reasoned about here: a gathered coordinate's ``index < size``
term is dropped at the load site where the tile masks imply it
(``gathered_bounds``), which leaves the tile masks themselves to this pass.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from ...language.memory_ops import load
from ..ast_extension import expr_from_string
from ._ast_pass_utils import _is_proven_relocatable_assignment
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import dealias_shared_nodes
from ._ast_pass_utils import is_constexpr_range
from ._ast_pass_utils import loop_target_names
from ._ast_pass_utils import statement_lists
from .packet_access import match_packet_access
from .persistent_branch_vec import _and_terms

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..device_ir import DeviceIR


def has_guarded_lane_sites(device_ir: DeviceIR) -> bool:
    """Whether the kernel has a load whose mask is more than its tiles' own:
    a coordinate computed from tile indices or gathered from a tensor, or an
    ``extra_mask``.  Structure only, so ``cute_guarded_lanes`` is offered at
    every shape of such a kernel; a kernel of plain tile loads is left to
    its dead-fill packets (``store_only_loads``)."""
    for info in device_ir.graphs:
        for node in info.graph.find_nodes(op="call_function", target=load):
            if len(node.args) > 2 and node.args[2] is not None:
                return True
            index = node.args[1] if len(node.args) > 1 else None
            if isinstance(index, (list, tuple)) and any(
                isinstance(entry, torch.fx.Node)
                and isinstance(entry.meta.get("val"), torch.Tensor)
                for entry in index
            ):
                return True
    return False


def _conjunction(terms: list[ast.expr]) -> ast.expr:
    if len(terms) == 1:
        return terms[0]
    return ast.BoolOp(op=ast.And(), values=terms)


def _guarded_stores(stmt: ast.stmt) -> bool:
    """``if S: <stores>``: ``S`` a pure expression (the run ahead of the
    statement moves behind its evaluation), every statement of the body a
    plain store or a packet flush, no ``else`` beyond ``pass``."""
    if not isinstance(stmt, ast.If) or not stmt.body:
        return False
    if not _pure_expression(stmt.test):
        return False
    if not all(isinstance(other, ast.Pass) for other in stmt.orelse):
        return False
    for inner in stmt.body:
        if not (isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Call)):
            return False
        access = match_packet_access(inner.value)
        if access is None or access.kind not in ("store", "flush"):
            return False
    return True


def _pure_expression(node: ast.AST) -> bool:
    return all(
        _is_proven_relocatable_call(call, allow_load=True)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    )


def _pure_bounds(call: ast.expr) -> bool:
    """Whether the bounds of the range call ``call`` are pure expressions."""
    return isinstance(call, ast.Call) and all(
        _pure_expression(bound)
        for bound in (*call.args, *(keyword.value for keyword in call.keywords))
    )


def _appended_list(stmt: ast.AST) -> str | None:
    """``X`` of the statement ``X.append(...)``: the compile-time list it
    extends."""
    if not isinstance(stmt, ast.Expr):
        return None
    call = stmt.value
    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
    ):
        return call.func.value.id
    return None


def _pure_statement(stmt: ast.stmt) -> bool:
    """A statement the region may hold: a pure assignment to a name, an
    append of a pure value to a compile-time list (which binds the list,
    ``_assigned``), or a constexpr loop, over pure bounds, of such
    statements."""
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Assign):
        return _is_proven_relocatable_assignment(stmt, allow_load=True)
    if isinstance(stmt, ast.Expr):
        call = stmt.value
        return (
            _appended_list(stmt) is not None
            and isinstance(call, ast.Call)
            and not call.keywords
            and all(_pure_expression(argument) for argument in call.args)
        )
    if isinstance(stmt, ast.For):
        return (
            is_constexpr_range(stmt)
            and _pure_bounds(stmt.iter)
            and not stmt.orelse
            and isinstance(stmt.target, ast.Name)
            and all(_pure_statement(inner) for inner in stmt.body)
        )
    return False


def _loop_target_names(loop: ast.For) -> list[ast.Name]:
    """The names a loop binds as its own target, whatever their context: a
    generated loop target may lack its ``Store`` context.  (The shared
    ``loop_target_names`` adds the targets of the loops nested in it.)"""
    return [node for node in ast.walk(loop.target) if isinstance(node, ast.Name)]


def _assigned(stmt: ast.AST) -> Counter[str]:
    """How often each name is bound under ``stmt``: stores, loop targets
    (``loop_target_names``) and appends, which give the list they extend a
    new value as an assignment would."""
    counts: Counter[str] = Counter()
    loop_targets = loop_target_names(stmt)
    targets = {id(name) for name in loop_targets}
    counts.update(name.id for name in loop_targets)
    for node in ast.walk(stmt):
        appended = _appended_list(node)
        if appended is not None:
            counts[appended] += 1
    for node in ast.walk(stmt):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and id(node) not in targets
        ):
            counts[node.id] += 1
    return counts


def _read(stmt: ast.AST) -> Counter[str]:
    """How often each name is read under ``stmt``; a loop's target is bound,
    not read, whatever its context."""
    targets = {id(name) for name in loop_target_names(stmt)}
    return Counter(
        node.id
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and id(node) not in targets
    )


def _reads_memory(node: ast.expr) -> bool:
    """Whether ``node`` may read memory, so that two evaluations of it can
    differ although every name it reads holds the same value in both: a call
    that is not proven pure without a load (a load, a load helper, an
    unknown call), or an element (``c[i]``, ``operator.getitem(c, i)``) of
    a container a store may change.  A named container may be a tensor or a
    fragment; a tuple display, a tensor's ``shape`` or ``stride`` and the
    value of a call (``cute.arch.thread_idx()[0]``; the call is judged as
    one) are immutable."""
    for inner in ast.walk(node):
        container = None
        if isinstance(inner, ast.Subscript):
            container = inner.value
        elif isinstance(inner, ast.Call):
            if not _is_proven_relocatable_call(inner, allow_load=False):
                return True
            if call_path_str(inner.func) == "operator.getitem" and inner.args:
                container = inner.args[0]
        if container is not None and not (
            isinstance(container, (ast.Tuple, ast.Call))
            or isinstance(container, ast.Attribute)
            and container.attr in ("shape", "stride")
        ):
            return True
    return False


def _reads_before_assignment(
    statements: list[ast.stmt], names: set[str], skipped: set[int]
) -> set[str]:
    """The ``names`` a run of pure statements reads before assigning them,
    in execution order (a constexpr loop's body once: a value its second
    trip reads from the first was assigned inside the run), the statements
    ``skipped`` (by ``id``) left out."""
    carried: set[str] = set()
    defined: set[str] = set()

    def visit(stmt: ast.stmt) -> None:
        if id(stmt) in skipped:
            return
        if isinstance(stmt, ast.For):
            carried.update((set(_read(stmt.iter)) & names) - defined)
            assert isinstance(stmt.target, ast.Name)
            defined.add(stmt.target.id)
            for inner in stmt.body:
                visit(inner)
            return
        carried.update((set(_read(stmt)) & names) - defined)
        defined.update(_assigned(stmt))

    for stmt in statements:
        visit(stmt)
    return carried


def _program_order(
    statements: list[ast.stmt], enclosing: tuple[int, ...] = ()
) -> list[tuple[ast.stmt, tuple[int, ...]]]:
    """Every statement under ``statements`` with the ``id`` of each compound
    statement around it (outermost first), each statement ahead of the
    statements nested in it and of the statements after it."""
    ordered: list[tuple[ast.stmt, tuple[int, ...]]] = []
    for stmt in statements:
        ordered.append((stmt, enclosing))
        for field in ("body", "orelse"):
            nested = getattr(stmt, field, None)
            if isinstance(nested, list):
                ordered.extend(_program_order(nested, (*enclosing, id(stmt))))
    return ordered


class _Binding(NamedTuple):
    """Where a name is bound: the program-order position of the statement,
    the statements around it, and the loop when it is the loop's target
    (bound for the statements of the loop only)."""

    position: int
    enclosing: tuple[int, ...]
    loop: int | None

    def reaches(self, position: int, enclosing: tuple[int, ...]) -> bool:
        """Whether the binding has run, and not again since, whenever the
        statement at ``position`` inside ``enclosing`` runs: it is ahead of
        the statement in program order, inside no branch or loop the
        statement is outside of, and a loop target is read inside its loop.
        (Ahead in program order, a binding runs before the statement in
        every iteration of the loops around both, and not again until the
        statement has run.)"""
        return (
            self.position < position
            and enclosing[: len(self.enclosing)] == self.enclosing
            and (self.loop is None or self.loop in enclosing)
        )


class _Kernel:
    """Name facts of the whole kernel body."""

    def __init__(self, body: list[ast.stmt]) -> None:
        module = ast.Module(body=body, type_ignores=[])
        self.assigned = _assigned(module)
        self.read = _read(module)
        # Every statement's program-order position and enclosing statements;
        # the value of the single assignment ``name = value`` of a name; and
        # the binding of every name bound once.
        self.where: dict[int, tuple[int, tuple[int, ...]]] = {}
        self.values: dict[str, ast.expr] = {}
        self.bindings: dict[str, _Binding] = {}
        for position, (stmt, enclosing) in enumerate(_program_order(body)):
            self.where[id(stmt)] = (position, enclosing)
            name = assigned_name(stmt)
            if name is not None and self.assigned[name] == 1:
                assert isinstance(stmt, ast.Assign)
                self.values[name] = stmt.value
            if isinstance(stmt, ast.For):
                for target in _loop_target_names(stmt):
                    self.bindings[target.id] = _Binding(position, enclosing, id(stmt))
            elif not isinstance(getattr(stmt, "body", None), list):
                # (The statements nested in a compound statement have
                # positions of their own.)
                for other in _assigned(stmt):
                    self.bindings[other] = _Binding(position, enclosing, None)

    def definition(self, name: str, reader: ast.stmt) -> ast.expr | None:
        """The value of ``name``'s only assignment when it is what the name
        holds wherever ``reader`` runs and its expression still evaluates
        to that value there, as far as it reads names: the assignment
        reaches ``reader`` (``_Binding.reaches``), and every name it reads
        is bound at most once, by a binding that reaches the assignment.
        (What it reads from memory may have changed: ``_Implied`` takes no
        conjunct that reads memory from it.)"""
        value = self.values.get(name)
        if value is None:
            return None
        binding = self.bindings[name]
        if not binding.reaches(*self.where[id(reader)]):
            return None
        for read in _read(value):
            if self.assigned[read] > 1:
                return None
            if self.assigned[read] == 1 and not self.bindings[read].reaches(
                binding.position, binding.enclosing
            ):
                return None
        return value


class _Implied:
    """The conjuncts the guard of the ``if`` statement ``guard`` implies by
    value (module docstring): its own and those of the definitions its
    names expand through, the ones that read memory left out.  Such a
    conjunct held when it was evaluated, which says nothing about another
    evaluation of the same expression: a store may run between the two."""

    def __init__(self, guard: ast.If, kernel: _Kernel) -> None:
        self.kernel = kernel
        self.guard = guard
        self.terms: set[str] = set()
        pending = _and_terms(guard.test)
        while pending:
            term = pending.pop()
            key = ast.dump(term)
            if key in self.terms or _reads_memory(term):
                continue
            self.terms.add(key)
            if isinstance(term, ast.Name):
                value = kernel.definition(term.id, guard)
                if value is not None:
                    pending.extend(_and_terms(value))

    def holds(self, term: ast.expr, depth: int = 0) -> bool:
        if isinstance(term, ast.Constant) and term.value is True:
            return True
        if ast.dump(term) in self.terms:
            # (No recorded term reads memory, so neither does ``term``.)
            return True
        if isinstance(term, ast.Name) and depth < 8:
            value = self.kernel.definition(term.id, self.guard)
            return value is not None and all(
                self.holds(inner, depth + 1) for inner in _and_terms(value)
            )
        return False

    def residual(self, test: ast.expr) -> list[ast.expr]:
        return [term for term in _and_terms(test) if not self.holds(term)]


class _DropImplied(ast.NodeTransformer):
    """Drop what the guard implies from the selects and the packet
    predicates of the statements now under it.  ``plain_loads`` maps the
    packets the lane-packet emitter loaded predicated to their unpredicated
    load, a template over ``{pointer}``
    (``CuteLaneState.zero_filled_packets``)."""

    def __init__(self, implied: _Implied, plain_loads: Mapping[str, str]) -> None:
        self.implied = implied
        self.plain_loads = plain_loads

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        name = assigned_name(node)
        access = match_packet_access(node.value)
        if (
            name in self.plain_loads
            and access is not None
            and access.kind == "masked_packet"
            and access.predicate is not None
            and not self.implied.residual(access.predicate)
        ):
            # The guard implies the packet's whole predicate: the load the
            # emitter would have placed without one.
            value = expr_from_string(self.plain_loads[name], pointer=access.pointer)
            assert isinstance(value, ast.expr)
            node.value = value
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        rest = self.implied.residual(node.test)
        if not rest:
            return node.body
        node.test = _conjunction(rest)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        access = match_packet_access(node)
        if access is not None and access.kind == "masked_packet":
            assert access.predicate is not None
            rest = self.implied.residual(access.predicate)
            # A predicate the guard implies entirely is left for
            # ``visit_Assign``, which knows the packet's unpredicated load.
            if rest:
                node.args[1] = _conjunction(rest)
        return node


def _guard_definitions(
    run: list[ast.stmt], guard_reads: set[str], kernel: _Kernel
) -> list[tuple[ast.For, ast.stmt]]:
    """The definitions inside the constexpr loops of ``run`` that the guard
    reads and that are candidates to move ahead of the run: the name's only
    assignment, pure without a load.  The caller keeps a candidate in its
    loop when it reads a name the run binds (the loop's index, a value of
    the lane): the run then starts after that loop."""
    found: list[tuple[ast.For, ast.stmt]] = []
    for loop in run:
        if not isinstance(loop, ast.For):
            continue
        for inner in loop.body:
            name = assigned_name(inner)
            if (
                name is not None
                and name in guard_reads
                and kernel.assigned[name] == 1
                and _is_proven_relocatable_assignment(inner, allow_load=False)
            ):
                found.append((loop, inner))
    return found


def _guard_region(
    body: list[ast.stmt],
    statements: list[ast.stmt],
    first: int,
    last: int,
    plain_loads: Mapping[str, str],
) -> int:
    """Move the pure run ahead of the guarded stores ``statements[first :
    last + 1]`` of the kernel ``body`` under their guard.  Returns the
    position of the first statement of ``statements`` that was moved or
    belongs to the stores: the statements ahead of it are as they were."""
    guard = statements[first]
    assert isinstance(guard, ast.If)
    start = first
    while start > 0 and _pure_statement(statements[start - 1]):
        start -= 1
    if start == first:
        return first
    kernel = _Kernel(body)
    guard_reads = set(_read(guard.test))
    store_reads: Counter[str] = Counter()
    for store in statements[first : last + 1]:
        assert isinstance(store, ast.If)
        store_reads.update(_read(ast.Module(body=store.body, type_ignores=[])))
    # The run starts after the last statement binding a name that could be
    # observed with the guard false.  The guard's own definitions inside a
    # constexpr loop of the run (``_guard_definitions``) are bound ahead of
    # the run instead, where they may not read what the run binds.
    hoisted: list[tuple[ast.For, ast.stmt]] = []
    while start < first:
        run = statements[start:first]
        hoisted = _guard_definitions(run, guard_reads, kernel)
        moving = {id(stmt) for _loop, stmt in hoisted}
        bound: Counter[str] = Counter()
        inside = Counter(store_reads)
        for stmt in run:
            if isinstance(stmt, ast.For):
                bound.update(name.id for name in _loop_target_names(stmt))
                inside.update(_read(stmt.iter))
                for inner in stmt.body:
                    if id(inner) not in moving:
                        bound.update(_assigned(inner))
                        inside.update(_read(inner))
            else:
                bound.update(_assigned(stmt))
                inside.update(_read(stmt))
        # (A definition that moves ahead reads outside the region: a name
        # of the run it reads escapes with the others.)
        escaping = {
            name
            for name, count in bound.items()
            if kernel.assigned[name] != count or kernel.read[name] != inside[name]
        }
        escaping |= _reads_before_assignment(run, set(bound), moving)
        if not escaping:
            break
        start = 1 + max(
            index
            for index in range(start, first)
            if set(_assigned(statements[index])) & escaping
        )
    if start == first:
        return first
    for loop, stmt in hoisted:
        loop.body.remove(stmt)
        if not loop.body:
            loop.body.append(ast.Pass())
    moved = statements[start:first]
    stores = [
        inner
        for store in statements[first : last + 1]
        for inner in store.body  # type: ignore[attr-defined]
    ]
    guard.body = [*moved, *stores]
    statements[start : last + 1] = [*(stmt for _loop, stmt in hoisted), guard]
    # What the guard implies, on the kernel as it is now.
    kernel = _Kernel(body)
    implied = _Implied(guard, kernel)
    dropper = _DropImplied(implied, plain_loads)
    guard.body = [dropper.visit(stmt) for stmt in guard.body]
    ast.fix_missing_locations(guard)
    return start


def guard_lanes(body: list[ast.stmt], plain_loads: Mapping[str, str]) -> list[ast.stmt]:
    """``body`` with the pure statements ahead of each group of guarded
    stores moved under the stores' guard (module docstring).  ``plain_loads``
    are the predicated packets of the lane-packet emitter with their
    unpredicated loads (``CuteLaneState.zero_filled_packets``)."""
    dealias_shared_nodes(body)
    for statements in statement_lists(body):
        last = len(statements) - 1
        while last >= 0:
            if not _guarded_stores(statements[last]):
                last -= 1
                continue
            guard = statements[last]
            assert isinstance(guard, ast.If)
            key = ast.dump(guard.test)
            first = last
            # (A test that reads memory is evaluated again after the stores
            # of the statement ahead: such stores are regions of their own.)
            while (
                first > 0
                and _guarded_stores(statements[first - 1])
                and ast.dump(statements[first - 1].test) == key  # type: ignore[attr-defined]
                and not _reads_memory(guard.test)
            ):
                first -= 1
            last = _guard_region(body, statements, first, last, plain_loads) - 1
    return body
