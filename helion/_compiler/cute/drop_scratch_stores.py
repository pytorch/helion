"""Drop the stores of kernel-local scratch tensors that no load reads (CuTe).

Role: device-body pass (pre-rename, after ``fuse_two_pass_loads``). A kernel
may write a host workspace in one sweep of a row and read it back in the
next (Inductor's looped reductions do; ports of them keep the workspace).
``scratch_tensors.kernel_local_scratch`` proves that nothing outside the
kernel observes such a tensor.  Once no load of it is left in the body (a
later sweep's reload is served from registers), the stores are dead: this
pass removes them wherever loops, branches and ``with`` blocks nest them,
the lists that only gather their packets and the loops and branches left
empty.  The finished kernel then reads none of the tensor, and
``drop_unused_scratch_arguments`` removes its parameter, its launcher
argument and, through ``GenerateAST.host_dead_code_elimination``, its
allocation.  That goes beyond Inductor, which always writes the workspace;
no result changes.

A scratch tensor any other reference names (a load, a pointer handed to a
call that is not one of its stores, a store whose operands are not proven
pure, anything inside a ``try``, which device code does not have) keeps its
stores; references are counted by occurrence, so a pointer node codegen
spliced into a store and a load keeps the store.  A list whose appends are
not proven pure, or whose name is bound to anything but a fresh ``[]``,
stays, and so does an emptied loop whose target a later statement reads.
Barriers stay: the barrier pass may count on a barrier the load's lowering
placed.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import TYPE_CHECKING

from ..ast_extension import create
from ..ast_read_writes import ReadWrites
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import dealias_shared_nodes
from ._ast_pass_utils import live_in
from ._ast_pass_utils import loop_live_end
from ._ast_pass_utils import names_stored_or_deleted
from ._ast_pass_utils import names_written
from .address_maps import _LOOP_CALLS
from .fuse_two_pass_loads import _tensor_arg_roots
from .packet_access import match_packet_access
from .sweep_values import is_pure_expression

if TYPE_CHECKING:
    from collections.abc import Collection

_STORE_KINDS = frozenset({"store", "flush"})


def _operands_pure(call: ast.Call) -> bool:
    """``call``'s receiver and arguments evaluate without an effect, so that
    dropping the call loses only its own."""
    operands = [*call.args, *(keyword.value for keyword in call.keywords)]
    if isinstance(call.func, ast.Attribute):
        operands.append(call.func.value)
    return all(is_pure_expression(operand) for operand in operands)


def _store_of(stmt: ast.stmt, tensors: set[str]) -> tuple[str, ast.expr] | None:
    """The scratch tensor ``stmt`` stores to and the store's pointer, when
    ``stmt`` is one store call through that tensor alone, with pure
    operands."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return None
    access = match_packet_access(stmt.value)
    if access is None or access.kind not in _STORE_KINDS:
        return None
    roots = _tensor_arg_roots(access.pointer, tensors)
    if roots is None or len(roots) != 1 or not _operands_pure(stmt.value):
        return None
    (root,) = roots
    return root, access.pointer


def _child_lists(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """The statement lists nested in ``stmt`` that the pass looks into (a
    ``try`` is one statement)."""
    if isinstance(stmt, (ast.For, ast.While, ast.If)):
        return [stmt.body, stmt.orelse]
    if isinstance(stmt, ast.With):
        return [stmt.body]
    return []


def _statement_lists(body: list[ast.stmt]) -> list[list[ast.stmt]]:
    lists = [body]
    for stmt in body:
        for inner in _child_lists(stmt):
            lists.extend(_statement_lists(inner))
    return lists


def _own_parts(stmt: ast.stmt) -> list[ast.AST]:
    """What ``stmt`` runs itself: a compound statement's header (a loop's
    target and iterator, a test, the context managers), not its body."""
    if isinstance(stmt, ast.For):
        return [stmt.target, stmt.iter]
    if isinstance(stmt, (ast.While, ast.If)):
        return [stmt.test]
    if isinstance(stmt, ast.With):
        return list(stmt.items)
    return [stmt]


def _own_reads(stmt: ast.stmt) -> set[str]:
    """The names ``stmt`` reads itself (``_own_parts``; an augmented
    assignment reads its target)."""
    return {
        name for part in _own_parts(stmt) for name in ReadWrites.from_ast(part).reads
    }


def _own_writes(stmt: ast.stmt) -> set[str]:
    """The names ``stmt`` binds or deletes itself (``_own_parts``: a loop's
    target, ``with`` variables, a walrus in a test)."""
    return {name for part in _own_parts(stmt) for name in names_stored_or_deleted(part)}


def _store_lists(stmt: ast.stmt) -> set[str]:
    """The names of the value lists a store statement consumes."""
    assert isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
    return {arg.id for arg in stmt.value.args if isinstance(arg, ast.Name)}


def _is_list_init(stmt: ast.stmt, names: Collection[str]) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id in names
        and isinstance(stmt.value, ast.List)
        and not stmt.value.elts
    )


def _is_append(stmt: ast.stmt, names: Collection[str]) -> bool:
    """``stmt`` appends a pure operand to one of the lists ``names``."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "append"
        and isinstance(stmt.value.func.value, ast.Name)
        and stmt.value.func.value.id in names
        and _operands_pure(stmt.value)
    )


def _nested_lists(
    stmt: ast.stmt, live: set[str]
) -> list[tuple[list[ast.stmt], set[str]]]:
    """The ``_child_lists`` of ``stmt``, each with the names read after it
    (``live``: the names read after ``stmt``)."""
    if isinstance(stmt, (ast.For, ast.While)):
        if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
            end = loop_live_end(stmt, live)
        else:
            # A later iteration may read anything the loop reads.
            end = live | set(ReadWrites.from_ast(stmt).reads)
        return [(stmt.body, end), (stmt.orelse, live)]
    return [(inner, live) for inner in _child_lists(stmt)]


def _fill_empty_body(stmt: ast.stmt) -> None:
    """``pass`` into a body the pruning emptied (a loop's control flow and a
    ``with``'s context stay)."""
    assert isinstance(stmt, (ast.For, ast.While, ast.If, ast.With))
    if not stmt.body:
        stmt.body.append(create(ast.Pass))


def _empty(stmt: ast.stmt, live: Collection[str]) -> bool:
    """A loop over a plain iterator or a branch whose body is gone and whose
    header is pure; a loop also needs its target dead after it (``live``:
    the names read after ``stmt``)."""
    if isinstance(stmt, ast.For):
        return (
            not stmt.body
            and not stmt.orelse
            and isinstance(stmt.iter, ast.Call)
            and call_path_str(stmt.iter.func) in _LOOP_CALLS
            and _operands_pure(stmt.iter)
            and names_written(stmt.target).isdisjoint(live)
        )
    if isinstance(stmt, ast.If):
        return not stmt.body and not stmt.orelse and is_pure_expression(stmt.test)
    return False


def drop_scratch_stores(
    body: list[ast.stmt], scratch: Collection[str]
) -> list[ast.stmt]:
    """Remove the stores to each tensor of ``scratch`` (device argument
    names of kernel-local scratch tensors) that nothing else in ``body``
    names."""
    # The pruning rewrites compound statements in place: a statement placed at
    # two positions would take one position's liveness at both.
    dealias_shared_nodes(body)
    tensors = set(scratch)

    def references(node: ast.AST) -> Counter[str]:
        # Occurrences, not nodes: codegen may splice one pointer node into a
        # store and a load (``statement_from_string`` does not copy its
        # placeholders), and ``ast.walk`` yields such a node once for each
        # place it occupies.
        return Counter(
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id in tensors
        )

    # Every statement, nested ones included (the pruning keeps the lists).
    flat = [stmt for statements in _statement_lists(body) for stmt in statements]
    stores: dict[int, str] = {}
    in_stores: Counter[str] = Counter()
    for stmt in flat:
        if (store := _store_of(stmt, tensors)) is not None:
            tensor, pointer = store
            stores[id(stmt)] = tensor
            # Only the tensor's own references: another scratch tensor in
            # the pointer (``ws.iterator + idx[o]``) is a read of it.
            in_stores[tensor] += references(pointer)[tensor]
    # A tensor every reference of which addresses one of its stores (a load
    # in a stored value names its tensor elsewhere).
    everywhere = references(ast.Module(body=body, type_ignores=[]))
    dead = {
        tensor for tensor in stores.values() if everywhere[tensor] == in_stores[tensor]
    }
    if not dead:
        return body
    lists: set[str] = set()
    for stmt in flat:
        if stores.get(id(stmt)) in dead:
            lists |= _store_lists(stmt)
    # A value list is a fresh list: every binding of its name is ``name =
    # []`` (an alias of another list would carry the appends the pruning
    # drops into a list something still reads).
    for stmt in flat:
        if not _is_list_init(stmt, lists):
            lists -= _own_writes(stmt)

    def dropped(stmt: ast.stmt) -> bool:
        return (
            stores.get(id(stmt)) in dead
            or _is_append(stmt, lists)
            or _is_list_init(stmt, lists)
        )

    # A value list read by nothing but its appends, the dropped stores and
    # the appends of other such lists.  A statement's reads are its own, not
    # those of the statements nested in it, and a list kept for a read keeps
    # the lists its appends read: iterate to a fixpoint.
    reads = {id(stmt): _own_reads(stmt) for stmt in flat}
    while True:
        kept_reads = {
            name for stmt in flat if not dropped(stmt) for name in reads[id(stmt)]
        }
        if lists.isdisjoint(kept_reads):
            break
        lists -= kept_reads

    def prune(statements: list[ast.stmt], live: set[str]) -> list[ast.stmt]:
        # Backwards, ``live`` holding the names read after the statement at
        # hand: an emptied loop whose target a later statement reads stays.
        kept: list[ast.stmt] = []
        for stmt in reversed(statements):
            if dropped(stmt):
                continue
            nested = _nested_lists(stmt, live)
            for inner, live_after in nested:
                inner[:] = prune(inner, live_after)
            if nested:
                if _empty(stmt, live):
                    continue
                _fill_empty_body(stmt)
            live = live_in([stmt], live)
            kept.append(stmt)
        kept.reverse()
        return kept

    return prune(body, set())
