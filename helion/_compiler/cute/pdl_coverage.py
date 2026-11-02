"""Does every global memory access of a generated CuTe body sit behind a
``griddepcontrol_wait`` on its own path?

A kernel launched as a programmatic dependent (``cute_pdl``) may start while
the previous kernel in the stream still writes its outputs; every read of
global memory must follow a ``cute.arch.griddepcontrol_wait()`` executed by
the same threads. The tcgen05 lowerings place their waits late (the
TMA-load warp ahead of its first load, the epilogue warps ahead of their
first auxiliary read) so the prologue overlaps the previous kernel; this
module checks, in program order, that nothing reads global memory ahead of
a wait on its path, and ``DeviceFunction._cute_pdl_applies`` puts a wait at
the top of the body when something does.

The analysis is structural: a wait at the top level of a statement list
covers every statement after it; a wait inside a conditional or loop body
covers only the rest of that body (the other branch, and the statements
after the compound statement, may run without it). It is conservative: a
``cute.copy`` with a TMA operand, a ``.load()`` / ``.store()`` through a
``.iterator`` or ``.llvm_ptr`` pointer, ``cp.async``, atomics and the
``ldg``/``ld_global``/``st_global`` helpers count as global accesses;
register / shared-memory copies and fragment loads do not.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_WAIT = "cute.arch.griddepcontrol_wait"
_GLOBAL_FUNC_MARKERS = (
    "cp_async",
    "atomic",
    "ldg",
    "ld_global",
    "st_global",
    "tma_load",
    "tma_store",
    "bulk",
)


def _is_wait(call: ast.Call) -> bool:
    return ast.unparse(call.func) == _WAIT


def _is_global_access(call: ast.Call) -> bool:
    func = ast.unparse(call.func)
    if func == _WAIT or func.endswith("prefetch_descriptor"):
        return False
    if func == "cute.copy":
        # a TMA copy names its atom / tensor / barrier; on-chip copies do not
        for node in ast.walk(call):
            if isinstance(node, ast.Name) and (
                node.id.startswith("tma_") or "_tma_" in node.id
            ):
                return True
        return any(keyword.arg == "tma_bar_ptr" for keyword in call.keywords)
    if isinstance(call.func, ast.Attribute) and call.func.attr in ("load", "store"):
        return any(
            isinstance(node, ast.Attribute) and node.attr in ("iterator", "llvm_ptr")
            for node in ast.walk(call.func.value)
        )
    tail = func.rsplit(".", 1)[-1]
    return any(marker in tail for marker in _GLOBAL_FUNC_MARKERS)


def _calls_outside_bodies(statement: ast.AST) -> list[ast.Call]:
    """Calls of ``statement`` itself (its test / iter / targets / value), not
    of the statement lists nested under it."""
    skip: set[int] = set()
    for field in ("body", "orelse", "finalbody", "handlers"):
        for inner in getattr(statement, field, None) or []:
            skip.update(id(node) for node in ast.walk(inner))
    return [
        node
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and id(node) not in skip
    ]


def waits_cover_global_reads(statements: Sequence[ast.AST]) -> bool:
    """Whether every global access in ``statements`` follows a wait on its path."""

    def covered(stmts: Sequence[ast.AST], waited: bool) -> bool:
        for statement in stmts:
            calls = _calls_outside_bodies(statement)
            if not waited and any(_is_global_access(call) for call in calls):
                return False
            if any(_is_wait(call) for call in calls):
                waited = True
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(statement, field, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                    if not covered(inner, waited):
                        return False
        return True

    return covered(statements, False)
