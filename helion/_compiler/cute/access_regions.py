"""Symbols of the access regions the CuTe memory-op codegen records.

``helion/language/memory_ops.py`` tags every load and store call it emits
with the elements the tile program's access covers, per tensor dimension, as
``[begin, end)`` bounds over sympy symbols (``HELION_ACCESS_REGIONS_ATTR``);
the barrier pass (``lane_loop_distribution.add_thread_barriers``) decides
from them whether two accesses can meet.  Both sides build the symbols here,
so they agree on what a symbol stands for, and both find the address of a
load or store call the same way (``access_address``).
"""

from __future__ import annotations

import ast
import itertools
from typing import Protocol

import sympy

_instances = itertools.count()
# The DSL's own load and store, whose address is their first argument.
_ARCH_ACCESS_CALLS = ("cute.arch.load", "cute.arch.store")


def access_address(call: ast.Call) -> ast.AST | None:
    """The addressed operand of a load or store call.

    The receiver of a pointer's ``.load()`` / ``.store()``; the first
    argument of ``cute.arch.load`` / ``cute.arch.store`` (a vector packet, a
    cache-hinted scalar) and of any other call (a store flush, a load
    helper).  None for a call without arguments.
    """
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr in ("load", "store")
        and ast.unparse(call.func) not in _ARCH_ACCESS_CALLS
    ):
        return call.func.value
    return call.args[0] if call.args else None


def new_loop_instance() -> int:
    """A serial for a loop state, unique for the process (``LoopInstance``)."""
    return next(_instances)


class LoopInstance(Protocol):
    """A device loop or grid state: its serial names its tiles' begins."""

    region_instance: int


def tile_begin_symbol(block_id: int, loop: LoopInstance | None) -> sympy.Symbol:
    """The begin of the tile over ``block_id`` that ``loop`` iterates.

    The value is one on every thread of the block, so two accesses of one
    loop instance cancel it.  Two device loops over one block id (a
    registered block size reused) have unrelated begins at any point of the
    program, so the symbol names the loop instance (the ``DeviceLoopState``
    or ``DeviceGridState`` active for the block when the access is emitted)
    by its serial, which no later state reuses (an address would be, once
    the state is collected); None stands for a block no loop is active for.
    """
    suffix = "" if loop is None else f"_{loop.region_instance}"
    return sympy.Symbol(f"_helion_tile_begin_{block_id}{suffix}", integer=True)


def block_size_symbol(var: str | None) -> sympy.Expr:
    """The block size named ``var``, a positive integer; 1 without a variable."""
    if var is None:
        return sympy.Integer(1)
    return sympy.Symbol(var, integer=True, positive=True)
