"""The pointer a CuTe atomic addresses an element through.

The atomic lowering (``atomic_ops.py``) emits it, and the privatized-atomics
rewrite (``atomic_privatization.py``) matches it at every site it redirects
and emits its flush through it, so the two must spell it alike.  A leaf
module: ``atomic_ops`` imports ``atomic_privatization``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def cute_coord_tuple(exprs: Sequence[object]) -> str:
    """A CuTe tuple of ``exprs``; a single element keeps its comma."""
    return f"({exprs[0]},)" if len(exprs) == 1 else f"({', '.join(map(str, exprs))})"


def cute_coord_pointer_expr(name: str, index_exprs: Sequence[object]) -> str:
    """``(name.iterator + cute.crd2idx(coord, name.layout)).llvm_ptr``: the
    pointer of the element of tensor ``name`` at ``index_exprs``."""
    coord = cute_coord_tuple(index_exprs)
    return f"({name}.iterator + cute.crd2idx({coord}, {name}.layout)).llvm_ptr"
