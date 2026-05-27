"""Pair-based jagged dispatch detection for the Pallas backend.

Helion already tracks every (``hl.jagged_tile`` child, ``hl.tile``
parent) pair in :attr:`CompileEnvironment.jagged_tile_parent_ids`.
This module adds a single decision on top: *is the kernel dispatchable
to the per-item DMA-orchestrated template* at
:func:`helion.runtime.pallas_templates.jagged_reduce_pallas`?

The detector stores no new IR. Its only output is a single ``int |
None``: the ``block_id`` of the items axis if the kernel is
dispatchable, ``None`` otherwise. Codegen reads this and walks
Helion's existing IR to derive everything else (jagged children of
that axis, compute slot, flush cast, tensor args, ...).

Dispatch rule
-------------

A kernel is dispatchable iff:

  1. It contains at least one ``hl.jagged_tile``.
  2. The kernel has a **single items axis** — every ``hl.jagged_tile``
     references exactly one outer ``hl.tile``, and they all reference
     the same one.

Number of jagged children is unbounded — a kernel with N
``hl.jagged_tile`` calls all parameterised by the same items axis
dispatches fine (e.g. ``jagged_mean`` has two: ``tile_m`` jagged in
features, ``tile_k`` jagged in seq length, both sharing parent
``tile_b``).

When the rule fails, the detector returns ``None`` and codegen falls
through to the existing Pallas lowering. Kernels with multi-parent
jagged tiles or multiple items axes are out of scope for this
template and would need a separate extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment


def detect_jagged_dispatch(env: CompileEnvironment) -> int | None:
    """Return the items-axis ``block_id`` if dispatchable, else ``None``.

    Reads only ``env.jagged_tile_parent_ids`` — no FX walk, no IR
    mutation. See module docstring for the rule.
    """
    parents_by_child = env.jagged_tile_parent_ids
    if not parents_by_child:
        return None  # no hl.jagged_tile in the kernel

    unique_parents: set[int] = set()
    for parent_ids in parents_by_child.values():
        if len(parent_ids) != 1:
            return None  # jagged_tile has 0 or >1 parents
        unique_parents.add(parent_ids[0])

    if len(unique_parents) != 1:
        return None  # multiple items axes (different parents)

    return next(iter(unique_parents))
