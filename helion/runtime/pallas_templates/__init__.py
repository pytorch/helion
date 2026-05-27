"""Pallas/TPU kernel templates dispatched by the Helion Pallas backend.

Each template implements the per-item DMA-orchestrated pattern needed
to lower a Helion ``hl.jagged_tile``-based kernel without going through
a gather lowering. See
:mod:`helion.runtime.pallas_templates.jagged_reduce`.
"""

from __future__ import annotations

from .jagged_reduce import jagged_reduce_pallas

__all__ = ["jagged_reduce_pallas"]
