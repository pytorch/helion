"""Pallas/TPU kernel templates dispatched by the Helion Pallas backend.

Each template implements a fixed computation pattern (e.g. jagged reduction)
in JAX/Pallas directly. The compiler's role is detection + dispatch; the
kernel body is hand-written here so we can rely on Mosaic features (e.g.
``pl.BoundedSlice`` + ``pl.ds``) that Helion's source-level lowering does
not yet emit.
"""
from .jagged_reduce import jagged_sum_pallas

__all__ = ["jagged_sum_pallas"]
