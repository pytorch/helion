"""Lowerings helion.inductor puts in torch._inductor.lowering.user_lowerings.

Each is inert unless cuda_backend is "helion" on a CUDA device: it then hands
the op to the entry it replaced (a user's lowering, or none, which leaves the
op to Inductor's own), as it does wherever it declines."""

from __future__ import annotations

from .reduction_order import register_reduction_order_lowerings
from .transposed_pads import register_transposed_pads_lowering


def register_lowerings() -> None:
    register_reduction_order_lowerings()
    register_transposed_pads_lowering()
