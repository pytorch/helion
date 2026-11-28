"""``cute_fragment_reduction``: the modes and the reductions the warp lowering supports.

The knob's roots and coverage are the ``REDUCTION`` row of
``cute.autotune.fragment_knobs``; the emitter and the root projection share
this eligibility predicate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ....inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from torch.fx import Node

    from .....autotuner.config_spec import ConfigSpec

KEY = "cute_fragment_reduction"
MODES = ("serial", "warp", "cta", "cta_unroll")


def reduction_modes(spec: ConfigSpec) -> tuple[str, ...]:
    if spec.cute_fragment_reduction_shared or (
        spec.cute_fragment_cta_requires_flat and "flat" not in spec.allowed_pid_types
    ):
        return MODES[:2]
    return MODES


def fragment_warp_reduction_supported(node: Node) -> bool:
    """Match the scalar combines and accumulator types supported by the emitter."""
    lowering = node.meta.get("lowering")
    value = node.meta.get("val")
    return (
        isinstance(lowering, ReductionLowering)
        and lowering.reduction_type in ("sum", "min", "max", "prod")
        and isinstance(value, torch.Tensor)
        and value.dtype
        in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        )
    )
