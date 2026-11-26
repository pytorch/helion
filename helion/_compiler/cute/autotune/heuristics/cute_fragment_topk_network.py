"""``cute_fragment_topk_network``: which selections the bounded network may own.

The knob's roots and coverage (and the dependent ``cute_fragment_cta_topk``
group) are the ``TOPK_NETWORK`` / ``CTA_TOPK`` rows of
``cute.autotune.fragment_knobs``; the emitter and the root projection share
this eligibility predicate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...fragment_launch import uses_shared_barrier

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .....autotuner.config_spec import ConfigSpec

KEY = "cute_fragment_topk_network"
CTA_KEY = "cute_fragment_cta_topk"


def active_network_roots(
    spec: ConfigSpec, config: Mapping[str, object]
) -> frozenset[int]:
    roots = spec.cute_fragment_topk_network_root_ids
    # A projected flat root has its own block; it does not require a barrier
    # with its siblings. Discovery has already checked its input/output storage.
    if config.get(
        "pid_type", spec.allowed_pid_types[0]
    ) != "flat" and not uses_shared_barrier(spec, config):
        return roots - spec.cute_shared_selection_network_root_ids
    return roots


def network_topk_supported(
    dtype: torch.dtype, n: int | None, k: int, *, values_only: bool = False
) -> bool:
    # At most 32 Int64 register keys per lane. Larger/different-typed selections
    # keep the serial implementation. Int64 values need no index packing when
    # the selection's public index output is proved unused.
    return (
        (
            dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32)
            or dtype == torch.int64
            and values_only
        )
        and n is not None
        and 0 < k <= n <= 1024
    )
