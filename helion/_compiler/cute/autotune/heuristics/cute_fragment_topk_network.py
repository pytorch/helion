"""``cute_fragment_topk_network``: which selections the bounded network may own.

The knob's roots and coverage (and the dependent ``cute_fragment_cta_topk``
group) are the ``TOPK_NETWORK`` / ``CTA_TOPK`` rows of
``cute.autotune.fragment_knobs``; the emitter and the root projection share
this eligibility predicate.
"""

from __future__ import annotations

import torch

KEY = "cute_fragment_topk_network"
CTA_KEY = "cute_fragment_cta_topk"


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
