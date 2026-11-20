"""Kernel configs for the Inkling megakernel variants built at serving time: a saved config's scalar choices with
tensor-descriptor indexing on the named tensors' memory ops and warp specialization of the roots whose dots read
them (the variants add memory ops, so a saved per-op indexing list does not carry over)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pretuned_kernels.megakernels._ring_config import ring_config_fields

import helion

if TYPE_CHECKING:
    from collections.abc import Callable

# The served scalar choices (vllm_helion_verify.py's defaults): "attn" for the attention kernels, "moe_nvfp4" /
# "moe_mxfp8" for the MoE kernels and the layer kernels of those layers.
CONFIGS = {
    name: {
        "block_sizes": [],
        "num_warps": num_warps,
        "num_stages": 3,
        "pid_type": "persistent_blocked",
        "cross_loop_pipeline": "dynamic",
        "num_sm_multiplier": 1,
    }
    for name, num_warps in (("attn", 8), ("moe_nvfp4", 4), ("moe_mxfp8", 4))
}
# Claim groups: a slot's quantize tasks run a wave of claims after its W13 tasks, inside the weight stream.
for name in ("moe_nvfp4", "moe_mxfp8"):
    CONFIGS[name]["cross_loop_claim_lag"] = 1


def variant_kernel(
    kern_fn: Callable[..., object],
    args: tuple[object, ...],
    saved_config: str,
    tma: str,
) -> helion.Kernel:
    """helion.kernel for kern_fn with saved_config's choices (a CONFIGS name or a JSON file) and, for these args,
    tensor descriptors for every load and store of the tensors named in tma (comma-separated) and warp
    specialization of the roots whose dots read them."""
    base = (
        dict(CONFIGS[saved_config])
        if saved_config in CONFIGS
        else json.load(open(saved_config))
    )
    for key in ("indexing", "range_warp_specializes", "load_eviction_policies"):
        base.pop(key, None)
    names = frozenset(tma.split(","))
    bound = helion.kernel(
        kern_fn, config=helion.Config(**base), static_shapes=True, backend="gluon"
    ).bind(args)
    base.update(ring_config_fields(bound, loads=names, stores=names))
    return helion.kernel(
        kern_fn, config=helion.Config(**base), static_shapes=True, backend="gluon"
    )
