"""Kernel configs for the Inkling megakernel variants built at serving time: a saved config with tensor-descriptor
indexing on the named tensors' memory ops and warp specialization of the roots whose dots read them, derived for each
kernel (the variants add memory ops, so a saved per-op indexing list does not carry over)."""

from __future__ import annotations

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
# Claim groups: the routed experts' W13 (R1) and W2 (R2) tasks are one claimed stream, W2 after W13's live tasks.
# NVFP4's quantize (Z) tasks wait key by key: a slot's Z tasks run a wave of claims after its W13 tasks, inside the
# weight stream (MXFP8 has no Z root).
for name in ("moe_nvfp4", "moe_mxfp8"):
    CONFIGS[name]["cross_loop_claim_groups"] = True


def variant_kernel(
    kern_fn: Callable[..., object],
    args: tuple[object, ...],
    saved_config: str,
    tma: str,
) -> helion.Kernel:
    """helion.kernel for kern_fn with saved_config (a CONFIGS name or a JSON file) and, for args, tensor descriptors
    for every load and store of the tensors named in tma (comma-separated) and warp specialization of the roots whose
    dots read them, in place of its indexing, range_warp_specializes and load_eviction_policies. Its other keys are
    kept. A JSON file may set only scalar keys, as the CONFIGS entries do: one saved config serves kernels of
    different loop structures (the driver's variables each reach several), so a per-loop list fits none of them
    reliably and is rejected. The derived config is the kernel's only config, so every binding runs it: the driver
    calls one kernel with every layer's arguments, and a layer whose arguments bind anew must not fall back to
    saved_config. The kernel is built with it from the start: a binding keys its tensor-descriptor alignment guards
    on the configs it has when it binds, so a config set after binding would leave the descriptors it enables
    unguarded. Deriving the config takes one bind under saved_config (a separate kernel object); args' binding of
    the returned kernel is cached, so each is traced once."""
    base = dict(
        CONFIGS[saved_config]
        if saved_config in CONFIGS
        else helion.Config.load(saved_config)
    )
    for key in ("indexing", "range_warp_specializes", "load_eviction_policies"):
        base.pop(key, None)
    per_loop = sorted(k for k, v in base.items() if isinstance(v, list) and v)
    if per_loop:
        raise ValueError(
            f"{saved_config}: per-loop keys {per_loop}; the Inkling configs take scalar keys only"
        )
    names = frozenset(tma.split(","))

    def build(config: dict[str, object]) -> helion.Kernel:
        return helion.kernel(
            kern_fn,
            config=helion.Config(**config),
            static_shapes=True,
            backend="gluon",
            ignore_warnings=[
                helion.exc.TensorOperationInWrapper,
                helion.exc.FastMathUnused,
            ],
        )

    base.update(ring_config_fields(build(base).bind(args), loads=names, stores=names))
    kernel = build(base)
    kernel.bind(args)
    return kernel
