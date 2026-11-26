"""Which computed-fragment modes can run in another root's CTA geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec


def fragment_option_fits_shared_block(
    key: str, value: object, *, spec: ConfigSpec
) -> bool:
    if key == "cute_fp32_dot":
        return value == "simt"
    if key == "cute_fragment_topk_network" and value is True:
        return not spec.cute_fragment_warp_topk_requires_flat
    if key == "cute_fragment_reduction" and value == "warp":
        return not spec.cute_fragment_warp_reduction_requires_flat
    return (
        not key.startswith("cute_fragment_")
        or value in (False, "serial")
        or (key, value) == ("cute_fragment_threads", 128)
    )
