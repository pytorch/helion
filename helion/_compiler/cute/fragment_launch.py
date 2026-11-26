"""Which computed-fragment modes can run in another root's CTA geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...autotuner.config_spec import ConfigSpec


def fragment_option_fits_shared_block(
    key: str,
    value: object,
    *,
    spec: ConfigSpec,
    config: Mapping[str, object] | None = None,
) -> bool:
    if key == "cute_fp32_dot":
        return value == "simt"
    if key == "cute_fragment_topk_network" and value is True:
        return not spec.cute_fragment_warp_topk_requires_flat
    if key == "cute_fragment_reduction" and value == "warp":
        return not spec.cute_fragment_warp_reduction_requires_flat
    if key == "cute_fragment_row_carry" and value is True:
        return spec.cute_fragment_row_carry_shared
    if config is not None and value is True:
        if key == "cute_fragment_unrolled_loops":
            from .warp_carry_packet import uses_warp_carry

            return uses_warp_carry(spec, config)
        if key == "cute_fragment_vector_io":
            from .warp_carry_packet import has_carry_packet

            return has_carry_packet(spec, config)
    return (
        not key.startswith("cute_fragment_")
        or value in (False, "serial")
        or (key, value) == ("cute_fragment_threads", 128)
    )
