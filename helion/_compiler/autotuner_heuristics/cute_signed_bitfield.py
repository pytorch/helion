"""An optional signed-field conversion sibling of an existing vector seed."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...runtime.config import Config

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment


def add_signed_bitfield_seeds(
    env: CompileEnvironment, seeds: list[Config]
) -> list[Config]:
    """Keep the default and old seed order; add one independent arithmetic choice."""
    from ..cute.signed_bitfield import KEY

    if not env.config_spec.cute_signed_bitfield_bf16_available:
        return seeds
    for index, seed in enumerate(seeds):
        widths = seed.config.get("cute_vector_widths", [])
        if not isinstance(widths, list) or not any(
            width in (2, 4, 8) for width in widths
        ):
            continue
        values = deepcopy(seed.config)
        values[KEY] = True
        sibling = Config.from_dict(values)
        env.config_spec.autotuner_heuristics.append(KEY)
        return [*seeds[: index + 1], sibling, *seeds[index + 1 :]]
    return seeds
