"""The Metal matmul heuristics' seed builder (``mpp_matmul_heuristic``,
``staged_matmul_heuristic``, ``fragment_matmul_heuristic``): configs that
set a few blocks of the default config, and keys over it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from ...autotuner.config_spec import ConfigSpec


def block_seeds(
    spec: ConfigSpec,
    tiles: Iterable[tuple[Mapping[int, int], Mapping[str, object]]],
) -> list[Config]:
    """A seed per ``(blocks, keys)`` of ``tiles``, in order: the default
    config with the sizes ``blocks`` maps block ids to in place and ``keys``
    set over it.  A tile with a size outside its block's search range is
    left out, and so is a repeat."""
    default = spec.default_config()
    seeds: list[Config] = []
    for blocks, keys in tiles:
        fragments = {
            b: spec.block_sizes.block_id_lookup(b)._fragment(spec) for b in blocks
        }
        if not all(
            fragments[b].low <= size <= fragments[b].high for b, size in blocks.items()
        ):
            continue
        block_sizes = [
            blocks.get(item.block_id, size)
            for item, size in zip(spec.block_sizes, default.block_sizes, strict=True)
        ]
        seed = Config.from_dict({**default.config, "block_sizes": block_sizes, **keys})
        if seed not in seeds:
            seeds.append(seed)
    return seeds
