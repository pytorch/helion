from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

from ...runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence

_T = TypeVar("_T")


def _round_robin(groups: Iterable[Sequence[_T]]) -> Iterator[_T]:
    pending = deque(deque(group) for group in groups if group)
    while pending:
        group = pending.popleft()
        yield group.popleft()
        if group:
            pending.append(group)


def _order_off_seeds(seeds: list[Config]) -> list[Config]:
    """Visit structural parents before their vector/cache alternatives.

    Interleave schedule families, then warp counts, then block/loop layouts. Keep
    the compiler-owned seed objects, their multiplicities, and their first seed
    unchanged; ordering within each parent's optional-knob children is stable.
    This only orders the existing pool, not the population size or seed domain.
    """
    families: dict[
        str, dict[int, dict[tuple[tuple[int, ...], str, int], list[Config]]]
    ] = {}
    for seed in seeds:
        family = cast("str", seed.config["cute_chained_mma_schedule"])
        geometry = (
            tuple(seed.block_sizes),
            seed.pid_type,
            cast("int", seed.config.get("cute_chained_pointwise_unroll", 1)),
        )
        families.setdefault(family, {}).setdefault(seed.num_warps, {}).setdefault(
            geometry, []
        ).append(seed)

    parents = list(
        _round_robin(
            [
                list(
                    _round_robin(
                        [list(geometries.values()) for geometries in warps.values()]
                    )
                )
                for warps in families.values()
            ]
        )
    )
    return list(_round_robin(parents))


def order_chained_seed_configs(seeds: list[Config]) -> list[Config]:
    """Insert optional siblings after their lowest-unroll structural parent.

    The legacy off ordering is untouched. Attaching a complete optional family
    to a stable parent (rather than taking turns with a variable-length queue)
    also preserves order when later unroll factors or optional modes are removed.
    Objects and multiplicities are retained; this never changes search budgets.
    """
    keys = (
        "cute_chained_pointwise_read_cache",
        "cute_chained_pointwise_inplace_async",
    )
    unroll = "cute_chained_pointwise_unroll"
    optional = (
        *keys,
        unroll,
        "cute_chained_pointwise_vectorize",
        "cute_chained_auxiliary_cache",
    )

    def flags(seed: Config) -> tuple[bool, bool]:
        return (
            bool(seed.config.get(keys[0], False)),
            bool(seed.config.get(keys[1], False)),
        )

    def anchor(seed: Config) -> Config:
        return Config.from_dict(
            {k: v for k, v in seed.config.items() if k not in optional}
        )

    off = _order_off_seeds([seed for seed in seeds if not any(flags(seed))])
    children: dict[Config, dict[tuple[bool, bool], list[Config]]] = {}
    for seed in seeds:
        mode = flags(seed)
        if any(mode):
            children.setdefault(anchor(seed), {}).setdefault(mode, []).append(seed)
    minimum: dict[Config, int] = {}
    for seed in off:
        key = anchor(seed)
        factor = cast("int", seed.config.get(unroll, 1))
        minimum[key] = min(minimum.get(key, factor), factor)
    result = []
    for seed in off:
        result.append(seed)
        key = anchor(seed)
        if seed.config.get(unroll, 1) == minimum[key] and key in children:
            modes = children.pop(key)
            result.extend(_round_robin([modes[mode] for mode in sorted(modes)]))
    # Preserve unusual caller-supplied optional objects without an off parent.
    for modes in children.values():
        result.extend(_round_robin([modes[mode] for mode in sorted(modes)]))
    return result
