"""Configuration vocabulary for native packed block-scaled contractions."""

from __future__ import annotations

from ...exc import InvalidConfig

BLOCK_SCALED_CHOICES: dict[str, tuple[object, ...]] = {
    "cute_scaled_mma": (False, True),
    "cute_scaled_tile_n": (256, 128),
    "cute_scaled_tile_k": (256, 128, 64),
    "cute_scaled_stages": (4, 2, 3, 5),
    "cute_scaled_cluster_m": (2, 1),
    "cute_scaled_persistent": (True, False),
}
BLOCK_SCALED_CONFIG_KEYS = frozenset(BLOCK_SCALED_CHOICES)


def block_scaled_workspace_shape(
    m: int, n: int, k: int, bn: int, bk: int
) -> tuple[int, int, int]:
    """Return physical M/N/K extents for the packed operands and their scales.

    A scale chunk contains 128 rows. SFB's descriptor must expose at least one
    complete N window: a unit N-chunk extent would otherwise disappear from
    CuTe's coordinate tensor even when the copy atom requests two chunks. The
    preparation kernel writes zeros into these extra rows on every invocation.
    """
    return (
        (m + 127) // 128 * 128,
        max((n + 127) // 128 * 128, bn),
        (k + bk - 1) // bk * bk,
    )


def normalize_block_scaled_config(
    config: dict[str, object],
    *,
    available: bool,
    capability: tuple[int, int] | None,
    fix_invalid: bool,
) -> None:
    selected = BLOCK_SCALED_CONFIG_KEYS.intersection(config)
    if not available:
        if selected and not fix_invalid:
            raise InvalidConfig(
                "native block scaling requires a proved packed contraction"
            )
        for key in selected:
            config.pop(key)
        return
    for key, choices in BLOCK_SCALED_CHOICES.items():
        value = config.setdefault(key, choices[0])
        if not any(
            type(value) is type(choice) and value == choice for choice in choices
        ):
            if fix_invalid:
                config[key] = choices[0]
            else:
                raise InvalidConfig(f"{key} must be one of {choices}")
    if config["cute_scaled_mma"] is True and (
        capability is None or capability[0] != 10
    ):
        if fix_invalid:
            config["cute_scaled_mma"] = False
        else:
            raise InvalidConfig("native block scaling requires SM100-family hardware")
