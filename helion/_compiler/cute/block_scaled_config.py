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
MXFP8_CONFIG_KEYS = frozenset(
    {
        "cute_scaled_fused_scales",
        "cute_scaled_tile_m",
        "cute_scaled_swap_ab",
        "cute_scaled_compact_m",
    }
)
BLOCK_SCALED_CONFIG_KEYS = frozenset(BLOCK_SCALED_CHOICES) | MXFP8_CONFIG_KEYS


def block_scaled_choices(*, mxfp8: bool = False) -> dict[str, tuple[object, ...]]:
    """Return format-specific geometry without extending NVFP4's vocabulary."""
    choices = dict(BLOCK_SCALED_CHOICES)
    if mxfp8:
        choices["cute_scaled_tile_n"] = (256, 224, 128, 64, 32, 16)
        choices["cute_scaled_stages"] = (4, 2, 3, 5, 6, 8, 10, 12)
        choices["cute_scaled_fused_scales"] = (False, True)
        choices["cute_scaled_tile_m"] = (128, 64, 32)
        choices["cute_scaled_swap_ab"] = (False, True)
        choices["cute_scaled_compact_m"] = (False, True)
    return choices


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
    mxfp8: bool = False,
    fused_scales_available: bool = False,
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
    for key, choices in block_scaled_choices(mxfp8=mxfp8).items():
        value = config.setdefault(key, choices[0])
        if not any(
            type(value) is type(choice) and value == choice for choice in choices
        ):
            if fix_invalid:
                config[key] = choices[0]
            else:
                raise InvalidConfig(f"{key} must be one of {choices}")
    if not mxfp8 and MXFP8_CONFIG_KEYS.intersection(config):
        if (
            any(config.get(key) not in (None, False, 128) for key in MXFP8_CONFIG_KEYS)
            and not fix_invalid
        ):
            raise InvalidConfig("fused scales require a proved MXFP8 contraction")
        for key in MXFP8_CONFIG_KEYS:
            config.pop(key, None)
    if config["cute_scaled_mma"] is True and (
        capability is None or capability[0] != 10
    ):
        if fix_invalid:
            config["cute_scaled_mma"] = False
        else:
            raise InvalidConfig("native block scaling requires SM100-family hardware")
    if mxfp8 and config["cute_scaled_tile_k"] == 64:
        if fix_invalid:
            config["cute_scaled_tile_k"] = 128
        else:
            raise InvalidConfig("MXFP8 block scaling requires K tiles of 128 or 256")
    if config.get("cute_scaled_fused_scales") is True and not fused_scales_available:
        if fix_invalid:
            config["cute_scaled_fused_scales"] = False
        else:
            raise InvalidConfig(
                "fused scale loading requires proved native packed INT32 scales"
            )
    if mxfp8 and config.get("cute_scaled_fused_scales") is not True:
        unsupported = (
            config.get("cute_scaled_tile_n") in (16, 32, 224)
            or config.get("cute_scaled_tile_m") != 128
            or config.get("cute_scaled_swap_ab") is True
            or config.get("cute_scaled_compact_m") is True
        )
        if unsupported:
            if not fix_invalid:
                raise InvalidConfig(
                    "small and transposed scaled tiles require fused packed scale loading"
                )
            if config.get("cute_scaled_tile_n") in (16, 32, 224):
                config["cute_scaled_tile_n"] = 128
            config["cute_scaled_tile_m"] = 128
            config["cute_scaled_swap_ab"] = False
            config["cute_scaled_compact_m"] = False
    if mxfp8 and config.get("cute_scaled_tile_m") == 128:
        config["cute_scaled_compact_m"] = False
