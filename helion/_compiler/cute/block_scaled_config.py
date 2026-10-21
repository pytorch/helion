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
        "cute_scaled_sf_tma",
        "cute_scaled_sf_inplace",
    }
)
BLOCK_SCALED_CONFIG_KEYS = frozenset(BLOCK_SCALED_CHOICES) | MXFP8_CONFIG_KEYS


def block_scaled_choices(*, mxfp8: bool = False) -> dict[str, tuple[object, ...]]:
    """Return format-specific geometry without extending NVFP4's vocabulary."""
    choices = dict(BLOCK_SCALED_CHOICES)
    if mxfp8:
        choices["cute_scaled_tile_n"] = (256, 224, 128, 64, 32, 16, 8)
        choices["cute_scaled_stages"] = (4, 2, 3, 5, 6, 8, 10, 12)
        choices["cute_scaled_fused_scales"] = (False, True)
        choices["cute_scaled_tile_m"] = (128, 64, 32)
        choices["cute_scaled_swap_ab"] = (False, True)
        choices["cute_scaled_compact_m"] = (False, True)
        choices["cute_scaled_sf_tma"] = (False, True)
        choices["cute_scaled_sf_inplace"] = (False, True)
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


def mxfp8_shared_memory_bytes(
    runtime_m: int,
    logical_m: int,
    bn: int,
    bk: int,
    stages: int,
    cluster_m: int,
    output_element_size: int,
    *,
    compact_m: bool,
    sf_tma: bool,
    sf_inplace: bool = False,
) -> int:
    """Account for native scales, optional raw words and the aliased A tail."""
    rows = logical_m if compact_m and logical_m < 128 and runtime_m > 1 else 128
    scales = (128 + ((bn + 127) // 128) * 128) * bk // 32
    stage_bytes = (rows + bn // cluster_m) * bk + scales * (
        2 if sf_tma and not sf_inplace else 1
    )
    return (
        stage_bytes * stages
        + (128 - rows) * 128
        + 128 * min(bn, 64) * output_element_size
        + 2048
    )


def normalize_block_scaled_config(
    config: dict[str, object],
    *,
    available: bool,
    capability: tuple[int, int] | None,
    fix_invalid: bool,
    mxfp8: bool = False,
    fused_scales_available: bool = False,
    sf_tma_available: bool = False,
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
    if (
        mxfp8
        and config["cute_scaled_tile_n"] == 8
        and config["cute_scaled_cluster_m"] != 1
    ):
        if fix_invalid:
            config["cute_scaled_cluster_m"] = 1
        else:
            raise InvalidConfig("MXFP8 N=8 requires a one-CTA MMA group")
    if config.get("cute_scaled_fused_scales") is True and not fused_scales_available:
        if fix_invalid:
            config["cute_scaled_fused_scales"] = False
        else:
            raise InvalidConfig(
                "fused scale loading requires proved native packed INT32 scales"
            )
    if config.get("cute_scaled_sf_tma") is True and not sf_tma_available:
        if fix_invalid:
            config["cute_scaled_sf_tma"] = False
        else:
            raise InvalidConfig(
                "SF TMA requires proved MN-major packed INT32 scales with legal non-unit tensor-map axes"
            )
    if (
        config.get("cute_scaled_sf_inplace") is True
        and config.get("cute_scaled_sf_tma") is not True
    ):
        if fix_invalid:
            config["cute_scaled_sf_inplace"] = False
        else:
            raise InvalidConfig("in-place scale transpose requires SF TMA")
    if mxfp8 and config.get("cute_scaled_fused_scales") is not True:
        unsupported = (
            config.get("cute_scaled_tile_n") in (8, 16, 32, 224)
            or config.get("cute_scaled_tile_m") != 128
            or config.get("cute_scaled_swap_ab") is True
            or config.get("cute_scaled_compact_m") is True
            or config.get("cute_scaled_sf_tma") is True
            or config.get("cute_scaled_sf_inplace") is True
        )
        if unsupported:
            if not fix_invalid:
                raise InvalidConfig(
                    "small and transposed scaled tiles require fused packed scale loading"
                )
            if config.get("cute_scaled_tile_n") in (8, 16, 32, 224):
                config["cute_scaled_tile_n"] = 128
            config["cute_scaled_tile_m"] = 128
            config["cute_scaled_swap_ab"] = False
            config["cute_scaled_compact_m"] = False
            config["cute_scaled_sf_tma"] = False
            config["cute_scaled_sf_inplace"] = False
    if mxfp8 and config.get("cute_scaled_tile_m") == 128:
        config["cute_scaled_compact_m"] = False
