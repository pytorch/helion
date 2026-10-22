"""Configuration vocabulary for native packed block-scaled contractions."""

from __future__ import annotations

from math import gcd

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
        "cute_scaled_epilogue_mode",
    }
)
BLOCK_SCALED_CONFIG_KEYS = frozenset(BLOCK_SCALED_CHOICES) | MXFP8_CONFIG_KEYS
MXFP8_EPILOGUE_MODES = {
    "default": (1, False, False),
    "tma_staged_2": (2, False, True),
    "tma_prefetch_1": (1, True, True),
    "tma_prefetch_2": (2, True, True),
}


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
        choices["cute_scaled_epilogue_mode"] = tuple(MXFP8_EPILOGUE_MODES)
    return choices


def mxfp8_epilogue_parameters(mode: str) -> tuple[int, bool, bool]:
    """Return accumulator banks, complete register preload and TMA output."""
    return MXFP8_EPILOGUE_MODES[mode]


def mxfp8_epilogue_supported(
    mode: str,
    bn: int,
    bk: int,
    logical_m: int,
    *,
    swap_ab: bool,
    fused_scales: bool,
    tma_output_available: bool,
) -> bool:
    """Keep the output descriptor and all accumulator/scale banks legal."""
    if mode == "default":
        return True
    if type(mode) is not str or mode not in MXFP8_EPILOGUE_MODES:
        return False
    accumulator_stages, prefetch, _tma = mxfp8_epilogue_parameters(mode)
    scale_columns = (1 + (bn + 127) // 128) * bk // 32
    return (
        tma_output_available
        and fused_scales
        and logical_m == 128
        and not swap_ab
        and bn >= 32
        and bn % 32 == 0
        and (not prefetch or bn <= 224)
        and bn * accumulator_stages + scale_columns <= 512
    )


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
    epilogue_mode: str = "default",
) -> int:
    """Account for native scales, optional raw words and the aliased A tail."""
    rows = logical_m if compact_m and logical_m < 128 and runtime_m > 1 else 128
    scales = (128 + ((bn + 127) // 128) * 128) * bk // 32
    stage_bytes = (rows + bn // cluster_m) * bk + scales * (
        2 if sf_tma and not sf_inplace else 1
    )
    _accumulator_stages, _prefetch, tma_output = mxfp8_epilogue_parameters(
        epilogue_mode
    )
    output_stages = 2 if tma_output else 1
    epi_n = min(gcd(bn, 64), 32) if tma_output else min(bn, 64)
    return (
        stage_bytes * stages
        + (128 - rows) * 128
        + output_stages * 128 * epi_n * output_element_size
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
    tma_output_available: bool = False,
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
            any(
                config.get(key) not in (None, False, 128, "default")
                for key in MXFP8_CONFIG_KEYS
            )
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
    if mxfp8 and config["cute_scaled_mma"] is not True:
        config["cute_scaled_epilogue_mode"] = "default"
    if mxfp8:
        mode = config["cute_scaled_epilogue_mode"]
        tile_n = config["cute_scaled_tile_n"]
        tile_k = config["cute_scaled_tile_k"]
        logical_m = config["cute_scaled_tile_m"]
        swap_ab = config["cute_scaled_swap_ab"]
        fused_scales = config["cute_scaled_fused_scales"]
        # The typed scalar choices were validated above before any repair.
        assert isinstance(mode, str)
        assert isinstance(tile_n, int)
        assert isinstance(tile_k, int)
        assert isinstance(logical_m, int)
        assert isinstance(swap_ab, bool)
        assert isinstance(fused_scales, bool)
        if not mxfp8_epilogue_supported(
            mode,
            tile_n,
            tile_k,
            logical_m,
            swap_ab=swap_ab,
            fused_scales=fused_scales,
            tma_output_available=tma_output_available,
        ):
            if fix_invalid:
                config["cute_scaled_epilogue_mode"] = "default"
            else:
                raise InvalidConfig(
                    "TMA epilogues require a proved BF16 row-major output, fused "
                    "MXFP8, unswapped logical M128, complete 32-column packets and "
                    "disjoint accumulator/scale TMEM banks within 512 columns"
                )
