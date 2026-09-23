"""Explicit schedule choices for proved native grouped RNA contractions."""

from __future__ import annotations

from typing import cast

from ...exc import InvalidConfig
from .tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from .tcgen05_grouped_descriptors import DYNAMIC
from .tcgen05_grouped_descriptors import WRAPPED
from .tcgen05_tma_rn import AUTO
from .tcgen05_tma_rn import CONVERSION_KEY
from .tcgen05_tma_rn import TMA_RN
from .tcgen05_tma_rn import WARP_RAW

WARPS_KEY = "cute_grouped_rna_warps"
K_KEY = "cute_grouped_rna_block_k"
STAGES_KEY = "cute_grouped_ab_stages"
RESIDENT_CTAS_KEY = "cute_grouped_ctas_per_sm"
PREFIX_SCAN_KEY = "cute_grouped_prefix_scan"
PREFIX_SCANS = ("warp", "block")
CONFIG_KEYS = frozenset(
    (
        WARPS_KEY,
        K_KEY,
        DESCRIPTOR_KEY,
        CONVERSION_KEY,
        STAGES_KEY,
        RESIDENT_CTAS_KEY,
        PREFIX_SCAN_KEY,
    )
)
CONVERTER_WARPS = (0, 4, 8, 2, 1)
K_STAGES = {32: 6, 64: 3}


def normalize_grouped_rna_config(
    config: dict[str, object],
    *,
    available_k: tuple[int, ...],
    fix_invalid: bool,
    wrapped_descriptors_available: bool = False,
    rn_two_stage_ctas: tuple[int, ...] = (),
    warp_raw_available: bool = False,
    block_prefix_recipes: tuple[tuple[int, int, int], ...] = (),
) -> None:
    prefix = config.get(PREFIX_SCAN_KEY, "warp")
    if type(prefix) is not str or prefix not in PREFIX_SCANS:
        if not fix_invalid:
            raise InvalidConfig(f"{PREFIX_SCAN_KEY} must be warp or block")
        prefix = "warp"
    config.pop(PREFIX_SCAN_KEY, None)
    conversion = config.get(CONVERSION_KEY, AUTO)
    if type(conversion) is not str or conversion not in (AUTO, TMA_RN, WARP_RAW):
        if not fix_invalid:
            raise InvalidConfig(f"{CONVERSION_KEY} must be auto, tma_rn or warp_raw")
        conversion = AUTO
    if prefix == "block" and conversion != TMA_RN:
        if not fix_invalid:
            raise InvalidConfig(f"{PREFIX_SCAN_KEY}=block requires TMA RN")
        prefix = "warp"
    if conversion == WARP_RAW:
        if not warp_raw_available:
            if not fix_invalid:
                raise InvalidConfig(
                    "raw warp TF32 requires a proved aligned direct FP32 TF32 contraction"
                )
            for key in CONFIG_KEYS:
                config.pop(key, None)
            return
        # One explicit numerical/geometry family: no RN descriptor, converter,
        # persistent grid or old K/stage option may silently alter its meaning.
        for key in CONFIG_KEYS - {CONVERSION_KEY}:
            config.pop(key, None)
        config[CONVERSION_KEY] = WARP_RAW
        return
    if conversion == AUTO:
        config.pop(CONVERSION_KEY, None)
    if conversion == TMA_RN:
        if not available_k:
            if not fix_invalid:
                raise InvalidConfig(
                    "TMA RN requires a proved direct FP32 TF32 contraction"
                )
            for key in CONFIG_KEYS:
                config.pop(key, None)
            return
        # Conversion has no SIMT role. The old zero sentinel still means off
        # only when the independent TMA-RN policy is absent.
        config.pop(WARPS_KEY, None)
        config[CONVERSION_KEY] = TMA_RN
        _normalize_grouped_geometry(
            config, available_k, fix_invalid, wrapped_descriptors_available
        )
        _normalize_rn_pipeline(config, rn_two_stage_ctas, fix_invalid)
        if prefix == "block":
            block_k = cast("int", config[K_KEY])
            recipe = (
                block_k,
                config.get(STAGES_KEY, K_STAGES[block_k]),
                config.get(RESIDENT_CTAS_KEY, 1),
            )
            if recipe not in block_prefix_recipes:
                if not fix_invalid:
                    raise InvalidConfig(
                        f"{PREFIX_SCAN_KEY}=block exceeds the proved CTA/storage domain"
                    )
            else:
                config[PREFIX_SCAN_KEY] = prefix
        return
    for key in (STAGES_KEY, RESIDENT_CTAS_KEY):
        if key in config:
            if not fix_invalid and config[key] != (0 if key == STAGES_KEY else 1):
                raise InvalidConfig(f"{key} requires TMA RN")
            config.pop(key)
    warps = config.get(WARPS_KEY, 0)
    if type(warps) is not int or warps not in CONVERTER_WARPS:
        if not fix_invalid:
            raise InvalidConfig(f"{WARPS_KEY} must be one of {CONVERTER_WARPS}")
        warps = 0
    if not available_k and warps != 0:
        if not fix_invalid:
            raise InvalidConfig(
                "native grouped RNA requires a proved static FP32 contraction, "
                "private output, aligned inputs and a complete resource budget"
            )
        warps = 0
    if warps == 0:
        # Off is the old path and retains its old canonical config identity.
        for key in CONFIG_KEYS:
            config.pop(key, None)
        return
    config[WARPS_KEY] = warps
    _normalize_grouped_geometry(
        config, available_k, fix_invalid, wrapped_descriptors_available
    )


def _normalize_grouped_geometry(
    config: dict[str, object],
    available_k: tuple[int, ...],
    fix_invalid: bool,
    wrapped_descriptors_available: bool,
) -> None:
    block_k = config.get(K_KEY, available_k[0])
    if type(block_k) is not int or block_k not in available_k:
        if not fix_invalid:
            raise InvalidConfig(f"{K_KEY} must be one of {available_k}")
        block_k = available_k[0]
    config[K_KEY] = block_k

    policy = config.get(DESCRIPTOR_KEY, DYNAMIC)
    if type(policy) is not str or policy not in (DYNAMIC, WRAPPED):
        if not fix_invalid:
            raise InvalidConfig(f"{DESCRIPTOR_KEY} must be dynamic or wrapped")
        policy = DYNAMIC
    if policy == WRAPPED and wrapped_descriptors_available:
        config[DESCRIPTOR_KEY] = WRAPPED
    else:
        # Explicit dynamic and statically unsupported requests canonicalize to
        # the exact prior descriptor path, including its existing cache key.
        config.pop(DESCRIPTOR_KEY, None)


def _normalize_rn_pipeline(
    config: dict[str, object], available_ctas: tuple[int, ...], fix_invalid: bool
) -> None:
    stages = config.get(STAGES_KEY, 0)
    if (
        type(stages) is not int
        or stages not in (0, 2)
        or (stages == 2 and (config[K_KEY] != 32 or not available_ctas))
    ):
        if not fix_invalid:
            raise InvalidConfig(
                f"{STAGES_KEY} requires an admitted RN K32/two-stage schedule"
            )
        stages = 0
    ctas = config.get(RESIDENT_CTAS_KEY, 1)
    if (
        type(ctas) is not int
        or ctas not in (1, 2)
        or (ctas == 2 and (stages != 2 or ctas not in available_ctas))
    ):
        if not fix_invalid:
            raise InvalidConfig(
                f"{RESIDENT_CTAS_KEY} exceeds the proved RN resource domain"
            )
        ctas = 1
    if stages:
        config[STAGES_KEY] = stages
    else:
        config.pop(STAGES_KEY, None)
    if ctas != 1:
        config[RESIDENT_CTAS_KEY] = ctas
    else:
        config.pop(RESIDENT_CTAS_KEY, None)
