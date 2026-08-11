"""Checked-in B200 heuristic for the grouped FP16 benchmark."""

from __future__ import annotations

from copy import deepcopy


_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY = "tcgen05_grouped_static_problem_signature"
_COMMON_CONFIG: dict[str, object] = {
    "block_sizes": [128, 64, 64],
    "loop_orders": [[0, 1]],
    "l2_groupings": [1],
    "num_warps": 8,
    "num_stages": 2,
    "pid_type": "persistent_interleaved",
    "tcgen05_cluster_m": 1,
    "tcgen05_cluster_n": 1,
    "tcgen05_num_epi_warps": 4,
    "tcgen05_grouped_mode": "direct",
    "tcgen05_grouped_external_direct_pointers": "direct_pointers",
    "tcgen05_grouped_external_direct_strides": "direct_strides",
}
_DEEP_CONFIG: dict[str, object] = {
    **_COMMON_CONFIG,
    "tcgen05_ab_stages": 8,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 4,
}
_SMALL_CONFIG: dict[str, object] = {
    **_COMMON_CONFIG,
    "tcgen05_ab_stages": 2,
    "tcgen05_acc_stages": 1,
    "tcgen05_c_stages": 2,
}

_PROBLEM_SIGNATURES = (
    (3, 128, 128, 128, 512, 128, 128, 128, 256, 128, 0, 0, 0),
    (4, 8192, 1280, 32, 128, 384, 1536, 640, 1280, 16, 640, 128, 16),
    (4, 8192, 1280, 32, 128, 384, 1536, 640, 1280, 16, 640, 192, 16),
    (4, 8192, 1280, 32, 16, 384, 1536, 640, 1280, 16, 640, 160, 16),
    (4, 8192, 1280, 32, 16, 384, 1536, 640, 1280, 16, 640, 192, 16),
    (4, 8192, 1280, 32, 16, 384, 1536, 640, 1280, 16, 640, 128, 16),
    (4, 8192, 1280, 32, 128, 384, 1536, 640, 1280, 16, 640, 160, 16),
)
_MEASURED_PIPELINES = (
    _SMALL_CONFIG,
    _DEEP_CONFIG,
    _DEEP_CONFIG,
    _DEEP_CONFIG,
    _DEEP_CONFIG,
    _DEEP_CONFIG,
    _DEEP_CONFIG,
)
_DEEP_PIPELINE_MIN_OUTPUT_TILES = 128


def _config_for_signature(
    signature: tuple[int, ...], pipeline: dict[str, object]
) -> dict[str, object]:
    group_count = signature[0]
    config = deepcopy(pipeline)
    config[_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY] = [
        int(value) for value in signature[: 1 + 3 * group_count]
    ]
    return config


CONFIGS = [
    _config_for_signature(signature, pipeline)
    for signature, pipeline in zip(
        _PROBLEM_SIGNATURES, _MEASURED_PIPELINES, strict=True
    )
]
_CONFIG_BY_SIGNATURE: dict[tuple[int, ...], dict[str, object]] = dict(
    zip(_PROBLEM_SIGNATURES, CONFIGS, strict=True)
)


def _fallback_pipeline(signature: tuple[int, ...]) -> dict[str, object]:
    """Choose a valid unmeasured pipeline from the static output-tile count."""
    group_count = signature[0]
    output_tiles = sum(
        ((signature[offset] + 127) // 128)
        * ((signature[offset + 1] + 63) // 64)
        for offset in range(1, 1 + 3 * group_count, 3)
    )
    return (
        _DEEP_CONFIG
        if output_tiles >= _DEEP_PIPELINE_MIN_OUTPUT_TILES
        else _SMALL_CONFIG
    )


def autotune_grouped_gemm(*args: int) -> dict[str, object]:
    """Select the measured pipeline and embed its exact active group shapes."""
    signature = tuple(args)
    measured = _CONFIG_BY_SIGNATURE.get(signature)
    if measured is not None:
        return deepcopy(measured)
    return _config_for_signature(signature, _fallback_pipeline(signature))
