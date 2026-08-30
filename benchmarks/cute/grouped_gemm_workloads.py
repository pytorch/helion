"""Shared workload and provider metadata for grouped-GEMM benchmarks."""

from __future__ import annotations

import random
from typing import NamedTuple

DEEPGEMM_VERSION = "2.6.1"
DEEPGEMM_COMMIT = "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
DEEPGEMM_CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
DEEPGEMM_FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
DEEPGEMM_M_ALIGNMENT = 224
BENCHMARK_B_LAYOUT = "k_major"
PROVIDER_DEFAULTS_MODE = "--provider-defaults"
PROVIDER_DEFAULTS_PLAN_MODE = "--provider-defaults-plan"
PROVIDER_DEFAULTS_WORKER_MODE = "--provider-defaults-worker"
PROVIDER_CLI_MODES = (
    PROVIDER_DEFAULTS_MODE,
    PROVIDER_DEFAULTS_PLAN_MODE,
    PROVIDER_DEFAULTS_WORKER_MODE,
)
PROVIDER_SELECTION_MODES = {
    "deepgemm": "public_kmajor_nk_no_psum_zero_padding_off",
    "quack": "public_api_default_tuned",
    "cudnn": "public_a_fallback_build_execute",
    "cublaslt": "grouped_shape_hints_heuristic_result_zero_requested_one",
    "cutlass": "all_supported_registry_operators_cold_l2_tuned",
}


class OfficialShape(NamedTuple):
    row_index: int
    groups: int
    expected_m_per_group: int
    n: int
    k: int


OFFICIAL_SHAPES = (
    OfficialShape(0, 4, 8192, 6144, 7168),
    OfficialShape(1, 4, 8192, 7168, 3072),
    OfficialShape(2, 4, 8192, 4096, 4096),
    OfficialShape(3, 4, 8192, 4096, 2048),
    OfficialShape(4, 8, 4096, 6144, 7168),
    OfficialShape(5, 8, 4096, 7168, 3072),
    OfficialShape(6, 8, 4096, 4096, 4096),
    OfficialShape(7, 8, 4096, 4096, 2048),
)


def official_actual_ms(seed: int = 0) -> tuple[tuple[int, ...], ...]:
    """Return the deterministic DeepGEMM-compatible logical M sizes."""

    rng = random.Random(seed)
    return tuple(
        tuple(
            int(shape.expected_m_per_group * rng.uniform(0.7, 1.3))
            for _ in range(shape.groups)
        )
        for shape in OFFICIAL_SHAPES
    )
