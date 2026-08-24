"""Reviewed grouped-GEMM shape profiles without runtime dependencies.

This is the shared source of truth for the reviewed Helion configuration and
physical-B policy. It stays standard-library-only so the checked-in AOT
heuristic can load it without importing Torch, Helion, or CUDA-facing code.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal
from typing import NamedTuple

LEGACY_M_ALIGNMENT = 224
SMALL_M_ALIGNMENT = 32
PROFILED_M_ALIGNMENT = 256
SOURCE_M_TILES = (SMALL_M_ALIGNMENT, LEGACY_M_ALIGNMENT, PROFILED_M_ALIGNMENT)
SOURCE_M_TILE_ERROR = "source_m_tile must be one of 32, 224, or 256"
REVIEWED_PROFILE_MANIFEST_SHA256 = (
    "82d3cf3b33b5e755c3dacbfe2f4d53163cb6ffaa5e72354b8ebafebf6f5c10ae"
)
BLOCK_M = 256
BLOCK_N = 128
DEFAULT_BLOCK_K = 64
LARGE_BLOCK_K = 128
# Intentionally mirrors the compiler seed builder. This module cannot import
# it because checked-in AOT dispatch requires this reviewed policy to stay
# stdlib-only; the schema-parity test guards the duplicate definition.
WORKLIST_CONFIG_KEYS = frozenset(
    {
        "block_sizes",
        "l2_groupings",
        "loop_orders",
        "num_stages",
        "num_warps",
        "pid_type",
        "tcgen05_cluster_m",
        "tcgen05_cluster_n",
        "tcgen05_ab_stages",
        "tcgen05_acc_stages",
        "tcgen05_c_stages",
        "tcgen05_num_epi_warps",
        "tcgen05_consumer_regs",
        "tcgen05_grouped_mode",
        "tcgen05_grouped_worklist_source_m_tile",
        "tcgen05_grouped_runtime_direct",
        "tcgen05_l2_swizzle_size",
        "tcgen05_grouped_static_reserved_sms",
        "tcgen05_strategy",
        "tcgen05_warp_spec_scheduler_warps",
        "tcgen05_persistence_model",
    }
)


class OfficialShape(NamedTuple):
    row_index: int
    groups: int
    expected_m_per_group: int
    n: int
    k: int


def validate_source_m_tile(source_m_tile: int) -> int:
    """Return one supported physical-A tile or raise the shared error."""
    if source_m_tile not in SOURCE_M_TILES:
        raise ValueError(SOURCE_M_TILE_ERROR)
    return source_m_tile


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


def sample_actual_ms(
    groups: int, expected_m: int, generator: random.Random
) -> tuple[int, ...]:
    return tuple(int(expected_m * generator.uniform(0.7, 1.3)) for _ in range(groups))


def official_actual_ms(seed: int = 0) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    return tuple(
        sample_actual_ms(shape.groups, shape.expected_m_per_group, rng)
        for shape in OFFICIAL_SHAPES
    )


@dataclass(frozen=True, slots=True)
class WorklistConfigSpec:
    """Stdlib-only inputs for one reviewed worklist-NM configuration."""

    ab_stages: int
    source_m_tile: int = LEGACY_M_ALIGNMENT
    block_k: int = DEFAULT_BLOCK_K
    consumer_regs: int = 240
    runtime_direct: bool = True
    clc: bool = False
    l2_swizzle_size: int | None = None
    reserved_sms: int | None = None

    @property
    def cluster_m(self) -> int:
        """Match the compiler seed builder's physical source-tile contract."""
        return 1 if self.source_m_tile == SMALL_M_ALIGNMENT else 2


@dataclass(frozen=True, slots=True)
class ReviewedWorklistProfile:
    """Named configuration and physical grouped-B storage for one shape."""

    config_name: str
    b_major: Literal["k", "n"]

    @property
    def source_m_tile(self) -> int:
        """Derive physical packing from the selected kernel configuration."""
        return _CONFIG_SPECS[self.config_name].source_m_tile


@dataclass(frozen=True, slots=True)
class ReviewedDispatchProfile:
    """One config and the exact normalized worklist on which it was reviewed."""

    config_name: str
    packed_m: int
    worklist_signature: str


def worklist_config_values(spec: WorklistConfigSpec) -> dict[str, object]:
    """Materialize fresh Helion-compatible values from a lightweight spec."""
    if spec.clc and not spec.runtime_direct:
        raise ValueError("grouped worklist CLC requires runtime_direct=True")
    values: dict[str, object] = {
        "block_sizes": [BLOCK_M, BLOCK_N, spec.block_k],
        "l2_groupings": [1],
        "loop_orders": [[0, 1, 2]],
        "num_stages": 7,
        "num_warps": 8,
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": spec.cluster_m,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": spec.ab_stages,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_consumer_regs": spec.consumer_regs,
        "tcgen05_grouped_mode": "worklist_nm",
        "tcgen05_grouped_worklist_source_m_tile": spec.source_m_tile,
    }
    if spec.runtime_direct:
        values["tcgen05_grouped_runtime_direct"] = True
    if spec.clc:
        values.update(
            {
                "tcgen05_strategy": "role_local_with_scheduler",
                "tcgen05_warp_spec_scheduler_warps": 1,
                "tcgen05_persistence_model": "clc_persistent",
            }
        )
    if spec.l2_swizzle_size is not None:
        values["tcgen05_l2_swizzle_size"] = spec.l2_swizzle_size
    if spec.reserved_sms is not None:
        values["tcgen05_grouped_static_reserved_sms"] = spec.reserved_sms
    if missing_schema_keys := values.keys() - WORKLIST_CONFIG_KEYS:
        raise RuntimeError(
            "reviewed worklist config emitted keys missing from its schema: "
            f"{sorted(missing_schema_keys)!r}"
        )
    return values


AOT_CONFIG_NAMES = tuple(f"_AOT_CONFIG_AB{ab_stages}" for ab_stages in range(4, 8))

_CONFIG_SPECS: dict[str, WorklistConfigSpec] = {
    **{
        name: WorklistConfigSpec(ab_stages, runtime_direct=False)
        for name, ab_stages in zip(AOT_CONFIG_NAMES, range(4, 8), strict=True)
    },
    "_SOURCE32_CONFIG": WorklistConfigSpec(
        7,
        source_m_tile=SMALL_M_ALIGNMENT,
        runtime_direct=False,
    ),
    "_SOURCE32_BK128_AB5_R256_DIRECT_CONFIG": WorklistConfigSpec(
        5,
        source_m_tile=SMALL_M_ALIGNMENT,
        block_k=LARGE_BLOCK_K,
        consumer_regs=256,
    ),
    "_SOURCE32_BK128_AB5_R256_RSV32_DIRECT_CONFIG": WorklistConfigSpec(
        5,
        source_m_tile=SMALL_M_ALIGNMENT,
        block_k=LARGE_BLOCK_K,
        consumer_regs=256,
        reserved_sms=32,
    ),
    "_SOURCE32_BK64_AB7_R240_DIRECT_CONFIG": WorklistConfigSpec(
        7,
        source_m_tile=SMALL_M_ALIGNMENT,
    ),
    "_SOURCE32_BK64_AB7_R240_RSV52_DIRECT_CONFIG": WorklistConfigSpec(
        7,
        source_m_tile=SMALL_M_ALIGNMENT,
        reserved_sms=52,
    ),
    "_SOURCE32_BK64_AB7_R240_PANEL4_RSV20_DIRECT_CONFIG": WorklistConfigSpec(
        7,
        source_m_tile=SMALL_M_ALIGNMENT,
        l2_swizzle_size=4,
        reserved_sms=20,
    ),
    "_SOURCE32_BK64_AB7_R240_PANEL8_RSV20_DIRECT_CONFIG": WorklistConfigSpec(
        7,
        source_m_tile=SMALL_M_ALIGNMENT,
        l2_swizzle_size=8,
        reserved_sms=20,
    ),
    "_SOURCE32_BK128_AB5_R256_RSV52_DIRECT_CONFIG": WorklistConfigSpec(
        5,
        source_m_tile=SMALL_M_ALIGNMENT,
        block_k=LARGE_BLOCK_K,
        consumer_regs=256,
        reserved_sms=52,
    ),
    "_SOURCE256_BK128_AB3_R240_CONFIG": WorklistConfigSpec(
        3,
        source_m_tile=PROFILED_M_ALIGNMENT,
        block_k=LARGE_BLOCK_K,
        runtime_direct=False,
    ),
    "_SOURCE256_BK128_AB3_R240_DIRECT_CONFIG": WorklistConfigSpec(
        3,
        source_m_tile=PROFILED_M_ALIGNMENT,
        block_k=LARGE_BLOCK_K,
    ),
    "_SOURCE256_BK128_AB3_R240_PANEL8_DIRECT_CONFIG": WorklistConfigSpec(
        3,
        source_m_tile=PROFILED_M_ALIGNMENT,
        block_k=LARGE_BLOCK_K,
        l2_swizzle_size=8,
    ),
    "_SOURCE256_BK64_AB6_R224_CONFIG": WorklistConfigSpec(
        6,
        source_m_tile=PROFILED_M_ALIGNMENT,
        consumer_regs=224,
        runtime_direct=False,
    ),
    "_SOURCE256_BK64_AB6_R240_CONFIG": WorklistConfigSpec(
        6,
        source_m_tile=PROFILED_M_ALIGNMENT,
        runtime_direct=False,
    ),
    "_SOURCE256_BK64_AB6_R240_DIRECT_CONFIG": WorklistConfigSpec(
        6,
        source_m_tile=PROFILED_M_ALIGNMENT,
    ),
    "_SOURCE256_BK64_AB6_R240_PANEL8_DIRECT_CLC_CONFIG": WorklistConfigSpec(
        6,
        source_m_tile=PROFILED_M_ALIGNMENT,
        clc=True,
        l2_swizzle_size=8,
    ),
    "_SOURCE256_BK64_AB6_R224_PANEL16_DIRECT_CONFIG": WorklistConfigSpec(
        6,
        source_m_tile=PROFILED_M_ALIGNMENT,
        consumer_regs=224,
        l2_swizzle_size=16,
    ),
    "_SOURCE256_BK64_AB6_R256_PANEL8_DIRECT_CLC_CONFIG": WorklistConfigSpec(
        6,
        source_m_tile=PROFILED_M_ALIGNMENT,
        consumer_regs=256,
        clc=True,
        l2_swizzle_size=8,
    ),
    "_SOURCE224_BK64_AB7_R256_PANEL8_DIRECT_CONFIG": WorklistConfigSpec(
        7,
        consumer_regs=256,
        l2_swizzle_size=8,
    ),
    "_SOURCE256_BK64_AB5_R224_PANEL8_DIRECT_CONFIG": WorklistConfigSpec(
        5,
        source_m_tile=PROFILED_M_ALIGNMENT,
        consumer_regs=224,
        l2_swizzle_size=8,
    ),
}
REVIEWED_CONFIG_NAMES = tuple(_CONFIG_SPECS)


def reviewed_config_spec(config_name: str) -> WorklistConfigSpec:
    """Return the immutable builder inputs for a reviewed configuration."""
    return _CONFIG_SPECS[config_name]


def reviewed_config_values(config_name: str) -> dict[str, object]:
    """Return a fresh configuration dictionary for a reviewed config name."""
    return worklist_config_values(_CONFIG_SPECS[config_name])


REVIEWED_PUBLIC_SHAPE_PROFILES: dict[
    tuple[int, int, int, int], ReviewedWorklistProfile
] = {
    (6, 20, 4096, 4096): ReviewedWorklistProfile(
        "_SOURCE32_BK128_AB5_R256_RSV52_DIRECT_CONFIG",
        "n",
    ),
    (6, 20, 4096, 2048): ReviewedWorklistProfile(
        "_SOURCE32_BK64_AB7_R240_RSV52_DIRECT_CONFIG",
        "n",
    ),
    (4, 8192, 6144, 7168): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB6_R256_PANEL8_DIRECT_CLC_CONFIG",
        "k",
    ),
    (4, 8192, 7168, 3072): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB6_R224_PANEL16_DIRECT_CONFIG",
        "n",
    ),
    (4, 8192, 4096, 4096): ReviewedWorklistProfile(
        "_SOURCE256_BK128_AB3_R240_DIRECT_CONFIG",
        "k",
    ),
    (4, 8192, 4096, 2048): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB6_R240_DIRECT_CONFIG",
        "k",
    ),
    (8, 4096, 6144, 7168): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB6_R240_PANEL8_DIRECT_CLC_CONFIG",
        "k",
    ),
    (8, 4096, 7168, 3072): ReviewedWorklistProfile(
        "_SOURCE256_BK128_AB3_R240_PANEL8_DIRECT_CONFIG",
        "k",
    ),
    (8, 4096, 4096, 4096): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB5_R224_PANEL8_DIRECT_CONFIG",
        "k",
    ),
    (8, 4096, 4096, 2048): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB6_R240_PANEL8_DIRECT_CLC_CONFIG",
        "n",
    ),
    (32, 192, 6144, 7168): ReviewedWorklistProfile(
        "_SOURCE256_BK128_AB3_R240_CONFIG",
        "k",
    ),
    (32, 192, 7168, 3072): ReviewedWorklistProfile(
        "_SOURCE256_BK128_AB3_R240_CONFIG",
        "n",
    ),
    (32, 192, 4096, 4096): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB6_R224_CONFIG",
        "n",
    ),
    (32, 192, 4096, 2048): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB6_R240_DIRECT_CONFIG",
        "n",
    ),
    (6, 1024, 6144, 7168): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB6_R240_CONFIG",
        "n",
    ),
    (6, 1024, 7168, 3072): ReviewedWorklistProfile(
        "_SOURCE224_BK64_AB7_R256_PANEL8_DIRECT_CONFIG",
        "k",
    ),
    (6, 1024, 4096, 4096): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB6_R240_CONFIG",
        "k",
    ),
    (6, 1024, 4096, 2048): ReviewedWorklistProfile(
        "_SOURCE256_BK64_AB6_R240_DIRECT_CONFIG",
        "n",
    ),
    (32, 20, 6144, 7168): ReviewedWorklistProfile(
        "_SOURCE32_BK64_AB7_R240_PANEL4_RSV20_DIRECT_CONFIG",
        "n",
    ),
    (32, 20, 7168, 3072): ReviewedWorklistProfile(
        "_SOURCE32_BK64_AB7_R240_PANEL8_RSV20_DIRECT_CONFIG",
        "n",
    ),
    (32, 20, 4096, 4096): ReviewedWorklistProfile(
        "_SOURCE32_BK64_AB7_R240_DIRECT_CONFIG",
        "n",
    ),
    (32, 20, 4096, 2048): ReviewedWorklistProfile(
        "_SOURCE32_BK64_AB7_R240_DIRECT_CONFIG",
        "n",
    ),
    (6, 20, 6144, 7168): ReviewedWorklistProfile(
        "_SOURCE32_BK64_AB7_R240_DIRECT_CONFIG",
        "n",
    ),
    (6, 20, 7168, 3072): ReviewedWorklistProfile(
        "_SOURCE32_BK128_AB5_R256_RSV32_DIRECT_CONFIG",
        "n",
    ),
}


def worklist_signature(rows: tuple[tuple[int, ...], ...]) -> str:
    """Encode validated worklist rows without relying on Python hash stability."""
    return ";".join(",".join(str(value) for value in row) for row in rows)


def tensor_layout_signature(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    storage_offset: int,
    dtype: str,
) -> str:
    """Encode one tensor's exact dtype, logical shape, and storage layout."""
    shape_text = ",".join(str(value) for value in shape)
    stride_text = ",".join(str(value) for value in strides)
    return f"{dtype}|{shape_text}|{stride_text}|{storage_offset}"


def _minimal_worklist_profile(
    actual_ms: tuple[int, ...],
    source_m_tile: int,
) -> tuple[int, str]:
    rows: list[tuple[int, int, int, int]] = []
    packed_m = 0
    for group, actual_m in enumerate(actual_ms):
        stored_m = (actual_m + source_m_tile - 1) // source_m_tile * source_m_tile
        rows.append((group, packed_m, actual_m, stored_m))
        packed_m += stored_m
    return packed_m, worklist_signature(tuple(rows))


def _build_reviewed_dispatch_profiles() -> dict[
    tuple[int, int, int, str, str, str, str, int],
    ReviewedDispatchProfile,
]:
    profiles = {}
    for shape, actual_ms in zip(
        OFFICIAL_SHAPES,
        official_actual_ms(seed=0),
        strict=True,
    ):
        profile = REVIEWED_PUBLIC_SHAPE_PROFILES[
            (shape.groups, shape.expected_m_per_group, shape.n, shape.k)
        ]
        if profile.b_major == "k":
            b_strides = (shape.n * shape.k, shape.k, 1)
        else:
            b_strides = (shape.n * shape.k, 1, shape.n)
        packed_m, signature = _minimal_worklist_profile(
            actual_ms,
            profile.source_m_tile,
        )
        a_layout = tensor_layout_signature(
            (packed_m, shape.k),
            (shape.k, 1),
            0,
            "torch.bfloat16",
        )
        b_layout = tensor_layout_signature(
            (shape.groups, shape.n, shape.k),
            b_strides,
            0,
            "torch.bfloat16",
        )
        worklist_layout = tensor_layout_signature(
            (shape.groups, 4),
            (4, 1),
            0,
            "torch.int32",
        )
        profiles[
            (
                shape.groups,
                shape.n,
                shape.k,
                profile.b_major,
                a_layout,
                b_layout,
                worklist_layout,
                profile.source_m_tile,
            )
        ] = ReviewedDispatchProfile(profile.config_name, packed_m, signature)
    return profiles


# Only the eight deterministic official rows have repository-owned input vectors
# that can bind a reviewed configuration to an exact runtime workload.
REVIEWED_DISPATCH_PROFILES = _build_reviewed_dispatch_profiles()
if len(REVIEWED_DISPATCH_PROFILES) != len(OFFICIAL_SHAPES):
    raise RuntimeError("official grouped-GEMM dispatch profiles must be unique")


def exact_reviewed_worklist_profile(
    groups: int,
    expected_m_per_group: int,
    n: int,
    k: int,
) -> ReviewedWorklistProfile:
    """Return an exact benchmark profile, never a generic fallback."""
    key = (groups, expected_m_per_group, n, k)
    try:
        return REVIEWED_PUBLIC_SHAPE_PROFILES[key]
    except KeyError as error:
        raise ValueError(f"no exact reviewed grouped-GEMM profile for {key}") from error


def reviewed_config_name(
    groups: int,
    n: int,
    k: int,
    b_major: str,
    a_layout: str,
    b_layout: str,
    worklist_layout: str,
    source_m_tile: int,
    packed_m: int,
    normalized_worklist: str,
) -> str | None:
    """Select an exact reviewed workload or abstain for compiler fallback."""

    source_m_tile = validate_source_m_tile(source_m_tile)
    if b_major not in ("k", "n"):
        raise ValueError("grouped B major must be 'k' or 'n'")
    if type(packed_m) is not int or packed_m <= 0:
        raise ValueError("packed M must be a positive integer")
    if packed_m % source_m_tile:
        return None
    profile = REVIEWED_DISPATCH_PROFILES.get(
        (
            groups,
            n,
            k,
            b_major,
            a_layout,
            b_layout,
            worklist_layout,
            source_m_tile,
        )
    )
    if (
        profile is None
        or packed_m != profile.packed_m
        or normalized_worklist != profile.worklist_signature
    ):
        return None
    return profile.config_name
