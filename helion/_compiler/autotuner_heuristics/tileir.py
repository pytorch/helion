from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...autotuner.config_spec import FULL_EXTENT_CATEGORIES
from ...autotuner.config_spec import ReductionCategory
from .common import dedupe_configs
from .common import is_canonical_row_reduction
from .registry import AutotunerHeuristic
from .triton import _materialize_config

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec
    from ...autotuner.config_spec import ReductionDescriptor
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def _block_sizes(spec: ConfigSpec, targets: dict[int, int]) -> list[int]:
    return [
        min(
            block.max_size,
            max(block.min_size, block.autotuner_min, targets[block.block_id]),
        )
        for block in spec.block_sizes
    ]


def _seed(
    spec: ConfigSpec,
    sizes: list[int],
    *,
    stages: int,
    pid_type: str = "flat",
    reduction_loops: list[int | None] | None = None,
) -> Config | None:
    # TileIR chooses warp layouts itself. Keep its occupancy/cluster requests
    # modest, and let normalization honor the kernel's legal execution choices.
    raw: dict[str, object] = {
        "block_sizes": sizes,
        "indexing": ["pointer"] * spec.indexing.length,
        "num_warps": 4,
        "num_stages": stages,
        "pid_type": pid_type,
        "num_sm_multiplier": 1,
        "num_ctas": 1,
        "occupancy": 1,
    }
    if reduction_loops is not None:
        raw["reduction_loops"] = reduction_loops
    # This also shrinks oversized tiles against the compiler's tensor-numel
    # constraints, as for the Triton seed families.
    config = _materialize_config(raw, config_spec=spec)
    sizes = config.block_sizes
    # Shrinking honors the hard minimum; a raised autotuner_min may make the
    # repaired candidate unreachable in the current search domain.
    if any(
        not max(block.min_size, block.autotuner_min) <= size <= block.max_size
        for block, size in zip(spec.block_sizes, sizes, strict=True)
    ) or any(
        not constraint.check_fn(*(sizes[i] for i in constraint.block_indices))
        for constraint in spec.tensor_numel_constraints
    ):
        return None
    return config


def _single_reduction(env: CompileEnvironment) -> ReductionDescriptor | None:
    spec = env.config_spec
    if env.backend_name != "tileir" or spec.matmul_facts:
        return None
    fact = spec.reduction_kernel_fact
    if fact is None or fact.non_reduction_loop_block_ids or not fact.reductions:
        return None
    first = fact.reductions[0]
    # Several operations may reduce the same axis (e.g. max, sum, and min in
    # chunk statistics). Decline kernels with unrelated reduction axes.
    if any(
        (d.block_id, d.category, d.size_hint)
        != (first.block_id, first.category, first.size_hint)
        for d in fact.reductions
    ):
        return None
    return first


class TileIRMatmulSeedHeuristic(AutotunerHeuristic):
    """Offer a larger half-precision dot tile to TileIR searches.

    Match the dot's three tunable axes, including registered tiles used with
    explicit grids and device-computed grouped offsets. Their logical M/N
    extents need not be static; the live block-size constraints bound the seed.
    """

    name = "tileir_matmul"
    backend = "tileir"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        if env.backend_name != cls.backend or len(spec.matmul_facts) != 1:
            return False
        fact = spec.matmul_facts[0]
        if (
            fact.lhs_ndim != 2
            or fact.rhs_ndim != 2
            or fact.lhs_dtype not in (torch.float16, torch.bfloat16)
            or fact.rhs_dtype != fact.lhs_dtype
            or spec.reduction_loops
        ):
            return False
        axes = (fact.m_block_id, fact.n_block_id, fact.k_block_id)
        return (
            None not in axes
            and len(set(axes)) == 3
            and set(axes) == set(spec.block_sizes.valid_block_ids())
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        sizes = _block_sizes(
            env.config_spec,
            dict.fromkeys(env.config_spec.block_sizes.valid_block_ids(), 128),
        )
        return _seed(env.config_spec, sizes, stages=3)


class TileIRRowReductionHeuristic(AutotunerHeuristic):
    """Offer one-row resident and streamed reduction candidates."""

    name = "tileir_row_reduction"
    backend = "tileir"
    FULL_WIDTH_LIMIT = 8192
    STREAMED_WIDTH = 4096

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        descriptor = _single_reduction(env)
        return (
            descriptor is not None
            and descriptor.category in FULL_EXTENT_CATEGORIES
            and descriptor.size_hint > 0
            and is_canonical_row_reduction(env)
            and env.config_spec.reduction_loops[0].block_id == descriptor.block_id
        )

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        spec = env.config_spec
        descriptor = _single_reduction(env)
        assert descriptor is not None
        seeds: list[Config | None] = []
        if descriptor.size_hint <= cls.FULL_WIDTH_LIMIT:
            seeds.append(_seed(spec, [1], stages=1, reduction_loops=[None]))
        if descriptor.size_hint > cls.STREAMED_WIDTH:
            seeds.append(
                _seed(
                    spec,
                    [1],
                    stages=2,
                    pid_type="persistent_blocked",
                    reduction_loops=[cls.STREAMED_WIDTH],
                )
            )
        return dedupe_configs(seed for seed in seeds if seed is not None)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None


class TileIRSmallReductionHeuristic(AutotunerHeuristic):
    """Batch independent short, fixed-width reductions into a larger tile."""

    name = "tileir_small_reduction"
    backend = "tileir"
    MAX_REDUCTION_WIDTH = 64
    GROUP_TILE = 128

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        descriptor = _single_reduction(env)
        if (
            descriptor is None
            or descriptor.category not in FULL_EXTENT_CATEGORIES
            or not 0 < descriptor.size_hint <= cls.MAX_REDUCTION_WIDTH
            or len(spec.block_sizes) != 1
            or spec.reduction_loops
        ):
            return False
        block_id = spec.block_sizes[0].block_id
        return block_id != descriptor.block_id and block_id in spec.grid_block_ids

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        sizes = _block_sizes(spec, {spec.block_sizes[0].block_id: cls.GROUP_TILE})
        return _seed(spec, sizes, stages=2)


class TileIRChunkReductionHeuristic(AutotunerHeuristic):
    """Offer a wider chunk to explicitly partitioned reduction grids.

    Registered ``hl.arange`` axes can have DECLINED logical reduction extents:
    the per-program chunk is known and tunable even when the full reduction
    length is encoded in index/mask arithmetic. Use its declared block bounds,
    without inferring the logical extent or changing the partitioning algorithm.
    """

    name = "tileir_chunk_reduction"
    backend = "tileir"
    CHUNK_TILE = 1024

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        descriptor = _single_reduction(env)
        return (
            descriptor is not None
            and descriptor.category
            in (ReductionCategory.USER_TILE, ReductionCategory.DECLINED)
            and len(spec.block_sizes) == 1
            and not spec.reduction_loops
            and spec.block_sizes[0].block_id == descriptor.block_id
            and descriptor.block_id not in spec.grid_block_ids
            and bool(spec.grid_block_ids)
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        sizes = _block_sizes(spec, {spec.block_sizes[0].block_id: cls.CHUNK_TILE})
        return _seed(spec, sizes, stages=2)
