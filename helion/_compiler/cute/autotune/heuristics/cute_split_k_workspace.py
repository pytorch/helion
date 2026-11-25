"""Expose native FP32 partial/reduction schedules beside atomic schedules."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING

from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ... import target_facts
from ...pipeline_smem import Tcgen05PipelineSmemFacts
from ...pipeline_smem import max_pipeline_ab_stages
from ...split_k_workspace_config import ATOMIC_EPILOGUE_KEY
from ...split_k_workspace_config import PARTIALS_KEY
from ...split_k_workspace_config import PARTIALS_OUTPUT
from ...split_k_workspace_config import PARTITIONS_KEY
from ...split_k_workspace_config import STAGES_KEY
from ...split_k_workspace_config import WORKSPACE_KEY
from ...strategies import TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY
from ...strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY
from ...strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY
from ...strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from ...strategies import Tcgen05LayoutStrategy
from ...tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY

if TYPE_CHECKING:
    from ....autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ....compile_environment import CompileEnvironment
    from ....compile_environment import ConfigValueExpression
    from ....device_ir import DeviceIR


def _config_keys(expression: ConfigValueExpression | int) -> set[str]:
    if isinstance(expression, int):
        return set()
    return {
        key
        for argument in expression.arguments
        for key in ({argument} if isinstance(argument, str) else _config_keys(argument))
    }


class CuteSplitKWorkspaceHeuristic(AutotunerHeuristic):
    name = "cute_split_k_workspace"
    backend = "cute"
    priority = 14
    CACHE_SPECIALIZATION_FACTS = frozenset({"config_num_sm"})

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ...split_k_workspace import analyze_split_k_workspace

        if device_ir.host_function is None:
            return frozenset()
        from ...split_k_workspace import WorkspaceGeometry
        from ...split_k_workspace import atomic_epilogue_admissible
        from ...split_k_workspace import partials_admissible
        from ...split_k_workspace import plain_store_partition_choices

        with device_ir.host_function:
            proof = analyze_split_k_workspace(env, device_ir)
            env.config_spec.cute_split_k_workspace_available = proof is not None
            env.config_spec.cute_split_k_partition_choices = (
                plain_store_partition_choices(proof, env.config_spec.num_sm)
                if proof is not None
                else ()
            )
            env.config_spec.cute_split_k_atomic_epilogue_available = (
                proof is not None and atomic_epilogue_admissible(proof)
            )
            env.config_spec.cute_split_k_partials_available = (
                proof is not None and partials_admissible(proof, env)
            )
            env.config_spec.cute_split_k_workspace_geometry = (
                WorkspaceGeometry.of(
                    proof, target_facts.per_cta_smem_capacity_bytes(env.device)
                )
                if proof is not None
                else None
            )
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec.cute_split_k_workspace_available

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        from ...split_k_workspace import analyze_split_k_workspace
        from ...split_k_workspace import wave_fill
        from ...split_k_workspace import workspace_schedule

        if not cls.is_eligible(env, device_ir):
            return []
        assert device_ir.host_function is not None
        with device_ir.host_function:
            proof = analyze_split_k_workspace(env, device_ir)
        assert proof is not None
        spec = env.config_spec
        capacity = target_facts.per_cta_smem_capacity_bytes(env.device)
        if capacity <= 0:
            return []
        base = spec._base_default_config()
        # The schedule builds its producer's config itself (a persistent
        # two-CTA GEMM over the partition views), so the pipeline surface a
        # default config carries is inert here; on a TVM-FFI-eligible shape
        # that default is the clustered direct-entry seed. Put the seeds on
        # the one-CTA surface with the promotion off: a seed that keeps the
        # cluster or requests the promotion is projected onto the 256x256
        # pair by the search fixers (``_fix_target1_tvm_ffi_search_config``,
        # ``_fix_cluster_m2_search_config``), which rewrite its tile and so
        # fail the strict seed transfer. The pid type is flat where the
        # kernel allows it (a persistent default keeps a one-CTA 256-row
        # tile off the search surface and the seed would be rewritten) and
        # stays the default's otherwise (force-persistent kernels).
        inert = {
            key: value
            for key, value in (
                *((("pid_type", "flat"),) if "flat" in spec.allowed_pid_types else ()),
                ("tcgen05_cluster_m", 1),
                ("tcgen05_cluster_n", 1),
                ("tcgen05_cta_group", "auto"),
                (TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY, False),
                (TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY, False),
                (
                    TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
                    Tcgen05LayoutStrategy.DEFAULT.value,
                ),
                (TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY, None),
                (TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY, None),
                (TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY, None),
            )
            if key in base.config
        }
        keys = _config_keys(proof.chunk)
        if len(keys) > 1:
            return []
        variations: list[dict[str, object]] = [{}]
        if keys:
            (key,) = keys
            if key == PARTITIONS_KEY:
                # The plain store pattern partitions through the schedule's
                # own knob (its last partition may run past K:
                # ``plain_store_chunk``).
                values: list[object] | None = [
                    split for split in spec.cute_split_k_partition_choices if split > 1
                ]
            else:
                fragment = spec.user_defined_tunables.get(key)
                searched = (
                    fragment.search_values(limit=32) if fragment is not None else None
                )
                values = list(searched) if searched is not None else None
            if values is None:
                return []
            variations = [
                {key: value} for value in values if type(value) is int and value > 0
            ]
        facts = Tcgen05PipelineSmemFacts(2, 4, capacity)
        small_grid = (
            -(-proof.m // 256) * -(-proof.n // 256) * 2 < (spec.num_sm or 1) // 2
        )
        geometry = sorted(
            (
                (bm, bn, bk)
                for bm in (128, 256)
                for bn in (64, 128, 256)
                for bk in (64, 128)
                if proof.m % bm == proof.n % bn == 0
                # (a plain store's partitions run whole K tiles past K:
                # ``plain_store_chunk``)
                and (proof.store or proof.k % bk == 0)
            ),
            key=lambda tile: (
                abs(math.log2(tile[0] / tile[1])),
                tile[0],
                tile[2],
                tile[1],
            ),
        )
        result: list[Config] = []
        for bm, bn, bk in geometry:
            overrides = {
                proof.m_block_id: bm,
                proof.n_block_id: bn,
                proof.k_block_id: bk,
            }
            block_sizes = []
            valid = True
            for item in spec.block_sizes:
                if len(item.block_ids) != 1:
                    valid = False
                    break
                fragment = item._fragment(spec)
                value = overrides.get(item.block_id, fragment.default())
                if not fragment.low <= value <= fragment.high:
                    valid = False
                    break
                block_sizes.append(value)
            if not valid:
                continue
            maximum = max_pipeline_ab_stages(
                facts, bm=bm, bn=bn, bk=bk, c_stages=2, acc_stages=2
            )
            if maximum <= 0:
                continue
            # Where the 16-bit output may take them (``partials_admissible``:
            # torch's default reduced-precision split-K flags), the seeds keep
            # their partials in the output dtype: half the bytes the producer
            # stores and the reducer loads, the sum still fp32. bf16, GB300,
            # the same configs with fp32 / bf16 partials, of cuBLAS: 256 x
            # 512 x 197952 (16 partitions of 256x128) 0.998 / 1.029, 128 x
            # 512 x 32768 CR (16 of 128x128) 0.971-1.033 / 1.017-1.082,
            # 256 x 256 x 32768 CR 1.125 / 1.186. fp32 partials stay a
            # mutation of the knob away.
            partials = (
                {PARTIALS_KEY: PARTIALS_OUTPUT}
                if spec.cute_split_k_partials_available
                else {}
            )
            candidates: list[tuple[float, int, Config]] = []
            for variation in variations:
                candidate = Config.from_dict(
                    base.config
                    | inert
                    | variation
                    | partials
                    | {
                        "block_sizes": block_sizes,
                        WORKSPACE_KEY: True,
                        STAGES_KEY: maximum,
                        "cute_collective_mma": False,
                    }
                )
                schedule = workspace_schedule(
                    proof, env, candidate, capacity_bytes=capacity
                )
                if schedule is None:
                    continue
                # On a grid that leaves half the SMs idle, target about one
                # native cluster per SM pair; on a plain store's fuller grid
                # (admitted where its waves run short:
                # ``plain_store_partition_choices``) the units that fill the
                # pairs' waves best, the fewer partitions on a tie: bf16 4096
                # x 2048 x 50264 has 128 tiles of 256x256, 84% of two waves of
                # 76 pairs, and 512 units in four partitions fill 96% of seven
                # (1.003 of cuBLAS against 0.936 in two partitions, 0.911 in
                # three). A kernel that splits K through its own tunable has
                # no such gate and keeps the one-cluster-per-pair target on
                # any grid: ranked by wave fill, its finest split would win
                # (bf16 4096 x 1536 x 16384: 256 one-tile partitions, 6.4 GB
                # of fp32 partials). Retain the two best partition counts so
                # traffic and occupancy can trade off in the initial
                # population. The parameter's spelling and numerical values
                # come only from its compiler expression.
                clusters = (proof.m // bm) * (proof.n // bn) * schedule.partitions
                num_sm = spec.num_sm or 1
                distance = (
                    -wave_fill(clusters, num_sm)
                    if proof.store and not small_grid
                    else abs(math.log2(clusters / max(1, num_sm // 2)))
                )
                candidates.append((distance, schedule.partitions, candidate))
            chosen: set[int] = set()
            for _distance, partitions, candidate in sorted(
                candidates, key=operator.itemgetter(slice(2))
            ):
                if partitions in chosen:
                    continue
                chosen.add(partitions)
                result.append(candidate)
                if spec.cute_split_k_atomic_epilogue_available:
                    # The reducer-free twin: the producer reduces into the
                    # zero-filled output itself (one kernel per call).
                    result.append(
                        Config.from_dict(candidate.config | {ATOMIC_EPILOGUE_KEY: True})
                    )
                if maximum > 2:
                    result.append(Config.from_dict(candidate.config | {STAGES_KEY: 2}))
                if len(chosen) == 2:
                    break
        return result
