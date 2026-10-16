"""Seeds for the ``cute_mma_dag`` warp-MMA state recurrence owner."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from ...runtime.config import Config
from ..compile_environment import CompileEnvironment
from ..cute.cute_mma_dag import MMA_DAG_KEY
from ..cute.cute_mma_dag import MMA_DAG_STAGES_KEY
from ..cute.cute_mma_dag import MMA_DAG_STORE_DIRECT
from ..cute.cute_mma_dag import MMA_DAG_STORE_KEY
from ..cute.cute_mma_dag import MMA_DAG_STORE_STAGED
from ..cute.cute_mma_dag import MMA_DAG_WARP
from ..cute.cute_mma_dag import MMA_DAG_WARPS_KEY
from ..cute.cute_mma_dag import CuteMmaDagFacts
from ..cute.cute_mma_dag import mma_dag_seed_configs
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteMmaDagHeuristic(AutotunerHeuristic):
    """Admit and seed the warp-level MMA lowering of serial state recurrences.

    ``register_facts`` runs the config-free matcher (``match_mma_dag``) on
    the device IR; where it matches, the knobs enter the search surface.
    The block-size fragments are left alone (``normalize`` turns tiles the
    mode cannot run into ``InvalidConfig`` / repairs, and the seeds emit
    legal tiles), so admission never changes the search space of the
    ordinary lowering.  The seeds walk the tile
    ladder {16, 32, 64, 128} over the tunable axes, take the most warps that
    tile every fragment within the register budget (and, ranked first, the
    most warps that feed every fragment operand of the loop from registers
    when fewer warps allow it), rank by CTA count (>= 128 first), root tile,
    shared memory for two CTAs per SM, loop tile and stages, and emit the
    top eight with their legal stage depths.  Promotion to the autotune-off
    default is limited to the GPU generations the lowering was validated on.
    """

    name = "cute_mma_dag"
    backend = "cute"
    promote_seed_to_default = True
    PROMOTE_TARGETS = (("cuda", "sm100"), ("cuda", "sm103"))
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})
    SEED_TILES = 8

    @classmethod
    def _facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> CuteMmaDagFacts | None:
        from ..cute.cute_mma_dag import match_mma_dag
        from ..cute.cute_mma_dag import mma_dag_facts

        host_function = device_ir.host_function
        if host_function is None:
            return None
        # The matcher resolves block ids and symbol origins through the
        # compile environment and the host function; the environment is not
        # current when the seeds are regenerated outside a compile.
        with contextlib.ExitStack() as stack:
            if (
                isinstance(env, CompileEnvironment)
                and not CompileEnvironment.has_current()
            ):
                stack.enter_context(env)
            stack.enter_context(host_function)
            program = match_mma_dag(device_ir.graphs)
            if program is None:
                return None
            return mma_dag_facts(program)

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        facts = cls._facts(env, device_ir)
        if facts is None:
            return frozenset()
        env.config_spec.enable_cute_mma_dag_search(facts)
        return cls.CACHE_SPECIALIZATION_FACTS

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec.cute_mma_dag is not None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        spec = env.config_spec
        facts = spec.cute_mma_dag_facts
        if facts is None or spec.cute_mma_dag is None:
            return None
        block_ids = spec.block_sizes.valid_block_ids()
        default_sizes = spec.default_config().config.get("block_sizes", [])
        assert isinstance(default_sizes, list)
        seeds: list[Config] = []
        # fp32 snapshots are 8-byte fragment pairs either way: the direct
        # pair stores beat the staging round trip (bwd_dh at s2: 255 us
        # against 263); 16-bit snapshots gain from the 16-byte packets of
        # the staging tile (fwd_h: 137 against 180).
        store = (
            MMA_DAG_STORE_DIRECT
            if any(elem_bytes == 4 for _rows, _cols, elem_bytes in facts.staging)
            else MMA_DAG_STORE_STAGED
        )
        for sizes, warps, stages in mma_dag_seed_configs(facts)[: cls.SEED_TILES]:
            block_sizes = list(default_sizes)
            for block_id, size in sizes.items():
                if block_id in block_ids:
                    block_sizes[spec.block_sizes.block_id_to_index(block_id)] = size
            seeds.append(
                Config.from_dict(
                    {
                        "block_sizes": block_sizes,
                        MMA_DAG_KEY: MMA_DAG_WARP,
                        MMA_DAG_WARPS_KEY: warps,
                        MMA_DAG_STAGES_KEY: stages,
                        MMA_DAG_STORE_KEY: store,
                    }
                )
            )
        return seeds or None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None
