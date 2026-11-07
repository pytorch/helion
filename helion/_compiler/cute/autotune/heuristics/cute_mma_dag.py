"""Admission and seeds of the ``cute_mma_dag`` lowering (the seeds: ``MmaDagSeedPlanner``)."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from ....autotuner_heuristics.registry import AutotunerHeuristic
from ....compile_environment import CompileEnvironment
from ...cute_mma_dag_seeds import SEED_TCGEN05
from ...cute_mma_dag_seeds import SEED_TILES
from ...cute_mma_dag_seeds import MmaDagSeedPlanner
from ...cute_mma_dag_seeds import streaming_regions
from ...cute_mma_dag_seeds import wide_epilogue_rows
from ..mma_dag_surface import enable_cute_mma_dag_search
from .cute_launch_bounds import MIN_BLOCKS_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .....runtime.config import Config
    from ....autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ....device_ir import DeviceIR
    from ...cute_mma_dag import CuteMmaDagFacts


class CuteMmaDagHeuristic(AutotunerHeuristic):
    """Admit and seed the warp-level MMA lowering of serial state recurrences.

    ``register_facts`` runs the config-free matcher (``match_mma_dag``) on
    the device IR; where it matches, the knobs enter the search surface.
    The block-size fragments are left alone (``normalize`` turns tiles the
    mode cannot run into ``InvalidConfig`` / repairs, and the seeds emit
    legal tiles), so admission never changes the search space of the
    ordinary lowering.  The seeds are ``MmaDagSeedPlanner``'s
    (cute_mma_dag_seeds.py): the warp ladder walks the tiles {16, 32, 64,
    128} over the tunable axes, takes the most warps that tile every
    fragment within the register budget (and, ranked first, the most warps
    that feed every fragment operand of the loop from registers when fewer
    warps allow it), ranks by CTA count (>= 128 first), root tile, shared
    memory for two CTAs per SM, loop tile and stages, and emits the top
    eight with their legal stage depths; the tensor-memory seeds and the
    knob twins follow the planner's steps.  Promotion to the autotune-off
    default is limited to the GPU generations the lowering was validated on.
    """

    name = "cute_mma_dag"
    backend = "cute"
    priority = 5
    promote_seed_to_default = True
    PROMOTE_TARGETS = (("cuda", "sm100"), ("cuda", "sm103"))
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})
    SEED_TILES = SEED_TILES
    SEED_TCGEN05 = SEED_TCGEN05

    @classmethod
    def _facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> CuteMmaDagFacts | None:
        from ...cute_mma_dag import match_mma_dag
        from ...cute_mma_dag import mma_dag_facts

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
            # The facts read the bound inputs' alignment (the side tiles of
            # the loops whose cp.async ring stages a tile for them:
            # ``mma_dag_ring_loops``), and seed registration precedes the
            # runtime's storage snapshot: evaluate the registered,
            # metadata-only classifiers here and restore the prior snapshot
            # afterwards (as the tcgen05 facts hook does).  Without a
            # binding (no runtime values) no base is provably aligned and
            # the knob is pinned to the lane loads.
            saved = env.bound_runtime_input_specialization_results
            try:
                if env.runtime_arg_values_by_name:
                    env.snapshot_runtime_input_specialization_results(
                        env.runtime_arg_values_by_name
                    )
                return mma_dag_facts(program)
            finally:
                env.bound_runtime_input_specialization_results = saved

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        facts = cls._facts(env, device_ir)
        if facts is None:
            return frozenset()
        enable_cute_mma_dag_search(env.config_spec, facts)
        return cls.CACHE_SPECIALIZATION_FACTS

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec.cute_mma_dag is not None

    @classmethod
    def should_promote(cls, env: CompileEnvironment) -> bool:
        # A kernel another whole-root owner lowers (the gdn recurrence's
        # tcgen05 schedule) keeps that owner's promoted default: the dag is
        # one of the search's choices there (its seeds and the
        # ``cute_mma_dag`` knob), not the autotune-off lowering.
        if env.config_spec.cute_gdn_recurrence_stages is not None:
            return False
        return super().should_promote(env)

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        spec = env.config_spec
        facts = spec.cute_mma_dag_facts
        if facts is None or spec.cute_mma_dag is None:
            return None
        planner = MmaDagSeedPlanner(facts, spec, occupancy_key=MIN_BLOCKS_KEY)
        return planner.configs(planner.ranked()) or None

    @classmethod
    def wide_epilogue_rows(
        cls, facts: CuteMmaDagFacts, sizes: Mapping[int, object]
    ) -> bool:
        return wide_epilogue_rows(facts, sizes)

    @classmethod
    def streaming_regions(cls, facts: CuteMmaDagFacts) -> bool:
        return streaming_regions(facts)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None
