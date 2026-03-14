from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .. import exc
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import performance
from .config_fragment import Category
from .effort_profile import BEAM_SEARCH_DEFAULTS
from .pattern_search import InitialPopulationStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .base_search import _AutotunableKernel
    from .config_generation import FlatConfig


class BeamSearch(PopulationBasedSearch):
    """Constructive beam search that builds configs dimension by dimension.

    Instead of perturbing complete configurations, this algorithm fixes one
    dimension at a time in order of importance (block sizes first, then
    num_warps, then everything else).  At each step the beam is expanded by
    trying all pattern neighbors for the current dimension and pruned back
    to ``beam_width`` best configs.

    After the constructive sweep an optional refinement phase runs
    PatternSearch-style single-parameter perturbations around the best
    config to exploit interactions missed by the left-to-right pass.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        beam_width: int = BEAM_SEARCH_DEFAULTS.beam_width,
        refinement_rounds: int = BEAM_SEARCH_DEFAULTS.refinement_rounds,
        initial_population_strategy: InitialPopulationStrategy | None = None,
        compile_timeout_lower_bound: float = BEAM_SEARCH_DEFAULTS.compile_timeout_lower_bound,
        compile_timeout_quantile: float = BEAM_SEARCH_DEFAULTS.compile_timeout_quantile,
    ) -> None:
        super().__init__(kernel, args)
        if initial_population_strategy is None:
            initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        self.beam_width = beam_width
        self.refinement_rounds = refinement_rounds
        self.initial_population_strategy = initial_population_strategy
        self.compile_timeout_lower_bound = compile_timeout_lower_bound
        self.compile_timeout_quantile = compile_timeout_quantile

    # ------------------------------------------------------------------
    # Dimension ordering
    # ------------------------------------------------------------------

    def _order_dimensions(self) -> list[int]:
        """Return flat_spec indices ordered by importance.

        Block-size dimensions first, then num_warps, then all remaining
        indices in their original order.
        """
        block_indices = set(self.config_gen.block_size_indices)
        warps_index = self.config_gen.num_warps_index
        ordered: list[int] = list(self.config_gen.block_size_indices)
        if warps_index >= 0 and warps_index not in block_indices:
            ordered.append(warps_index)
        important = block_indices | ({warps_index} if warps_index >= 0 else set())
        for i in range(len(self.config_gen.flat_spec)):
            if i not in important:
                ordered.append(i)
        return ordered

    # ------------------------------------------------------------------
    # Initial beam
    # ------------------------------------------------------------------

    def _generate_initial_beam(self) -> list[FlatConfig]:
        if self.initial_population_strategy == InitialPopulationStrategy.FROM_DEFAULT:
            return [self.config_gen.default_flat()]
        if (
            self.initial_population_strategy
            == InitialPopulationStrategy.FROM_BEST_AVAILABLE
        ):
            return self._generate_best_available_population_flat()
        return self.config_gen.random_population_flat(self.beam_width)

    # ------------------------------------------------------------------
    # Neighbor generation (for refinement phase)
    # ------------------------------------------------------------------

    def _generate_neighbors(self, base: FlatConfig) -> list[FlatConfig]:
        """Generate single-parameter perturbations plus block-size pairs.

        Reuses the same logic as PatternSearch so the refinement phase
        explores the same neighborhood structure.
        """
        candidates_by_index = [
            spec.pattern_neighbors(base[index])
            for index, spec in enumerate(self.config_gen.flat_spec)
        ]
        neighbors: list[FlatConfig] = []

        for index, candidates in enumerate(candidates_by_index):
            for candidate_value in candidates:
                new_flat = [*base]
                new_flat[index] = candidate_value
                neighbors.append(new_flat)

        block_indices = self.config_gen.block_size_indices
        for i_pos, first in enumerate(block_indices):
            first_candidates = candidates_by_index[first]
            if not first_candidates:
                continue
            for second in block_indices[i_pos + 1 :]:
                second_candidates = candidates_by_index[second]
                if not second_candidates:
                    continue
                for first_value in first_candidates:
                    for second_value in second_candidates:
                        new_flat = [*base]
                        new_flat[first] = first_value
                        new_flat[second] = second_value
                        neighbors.append(new_flat)

        return neighbors

    # ------------------------------------------------------------------
    # Main algorithm
    # ------------------------------------------------------------------

    def _autotune(self) -> Config:
        self.log(
            f"Starting BeamSearch with beam_width={self.beam_width}, "
            f"refinement_rounds={self.refinement_rounds}, "
            f"strategy={self.initial_population_strategy.name}"
        )

        ordered_dims = self._order_dimensions()
        visited: set[Config] = set()

        # --- initial beam ---
        beam_members: list[PopulationMember] = []
        for flat_config in self._generate_initial_beam():
            member = self.make_unbenchmarked(flat_config)
            if member.config not in visited:
                visited.add(member.config)
                beam_members.append(member)

        self.parallel_benchmark_population(beam_members, desc="Initial beam")
        self.set_adaptive_compile_timeout(
            beam_members,
            min_seconds=self.compile_timeout_lower_bound,
            quantile=self.compile_timeout_quantile,
        )
        self.rebenchmark_population(beam_members, desc="Verifying initial beam")
        beam_members.sort(key=performance)
        beam_members = [
            m for m in beam_members[: self.beam_width] if math.isfinite(m.perf)
        ]
        if not beam_members:
            raise exc.NoConfigFound

        self.population = list(beam_members)
        self.log(
            f"Initial beam of {len(beam_members)} config(s):", self.statistics
        )

        # --- dimension-by-dimension sweep ---
        for gen, dim_idx in enumerate(ordered_dims, start=1):
            self.set_generation(gen)
            spec = self.config_gen.flat_spec[dim_idx]
            candidates: list[PopulationMember] = []
            seen_in_step: set[Config] = set()

            for member in beam_members:
                # keep the current value as a candidate
                if member.config not in seen_in_step:
                    seen_in_step.add(member.config)
                    candidates.append(member)
                # expand with pattern neighbors for this dimension
                for neighbor_val in spec.pattern_neighbors(member.flat_values[dim_idx]):
                    new_flat = [*member.flat_values]
                    new_flat[dim_idx] = neighbor_val
                    new_member = self.make_unbenchmarked(new_flat)
                    if (
                        new_member.config not in visited
                        and new_member.config not in seen_in_step
                    ):
                        seen_in_step.add(new_member.config)
                        visited.add(new_member.config)
                        candidates.append(new_member)

            unbenchmarked = [c for c in candidates if len(c.perfs) == 0]
            if unbenchmarked:
                self.parallel_benchmark_population(
                    unbenchmarked,
                    desc=f"Dim {dim_idx} ({spec.category().name})",
                )
            self.rebenchmark_population(
                candidates,
                desc=f"Dim {dim_idx}: verifying",
            )

            candidates.sort(key=performance)
            beam_members = [
                m for m in candidates[: self.beam_width] if math.isfinite(m.perf)
            ]
            self.population = list(beam_members)
            self.log(
                f"Dim {dim_idx} ({spec.category().name}): "
                f"{len(candidates)} candidates -> beam of {len(beam_members)}",
                self.statistics,
            )

            if not beam_members:
                raise exc.NoConfigFound

        # --- refinement phase ---
        for rnd in range(1, self.refinement_rounds + 1):
            current = self.best
            neighbor_flats = self._generate_neighbors(current.flat_values)
            new_members: list[PopulationMember] = []
            for nf in neighbor_flats:
                m = self.make_unbenchmarked(nf)
                if m.config not in visited:
                    visited.add(m.config)
                    new_members.append(m)
            if not new_members:
                self.log(f"Refinement round {rnd}: no new neighbors, stopping")
                break
            self.set_generation(len(ordered_dims) + rnd)
            self.parallel_benchmark_population(
                new_members, desc=f"Refinement round {rnd}"
            )
            self.population = [current, *new_members]
            self.rebenchmark_population(
                self.population, desc=f"Refinement {rnd}: verifying"
            )
            self.log(f"Refinement round {rnd} complete:", self.statistics)
            if self.best is current:
                self.log(f"Refinement round {rnd}: no improvement, stopping")
                break

        best = self.run_finishing_phase(self.best, self.finishing_rounds)
        return best.config
