from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
import random
from typing import TYPE_CHECKING
from typing import cast

from .. import exc
from .base_search import BenchmarkResult
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import check_population_consistency
from .base_search import performance
from .benchmark_provider import MultiShapeBenchmarkProvider
from .benchmark_provider import _compile_config_failure_source_hash
from .benchmark_provider import _MultiShapeAutotuneArgs
from .benchmark_provider import _unset_fn
from .effort_profile import PATTERN_SEARCH_DEFAULTS
from .effort_profile import FlashStructuralSearchConfig
from .pattern_search import InitialPopulationStrategy
from .pattern_search import PatternSearch
from .search_space_logger import canonical_config_id
from helion._dist_utils import sync_seed

# 17: a pipeline lane whose conditional candidate search finds no unmeasured
# config counts as exhausted instead of leaving the qualification incomplete.
CUTE_FLASH_LANE_POLICY_VERSION = 17
# Raw per-family product up to which the structural qualification enumerates
# the effective (normalized) CuTe-flash search space (0.4 s for 1024 at the
# d256 pair cell); larger spaces stay sampled.
FLASH_EXACT_SPACE_RAW_FLOOR = 1024
_FLASH_TERMINAL_REFINEMENT_SCHEMA_VERSION = 2
_FLASH_TERMINAL_REFINEMENT_POLICY_VERSION = 2
_FLASH_TERMINAL_COORDINATE_POLICY = "same_leaf_full_surface_normalized_coordinate_v2"
_FLASH_TERMINAL_REFINEMENT_TARGET_MS = 200.0
_FLASH_TERMINAL_CONFIRMATION_TARGET_MS = 5000.0
_FLASH_TERMINAL_MEASUREMENT_POLICY = "mirrored_rotating_batched_wall_v2"

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .._compiler.cute.autotune.config_generation_ext import (
        CuteConfigGenerationExtension,
    )
    from .._compiler.cute.cute_flash import FlashStructuralLeaf
    from ..autotuner.effort_profile import AutotuneEffortProfile
    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from .base_search import _AutotunableKernel
    from .benchmarking import MirroredBenchmarkTrace
    from .config_generation import ConfigGeneration
    from .config_generation import CoordinateNeighborProjection
    from .config_generation import FlatConfig

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    HAS_ML_DEPS = True
except ImportError as e:
    HAS_ML_DEPS = False
    _IMPORT_ERROR = e


def flash_terminal_measurement_is_valid(
    record: Mapping[str, object], *, allow_projection_rejected: bool = False
) -> bool:
    """Validate one normalized terminal structural-qualification measurement."""
    status = record.get("status")
    attempt_perf = record.get("attempt_perf")
    selection_perf = record.get("selection_perf")
    succeeded = status in {"ok", "deduplicated"}
    finite_perfs = bool(
        not isinstance(attempt_perf, bool)
        and isinstance(attempt_perf, (int, float))
        and math.isfinite(attempt_perf)
        and attempt_perf > 0
        and not isinstance(selection_perf, bool)
        and isinstance(selection_perf, (int, float))
        and math.isfinite(selection_perf)
        and selection_perf > 0
    )
    if succeeded:
        return finite_perfs
    if status in {"error", "timeout", "peer_compilation_fail"}:
        return attempt_perf is None and selection_perf is None
    return bool(
        allow_projection_rejected
        and status == "projection_rejected"
        and record.get("config_id") is None
        and record.get("projected_config_id") is None
        and attempt_perf is None
        and selection_perf is None
    )


def flash_terminal_refinement_result_is_valid(record: Mapping[str, object]) -> bool:
    """Validate a candidate result recorded by terminal coordinate refinement."""
    if flash_terminal_measurement_is_valid(record):
        return True
    return bool(
        record.get("status") in {"accuracy_error", "source_rejected"}
        and record.get("attempt_perf") is None
        and record.get("selection_perf") is None
    )


class LFBOPatternSearch(PatternSearch):
    """
    Batch Likelihood-Free Bayesian Optimization (LFBO) Pattern Search.

    This algorithm enhances PatternSearch by using a Random Forest classifier as a surrogate
    model to select which configurations to benchmark, reducing the number of
    kernel compilations and runs needed to find optimal configurations.
    It imposes a similarity penalty to encourage diverse config selection.

    Algorithm Overview:
        1. Generate an initial population (random or default) and benchmark all configurations
        2. Fit a Random Forest classifier to predict "good" vs "bad" configurations:
           - Configs with performance < quantile threshold are labeled as "good" (class 1)
           - Configs with performance >= quantile threshold are labeled as "bad" (class 0)
           - Weighted classification emphasize configs that are much better than the threshold
        3. For each generation:
           - Generate random neighbors around the current best configurations
           - Score all neighbors using the classifier's predicted probability of being "good"
           - Penalizes points that are similar to previously selected points
           - Selects points to benchmark via sequential greedy optimization
           - Retrain the classifier on all observed data (not incremental)
           - Update search trajectories based on new results

    The weighted classification model learns to identify which configs maximize
    expected improvement over the current best config. Compared to fitting a surrogate
    to fit the config performances themselves, since this method is based on classification,
    it can also learn from configs that timeout or have unacceptable accuracy.

    References:
    - Song, J., et al. (2022). "A General Recipe for Likelihood-free Bayesian Optimization."

    Args:
        kernel: The kernel to be autotuned.
        args: The arguments to be passed to the kernel during benchmarking.
        initial_population: Number of random configurations in initial population.
            Default from PATTERN_SEARCH_DEFAULTS. Ignored when using DEFAULT strategy.
        copies: Number of top configurations to run pattern search from.
            Full CuTe-flash searches qualify every structural leaf, give every
            live family one measured probe generation, then continue the best
            evidence-ranked parent families and every compound leaf.
            Default from PATTERN_SEARCH_DEFAULTS.
        max_generations: Maximum number of search iterations per copy.
            Default from PATTERN_SEARCH_DEFAULTS.
        min_improvement_delta: Early stopping threshold. Search stops if the relative
            improvement abs(best/current - 1) < min_improvement_delta.
            Default: 0.001 (0.1% improvement threshold).
        frac_selected: Fraction of generated neighbors to actually benchmark, after
            filtering by classifier score. Range: (0, 1]. Lower values reduce benchmarking
            cost but may miss good configurations. Default: 0.15.
        num_neighbors: Number of random neighbor configurations to generate around
            each search point per generation. Default: 300.
        radius: Maximum perturbation distance in configuration space. For power-of-two
            parameters, this is the max change in log2 space. For other parameters,
            this limits how many parameters can be changed. Default: 2.
        quantile: Threshold for labeling configs as "good" (class 1) vs "bad" (class 0).
            Configs with performance below this quantile are labeled as good.
            Range: (0, 1). Lower values create a more selective definition of "good".
            Default: 0.3 (top 30% are considered good).
        patience: Number of generations without improvement before stopping
            the search copy. Default: 2.
        similarity_penalty: Penalty for selecting points that are similar to points
            already selected in the batch. Default: 1.0.
        initial_population_strategy: Strategy for generating the initial population.
            FROM_RANDOM generates initial_population random configs.
            FROM_BEST_AVAILABLE uses cached configs from prior runs, and fills the
            remainder with random configs when best_available_pad_random is True.
            Can be overridden by HELION_AUTOTUNER_INITIAL_POPULATION env var.
    """

    # Keep old serialized/test-created search state usable; normal construction
    # replaces this with a per-instance list for CuTe flash.
    train_source_hashes: list[str | None] | None = None

    # Written by ``FlashStructuralQualification`` (absent until it ran); read
    # by ``_select_starting_paths`` and the lane helpers through ``getattr``.
    _flash_qualified_pipeline_lanes: dict[
        FlashStructuralLeaf, tuple[tuple[str, object], ...]
    ]
    _flash_qualified_compound_config_ids: dict[FlashStructuralLeaf, set[str]]
    _flash_parent_score_config_ids: set[str]

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        initial_population: int = PATTERN_SEARCH_DEFAULTS.initial_population,
        copies: int = PATTERN_SEARCH_DEFAULTS.copies,
        max_generations: int = PATTERN_SEARCH_DEFAULTS.max_generations,
        min_improvement_delta: float = 0.001,
        frac_selected: float = 0.10,
        num_neighbors: int = 300,
        radius: int = 2,
        quantile: float = 0.1,
        patience: int = 1,
        similarity_penalty: float = 1.0,
        initial_population_strategy: InitialPopulationStrategy | None = None,
        best_available_pad_random: bool = PATTERN_SEARCH_DEFAULTS.best_available_pad_random,
        num_neighbors_cap: int = -1,
        finishing_rounds: int = 0,
        polish_rounds: int = 10,
        compile_timeout_lower_bound: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_lower_bound,
        compile_timeout_quantile: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_quantile,
        flash_structural_search: FlashStructuralSearchConfig | None = None,
    ) -> None:
        if not HAS_ML_DEPS:
            raise exc.AutotuneError(
                "LFBOPatternSearch requires numpy and scikit-learn."
                "Install them with: pip install helion[surrogate]"
            ) from _IMPORT_ERROR

        super().__init__(
            kernel=kernel,
            args=args,
            initial_population=initial_population,
            copies=copies,
            max_generations=max_generations,
            min_improvement_delta=min_improvement_delta,
            initial_population_strategy=initial_population_strategy,
            best_available_pad_random=best_available_pad_random,
            num_neighbors_cap=num_neighbors_cap,
            finishing_rounds=finishing_rounds,
            compile_timeout_lower_bound=compile_timeout_lower_bound,
            compile_timeout_quantile=compile_timeout_quantile,
        )
        # Parallel to train_x/train_y: the live member behind each training
        # row (None for externally seeded rows). Used to sync labels with
        # rebenchmarked timings before each surrogate fit.
        self._train_members: list[PopulationMember | None] = []

        # Number of neighbors and how many to evalaute
        self.num_neighbors = num_neighbors
        self.radius = radius
        self.frac_selected = frac_selected
        self.patience = patience
        self.similarity_penalty = similarity_penalty
        self.polish_rounds = polish_rounds
        self.surrogate: RandomForestClassifier | None = None

        # Save training data
        self.train_x = []
        self.train_y = []
        self.train_configs: list[Config] | None = (
            [] if self.config_spec.cute_flash_search_enabled else None
        )
        self.train_source_hashes: list[str | None] | None = (
            [] if self.config_spec.cute_flash_search_enabled else None
        )
        self.quantile = quantile
        self.flash_structural_search = flash_structural_search
        self._flash_family_probe_path_limit = 0
        self._flash_promoted_path_limit = self.copies
        self._cute_flash_lane_policy_enabled = (
            self.config_spec.cute_flash_search_enabled
            and flash_structural_search is not None
        )
        if self._cute_flash_lane_policy_enabled:
            assert flash_structural_search is not None
            if (
                flash_structural_search.terminal_coordinate_rounds > 0
                and flash_structural_search.terminal_coordinate_beam_width > 0
                and self.settings.autotune_budget_seconds is None
                and self.settings.autotune_benchmark_fn is None
                and not isinstance(self.args, _MultiShapeAutotuneArgs)
            ):
                self._terminal_refinement_members = {}
            self._flash_promoted_path_limit = (
                self._flash_generation().flash_structural_starting_path_limit(
                    minimum=max(self.copies, flash_structural_search.starting_paths),
                    retained_families=flash_structural_search.retained_families,
                    retained_candidates_per_leaf=(
                        flash_structural_search.retained_candidates_per_leaf
                    ),
                )
            )
            self._flash_family_probe_path_limit = (
                self._flash_generation().flash_structural_family_probe_path_limit(
                    flash_structural_search.retained_families,
                    flash_structural_search.family_probe_generations,
                )
            )
            self.copies = max(
                self._flash_promoted_path_limit,
                self._flash_family_probe_path_limit,
            )

    def _algorithm_cache_policy(self) -> dict[str, object]:
        policy = super()._algorithm_cache_policy()
        policy.update(
            {
                # 2: benchmarked-only visited set, per-copy selection floor,
                # incumbent retention, and rebenchmark-synced surrogate labels
                # became unconditional.
                # 3: a full-neighborhood polish descent runs after the main
                # loop (see polish_rounds).
                "lfbo_version": 3,
                "polish_rounds": self.polish_rounds,
                "num_neighbors": self.num_neighbors,
                "radius": self.radius,
                "frac_selected": self.frac_selected,
                "quantile": self.quantile,
                "patience": self.patience,
                "similarity_penalty": self.similarity_penalty,
                "flash_structural_search": (
                    self.flash_structural_search
                    if self._cute_flash_lane_policy_enabled
                    else None
                ),
            }
        )
        if self._cute_flash_lane_policy_enabled:
            policy["cute_flash_lane_policy_version"] = CUTE_FLASH_LANE_POLICY_VERSION
            assert self.flash_structural_search is not None
            policy["cute_flash_terminal_coordinate_refinement"] = {
                "schema_version": _FLASH_TERMINAL_REFINEMENT_SCHEMA_VERSION,
                "policy_version": _FLASH_TERMINAL_REFINEMENT_POLICY_VERSION,
                "coordinate_policy": _FLASH_TERMINAL_COORDINATE_POLICY,
                "rounds": self.flash_structural_search.terminal_coordinate_rounds,
                "beam_width": (
                    self.flash_structural_search.terminal_coordinate_beam_width
                ),
                "radius": self.radius,
                "minimum_improvement_fraction": self.min_improvement_delta,
                "measurement_policy": _FLASH_TERMINAL_MEASUREMENT_POLICY,
                "round_target_ms": _FLASH_TERMINAL_REFINEMENT_TARGET_MS,
                "confirmation_target_ms": (_FLASH_TERMINAL_CONFIRMATION_TARGET_MS),
            }
            policy["cute_flash_starting_path_limit"] = self._flash_promoted_path_limit
            policy["cute_flash_family_probe_path_limit"] = (
                self._flash_family_probe_path_limit
            )
            policy["cute_flash_maximum_path_capacity"] = self.copies
        return policy

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        from ..runtime.settings import _env_get_int
        from ..runtime.settings import _get_initial_population_strategy

        assert profile.lfbo_pattern_search is not None
        strategy = _get_initial_population_strategy(
            profile.lfbo_pattern_search.initial_population_strategy,
            settings.autotune_initial_population_strategy,
        )
        return {
            "initial_population": profile.lfbo_pattern_search.initial_population,
            "copies": profile.lfbo_pattern_search.copies,
            "max_generations": profile.lfbo_pattern_search.max_generations,
            "initial_population_strategy": strategy,
            "best_available_pad_random": profile.lfbo_pattern_search.best_available_pad_random,
            "num_neighbors_cap": _env_get_int("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS", -1),
            "flash_structural_search": profile.flash_structural_search,
            **PopulationBasedSearch.get_kwargs_from_profile(profile, settings),
        }

    def seed_training_data(
        self,
        results: Sequence[BenchmarkResult],
    ) -> None:
        """Pre-populate the surrogate's training set with externally-benchmarked configs.

        Useful when an outer loop (e.g. a hybrid LLM+LFBO search) has already
        benchmarked configs and wants the LFBO surrogate to learn from them
        rather than starting from scratch. Failed configs (perf=inf) are
        kept since the surrogate's binary classifier learns from negatives too.
        """
        for result in results:
            try:
                flat_values = self.config_gen.flatten(result.config)
                encoded = self.config_gen.encode_config(flat_values)
            except Exception as e:
                self.log.debug(f"seed_training_data: skipping config: {e}")
                continue
            self._append_training_sample(
                encoded,
                result.perf,
                result.config,
                result.fn,
            )

    def _append_training_sample(
        self,
        encoded: object,
        perf: float,
        config: Config,
        fn: object,
        member: PopulationMember | None = None,
    ) -> None:
        self.train_x.append(encoded)
        self.train_y.append(perf)
        self._train_members.append(member)
        if self.train_configs is None:
            return
        if self.train_source_hashes is None:
            self.train_source_hashes = [None] * len(self.train_configs)
        self.train_configs.append(config)
        self.train_source_hashes.append(
            self.config_spec.backend.generated_source_hash(fn)
        )

    def _apply_effective_source_repairs(
        self,
        repairs: dict[Config, BenchmarkResult],
        current_members: Sequence[PopulationMember],
    ) -> None:
        super()._apply_effective_source_repairs(repairs, current_members)
        if self.train_configs is None:
            return
        if self.train_source_hashes is None:
            self.train_source_hashes = [None] * len(self.train_configs)
        for index, config in enumerate(self.train_configs):
            repair = repairs.get(config)
            if (
                repair is not None
                and index < len(self.train_y)
                and not math.isfinite(self.train_y[index])
            ):
                self.train_y[index] = repair.perf
                self.train_source_hashes[index] = (
                    self.config_spec.backend.generated_source_hash(repair.fn)
                )

    def _invalidate_rebenchmark_training_targets(
        self,
        failed_configs: set[Config],
        failed_source_hashes: set[str],
    ) -> None:
        if self.train_configs is None:
            return
        if self.train_source_hashes is None:
            self.train_source_hashes = [None] * len(self.train_configs)
        for index, (config, source_hash) in enumerate(
            zip(self.train_configs, self.train_source_hashes, strict=True)
        ):
            if index < len(self.train_y) and (
                config in failed_configs or source_hash in failed_source_hashes
            ):
                self.train_y[index] = math.inf

    def _sync_training_labels(self) -> None:
        """Refresh train_y from live members so rebenchmarked timings (which
        replace noisy one-shot measurements) are what the surrogate learns
        from. Rows without a live member keep their recorded label."""
        for index, member in enumerate(self._train_members):
            if member is None or index >= len(self.train_y):
                continue
            if member.perfs and math.isfinite(self.train_y[index]):
                self.train_y[index] = member.perf

    def _fit_surrogate(self) -> None:
        self._sync_training_labels()
        train_x = np.array(self.train_x)
        train_y = np.array(self.train_y)

        # Compute labels based on quantile threshold
        finite_mask = ~np.isinf(train_y)
        if finite_mask.any():
            # Compute quantile among finite performance values
            train_y_quantile = np.quantile(train_y[finite_mask], self.quantile)
            pos_mask: np.ndarray = train_y <= train_y_quantile
            train_labels: np.ndarray = 1.0 * (pos_mask)

            # Sample weights to emphasize configs that are much better than the threshold
            # Clip this difference to a small number (e.g. 1e-5) so that in the case that all perfs
            # are equal (and train_y_quantile - train_y = 0) we avoid dividing by zero.
            # Instead, we will have all sample weights = 1 for all positive points.
            pos_weights = np.maximum(1e-5, train_y_quantile - train_y) * train_labels
            normalizing_factor = np.mean(pos_weights[pos_mask])
            # Normalize weights so on average they are 1.0
            pos_weights = pos_weights / normalizing_factor
            # Weights for negative labels are 1.0
            sample_weight: np.ndarray = np.where(pos_mask, pos_weights, 1.0)
        else:
            # If all targets are inf, then all labels are 0 (except the first one)
            train_labels: np.ndarray = np.zeros(len(train_y))
            sample_weight: np.ndarray = np.ones(len(train_y))

        # Ensure we have at least 2 classes for the classifier
        # If all labels are the same, we need to handle this case
        if np.all(train_labels == train_labels[0]):
            self.log("All labels are identical, skip training surrogate.")
            self.surrogate = None
        else:
            self.log(
                f"Fitting surrogate: {len(train_x)} points, {len(train_y)} targets"
            )
            self.surrogate = RandomForestClassifier(
                criterion="log_loss",
                random_state=42,
                n_estimators=100,
                n_jobs=-1,
            )
            self.surrogate.fit(train_x, train_labels, sample_weight=sample_weight)
            assert len(self.surrogate.classes_) == 2

    def compute_leaf_similarity(
        self, surrogate: RandomForestClassifier, X_test: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix using leaf node co-occurrence.

        For RandomForest, two samples are similar if they land in the same leaf nodes
        across trees. This is the Jaccard similarity of their leaf assignments.

        Args:
            model: Fitted RandomForestClassifier
            X_test: Test samples (n_samples, n_features)

        Returns:
            similarity_matrix: (n_samples, n_samples) matrix where entry [i,j] is
                            the fraction of trees where samples i and j land in the same leaf
        """
        n_samples = X_test.shape[0]

        # Get leaf indices for each sample across all trees
        # leaf_indices shape: (n_samples, n_trees)
        leaf_indices = surrogate.apply(X_test)
        n_trees = leaf_indices.shape[1]

        # Compute similarity: fraction of trees where samples land in same leaf
        # This is equivalent to Jaccard similarity on the leaf assignments
        similarity_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            # Vectorized comparison: how many trees have same leaf as sample i
            same_leaf: np.ndarray = (
                leaf_indices == leaf_indices[i : i + 1, :]
            )  # (n_samples, n_trees)
            similarity_matrix[i, :] = same_leaf.sum(axis=1) / n_trees

        return similarity_matrix

    def _surrogate_select(
        self, candidates: list[PopulationMember], n_sorted: int
    ) -> list[PopulationMember]:
        """
        Select top candidates using the surrogate model with diversity-aware scoring.

        Uses sequential greedy selection to pick candidates that balance high predicted
        probability of being "good" (from the Random Forest classifier) with diversity
        (avoiding candidates too similar to already-selected ones).

        The selection process:
        1. Score each candidate using the surrogate's predicted probability of class 1 ("good")
        2. Compute pairwise similarity between candidates using leaf node co-occurrence
        3. Greedily select candidates one at a time:
           - First candidate: highest probability
           - Subsequent candidates: highest (probability - similarity_penalty * mean_similarity)
             where mean_similarity is the average similarity to already-selected candidates
        4. Return the top n_sorted candidates based on selection order

        If no surrogate model is available (e.g., all training labels were identical),
        candidates are scored randomly.

        Args:
            candidates: List of PopulationMember configurations to score and select from.
            n_sorted: Number of top candidates to return.

        Returns:
            List of the top n_sorted PopulationMember candidates, ordered by selection rank.
        """
        if n_sorted <= 0 or not candidates:
            return []

        # Score candidates
        candidate_X = np.array(
            [self.config_gen.encode_config(member.flat_values) for member in candidates]
        )

        n_samples = len(candidate_X)
        n_selected = min(n_sorted, n_samples)

        # Get predicted probabilities (higher = more likely to be good)
        surrogate: RandomForestClassifier | None = self.surrogate
        if surrogate is None:
            # If surrogate is None, scores are random
            with sync_seed(process_group_name=self.kernel.env.process_group_name):
                scores = [random.random() for _ in range(n_samples)]
            candidates_sorted = sorted(
                zip(candidates, scores, strict=True),
                key=operator.itemgetter(1),
            )[:n_selected]
            candidates_sorted = [member for member, _ in candidates_sorted]
        else:
            proba = np.asarray(surrogate.predict_proba(candidate_X))[:, 1]

            # Track the cumulative similarity to already-selected points and update
            # it incrementally. This preserves the original ranking while avoiding
            # the dense n_samples x n_samples similarity matrix.
            leaf_indices = surrogate.apply(candidate_X)
            n_trees = leaf_indices.shape[1]
            similarity_sums = np.zeros(n_samples)
            remaining_indices = list(range(n_samples))
            selected_indices: list[int] = []

            for _rank in range(n_selected):
                if selected_indices:
                    mean_similarities = similarity_sums[remaining_indices] / len(
                        selected_indices
                    )
                    proba_minus_similarity = (
                        proba[remaining_indices]
                        - self.similarity_penalty * mean_similarities
                    )
                else:
                    proba_minus_similarity = proba[remaining_indices]

                best_local_idx = int(np.argmax(proba_minus_similarity))
                best_global_idx = remaining_indices.pop(best_local_idx)
                selected_indices.append(best_global_idx)

                if len(selected_indices) == n_selected:
                    break

                same_leaf: np.ndarray = (
                    leaf_indices == leaf_indices[best_global_idx : best_global_idx + 1]
                )
                similarity_sums += same_leaf.sum(axis=1) / n_trees

            candidates_sorted = [candidates[idx] for idx in selected_indices]

        self.log.debug(
            f"Scoring {len(candidate_X)} neighbors, selecting {(n_selected / len(candidate_X)) * 100:.0f}% neighbors: {len(candidates_sorted)}"
        )

        return candidates_sorted

    def _autotune(self) -> Config:
        initial_population_name = self.initial_population_strategy.name
        self.log(
            f"Starting {self.__class__.__name__} with initial_population={initial_population_name},"
            f" copies={self.copies},"
            f" max_generations={self.max_generations},"
            f" similarity_penalty={self.similarity_penalty}"
        )
        visited: set[Config] = set()
        self.population = []
        for flat_config in self._generate_initial_population_flat():
            member = self.make_unbenchmarked(flat_config)
            if member is not None and member.config not in visited:
                visited.add(member.config)
                self.population.append(member)
        initial_population = list(self.population)
        self.set_generation(0)
        self.benchmark_initial_population(
            self.population,
            random_fallback_target=self._random_fallback_population_target(),
            visited=visited,
        )
        # Configs the budget never attempted are no evidence for the
        # structural qualification (they were dropped from the population).
        initial_population = [
            member for member in initial_population if member.status != "skipped"
        ]

        # Compute adaptive compile timeout based on initial population compile times
        self.set_adaptive_compile_timeout(
            self.population,
            min_seconds=self.compile_timeout_lower_bound,
            quantile=self.compile_timeout_quantile,
        )

        # again with higher accuracy
        self.rebenchmark_population(self.population, desc="Verifying initial results")
        check_population_consistency(
            self.population, process_group_name=self.kernel.env.process_group_name
        )
        # Snapshot compiler-seeded members so they survive the search-loop
        # pruning into the final-pick verification candidate pool.
        self.capture_compiler_seed_members(self.population)
        self.population.sort(key=performance)
        if not any(math.isfinite(member.perf) for member in self.population):
            raise exc.NoConfigFound

        # Save to training data
        for member in self.population:
            self._append_training_sample(
                self.config_gen.encode_config(member.flat_values),
                member.perf,
                member.config,
                member.fn,
                member=member,
            )

        # Fit model
        self._fit_surrogate()

        # Initial witnesses can underrate a family whose useful child settings
        # are non-default. Full CuTe-flash tuning qualifies every ordinary leaf
        # before transferring its best representatives to compound leaves and
        # promoting parent families.
        # Quick tuning keeps its historical generation budget.
        qualification_generations = self._run_flash_structural_qualification(
            visited,
            initial_population=initial_population,
        )
        first_main_generation = 1 + qualification_generations
        phase = self._autotune_metrics.search_phase_metrics
        if phase is not None and phase.get("completed") is not True:
            if self._autotune_budget_exceeded_across_ranks():
                return self._finalize()
            # Main search cannot replace missing qualification witnesses. Keep
            # the recorded failure and stop before promotion or final selection.
            failures = [
                (
                    f"passes={phase.get('qualification_passes_completed')}/"
                    f"{phase.get('qualification_passes_planned')}"
                )
            ]
            for key in (
                "schedule_anchor_complete",
                "compound_catalog_complete",
                "family_probe_complete",
            ):
                if phase.get(key) is False:
                    failures.append(f"{key}=False")
            for key in ("leaf_results", "clc_families", "compound_transfers"):
                rows = phase.get(key, [])
                assert isinstance(rows, list)
                for row in rows:
                    if row["complete"] is True:
                        continue
                    detail = (
                        f"{key}: family={row['family']}, "
                        f"compound_packet={row.get('compound_packet')}, "
                        f"softmax_disc={row['softmax_disc']}"
                    )
                    if key == "compound_transfers":
                        detail += (
                            " successful transfers="
                            f"{len(row['successful_transfer_config_ids'])}/"
                            f"{row['transfer_target_count']}"
                        )
                    failures.append(detail)
            raise exc.AutotuneError(
                "required CuTe flash structural qualification did not complete: "
                + "; ".join(failures)
            )
        if phase is not None:
            # A complete qualification's lanes carry their metrics; the lanes
            # the conditional search exhausted (no unmeasured candidate within
            # reach of a measured parent) are logged once here.
            leaf_rows = phase.get("leaf_results", [])
            assert isinstance(leaf_rows, list)
            exhausted_lanes = []
            for row in leaf_rows:
                lanes = row["pipeline_lanes"]
                assert isinstance(lanes, list)
                exhausted_lanes.extend(
                    f"{row['family']}:{lane['key']}={lane['value']!r}"
                    for lane in lanes
                    if lane["conditional_exhausted"]
                )
            if exhausted_lanes:
                self.log(
                    "Structural qualification: no unmeasured conditional candidate "
                    "is reachable in "
                    f"{', '.join(exhausted_lanes)}; the lane(s) count as exhausted."
                )

        starting_paths = self._select_starting_paths()
        starting_points = [member for member, _constraints in starting_paths]
        self.log(
            f"Qualified population of {len(self.population)}, "
            f"{len(starting_points)} retained search paths:",
            self.statistics,
        )
        if not starting_points:
            raise exc.NoConfigFound
        if self._autotune_metrics.search_phase_metrics is not None:
            self._autotune_metrics.search_phase_metrics["retained_path_count"] = len(
                starting_paths
            )

        search_copies = []
        for idx, (member, constraints) in enumerate(starting_paths):
            required_leaf = self._flash_structural_leaf(member) if constraints else None
            # A retained leaf owns a conditional field set. Generating from the
            # global union mostly produces inactive aliases and silently spends
            # this path's bounded generations without exploring its children.
            search_copies.append(
                self._pruned_pattern_search_from(
                    idx,
                    member,
                    visited,
                    constraints,
                    required_leaf=required_leaf,
                    conditional_surface=required_leaf is not None,
                    disable_early_stopping=self._path_exhausts_generation_budget(
                        constraints
                    ),
                )
            )

        for generation in self._budgeted_range(
            first_main_generation, self.max_generations + 1
        ):
            prior_best = self.best
            new_population = {id(prior_best): prior_best}
            num_neighbors = 0
            num_active = 0
            for search_copy in search_copies:
                added = next(search_copy, ())
                if added:
                    assert len(added) > 1
                    num_active += 1
                    num_neighbors += len(added) - 1
                    for member in added:
                        new_population[id(member)] = member
            if num_active == 0:
                self.log(
                    f"Autotuning stop at generation {generation} because of no active search path"
                )
                break

            # Log generation header before compiling/benchmarking
            self.log(
                f"Generation {generation} starting: {num_neighbors} neighbors, {num_active} active search path(s)"
            )

            self.population = [*new_population.values()]
            # compile any unbenchmarked members in parallel
            unbenchmarked = [m for m in self.population if len(m.perfs) == 0]
            if unbenchmarked:
                self.set_generation(generation)
                self.benchmark_population(
                    unbenchmarked, desc=f"Generation {generation}:"
                )
            # higher-accuracy rebenchmark
            self.rebenchmark_population(
                self.population, desc=f"Generation {generation}: verifying top configs"
            )
            # Log final statistics for this generation
            self.log(f"Generation {generation} complete:", self.statistics)

            # no need to retrain the model for the last generation
            if generation != self.max_generations:
                # Update training data with newly benchmarked members only
                for member in unbenchmarked:
                    self._append_training_sample(
                        self.config_gen.encode_config(member.flat_values),
                        member.perf,
                        member.config,
                        member.fn,
                        member=member,
                    )
                # Fit model
                self._fit_surrogate()

        self._polish_descent(visited)
        # Final verification, finishing phase, and (TPU-only) final-pick re-rank.
        return self._finalize()

    def _polish_descent(self, visited: set[Config]) -> None:
        """Full-neighborhood descent after the surrogate-guided main loop.

        Run plain pattern-search descent from the incumbent: benchmark the
        *entire* deterministic radius-1 neighborhood (no surrogate pruning)
        and move to the best neighbor until a round fails to improve, up to
        ``polish_rounds`` rounds. Recovers local wins the surrogate's
        selection fraction skipped, at a bounded extra eval cost.
        """
        rounds = self.polish_rounds
        if rounds <= 0 or self.config_spec.cute_flash_search_enabled:
            return
        current = self.best
        for round_num in self._budgeted_range(1, rounds + 1):
            candidates = [current]
            for flat_config in PatternSearch._generate_neighbors(
                self, current.flat_values
            ):
                member = self.make_unbenchmarked(flat_config)
                if member is not None and member.config not in visited:
                    visited.add(member.config)
                    candidates.append(member)
            if len(candidates) <= 1:
                self.log(f"Polish round {round_num}: no unvisited neighbors")
                break
            self.set_generation(self._autotune_metrics.num_generations + 1)
            self.benchmark_population(candidates[1:], desc=f"Polish round {round_num}")
            self.rebenchmark_population(
                candidates, desc=f"Polish round {round_num}: verifying"
            )
            self.population.extend(candidates[1:])
            best = min(candidates, key=performance)
            self.log(
                f"Polish round {round_num}: {len(candidates) - 1} neighbors, "
                f"best {self.format_performance(best.perf)}"
            )
            if self._check_early_stopping(best, current):
                break
            current = best

    def _flash_generation(
        self, generation: ConfigGeneration | None = None
    ) -> CuteConfigGenerationExtension:
        """The CuTe generation extension the flash lane policy reads (the
        policy runs only under the CuTe flash surface)."""
        return cast(
            "CuteConfigGenerationExtension",
            (self.config_gen if generation is None else generation).extension,
        )

    @staticmethod
    def _flash_structural_leaf(
        member: PopulationMember,
    ) -> FlashStructuralLeaf | None:
        from .._compiler.cute.cute_flash import flash_structural_leaf_from_config

        return flash_structural_leaf_from_config(member.config.config)

    @staticmethod
    def _flash_leaf_constraints(
        leaf: FlashStructuralLeaf,
    ) -> tuple[tuple[str, object], ...]:
        from .._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
        from .._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
        from .._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY

        constraints: list[tuple[str, object]] = [
            (FLASH_PIPELINE_FAMILY_KEY, leaf.pipeline_family),
            (FLASH_SOFTMAX_DISC_KEY, leaf.softmax_disc),
        ]
        if leaf.compound_exp2_packet is not None:
            constraints.append((FLASH_EXP2_PACKET_KEY, leaf.compound_exp2_packet))
        return tuple(constraints)

    @staticmethod
    def _flash_member_rank_key(member: PopulationMember) -> tuple[float, str]:
        """Order equal-performing flash configs reproducibly."""
        return performance(member), canonical_config_id(member.config)

    @staticmethod
    def _flash_member_succeeded(member: PopulationMember) -> bool:
        return (
            member.status in {"ok", "deduplicated"}
            and bool(member.perfs)
            and math.isfinite(member.perfs[0])
            and math.isfinite(member.perf)
        )

    @staticmethod
    def _flash_member_has_retryable_failure(member: PopulationMember) -> bool:
        """Return whether a structural alternate may repair this failed attempt."""
        return member.status in {"error", "timeout", "peer_compilation_fail"}

    def _flash_member_source_hash(self, member: PopulationMember) -> str | None:
        """Return the effective generated-source identity for one flash member."""
        if member.fn is _unset_fn and member.status == "error":
            return _compile_config_failure_source_hash(member.config)
        return self.config_spec.backend.generated_source_hash(member.fn)

    @staticmethod
    def _flash_clc_combination_statuses_allowed(
        cells: Sequence[Mapping[str, object]],
    ) -> bool:
        """Reject correctness and policy failures hidden by marginal coverage."""
        return all(
            flash_terminal_measurement_is_valid(cell, allow_projection_rejected=True)
            for cell in cells
        )

    @staticmethod
    def _flash_pipeline_qualification_keys() -> tuple[str, str]:
        from .._compiler.cute.cute_flash import FLASH_KV_STAGE_KEY
        from .._compiler.cute.cute_flash import FLASH_S_STAGE_KEY

        # KV depth controls the long producer pipeline and is the primary axis.
        # S depth is still qualified when it has a live family-conditional choice.
        return FLASH_KV_STAGE_KEY, FLASH_S_STAGE_KEY

    def _flash_pipeline_lanes(
        self,
        leaf: FlashStructuralLeaf,
    ) -> tuple[tuple[str, object], ...]:
        """Return ConfigGeneration's exact normalized depth catalog for ``leaf``."""
        return self._flash_generation().flash_pipeline_lane_catalog().get(leaf, ())

    @staticmethod
    def _flash_member_matches_pipeline_lane(
        member: PopulationMember, lane: tuple[str, object]
    ) -> bool:
        key, value = lane
        return member.config.config.get(key) == value

    @staticmethod
    def _flash_pipeline_lane_metric(
        lane: tuple[str, object] | None,
    ) -> dict[str, object] | None:
        if lane is None:
            return None
        return {"key": lane[0], "value": lane[1]}

    def _flash_lane_diverse_members(
        self,
        members: Sequence[PopulationMember],
        lanes: Sequence[tuple[str, object]],
        limit: int,
    ) -> list[tuple[PopulationMember, tuple[str, object] | None]]:
        """Retain fast members while covering primary depth values first."""
        remaining = sorted(members, key=self._flash_member_rank_key)
        if limit <= 0 or not remaining:
            return []
        selected: list[tuple[PopulationMember, tuple[str, object] | None]] = [
            (remaining.pop(0), None)
        ]
        covered = {
            lane
            for lane in lanes
            if self._flash_member_matches_pipeline_lane(selected[0][0], lane)
        }
        key_order = self._flash_pipeline_qualification_keys()
        while remaining and len(selected) < limit:

            def rank(member: PopulationMember) -> tuple[object, ...]:
                newly_covered = {
                    lane
                    for lane in lanes
                    if lane not in covered
                    and self._flash_member_matches_pipeline_lane(member, lane)
                }
                coverage_by_key = tuple(
                    -sum(lane[0] == key for lane in newly_covered) for key in key_order
                )
                return (
                    *coverage_by_key,
                    self._flash_member_rank_key(member),
                )

            member = min(remaining, key=rank)
            remaining.remove(member)
            newly_covered = [
                lane
                for lane in lanes
                if lane not in covered
                and self._flash_member_matches_pipeline_lane(member, lane)
            ]
            assigned_lane = newly_covered[0] if newly_covered else None
            selected.append((member, assigned_lane))
            covered.update(
                lane
                for lane in lanes
                if self._flash_member_matches_pipeline_lane(member, lane)
            )
        return selected

    def _flash_pipeline_lane_witness(
        self,
        leaf: FlashStructuralLeaf,
        lane: tuple[str, object],
    ) -> PopulationMember | None:
        """Create the deterministic normalized candidate for a missing lane."""
        config = (
            self._flash_generation()
            .flash_pipeline_lane_witnesses()
            .get((leaf, lane[0], lane[1]))
        )
        if config is None:
            return None
        global_flat = self.config_gen.flatten(config)
        global_flat, global_config = self.config_gen.canonicalize_flat(global_flat)
        assert self._flash_structural_leaf_from_config(global_config) == leaf
        assert global_config.config.get(lane[0]) == lane[1]
        return self.make_unbenchmarked(global_flat)

    def _flash_clc_lane_witness(
        self,
        leaf: FlashStructuralLeaf,
        value: int,
    ) -> PopulationMember | None:
        """Create the deterministic normalized candidate for a CLC divisor."""
        config = self._flash_generation().flash_clc_lane_witnesses().get((leaf, value))
        if config is None:
            return None
        global_flat = self.config_gen.flatten(config)
        global_flat, global_config = self.config_gen.canonicalize_flat(global_flat)
        assert self._flash_structural_leaf_from_config(global_config) == leaf
        from .._compiler.cute.cute_flash import FLASH_CLC_HEADS_PER_BATCH_KEY

        assert global_config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY) == value
        return self.make_unbenchmarked(global_flat)

    def _flash_config_variant(
        self,
        member: PopulationMember,
        overrides: Mapping[str, object],
        *,
        expected_leaf: FlashStructuralLeaf,
    ) -> PopulationMember | None:
        """Normalize a structural transfer while preserving the source fields."""
        config = copy.deepcopy(member.config)
        config.config.update(overrides)
        try:
            global_flat = self.config_gen.flatten(config)
            global_flat, global_config = self.config_gen.canonicalize_flat(global_flat)
        except exc.InvalidConfig:
            return None
        if self._flash_structural_leaf_from_config(global_config) != expected_leaf:
            return None
        return self.make_unbenchmarked(global_flat)

    def _flash_pipeline_values_survive(
        self,
        source: PopulationMember,
        candidate: PopulationMember,
    ) -> bool:
        """Return whether normalization preserved the source pipeline depths."""
        return all(
            key not in source.config.config
            or candidate.config.config.get(key) == source.config.config[key]
            for key in self._flash_pipeline_qualification_keys()
        )

    def _flash_clc_depth_variant(
        self,
        depth_member: PopulationMember,
        value: int,
        *,
        expected_leaf: FlashStructuralLeaf,
    ) -> PopulationMember | None:
        """Apply a CLC divisor without changing the selected depth schedule."""
        from .._compiler.cute.cute_flash import FLASH_CLC_HEADS_PER_BATCH_KEY

        candidate = self._flash_config_variant(
            depth_member,
            {FLASH_CLC_HEADS_PER_BATCH_KEY: value},
            expected_leaf=expected_leaf,
        )
        if (
            candidate is None
            or candidate.config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY) != value
            or not self._flash_pipeline_values_survive(depth_member, candidate)
        ):
            return None
        return candidate

    def _flash_compound_variant(
        self,
        source: PopulationMember,
        packet: object,
        *,
        expected_leaf: FlashStructuralLeaf,
    ) -> PopulationMember | None:
        """Apply a compound packet only when source pipeline depths survive."""
        from .._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY

        candidate = self._flash_config_variant(
            source,
            {FLASH_EXP2_PACKET_KEY: packet},
            expected_leaf=expected_leaf,
        )
        if candidate is None or not self._flash_pipeline_values_survive(
            source, candidate
        ):
            return None
        return candidate

    @staticmethod
    def _flash_lane_qualification_passes(
        lanes: Sequence[tuple[str, object]],
        *,
        candidate_limit: int,
        conditional_candidates_per_lane: int,
        minimum_passes: int,
        conditional_lanes: Sequence[tuple[str, object]] | None = None,
    ) -> list[list[tuple[str, tuple[str, object] | None]]]:
        """Build dependency-ordered witness and conditional qualification passes."""
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        if conditional_candidates_per_lane < 0:
            raise ValueError("conditional_candidates_per_lane must be nonnegative")
        if not lanes:
            return [[("ordinary", None)] for _ in range(minimum_passes)]

        passes: list[list[tuple[str, tuple[str, object] | None]]] = []
        witness_jobs: list[tuple[str, tuple[str, object] | None]] = [
            ("witness", lane) for lane in lanes
        ]
        conditional_jobs: list[tuple[str, tuple[str, object] | None]] = [
            ("conditional", lane)
            for _ in range(conditional_candidates_per_lane)
            for lane in lanes
            if conditional_lanes is None or lane in conditional_lanes
        ]
        for jobs in (witness_jobs, conditional_jobs):
            passes.extend(
                jobs[offset : offset + candidate_limit]
                for offset in range(0, len(jobs), candidate_limit)
            )
        passes.extend([] for _ in range(max(0, minimum_passes - len(passes))))
        return passes

    @staticmethod
    def _flash_lane_neighbor_limits(
        quotas: Sequence[tuple[tuple[str, object] | None, int]],
        total_neighbors: int,
    ) -> list[int]:
        """Divide one historical neighbor-generation budget across lanes."""
        total_quota = sum(quota for _lane, quota in quotas)
        if total_quota <= 0:
            return [0] * len(quotas)
        limits: list[int] = []
        cumulative = 0
        for _lane, quota in quotas:
            start = cumulative * total_neighbors // total_quota
            cumulative += quota
            limits.append(cumulative * total_neighbors // total_quota - start)
        return limits

    def _flash_qualification_neighbor_limit(self) -> int:
        """Return the one effective raw-neighbor budget shared by a leaf."""
        if self.num_neighbors_cap > 0:
            return min(self.num_neighbors, self.num_neighbors_cap)
        return self.num_neighbors

    def _flash_family_probe_paths(
        self, population: Sequence[PopulationMember]
    ) -> list[
        tuple[
            PopulationMember,
            FlashStructuralLeaf | None,
            tuple[tuple[str, object], ...],
            bool,
        ]
    ]:
        """Select one measured probe start per family/compound leaf and globally."""
        eligible = [
            member
            for member in population
            if self._flash_member_succeeded(member)
            and self._flash_structural_leaf(member) is not None
        ]
        if not eligible:
            return []

        ordinary_by_family: dict[
            str, list[tuple[PopulationMember, FlashStructuralLeaf]]
        ] = {}
        compound_by_leaf: dict[FlashStructuralLeaf, list[PopulationMember]] = {}
        probe_eligible: list[PopulationMember] = []
        qualified_compound_config_ids = getattr(
            self, "_flash_qualified_compound_config_ids", {}
        )
        for member in eligible:
            leaf = self._flash_structural_leaf(member)
            assert leaf is not None
            if leaf.compound_exp2_packet is None:
                probe_eligible.append(member)
                ordinary_by_family.setdefault(leaf.pipeline_family, []).append(
                    (member, leaf)
                )
            elif canonical_config_id(
                member.config
            ) in qualified_compound_config_ids.get(leaf, set()):
                probe_eligible.append(member)
                compound_by_leaf.setdefault(leaf, []).append(member)

        paths: list[
            tuple[
                PopulationMember,
                FlashStructuralLeaf | None,
                tuple[tuple[str, object], ...],
                bool,
            ]
        ] = []
        family_starts = [
            min(members, key=lambda item: self._flash_member_rank_key(item[0]))
            for members in ordinary_by_family.values()
        ]
        for member, leaf in sorted(
            family_starts,
            key=lambda item: (
                self._flash_member_rank_key(item[0]),
                item[1].pipeline_family,
            ),
        ):
            paths.append((member, leaf, self._flash_leaf_constraints(leaf), False))

        for leaf, members in sorted(
            compound_by_leaf.items(),
            key=lambda item: (
                self._flash_member_rank_key(
                    min(item[1], key=self._flash_member_rank_key)
                ),
                item[0].pipeline_family,
                item[0].compound_exp2_packet or "",
                item[0].softmax_disc,
            ),
        ):
            member = min(members, key=self._flash_member_rank_key)
            paths.append((member, leaf, self._flash_leaf_constraints(leaf), False))

        if not probe_eligible:
            return []
        global_best = min(probe_eligible, key=self._flash_member_rank_key)
        paths.append((global_best, None, (), True))
        return paths

    def _run_flash_structural_qualification(
        self,
        visited: set[Config],
        *,
        initial_population: Sequence[PopulationMember] | None = None,
    ) -> int:
        """Qualify ordinary schedules, then transfer their best representatives.

        The phase lives in ``FlashStructuralQualification``; this seam stays a
        method so a search can be built with it mocked out.
        """
        from .._compiler.cute.autotune.flash_search import FlashStructuralQualification

        return FlashStructuralQualification(
            self, visited, initial_population=initial_population
        ).run()

    def _path_exhausts_generation_budget(
        self, constraints: tuple[tuple[str, object], ...]
    ) -> bool:
        """Keep the full-search winner path alive through the bounded budget."""
        policy = getattr(self, "flash_structural_search", None)
        return bool(
            self.config_spec.cute_flash_search_enabled
            and policy is not None
            and policy.exhaust_unrestricted_path
            and not constraints
        )

    def _select_starting_paths(
        self,
    ) -> list[tuple[PopulationMember, tuple[tuple[str, object], ...]]]:
        """Select starts and structural values each qualification path must retain."""
        if not self.config_spec.cute_flash_search_enabled:
            return [
                (member, ())
                for member in self.population[: self.copies]
                if math.isfinite(member.perf)
            ]
        eligible = [member for member in self.population if math.isfinite(member.perf)]
        selection_limit = getattr(self, "_flash_promoted_path_limit", self.copies)
        if selection_limit <= 1:
            return [(member, ()) for member in eligible[:selection_limit]]

        policy = getattr(self, "flash_structural_search", None)
        if policy is not None:
            path_limit = selection_limit
            by_leaf: dict[FlashStructuralLeaf, list[PopulationMember]] = {}
            qualified_compound_config_ids = getattr(
                self, "_flash_qualified_compound_config_ids", None
            )
            for member in eligible:
                if not self._flash_member_succeeded(member):
                    continue
                leaf = self._flash_structural_leaf(member)
                if (
                    leaf is not None
                    and leaf.compound_exp2_packet is not None
                    and qualified_compound_config_ids is not None
                    and canonical_config_id(member.config)
                    not in qualified_compound_config_ids.get(leaf, set())
                ):
                    continue
                if leaf is not None:
                    by_leaf.setdefault(leaf, []).append(member)
            for members in by_leaf.values():
                members.sort(key=self._flash_member_rank_key)

            qualified_pipeline_lanes = getattr(
                self, "_flash_qualified_pipeline_lanes", {}
            )
            leaf_pipeline_lanes = {
                leaf: (
                    qualified_pipeline_lanes[leaf]
                    if leaf in qualified_pipeline_lanes
                    else self._flash_pipeline_lanes(leaf)
                )
                for leaf in by_leaf
            }
            retained_by_leaf = {
                leaf: self._flash_lane_diverse_members(
                    members,
                    leaf_pipeline_lanes[leaf],
                    policy.retained_candidates_per_leaf,
                )
                for leaf, members in by_leaf.items()
            }

            family_queues: dict[
                str,
                list[
                    tuple[
                        PopulationMember,
                        FlashStructuralLeaf,
                        tuple[str, object] | None,
                    ]
                ],
            ] = {}
            family_score_entries: dict[
                str, tuple[PopulationMember, FlashStructuralLeaf]
            ] = {}
            parent_score_families: set[str] = set()
            parent_score_config_ids = getattr(
                self, "_flash_parent_score_config_ids", None
            )
            families = list(
                dict.fromkeys(leaf.pipeline_family for leaf in by_leaf if by_leaf[leaf])
            )
            for family in families:
                leaves = [
                    leaf
                    for leaf in by_leaf
                    if leaf.pipeline_family == family and by_leaf[leaf]
                ]
                queue: list[
                    tuple[
                        PopulationMember,
                        FlashStructuralLeaf,
                        tuple[str, object] | None,
                    ]
                ] = []
                for rank in range(policy.retained_candidates_per_leaf):
                    layer = [
                        (*retained_by_leaf[leaf][rank], leaf)
                        for leaf in leaves
                        if rank < len(retained_by_leaf[leaf])
                    ]
                    queue.extend(
                        sorted(
                            ((member, leaf, lane) for member, lane, leaf in layer),
                            key=lambda item: (
                                performance(item[0]),
                                item[1].compound_exp2_packet or "",
                                item[1].softmax_disc,
                                canonical_config_id(item[0].config),
                            ),
                        )
                    )
                if queue:
                    family_queues[family] = queue
                    ordinary: list[tuple[PopulationMember, FlashStructuralLeaf]] = []
                    for leaf in leaves:
                        if leaf.compound_exp2_packet is not None:
                            continue
                        score_members = [
                            member
                            for member in by_leaf[leaf]
                            if parent_score_config_ids is None
                            or canonical_config_id(member.config)
                            in parent_score_config_ids
                        ]
                        if score_members:
                            ordinary.append((score_members[0], leaf))
                    if ordinary:
                        parent_score_families.add(family)
                    family_score_entries[family] = min(
                        ordinary or [(queue[0][0], queue[0][1])],
                        key=lambda item: (
                            performance(item[0]),
                            canonical_config_id(item[0].config),
                        ),
                    )

            ranked_families = sorted(
                parent_score_families,
                key=lambda family: (
                    performance(family_score_entries[family][0]),
                    family,
                ),
            )
            if not family_queues:
                phase = getattr(
                    getattr(self, "_autotune_metrics", None),
                    "search_phase_metrics",
                    None,
                )
                if phase is not None:
                    phase["retained_families"] = []
                return [(member, ()) for member in eligible[:1]]
            # Score every parent by its ordinary leaf. Compound packets receive
            # only transferred ordinary representatives and can earn a leaf path
            # below, but a family with more packet variants gets no
            # multiple-comparison advantage during parent promotion.
            competitive_families: list[str] = []
            if policy.retained_families is None:
                # Full effort continues every qualified live family. A family
                # with a weak pipeline-only witness may still own the best
                # arithmetic settings, so qualification rank cannot prune it.
                competitive_families = ranked_families
            elif ranked_families:
                best_family_perf = performance(
                    family_score_entries[ranked_families[0]][0]
                )
                competitive_families = [
                    family
                    for family in ranked_families
                    if performance(family_score_entries[family][0])
                    <= best_family_perf * policy.retained_family_slowdown_limit
                ]
            best_leaf_entry = min(
                ((by_leaf[leaf][0], leaf) for leaf in by_leaf if by_leaf[leaf]),
                key=lambda item: (
                    performance(item[0]),
                    item[1].pipeline_family,
                    item[1].compound_exp2_packet or "",
                    item[1].softmax_disc,
                    canonical_config_id(item[0].config),
                ),
            )
            best_member, best_leaf = best_leaf_entry
            best_family = best_leaf.pipeline_family
            alternate_leaf_order = [best_leaf]
            family_score_leaf = family_score_entries[best_family][1]
            if family_score_leaf != best_leaf:
                alternate_leaf_order.append(family_score_leaf)
            alternate_leaf_order.extend(
                leaf
                for leaf in sorted(
                    (
                        leaf
                        for leaf in by_leaf
                        if leaf.pipeline_family == best_family
                        and leaf not in alternate_leaf_order
                    ),
                    key=lambda leaf: self._flash_member_rank_key(by_leaf[leaf][0]),
                )
            )
            best_lane_alternate = next(
                (
                    (member, leaf, lane)
                    for leaf in alternate_leaf_order
                    for member, lane in retained_by_leaf[leaf]
                    if member.config != best_member.config and lane is not None
                ),
                None,
            )
            # Reserve one path for the unrestricted winner. It is appended last
            # so its global neighbor generation cannot consume candidates before
            # the constrained structural paths have selected theirs.
            constrained_limit = max(0, path_limit - 1)
            retained_parent_families = competitive_families[
                : min(
                    len(competitive_families)
                    if policy.retained_families is None
                    else policy.retained_families,
                    constrained_limit,
                )
            ]
            selected_leaf_paths: list[
                tuple[
                    PopulationMember,
                    FlashStructuralLeaf,
                    bool,
                    tuple[str, object] | None,
                ]
            ] = []
            for family in retained_parent_families:
                member, leaf = family_score_entries[family]
                if len(selected_leaf_paths) >= constrained_limit:
                    break
                selected_leaf_paths.append((member, leaf, False, None))
            selected_configs = {
                member.config
                for member, _leaf, _unrestricted, _lane in selected_leaf_paths
            }
            selected_leaves = {
                leaf for _member, leaf, _unrestricted, _lane in selected_leaf_paths
            }

            # Retain every ordinary protocol in the promoted families before
            # spending a path on a compound packet.
            ordinary_leaf_candidates = sorted(
                (
                    (by_leaf[leaf][0], leaf)
                    for family in retained_parent_families
                    for leaf in by_leaf
                    if leaf.pipeline_family == family
                    and leaf.compound_exp2_packet is None
                    and leaf not in selected_leaves
                ),
                key=lambda item: (
                    performance(item[0]),
                    item[1].pipeline_family,
                    item[1].softmax_disc,
                    canonical_config_id(item[0].config),
                ),
            )
            for member, leaf in ordinary_leaf_candidates:
                if len(selected_leaf_paths) >= constrained_limit:
                    break
                if member.config in selected_configs:
                    continue
                selected_leaf_paths.append((member, leaf, False, None))
                selected_configs.add(member.config)
                selected_leaves.add(leaf)

            # Keep one measured alternate pipeline lane for the global winner's
            # family. This reservation precedes all compound continuations.
            if (
                best_lane_alternate is not None
                and best_family in retained_parent_families
                and len(selected_leaf_paths) < constrained_limit
            ):
                alternate_member, alternate_leaf, alternate_lane = best_lane_alternate
                if alternate_member.config not in selected_configs:
                    selected_leaf_paths.append(
                        (alternate_member, alternate_leaf, False, alternate_lane)
                    )
                    selected_configs.add(alternate_member.config)
                    selected_leaves.add(alternate_leaf)

            # Qualification ranking is noisy enough that the measured leading
            # family may not own the best arithmetic basin. Give every promoted
            # family one available ordinary secondary before allocating paths to
            # compound schedules. The global-family lane alternate above counts
            # when it is already a secondary for its ordinary leaf.
            families_with_ordinary_secondary = {
                leaf.pipeline_family
                for member, leaf, _unrestricted, _lane in selected_leaf_paths
                if leaf.compound_exp2_packet is None
                and member.config != by_leaf[leaf][0].config
            }
            for family in retained_parent_families:
                if (
                    len(selected_leaf_paths) >= constrained_limit
                    or family in families_with_ordinary_secondary
                ):
                    continue
                secondary = next(
                    (
                        (member, leaf, lane)
                        for member, leaf, lane in family_queues[family]
                        if leaf.compound_exp2_packet is None
                        and member.config != by_leaf[leaf][0].config
                        and member.config not in selected_configs
                    ),
                    None,
                )
                if secondary is None:
                    continue
                member, leaf, lane = secondary
                selected_leaf_paths.append((member, leaf, False, lane))
                selected_configs.add(member.config)
                selected_leaves.add(leaf)
                families_with_ordinary_secondary.add(family)

            compound_leaf_candidates = sorted(
                (
                    (by_leaf[leaf][0], leaf)
                    for leaf in by_leaf
                    if leaf.compound_exp2_packet is not None
                    and leaf not in selected_leaves
                ),
                key=lambda item: (
                    performance(item[0]),
                    item[1].pipeline_family,
                    item[1].compound_exp2_packet or "",
                    item[1].softmax_disc,
                    canonical_config_id(item[0].config),
                ),
            )
            for member, leaf in compound_leaf_candidates:
                if len(selected_leaf_paths) >= constrained_limit:
                    break
                if member.config in selected_configs:
                    continue
                selected_leaf_paths.append((member, leaf, False, None))
                selected_configs.add(member.config)
                selected_leaves.add(leaf)

            queue_offsets = dict.fromkeys(retained_parent_families, 0)
            while len(selected_leaf_paths) < constrained_limit:
                added = False
                for family in retained_parent_families:
                    queue = family_queues[family]
                    offset = queue_offsets[family]
                    while (
                        offset < len(queue)
                        and queue[offset][0].config in selected_configs
                    ):
                        offset += 1
                    queue_offsets[family] = offset
                    if offset >= len(queue):
                        continue
                    item = queue[offset]
                    queue_offsets[family] += 1
                    member, leaf, lane = item
                    selected_leaf_paths.append((member, leaf, False, lane))
                    selected_configs.add(item[0].config)
                    added = True
                    if len(selected_leaf_paths) >= constrained_limit:
                        break
                if not added:
                    break

            if path_limit > 0:
                selected_leaf_paths.append((best_member, best_leaf, True, None))

            paths = [
                (
                    member,
                    ()
                    if unrestricted
                    else (
                        *self._flash_leaf_constraints(leaf),
                        *((lane,) if lane is not None else ()),
                    ),
                )
                for member, leaf, unrestricted, lane in selected_leaf_paths
            ]
            metrics = getattr(self, "_autotune_metrics", None)
            phase = (
                getattr(metrics, "search_phase_metrics", None)
                if metrics is not None
                else None
            )
            if phase is not None:
                reported_families = list(
                    dict.fromkeys(
                        (
                            best_family,
                            *retained_parent_families,
                            *(
                                leaf.pipeline_family
                                for _member, leaf, _unrestricted, _lane in selected_leaf_paths
                            ),
                        )
                    )
                )
                phase["retained_families"] = [
                    {
                        "family": family,
                        "score": family_score_entries[family][0].perf,
                        "score_compound_packet": family_score_entries[family][
                            1
                        ].compound_exp2_packet,
                        "score_softmax_disc": family_score_entries[family][
                            1
                        ].softmax_disc,
                        "parent_promoted": family in retained_parent_families,
                        "starting_paths": [
                            {
                                "family": leaf.pipeline_family,
                                "compound_packet": leaf.compound_exp2_packet,
                                "softmax_disc": leaf.softmax_disc,
                                "config_id": canonical_config_id(member.config),
                                "unrestricted": unrestricted,
                                "pipeline_lane": self._flash_pipeline_lane_metric(lane),
                            }
                            for member, leaf, unrestricted, lane in selected_leaf_paths
                            if leaf.pipeline_family == family
                        ],
                    }
                    for family in reported_families
                ]
            return paths

        selected: list[tuple[PopulationMember, tuple[tuple[str, object], ...]]] = []
        selected_configs: set[Config] = set()
        if eligible:
            selected.append((eligible[0], ()))
            selected_configs.add(eligible[0].config)
        for path in sorted(
            self._flash_structural_paths(),
            key=lambda item: self._flash_member_rank_key(item[0]),
        ):
            if len(selected) >= self.copies:
                break
            if path[0].config in selected_configs:
                continue
            selected.append(path)
            selected_configs.add(path[0].config)
        for member in eligible:
            if len(selected) >= self.copies:
                break
            if member.config in selected_configs:
                continue
            selected.append((member, ()))
            selected_configs.add(member.config)
        return selected

    def _flash_structural_paths(
        self,
        *,
        include_nonfinite: bool = False,
    ) -> list[tuple[PopulationMember, tuple[tuple[str, object], ...]]]:
        """Return one measured representative for each live structure.

        Qualification may start from a failed witness because a nearby child
        configuration can still be valid. Long-running paths remain finite-only.
        """
        if not self.config_spec.cute_flash_search_enabled:
            return []
        eligible = (
            self.population
            if include_nonfinite
            else [member for member in self.population if math.isfinite(member.perf)]
        )

        by_leaf: dict[FlashStructuralLeaf, PopulationMember] = {}
        for member in eligible:
            leaf = self._flash_structural_leaf(member)
            if leaf is not None and (
                leaf not in by_leaf
                or self._flash_member_rank_key(member)
                < self._flash_member_rank_key(by_leaf[leaf])
            ):
                by_leaf[leaf] = member
        catalog = self._flash_generation().flash_structural_leaf_catalog()
        return [
            (by_leaf[leaf], self._flash_leaf_constraints(leaf))
            for leaf in catalog
            if leaf in by_leaf
        ]

    def _generate_neighbors(
        self,
        base: FlatConfig,
        *,
        fixed_flat_values: Mapping[int, object] | None = None,
        config_gen: ConfigGeneration | None = None,
        num_neighbors: int | None = None,
        proposal_callback: Callable[[FlatConfig], None] | None = None,
    ) -> list[FlatConfig]:
        """
        Generate neighboring configurations randomly within a specified radius.

        Strategy:
        1. Sample one block size index and change it by at most radius (in log2 space)
        2. Sample the num_warps index and change it by at most radius (in log2 space)
        3. For at most radius remaining indices, randomly select pattern neighbors

        Args:
            base: The base configuration to generate neighbors from

        Returns:
            A list of neighboring configurations
        """
        neighbors: list[FlatConfig] = []
        config_gen = self.config_gen if config_gen is None else config_gen

        # Generate num_neighbors random neighbors
        frozen = set(config_gen.overridden_flat_indices)
        if fixed_flat_values:
            frozen.update(fixed_flat_values)
        eligible_block = [i for i in config_gen.block_size_indices if i not in frozen]
        warp_idx = config_gen.num_warps_index
        tune_warps = warp_idx >= 0 and warp_idx not in frozen
        for _ in range(self.num_neighbors if num_neighbors is None else num_neighbors):
            new_flat = [*base]  # Copy the base configuration
            modified_indices = set()

            # 1. Sample a block size index and change it
            if eligible_block:
                block_idx = random.choice(eligible_block)
                modified_indices.add(block_idx)

                block_spec = config_gen.flat_spec[block_idx]
                block_neighbors = block_spec.pattern_neighbors(
                    base[block_idx], self.radius
                )
                if block_neighbors:
                    new_flat[block_idx] = random.choice(block_neighbors)

            # 2. Sample the num_warps index and change it
            if tune_warps:
                modified_indices.add(warp_idx)

                warp_spec = config_gen.flat_spec[warp_idx]
                warp_neighbors = warp_spec.pattern_neighbors(
                    base[warp_idx], self.radius
                )
                if warp_neighbors:
                    new_flat[warp_idx] = random.choice(warp_neighbors)

            # 3. For at most radius remaining indices, use pattern neighbors
            # Exclude the already-modified block size and warp indices

            # Collect available pattern neighbors for remaining indices
            remaining_pattern_neighbors = []
            for index, spec in enumerate(config_gen.flat_spec):
                if index not in modified_indices and index not in frozen:
                    pattern_neighbors = spec.pattern_neighbors(base[index])
                    if pattern_neighbors:
                        remaining_pattern_neighbors.append((index, pattern_neighbors))

            # Randomly select at most radius indices to change
            if remaining_pattern_neighbors:
                num_to_change = random.randint(
                    0, min(self.radius, len(remaining_pattern_neighbors))
                )
                if num_to_change > 0:
                    indices_to_change = random.sample(
                        remaining_pattern_neighbors, num_to_change
                    )
                    for idx, pattern_neighbors in indices_to_change:
                        new_flat[idx] = random.choice(pattern_neighbors)

            if proposal_callback is not None:
                proposal_callback(new_flat)
            # Only add if it's different from the base
            if new_flat != base:
                neighbors.append(new_flat)

        return self.shrink_neighbors(neighbors)

    def _flash_leaf_config_generation(
        self,
        leaf: FlashStructuralLeaf,
        lane_constraints: tuple[tuple[str, object], ...] = (),
    ) -> ConfigGeneration | None:
        if not hasattr(self.config_spec, "create_config_generation"):
            return None
        cache = getattr(self, "_flash_leaf_config_generation_cache", None)
        if cache is None:
            cache = {}
            self._flash_leaf_config_generation_cache = cache
        cache_key = (leaf, lane_constraints)
        if cache_key in cache:
            return cache[cache_key]

        from .._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
        from .._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
        from .._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY

        overrides = dict(self.config_gen._override_values)
        overrides[FLASH_PIPELINE_FAMILY_KEY] = leaf.pipeline_family
        overrides[FLASH_SOFTMAX_DISC_KEY] = leaf.softmax_disc
        if leaf.compound_exp2_packet is not None:
            overrides[FLASH_EXP2_PACKET_KEY] = leaf.compound_exp2_packet
        overrides.update(lane_constraints)
        config_gen = self.config_spec.create_config_generation(
            overrides=overrides,
            advanced_controls_files=self.config_gen._advanced_controls_files,
            process_group_name=self.config_gen.process_group_name,
        )
        cache[cache_key] = config_gen
        return config_gen

    @staticmethod
    def _flash_terminal_projection_metric(
        projection: CoordinateNeighborProjection,
        *,
        outcome: str,
    ) -> dict[str, object]:
        return {
            "flat_index": projection.flat_index,
            "key": projection.key,
            "sequence_index": projection.sequence_index,
            "from_value": copy.deepcopy(projection.from_value),
            "to_value": copy.deepcopy(projection.to_value),
            "outcome": outcome,
            "config_id": (
                canonical_config_id(projection.config)
                if projection.config is not None
                else None
            ),
        }

    def _flash_terminal_member_result(
        self, member: PopulationMember
    ) -> dict[str, object]:
        succeeded = self._flash_member_succeeded(member)
        return {
            "config_id": canonical_config_id(member.config),
            "attempt_perf": (
                member.perfs[0]
                if member.perfs and math.isfinite(member.perfs[0])
                else None
            ),
            "selection_perf": member.perf if succeeded else None,
            "status": member.status,
            "source_hash": self._flash_member_source_hash(member),
        }

    @staticmethod
    def _flash_terminal_trace_metric(
        member_ids: Sequence[str], trace: MirroredBenchmarkTrace
    ) -> dict[str, object]:
        return {
            "base_order": list(member_ids),
            "target_ms": trace.target_ms,
            "repeat_reference_perf_ms": trace.repeat_reference_perf_ms,
            "sweep_count": trace.sweep_count,
            "calls_per_sample": trace.calls_per_sample,
            "total_calls": trace.total_calls,
            "elapsed_ms": [list(times) for times in trace.elapsed_ms],
            "median_ms": [
                {"config_id": config_id, "value": timing}
                for config_id, timing in zip(
                    member_ids,
                    trace.medians_ms,
                    strict=True,
                )
            ],
        }

    def run_terminal_refinement(self, best: PopulationMember) -> PopulationMember:
        """Close the final CuTe-flash basin with a deterministic coordinate beam."""
        policy = getattr(self, "flash_structural_search", None)
        phase = getattr(
            getattr(self, "_autotune_metrics", None),
            "search_phase_metrics",
            None,
        )
        if (
            not getattr(self, "_cute_flash_lane_policy_enabled", False)
            or policy is None
            or policy.terminal_coordinate_rounds <= 0
            or policy.terminal_coordinate_beam_width <= 0
            or not isinstance(phase, dict)
            or self.performance_unit != "ms"
            or self.settings.autotune_benchmark_fn is not None
            or self.settings.autotune_budget_seconds is not None
            or isinstance(
                getattr(self, "benchmark_provider", None),
                MultiShapeBenchmarkProvider,
            )
        ):
            return best

        search_generation = self._autotune_metrics.num_generations
        initial_config_id = canonical_config_id(best.config)
        config_manifest: dict[str, dict[str, object]] = {}
        unique_candidate_ids: set[str] = set()
        new_candidate_ids: set[str] = set()
        reused_candidate_ids: set[str] = set()
        intra_terminal_reused_candidate_ids: set[str] = set()
        prior_failed_candidate_ids: set[str] = set()
        projection_attempt_count = 0
        projection_parent_count = 0
        transcript: dict[str, object] = {
            "schema_version": _FLASH_TERMINAL_REFINEMENT_SCHEMA_VERSION,
            "policy_version": _FLASH_TERMINAL_REFINEMENT_POLICY_VERSION,
            "lane_policy_version": CUTE_FLASH_LANE_POLICY_VERSION,
            "coordinate_policy": _FLASH_TERMINAL_COORDINATE_POLICY,
            "measurement_policy": _FLASH_TERMINAL_MEASUREMENT_POLICY,
            "rounds_planned": policy.terminal_coordinate_rounds,
            "beam_width": policy.terminal_coordinate_beam_width,
            "maximum_projection_parent_count": 1
            + policy.terminal_coordinate_beam_width
            * max(policy.terminal_coordinate_rounds - 1, 0),
            "projection_parent_count": 0,
            "rounds_started": 0,
            "rounds_completed": 0,
            "completed": False,
            "budget_exhausted": False,
            "termination_reason": None,
            "search_generation": search_generation,
            "preterminal_num_configs_tested": getattr(
                self._autotune_metrics,
                "num_configs_tested",
                0,
            ),
            "preterminal_registry_config_count": 0,
            "preterminal_registry_config_ids_hash_policy": (
                "sorted_compact_json_sha256_v1"
            ),
            "preterminal_registry_config_ids_sha256": None,
            "radius": self.radius,
            "minimum_improvement_fraction": self.min_improvement_delta,
            "initial_incumbent_config_id": initial_config_id,
            "refined_config_id": initial_config_id,
            "final_config_id": initial_config_id,
            "projection_attempt_count": 0,
            "unique_candidate_count": 0,
            "new_candidate_count": 0,
            "reused_candidate_count": 0,
            "intra_terminal_reused_candidate_count": 0,
            "prior_failed_candidate_count": 0,
            "accepted_config_ids": [],
            "config_manifest_sha256": None,
            "config_manifest": config_manifest,
            "rounds": [],
            "confirmation": None,
        }
        phase["terminal_coordinate_refinement"] = transcript

        registry = getattr(self, "_terminal_refinement_members", None)
        if registry is None:
            registry = {}
            self._terminal_refinement_members = registry
        self._record_best_member_for_config(
            registry,
            best.config,
            best,
            replace=True,
        )
        preterminal_registry_config_ids = sorted(
            canonical_config_id(config) for config in registry
        )
        preterminal_configs = set(registry)
        transcript["preterminal_registry_config_count"] = len(
            preterminal_registry_config_ids
        )
        transcript["preterminal_registry_config_ids_sha256"] = hashlib.sha256(
            json.dumps(
                preterminal_registry_config_ids,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        def add_manifest_config(member_config: Config) -> None:
            config_id = canonical_config_id(member_config)
            config_manifest[config_id] = {"config": copy.deepcopy(member_config.config)}

        def finish_transcript(current: PopulationMember) -> PopulationMember:
            transcript["final_config_id"] = canonical_config_id(current.config)
            transcript["unique_candidate_count"] = len(unique_candidate_ids)
            transcript["new_candidate_count"] = len(new_candidate_ids)
            transcript["reused_candidate_count"] = len(reused_candidate_ids)
            transcript["intra_terminal_reused_candidate_count"] = len(
                intra_terminal_reused_candidate_ids
            )
            transcript["prior_failed_candidate_count"] = len(prior_failed_candidate_ids)
            sorted_manifest = dict(sorted(config_manifest.items()))
            transcript["config_manifest"] = sorted_manifest
            transcript["config_manifest_sha256"] = hashlib.sha256(
                json.dumps(
                    sorted_manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if self._autotune_metrics.num_generations != search_generation:
                raise AssertionError(
                    "terminal refinement changed LFBO generation count"
                )
            if current.config != best.config:
                self.log(
                    "Terminal coordinate refinement selected "
                    f"{current.config} ({self.format_performance(current.perf)})"
                )
            self._selected_member = current
            return current

        add_manifest_config(best.config)
        initial_leaf = self._flash_structural_leaf(best)
        if initial_leaf is None:
            transcript["termination_reason"] = "no_candidates"
            transcript["completed"] = True
            transcript["confirmation"] = {
                "candidate_config_ids": [initial_config_id],
                "measurement": None,
                "best_config_id": initial_config_id,
                "selected_config_id": initial_config_id,
                "accepted": False,
                "improvement_fraction": 0.0,
                "skipped_reason": "missing_structural_leaf",
            }
            return finish_transcript(best)

        incumbent = best
        beam = [best]
        final_beam = [best]
        accepted_round_winners: list[PopulationMember] = []
        accepted_configs: set[Config] = set()
        round_metrics = transcript["rounds"]
        assert isinstance(round_metrics, list)
        termination_reason = "round_limit"

        for round_index in range(1, policy.terminal_coordinate_rounds + 1):
            transcript["rounds_started"] = round_index
            parent_ids = [canonical_config_id(parent.config) for parent in beam]
            projection_parent_count += len(parent_ids)
            transcript["projection_parent_count"] = projection_parent_count
            parent_configs = {parent.config for parent in beam}
            parent_projections: list[dict[str, object]] = []
            candidate_member_by_config: dict[Config, PopulationMember] = {}
            candidate_configs: list[Config] = []
            round_seen_configs: set[Config] = set()
            newly_unbenchmarked: list[PopulationMember] = []
            round_new_ids: list[str] = []
            round_reused_ids: list[str] = []
            round_intra_terminal_reused_ids: list[str] = []
            prior_failed_ids: list[str] = []
            for parent in beam:
                leaf_config_gen = self._flash_leaf_config_generation(initial_leaf)
                if leaf_config_gen is None:
                    termination_reason = "no_candidates"
                    break
                projections = self.config_gen.canonicalize_coordinate_projections(
                    leaf_config_gen.coordinate_neighbor_projections(
                        leaf_config_gen.flatten(parent.config),
                        radius=self.radius,
                    ),
                    base_config=parent.config,
                )
                requests: list[dict[str, object]] = []
                projection_attempt_count += len(projections)
                for projection in projections:
                    outcome = projection.outcome
                    config = projection.config
                    if (
                        outcome == "candidate"
                        and config is not None
                        and self._flash_structural_leaf_from_config(config)
                        != initial_leaf
                    ):
                        outcome = "different_leaf"
                    elif outcome == "candidate" and config in parent_configs:
                        outcome = "beam_alias"
                    elif outcome == "candidate" and config in round_seen_configs:
                        outcome = "round_candidate_alias"
                    requests.append(
                        self._flash_terminal_projection_metric(
                            projection,
                            outcome=outcome,
                        )
                    )
                    if config is not None:
                        add_manifest_config(config)
                    if outcome != "candidate" or config is None:
                        continue
                    round_seen_configs.add(config)
                    candidate_configs.append(config)
                    config_id = canonical_config_id(config)
                    unique_candidate_ids.add(config_id)
                    existing = registry.get(config)
                    if existing is not None:
                        candidate_member_by_config[config] = existing
                        if self._flash_member_succeeded(existing):
                            if config in preterminal_configs:
                                round_reused_ids.append(config_id)
                                reused_candidate_ids.add(config_id)
                            else:
                                round_intra_terminal_reused_ids.append(config_id)
                                intra_terminal_reused_candidate_ids.add(config_id)
                        else:
                            prior_failed_ids.append(config_id)
                            prior_failed_candidate_ids.add(config_id)
                        continue

                    member = self.make_unbenchmarked(self.config_gen.flatten(config))
                    if member is None:
                        prior_failed_ids.append(config_id)
                        prior_failed_candidate_ids.add(config_id)
                        continue
                    candidate_member_by_config[config] = member
                    newly_unbenchmarked.append(member)
                    round_new_ids.append(config_id)
                    new_candidate_ids.add(config_id)
                parent_projections.append(
                    {
                        "parent_config_id": canonical_config_id(parent.config),
                        "coordinate_requests": requests,
                    }
                )
            if termination_reason == "no_candidates":
                break

            transcript["projection_attempt_count"] = projection_attempt_count
            if newly_unbenchmarked:
                self.benchmark_population(
                    newly_unbenchmarked,
                    desc=f"Terminal coordinate refinement {round_index}:",
                )

            measured: list[PopulationMember] = []
            measured_configs: set[Config] = set()
            for member in (*beam, *candidate_member_by_config.values()):
                if (
                    self._flash_member_succeeded(member)
                    and member.config not in measured_configs
                ):
                    measured.append(member)
                    measured_configs.add(member.config)

            candidate_ids = [
                canonical_config_id(config) for config in candidate_configs
            ]
            candidate_results = [
                self._flash_terminal_member_result(candidate_member_by_config[config])
                for config in candidate_configs
                if config in candidate_member_by_config
            ]

            round_metric: dict[str, object] = {
                "round_index": round_index,
                "incumbent_config_id": canonical_config_id(incumbent.config),
                "leaf": {
                    "family": initial_leaf.pipeline_family,
                    "compound_packet": initial_leaf.compound_exp2_packet,
                    "softmax_disc": initial_leaf.softmax_disc,
                },
                "parent_config_ids": parent_ids,
                "parent_projections": parent_projections,
                "candidate_config_ids": candidate_ids,
                "new_candidate_ids": round_new_ids,
                "reused_candidate_ids": round_reused_ids,
                "intra_terminal_reused_candidate_ids": (
                    round_intra_terminal_reused_ids
                ),
                "prior_failed_candidate_ids": prior_failed_ids,
                "candidate_results": candidate_results,
                "comparison_config_ids": [],
                "measurement": None,
                "round_best_config_id": canonical_config_id(incumbent.config),
                "selected_config_id": canonical_config_id(incumbent.config),
                "accepted": False,
                "improvement_fraction": 0.0,
                "beam_config_ids": parent_ids,
            }
            round_metrics.append(round_metric)

            if len(measured) < 2:
                transcript["rounds_completed"] = round_index
                final_beam = beam
                termination_reason = "no_candidates"
                break

            member_ids = [canonical_config_id(member.config) for member in measured]
            round_metric["comparison_config_ids"] = member_ids
            trace = self.mirrored_rebenchmark(
                measured,
                desc=f"Terminal coordinate refinement {round_index}: comparing",
                target_ms=_FLASH_TERMINAL_REFINEMENT_TARGET_MS,
            )
            round_metric["measurement"] = self._flash_terminal_trace_metric(
                member_ids,
                trace,
            )
            round_metric["candidate_results"] = [
                self._flash_terminal_member_result(candidate_member_by_config[config])
                for config in candidate_configs
                if config in candidate_member_by_config
            ]

            round_best = min(measured, key=self._flash_member_rank_key)
            improvement = (
                1.0 - round_best.perf / incumbent.perf
                if math.isfinite(round_best.perf)
                and math.isfinite(incumbent.perf)
                and incumbent.perf > 0.0
                else 0.0
            )
            accepted = bool(
                round_best.config != incumbent.config
                and improvement >= self.min_improvement_delta
            )
            if accepted:
                incumbent = round_best
                add_manifest_config(incumbent.config)
                if incumbent.config not in accepted_configs:
                    accepted_configs.add(incumbent.config)
                    accepted_round_winners.append(incumbent)
                    accepted_config_ids = transcript["accepted_config_ids"]
                    assert isinstance(accepted_config_ids, list)
                    accepted_config_ids.append(canonical_config_id(incumbent.config))

            ranked = sorted(measured, key=self._flash_member_rank_key)
            next_beam = [incumbent]
            for member in ranked:
                if member.config in {item.config for item in next_beam}:
                    continue
                next_beam.append(member)
                if len(next_beam) >= policy.terminal_coordinate_beam_width:
                    break
            beam = next_beam
            final_beam = beam
            round_metric["round_best_config_id"] = canonical_config_id(
                round_best.config
            )
            round_metric["selected_config_id"] = canonical_config_id(incumbent.config)
            round_metric["accepted"] = accepted
            round_metric["improvement_fraction"] = improvement
            round_metric["beam_config_ids"] = [
                canonical_config_id(member.config) for member in beam
            ]
            transcript["rounds_completed"] = round_index

        transcript["termination_reason"] = termination_reason
        transcript["refined_config_id"] = canonical_config_id(incumbent.config)
        confirmation_members: list[PopulationMember] = []
        confirmation_configs: set[Config] = set()
        for member in (best, *accepted_round_winners, *final_beam):
            if member.config in confirmation_configs:
                continue
            confirmation_configs.add(member.config)
            confirmation_members.append(member)
            add_manifest_config(member.config)

        current = best
        if len(confirmation_members) > 1:
            confirmation_ids = [
                canonical_config_id(member.config) for member in confirmation_members
            ]
            confirmation_trace = self.mirrored_rebenchmark(
                confirmation_members,
                desc="Terminal coordinate refinement: confirming",
                target_ms=_FLASH_TERMINAL_CONFIRMATION_TARGET_MS,
            )
            confirmed_best = min(
                confirmation_members,
                key=self._flash_member_rank_key,
            )
            confirmation_improvement = (
                1.0 - confirmed_best.perf / best.perf
                if math.isfinite(confirmed_best.perf)
                and math.isfinite(best.perf)
                and best.perf > 0.0
                else 0.0
            )
            confirmation_accepted = bool(
                confirmed_best.config != best.config
                and confirmation_improvement >= self.min_improvement_delta
            )
            if confirmation_accepted:
                current = confirmed_best
            transcript["confirmation"] = {
                "candidate_config_ids": confirmation_ids,
                "measurement": self._flash_terminal_trace_metric(
                    confirmation_ids,
                    confirmation_trace,
                ),
                "best_config_id": canonical_config_id(confirmed_best.config),
                "selected_config_id": canonical_config_id(current.config),
                "accepted": confirmation_accepted,
                "improvement_fraction": confirmation_improvement,
                "skipped_reason": None,
            }
        else:
            transcript["confirmation"] = {
                "candidate_config_ids": [canonical_config_id(best.config)],
                "measurement": None,
                "best_config_id": canonical_config_id(best.config),
                "selected_config_id": canonical_config_id(best.config),
                "accepted": False,
                "improvement_fraction": 0.0,
                "skipped_reason": "single_candidate",
            }
        transcript["completed"] = True
        return finish_transcript(current)

    def _flash_conditional_parent_candidates(
        self,
        members: Sequence[PopulationMember],
        leaf: FlashStructuralLeaf,
        constraints: tuple[tuple[str, object], ...],
        known_configs: set[Config],
        neighbor_limit: int,
    ) -> tuple[PopulationMember | None, list[PopulationMember], dict[str, object]]:
        """Find one conditional child within a shared ranked-parent budget.

        A parent's projected neighborhood can already be measured even when
        another parent in the same lane has novel children. Keep the complete
        attempt ledger so strict qualification can distinguish that local
        saturation from exhaustion of the lane.
        """
        ranked = sorted(members, key=self._flash_member_rank_key)
        parent_count = min(len(ranked), max(0, neighbor_limit))
        allocations = [
            (index + 1) * neighbor_limit // parent_count
            - index * neighbor_limit // parent_count
            for index in range(parent_count)
        ]
        attempts: list[dict[str, object]] = []
        ledger: dict[str, object] = {
            "schema_version": 2,
            "neighbor_limit": neighbor_limit,
            "radius": self.radius,
            "known_config_ids": sorted(map(canonical_config_id, known_configs)),
            "allocations": [
                {"parent_config_id": canonical_config_id(parent.config), "limit": limit}
                for parent, limit in zip(ranked, allocations, strict=False)
            ],
            "attempts": attempts,
            "consumed": 0,
            "selected_parent_config_id": None,
            "selected_child_config_id": None,
        }
        consumed_total = 0
        for parent, limit in zip(ranked, allocations, strict=False):
            generation = self._flash_leaf_config_generation(leaf, constraints)
            assert generation is not None
            base, parent_config = generation.canonicalize_flat(
                generation.flatten(parent.config)
            )
            owned_indices = self._flash_generation(
                generation
            ).flash_owned_coordinate_indices(parent_config)
            proposals: list[dict[str, object]] = []
            candidates: list[PopulationMember] = []
            proposed_configs: set[Config] = set()

            def record_proposal(
                raw: FlatConfig,
                kind: str,
                *,
                generation: ConfigGeneration = generation,
                proposals: list[dict[str, object]] = proposals,
                candidates: list[PopulationMember] = candidates,
                proposed_configs: set[Config] = proposed_configs,
            ) -> None:
                record: dict[str, object] = {
                    "kind": kind,
                    "raw_flat_values": copy.deepcopy(raw),
                    "config": None,
                    "config_id": None,
                    "outcome": "invalid",
                }
                proposals.append(record)
                try:
                    _, local = generation.canonicalize_flat(raw)
                    flat, config = self.config_gen.canonicalize_flat(
                        self.config_gen.flatten(local)
                    )
                except exc.InvalidConfig:
                    return
                record["config"] = dict(config.config)
                record["config_id"] = canonical_config_id(config)
                if self._flash_structural_leaf_from_config(config) != leaf or any(
                    config.config.get(key) != value for key, value in constraints
                ):
                    record["outcome"] = "constraint_mismatch"
                elif not self._backend_config_is_viable(config):
                    record["outcome"] = "backend_invalid"
                elif config in known_configs:
                    record["outcome"] = "known"
                elif config in proposed_configs:
                    record["outcome"] = "duplicate"
                else:
                    record["outcome"] = "novel"
                    proposed_configs.add(config)
                    candidates.append(PopulationMember(_unset_fn, [], flat, config))

            with sync_seed(process_group_name=self.kernel.env.process_group_name):
                projections = generation.coordinate_neighbor_projections(
                    base,
                    radius=self.radius,
                    limit=limit,
                    frozen_indices=owned_indices,
                )
                for projection in projections:
                    raw = copy.deepcopy(base)
                    raw[projection.flat_index] = copy.deepcopy(projection.to_value)
                    record_proposal(raw, "coordinate")
                remaining = limit - len(proposals)
                random_state = random.getstate() if remaining else None
                if remaining:
                    LFBOPatternSearch._generate_neighbors(
                        self,
                        base,
                        config_gen=generation,
                        num_neighbors=remaining,
                        proposal_callback=lambda raw: record_proposal(raw, "random"),
                    )
            assert len(proposals) == limit
            attempt: dict[str, object] = {
                "parent_config_id": canonical_config_id(parent.config),
                "owned_coordinate_indices": owned_indices,
                "allocated": limit,
                "consumed": len(proposals),
                "random_state": random_state,
                "proposals": proposals,
                "novel_config_ids": [
                    canonical_config_id(item.config) for item in candidates
                ],
            }
            attempts.append(attempt)
            consumed_total += len(proposals)
            ledger["consumed"] = consumed_total
            if not candidates:
                continue
            selected = self._surrogate_select([parent, *candidates], 2)
            child = next(item for item in selected if item.config != parent.config)
            ledger["selected_parent_config_id"] = canonical_config_id(parent.config)
            ledger["selected_child_config_id"] = canonical_config_id(child.config)
            return parent, [child], ledger
        return None, [], ledger

    def _generate_flash_leaf_neighbors(
        self,
        current: PopulationMember,
        leaf: FlashStructuralLeaf,
        lane_constraints: tuple[tuple[str, object], ...] = (),
        neighbor_limit: int | None = None,
    ) -> list[FlatConfig]:
        """Generate on the family-conditional surface, then map back globally."""
        config_gen = self._flash_leaf_config_generation(leaf, lane_constraints)
        if config_gen is None:
            return self._generate_neighbors(
                current.flat_values, num_neighbors=neighbor_limit
            )
        base = config_gen.flatten(current.config)
        child_neighbors = LFBOPatternSearch._generate_neighbors(
            self,
            base,
            config_gen=config_gen,
            num_neighbors=neighbor_limit,
        )
        result: list[FlatConfig] = []
        seen: set[Config] = set()
        for child_flat in child_neighbors:
            try:
                _child_flat, config = config_gen.canonicalize_flat(child_flat)
                if self._flash_structural_leaf_from_config(config) != leaf:
                    continue
                global_flat = self.config_gen.flatten(config)
                global_flat, global_config = self.config_gen.canonicalize_flat(
                    global_flat
                )
            except exc.InvalidConfig:
                continue
            if (
                global_config in seen
                or self._flash_structural_leaf_from_config(global_config) != leaf
            ):
                continue
            seen.add(global_config)
            result.append(global_flat)
        return self.shrink_neighbors(result)

    @staticmethod
    def _flash_structural_leaf_from_config(
        config: Config,
    ) -> FlashStructuralLeaf | None:
        from .._compiler.cute.cute_flash import flash_structural_leaf_from_config

        return flash_structural_leaf_from_config(config.config)

    def _pruned_pattern_search_from(
        self,
        copy_idx: int,
        current: PopulationMember,
        visited: set[Config],
        constraints: tuple[tuple[str, object], ...] = (),
        selected_limit: int | None = None,
        required_leaf: FlashStructuralLeaf | None = None,
        conditional_surface: bool = False,
        disable_early_stopping: bool = False,
        neighbor_limit: int | None = None,
    ) -> Iterator[list[PopulationMember]]:
        """
        Run a single copy of pattern search from the given starting point.

        We use a generator and yield the new population at each generation so that we can
        run multiple copies of pattern search in parallel.

        Only keep self.frac_selected of the neighbors generated from the current
        search_copy using _surrogate_select.

        Args:
            current: The current best configuration.
            visited: A set of visited configurations.

        Returns:
            A generator that yields the new population at each generation.
        """
        patience = self.patience
        fixed_flat_values: dict[int, object] = {}
        for key, _value in constraints:
            indices, _is_sequence = self.config_gen._key_to_flat_indices[key]
            for index in indices:
                fixed_flat_values[index] = current.flat_values[index]
        for _ in range(self.max_generations):
            candidates: list[PopulationMember] = [current]
            generated_configs: set[Config] = set()
            with sync_seed(process_group_name=self.kernel.env.process_group_name):
                if required_leaf is not None and conditional_surface:
                    all_neighbors = self._generate_flash_leaf_neighbors(
                        current, required_leaf, constraints, neighbor_limit
                    )
                elif fixed_flat_values:
                    if neighbor_limit is None:
                        all_neighbors = self._generate_neighbors(
                            current.flat_values,
                            fixed_flat_values=fixed_flat_values,
                        )
                    else:
                        all_neighbors = self._generate_neighbors(
                            current.flat_values,
                            fixed_flat_values=fixed_flat_values,
                            num_neighbors=neighbor_limit,
                        )
                else:
                    if neighbor_limit is None:
                        all_neighbors = self._generate_neighbors(current.flat_values)
                    else:
                        all_neighbors = self._generate_neighbors(
                            current.flat_values, num_neighbors=neighbor_limit
                        )
            for flat_config in all_neighbors:
                new_member = self.make_unbenchmarked(flat_config)
                if new_member is None or any(
                    new_member.config.config.get(key) != value
                    for key, value in constraints
                ):
                    continue
                if (
                    required_leaf is not None
                    and self._flash_structural_leaf(new_member) != required_leaf
                ):
                    continue
                if (
                    new_member.config not in visited
                    and new_member.config not in generated_configs
                ):
                    candidates.append(new_member)
                    generated_configs.add(new_member.config)

            # Score candidates. Only the selected (i.e. benchmarked)
            # candidates enter `visited`, so proposals the surrogate rejects
            # can be re-proposed after it learns more. The incumbent is always
            # retained and at least one real neighbor is selected so a copy
            # cannot die from selection-quota truncation.
            n_sorted = int(len(candidates) * self.frac_selected)
            if len(candidates) > 1:
                n_sorted = max(2, n_sorted)
            if selected_limit is not None:
                n_sorted = min(n_sorted, selected_limit)
            candidates = self._surrogate_select(candidates, n_sorted)
            selected_neighbors = [
                member for member in candidates if member.config != current.config
            ]
            candidates = [current, *selected_neighbors[: max(0, n_sorted - 1)]]
            visited.update(member.config for member in candidates)

            if len(candidates) <= 1:
                self.log(f"Copy {copy_idx} finish because of no candidates")
                return  # no new candidates, stop searching
            yield candidates  # yield new population to benchmark in parallel
            best = min(candidates, key=performance)
            if not disable_early_stopping and self._check_early_stopping(best, current):
                if patience > 0:
                    patience -= 1
                else:
                    self.log(f"Copy {copy_idx} finish because of no improvement")
                    return
            current = best


class LFBOTreeSearch(LFBOPatternSearch):
    """
    LFBO Tree Search: Likelihood-Free Bayesian Optimization with tree-guided neighbor generation.

    This algorithm uses a Random Forest classifier as a surrogate model to both
    select which configurations to benchmark and to guide the generation of new
    candidate configurations via greedy decision tree traversal.

    Algorithm Overview:
        1. Generate an initial population (random or default) and benchmark all configurations
        2. Fit a Random Forest classifier to predict "good" vs "bad" configurations:
           - Configs with performance < quantile threshold are labeled as "good" (class 1)
           - Configs with performance >= quantile threshold are labeled as "bad" (class 0)
           - Weighted classification emphasizes configs that are much better than the threshold
        3. For the first generation, generate neighbors via random perturbation
           since the surrogate is not yet fitted
        4. For subsequent generations, generate neighbors via greedy tree traversal:
           a. For each of num_neighbors trials:
              - Pick a random decision tree from the Random Forest
              - Trace the decision path for the current best config through that tree
              - Extract the configuration parameters used in the tree's split decisions
              - For each parameter on the path, greedily optimize it:
                  * Generate pattern_neighbors within the configured radius
                  * Score candidates using the single tree's predicted probability
                  * Accept the best value (ties broken randomly) and incrementally
                    update the encoded representation
              - Keep the result only if it differs from the base configuration
           b. Score candidates using the full ensemble's predicted probability
              with a diversity-aware similarity penalty, then select top candidates
        5. Benchmark selected candidates, retrain the classifier on all observed data

    The tree-guided traversal focuses search on parameters the surrogate has identified
    as important (those used in tree splits). Using a single tree per trial (rather
    than the full ensemble) introduces diversity since different trees may emphasize
    different parameters.

    References:
    - Song, J., et al. (2022). "A General Recipe for Likelihood-free Bayesian Optimization."
    - Mišić, Velibor V. "Optimization of tree ensembles." Operations Research 68.5 (2020): 1605-1624.

    Args:
        kernel: The kernel to be autotuned.
        args: The arguments to be passed to the kernel during benchmarking.
        initial_population: Number of random configurations in initial population.
            Default from PATTERN_SEARCH_DEFAULTS. Ignored when using DEFAULT strategy.
        copies: Number of top configurations to run pattern search from.
            Full CuTe-flash searches give every ordinary structural leaf bounded
            qualification work and transfer top representatives to compound leaves,
            then retain at most ``copies`` paths across the best parent families.
            Default from PATTERN_SEARCH_DEFAULTS.
        max_generations: Maximum number of search iterations per copy.
            Default from PATTERN_SEARCH_DEFAULTS.
        min_improvement_delta: Early stopping threshold. Search stops if the relative
            improvement abs(best/current - 1) < min_improvement_delta.
            Default: 0.001 (0.1% improvement threshold).
        frac_selected: Fraction of generated neighbors to actually benchmark, after
            filtering by classifier score. Range: (0, 1]. Lower values reduce benchmarking
            cost but may miss good configurations. Default: 0.15.
        num_neighbors: Number of greedy tree traversal trials to run per generation.
            Each trial picks a random tree, traces its decision path, and greedily
            optimizes parameters along that path. Default: 100.
        radius: Maximum perturbation distance when generating pattern neighbors for
            each parameter during tree traversal. For power-of-two parameters, this
            is the max change in log2 space. For other parameters, this limits the
            neighborhood size. Default: 3.
        quantile: Threshold for labeling configs as "good" (class 1) vs "bad" (class 0).
            Configs with performance below this quantile are labeled as good.
            Range: (0, 1). Lower values create a more selective definition of "good".
            Default: 0.1 (top 10% are considered good).
        patience: Number of generations without improvement before stopping
            the search copy. Default: 1.
        similarity_penalty: Penalty for selecting points that are similar to points
            already selected in the batch. Default: 1.0.
        initial_population_strategy: Strategy for generating the initial population.
            FROM_RANDOM generates initial_population random configs.
            FROM_BEST_AVAILABLE uses cached configs from prior runs, and fills the
            remainder with random configs when best_available_pad_random is True.
            Can be overridden by HELION_AUTOTUNER_INITIAL_POPULATION env var.
        num_neighbors_cap: Maximum number of neighbors to explore per generation.
            -1 means no cap. Set HELION_CAP_AUTOTUNE_NUM_NEIGHBORS=N to override.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        num_neighbors: int = 200,
        frac_selected: float = 0.10,
        radius: int = 2,
        initial_population: int = PATTERN_SEARCH_DEFAULTS.initial_population,
        copies: int = PATTERN_SEARCH_DEFAULTS.copies,
        max_generations: int = PATTERN_SEARCH_DEFAULTS.max_generations,
        min_improvement_delta: float = 0.001,
        quantile: float = 0.1,
        patience: int = 1,
        similarity_penalty: float = 1.0,
        initial_population_strategy: InitialPopulationStrategy | None = None,
        best_available_pad_random: bool = PATTERN_SEARCH_DEFAULTS.best_available_pad_random,
        num_neighbors_cap: int = -1,
        finishing_rounds: int = 0,
        polish_rounds: int = 10,
        compile_timeout_lower_bound: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_lower_bound,
        compile_timeout_quantile: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_quantile,
        flash_structural_search: FlashStructuralSearchConfig | None = None,
    ) -> None:
        super().__init__(
            kernel=kernel,
            args=args,
            num_neighbors=num_neighbors,
            frac_selected=frac_selected,
            radius=radius,
            initial_population=initial_population,
            copies=copies,
            max_generations=max_generations,
            min_improvement_delta=min_improvement_delta,
            quantile=quantile,
            patience=patience,
            similarity_penalty=similarity_penalty,
            initial_population_strategy=initial_population_strategy,
            best_available_pad_random=best_available_pad_random,
            num_neighbors_cap=num_neighbors_cap,
            finishing_rounds=finishing_rounds,
            polish_rounds=polish_rounds,
            compile_timeout_lower_bound=compile_timeout_lower_bound,
            compile_timeout_quantile=compile_timeout_quantile,
            flash_structural_search=flash_structural_search,
        )
        self._encoded_to_flat_mapping: list[tuple[int, int, int]] | None = None

    def _get_encoded_to_flat_mapping(self) -> list[tuple[int, int, int]]:
        """Build and cache mapping from encoded feature indices to flat_spec indices."""
        if self._encoded_to_flat_mapping is None:
            mapping: list[tuple[int, int, int]] = []
            offset = 0
            for flat_idx, spec in enumerate(self.config_gen.flat_spec):
                d = spec.dim()
                mapping.append((offset, offset + d, flat_idx))
                offset += d
            self._encoded_to_flat_mapping = mapping
        return self._encoded_to_flat_mapping

    @staticmethod
    def _encoded_index_to_flat_index(
        mapping: list[tuple[int, int, int]], encoded_idx: int
    ) -> int:
        """Map an encoded feature index used in tree splits to its flat_spec index."""
        for start, end, flat_idx in mapping:
            if start <= encoded_idx < end:
                return flat_idx
        raise ValueError(f"Encoded index {encoded_idx} out of range")

    def _generate_neighbors(
        self,
        base: FlatConfig,
        *,
        fixed_flat_values: Mapping[int, object] | None = None,
        config_gen: ConfigGeneration | None = None,
        num_neighbors: int | None = None,
        proposal_callback: Callable[[FlatConfig], None] | None = None,
    ) -> list[FlatConfig]:
        """
        Generate neighbors via greedy tree traversal with incremental encoding.

        For each of num_neighbors trials:
        1. Pick a random tree from the Random Forest surrogate
        2. Get its decision path for the base config
        3. Extract unique flat_spec indices from the path's split features
        4. Augment with a random block_size index and the num_warps index
        5. For each parameter on the path:
           - Generate pattern_neighbors with the configured radius
           - Score current value + neighbors with single tree (ties broken randomly)
           - Only re-encode the changed parameter's features (incremental)

        Returns all distinct candidates.
        Falls back to the parent's random neighbor generation if no surrogate is fitted.
        """
        if config_gen is not None or proposal_callback is not None:
            return super()._generate_neighbors(
                base,
                fixed_flat_values=fixed_flat_values,
                config_gen=config_gen,
                num_neighbors=num_neighbors,
                proposal_callback=proposal_callback,
            )

        surrogate = self.surrogate
        if surrogate is None or self._autotune_metrics.num_generations <= 1:
            return super()._generate_neighbors(
                base,
                fixed_flat_values=fixed_flat_values,
                num_neighbors=num_neighbors,
            )

        config_gen = self.config_gen
        mapping = self._get_encoded_to_flat_mapping()
        n_trees = len(surrogate.estimators_)
        base_list = list(base)
        base_encoded = np.array(config_gen.encode_config(base), dtype=np.float64)
        frozen = set(config_gen.overridden_flat_indices)
        if fixed_flat_values:
            frozen.update(fixed_flat_values)
        eligible_block = [i for i in config_gen.block_size_indices if i not in frozen]
        warp_idx = config_gen.num_warps_index
        tune_warps = warp_idx >= 0 and warp_idx not in frozen

        all_results: list[FlatConfig] = []

        for _ in range(self.num_neighbors if num_neighbors is None else num_neighbors):
            # 1. Pick a random tree
            tree_idx = random.randint(0, n_trees - 1)
            estimator = surrogate.estimators_[tree_idx]
            tree = estimator.tree_

            # 2. Get decision path for base config
            decision_path = estimator.decision_path(base_encoded.reshape(1, -1))
            path_node_indices = decision_path.indices.tolist()  # type: ignore[union-attr]

            # 3. Extract flat_spec indices (deduplicated, order-preserving)
            seen: set[int] = set(frozen)
            path_flat_indices: list[int] = []
            for node_id in path_node_indices:
                feat = tree.feature[node_id]  # pyrefly: ignore [missing-attribute]
                if feat >= 0:
                    flat_idx = self._encoded_index_to_flat_index(mapping, feat)
                    if flat_idx not in seen:
                        seen.add(flat_idx)
                        path_flat_indices.append(flat_idx)

            # 4. Augment with block_size and num_warps indices
            if eligible_block:
                bs_idx = random.choice(eligible_block)
                if bs_idx not in seen:
                    seen.add(bs_idx)
                    path_flat_indices.append(bs_idx)
            if tune_warps and warp_idx not in seen:
                seen.add(warp_idx)
                path_flat_indices.append(warp_idx)

            # 5. Greedy traversal with incremental encoding
            current_flat: FlatConfig = list(base)
            current_encoded = base_encoded.copy()

            for flat_idx in path_flat_indices:
                spec = config_gen.flat_spec[flat_idx]
                current_val = current_flat[flat_idx]
                neighbors = spec.pattern_neighbors(current_val, self.radius)

                if not neighbors:
                    continue

                # Build candidate encodings by patching only the changed slice
                candidate_vals = [current_val, *neighbors]
                enc_start, enc_end, _ = mapping[flat_idx]
                n_candidates = len(candidate_vals)
                candidate_encoded = np.tile(current_encoded, (n_candidates, 1))
                for i, val in enumerate(candidate_vals):
                    candidate_encoded[i, enc_start:enc_end] = spec.encode(val)

                # Score with single tree (ties broken randomly)
                probas = np.asarray(estimator.predict_proba(candidate_encoded))[:, 1]

                # Greedy: pick the best, with random tie-breaking
                max_proba = float(np.max(probas))
                top_indices = [i for i, p in enumerate(probas) if p == max_proba]
                chosen_idx = random.choice(top_indices)

                current_flat[flat_idx] = candidate_vals[chosen_idx]
                current_encoded[enc_start:enc_end] = candidate_encoded[
                    chosen_idx, enc_start:enc_end
                ]

            # Only keep if different from base
            if current_flat != base_list:
                all_results.append(list(current_flat))

        return self.shrink_neighbors(all_results)
