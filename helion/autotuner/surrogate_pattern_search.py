from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
import random
from typing import TYPE_CHECKING

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

_CUTE_FLASH_LANE_POLICY_VERSION = 14
_FLASH_TERMINAL_REFINEMENT_SCHEMA_VERSION = 2
_FLASH_TERMINAL_REFINEMENT_POLICY_VERSION = 2
_FLASH_TERMINAL_COORDINATE_POLICY = "same_leaf_full_surface_normalized_coordinate_v2"
_FLASH_TERMINAL_REFINEMENT_TARGET_MS = 200.0
_FLASH_TERMINAL_CONFIRMATION_TARGET_MS = 5000.0
_FLASH_TERMINAL_MEASUREMENT_POLICY = "mirrored_rotating_batched_wall_v2"

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

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
                self.config_gen.flash_structural_starting_path_limit(
                    minimum=max(self.copies, flash_structural_search.starting_paths),
                    retained_families=flash_structural_search.retained_families,
                    retained_candidates_per_leaf=(
                        flash_structural_search.retained_candidates_per_leaf
                    ),
                )
            )
            self._flash_family_probe_path_limit = (
                self.config_gen.flash_structural_family_probe_path_limit(
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
            policy["cute_flash_lane_policy_version"] = _CUTE_FLASH_LANE_POLICY_VERSION
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
        self.benchmark_population(self.population, desc="Initial population")

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
        if (
            phase is not None
            and phase.get("family_probe_required") is True
            and phase.get("family_probe_complete") is not True
        ):
            if self._autotune_budget_exceeded_across_ranks():
                return self._finalize()
            raise exc.AutotuneError("required CuTe flash family probe did not complete")

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
        return self.config_gen.flash_pipeline_lane_catalog().get(leaf, ())

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
        config = self.config_gen.flash_pipeline_lane_witnesses().get(
            (leaf, lane[0], lane[1])
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
        config = self.config_gen.flash_clc_lane_witnesses().get((leaf, value))
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
        """Qualify ordinary schedules, then transfer their best representatives."""
        policy = getattr(self, "flash_structural_search", None)
        if (
            not self.config_spec.cute_flash_search_enabled
            or policy is None
            or self.max_generations <= 1
        ):
            return 0

        from .._compiler.cute.cute_flash import FLASH_CLC_HEADS_PER_BATCH_KEY
        from .._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY

        if initial_population is None:
            initial_population = self.population
        initial_population = list(initial_population)
        leaf_catalog = self.config_gen.flash_structural_leaf_catalog()
        ordinary_leaves = [
            leaf for leaf in leaf_catalog if leaf.compound_exp2_packet is None
        ]
        compound_leaves = [
            leaf for leaf in leaf_catalog if leaf.compound_exp2_packet is not None
        ]
        clc_catalog = self.config_gen.flash_clc_lane_catalog()
        initial_config_ids = [
            canonical_config_id(member.config) for member in initial_population
        ]
        initial_measurements = {
            id(member): {
                "attempt_perf": (
                    member.perfs[0]
                    if member.perfs and math.isfinite(member.perfs[0])
                    else None
                ),
                "selection_perf": (
                    member.perf if member.perfs and math.isfinite(member.perf) else None
                ),
                "status": member.status,
                "source_hash": self._flash_member_source_hash(member),
                "measurement_pass_index": 0,
            }
            for member in initial_population
        }
        exact_space_raw_budget = max(
            1,
            self.initial_population,
            len(initial_population),
        )
        exact_space = self.config_gen.flash_exact_effective_search_space_configs(
            exact_space_raw_budget
        )
        measured_initial_configs = {
            member.config for member in initial_population if member.perfs
        }

        def measured_space_exhausted(configs: Sequence[Config] | None) -> bool:
            return bool(configs) and all(
                config in measured_initial_configs for config in configs or ()
            )

        def space_config_count(configs: Sequence[Config] | None) -> int | None:
            return None if configs is None else len(configs)

        def hierarchical_clc_values_covered(
            leaf: FlashStructuralLeaf,
            configs: Sequence[Config] | None,
        ) -> bool:
            catalog = clc_catalog.get(leaf)
            if catalog is None:
                return True
            present_values = {
                config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                for config in configs or ()
            }
            return set(catalog["attempted_values"]).issubset(present_values)

        def hierarchical_space_exhausted(
            leaf: FlashStructuralLeaf,
            configs: Sequence[Config] | None,
        ) -> bool:
            return measured_space_exhausted(
                configs
            ) and hierarchical_clc_values_covered(leaf, configs)

        exact_space_exhausted = measured_space_exhausted(exact_space) and all(
            hierarchical_clc_values_covered(leaf, exact_space) for leaf in clc_catalog
        )
        leaf_metrics: dict[FlashStructuralLeaf, dict[str, object]] = {}
        leaf_pipeline_lanes: dict[
            FlashStructuralLeaf, tuple[tuple[str, object], ...]
        ] = {}
        lane_metrics: dict[
            tuple[FlashStructuralLeaf, tuple[str, object]], dict[str, object]
        ] = {}
        for leaf in ordinary_leaves:
            initial_members = [
                member
                for member in initial_population
                if self._flash_structural_leaf(member) == leaf
            ]
            lanes = self._flash_pipeline_lanes(leaf)
            leaf_pipeline_lanes[leaf] = lanes
            leaf_space_configs = (
                None
                if exact_space is None
                else [
                    config
                    for config in exact_space
                    if self._flash_structural_leaf_from_config(config) == leaf
                ]
            )
            lane_space_configs = {
                lane: (
                    None
                    if exact_space is None
                    else [
                        config
                        for config in exact_space
                        if self._flash_structural_leaf_from_config(config) == leaf
                        and config.config.get(lane[0]) == lane[1]
                    ]
                )
                for lane in lanes
            }
            leaf_metrics[leaf] = {
                "family": leaf.pipeline_family,
                "compound_packet": leaf.compound_exp2_packet,
                "softmax_disc": leaf.softmax_disc,
                "initial_config_ids": [
                    canonical_config_id(member.config) for member in initial_members
                ],
                "space_exhausted": hierarchical_space_exhausted(
                    leaf, leaf_space_configs
                ),
                "space_config_count": space_config_count(leaf_space_configs),
                "ordinary_search_required": not lanes
                and not hierarchical_space_exhausted(leaf, leaf_space_configs),
                "rounds": [],
                "pipeline_lanes": [
                    {
                        "key": lane[0],
                        "value": lane[1],
                        "initial_config_ids": [
                            canonical_config_id(member.config)
                            for member in initial_members
                            if self._flash_member_matches_pipeline_lane(member, lane)
                        ],
                        "rounds": [],
                        "witness_attempted": False,
                        "witness_config_id": None,
                        "witness_succeeded": False,
                        "space_exhausted": hierarchical_space_exhausted(
                            leaf, lane_space_configs[lane]
                        ),
                        "space_config_count": space_config_count(
                            lane_space_configs[lane]
                        ),
                        "conditional_required": bool(
                            policy.conditional_candidates_per_pipeline_lane
                        )
                        and not hierarchical_space_exhausted(
                            leaf, lane_space_configs[lane]
                        ),
                        "conditional_candidate_ids": [],
                        "successful_conditional_candidate_ids": [],
                        "repair_candidate_ids": [],
                        "successful_repair_candidate_ids": [],
                        "repair_parent_decisions": [],
                        "terminal_failure_exhausted": False,
                        "complete": False,
                    }
                    for lane in lanes
                ],
            }
            cast_lanes = leaf_metrics[leaf]["pipeline_lanes"]
            assert isinstance(cast_lanes, list)
            for lane, lane_metric in zip(lanes, cast_lanes, strict=True):
                lane_metrics[(leaf, lane)] = lane_metric
        # Selection must replay the same normalized catalog that qualification
        # measured. Re-deriving it from later children could make provenance
        # base-dependent and promote an unrecorded lane.
        self._flash_qualified_pipeline_lanes = dict(leaf_pipeline_lanes)
        qualified_population = {id(member): member for member in self.population}
        candidate_ids: set[str] = set()
        leaves_with_candidates: set[FlashStructuralLeaf] = set()
        rounds_started = 0
        generation_rounds_started = 0
        rounds_completed = 0
        budget_exhausted = False
        qualification_neighbor_limit = self._flash_qualification_neighbor_limit()

        def member_measurement_state(
            member: PopulationMember,
        ) -> dict[str, object]:
            return {
                "attempt_perf": (
                    member.perfs[0]
                    if member.perfs and math.isfinite(member.perfs[0])
                    else None
                ),
                "selection_perf": (
                    member.perf if member.perfs and math.isfinite(member.perf) else None
                ),
                "status": member.status,
                "source_hash": self._flash_member_source_hash(member),
            }

        measurement_states_by_id: dict[str, dict[str, object]] = {}
        measurement_timeline: list[dict[str, object]] = []

        def record_measurement_pass(pass_index: int) -> None:
            """Record new or changed member states after one qualification pass."""
            nonlocal measurement_states_by_id
            current: dict[str, dict[str, object]] = {}
            for member in qualified_population.values():
                if not member.perfs:
                    continue
                config_id = canonical_config_id(member.config)
                state = member_measurement_state(member)
                existing = current.get(config_id)
                if existing is not None and existing != state:
                    raise AssertionError(
                        f"conflicting measurement state for config {config_id}"
                    )
                current[config_id] = state
            if not set(measurement_states_by_id) <= set(current):
                raise AssertionError(
                    "qualified measurement state removed between passes"
                )
            updates = [
                {"config_id": config_id, **current[config_id]}
                for config_id in sorted(current)
                if measurement_states_by_id.get(config_id) != current[config_id]
            ]
            measurement_timeline.append({"pass_index": pass_index, "updates": updates})
            measurement_states_by_id = current

        record_measurement_pass(0)

        def measurement_result(member: PopulationMember) -> dict[str, object]:
            """Snapshot the measurements visible at the current pass boundary."""
            state = member_measurement_state(member)
            config_id = canonical_config_id(member.config)
            if not member.perfs:
                return {
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "unknown",
                    "source_hash": None,
                    "measurement_pass_index": None,
                }
            if measurement_states_by_id.get(config_id) != state:
                raise AssertionError(
                    f"measurement state for {config_id} was not recorded at pass "
                    f"{rounds_completed}"
                )
            return {**state, "measurement_pass_index": rounds_completed}

        def decision_member_result(member: PopulationMember) -> dict[str, object]:
            """Snapshot the measurements visible to one structural decision."""
            return {
                "config_id": canonical_config_id(member.config),
                **measurement_result(member),
            }

        def ranked_decision_results(
            members: Sequence[PopulationMember],
        ) -> list[dict[str, object]]:
            return [
                decision_member_result(member)
                for member in sorted(members, key=self._flash_member_rank_key)
            ]

        def successful_leaf_members(
            leaf: FlashStructuralLeaf,
        ) -> list[PopulationMember]:
            return sorted(
                (
                    member
                    for member in qualified_population.values()
                    if self._flash_structural_leaf(member) == leaf
                    and self._flash_member_succeeded(member)
                ),
                key=self._flash_member_rank_key,
            )

        def add_members(
            members: Sequence[PopulationMember],
            round_members: dict[int, PopulationMember],
        ) -> list[str]:
            ids: list[str] = []
            for member in members:
                qualified_population[id(member)] = member
                if member.perfs:
                    continue
                round_members[id(member)] = member
                config_id = canonical_config_id(member.config)
                candidate_ids.add(config_id)
                ids.append(config_id)
            return ids

        def novel_unbenchmarked_members(
            members: Sequence[PopulationMember],
        ) -> list[PopulationMember]:
            """Keep only configs that this phase has not already measured or queued."""
            known_configs = {member.config for member in qualified_population.values()}
            result: list[PopulationMember] = []
            for member in members:
                if member.perfs or member.config in known_configs:
                    continue
                known_configs.add(member.config)
                result.append(member)
            return result

        def reserve_pass(*, counts_toward_generation_budget: bool = True) -> bool:
            """Reserve a measurement pass before candidate generation mutates state."""
            nonlocal rounds_started, generation_rounds_started, budget_exhausted
            if budget_exhausted:
                return False
            if (
                counts_toward_generation_budget
                and generation_rounds_started >= self.max_generations - 1
            ):
                budget_exhausted = True
                return False
            if not any(True for _ in self._budgeted_range(1)):
                budget_exhausted = True
                return False
            rounds_started += 1
            generation_rounds_started += int(counts_toward_generation_budget)
            return True

        def run_pass(
            round_members: dict[int, PopulationMember],
            *,
            desc: str,
        ) -> None:
            nonlocal rounds_completed
            self.population = [*qualified_population.values()]
            unbenchmarked = [
                member for member in round_members.values() if not member.perfs
            ]
            if unbenchmarked:
                # Non-budgeted anchor work extends generation zero. Counted
                # qualification passes retain the historical generation IDs,
                # while measurement_timeline records every physical pass.
                self.set_generation(generation_rounds_started)
                self.benchmark_population(unbenchmarked, desc=desc)
                self.rebenchmark_population(
                    self.population,
                    desc=f"{desc} verifying",
                )
                for member in unbenchmarked:
                    self._append_training_sample(
                        self.config_gen.encode_config(member.flat_values),
                        member.perf,
                        member.config,
                        member.fn,
                        member=member,
                    )
                self._fit_surrogate()
                self.population.sort(key=performance)
            record_measurement_pass(rounds_completed + 1)
            rounds_completed += 1

        schedule_anchor_configs = (
            self.config_gen.flash_low_confound_schedule_anchor_configs()
        )
        schedule_anchor_members: list[PopulationMember] = []
        pending_schedule_anchors: list[PopulationMember] = []
        for config in schedule_anchor_configs:
            existing = next(
                (
                    member
                    for member in qualified_population.values()
                    if member.config == config
                ),
                None,
            )
            if existing is None:
                existing = self.make_unbenchmarked(self.config_gen.flatten(config))
                if existing is None:
                    continue
                pending_schedule_anchors.append(existing)
            schedule_anchor_members.append(existing)
        schedule_anchor_pass_planned = bool(pending_schedule_anchors)
        schedule_anchor_pass_started = schedule_anchor_pass_planned and reserve_pass(
            counts_toward_generation_budget=False
        )
        if schedule_anchor_pass_started:
            schedule_anchor_round_members: dict[int, PopulationMember] = {}
            for member in pending_schedule_anchors:
                visited.add(member.config)
                anchor_ids = add_members([member], schedule_anchor_round_members)
                if anchor_ids:
                    leaf = self._flash_structural_leaf(member)
                    if leaf is not None:
                        leaves_with_candidates.add(leaf)
            run_pass(
                schedule_anchor_round_members,
                desc="Low-confound schedule anchors:",
            )
        elif pending_schedule_anchors:
            pending_ids = {id(member) for member in pending_schedule_anchors}
            schedule_anchor_members = [
                member
                for member in schedule_anchor_members
                if id(member) not in pending_ids
            ]
        schedule_anchor_results: list[dict[str, object]] = []
        for member in schedule_anchor_members:
            leaf = self._flash_structural_leaf(member)
            if leaf is None:
                continue
            schedule_anchor_results.append(
                {
                    "config_id": canonical_config_id(member.config),
                    "family": leaf.pipeline_family,
                    "compound_packet": leaf.compound_exp2_packet,
                    "softmax_disc": leaf.softmax_disc,
                    **measurement_result(member),
                }
            )
        schedule_anchor_complete = (
            (not pending_schedule_anchors or schedule_anchor_pass_started)
            and len(schedule_anchor_results) == len(schedule_anchor_configs)
            and all(
                flash_terminal_measurement_is_valid(result)
                for result in schedule_anchor_results
            )
        )

        lane_passes: dict[
            FlashStructuralLeaf,
            list[list[tuple[str, tuple[str, object] | None]]],
        ] = {}
        for leaf in ordinary_leaves:
            minimum_passes = (
                0
                if not leaf_pipeline_lanes[leaf]
                and bool(leaf_metrics[leaf]["space_exhausted"])
                else policy.qualification_rounds
            )
            lane_passes[leaf] = self._flash_lane_qualification_passes(
                leaf_pipeline_lanes[leaf],
                candidate_limit=policy.pipeline_candidates_per_leaf_per_round,
                conditional_candidates_per_lane=(
                    policy.conditional_candidates_per_pipeline_lane
                ),
                minimum_passes=minimum_passes,
                conditional_lanes=tuple(
                    lane
                    for lane in leaf_pipeline_lanes[leaf]
                    if lane_metrics[(leaf, lane)]["conditional_required"]
                ),
            )
        pipeline_pass_count = max(map(len, lane_passes.values()), default=0)
        for pass_index in range(pipeline_pass_count):
            if not reserve_pass():
                break
            round_members: dict[int, PopulationMember] = {}
            for leaf_index, leaf in enumerate(ordinary_leaves):
                jobs = (
                    lane_passes[leaf][pass_index]
                    if pass_index < len(lane_passes[leaf])
                    else []
                )
                leaf_round_ids: list[str] = []
                lane_round_ids = {lane: [] for lane in leaf_pipeline_lanes[leaf]}
                parent_decisions: list[dict[str, object]] = []
                conditional_jobs = [job for job in jobs if job[0] == "conditional"]
                conditional_limits = self._flash_lane_neighbor_limits(
                    [(job[1], 1) for job in conditional_jobs],
                    qualification_neighbor_limit,
                )
                conditional_limit_iter = iter(conditional_limits)
                neighbor_limit_by_lane: dict[tuple[str, object], int] = {}
                ordinary_neighbor_limit = 0
                for job_index, (kind, lane) in enumerate(jobs):
                    if kind == "witness":
                        assert lane is not None
                        lane_metric = lane_metrics[(leaf, lane)]
                        successful_lane_members = [
                            member
                            for member in qualified_population.values()
                            if self._flash_structural_leaf(member) == leaf
                            and self._flash_member_matches_pipeline_lane(member, lane)
                            and self._flash_member_succeeded(member)
                        ]
                        if successful_lane_members:
                            witness_candidates = sorted(
                                successful_lane_members,
                                key=self._flash_member_rank_key,
                            )
                            witness = witness_candidates[0]
                            selection_kind = "ranked_existing"
                        else:
                            witness = self._flash_pipeline_lane_witness(leaf, lane)
                            if witness is not None:
                                witness = next(
                                    (
                                        member
                                        for member in qualified_population.values()
                                        if member.config == witness.config
                                    ),
                                    witness,
                                )
                            witness_candidates = [] if witness is None else [witness]
                            selection_kind = "catalog_witness"
                        witness_decision: dict[str, object] = {
                            "job_index": job_index,
                            "kind": "witness",
                            "pipeline_lane": self._flash_pipeline_lane_metric(lane),
                            "selection_kind": selection_kind,
                            "candidate_results": [
                                decision_member_result(member)
                                for member in witness_candidates
                            ],
                            "selected_config_id": (
                                None
                                if witness is None
                                else canonical_config_id(witness.config)
                            ),
                            "generated_config_ids": [],
                        }
                        parent_decisions.append(witness_decision)
                        if witness is None:
                            continue
                        existing = next(
                            (
                                member
                                for member in qualified_population.values()
                                if member.config == witness.config
                            ),
                            None,
                        )
                        if existing is None:
                            visited.add(witness.config)
                            existing = witness
                        witness_id = canonical_config_id(existing.config)
                        lane_metric["witness_attempted"] = True
                        lane_metric["witness_config_id"] = witness_id
                        lane_round_ids[lane].append(witness_id)
                        ids = add_members([existing], round_members)
                        witness_decision["generated_config_ids"] = ids
                        leaf_round_ids.extend(ids)
                        if ids:
                            leaves_with_candidates.add(leaf)
                        continue

                    if kind == "ordinary":
                        members = successful_leaf_members(leaf)
                        if not members:
                            parent_decisions.append(
                                {
                                    "job_index": job_index,
                                    "kind": kind,
                                    "pipeline_lane": None,
                                    "selection_kind": "ranked_parent",
                                    "candidate_results": [],
                                    "selected_config_id": None,
                                    "generated_config_ids": [],
                                }
                            )
                            continue
                        member = members[0]
                        quota = policy.pipeline_candidates_per_leaf_per_round
                        neighbor_limit = qualification_neighbor_limit
                        ordinary_neighbor_limit += neighbor_limit
                        constraints = self._flash_leaf_constraints(leaf)
                    else:
                        assert lane is not None and kind == "conditional"
                        neighbor_limit = next(conditional_limit_iter)
                        neighbor_limit_by_lane[lane] = (
                            neighbor_limit_by_lane.get(lane, 0) + neighbor_limit
                        )
                        members = [
                            member
                            for member in qualified_population.values()
                            if self._flash_structural_leaf(member) == leaf
                            and member.perfs
                            if self._flash_member_matches_pipeline_lane(member, lane)
                        ]
                        if not members:
                            parent_decisions.append(
                                {
                                    "job_index": job_index,
                                    "kind": kind,
                                    "pipeline_lane": (
                                        self._flash_pipeline_lane_metric(lane)
                                    ),
                                    "selection_kind": "ranked_parent",
                                    "candidate_results": [],
                                    "selected_config_id": None,
                                    "generated_config_ids": [],
                                }
                            )
                            continue
                        member = min(members, key=self._flash_member_rank_key)
                        # One scheduled job contributes one child. Repeating the
                        # job N times makes the policy's N accounting linear.
                        quota = 1
                        constraints = (*self._flash_leaf_constraints(leaf), lane)

                    parent_candidate_results = ranked_decision_results(members)
                    parent_config_id = canonical_config_id(member.config)
                    search_copy = self._pruned_pattern_search_from(
                        pass_index
                        * max(1, len(ordinary_leaves))
                        * policy.pipeline_candidates_per_leaf_per_round
                        + leaf_index * policy.pipeline_candidates_per_leaf_per_round
                        + job_index,
                        member,
                        visited,
                        constraints,
                        selected_limit=quota + 1,
                        neighbor_limit=neighbor_limit,
                        required_leaf=leaf,
                        conditional_surface=True,
                        disable_early_stopping=True,
                    )
                    added = next(search_copy, ())
                    ids = add_members(
                        novel_unbenchmarked_members(added)[:quota],
                        round_members,
                    )
                    parent_decisions.append(
                        {
                            "job_index": job_index,
                            "kind": kind,
                            "pipeline_lane": self._flash_pipeline_lane_metric(lane),
                            "selection_kind": "ranked_parent",
                            "candidate_results": parent_candidate_results,
                            "selected_config_id": parent_config_id,
                            "generated_config_ids": ids,
                        }
                    )
                    leaf_round_ids.extend(ids)
                    if ids:
                        leaves_with_candidates.add(leaf)
                    if lane is not None:
                        lane_round_ids[lane].extend(ids)
                        cast_ids = lane_metrics[(leaf, lane)][
                            "conditional_candidate_ids"
                        ]
                        assert isinstance(cast_ids, list)
                        cast_ids.extend(ids)

                cast_rounds = leaf_metrics[leaf]["rounds"]
                assert isinstance(cast_rounds, list)
                cast_rounds.append(
                    {
                        "candidate_config_ids": leaf_round_ids,
                        "neighbor_generation_limit": ordinary_neighbor_limit
                        + sum(neighbor_limit_by_lane.values()),
                        "ordinary_neighbor_generation_limit": (ordinary_neighbor_limit),
                        "parent_decisions": parent_decisions,
                    }
                )
                for lane in leaf_pipeline_lanes[leaf]:
                    cast_rounds = lane_metrics[(leaf, lane)]["rounds"]
                    assert isinstance(cast_rounds, list)
                    cast_rounds.append(
                        {
                            "candidate_config_ids": lane_round_ids[lane],
                            "neighbor_generation_limit": neighbor_limit_by_lane.get(
                                lane, 0
                            ),
                        }
                    )
            run_pass(
                round_members,
                desc=f"Structural qualification {pass_index + 1}:",
            )

        def pipeline_lane_attempts(
            leaf: FlashStructuralLeaf,
            lane: tuple[str, object],
        ) -> list[PopulationMember]:
            """Return the measured attempts explicitly tracked for one lane."""
            metric = lane_metrics[(leaf, lane)]
            config_ids = [metric["witness_config_id"]]
            for key in ("conditional_candidate_ids", "repair_candidate_ids"):
                values = metric[key]
                assert isinstance(values, list)
                config_ids.extend(values)
            wanted = {
                config_id for config_id in config_ids if isinstance(config_id, str)
            }
            by_config_id = {
                canonical_config_id(member.config): member
                for member in qualified_population.values()
                if canonical_config_id(member.config) in wanted
                and self._flash_structural_leaf(member) == leaf
                and self._flash_member_matches_pipeline_lane(member, lane)
                and member.perfs
            }
            return [
                by_config_id[config_id]
                for config_id in sorted(wanted & by_config_id.keys())
            ]

        pipeline_repair_pass_count = 0
        repair_pass_ordinal = 0
        for repair_index in range(policy.qualification_failure_retries):
            repair_job_lanes: dict[FlashStructuralLeaf, list[tuple[str, object]]] = {}
            for leaf in ordinary_leaves:
                leaf_lanes = []
                for lane in leaf_pipeline_lanes[leaf]:
                    attempts = pipeline_lane_attempts(leaf, lane)
                    if not attempts or not all(
                        self._flash_member_has_retryable_failure(member)
                        for member in attempts
                    ):
                        continue
                    leaf_lanes.append(lane)
                repair_job_lanes[leaf] = leaf_lanes
            if not any(repair_job_lanes.values()):
                break
            batch_index = 0
            while any(repair_job_lanes.values()):
                current_jobs: dict[
                    FlashStructuralLeaf,
                    list[tuple[tuple[str, object], list[PopulationMember]]],
                ] = {}
                for leaf in ordinary_leaves:
                    leaf_jobs = []
                    pending_lanes = []
                    for lane in repair_job_lanes[leaf]:
                        attempts = pipeline_lane_attempts(leaf, lane)
                        if attempts and all(
                            self._flash_member_has_retryable_failure(member)
                            for member in attempts
                        ):
                            if (
                                len(leaf_jobs)
                                < policy.pipeline_candidates_per_leaf_per_round
                            ):
                                leaf_jobs.append(
                                    (
                                        lane,
                                        sorted(
                                            attempts,
                                            key=self._flash_member_rank_key,
                                        ),
                                    )
                                )
                            else:
                                pending_lanes.append(lane)
                    repair_job_lanes[leaf] = pending_lanes
                    current_jobs[leaf] = leaf_jobs
                if not any(current_jobs.values()):
                    break
                if not reserve_pass():
                    break
                pipeline_repair_pass_count += 1
                repair_pass_ordinal += 1
                round_members = {}
                for leaf_index, leaf in enumerate(ordinary_leaves):
                    jobs = current_jobs[leaf]
                    neighbor_limits = self._flash_lane_neighbor_limits(
                        [(lane, 1) for lane, _members in jobs],
                        qualification_neighbor_limit,
                    )
                    leaf_round_ids: list[str] = []
                    parent_decisions: list[dict[str, object]] = []
                    lane_round_ids = {lane: [] for lane in leaf_pipeline_lanes[leaf]}
                    lane_neighbor_limits = {
                        lane: limit
                        for (lane, _members), limit in zip(
                            jobs, neighbor_limits, strict=True
                        )
                    }
                    for job_index, ((lane, members), neighbor_limit) in enumerate(
                        zip(jobs, neighbor_limits, strict=True)
                    ):
                        parent = members[0]
                        parent_config_id = canonical_config_id(parent.config)
                        search_copy = self._pruned_pattern_search_from(
                            200_000
                            + repair_index * 10_000
                            + batch_index * max(1, len(ordinary_leaves)) * 100
                            + leaf_index * 100
                            + job_index,
                            parent,
                            visited,
                            (*self._flash_leaf_constraints(leaf), lane),
                            selected_limit=2,
                            neighbor_limit=neighbor_limit,
                            required_leaf=leaf,
                            conditional_surface=True,
                            disable_early_stopping=True,
                        )
                        added = next(search_copy, ())
                        ids = add_members(
                            novel_unbenchmarked_members(added)[:1],
                            round_members,
                        )
                        metric = lane_metrics[(leaf, lane)]
                        cast_ids = metric["repair_candidate_ids"]
                        cast_decisions = metric["repair_parent_decisions"]
                        assert isinstance(cast_ids, list)
                        assert isinstance(cast_decisions, list)
                        cast_ids.extend(ids)
                        lane_decision = {
                            "repair_index": repair_index,
                            "candidate_results": ranked_decision_results(members),
                            "selected_config_id": parent_config_id,
                            "generated_config_ids": ids,
                        }
                        cast_decisions.append(lane_decision)
                        parent_decisions.append(
                            {
                                "job_index": job_index,
                                "kind": "failure_repair",
                                "pipeline_lane": self._flash_pipeline_lane_metric(lane),
                                "selection_kind": "ranked_failed_parent",
                                **lane_decision,
                            }
                        )
                        leaf_round_ids.extend(ids)
                        lane_round_ids[lane].extend(ids)
                        if ids:
                            leaves_with_candidates.add(leaf)
                    cast_rounds = leaf_metrics[leaf]["rounds"]
                    assert isinstance(cast_rounds, list)
                    cast_rounds.append(
                        {
                            "candidate_config_ids": leaf_round_ids,
                            "neighbor_generation_limit": sum(neighbor_limits),
                            "ordinary_neighbor_generation_limit": 0,
                            "parent_decisions": parent_decisions,
                        }
                    )
                    for lane in leaf_pipeline_lanes[leaf]:
                        cast_lane_rounds = lane_metrics[(leaf, lane)]["rounds"]
                        assert isinstance(cast_lane_rounds, list)
                        cast_lane_rounds.append(
                            {
                                "candidate_config_ids": lane_round_ids[lane],
                                "neighbor_generation_limit": (
                                    lane_neighbor_limits.get(lane, 0)
                                ),
                            }
                        )
                run_pass(
                    round_members,
                    desc=(
                        f"Structural qualification failure repairs "
                        f"{repair_pass_ordinal}:"
                    ),
                )
                batch_index += 1
            if budget_exhausted:
                break

        clc_value_space_exhausted = {
            leaf: {
                value: measured_space_exhausted(
                    None
                    if exact_space is None
                    else [
                        config
                        for config in exact_space
                        if self._flash_structural_leaf_from_config(config) == leaf
                        and config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY) == value
                    ]
                )
                for value in catalog["attempted_values"]
            }
            for leaf, catalog in clc_catalog.items()
            if leaf in ordinary_leaves
        }
        clc_metrics: dict[FlashStructuralLeaf, dict[str, object]] = {
            leaf: {
                "family": leaf.pipeline_family,
                "softmax_disc": leaf.softmax_disc,
                "space_exhausted": bool(leaf_metrics[leaf]["space_exhausted"]),
                "legal_values": list(catalog["legal_values"]),
                "search_values": list(catalog["search_values"]),
                "anchor_values": list(catalog["anchor_values"]),
                "refinement_values": list(catalog["refinement_values"]),
                "planned_values": list(catalog["attempted_values"]),
                "attempted_values": [],
                "witness_config_ids": {},
                "witness_repair_candidate_ids": {},
                "witness_repair_parent_decisions": [],
                "value_space_exhausted": {
                    str(value): exhausted
                    for value, exhausted in clc_value_space_exhausted[leaf].items()
                },
                "witness_candidate_results": [],
                "witness_selection_results": [],
                "selected_values": [],
                "selected_config_ids": [],
                "conditional_values": [],
                "conditional_neighbor_generation_limit": 0,
                "conditional_parent_decisions": [],
                "conditional_repair_candidate_ids": {},
                "conditional_repair_parent_decisions": [],
                "retained_values": [],
                "retained_config_ids": [],
                "retained_value_decisions": [],
                "retained_ranking_results": [],
                "conditional_candidate_ids": {},
                "combination_required": not bool(leaf_metrics[leaf]["space_exhausted"]),
                "depth_selection": {
                    "candidate_results": [],
                    "selected_representatives": [],
                },
                "combination_candidate_ids": [],
                "combination_depth_config_ids": [],
                "combination_divisor_values": [],
                "combination_cells": [],
                "combination_projection_complete": True,
                "successful_combination_depth_config_ids": [],
                "successful_combination_divisor_values": [],
                "combination_row_coverage_complete": True,
                "combination_column_coverage_complete": True,
                "combination_failure_statuses_allowed": True,
            }
            for leaf, catalog in clc_catalog.items()
            if leaf in ordinary_leaves
        }
        # All legal CLC decompositions share one common-context generation. A
        # per-generation slice would make coverage depend on the bounded
        # generation budget for highly composite B*H grids.
        clc_witness_pass_count = int(
            any(catalog["attempted_values"] for catalog in clc_catalog.values())
        )
        for clc_pass_index in range(clc_witness_pass_count):
            if not reserve_pass():
                break
            round_members = {}
            for leaf, catalog in clc_catalog.items():
                values = catalog["attempted_values"]
                for value in values:
                    witness = self._flash_clc_lane_witness(leaf, value)
                    if witness is None:
                        continue
                    existing = next(
                        (
                            member
                            for member in qualified_population.values()
                            if member.config == witness.config
                        ),
                        None,
                    )
                    if existing is None:
                        visited.add(witness.config)
                        existing = witness
                    cast_attempted = clc_metrics[leaf]["attempted_values"]
                    assert isinstance(cast_attempted, list)
                    cast_attempted.append(value)
                    cast_witnesses = clc_metrics[leaf]["witness_config_ids"]
                    assert isinstance(cast_witnesses, dict)
                    cast_witnesses[str(value)] = canonical_config_id(existing.config)
                    ids = add_members([existing], round_members)
                    if ids:
                        leaves_with_candidates.add(leaf)
            run_pass(
                round_members,
                desc=f"CLC divisor witnesses {clc_pass_index + 1}:",
            )

        def clc_member(
            leaf: FlashStructuralLeaf,
            value: int,
            config_id: object,
        ) -> PopulationMember | None:
            if not isinstance(config_id, str):
                return None
            members = [
                member
                for member in qualified_population.values()
                if canonical_config_id(member.config) == config_id
                and self._flash_structural_leaf(member) == leaf
                and member.config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY) == value
            ]
            return min(members, key=self._flash_member_rank_key, default=None)

        def clc_attempt_members(
            leaf: FlashStructuralLeaf,
            value: int,
            *,
            primary_key: str,
            repair_key: str,
        ) -> list[PopulationMember]:
            """Return all measured primary and repair attempts for one divisor."""
            metric = clc_metrics[leaf]
            primary = metric[primary_key]
            assert isinstance(primary, dict)
            primary_value = primary.get(str(value))
            config_ids = (
                [primary_value]
                if isinstance(primary_value, str)
                else list(primary_value or ())
            )
            repairs = metric[repair_key]
            assert isinstance(repairs, dict)
            config_ids.extend(repairs.get(str(value), ()))
            members: dict[str, PopulationMember] = {}
            for config_id in config_ids:
                member = clc_member(leaf, value, config_id)
                if member is not None and member.perfs:
                    members[canonical_config_id(member.config)] = member
            return sorted(members.values(), key=self._flash_member_rank_key)

        def run_clc_failure_repairs(
            values_by_leaf: Mapping[FlashStructuralLeaf, Sequence[int]],
            *,
            primary_key: str,
            repair_key: str,
            decision_key: str,
            kind: str,
            desc: str,
            missing_attempt_parent_keys: tuple[str, str] | None = None,
        ) -> int:
            """Try one bounded alternate for each failed mandatory CLC obligation."""

            def repair_parents(
                leaf: FlashStructuralLeaf,
                value: int,
            ) -> list[PopulationMember]:
                attempts = clc_attempt_members(
                    leaf,
                    value,
                    primary_key=primary_key,
                    repair_key=repair_key,
                )
                if attempts:
                    if all(
                        self._flash_member_has_retryable_failure(member)
                        for member in attempts
                    ):
                        return attempts
                    return []
                if missing_attempt_parent_keys is None:
                    return []
                fallback_primary_key, fallback_repair_key = missing_attempt_parent_keys
                return [
                    member
                    for member in clc_attempt_members(
                        leaf,
                        value,
                        primary_key=fallback_primary_key,
                        repair_key=fallback_repair_key,
                    )
                    if self._flash_member_succeeded(member)
                ][:1]

            repair_pass_count = 0
            repair_pass_ordinal = 0
            for repair_index in range(policy.qualification_failure_retries):
                repair_job_values: dict[FlashStructuralLeaf, list[int]] = {}
                for leaf in clc_metrics:
                    leaf_values = []
                    for value in values_by_leaf.get(leaf, ()):
                        if not repair_parents(leaf, value):
                            continue
                        leaf_values.append(value)
                    repair_job_values[leaf] = leaf_values
                if not any(repair_job_values.values()):
                    break
                batch_index = 0
                while any(repair_job_values.values()):
                    current_jobs: dict[
                        FlashStructuralLeaf, list[tuple[int, list[PopulationMember]]]
                    ] = {}
                    for leaf in clc_metrics:
                        leaf_jobs = []
                        pending_values = []
                        for value in repair_job_values[leaf]:
                            parents = repair_parents(leaf, value)
                            if parents:
                                if (
                                    len(leaf_jobs)
                                    < policy.pipeline_candidates_per_leaf_per_round
                                ):
                                    leaf_jobs.append(
                                        (
                                            value,
                                            sorted(
                                                parents,
                                                key=self._flash_member_rank_key,
                                            ),
                                        )
                                    )
                                else:
                                    pending_values.append(value)
                        repair_job_values[leaf] = pending_values
                        current_jobs[leaf] = leaf_jobs
                    if not any(current_jobs.values()):
                        break
                    if not reserve_pass():
                        break
                    repair_pass_count += 1
                    repair_pass_ordinal += 1
                    round_members: dict[int, PopulationMember] = {}
                    for leaf_index, leaf in enumerate(clc_metrics):
                        jobs = current_jobs[leaf]
                        neighbor_limits = self._flash_lane_neighbor_limits(
                            [
                                ((FLASH_CLC_HEADS_PER_BATCH_KEY, value), 1)
                                for value, _members in jobs
                            ],
                            qualification_neighbor_limit,
                        )
                        for job_index, ((value, members), neighbor_limit) in enumerate(
                            zip(jobs, neighbor_limits, strict=True)
                        ):
                            parent = members[0]
                            parent_config_id = canonical_config_id(parent.config)
                            lane = (FLASH_CLC_HEADS_PER_BATCH_KEY, value)
                            search_copy = self._pruned_pattern_search_from(
                                300_000
                                + repair_index * 10_000
                                + batch_index * max(1, len(clc_metrics)) * 100
                                + leaf_index * 100
                                + job_index,
                                parent,
                                visited,
                                (*self._flash_leaf_constraints(leaf), lane),
                                selected_limit=2,
                                neighbor_limit=neighbor_limit,
                                required_leaf=leaf,
                                conditional_surface=True,
                                disable_early_stopping=True,
                            )
                            added = next(search_copy, ())
                            ids = add_members(
                                novel_unbenchmarked_members(added)[:1],
                                round_members,
                            )
                            metric = clc_metrics[leaf]
                            repair_ids = metric[repair_key]
                            repair_decisions = metric[decision_key]
                            assert isinstance(repair_ids, dict)
                            assert isinstance(repair_decisions, list)
                            repair_ids.setdefault(str(value), []).extend(ids)
                            repair_decisions.append(
                                {
                                    "kind": kind,
                                    "value": value,
                                    "repair_index": repair_index,
                                    "candidate_results": ranked_decision_results(
                                        members
                                    ),
                                    "selected_config_id": parent_config_id,
                                    "generated_config_ids": ids,
                                    "neighbor_generation_limit": neighbor_limit,
                                }
                            )
                            if ids:
                                leaves_with_candidates.add(leaf)
                    run_pass(
                        round_members,
                        desc=f"{desc} {repair_pass_ordinal}:",
                    )
                    batch_index += 1
                if budget_exhausted:
                    break
            return repair_pass_count

        clc_witness_repair_pass_count = run_clc_failure_repairs(
            {
                leaf: tuple(catalog["attempted_values"])
                for leaf, catalog in clc_catalog.items()
                if leaf in clc_metrics
            },
            primary_key="witness_config_ids",
            repair_key="witness_repair_candidate_ids",
            decision_key="witness_repair_parent_decisions",
            kind="witness_failure_repair",
            desc="CLC divisor witness failure repairs",
        )

        def clc_witness_member(
            leaf: FlashStructuralLeaf, value: int
        ) -> PopulationMember | None:
            attempts = clc_attempt_members(
                leaf,
                value,
                primary_key="witness_config_ids",
                repair_key="witness_repair_candidate_ids",
            )
            return next(
                (member for member in attempts if self._flash_member_succeeded(member)),
                None,
            )

        def ranked_clc_witness_values(
            leaf: FlashStructuralLeaf, values: Sequence[int]
        ) -> list[tuple[int, PopulationMember]]:
            """Rank one dedicated common-context witness per divisor."""
            ranked: list[tuple[int, PopulationMember]] = []
            for value in values:
                member = clc_witness_member(leaf, value)
                if member is not None:
                    ranked.append((value, member))
            return sorted(
                ranked,
                key=lambda item: (
                    self._flash_member_rank_key(item[1]),
                    item[0],
                ),
            )

        def tracked_clc_candidates(
            leaf: FlashStructuralLeaf, value: int
        ) -> list[PopulationMember]:
            """Rank every tracked candidate for one selected divisor."""
            candidates = {
                canonical_config_id(member.config): member
                for member in clc_attempt_members(
                    leaf,
                    value,
                    primary_key="witness_config_ids",
                    repair_key="witness_repair_candidate_ids",
                )
            }
            for member in clc_attempt_members(
                leaf,
                value,
                primary_key="conditional_candidate_ids",
                repair_key="conditional_repair_candidate_ids",
            ):
                candidates[canonical_config_id(member.config)] = member
            return sorted(candidates.values(), key=self._flash_member_rank_key)

        def clc_scoped_candidates(
            leaf: FlashStructuralLeaf, value: int
        ) -> list[PopulationMember]:
            return [
                member
                for member in tracked_clc_candidates(leaf, value)
                if self._flash_member_succeeded(member)
            ]

        clc_selected_values: dict[FlashStructuralLeaf, tuple[int, ...]] = {}
        clc_conditional_values: dict[FlashStructuralLeaf, tuple[int, ...]] = {}
        for leaf, catalog in clc_catalog.items():
            ranked = ranked_clc_witness_values(leaf, catalog["attempted_values"])
            selected = ranked
            selected_values = tuple(value for value, _member in selected)
            conditional_values = tuple(
                value
                for value in selected_values
                if not clc_value_space_exhausted[leaf][value]
            )
            clc_selected_values[leaf] = selected_values
            clc_conditional_values[leaf] = conditional_values
            witness_candidate_results: list[dict[str, object]] = []
            for value in catalog["attempted_values"]:
                members = clc_attempt_members(
                    leaf,
                    value,
                    primary_key="witness_config_ids",
                    repair_key="witness_repair_candidate_ids",
                )
                for member in members:
                    witness_candidate_results.append(
                        {"value": value, **decision_member_result(member)}
                    )
            clc_metrics[leaf]["witness_candidate_results"] = witness_candidate_results
            clc_metrics[leaf]["witness_selection_results"] = [
                {"value": value, **decision_member_result(member)}
                for value, member in ranked
            ]
            clc_metrics[leaf]["selected_values"] = list(selected_values)
            clc_metrics[leaf]["selected_config_ids"] = [
                canonical_config_id(member.config) for _value, member in selected
            ]
            clc_metrics[leaf]["conditional_values"] = list(conditional_values)

        if any(clc_conditional_values.values()) and reserve_pass():
            round_members = {}
            for leaf_index, (leaf, values) in enumerate(clc_conditional_values.items()):
                conditional_neighbor_limit = max(
                    qualification_neighbor_limit,
                    len(values),
                )
                clc_metrics[leaf]["conditional_neighbor_generation_limit"] = (
                    conditional_neighbor_limit
                )
                neighbor_limits = self._flash_lane_neighbor_limits(
                    [((FLASH_CLC_HEADS_PER_BATCH_KEY, value), 1) for value in values],
                    conditional_neighbor_limit,
                )
                for value_index, (value, neighbor_limit) in enumerate(
                    zip(values, neighbor_limits, strict=True)
                ):
                    ranked = ranked_clc_witness_values(leaf, (value,))
                    if not ranked:
                        witness_config_ids = clc_metrics[leaf]["witness_config_ids"]
                        assert isinstance(witness_config_ids, dict)
                        witness = clc_member(
                            leaf,
                            value,
                            witness_config_ids.get(str(value)),
                        )
                        cast_decisions = clc_metrics[leaf][
                            "conditional_parent_decisions"
                        ]
                        assert isinstance(cast_decisions, list)
                        cast_decisions.append(
                            {
                                "value": value,
                                "candidate_results": (
                                    []
                                    if witness is None
                                    else [decision_member_result(witness)]
                                ),
                                "selected_config_id": None,
                                "generated_config_ids": [],
                                "neighbor_generation_limit": neighbor_limit,
                            }
                        )
                        continue
                    member = ranked[0][1]
                    parent_result = decision_member_result(member)
                    parent_config_id = canonical_config_id(member.config)
                    lane = (FLASH_CLC_HEADS_PER_BATCH_KEY, value)
                    search_copy = self._pruned_pattern_search_from(
                        100_000 + leaf_index * max(1, len(values)) + value_index,
                        member,
                        visited,
                        (*self._flash_leaf_constraints(leaf), lane),
                        selected_limit=2,
                        neighbor_limit=neighbor_limit,
                        required_leaf=leaf,
                        conditional_surface=True,
                        disable_early_stopping=True,
                    )
                    added = next(search_copy, ())
                    ids = add_members(
                        novel_unbenchmarked_members(added)[:1],
                        round_members,
                    )
                    cast_ids = clc_metrics[leaf]["conditional_candidate_ids"]
                    assert isinstance(cast_ids, dict)
                    cast_ids[str(value)] = ids
                    cast_decisions = clc_metrics[leaf]["conditional_parent_decisions"]
                    assert isinstance(cast_decisions, list)
                    cast_decisions.append(
                        {
                            "value": value,
                            "candidate_results": [parent_result],
                            "selected_config_id": parent_config_id,
                            "generated_config_ids": ids,
                            "neighbor_generation_limit": neighbor_limit,
                        }
                    )
            run_pass(round_members, desc="CLC divisor conditional children:")

        clc_conditional_repair_pass_count = run_clc_failure_repairs(
            clc_conditional_values,
            primary_key="conditional_candidate_ids",
            repair_key="conditional_repair_candidate_ids",
            decision_key="conditional_repair_parent_decisions",
            kind="conditional_failure_repair",
            desc="CLC divisor conditional failure repairs",
            missing_attempt_parent_keys=(
                "witness_config_ids",
                "witness_repair_candidate_ids",
            ),
        )

        retained_clc_members: dict[
            FlashStructuralLeaf, list[tuple[int, PopulationMember]]
        ] = {}
        for leaf in clc_catalog:
            value_decisions: list[dict[str, object]] = []
            representatives: list[tuple[int, PopulationMember]] = []
            for value in clc_selected_values[leaf]:
                tracked_candidates = tracked_clc_candidates(leaf, value)
                candidates = clc_scoped_candidates(leaf, value)
                selected_member = candidates[0] if candidates else None
                value_decisions.append(
                    {
                        "value": value,
                        "candidate_results": [
                            decision_member_result(member)
                            for member in tracked_candidates
                        ],
                        "selected_config_id": (
                            None
                            if selected_member is None
                            else canonical_config_id(selected_member.config)
                        ),
                    }
                )
                if selected_member is not None:
                    representatives.append((value, selected_member))
            ranked = sorted(
                representatives,
                key=lambda item: (
                    self._flash_member_rank_key(item[1]),
                    item[0],
                ),
            )
            retained_clc_members[leaf] = ranked
            clc_metrics[leaf]["retained_value_decisions"] = value_decisions
            clc_metrics[leaf]["retained_ranking_results"] = [
                {"value": value, **decision_member_result(member)}
                for value, member in sorted(
                    representatives,
                    key=lambda item: (
                        self._flash_member_rank_key(item[1]),
                        item[0],
                    ),
                )
            ]
            clc_metrics[leaf]["retained_values"] = [value for value, _ in ranked]
            clc_metrics[leaf]["retained_config_ids"] = [
                canonical_config_id(member.config) for _value, member in ranked
            ]

        combination_members: dict[FlashStructuralLeaf, list[PopulationMember]] = {}
        clc_combination_leaves = [
            leaf
            for leaf in clc_catalog
            if bool(clc_metrics[leaf]["combination_required"])
        ]
        if clc_combination_leaves and reserve_pass():
            combination_round_members: dict[int, PopulationMember] = {}
            for leaf in clc_combination_leaves:
                divisor_members = retained_clc_members[leaf]
                depth_candidates = successful_leaf_members(leaf)
                depth_representatives = self._flash_lane_diverse_members(
                    depth_candidates,
                    leaf_pipeline_lanes[leaf],
                    policy.retained_candidates_per_leaf,
                )
                clc_metrics[leaf]["depth_selection"] = {
                    "candidate_results": [
                        decision_member_result(member) for member in depth_candidates
                    ],
                    "selected_representatives": [
                        {
                            "config_id": canonical_config_id(member.config),
                            "assigned_pipeline_lane": (
                                self._flash_pipeline_lane_metric(lane)
                            ),
                        }
                        for member, lane in depth_representatives
                    ],
                }
                depth_members = [member for member, _lane in depth_representatives]
                depth_config_ids = [
                    canonical_config_id(member.config) for member in depth_members
                ]
                divisor_values = [value for value, _member in divisor_members]
                clc_metrics[leaf]["combination_depth_config_ids"] = depth_config_ids
                clc_metrics[leaf]["combination_divisor_values"] = divisor_values
                combined: list[PopulationMember] = []
                seen_configs: set[Config] = set()
                cells: list[dict[str, object]] = []
                for depth_member, depth_config_id in zip(
                    depth_members, depth_config_ids, strict=True
                ):
                    for value, _divisor_member in divisor_members:
                        candidate = self._flash_clc_depth_variant(
                            depth_member,
                            value,
                            expected_leaf=leaf,
                        )
                        if candidate is None:
                            cells.append(
                                {
                                    "depth_config_id": depth_config_id,
                                    "divisor_value": value,
                                    "projected_config_id": None,
                                    "config_id": None,
                                    "attempt_perf": None,
                                    "selection_perf": None,
                                    "status": "projection_rejected",
                                    "source_hash": None,
                                    "measurement_pass_index": None,
                                }
                            )
                            continue
                        existing = next(
                            (
                                member
                                for member in qualified_population.values()
                                if member.config == candidate.config
                            ),
                            None,
                        )
                        combined_member = candidate if existing is None else existing
                        projected_config_id = canonical_config_id(
                            combined_member.config
                        )
                        cells.append(
                            {
                                "depth_config_id": depth_config_id,
                                "divisor_value": value,
                                "projected_config_id": projected_config_id,
                            }
                        )
                        if candidate.config in seen_configs:
                            continue
                        seen_configs.add(candidate.config)
                        if existing is None:
                            visited.add(candidate.config)
                        combined.append(combined_member)
                        add_members([combined_member], combination_round_members)
                combination_members[leaf] = combined
                cast_ids = clc_metrics[leaf]["combination_candidate_ids"]
                assert isinstance(cast_ids, list)
                cast_ids.extend(
                    canonical_config_id(member.config) for member in combined
                )
                clc_metrics[leaf]["combination_cells"] = cells
            run_pass(
                combination_round_members,
                desc="CLC depth/divisor combinations:",
            )
            for leaf in clc_combination_leaves:
                metric_cells = clc_metrics[leaf]["combination_cells"]
                assert isinstance(metric_cells, list)
                members_by_id = {
                    canonical_config_id(member.config): member
                    for member in qualified_population.values()
                }
                for cell in metric_cells:
                    assert isinstance(cell, dict)
                    projected_config_id = cell["projected_config_id"]
                    if not isinstance(projected_config_id, str):
                        continue
                    member = members_by_id[projected_config_id]
                    cell.update(decision_member_result(member))

        compound_transfer_metrics: list[dict[str, object]] = []
        compound_round_members: dict[int, PopulationMember] = {}
        compound_transfer_members: dict[str, PopulationMember] = {}
        qualified_compound_config_ids: dict[FlashStructuralLeaf, set[str]] = {}
        compound_pass_reserved = bool(compound_leaves) and reserve_pass()
        compound_states: dict[FlashStructuralLeaf, dict[str, object]] = {}
        ordinary_protocol_leaves = {
            (leaf.pipeline_family, leaf.softmax_disc): leaf for leaf in ordinary_leaves
        }
        compound_catalog_errors: list[dict[str, object]] = []

        def add_compound_transfers(
            compound_leaf: FlashStructuralLeaf,
            *,
            count: int,
            round_members: dict[int, PopulationMember],
        ) -> list[str]:
            """Project the next ranked sources into distinct compound candidates."""
            state = compound_states[compound_leaf]
            source_pool = state["source_pool"]
            seen_transfers = state["seen_transfers"]
            transfers = state["transfers"]
            source_selection = state["source_selection"]
            assert isinstance(source_pool, list)
            assert isinstance(seen_transfers, set)
            assert isinstance(transfers, list)
            assert isinstance(source_selection, dict)
            attempted_source_ids = source_selection["attempted_config_ids"]
            selected_source_ids = source_selection["selected_config_ids"]
            assert isinstance(attempted_source_ids, list)
            assert isinstance(selected_source_ids, list)
            generated_ids: list[str] = []
            source_index = state["next_source_index"]
            assert isinstance(source_index, int)
            while count > 0 and source_index < len(source_pool):
                source_member = source_pool[source_index]
                source_index += 1
                state["next_source_index"] = source_index
                source_config_id = canonical_config_id(source_member.config)
                attempted_source_ids.append(source_config_id)
                candidate = self._flash_compound_variant(
                    source_member,
                    compound_leaf.compound_exp2_packet,
                    expected_leaf=compound_leaf,
                )
                if candidate is None or candidate.config in seen_transfers:
                    continue
                seen_transfers.add(candidate.config)
                existing = next(
                    (
                        member
                        for member in qualified_population.values()
                        if member.config == candidate.config
                    ),
                    None,
                )
                if (
                    existing is not None
                    and existing.perfs
                    and self._flash_member_has_retryable_failure(existing)
                ):
                    # A later source may project to a viable candidate. Preserve
                    # this source attempt in the immutable decision record, but
                    # do not spend one of the leaf's transfer slots on a known
                    # retryable failure.
                    continue
                if existing is None:
                    visited.add(candidate.config)
                transferred = candidate if existing is None else existing
                ids = add_members([transferred], round_members)
                transferred_config_id = canonical_config_id(transferred.config)
                generated_ids.append(transferred_config_id)
                compound_transfer_members[transferred_config_id] = transferred
                selected_source_ids.append(source_config_id)
                transfers.append(
                    {
                        "source_config_id": source_config_id,
                        "source_config": copy.deepcopy(source_member.config.config),
                        "transferred_config_id": transferred_config_id,
                        "projection_overrides": {
                            FLASH_EXP2_PACKET_KEY: compound_leaf.compound_exp2_packet
                        },
                        "projected_config_id": canonical_config_id(candidate.config),
                        "projected_config": copy.deepcopy(candidate.config.config),
                        "preserved_pipeline_values": {
                            key: source_member.config.config[key]
                            for key in self._flash_pipeline_qualification_keys()
                            if key in source_member.config.config
                        },
                    }
                )
                if ids:
                    leaves_with_candidates.add(compound_leaf)
                count -= 1
            return generated_ids

        for compound_leaf in compound_leaves:
            ordinary_leaf = ordinary_protocol_leaves.get(
                (compound_leaf.pipeline_family, compound_leaf.softmax_disc)
            )
            if ordinary_leaf is None:
                catalog_error: dict[str, object] = {
                    "family": compound_leaf.pipeline_family,
                    "compound_packet": compound_leaf.compound_exp2_packet,
                    "softmax_disc": compound_leaf.softmax_disc,
                    "error": "missing_ordinary_protocol_leaf",
                    "required_parent": {
                        "family": compound_leaf.pipeline_family,
                        "compound_packet": None,
                        "softmax_disc": compound_leaf.softmax_disc,
                    },
                }
                compound_catalog_errors.append(catalog_error)
                compound_transfer_metrics.append(
                    {
                        "family": compound_leaf.pipeline_family,
                        "compound_packet": compound_leaf.compound_exp2_packet,
                        "softmax_disc": compound_leaf.softmax_disc,
                        "limit": policy.retained_candidates_per_leaf,
                        "transfer_target_count": 0,
                        "transfer_count": 0,
                        "primary_transfer_config_ids": [],
                        "backfill_rounds": [],
                        "successful_transfer_config_ids": [],
                        "qualified_transfer_config_ids": [],
                        "failure_statuses_allowed": False,
                        "source_selection": {
                            "candidate_results": [],
                            "combination_prefix_count": 0,
                            "attempted_config_ids": [],
                            "selected_config_ids": [],
                        },
                        "transfers": [],
                        "catalog_error": catalog_error["error"],
                        "complete": False,
                    }
                )
                qualified_compound_config_ids[compound_leaf] = set()
                continue
            transfers: list[dict[str, object]] = []
            source_selection: dict[str, object] = {
                "candidate_results": [],
                "combination_prefix_count": 0,
                "attempted_config_ids": [],
                "selected_config_ids": [],
            }
            combined_source_pool = sorted(
                (
                    member
                    for member in combination_members.get(ordinary_leaf, [])
                    if self._flash_member_succeeded(member)
                ),
                key=self._flash_member_rank_key,
            )
            source_configs = {member.config for member in combined_source_pool}
            source_pool = [
                *combined_source_pool,
                *(
                    member
                    for member in successful_leaf_members(ordinary_leaf)
                    if member.config not in source_configs
                ),
            ]
            source_selection["candidate_results"] = [
                decision_member_result(member) for member in source_pool
            ]
            source_selection["combination_prefix_count"] = len(combined_source_pool)
            metric: dict[str, object] = {
                "family": compound_leaf.pipeline_family,
                "compound_packet": compound_leaf.compound_exp2_packet,
                "softmax_disc": compound_leaf.softmax_disc,
                "limit": policy.retained_candidates_per_leaf,
                "transfer_target_count": 0,
                "transfer_count": 0,
                "primary_transfer_config_ids": [],
                "backfill_rounds": [],
                "successful_transfer_config_ids": [],
                "qualified_transfer_config_ids": [],
                "failure_statuses_allowed": True,
                "source_selection": source_selection,
                "transfers": transfers,
                "complete": False,
            }
            compound_states[compound_leaf] = {
                "source_pool": source_pool,
                "next_source_index": 0,
                "seen_transfers": set(),
                "transfers": transfers,
                "source_selection": source_selection,
                "metric": metric,
            }
            compound_transfer_metrics.append(metric)
            if compound_pass_reserved:
                primary_ids = add_compound_transfers(
                    compound_leaf,
                    count=policy.retained_candidates_per_leaf,
                    round_members=compound_round_members,
                )
                metric["primary_transfer_config_ids"] = primary_ids
                metric["transfer_target_count"] = len(primary_ids)
        if compound_pass_reserved:
            run_pass(compound_round_members, desc="Compound packet transfers:")

        compound_backfill_pass_count = 0
        for repair_index in range(policy.qualification_failure_retries):
            repair_needs: dict[FlashStructuralLeaf, tuple[int, list[str]]] = {}
            for compound_leaf, state in compound_states.items():
                state_metric = state["metric"]
                assert isinstance(state_metric, dict)
                target_count = state_metric["transfer_target_count"]
                state_transfers = state_metric["transfers"]
                assert isinstance(target_count, int)
                assert isinstance(state_transfers, list)
                attempted_members = [
                    compound_transfer_members[transfer["transferred_config_id"]]
                    for transfer in state_transfers
                    if isinstance(transfer, dict)
                    and isinstance(transfer.get("transferred_config_id"), str)
                ]
                successful_count = sum(
                    self._flash_member_succeeded(member) for member in attempted_members
                )
                failed_members = [
                    member
                    for member in attempted_members
                    if not self._flash_member_succeeded(member)
                ]
                if not failed_members or not all(
                    self._flash_member_has_retryable_failure(member)
                    for member in failed_members
                ):
                    continue
                missing = target_count - successful_count
                if missing > 0:
                    repair_needs[compound_leaf] = (
                        missing,
                        [
                            canonical_config_id(member.config)
                            for member in failed_members
                        ],
                    )
            if not repair_needs:
                break
            compound_backfill_pass_count += 1
            if not reserve_pass():
                break
            backfill_round_members: dict[int, PopulationMember] = {}
            for compound_leaf, (missing, failed_ids) in repair_needs.items():
                state = compound_states[compound_leaf]
                state_metric = state["metric"]
                state_source_selection = state["source_selection"]
                assert isinstance(state_metric, dict)
                assert isinstance(state_source_selection, dict)
                attempted_sources = state_source_selection["attempted_config_ids"]
                assert isinstance(attempted_sources, list)
                attempted_start = len(attempted_sources)
                generated_ids = add_compound_transfers(
                    compound_leaf,
                    count=missing,
                    round_members=backfill_round_members,
                )
                backfill_rounds = state_metric["backfill_rounds"]
                assert isinstance(backfill_rounds, list)
                backfill_rounds.append(
                    {
                        "repair_index": repair_index,
                        "required_successes": missing,
                        "failed_transfer_config_ids": failed_ids,
                        "attempted_source_config_ids": attempted_sources[
                            attempted_start:
                        ],
                        "generated_config_ids": generated_ids,
                    }
                )
            run_pass(
                backfill_round_members,
                desc=f"Compound packet failure backfills {repair_index + 1}:",
            )
            if budget_exhausted:
                break

        for metric in compound_transfer_metrics:
            metric_transfer_entries = metric["transfers"]
            assert isinstance(metric_transfer_entries, list)
            for transfer in metric_transfer_entries:
                assert isinstance(transfer, dict)
                transferred_config_id = transfer["transferred_config_id"]
                assert isinstance(transferred_config_id, str)
                transferred = compound_transfer_members[transferred_config_id]
                transfer.update(measurement_result(transferred))
            metric["transfer_count"] = len(metric_transfer_entries)
            successful_transfer_ids = [
                transfer["transferred_config_id"]
                for transfer in metric_transfer_entries
                if isinstance(transfer, dict)
                and isinstance(transfer.get("transferred_config_id"), str)
                and self._flash_member_succeeded(
                    compound_transfer_members[transfer["transferred_config_id"]]
                )
            ]
            metric["successful_transfer_config_ids"] = successful_transfer_ids
            target_count = metric["transfer_target_count"]
            assert isinstance(target_count, int)
            qualified_transfer_ids = successful_transfer_ids[:target_count]
            metric["qualified_transfer_config_ids"] = qualified_transfer_ids
            failure_statuses_allowed = metric.get("catalog_error") is None and all(
                isinstance(transfer, dict)
                and flash_terminal_measurement_is_valid(transfer)
                for transfer in metric_transfer_entries
            )
            metric["failure_statuses_allowed"] = failure_statuses_allowed
            compound_leaf = next(
                leaf
                for leaf in compound_leaves
                if leaf.pipeline_family == metric["family"]
                and leaf.compound_exp2_packet == metric["compound_packet"]
                and leaf.softmax_disc == metric["softmax_disc"]
            )
            qualified_compound_config_ids[compound_leaf] = set(qualified_transfer_ids)
        self._flash_qualified_compound_config_ids = qualified_compound_config_ids

        parent_score_config_ids = {
            canonical_config_id(member.config)
            for member in qualified_population.values()
            if (leaf := self._flash_structural_leaf(member)) is not None
            and leaf.compound_exp2_packet is None
        }
        live_family_count = len({leaf.pipeline_family for leaf in ordinary_leaves})
        family_probe_required = bool(
            policy.family_probe_generations > 0
            and policy.retained_families is not None
            and live_family_count > policy.retained_families
            and not exact_space_exhausted
        )
        family_probe_paths = (
            self._flash_family_probe_paths([*qualified_population.values()])
            if family_probe_required
            else []
        )
        family_probe_path_limit = getattr(
            self, "_flash_family_probe_path_limit", len(family_probe_paths)
        )
        if family_probe_required and len(family_probe_paths) != family_probe_path_limit:
            raise AssertionError(
                "family probe path count does not match the live structural catalog"
            )
        family_probe_metrics: list[dict[str, object]] = []
        family_probe_generators = []
        for copy_index, (member, required_leaf, constraints, unrestricted) in enumerate(
            family_probe_paths
        ):
            start_leaf = self._flash_structural_leaf(member)
            assert start_leaf is not None
            family_probe_metrics.append(
                {
                    "family": start_leaf.pipeline_family,
                    "compound_packet": start_leaf.compound_exp2_packet,
                    "softmax_disc": start_leaf.softmax_disc,
                    "starting_config_id": canonical_config_id(member.config),
                    "unrestricted": unrestricted,
                    "rounds": [],
                }
            )
            family_probe_generators.append(
                self._pruned_pattern_search_from(
                    copy_index,
                    member,
                    visited,
                    constraints,
                    selected_limit=policy.family_probe_candidates_per_path,
                    required_leaf=required_leaf,
                    conditional_surface=required_leaf is not None,
                    disable_early_stopping=True,
                )
            )

        family_probe_generations_started = 0
        family_probe_generations_completed = 0
        for probe_generation in range(policy.family_probe_generations):
            if not family_probe_required or not reserve_pass():
                break
            family_probe_generations_started += 1
            probe_round_members: dict[int, PopulationMember] = {}
            round_members_by_path: list[list[PopulationMember]] = []
            for path_metric, generator, path in zip(
                family_probe_metrics,
                family_probe_generators,
                family_probe_paths,
                strict=True,
            ):
                added = next(generator, ())
                new_members = list(added[1:]) if added else []
                generated_ids = add_members(new_members, probe_round_members)
                required_leaf = path[1]
                if (
                    required_leaf is not None
                    and required_leaf.compound_exp2_packet is None
                ):
                    parent_score_config_ids.update(generated_ids)
                for candidate in new_members:
                    candidate_leaf = self._flash_structural_leaf(candidate)
                    if candidate_leaf is not None:
                        leaves_with_candidates.add(candidate_leaf)
                cast_rounds = path_metric["rounds"]
                assert isinstance(cast_rounds, list)
                cast_rounds.append(
                    {
                        "probe_generation": probe_generation + 1,
                        "measurement_pass_index": rounds_completed + 1,
                        "candidate_ids": generated_ids,
                        "results": [],
                    }
                )
                round_members_by_path.append(new_members)
            run_pass(
                probe_round_members,
                desc=f"Structural family probe {probe_generation + 1}:",
            )
            for path_metric, members in zip(
                family_probe_metrics, round_members_by_path, strict=True
            ):
                cast_rounds = path_metric["rounds"]
                assert isinstance(cast_rounds, list)
                round_metric = cast_rounds[-1]
                assert isinstance(round_metric, dict)
                round_metric["results"] = [
                    {
                        "config_id": canonical_config_id(member.config),
                        **measurement_result(member),
                    }
                    for member in members
                ]
                for member in members:
                    leaf = self._flash_structural_leaf(member)
                    if (
                        leaf is not None
                        and leaf.compound_exp2_packet is not None
                        and self._flash_member_succeeded(member)
                    ):
                        qualified_compound_config_ids.setdefault(leaf, set()).add(
                            canonical_config_id(member.config)
                        )
            family_probe_generations_completed += 1
        family_probe_complete = bool(
            not family_probe_required
            or (
                family_probe_generations_started == policy.family_probe_generations
                and family_probe_generations_completed
                == policy.family_probe_generations
            )
        )
        self._flash_parent_score_config_ids = parent_score_config_ids

        successful_config_ids = {
            canonical_config_id(member.config)
            for member in qualified_population.values()
            if self._flash_member_succeeded(member)
        }
        for leaf in ordinary_leaves:
            members = sorted(
                (
                    member
                    for member in qualified_population.values()
                    if self._flash_structural_leaf(member) == leaf
                ),
                key=self._flash_member_rank_key,
            )
            successful = [
                member for member in members if self._flash_member_succeeded(member)
            ]
            retained = self._flash_lane_diverse_members(
                successful,
                leaf_pipeline_lanes[leaf],
                policy.retained_candidates_per_leaf,
            )
            leaf_metrics[leaf]["qualified_results"] = [
                {
                    "config_id": canonical_config_id(member.config),
                    **measurement_result(member),
                    "pipeline_lanes": [
                        self._flash_pipeline_lane_metric(lane)
                        for lane in leaf_pipeline_lanes[leaf]
                        if self._flash_member_matches_pipeline_lane(member, lane)
                    ],
                }
                for member in members
            ]
            leaf_metrics[leaf]["retained_config_ids"] = [
                canonical_config_id(member.config) for member, _lane in retained
            ]
            cast_lanes = leaf_metrics[leaf]["pipeline_lanes"]
            assert isinstance(cast_lanes, list)
            for _lane, lane_metric in zip(
                leaf_pipeline_lanes[leaf], cast_lanes, strict=True
            ):
                cast_ids = lane_metric["conditional_candidate_ids"]
                successful_conditional_ids = lane_metric[
                    "successful_conditional_candidate_ids"
                ]
                repair_ids = lane_metric["repair_candidate_ids"]
                successful_repair_ids = lane_metric["successful_repair_candidate_ids"]
                assert isinstance(cast_ids, list)
                assert isinstance(successful_conditional_ids, list)
                assert isinstance(repair_ids, list)
                assert isinstance(successful_repair_ids, list)
                repair_parent_decisions = lane_metric["repair_parent_decisions"]
                assert isinstance(repair_parent_decisions, list)
                witness_config_id = lane_metric["witness_config_id"]
                witness_succeeded = witness_config_id in successful_config_ids
                successful_conditional_ids.extend(
                    config_id
                    for config_id in cast_ids
                    if config_id in successful_config_ids
                )
                successful_repair_ids.extend(
                    config_id
                    for config_id in repair_ids
                    if config_id in successful_config_ids
                )
                lane_metric["witness_succeeded"] = witness_succeeded
                has_success = bool(
                    witness_succeeded
                    or successful_conditional_ids
                    or successful_repair_ids
                )
                attempts = pipeline_lane_attempts(leaf, _lane)
                terminal_failure_exhausted = bool(
                    not has_success
                    and lane_metric["witness_attempted"]
                    and (
                        not lane_metric["conditional_required"]
                        or len(set(cast_ids))
                        >= policy.conditional_candidates_per_pipeline_lane
                    )
                    and len(repair_ids) == policy.qualification_failure_retries
                    and len(repair_parent_decisions)
                    == policy.qualification_failure_retries
                    and len(attempts) == 1 + len(cast_ids) + len(repair_ids)
                    and all(
                        self._flash_member_has_retryable_failure(member)
                        for member in attempts
                    )
                )
                lane_metric["terminal_failure_exhausted"] = terminal_failure_exhausted
                lane_metric["complete"] = bool(
                    lane_metric["witness_attempted"]
                    and (
                        not lane_metric["conditional_required"]
                        or len(set(cast_ids))
                        >= policy.conditional_candidates_per_pipeline_lane
                    )
                    and (has_success or terminal_failure_exhausted)
                )
            leaf_metrics[leaf]["complete"] = bool(successful) and all(
                bool(metric["complete"]) for metric in cast_lanes
            )
        lane_complete = all(
            bool(metric["complete"]) for metric in lane_metrics.values()
        )
        ordinary_complete = all(
            bool(leaf_metrics[leaf]["complete"]) for leaf in ordinary_leaves
        )
        for metric in clc_metrics.values():
            planned_values = metric["planned_values"]
            attempted_values = metric["attempted_values"]
            witness_config_ids = metric["witness_config_ids"]
            witness_repair_candidate_ids = metric["witness_repair_candidate_ids"]
            conditional_values = metric["conditional_values"]
            conditional_candidate_ids = metric["conditional_candidate_ids"]
            conditional_repair_candidate_ids = metric[
                "conditional_repair_candidate_ids"
            ]
            retained_values = metric["retained_values"]
            selected_values = metric["selected_values"]
            combination_candidate_ids = metric["combination_candidate_ids"]
            combination_depth_config_ids = metric["combination_depth_config_ids"]
            combination_divisor_values = metric["combination_divisor_values"]
            combination_cells = metric["combination_cells"]
            assert isinstance(planned_values, list)
            assert isinstance(attempted_values, list)
            assert isinstance(witness_config_ids, dict)
            assert isinstance(witness_repair_candidate_ids, dict)
            assert isinstance(conditional_values, list)
            assert isinstance(conditional_candidate_ids, dict)
            assert isinstance(conditional_repair_candidate_ids, dict)
            assert isinstance(retained_values, list)
            assert isinstance(selected_values, list)
            assert isinstance(combination_candidate_ids, list)
            assert isinstance(combination_depth_config_ids, list)
            assert isinstance(combination_divisor_values, list)
            assert isinstance(combination_cells, list)
            successful_depth_ids = [
                depth_config_id
                for depth_config_id in combination_depth_config_ids
                if any(
                    isinstance(cell, dict)
                    and cell.get("depth_config_id") == depth_config_id
                    and cell.get("config_id") in successful_config_ids
                    for cell in combination_cells
                )
            ]
            successful_divisor_values = [
                value
                for value in combination_divisor_values
                if any(
                    isinstance(cell, dict)
                    and cell.get("divisor_value") == value
                    and cell.get("config_id") in successful_config_ids
                    for cell in combination_cells
                )
            ]
            projection_complete = bool(
                len(combination_cells)
                == len(combination_depth_config_ids) * len(combination_divisor_values)
                and len(
                    {
                        (cell.get("depth_config_id"), cell.get("divisor_value"))
                        for cell in combination_cells
                        if isinstance(cell, dict)
                    }
                )
                == len(combination_cells)
            )
            row_coverage_complete = successful_depth_ids == combination_depth_config_ids
            column_coverage_complete = (
                successful_divisor_values == combination_divisor_values
            )
            failure_statuses_allowed = self._flash_clc_combination_statuses_allowed(
                [cell for cell in combination_cells if isinstance(cell, dict)]
            ) and all(isinstance(cell, dict) for cell in combination_cells)
            metric["combination_projection_complete"] = projection_complete
            metric["successful_combination_depth_config_ids"] = successful_depth_ids
            metric["successful_combination_divisor_values"] = successful_divisor_values
            metric["combination_row_coverage_complete"] = row_coverage_complete
            metric["combination_column_coverage_complete"] = column_coverage_complete
            metric["combination_failure_statuses_allowed"] = failure_statuses_allowed
            metric["complete"] = bool(
                attempted_values == planned_values
                and len(selected_values) == len(planned_values)
                and set(selected_values) == set(planned_values)
                and all(
                    any(
                        config_id in successful_config_ids
                        for config_id in (
                            witness_config_ids.get(str(value)),
                            *witness_repair_candidate_ids.get(str(value), ()),
                        )
                    )
                    for value in planned_values
                )
                and all(
                    any(
                        config_id in successful_config_ids
                        for config_id in (
                            list(conditional_candidate_ids.get(str(value), ()))
                            + list(conditional_repair_candidate_ids.get(str(value), ()))
                        )
                    )
                    for value in conditional_values
                )
                and len(retained_values) == len(planned_values)
                and set(retained_values) == set(planned_values)
                and (
                    not metric["combination_required"]
                    or (
                        bool(combination_candidate_ids)
                        and len(combination_candidate_ids)
                        <= len(combination_depth_config_ids)
                        * len(combination_divisor_values)
                        and bool(combination_depth_config_ids)
                        and bool(combination_divisor_values)
                        and projection_complete
                        and row_coverage_complete
                        and column_coverage_complete
                        and failure_statuses_allowed
                    )
                )
            )
        clc_complete = all(bool(metric["complete"]) for metric in clc_metrics.values())
        for metric in compound_transfer_metrics:
            metric_transfers = metric["transfers"]
            limit = metric["limit"]
            target_count = metric["transfer_target_count"]
            successful_transfer_ids = metric["successful_transfer_config_ids"]
            qualified_transfer_ids = metric["qualified_transfer_config_ids"]
            assert isinstance(metric_transfers, list)
            assert isinstance(limit, int)
            assert isinstance(target_count, int)
            assert isinstance(successful_transfer_ids, list)
            assert isinstance(qualified_transfer_ids, list)
            metric["complete"] = bool(
                metric.get("catalog_error") is None
                and len(metric_transfers) == metric["transfer_count"]
                and 0 < target_count <= limit
                and len(metric_transfers)
                <= limit * (1 + policy.qualification_failure_retries)
                and len(successful_transfer_ids) >= target_count
                and len(set(successful_transfer_ids)) == len(successful_transfer_ids)
                and set(successful_transfer_ids) <= successful_config_ids
                and qualified_transfer_ids == successful_transfer_ids[:target_count]
                and len(qualified_transfer_ids) == target_count
                and len(set(qualified_transfer_ids)) == target_count
                and metric["failure_statuses_allowed"] is True
            )
        compound_catalog_complete = bool(
            not compound_catalog_errors
            and len(compound_transfer_metrics) == len(compound_leaves)
        )
        compound_complete = bool(
            compound_catalog_complete
            and all(bool(metric["complete"]) for metric in compound_transfer_metrics)
        )
        qualification_passes_planned = (
            int(schedule_anchor_pass_planned)
            + pipeline_pass_count
            + pipeline_repair_pass_count
            + clc_witness_pass_count
            + clc_witness_repair_pass_count
            + int(any(clc_conditional_values.values()))
            + clc_conditional_repair_pass_count
            + int(bool(clc_combination_leaves))
            + int(bool(compound_leaves))
            + compound_backfill_pass_count
            + (policy.family_probe_generations if family_probe_required else 0)
        )
        manifest_configs: dict[str, Config] = {}

        def add_manifest_config(
            config: Config,
        ) -> None:
            config_id = canonical_config_id(config)
            existing_config = manifest_configs.get(config_id)
            if existing_config is not None and existing_config != config:
                raise AssertionError(f"canonical config ID collision for {config_id}")
            manifest_configs[config_id] = config

        for config in exact_space or ():
            add_manifest_config(config)
        for member in qualified_population.values():
            add_manifest_config(member.config)

        config_manifest: dict[str, dict[str, object]] = {}
        for config_id, config in manifest_configs.items():
            config_manifest[config_id] = {
                "config": copy.deepcopy(config.config),
            }
        initial_results = []
        for member in initial_population:
            leaf = self._flash_structural_leaf(member)
            assert leaf is not None
            lanes = leaf_pipeline_lanes.get(leaf, ())
            initial_results.append(
                {
                    "config_id": canonical_config_id(member.config),
                    "family": leaf.pipeline_family,
                    "compound_packet": leaf.compound_exp2_packet,
                    "softmax_disc": leaf.softmax_disc,
                    **initial_measurements[id(member)],
                    "pipeline_lanes": [
                        self._flash_pipeline_lane_metric(lane)
                        for lane in lanes
                        if self._flash_member_matches_pipeline_lane(member, lane)
                    ],
                }
            )
        retained_family_limit = (
            live_family_count
            if policy.retained_families is None
            else min(policy.retained_families, live_family_count)
        )
        self._autotune_metrics.search_phase_metrics = {
            "phase": "cute_flash_structural_qualification_v22",
            "cute_flash_lane_policy_version": _CUTE_FLASH_LANE_POLICY_VERSION,
            "completed": bool(
                rounds_completed == qualification_passes_planned
                and schedule_anchor_complete
                and lane_complete
                and ordinary_complete
                and clc_complete
                and compound_complete
                and family_probe_complete
            ),
            "initial_config_count": len(initial_config_ids),
            "initial_config_ids": initial_config_ids,
            "initial_results": initial_results,
            "schedule_anchor_design_source": (
                "live family x ordinary packet x softmax protocol from fragment defaults"
            ),
            "schedule_anchor_pass_planned": schedule_anchor_pass_planned,
            "schedule_anchor_pass_started": schedule_anchor_pass_started,
            "schedule_anchor_count": len(schedule_anchor_members),
            "schedule_anchor_complete": schedule_anchor_complete,
            "schedule_anchor_results": schedule_anchor_results,
            "measurement_timeline": measurement_timeline,
            "config_manifest": config_manifest,
            "exact_space_enumerated": exact_space is not None,
            "exact_space_exhausted": exact_space_exhausted,
            "exact_space_raw_budget": exact_space_raw_budget,
            "exact_space_config_ids": [
                canonical_config_id(config) for config in exact_space or ()
            ],
            "leaf_count": len(leaf_catalog),
            "ordinary_leaf_count": len(ordinary_leaves),
            "compound_leaf_count": len(compound_leaves),
            "leaf_results": [leaf_metrics[leaf] for leaf in ordinary_leaves],
            "pipeline_qualification_keys": list(
                self._flash_pipeline_qualification_keys()
            ),
            "qualification_rounds": policy.qualification_rounds,
            "qualification_rounds_started": rounds_started,
            "qualification_rounds_completed": rounds_completed,
            "qualification_passes_planned": qualification_passes_planned,
            "qualification_passes_started": rounds_started,
            "qualification_passes_completed": rounds_completed,
            "budget_exhausted": budget_exhausted,
            "pipeline_candidate_limit_per_leaf_per_round": (
                policy.pipeline_candidates_per_leaf_per_round
            ),
            "conditional_candidates_per_pipeline_lane": (
                policy.conditional_candidates_per_pipeline_lane
            ),
            "qualification_failure_retries": policy.qualification_failure_retries,
            "family_probe_generations": policy.family_probe_generations,
            "family_probe_generations_started": family_probe_generations_started,
            "family_probe_generations_completed": family_probe_generations_completed,
            "family_probe_candidates_per_path": (
                policy.family_probe_candidates_per_path
            ),
            "family_probe_required": family_probe_required,
            "family_probe_complete": family_probe_complete,
            "family_probe_path_limit": family_probe_path_limit,
            "family_probe_paths": family_probe_metrics,
            "neighbor_generation_limit_per_leaf_per_round": (
                qualification_neighbor_limit
            ),
            "candidate_count": len(candidate_ids),
            "leaves_with_candidates": len(leaves_with_candidates),
            "retained_candidates_per_leaf": policy.retained_candidates_per_leaf,
            "retained_family_cap": policy.retained_families,
            "retained_family_limit": retained_family_limit,
            "retained_family_slowdown_limit": (policy.retained_family_slowdown_limit),
            "clc_families": [clc_metrics[leaf] for leaf in clc_metrics],
            "compound_catalog_complete": compound_catalog_complete,
            "compound_catalog_errors": compound_catalog_errors,
            "compound_transfers": compound_transfer_metrics,
            "starting_path_limit": getattr(
                self, "_flash_promoted_path_limit", self.copies
            ),
            "maximum_path_capacity": self.copies,
            "unrestricted_path_exhausts_generation_budget": (
                policy.exhaust_unrestricted_path
            ),
        }
        self.population = [*qualified_population.values()]
        self.population.sort(key=performance)
        return generation_rounds_started

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
        catalog = (
            self.config_gen.flash_structural_leaf_catalog()
            if hasattr(self.config_gen, "flash_structural_leaf_catalog")
            else list(by_leaf)
        )
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
            "lane_policy_version": _CUTE_FLASH_LANE_POLICY_VERSION,
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
        if config_gen is not None:
            return super()._generate_neighbors(
                base,
                fixed_flat_values=fixed_flat_values,
                config_gen=config_gen,
                num_neighbors=num_neighbors,
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
