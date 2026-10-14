from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Literal

AutotuneEffort = Literal["none", "quick", "full"]
InitialPopulation = Literal["from_random", "from_best_available"]

DEFAULT_LLM_MODEL = "gpt-5-2"
DEFAULT_LLM_CONFIGS_PER_ROUND = 15
DEFAULT_LLM_MAX_ROUNDS = 4
DEFAULT_LLM_INITIAL_RANDOM_CONFIGS = 10
DEFAULT_LLM_COMPILE_TIMEOUT_S: int | None = 15


@dataclass(frozen=True)
class PatternSearchConfig:
    initial_population: int
    copies: int
    max_generations: int
    initial_population_strategy: InitialPopulation = "from_random"
    best_available_pad_random: bool = True
    compile_timeout_lower_bound: float = 30.0
    compile_timeout_quantile: float = 0.9


@dataclass(frozen=True)
class FlashStructuralSearchConfig:
    qualification_rounds: int
    pipeline_candidates_per_leaf_per_round: int
    conditional_candidates_per_pipeline_lane: int
    qualification_failure_retries: int
    family_probe_generations: int
    family_probe_candidates_per_path: int
    retained_candidates_per_leaf: int
    # None continues every successful live parent family. Finite caps rank
    # families after the measured probe, then apply the slowdown filter and cap.
    retained_families: int | None
    retained_family_slowdown_limit: float
    exhaust_unrestricted_path: bool
    starting_paths: int = 14
    terminal_coordinate_rounds: int = 2
    terminal_coordinate_beam_width: int = 4


@dataclass(frozen=True)
class DifferentialEvolutionConfig:
    population_size: int
    max_generations: int
    initial_population_strategy: InitialPopulation = "from_random"
    best_available_pad_random: bool = True
    compile_timeout_lower_bound: float = 30.0
    compile_timeout_quantile: float = 0.9


@dataclass(frozen=True)
class RandomSearchConfig:
    count: int


@dataclass(frozen=True)
class LLMSearchConfig:
    model: str = DEFAULT_LLM_MODEL
    configs_per_round: int = DEFAULT_LLM_CONFIGS_PER_ROUND
    max_rounds: int = DEFAULT_LLM_MAX_ROUNDS
    initial_random_configs: int = DEFAULT_LLM_INITIAL_RANDOM_CONFIGS
    compile_timeout_s: int | None = DEFAULT_LLM_COMPILE_TIMEOUT_S


# Default values for each algorithm (single source of truth)
PATTERN_SEARCH_DEFAULTS = PatternSearchConfig(
    initial_population=100,
    copies=5,
    max_generations=20,
)

FLASH_STRUCTURAL_SEARCH_DEFAULTS = FlashStructuralSearchConfig(
    qualification_rounds=2,
    pipeline_candidates_per_leaf_per_round=4,
    conditional_candidates_per_pipeline_lane=1,
    qualification_failure_retries=1,
    family_probe_generations=1,
    family_probe_candidates_per_path=20,
    retained_candidates_per_leaf=2,
    retained_families=4,
    retained_family_slowdown_limit=2.0,
    exhaust_unrestricted_path=True,
    starting_paths=14,
)

DIFFERENTIAL_EVOLUTION_DEFAULTS = DifferentialEvolutionConfig(
    population_size=40,
    max_generations=40,
)

RANDOM_SEARCH_DEFAULTS = RandomSearchConfig(
    count=1000,
)

# Full guided-search defaults used by direct LLMGuidedSearch construction.
LLM_SEARCH_DEFAULTS = LLMSearchConfig()

# Quick LLM preset used by the quick effort profile and the hybrid seed stage.
QUICK_LLM_SEARCH_DEFAULTS = LLMSearchConfig(
    max_rounds=1,
)


@dataclass(frozen=True)
class FinalSelectionConfig:
    """Harness-grade shootout that picks the config a search returns.

    The search's own candidate timings are short windows taken on a GPU that
    idles through most of the search (compiles), so they sit at boost clocks
    and a single reading can be bimodal. The final selection first drives the
    GPU to its sustained clock state (``thermal_warmup_ms`` of back-to-back
    launches), screens the retained shortlist with one
    ``screen_warmup_ms``+``screen_rep_ms`` window each when it is longer than
    ``top_k + 1`` (``screen_rep_ms`` 0 never screens), then times the best
    ``top_k`` survivors plus the best pinned seed and the search's incumbent
    for ``rounds`` rounds of ``warmup_ms``+``rep_ms`` steady windows,
    alternating the order every round so clock drift hits every finalist
    alike, and ranks by the median across rounds. A finalist needs a reading
    in every round to compete; one that fails or misses a round is not timed
    again.
    Each finalist reads private copies of the kernel arguments when device
    memory allows, so an L2-resident input of one candidate cannot speed up
    its rivals.

    Under ``HELION_AUTOTUNE_BUDGET_SECONDS`` the search phase keeps
    :meth:`estimated_seconds` back for the phase (at most a quarter of the
    budget, shared with the allowance for the search's generation in
    flight, though the smallest plan stays reserved past that generation
    whenever a quarter of the budget covers it) and the phase is sized to
    whatever is left when it starts
    (:meth:`fit_to_seconds`): the screen goes first, then rounds (down to
    two), then survivors (down to two), then the thermal warmup; the phase
    is skipped when even that plan does not fit.
    """

    # Survivors of the screen: the harness-best config of every validated
    # B200 flash search sat within the search's top five by its own readings,
    # so eight keeps a margin at one steady window per round each.
    top_k: int = 8
    # Even, so the alternating order cancels linear clock drift: with four
    # rounds (forward, reverse, forward, reverse) the two middle samples of
    # every finalist sit at the same summed window position, so the median is
    # drift-free; three rounds leave a position-dependent bias of up to the
    # drift over half a round (a 0.1%/window drift mis-ranked a 0.6%-slower
    # finalist in simulation). Each extra pair of rounds costs about
    # ``2 * finalists * (warmup + rep + overhead)`` seconds.
    rounds: int = 4
    # The steady window. Every window follows an idle gap (the worker round
    # trip), and the B200 power limiter takes about 400 ms of back-to-back
    # launches to settle again: per-launch traces of a dense flash kernel on
    # an already warm GPU read 1.37 ms (boost) for the first 100 ms, 1.48 ms
    # for the next 300 ms and 1.52 ms from then on. A 100 ms warmup left the
    # 500 ms window straddling the 1.48 / 1.52 regimes, so the median flipped
    # between them by 2-3% (readings of one finalist spanned 1.54-1.58 ms
    # across rounds) while the finalists differed by 0.3%; 500 ms of warmup
    # starts the window in the settled regime (twelve windows read within
    # 0.35% of each other).
    warmup_ms: int = 500
    rep_ms: int = 500
    # Back-to-back launches drive the B200 power limiter from boost (~1965
    # MHz) to the sustained ~1230 MHz within a few seconds; five seconds is
    # the deployment harness's own warmup.
    thermal_warmup_ms: int = 5000
    # The screen ranks the shortlist for the rounds, so it takes the same
    # steady window. The search's verification window (50 + 200 ms) sat in
    # the boost and settling regimes: it read finalists up to 16% below
    # their round medians and ranked them with no correlation to the rounds
    # (Spearman -0.6 to +0.3 at the large shapes of a 33-search campaign,
    # where the search's own first readings still correlated 0.6-0.95).
    screen_warmup_ms: int = 500
    screen_rep_ms: int = 500

    # Worker round trip, wrapper load and the timer's estimate pass (measured
    # ~0.15 s per window for CuTe flash kernels on B200).
    WINDOW_OVERHEAD_S = 0.2
    # Minimum plan: two finalists and two rounds keep the alternating order.
    MIN_FINALISTS = 2
    MIN_ROUNDS = 2

    def screens(self, shortlist: int) -> bool:
        """Whether a shortlist of ``shortlist`` configs is screened first.

        A screen costs one window per shortlisted config and saves one round
        of windows per dropped finalist, so it only pays off when the
        shortlist is longer than the ``top_k + 1`` finalists the rounds keep
        at the least.
        """
        return self.screen_rep_ms > 0 and shortlist > self.top_k + 1

    def finalists(self, shortlist: int) -> int:
        """Most finalists the rounds can time: the ``top_k`` best ranked plus
        the best pinned seed plus the search's incumbent.

        The seed and the incumbent survive even when they rank outside the
        ``top_k`` (the incumbent does when a later phase re-timed it above
        the shortlist's search numbers), so a shortlist of ``top_k + 2`` can
        reach the rounds whole; the estimate counts that upper bound.
        """
        return min(shortlist, self.top_k + 2)

    def estimated_seconds(self, shortlist: int) -> float:
        """Wall-clock estimate for a shortlist of ``shortlist`` configs.

        The reserve the search phase keeps for the phase under a budget.
        """
        window_s = (self.warmup_ms + self.rep_ms) / 1000.0 + self.WINDOW_OVERHEAD_S
        screen_s = 0.0
        if self.screens(shortlist):
            screen_window_s = (self.screen_warmup_ms + self.screen_rep_ms) / 1000.0
            screen_s = shortlist * (screen_window_s + self.WINDOW_OVERHEAD_S)
        rounds_s = self.finalists(shortlist) * self.rounds * window_s
        # 1.25x covers the timer sizing its repetitions from an estimate pass
        # (a window runs up to a quarter longer than requested); the 5 s
        # covers one worker spawn with wrapper and argument loads.
        return 1.25 * (self.thermal_warmup_ms / 1000.0 + screen_s + rounds_s) + 5.0

    def fit_to_seconds(
        self, seconds: float, shortlist: int
    ) -> FinalSelectionConfig | None:
        """The most complete plan whose estimate fits in ``seconds``.

        Sheds cost in the order: the screen, rounds (two at a time, keeping
        the count even, down to two), survivors (down to two), then the
        thermal warmup. Returns None when even the minimum plan does not fit.
        """
        for policy in self.budget_ladder():
            if policy.estimated_seconds(shortlist) <= seconds:
                return policy
        return None

    def budget_ladder(self) -> list[FinalSelectionConfig]:
        """Plans from this policy down to the minimum, cheapest last."""
        ladder: list[FinalSelectionConfig] = [self]
        policy = replace(self, screen_rep_ms=0)
        ladder.append(policy)
        while policy.rounds - 2 >= self.MIN_ROUNDS:
            policy = replace(policy, rounds=policy.rounds - 2)
            ladder.append(policy)
        while policy.top_k > self.MIN_FINALISTS:
            policy = replace(policy, top_k=policy.top_k - 1)
            ladder.append(policy)
        if policy.thermal_warmup_ms > 0:
            ladder.append(replace(policy, thermal_warmup_ms=0))
        return ladder


FINAL_SELECTION_DEFAULTS = FinalSelectionConfig()


@dataclass(frozen=True)
class AutotuneEffortProfile:
    pattern_search: PatternSearchConfig | None
    lfbo_pattern_search: PatternSearchConfig | None
    differential_evolution: DifferentialEvolutionConfig | None
    random_search: RandomSearchConfig | None
    llm_search: LLMSearchConfig | None = None
    finishing_rounds: int = 0
    rebenchmark_threshold: float = 1.5
    flash_structural_search: FlashStructuralSearchConfig | None = None
    # None keeps the single-pass final verification; lower efforts stay cheap.
    final_selection: FinalSelectionConfig | None = None


_PROFILES: dict[AutotuneEffort, AutotuneEffortProfile] = {
    "none": AutotuneEffortProfile(
        pattern_search=None,
        lfbo_pattern_search=None,
        differential_evolution=None,
        random_search=None,
    ),
    "quick": AutotuneEffortProfile(
        # CuTe flash surfaces raise this population to one parent row per
        # structural leaf (``ConfigGeneration.flash_population_floor``), so a
        # surface wider than this budget still measures every pipeline
        # family without charging other kernels for it.
        pattern_search=PatternSearchConfig(
            initial_population=30,
            copies=2,
            max_generations=5,
            initial_population_strategy="from_best_available",
            best_available_pad_random=False,
        ),
        lfbo_pattern_search=PatternSearchConfig(
            initial_population=30,
            copies=2,
            max_generations=5,
            initial_population_strategy="from_best_available",
            best_available_pad_random=False,
        ),
        differential_evolution=DifferentialEvolutionConfig(
            population_size=20,
            max_generations=8,
            initial_population_strategy="from_best_available",
            best_available_pad_random=False,
        ),
        random_search=RandomSearchConfig(
            count=100,
        ),
        llm_search=QUICK_LLM_SEARCH_DEFAULTS,
        finishing_rounds=0,
        rebenchmark_threshold=0.9,  # <1.0 effectively disables rebenchmarking
    ),
    "full": AutotuneEffortProfile(
        pattern_search=PATTERN_SEARCH_DEFAULTS,
        lfbo_pattern_search=PATTERN_SEARCH_DEFAULTS,
        differential_evolution=DIFFERENTIAL_EVOLUTION_DEFAULTS,
        random_search=RANDOM_SEARCH_DEFAULTS,
        llm_search=LLM_SEARCH_DEFAULTS,
        flash_structural_search=FLASH_STRUCTURAL_SEARCH_DEFAULTS,
        final_selection=FINAL_SELECTION_DEFAULTS,
    ),
}


def get_effort_profile(effort: AutotuneEffort) -> AutotuneEffortProfile:
    return _PROFILES[effort]
