"""Search for autotune configs by iteratively querying an LLM.

High-level flow:
1. Initialize the prompt context from the kernel, config space, and default
   config so the first LLM call sees both the workload description and the
   available tuning knobs.
2. Round 0 launches the first LLM call immediately, then benchmarks the
   default config plus a few random seed configs while that request is in
   flight.
3. When the round-0 LLM response arrives, the search benchmarks its new unique
   configs and folds those results into the running set of top configs.
4. The top configs are then rebenchmarked before the next prompt is built, so each
   later LLM round sees the latest stabilized timings instead of only one-shot
   measurements.
5. Later rounds repeat a synchronous cycle: build prompt from the latest
   search state, query the LLM, benchmark new configs, then rebenchmark the
   strongest configs.
6. The final returned config comes from the best rebenchmarked config,
   not from an unrechecked one-shot LLM suggestion.

The implementation keeps config parsing, workload analysis, prompting,
transport, and search orchestration separate:
- `configs.py` parses and validates sparse configs from LLM responses.
- `workload.py` analyzes the kernel and hardware for prompt context.
- `feedback.py` summarizes benchmark results for prompts.
- `prompting.py` builds the actual prompt text.
- `transport.py` handles provider I/O.
- This file owns the round-by-round search state machine.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
import os
import time
from typing import TYPE_CHECKING

from .. import exc
from .base_search import BenchmarkResult
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import check_population_consistency
from .candidate_budget import AttemptCategory
from .candidate_budget import random_replacement_draw_cap
from .effort_profile import DEFAULT_LLM_CONFIGS_PER_ROUND
from .effort_profile import DEFAULT_LLM_INITIAL_RANDOM_CONFIGS
from .effort_profile import DEFAULT_LLM_MAX_ROUNDS
from .effort_profile import DEFAULT_LLM_MODEL
from .llm.configs import parse_response_configs
from .llm.feedback import analyze_top_configs
from .llm.feedback import failed_benchmark_results
from .llm.feedback import format_results_for_llm
from .llm.feedback import summarize_anchor_configs_for_llm
from .llm.feedback import summarize_failed_configs_for_llm
from .llm.feedback import summarize_search_state_for_llm
from .llm.prompting import build_initial_prompt
from .llm.prompting import build_initial_search_guidance
from .llm.prompting import build_refinement_prompt
from .llm.prompting import build_system_prompt
from .llm.transport import DEFAULT_REQUEST_TIMEOUT_S
from .llm.transport import ProviderMetadata as _ProviderMetadata
from .llm.transport import ProviderReplayRecord as _ProviderReplayRecord
from .llm.transport import TokenUsage as _TokenUsage
from .llm.transport import append_provider_replay_record
from .llm.transport import call_provider_with_usage as _call_provider_with_usage
from .llm.transport import infer_provider as _infer_provider
from .llm.transport import provider_replay_extra_headers_hash
from .llm.transport import provider_replay_request_id
from .llm.transport import provider_replay_response_hash
from .rag.seeding import build_seeded_population

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from .base_search import _AutotunableKernel
    from .effort_profile import AutotuneEffortProfile
    from .effort_profile import LLMSearchConfig

# Keep system + initial prompt plus this many recent round-trip exchanges
# to avoid exceeding LLM context limits on long sessions.
_MAX_CONTEXT_ROUNDS = 3
_MAX_STAGNANT_ROUNDS = 2


@dataclass
class _SearchLoopState:
    """Track dedupe and early-stop state across guided-search rounds."""

    seen_config_keys: set[str]
    prev_best_perf: float = math.inf
    rounds_without_improvement: int = 0


def guided_search_kwargs_from_config(
    config: LLMSearchConfig | None,
    settings: Settings,
) -> dict[str, object]:
    """Merge LLM config defaults with the supported HELION_LLM_* overrides."""
    del settings
    kwargs: dict[str, object] = {}

    if config is not None:
        kwargs.update(
            {
                "model": config.model,
                "configs_per_round": config.configs_per_round,
                "max_rounds": config.max_rounds,
                "initial_random_configs": config.initial_random_configs,
                "compile_timeout_s": config.compile_timeout_s,
            }
        )

    if (provider := os.environ.get("HELION_LLM_PROVIDER")) is not None:
        kwargs["provider"] = provider
    if (model := os.environ.get("HELION_LLM_MODEL")) is not None:
        kwargs["model"] = model
    if (effort_level := os.environ.get("HELION_LLM_EFFORT_LEVEL")) is not None:
        kwargs["effort_level"] = effort_level
    if os.environ.get("HELION_LLM_FAST_MODE") is not None:
        from ..runtime.settings import _env_get_bool

        kwargs["fast_mode"] = _env_get_bool("HELION_LLM_FAST_MODE", False)
    if (value := os.environ.get("HELION_LLM_COMPILE_TIMEOUT_S")) is not None:
        kwargs["compile_timeout_s"] = int(value)
    if (value := os.environ.get("HELION_LLM_REQUEST_TIMEOUT_S")) is not None:
        kwargs["request_timeout_s"] = float(value)
    return kwargs


def guided_search_kwargs_from_profile(
    profile: AutotuneEffortProfile,
    settings: Settings,
) -> dict[str, object]:
    """Merge effort-profile defaults with the supported HELION_LLM_* overrides."""
    return guided_search_kwargs_from_config(profile.llm_search, settings)


class LLMGuidedSearch(PopulationBasedSearch):
    """
    LLM-Guided autotuner that uses a language model to suggest kernel configurations.

    Instead of random or evolutionary search, this strategy uses an LLM to propose
    configurations based on:
    - The kernel's source code and structure
    - The configuration space (parameter types, ranges)
    - GPU hardware information
    - Benchmark results from previous rounds (iterative refinement)

    The search overlaps only the initial round-0 request with seed
    benchmarking. After that, refinement rounds are synchronous: each round
    asks the LLM for a batch of configs, benchmarks them, rebenchmarks the
    strongest configs, and only then builds the next prompt.

    Common providers (OpenAI Responses, Anthropic Messages, and compatible
    proxies) work via direct HTTP without extra dependencies.

    Args:
        kernel: The kernel to be autotuned.
        args: Arguments passed to the kernel during benchmarking.
        provider: Optional explicit provider override. Use this when a proxy
            serves a model family behind a different API shape than its name
            implies.
        model: LLM model name (e.g. "gpt-5-2", "claude-haiku-4.5",
            "claude-3-5-haiku-latest"). Can also be set via HELION_LLM_MODEL.
        configs_per_round: Number of configs to request from the LLM per round.
        max_rounds: Total number of LLM query rounds, including the initial
            suggestion round. ``max_rounds=1`` means one LLM call total.
        initial_random_configs: Number of random configs to add alongside LLM
            suggestions in the first round, for diversity.
        finishing_rounds: Number of finishing rounds to simplify the best config.
        api_base: Optional custom API base URL for the LLM provider.
        api_key: Optional API key. Defaults to the provider's env var (e.g. OPENAI_API_KEY).
        compile_timeout_s: Optional compile-time cap applied only while the LLM
            search benchmarks its exploratory configs.
        effort_level: Optional provider-specific effort-level hint (none / low /
            medium / high / max). Can also be set via HELION_LLM_EFFORT_LEVEL.
        fast_mode: Opt into Anthropic Opus 4.6/4.7 fast mode (faster output
            tokens, no extended thinking). Ignored by non-Anthropic providers.
            Can also be set via HELION_LLM_FAST_MODE.
    """

    _provider_requests = 0

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        provider: str | None = None,
        model: str = DEFAULT_LLM_MODEL,
        configs_per_round: int = DEFAULT_LLM_CONFIGS_PER_ROUND,
        max_rounds: int = DEFAULT_LLM_MAX_ROUNDS,
        initial_random_configs: int = DEFAULT_LLM_INITIAL_RANDOM_CONFIGS,
        finishing_rounds: int = 0,
        min_improvement_delta: float = 0.005,
        api_base: str | None = None,
        api_key: str | None = None,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        compile_timeout_s: int | None = None,
        effort_level: str | None = None,
        fast_mode: bool = False,
        retrieved_examples: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__(kernel, args, finishing_rounds=finishing_rounds)
        if max_rounds < 1:
            raise ValueError("LLMGuidedSearch max_rounds must be >= 1")
        self.provider = (
            _infer_provider(model, provider) if provider is not None else None
        )
        self.model = model
        self.configs_per_round = configs_per_round
        self.max_rounds = max_rounds
        self.initial_random_configs = initial_random_configs
        self.min_improvement_delta = min_improvement_delta
        self.api_base = api_base
        self.api_key = api_key
        self.request_timeout_s = request_timeout_s
        self.compile_timeout_s = compile_timeout_s
        self.effort_level = effort_level
        self.fast_mode = fast_mode
        self._retrieved_examples = [
            dict(example) for example in retrieved_examples or ()
        ]

        self._messages: list[dict[str, str]] = []
        self._all_benchmark_results: list[BenchmarkResult] = []
        self._latest_results_by_config_key: dict[str, BenchmarkResult] = {}
        self._llm_call_times: list[float] = []
        self._token_usage: list[_TokenUsage] = []
        self._provider_metadata: list[_ProviderMetadata] = []
        self._provider_replay_identities: list[tuple[str, str | None]] = []
        self._provider_requests = 0
        self._benchmark_times: list[float] = []
        self._llm_executor: concurrent.futures.ThreadPoolExecutor | None = None

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        """Merge shared search kwargs with LLM-specific profile settings."""
        return {
            **super().get_kwargs_from_profile(profile, settings),
            **guided_search_kwargs_from_profile(profile, settings),
        }

    # ── Prompt building ─────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Return the fixed instruction block shared by every LLM request."""
        return build_system_prompt()

    def _build_initial_search_guidance(self) -> str:
        """Describe the round-0 search strategy for this config space."""
        return build_initial_search_guidance(
            configs_per_round=self.configs_per_round,
            compile_timeout_s=self.settings.autotune_compile_timeout,
            flat_fields=self.config_spec._flat_fields(),
        )

    def _build_initial_prompt(self) -> str:
        """Describe the kernel and ask the LLM for the first batch of configs."""
        return build_initial_prompt(
            kernel=self.kernel,
            args=self.args,
            config_spec=self.config_spec,
            configs_per_round=self.configs_per_round,
            compile_timeout_s=self.settings.autotune_compile_timeout,
            retrieved_examples=self._retrieved_examples,
        )

    def set_retrieved_examples(self, examples: Sequence[Mapping[str, object]]) -> None:
        """Replace the structured Tier-1 examples included in the first prompt."""
        self._retrieved_examples = [dict(example) for example in examples]

    def _prepare_best_available_initial_evidence(self) -> None:
        """Add live best-available configs to round-0 seeds and prompt context."""
        if (
            not self.settings.autotune_best_available_read
            or self.fixed_initial_population_flat() is not None
        ):
            return

        cached = self._find_similar_cached_configs(
            self.settings.autotune_best_available_max_configs
        )
        best_available: list[Config] = []
        seen: set[Config] = set()
        for entry in cached:
            try:
                config = self.config_gen.unflatten(entry.to_mutable_flat_config())
            except (exc.InvalidConfig, ValueError, TypeError, KeyError, AssertionError):
                continue
            if config not in seen:
                seen.add(config)
                best_available.append(config)
        if not best_available:
            return

        target_size = self.initial_random_configs + 1
        seeds = [*best_available, self.config_spec.default_config()]

        def draw_random() -> Config:
            return self.config_gen.unflatten(self.config_gen.random_flat())

        def is_valid(config: Config) -> bool:
            try:
                self.config_gen.unflatten(self.config_gen.flatten(config))
            except (exc.InvalidConfig, ValueError, TypeError, KeyError, AssertionError):
                return False
            return True

        population, _ = build_seeded_population(
            seeds,
            n=target_size,
            num_neighbors_cap=target_size,
            draw_random=draw_random,
            is_valid=is_valid,
            flatten_key=lambda config: config,
            budget=self.candidate_attempt_budget,
            draw_cap=random_replacement_draw_cap(target_size),
        )
        self.set_fixed_initial_population_configs(
            population, attempts_already_recorded=True
        )
        best_available_set = set(best_available)
        for config in population:
            self.set_candidate_source(
                config,
                "best_available" if config in best_available_set else "initial_seed",
            )
        self._retrieved_examples.extend(
            {
                "kernel_name": "best_available",
                "config": dict(config.config),
            }
            for config in best_available
        )

    def _build_refinement_prompt(self, round_num: int) -> str:
        """Summarize search progress so the LLM can propose the next batch."""
        del round_num
        return build_refinement_prompt(
            configs_per_round=self.configs_per_round,
            compile_timeout_s=self.settings.autotune_compile_timeout,
            failed_count=len(failed_benchmark_results(self._all_benchmark_results)),
            total_count=len(self._all_benchmark_results),
            search_state=summarize_search_state_for_llm(
                self._all_benchmark_results,
                self._default_config_dict,
            ),
            anchor_configs=summarize_anchor_configs_for_llm(
                self._all_benchmark_results,
                self._default_config_dict,
            ),
            results=format_results_for_llm(
                self._all_benchmark_results,
                self._default_config_dict,
            ),
            top_patterns=analyze_top_configs(
                self._all_benchmark_results,
                self._default_config_dict,
            ),
            failed_patterns=summarize_failed_configs_for_llm(
                self._all_benchmark_results,
                self._default_config_dict,
            ),
        )

    # ── LLM transport ────────────────────────────────────────────

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Send one synchronous request to the configured provider and time it."""
        t0 = time.perf_counter()
        provider: str | None = None
        max_output_tokens = self._max_output_tokens_for_request()
        result = None
        request_hash: str | None = None
        try:
            provider = self.provider or _infer_provider(self.model)
            if provider == "unsupported":
                raise RuntimeError(
                    f"Unsupported LLM provider for model={self.model!r}. "
                    "Supported providers are Anthropic Messages and OpenAI "
                    "Responses. Set HELION_LLM_PROVIDER to override the provider "
                    "when using a proxy."
                )
            effective_api_base = self.api_base or os.environ.get("HELION_LLM_API_BASE")
            request_hash = provider_replay_request_id(
                provider,
                model=self.model,
                api_base=effective_api_base,
                messages=messages,
                max_output_tokens=max_output_tokens,
                request_timeout_s=self.request_timeout_s,
                effort_level=self.effort_level,
                fast_mode=self.fast_mode,
                extra_headers_sha256=provider_replay_extra_headers_hash(),
            )
            self._provider_requests += 1
            result = _call_provider_with_usage(
                provider,
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                messages=messages,
                max_output_tokens=max_output_tokens,
                request_timeout_s=self.request_timeout_s,
                effort_level=self.effort_level,
                fast_mode=self.fast_mode,
            )
            self._token_usage.append(result.usage)
            self._provider_metadata.append(
                _ProviderMetadata(
                    request_id=result.request_id,
                    response_id=result.response_id,
                    cache_state=result.cache_state,
                )
            )
            append_provider_replay_record(
                _ProviderReplayRecord(
                    provider=provider,
                    model=self.model,
                    api_base=effective_api_base,
                    messages=tuple(dict(message) for message in messages),
                    max_output_tokens=max_output_tokens,
                    request_timeout_s=self.request_timeout_s,
                    effort_level=self.effort_level,
                    fast_mode=self.fast_mode,
                    request_id=result.request_id or request_hash,
                    response_id=result.response_id,
                    cache_state=result.cache_state,
                    response_text=result.text,
                    usage=result.usage,
                    error_type=None,
                )
            )
            self._provider_replay_identities.append(
                (request_hash, provider_replay_response_hash(result.text))
            )
            return result.text
        except Exception as e:
            if provider is not None and request_hash is not None and result is None:
                append_provider_replay_record(
                    _ProviderReplayRecord(
                        provider=provider,
                        model=self.model,
                        api_base=self.api_base or os.environ.get("HELION_LLM_API_BASE"),
                        messages=tuple(dict(message) for message in messages),
                        max_output_tokens=max_output_tokens,
                        request_timeout_s=self.request_timeout_s,
                        effort_level=self.effort_level,
                        fast_mode=self.fast_mode,
                        request_id=request_hash,
                        response_id=None,
                        cache_state=None,
                        response_text=None,
                        usage=_TokenUsage(),
                        error_type=type(e).__name__,
                    )
                )
                self._provider_replay_identities.append((request_hash, None))
            self.log.warning(f"LLM call failed: {type(e).__name__}: {e}")
            raise
        finally:
            elapsed = time.perf_counter() - t0
            self._llm_call_times.append(elapsed)
            collector = self._attempt_instrumentation
            if collector is not None:
                from .rag.types import Phase

                collector.record_phase_transition(Phase.PROVIDER, phase_seconds=elapsed)

    def _call_llm_async(
        self, messages: list[dict[str, str]]
    ) -> concurrent.futures.Future[str]:
        """Launch the round-0 LLM request so seed benchmarking can overlap it."""
        # Round 0 is the only safe overlap point because the first prompt does not
        # depend on benchmark feedback from earlier rounds.
        if self._llm_executor is None:
            self._llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return self._llm_executor.submit(self._call_llm, messages)

    def _max_output_tokens_for_request(self) -> int:
        """Response budget sized to fit verbose JSON from models like Opus 4.7."""
        return max(512, min(4096, 256 + self.configs_per_round * 128))

    def _get_context_messages(self) -> list[dict[str, str]]:
        """Keep the fixed prompt prefix plus only the most recent round history."""
        prefix = self._messages[:2]
        suffix = self._messages[2:]
        max_suffix = _MAX_CONTEXT_ROUNDS * 2
        if len(suffix) > max_suffix:
            suffix = suffix[-max_suffix:]
        return prefix + suffix

    def _parse_configs(self, response: str) -> list[Config]:
        """Parse and validate candidate configs from a raw LLM response."""
        parsing_started = time.perf_counter()
        budget = self.candidate_attempt_budget

        def record_invalid() -> bool:
            return budget.record(AttemptCategory.INVALID)

        def admit_config(config: Config) -> bool:
            return self._admit_config_candidate(config, AttemptCategory.LLM_PROPOSED)

        def budget_exhausted() -> bool:
            return budget.exhausted

        try:
            return parse_response_configs(
                response,
                config_spec=self.config_spec,
                default_config_dict=self._default_config_dict,
                log=self.log,
                admit_config=admit_config if budget.limit is not None else None,
                record_invalid=record_invalid if budget.limit is not None else None,
                budget_exhausted=(
                    budget_exhausted if budget.limit is not None else None
                ),
            )
        finally:
            self._record_candidate_generation_time(
                time.perf_counter() - parsing_started
            )

    # ── Search loop ──────────────────────────────────────────────

    @contextlib.contextmanager
    def _llm_search_settings_context(self) -> Iterator[None]:
        """LLM proposals timed out more often per config, so fail them fast."""
        if self.compile_timeout_s is None:
            yield
            return

        original_compile_timeout = self.settings.autotune_compile_timeout
        self.settings.autotune_compile_timeout = min(
            original_compile_timeout,
            self.compile_timeout_s,
        )
        self.log(
            f"LLM compile timeout capped at {self.settings.autotune_compile_timeout}s"
        )
        try:
            yield
        finally:
            self.settings.autotune_compile_timeout = original_compile_timeout

    def _config_key(self, cfg: Config) -> str:
        """Return the stable key used to dedupe configs across rounds."""
        # Use the normalized repr so identical configs collapse across round boundaries.
        return repr(cfg)

    def _initialize_prompt_state(self) -> None:
        """Reset prompt state for a fresh guided-search run."""
        # Start each run from the fixed system prompt and the initial request.
        self._default_config_dict = dict(self.config_spec.default_config())
        self._messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_initial_prompt()},
        ]

    def _build_seed_configs(self) -> list[Config]:
        """Build the initial benchmark set: default plus a few random seeds."""
        if self.candidate_attempt_budget.limit is not None:
            fixed = self.fixed_initial_population_flat()
            if fixed is None:
                target_size = self.initial_random_configs + 1
                default_flat = self.config_gen.flatten(
                    self.config_spec.default_config()
                )
                generated_size = self._candidate_generation_limit(target_size)
                random_flats = self.config_gen.random_population_flat(generated_size)[
                    1:
                ]
                initial_flats = (
                    [default_flat, *random_flats] if generated_size > 0 else []
                )
            else:
                initial_flats = fixed
                target_size = len(fixed)

            try:
                population, _ = self.make_initial_population(
                    initial_flats, target_size=target_size
                )
            finally:
                self._finish_initial_candidate_attempts()
            return [member.config for member in population]

        # Start from default and add only distinct random configs that unflatten cleanly.
        seed_configs: list[Config] = [self.config_spec.default_config()]
        seen_config_keys = {self._config_key(seed_configs[0])}
        for flat in self.config_gen.random_population_flat(
            self.initial_random_configs + 1
        )[1:]:
            try:
                cfg = self.config_gen.unflatten(flat)
            except Exception:
                continue
            key = self._config_key(cfg)
            if key in seen_config_keys:
                continue
            seen_config_keys.add(key)
            seed_configs.append(cfg)
        return seed_configs

    def _dedupe_new_configs(
        self,
        configs: list[Config],
        seen_config_keys: set[str],
        *,
        attempts_already_recorded: bool = False,
    ) -> list[Config]:
        """Filter out configs that have already been seen in earlier rounds."""
        # Drop configs that were already benchmarked or queued in prior rounds.
        new_configs: list[Config] = []
        for cfg in configs:
            if (
                self.candidate_attempt_budget.exhausted
                and not attempts_already_recorded
            ):
                break
            if not attempts_already_recorded and not self._admit_config_candidate(
                cfg, AttemptCategory.LLM_PROPOSED
            ):
                continue
            key = self._config_key(cfg)
            if key in seen_config_keys:
                continue
            seen_config_keys.add(key)
            new_configs.append(cfg)
        return new_configs

    def _benchmark_and_ingest(
        self,
        configs: list[Config],
        *,
        generation: int,
        desc: str,
    ) -> None:
        """Benchmark a batch of configs and fold the results into search state."""
        # Benchmark one batch and feed the outcomes back into prompt and top-config state.
        self.set_generation(generation)
        bench_t0 = time.perf_counter()
        results = self.benchmark_batch(configs, desc=desc)
        self._benchmark_times.append(time.perf_counter() - bench_t0)
        self._ingest_results(results)

    def _ingest_results(self, results: list[BenchmarkResult]) -> None:
        """Store raw results and keep a bounded set of top configs for rebenchmarking."""
        # Retain full results for prompts while keeping only a small top-config set in memory.
        self._store_latest_results(results)
        members = [
            PopulationMember(
                fn=result.fn,
                perfs=[result.perf],
                flat_values=self.config_gen.flatten(result.config),
                config=result.config,
                status=result.status,
                compile_time=result.compile_time,
            )
            for result in results
        ]
        for member in members:
            self._record_benchmarked_member(member)
        self.population.extend(members)
        self._trim_population()

    def _trim_population(self) -> None:
        """Keep only the current top configs that future rebenchmarking needs."""
        # Bound population size because rebenchmarking cost scales with how many
        # top configs we keep.
        max_population = self.configs_per_round * 2
        if len(self.population) > max_population:
            self.population.sort(key=lambda member: member.perf)
            self.population = self.population[:max_population]

    def _store_latest_results(self, results: list[BenchmarkResult]) -> None:
        """Replace each config's prompt-facing result with its newest known timing."""
        # Keep one latest result per config so later prompts can see rebenchmark updates.
        for result in results:
            self._latest_results_by_config_key[self._config_key(result.config)] = result
        self._all_benchmark_results = list(self._latest_results_by_config_key.values())

    def _result_from_population_member(
        self, member: PopulationMember
    ) -> BenchmarkResult:
        """Convert one top config into a prompt-facing benchmark result."""
        # Reuse the latest top-config timing so prompts reflect post-rebenchmark winners.
        status = member.status
        if status == "unknown":
            status = "error"
        return BenchmarkResult(
            config=member.config,
            fn=member.fn,
            perf=member.perf,
            status=status,
            compile_time=member.compile_time,
        )

    def _refresh_prompt_results_from_population(self) -> None:
        """Push rebenchmarked top-config timings back into the prompt-facing history."""
        # Update only configs still in the top set; older off-top-set configs keep their
        # latest one-shot results.
        self._store_latest_results(
            [self._result_from_population_member(member) for member in self.population]
        )

    def _build_llm_messages(self, prompt: str | None = None) -> list[dict[str, str]]:
        """Build the message list for the next LLM request."""
        # Start from the rolling context window and optionally append a fresh prompt.
        messages = self._get_context_messages()
        if prompt is not None:
            messages = [*messages, {"role": "user", "content": prompt}]
        return messages

    def _wait_for_initial_llm_response(
        self,
        future: concurrent.futures.Future[str] | None,
    ) -> str | None:
        """Finish the overlapped round-0 LLM request after seed benchmarking."""
        # Wait only after the seed batch so round 0 can hide some initial LLM latency.
        if future is None:
            return None
        # LLM failures are intentionally fatal: silently falling back to plain
        # LFBO when the user opted into the LLM autotuner masks real config or
        # connectivity bugs (e.g. wrong API key, missing mTLS cert).
        return future.result(timeout=self.request_timeout_s)

    def _finalize_round(self, round_num: int) -> None:
        """Rebenchmark the current top configs and log the stabilized round summary."""
        # Rebenchmark before the next prompt so prompts and stop checks use stable winners.
        self.rebenchmark_population(desc=f"Round {round_num}: verifying top configs")
        self._refresh_prompt_results_from_population()
        check_population_consistency(
            self.population,
            process_group_name=self.kernel.env.process_group_name,
        )
        self.log(f"Round {round_num} complete:", self.statistics)

    def _update_early_stop_state(self, state: _SearchLoopState) -> bool:
        """Track weak-improvement rounds and decide whether to stop early."""
        if self.settings.autotune_disable_trajectory_early_stop:
            return False
        # Stop after repeated weak rounds so extra LLM calls do not just churn.
        current_best = self.best.perf
        if (
            math.isfinite(current_best)
            and math.isfinite(state.prev_best_perf)
            and state.prev_best_perf > 0
        ):
            relative_improvement = (
                state.prev_best_perf - current_best
            ) / state.prev_best_perf
            if relative_improvement < self.min_improvement_delta:
                state.rounds_without_improvement += 1
                if state.rounds_without_improvement >= _MAX_STAGNANT_ROUNDS:
                    self.log(
                        "Early stopping: no significant improvement "
                        f"for {state.rounds_without_improvement} rounds"
                    )
                    return True
            else:
                state.rounds_without_improvement = 0
        state.prev_best_perf = current_best
        return False

    def _run_initial_round(self, state: _SearchLoopState) -> None:
        """Run round 0 by overlapping the initial LLM request with seed benchmarking."""
        # Launch the first request before benchmarking because round 0 does not need
        # any prior search feedback to build its prompt.
        seeding_started = time.perf_counter()
        seed_configs = self._build_seed_configs()
        self._record_candidate_generation_time(time.perf_counter() - seeding_started)
        state.seen_config_keys.update(self._config_key(cfg) for cfg in seed_configs)

        self.log(
            f"Round 0: starting initial LLM call while benchmarking "
            f"{len(seed_configs)} seed configs (1 default + "
            f"{max(0, len(seed_configs) - 1)} random)"
        )

        # Do not generate proposals after the frozen candidate budget is spent.
        # Failure to dispatch an eligible request remains fatal (see
        # _wait_for_initial_llm_response for the rationale).
        if self.candidate_attempt_budget.exhausted:
            llm_future = None
        else:
            prompt_started = time.perf_counter()
            messages = self._build_llm_messages()
            self._record_candidate_generation_time(time.perf_counter() - prompt_started)
            llm_future = self._call_llm_async(messages)

        if seed_configs:
            self._benchmark_and_ingest(seed_configs, generation=0, desc="Round 0 seed")

        llm_response = self._wait_for_initial_llm_response(llm_future)

        llm_configs: list[Config] = []
        if llm_response is not None:
            self._messages.append({"role": "assistant", "content": llm_response})
            llm_configs = self._parse_configs(llm_response)

        round0_configs = self._dedupe_new_configs(
            llm_configs,
            state.seen_config_keys,
            attempts_already_recorded=(self.candidate_attempt_budget.limit is not None),
        )
        if round0_configs:
            self.log(
                f"Round 0: benchmarking {len(round0_configs)} new configs from the LLM"
            )
            self._benchmark_and_ingest(round0_configs, generation=0, desc="Round 0 LLM")
        else:
            self.log("Round 0: no new unique configs from the LLM")

        self._finalize_round(0)
        state.prev_best_perf = self.best.perf

    def _run_refinement_round(self, round_num: int, state: _SearchLoopState) -> bool:
        """Run one post-seed refinement round and report whether search should stop."""
        # Build the next prompt from the stabilized prior round, then benchmark new configs.
        prompt_started = time.perf_counter()
        prompt = self._build_refinement_prompt(round_num)
        messages = self._build_llm_messages(prompt)
        self._record_candidate_generation_time(time.perf_counter() - prompt_started)
        # LLM failures are intentionally fatal (see _wait_for_initial_llm_response).
        llm_response = self._call_llm(messages)

        self._messages.append({"role": "user", "content": prompt})
        self._messages.append({"role": "assistant", "content": llm_response})

        new_configs = self._dedupe_new_configs(
            self._parse_configs(llm_response),
            state.seen_config_keys,
            attempts_already_recorded=(self.candidate_attempt_budget.limit is not None),
        )
        if not new_configs:
            self.log(f"Round {round_num}: no new unique configs from LLM, stopping")
            return True

        self.log(f"Round {round_num}: benchmarking {len(new_configs)} new configs")
        self._benchmark_and_ingest(
            new_configs,
            generation=round_num,
            desc=f"Round {round_num}",
        )

        self._finalize_round(round_num)
        return self._update_early_stop_state(state)

    def _autotune(self) -> Config:
        """Run the guided search and emit per-run timing logs."""
        self.log(
            f"Starting LLMGuidedSearch with model={self.model}, "
            f"configs_per_round={self.configs_per_round}, "
            f"max_rounds={self.max_rounds}"
        )
        try:
            with self._llm_search_settings_context():
                return self._autotune_inner()
        finally:
            if (executor := getattr(self, "_llm_executor", None)) is not None:
                executor.shutdown(wait=False, cancel_futures=True)
                self._llm_executor = None
            self._log_search_stats()

    def _autotune_inner(self) -> Config:
        """Run round 0 once, then iterate the synchronized refinement rounds."""
        self._prepare_best_available_initial_evidence()
        prompt_started = time.perf_counter()
        self._initialize_prompt_state()
        self._record_candidate_generation_time(time.perf_counter() - prompt_started)
        state = _SearchLoopState(seen_config_keys=set())
        self._run_initial_round(state)

        for round_num in self._budgeted_range(1, self.max_rounds):
            if self._run_refinement_round(round_num, state):
                break

        best = self.final_rebenchmark_best(self.best)
        best = self.run_finishing_phase(best, self.finishing_rounds)
        return best.config

    def aggregate_token_usage(self) -> dict[str, int | None]:
        """Sum provider-reported token usage across every request this search (§6.3).

        Reports the total tokens to readiness plus the request count; a field is
        ``None`` when no request reported it (never a tokenizer estimate). The
        per-request breakdown lives in ``self._token_usage``.
        """

        def _sum(field: str) -> int | None:
            if len(self._token_usage) != self._provider_requests:
                return None
            values = [getattr(usage, field) for usage in self._token_usage]
            if not values or any(value is None for value in values):
                return None
            return sum(value for value in values if value is not None)

        return {
            "requests": self._provider_requests,
            "input_tokens": _sum("input_tokens"),
            "cached_input_tokens": _sum("cached_input_tokens"),
            "output_tokens": _sum("output_tokens"),
            "reasoning_tokens": _sum("reasoning_tokens"),
        }

    @property
    def provider_request_metadata(self) -> tuple[_ProviderMetadata, ...]:
        """Return the non-secret metadata retained for successful requests."""
        return tuple(self._provider_metadata)

    @property
    def provider_replay_identities(self) -> tuple[tuple[str, str | None], ...]:
        """Return ordered canonical request/response identities for this attempt."""
        return tuple(self._provider_replay_identities)

    def aggregate_provider_identity(self) -> dict[str, str | None]:
        """Project per-request identities into one deterministic attempt identity."""

        def aggregate(field: str) -> str | None:
            values = [getattr(item, field) for item in self._provider_metadata]
            if not values:
                return None
            if len(values) == 1:
                return values[0]
            canonical = json.dumps(
                values, ensure_ascii=True, separators=(",", ":")
            ).encode("utf-8")
            return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

        cache_states = [item.cache_state for item in self._provider_metadata]
        cache_state = (
            None
            if not cache_states
            else "unknown"
            if "unknown" in cache_states
            else "miss"
            if "miss" in cache_states
            else "hit"
        )
        return {
            "request_id": aggregate("request_id"),
            "response_id": aggregate("response_id"),
            "cache_state": cache_state,
        }

    def _log_search_stats(self) -> None:
        """Report how much time went to LLM calls and benchmarking."""
        if not self._llm_call_times:
            return
        avg_llm = sum(self._llm_call_times) / len(self._llm_call_times)
        avg_bench = (
            sum(self._benchmark_times) / len(self._benchmark_times)
            if self._benchmark_times
            else 0.0
        )
        tokens = self.aggregate_token_usage()
        self.log(
            f"LLM search stats: avg LLM call={avg_llm:.1f}s, "
            f"avg benchmark={avg_bench:.1f}s, "
            f"requests={tokens['requests']}, "
            f"total input/output tokens={tokens['input_tokens']}/"
            f"{tokens['output_tokens']}"
        )
