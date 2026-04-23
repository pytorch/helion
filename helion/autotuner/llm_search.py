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
import math
import os
import time
from typing import TYPE_CHECKING

from .base_search import BenchmarkResult
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import check_population_consistency
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
from .llm.transport import call_provider as _call_provider
from .llm.transport import infer_provider as _infer_provider

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from .base_search import _AutotunableKernel
    from .effort_profile import AutotuneEffortProfile
    from .effort_profile import LLMSearchConfig

# Keep system + initial prompt plus this many recent round-trip exchanges
# to avoid exceeding LLM context limits on long sessions.
_MAX_CONTEXT_ROUNDS = 3
_EMPTY_LLM_RESPONSE = '{"configs": []}'
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
    if (value := os.environ.get("HELION_LLM_COMPILE_TIMEOUT_S")) is not None:
        kwargs["compile_timeout_s"] = int(value)
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
    """

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

        self._messages: list[dict[str, str]] = []
        self._all_benchmark_results: list[BenchmarkResult] = []
        self._latest_results_by_config_key: dict[str, BenchmarkResult] = {}
        self._llm_call_times: list[float] = []
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
        try:
            provider = self.provider or _infer_provider(self.model)
            if provider == "unsupported":
                raise RuntimeError(
                    f"Unsupported LLM provider for model={self.model!r}. "
                    "Supported providers are Anthropic Messages and OpenAI "
                    "Responses. Set HELION_LLM_PROVIDER to override the provider "
                    "when using a proxy."
                )
            return _call_provider(
                provider,
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                messages=messages,
                max_output_tokens=self._max_output_tokens_for_request(),
                request_timeout_s=self.request_timeout_s,
            )
        except Exception as e:
            self.log.warning(f"LLM call failed: {type(e).__name__}: {e}")
            raise
        finally:
            self._llm_call_times.append(time.perf_counter() - t0)

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
        """Choose a small response budget that scales with configs_per_round."""
        return max(384, min(1536, 192 + self.configs_per_round * 56))

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
        return parse_response_configs(
            response,
            config_spec=self.config_spec,
            default_config_dict=self._default_config_dict,
            log=self.log,
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
        self, configs: list[Config], seen_config_keys: set[str]
    ) -> list[Config]:
        """Filter out configs that have already been seen in earlier rounds."""
        # Drop configs that were already benchmarked or queued in prior rounds.
        new_configs: list[Config] = []
        for cfg in configs:
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
        self.population.extend(
            PopulationMember(
                fn=result.fn,
                perfs=[result.perf],
                flat_values=self.config_gen.flatten(result.config),
                config=result.config,
                status=result.status,
                compile_time=result.compile_time,
            )
            for result in results
        )
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
        try:
            return future.result(timeout=self.request_timeout_s)
        except Exception:
            self.log.warning(
                "Round 0: initial LLM call failed, continuing with seed configs"
            )
            return None

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
        seed_configs = self._build_seed_configs()
        state.seen_config_keys.update(self._config_key(cfg) for cfg in seed_configs)

        self.log(
            f"Round 0: starting initial LLM call while benchmarking "
            f"{len(seed_configs)} seed configs (1 default + "
            f"{max(0, len(seed_configs) - 1)} random)"
        )

        llm_future: concurrent.futures.Future[str] | None = None
        try:
            llm_future = self._call_llm_async(self._build_llm_messages())
        except Exception:
            self.log.warning(
                "Round 0: could not start initial LLM call, continuing with seed configs"
            )

        if seed_configs:
            self._benchmark_and_ingest(seed_configs, generation=0, desc="Round 0 seed")

        llm_response = self._wait_for_initial_llm_response(llm_future)

        llm_configs: list[Config] = []
        if llm_response is not None:
            self._messages.append({"role": "assistant", "content": llm_response})
            llm_configs = self._parse_configs(llm_response)

        round0_configs = self._dedupe_new_configs(llm_configs, state.seen_config_keys)
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
        prompt = self._build_refinement_prompt(round_num)
        try:
            llm_response = self._call_llm(self._build_llm_messages(prompt))
        except Exception:
            self.log.warning(
                f"Round {round_num}: LLM call failed, generating no new configs instead"
            )
            llm_response = _EMPTY_LLM_RESPONSE

        self._messages.append({"role": "user", "content": prompt})
        self._messages.append({"role": "assistant", "content": llm_response})

        new_configs = self._dedupe_new_configs(
            self._parse_configs(llm_response),
            state.seen_config_keys,
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
        self._initialize_prompt_state()
        state = _SearchLoopState(seen_config_keys=set())
        self._run_initial_round(state)

        for round_num in range(1, self.max_rounds):
            if self._run_refinement_round(round_num, state):
                break

        best = self.run_finishing_phase(self.best, self.finishing_rounds)
        return best.config

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
        self.log(
            f"LLM search stats: avg LLM call={avg_llm:.1f}s, "
            f"avg benchmark={avg_bench:.1f}s"
        )
