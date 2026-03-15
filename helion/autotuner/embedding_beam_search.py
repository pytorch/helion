"""
EmbeddingBeamSearch: Beam-search autotuner guided by an external embedding model.

Overview
--------
Instead of a surrogate classifier trained locally, this algorithm asks an
external HTTP endpoint to embed each kernel configuration as a dense vector.
Cosine similarity between embeddings drives two complementary signals:

  diversity_score    = 1 - max_cosine_sim(candidate, all_observed)
                       High → candidate is a novel region of config space.

  exploitation_score = max_cosine_sim(candidate, beam_embeddings)
                       High → candidate is close to configs that already
                       performed well.

A user-configurable ExplorationSchedule blends the two signals:

  score = exploration_ratio * diversity + (1 - exploration_ratio) * exploitation

At the start exploration_ratio is high (broad search); it decays over
generations, eventually concentrating the search near the best observed
configs.  The ThresholdTriggerSchedule additionally drops to a minimum
exploitation_ratio when improvement stalls for several consecutive generations.

HTTP endpoint contract
----------------------
  POST <endpoint>
  Content-Type: application/json

  Single text  → body: {"input": "<text>"}
               → response: {"embedding": [float, ...]}

  Batch texts  → body: {"input": ["t1", "t2", ...]}
               → response: {"embeddings": [[float, ...], ...]}
"""

from __future__ import annotations

import abc
import json
import math
import random
import time
from typing import TYPE_CHECKING
import urllib.error
import urllib.request

import numpy as np

from .. import exc
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import performance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .base_search import _AutotunableKernel
    from .config_generation import FlatConfig


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Embedding Client
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingClient:
    """
    HTTP client that calls an external text-embedding endpoint.

    Supports three response formats transparently:
      - Simple single:   ``{"embedding": [...]}``
      - Simple batch:    ``{"embeddings": [[...], ...]}``
      - OpenAI/Together: ``{"data": [{"embedding": [...]}, ...]}``

    Args:
        endpoint:  Full URL to POST embedding requests to.
        model:     Model name passed as ``"model"`` in every request body.
                   Required by OpenAI-compatible APIs such as Together AI.
        api_token: Optional bearer token sent as ``Authorization: Bearer <token>``.
        timeout:   Per-request timeout in seconds.
        batch_size: Maximum number of texts per HTTP request.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        api_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_token = api_token
        self.timeout = timeout
        self.total_embedding_secs: float = 0.0
        self.total_embedding_calls: int = 0

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        data = json.dumps(payload).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            _t0 = time.perf_counter()
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
            self.total_embedding_secs += time.perf_counter() - _t0
            self.total_embedding_calls += 1
            return result  # type: ignore[no-any-return]
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                hint = (
                    "Set HELION_EMBEDDING_API_TOKEN to a valid bearer token."
                    if not self.api_token
                    else "Check that HELION_EMBEDDING_API_TOKEN is correct and has not expired."
                )
                raise exc.AutotuneError(
                    f"EmbeddingBeamSearch: embedding endpoint {self.endpoint!r} "
                    f"returned HTTP {e.code} {e.reason}. {hint}"
                ) from e
            raise exc.AutotuneError(
                f"EmbeddingBeamSearch: embedding endpoint {self.endpoint!r} "
                f"returned HTTP {e.code} {e.reason}."
            ) from e
        except urllib.error.URLError as e:
            raise exc.AutotuneError(
                f"EmbeddingBeamSearch: failed to reach embedding endpoint {self.endpoint!r}. "
                f"Error: {e}\n"
                "Check that the endpoint is reachable and the URL is correct."
            ) from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Return one embedding list per input text in a single HTTP request.
        """
        if not texts:
            return []

        payload: dict[str, object] = {
            "model": self.model,
            "input": texts if len(texts) > 1 else texts[0],
        }
        response = self._post(payload)
        return self._parse_embeddings(response, expected=len(texts))

    def _parse_embeddings(
        self, response: dict[str, object], expected: int
    ) -> list[list[float]]:
        """
        Extract a flat list of embedding vectors from an API response.

        Handles three response shapes:
          - OpenAI/Together: ``{"data": [{"embedding": [...]}, ...]}``
          - Simple batch:    ``{"embeddings": [[...], ...]}``
          - Simple single:   ``{"embedding": [...]}``
        """
        # OpenAI / Together AI format
        if "data" in response:
            data = response["data"]
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return [item["embedding"] for item in data]  # type: ignore[index]

        # Simple batch format
        if "embeddings" in response:
            embeddings = response["embeddings"]
            if isinstance(embeddings, list):
                return embeddings  # type: ignore[return-value]

        # Simple single format
        if "embedding" in response:
            return [response["embedding"]]  # type: ignore[list-item]

        raise exc.AutotuneError(
            f"EmbeddingBeamSearch: unexpected response format from {self.endpoint!r}: "
            f"{response!r}.\n"
            "Supported formats: "
            '{"data": [{"embedding": [...]}]} (OpenAI/Together), '
            '{"embeddings": [[...]]} (batch), '
            '{"embedding": [...]} (single).'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Cache
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingCache:
    """
    In-memory cache that maps FlatConfig → L2-normalised embedding array.

    Normalisation is applied once on insertion so that all subsequent cosine
    similarity calculations reduce to a simple dot product.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[object, ...], np.ndarray] = {}

    @staticmethod
    def _key(flat_config: FlatConfig) -> tuple[object, ...]:
        """Convert a FlatConfig (which may contain lists) into a hashable key."""
        return tuple(tuple(v) if isinstance(v, list) else v for v in flat_config)

    def get(self, flat_config: FlatConfig) -> np.ndarray | None:
        return self._store.get(self._key(flat_config))

    def put(self, flat_config: FlatConfig, raw_embedding: list[float]) -> None:
        """Store a raw embedding, L2-normalising it in place."""
        arr = np.array(raw_embedding, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm > 0.0:
            arr = arr / norm
        self._store[self._key(flat_config)] = arr

    def get_or_fetch(
        self,
        flat_configs: list[FlatConfig],
        client: EmbeddingClient,
        texts: list[str],
    ) -> list[np.ndarray]:
        """
        Return embeddings for every flat_config, fetching missing ones in a
        single batched HTTP call.

        Args:
            flat_configs: Configs whose embeddings are needed.
            client:       EmbeddingClient to call for cache misses.
            texts:        Parallel list of text representations (same order).

        Returns:
            One L2-normalised embedding array per flat_config.
        """
        results: list[np.ndarray | None] = [self.get(f) for f in flat_configs]
        missing_idx = [i for i, r in enumerate(results) if r is None]

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            fetched = client.embed_batch(missing_texts)
            if len(fetched) != len(missing_idx):
                raise exc.AutotuneError(
                    f"EmbeddingBeamSearch: embedding endpoint returned {len(fetched)} "
                    f"embeddings but {len(missing_idx)} were requested."
                )
            for i, raw_embedding in zip(missing_idx, fetched, strict=True):
                self.put(flat_configs[i], raw_embedding)
                results[i] = self.get(flat_configs[i])

        return results  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Index
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingIndex:
    """
    Accumulates (embedding, flat_config, perf) observations and provides
    cosine-similarity-based scoring for candidate configs.

    All stored embeddings are already L2-normalised (guaranteed by
    EmbeddingCache), so dot product == cosine similarity everywhere.
    """

    def __init__(self) -> None:
        self._embeddings: list[np.ndarray] = []
        self._flat_configs: list[FlatConfig] = []
        self._perfs: list[float] = []
        # Lazily rebuilt when new entries are added.
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._embeddings)

    def add(self, embedding: np.ndarray, flat_config: FlatConfig, perf: float) -> None:
        """Record an observed (embedding, config, perf) triple."""
        self._embeddings.append(embedding)
        self._flat_configs.append(flat_config)
        self._perfs.append(perf)
        self._matrix = None  # invalidate stacked matrix

    @property
    def _emb_matrix(self) -> np.ndarray | None:
        """(N, dim) matrix of all stored embeddings; rebuilt lazily."""
        if not self._embeddings:
            return None
        if self._matrix is None:
            self._matrix = np.stack(self._embeddings)
        return self._matrix

    def _cosine_sims(self, queries: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarities between queries and all stored
        embeddings.

        Args:
            queries: (Q, dim) matrix of L2-normalised query vectors.

        Returns:
            (Q, N) similarity matrix, or (Q, 0) when the index is empty.
        """
        mat = self._emb_matrix
        if mat is None:
            return np.zeros((len(queries), 0))
        return queries @ mat.T  # dot product of L2-normalised vecs = cosine sim

    def diversity_scores(self, queries: np.ndarray) -> np.ndarray:
        """
        Novelty of each query w.r.t. all previously observed configs.

        Returns ``1 − max_cosine_similarity`` ∈ [0, 1].
        A score of 1.0 means the candidate is completely unlike anything seen.
        Returns 1.0 for all queries when the index is empty.
        """
        sims = self._cosine_sims(queries)
        if sims.shape[1] == 0:
            return np.ones(len(queries))
        return 1.0 - sims.max(axis=1)

    def exploitation_scores(
        self, queries: np.ndarray, beam_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Proximity of each query to the current beam (best-performing configs).

        Returns ``max_cosine_similarity`` to any beam member ∈ [0, 1].
        A score of 1.0 means the candidate is essentially the same as a beam
        member — very attractive for exploitation.
        Returns 0.5 for all queries when the beam is empty (neutral baseline).

        Args:
            queries:         (Q, dim) candidate embedding matrix.
            beam_embeddings: (B, dim) beam member embedding matrix.
        """
        if len(beam_embeddings) == 0:
            return np.full(len(queries), 0.5)
        sims = queries @ beam_embeddings.T  # (Q, B)
        return sims.max(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Exploration / Exploitation Schedule
# ─────────────────────────────────────────────────────────────────────────────


class ExplorationSchedule(abc.ABC):
    """
    Controls the exploration ratio at each generation.

    The ratio ∈ [0, 1] determines how candidate scores are blended:
        score = ratio * diversity_score + (1 - ratio) * exploitation_score

    ratio = 1.0  →  pure exploration  (prefer novel / diverse configs)
    ratio = 0.0  →  pure exploitation (prefer configs near the current best)
    """

    @abc.abstractmethod
    def get_ratio(self, generation: int, stagnation_count: int) -> float:
        """Return the exploration ratio for this generation."""
        ...

    @abc.abstractmethod
    def description(self) -> str:
        """Return a human-readable description for logging."""
        ...


class LinearDecaySchedule(ExplorationSchedule):
    """
    Linearly decays the exploration ratio from ``initial_ratio`` to
    ``final_ratio`` over ``decay_generations`` generations, then stays at
    ``final_ratio``.  Stagnation count is ignored.

    Args:
        initial_ratio:     Starting exploration ratio (generation 0).
        final_ratio:       Minimum exploration ratio (reached at decay_generations).
        decay_generations: Number of generations over which to decay.
    """

    def __init__(
        self,
        initial_ratio: float = 0.8,
        final_ratio: float = 0.1,
        decay_generations: int = 10,
    ) -> None:
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.decay_generations = decay_generations

    def get_ratio(self, generation: int, stagnation_count: int) -> float:
        t = min(generation, self.decay_generations) / self.decay_generations
        return self.initial_ratio + t * (self.final_ratio - self.initial_ratio)

    def description(self) -> str:
        return (
            f"LinearDecay(initial={self.initial_ratio}, final={self.final_ratio}, "
            f"decay_over={self.decay_generations} generations)"
        )


class ThresholdTriggerSchedule(ExplorationSchedule):
    """
    Two-phase exploration schedule:

    Phase 1 – Gradual decay: linear decay from ``initial_ratio`` toward
               ``final_ratio`` over ``decay_generations`` while the search is
               making progress.

    Phase 2 – Exploitation lock-in: once ``stagnation_count`` reaches
               ``stagnation_threshold``, the ratio immediately drops to
               ``exploitation_ratio`` and stays there for the remainder of
               the search.

    This models the intuition: explore broadly while finding improvements,
    then commit to intensive exploitation once the search stalls.

    Args:
        initial_ratio:        Starting exploration ratio.
        final_ratio:          Minimum ratio during gradual decay (Phase 1).
        exploitation_ratio:   Ratio after stagnation threshold (Phase 2).
        stagnation_threshold: stagnation_count value that triggers Phase 2.
        decay_generations:    Generations over which Phase 1 decays.
    """

    def __init__(
        self,
        initial_ratio: float = 0.8,
        final_ratio: float = 0.2,
        exploitation_ratio: float = 0.05,
        stagnation_threshold: int = 3,
        decay_generations: int = 10,
    ) -> None:
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.exploitation_ratio = exploitation_ratio
        self.stagnation_threshold = stagnation_threshold
        self.decay_generations = decay_generations

    def get_ratio(self, generation: int, stagnation_count: int) -> float:
        if stagnation_count >= self.stagnation_threshold:
            return self.exploitation_ratio
        t = min(generation, self.decay_generations) / self.decay_generations
        return self.initial_ratio + t * (self.final_ratio - self.initial_ratio)

    def description(self) -> str:
        return (
            f"ThresholdTrigger(initial={self.initial_ratio}, "
            f"final={self.final_ratio}, "
            f"exploitation={self.exploitation_ratio}, "
            f"stagnation_threshold={self.stagnation_threshold}, "
            f"decay_over={self.decay_generations} generations)"
        )


_SCHEDULE_REGISTRY: dict[str, type[ExplorationSchedule]] = {
    "linear_decay": LinearDecaySchedule,
    "threshold_trigger": ThresholdTriggerSchedule,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Algorithm
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingBeamSearch(PopulationBasedSearch):
    """
    Beam-search autotuner guided by an external text-embedding model.

    Algorithm
    ---------
    Phase 1 – Initial population
        Generate ``initial_population`` random configs, benchmark them all,
        embed them, populate the EmbeddingIndex, and seed the beam with the
        top ``beam_width`` successful configs.

    Phase 2 – Generational beam search (repeated ``max_generations`` times)
        1. Ask the ExplorationSchedule for the current exploration_ratio.
        2. Generate neighbor flat configs from every beam member using
           ``pattern_neighbors`` (single-parameter changes + block-size pairs),
           sampling up to ``num_neighbors // beam_width`` per member.
        3. Embed all new candidates (cache avoids re-fetching known configs).
        4. Score each candidate:
               score = exploration_ratio × diversity_score
                     + (1 − exploration_ratio) × exploitation_score
        5. Select the top ``frac_selected`` fraction by score and benchmark them.
        6. Add newly benchmarked configs to the EmbeddingIndex and update the
           beam (top ``beam_width`` finite-perf configs seen so far).
        7. Update stagnation_count if improvement < ``min_improvement_delta``.

    Args:
        kernel:              The kernel to autotune.
        args:                Example arguments used for benchmarking.
        embedding_endpoint:  URL that serves embedding requests (required).
        embedding_model:     Model name sent in every request body (required).
                             Example: ``"intfloat/multilingual-e5-large-instruct"``.
        embedding_api_token: Optional bearer token for authenticating with the
                             embedding endpoint. Sent as
                             ``Authorization: Bearer <token>``. Can also be
                             supplied via the ``HELION_EMBEDDING_API_TOKEN``
                             environment variable.
        beam_width:          Number of configs kept in the beam at all times.
        initial_population:  Size of the initial random population.
        max_generations:     Maximum number of search generations.
        num_neighbors:       Total neighbor configs generated per generation
                             (split evenly across beam members).
        frac_selected:       Fraction of generated neighbors to benchmark.
        min_improvement_delta: Relative improvement below which a generation
                             is counted as stagnant (0.001 = 0.1%).
        embedding_timeout:   HTTP timeout in seconds for each embedding call.
        embedding_batch_size: Max texts per HTTP request to the endpoint.
        schedule:            Either an ExplorationSchedule instance, or a
                             string key from {"linear_decay",
                             "threshold_trigger"} (default).
        initial_exploration: Starting exploration ratio (see schedule docs).
        final_exploration:   Minimum ratio during gradual decay.
        exploitation_exploration: Ratio after stagnation triggers (only used
                             by ThresholdTriggerSchedule).
        stagnation_patience: stagnation_count value that triggers Phase 2 of
                             ThresholdTriggerSchedule.
        exploration_decay_generations: Generations over which the ratio decays.
                             Defaults to ``max_generations`` when None.
        compile_timeout_lower_bound: Lower bound for adaptive compile timeout.
        compile_timeout_quantile: Quantile for adaptive compile timeout.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        embedding_endpoint: str,
        embedding_model: str,
        embedding_api_token: str | None = None,
        beam_width: int = 5,
        initial_population: int = 50,
        max_generations: int = 20,
        num_neighbors: int = 100,
        frac_selected: float = 0.3,
        min_improvement_delta: float = 0.001,
        embedding_timeout: float = 30.0,
        schedule: str | ExplorationSchedule = "threshold_trigger",
        initial_exploration: float = 0.8,
        final_exploration: float = 0.2,
        exploitation_exploration: float = 0.05,
        stagnation_patience: int = 3,
        exploration_decay_generations: int | None = None,
        compile_timeout_lower_bound: float = 30.0,
        compile_timeout_quantile: float = 0.9,
    ) -> None:
        super().__init__(kernel, args)
        self.beam_width = beam_width
        self.initial_population = initial_population
        self.max_generations = max_generations
        self.num_neighbors = num_neighbors
        self.frac_selected = frac_selected
        self.min_improvement_delta = min_improvement_delta
        self.compile_timeout_lower_bound = compile_timeout_lower_bound
        self.compile_timeout_quantile = compile_timeout_quantile

        self._client = EmbeddingClient(
            embedding_endpoint,
            model=embedding_model,
            api_token=embedding_api_token,
            timeout=embedding_timeout,
        )
        self._cache = EmbeddingCache()
        self._index = EmbeddingIndex()
        self._benchmark_secs: float = 0.0
        self._configs_benchmarked: int = 0
        self._kernel_name: str = getattr(
            getattr(self.kernel, "kernel", None), "name", "unknown"
        )
        self._kernel_code: str | None = None  # lazily populated on first embed
        self._schedule = self._build_schedule(
            schedule,
            initial_exploration=initial_exploration,
            final_exploration=final_exploration,
            exploitation_exploration=exploitation_exploration,
            stagnation_patience=stagnation_patience,
            decay_generations=exploration_decay_generations or max_generations,
        )

    # ── Timer hook ───────────────────────────────────────────────────────────

    def autotune(self, *, skip_cache: bool = False) -> object:
        """Override to capture wall time from before _prepare() (baseline compile)."""
        self._t_autotune_start = time.perf_counter()
        return super().autotune(skip_cache=skip_cache)

    # ── Timed benchmark helpers ───────────────────────────────────────────────

    def _timed_benchmark(
        self, members: list[PopulationMember], *, desc: str
    ) -> list[PopulationMember]:
        """Run parallel_benchmark_population and accumulate wall-clock time."""
        _t0 = time.perf_counter()
        result = self.parallel_benchmark_population(members, desc=desc)
        self._benchmark_secs += time.perf_counter() - _t0
        self._configs_benchmarked += len(members)
        return result

    def _timed_rebenchmark(self, members: list[PopulationMember], *, desc: str) -> None:
        """Run rebenchmark_population and accumulate wall-clock time."""
        _t0 = time.perf_counter()
        self.rebenchmark_population(members, desc=desc)
        self._benchmark_secs += time.perf_counter() - _t0

    # ── Schedule construction ─────────────────────────────────────────────────

    @staticmethod
    def _build_schedule(
        schedule: str | ExplorationSchedule,
        *,
        initial_exploration: float,
        final_exploration: float,
        exploitation_exploration: float,
        stagnation_patience: int,
        decay_generations: int,
    ) -> ExplorationSchedule:
        if isinstance(schedule, ExplorationSchedule):
            return schedule
        if schedule not in _SCHEDULE_REGISTRY:
            raise exc.InvalidAPIUsage(
                f"Unknown exploration schedule {schedule!r}. "
                f"Valid choices: {sorted(_SCHEDULE_REGISTRY)}"
            )
        if schedule == "linear_decay":
            return LinearDecaySchedule(
                initial_ratio=initial_exploration,
                final_ratio=final_exploration,
                decay_generations=decay_generations,
            )
        # "threshold_trigger"
        return ThresholdTriggerSchedule(
            initial_ratio=initial_exploration,
            final_ratio=final_exploration,
            exploitation_ratio=exploitation_exploration,
            stagnation_threshold=stagnation_patience,
            decay_generations=decay_generations,
        )

    # ── Kernel code helper ────────────────────────────────────────────────────

    def _get_kernel_code(self) -> str:
        """
        Return the Triton source for this kernel, computed once from the default
        config and cached.  Falls back to the kernel name if codegen fails.
        """
        if self._kernel_code is None:
            try:
                default_config = self.config_gen.unflatten(
                    self.config_gen.default_flat()
                )
                self._kernel_code = (
                    self.kernel.to_triton_code(default_config) or self._kernel_name
                )
            except Exception:
                self._kernel_code = self._kernel_name
        return self._kernel_code

    # ── Config → text serialisation ───────────────────────────────────────────

    def _config_to_text(
        self,
        flat_config: FlatConfig,
        beam: list[PopulationMember] | None = None,
    ) -> str:
        """
        Serialise a flat config to a human-readable string for embedding.

        When ``beam`` is provided (generation 1+), the text is prefixed with
        the kernel name and the current beam configs with their measured latencies
        so the embedding model has performance context when encoding the candidate.
        Without beam context the simpler format is used for the initial population.
        """
        kernel_code = self._get_kernel_code()
        config = self.config_gen.unflatten(flat_config)
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(config.config.items()))
        if not beam:
            return f"Triton Kernel code: {kernel_code}\nconfig: {pairs}"
        beam_lines = [
            "  "
            + ", ".join(
                f"{k}={v}"
                for k, v in sorted(
                    self.config_gen.unflatten(m.flat_values).config.items()
                )
            )
            + f" → {m.perf:.3f} ms"
            for m in beam
        ]
        return (
            f"Triton Kernel code: {kernel_code}\n"
            f"best configs so far:\n" + "\n".join(beam_lines) + f"\ncandidate: {pairs}"
        )

    # ── Embedding helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_embedding(raw: list[float]) -> np.ndarray:
        arr = np.array(raw, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm > 0.0:
            arr = arr / norm
        return arr

    def _embed_configs(
        self,
        flat_configs: list[FlatConfig],
        beam: list[PopulationMember] | None = None,
    ) -> list[np.ndarray]:
        """
        Return one L2-normalised embedding per flat_config.

        Without beam context (initial population): uses the cache so repeated
        configs are free and the results can be reused for index/beam lookups.

        With beam context (generation 1+): skips the cache entirely because the
        text changes every generation as the beam evolves, so cached embeddings
        from a prior generation would be stale.
        """
        texts = [self._config_to_text(f, beam) for f in flat_configs]
        if beam:
            if not texts:
                return []
            raw_list = self._client.embed_batch(texts)
            return [self._normalize_embedding(r) for r in raw_list]
        return self._cache.get_or_fetch(flat_configs, self._client, texts)

    def _beam_embedding_matrix(self, beam: list[PopulationMember]) -> np.ndarray:
        """
        Stack cached embeddings of beam members into a (B, dim) matrix.
        Members whose embedding is not cached are skipped (should not happen
        in normal operation because the beam is always seeded from embedded
        configs).
        """
        vecs = [
            self._cache.get(m.flat_values)
            for m in beam
            if self._cache.get(m.flat_values) is not None
        ]
        if not vecs:
            return np.zeros((0,))
        return np.stack(vecs)

    # ── Neighbor generation ───────────────────────────────────────────────────

    def _generate_candidate_flat_configs(
        self,
        beam: list[PopulationMember],
        visited: set[Config],
    ) -> list[FlatConfig]:
        """
        Generate candidate flat configs by applying ``pattern_neighbors`` to
        every beam member.

        For each beam member we collect all possible single-parameter changes
        and block-size pair changes (identical to PatternSearch's neighborhood),
        then randomly sample up to ``num_neighbors // beam_width`` of them.
        Results are deduplicated across beam members and filtered against the
        visited set.

        Args:
            beam:    Current beam of PopulationMember objects.
            visited: Set of already-benchmarked Config objects.

        Returns:
            List of unique unvisited FlatConfig candidates.
        """
        neighbors_per_member = max(1, self.num_neighbors // max(1, len(beam)))
        seen_keys: set[tuple[object, ...]] = set()
        candidates: list[FlatConfig] = []

        for member in beam:
            base = member.flat_values
            neighbors_by_idx = [
                spec.pattern_neighbors(base[idx])
                for idx, spec in enumerate(self.config_gen.flat_spec)
            ]

            # Single-parameter changes
            pool: list[FlatConfig] = []
            for idx, neighbor_vals in enumerate(neighbors_by_idx):
                for val in neighbor_vals:
                    new_flat: FlatConfig = [*base]
                    new_flat[idx] = val
                    pool.append(new_flat)

            # Block-size pair changes (cross-dimension exploration)
            block_indices = self.config_gen.block_size_indices
            for i, first in enumerate(block_indices):
                for second in block_indices[i + 1 :]:
                    for fv in neighbors_by_idx[first]:
                        for sv in neighbors_by_idx[second]:
                            new_flat = [*base]
                            new_flat[first] = fv
                            new_flat[second] = sv
                            pool.append(new_flat)

            # Randomly sample to stay within the per-member budget
            random.shuffle(pool)
            for flat in pool[:neighbors_per_member]:
                key = EmbeddingCache._key(flat)
                if key in seen_keys:
                    continue
                config = self.config_gen.unflatten(flat)
                if config in visited:
                    continue
                seen_keys.add(key)
                candidates.append(flat)

        return candidates

    # ── Candidate scoring and selection ───────────────────────────────────────

    def _select_candidates(
        self,
        candidates: list[FlatConfig],
        candidate_embeddings: list[np.ndarray],
        beam: list[PopulationMember],
        exploration_ratio: float,
    ) -> list[FlatConfig]:
        """
        Score candidates and return the top ``frac_selected`` fraction.

        Score = exploration_ratio       × diversity_score(candidate)
              + (1 - exploration_ratio) × exploitation_score(candidate)

        diversity_score:    how novel the candidate is compared to everything
                            seen so far (1 = completely new region).
        exploitation_score: how similar the candidate is to current beam
                            members (1 = very close to a good config).

        Args:
            candidates:           FlatConfig list to score.
            candidate_embeddings: Parallel list of L2-normalised embeddings.
            beam:                 Current beam of PopulationMember objects.
            exploration_ratio:    Blend factor ∈ [0, 1].

        Returns:
            Top ``frac_selected`` fraction of candidates ordered by score.
        """
        n_to_select = max(1, int(len(candidates) * self.frac_selected))

        if not candidate_embeddings:
            # No embeddings available — fall back to random selection.
            selected = list(candidates)
            random.shuffle(selected)
            return selected[:n_to_select]

        cand_matrix = np.stack(candidate_embeddings)  # (C, dim)
        diversity = self._index.diversity_scores(cand_matrix)
        exploitation = self._index.exploitation_scores(
            cand_matrix, self._beam_embedding_matrix(beam)
        )
        scores = (
            exploration_ratio * diversity + (1.0 - exploration_ratio) * exploitation
        )

        top_indices = np.argsort(scores)[::-1][:n_to_select]
        return [candidates[i] for i in top_indices]

    # ── Beam update ───────────────────────────────────────────────────────────

    def _update_beam(
        self,
        beam: list[PopulationMember],
        new_members: list[PopulationMember],
    ) -> list[PopulationMember]:
        """
        Merge the current beam with newly benchmarked members and return the
        top ``beam_width`` configs by performance.

        Only members with a finite (non-inf) perf value are considered.
        If no finite results exist the current beam is returned unchanged.
        """
        candidates = [m for m in [*beam, *new_members] if math.isfinite(m.perf)]
        if not candidates:
            return beam
        candidates.sort(key=performance)
        return candidates[: self.beam_width]

    # ── Stagnation detection ──────────────────────────────────────────────────

    def _is_stagnant(self, current_best_perf: float, prior_best_perf: float) -> bool:
        """
        Return True if the relative improvement is below ``min_improvement_delta``
        or performance did not improve at all.
        """
        if current_best_perf >= prior_best_perf:
            return True
        if prior_best_perf == 0.0:
            return False
        return (
            abs(current_best_perf / prior_best_perf - 1.0) < self.min_improvement_delta
        )

    # ── Main autotune loop ────────────────────────────────────────────────────

    def _autotune(self) -> Config:
        self.log(
            f"Starting EmbeddingBeamSearch: "
            f"beam_width={self.beam_width}, "
            f"initial_population={self.initial_population}, "
            f"max_generations={self.max_generations}, "
            f"num_neighbors={self.num_neighbors}, "
            f"frac_selected={self.frac_selected}, "
            f"schedule={self._schedule.description()}"
        )

        _t_start = time.perf_counter()
        visited: set[Config] = set()

        # ── Phase 1: initial population ───────────────────────────────────────
        init_flat_configs = self.config_gen.random_population_flat(
            self.initial_population
        )
        self.population = []
        for flat in init_flat_configs:
            member = self.make_unbenchmarked(flat)
            if member.config not in visited:
                visited.add(member.config)
                self.population.append(member)

        self.set_generation(0)
        self._timed_benchmark(self.population, desc="Initial population")
        self.set_adaptive_compile_timeout(
            self.population,
            min_seconds=self.compile_timeout_lower_bound,
            quantile=self.compile_timeout_quantile,
        )
        self._timed_rebenchmark(self.population, desc="Verifying initial population")

        # Embed the initial population and populate the index.
        init_flat_list = [m.flat_values for m in self.population]
        init_embeddings = self._embed_configs(init_flat_list)
        for member, emb in zip(self.population, init_embeddings, strict=True):
            self._index.add(emb, member.flat_values, member.perf)

        # Seed the beam with the best successful configs.
        successful = sorted(
            (m for m in self.population if math.isfinite(m.perf)),
            key=performance,
        )
        if not successful:
            raise exc.NoConfigFound
        beam: list[PopulationMember] = successful[: self.beam_width]

        self.log(
            f"Initial population of {len(self.population)}: "
            f"beam seeded with {len(beam)} configs. "
            f"Best: {beam[0].perf:.3f} ms",
            self.statistics,
        )

        # ── Phase 2: generational beam search ────────────────────────────────
        stagnation_count = 0
        prior_best_perf = beam[0].perf

        for generation in range(1, self.max_generations + 1):
            exploration_ratio = self._schedule.get_ratio(generation, stagnation_count)

            # 1. Generate candidates from all beam members.
            candidates = self._generate_candidate_flat_configs(beam, visited)
            if not candidates:
                self.log(f"Generation {generation}: no new candidates, stopping early.")
                break

            # 2. Embed candidates with beam context (generation 1+).
            #    Beam members have measured latencies from the previous phase,
            #    so the model can see what good configs look like when encoding
            #    each candidate.  These embeddings are NOT cached because the
            #    beam (and therefore the text) changes every generation.
            candidate_embeddings = self._embed_configs(candidates, beam)

            # 3. Score and select.
            selected_flat = self._select_candidates(
                candidates, candidate_embeddings, beam, exploration_ratio
            )

            self.log(
                f"Generation {generation}: "
                f"exploration_ratio={exploration_ratio:.3f}, "
                f"{len(candidates)} candidates generated → "
                f"{len(selected_flat)} selected for benchmarking"
            )

            # 4. Benchmark selected candidates.
            new_members: list[PopulationMember] = []
            for flat in selected_flat:
                member = self.make_unbenchmarked(flat)
                visited.add(member.config)
                new_members.append(member)

            self.set_generation(generation)
            self._timed_benchmark(new_members, desc=f"Generation {generation}")
            self._timed_rebenchmark(
                [*beam, *new_members],
                desc=f"Generation {generation}: verifying top configs",
            )

            # 5. Add newly benchmarked configs to the embedding index.
            #    Embeddings are already in _cache from step 2; look them up.
            for member in new_members:
                emb = self._cache.get(member.flat_values)
                if emb is not None:
                    self._index.add(emb, member.flat_values, member.perf)

            # 6. Update population (for statistics) and beam.
            self.population = [*beam, *new_members]
            beam = self._update_beam(beam, new_members)

            # 7. Track stagnation.
            current_best_perf = beam[0].perf
            if self._is_stagnant(current_best_perf, prior_best_perf):
                stagnation_count += 1
            else:
                stagnation_count = 0
                prior_best_perf = current_best_perf

            self.log(
                f"Generation {generation} complete: "
                f"best={current_best_perf:.3f} ms, "
                f"stagnation_count={stagnation_count}",
                self.statistics,
            )

        emb_secs = self._client.total_embedding_secs
        emb_calls = self._client.total_embedding_calls
        bench_secs = self._benchmark_secs
        search_secs = time.perf_counter() - _t_start
        total_secs = time.perf_counter() - getattr(self, "_t_autotune_start", _t_start)
        baseline_secs = total_secs - search_secs
        other_secs = search_secs - emb_secs - bench_secs
        self.log(
            f"\n┌─ EmbeddingBeamSearch timing summary ─────────────────────\n"
            f"│  Total wall    : {total_secs:6.1f}s\n"
            f"│  Baseline/setup: {baseline_secs:6.1f}s  (default config compile + _prepare)\n"
            f"│  GPU benchmark : {bench_secs:6.1f}s  ({self._configs_benchmarked} configs)\n"
            f"│  Embedding API : {emb_secs:6.1f}s  ({emb_calls} requests)\n"
            f"│  Other         : {other_secs:6.1f}s  (scoring, neighbor gen, overhead)\n"
            f"│  Best config   : {beam[0].perf:.3f} ms\n"
            f"└──────────────────────────────────────────────────────────"
        )

        return beam[0].config
