from __future__ import annotations

import dataclasses
import functools
import logging
import math
import time
from typing import TYPE_CHECKING
from typing import Callable

from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .benchmarking import do_bench
from .benchmarking import sync_object

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .base_search import _AutotunableKernel


log = logging.getLogger(__name__)


@dataclasses.dataclass
class FidelityLevel:
    """A single level in the multi-fidelity evaluation schedule.

    Attributes:
        scale_factor: Controls benchmark precision at this level.
            Values < 1.0 use fewer repetitions (proportional to scale_factor).
            The last level (1.0) uses the full benchmark path.
        keep_ratio: Fraction of candidates to keep after this level (0.0, 1.0].
    """

    scale_factor: float
    keep_ratio: float


DEFAULT_SCHEDULE: list[FidelityLevel] = [
    FidelityLevel(scale_factor=0.25, keep_ratio=0.25),
    FidelityLevel(scale_factor=0.5, keep_ratio=0.5),
    FidelityLevel(scale_factor=1.0, keep_ratio=1.0),
]

# Minimum dimension size for scaled tensors (GPU-friendly)
MIN_DIM_SIZE = 16

# Minimum Spearman rank correlation to trust low-fidelity rankings
MIN_RANK_CORRELATION = 0.5


def _validate_schedule(schedule: list[FidelityLevel]) -> None:
    """Validate the fidelity schedule."""
    assert len(schedule) >= 1, "Schedule must have at least one level"
    assert schedule[-1].scale_factor == 1.0, "Last level must have scale_factor=1.0"
    for i in range(1, len(schedule)):
        assert schedule[i].scale_factor >= schedule[i - 1].scale_factor, (
            f"scale_factors must be non-decreasing: "
            f"{schedule[i - 1].scale_factor} -> {schedule[i].scale_factor}"
        )


def _round_to_power_of_two(n: int) -> int:
    """Round n to the nearest power of 2."""
    if n <= 0:
        return 1
    low = 1 << (n.bit_length() - 1)
    high = low << 1
    return low if (n - low) <= (high - n) else high


def scale_tensor_size(size: int, scale_factor: float) -> int:
    """Scale a single tensor dimension by scale_factor.

    - Dimensions of size 1 are unchanged.
    - Rounds to nearest power of 2.
    - Enforces minimum of MIN_DIM_SIZE.
    - Never exceeds original size.
    """
    if size <= 1:
        return size
    result = int(size * scale_factor)
    result = _round_to_power_of_two(result)
    result = max(result, MIN_DIM_SIZE)
    return min(result, size)


def _scale_args(
    args: Sequence[object], scale_factor: float
) -> tuple[list[object], bool]:
    """Scale tensor arguments by scale_factor.

    Returns:
        A tuple of (scaled_args, changed) where changed indicates
        whether any tensor was actually resized.

    Note: Helion compiles kernels with hardcoded tensor dimensions, so
    scaled tensors cannot be used with kernels compiled for original sizes.
    This function is kept for potential future use with recompilation.
    """
    import torch

    if scale_factor >= 1.0:
        return list(args), False

    scaled: list[object] = []
    changed = False
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.ndim > 0:
            new_shape = tuple(scale_tensor_size(s, scale_factor) for s in arg.shape)
            if new_shape != arg.shape:
                scaled_tensor = torch.randn(
                    new_shape, dtype=arg.dtype, device=arg.device
                )
                scaled.append(scaled_tensor)
                changed = True
            else:
                scaled.append(arg)
        else:
            scaled.append(arg)
    return scaled, changed


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two lists.

    Uses the standard formula: 1 - 6*sum(d_i^2) / (n*(n^2-1))
    where d_i is the difference in ranks.
    """
    n = len(x)
    assert n == len(y), "Lists must have equal length"
    if n <= 1:
        return 1.0

    def _ranks(values: list[float]) -> list[float]:
        indexed = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(indexed):
            ranks[idx] = float(rank)
        return ranks

    rx = _ranks(x)
    ry = _ranks(y)
    d_sq_sum = sum((a - b) ** 2 for a, b in zip(rx, ry, strict=True))
    return 1.0 - 6.0 * d_sq_sum / (n * (n * n - 1))


class MultiFidelitySearch(PopulationBasedSearch):
    """Wraps any PopulationBasedSearch with multi-fidelity staged evaluation.

    Instead of benchmarking every candidate at full precision, evaluates
    in stages with increasing benchmark precision. Bad configs are filtered
    out at cheap low-fidelity stages (fewer benchmark repetitions), so only
    a fraction of candidates receive expensive full-fidelity evaluation.

    This is inspired by Successive Halving from hyperparameter optimization,
    applied to GPU kernel autotuning.

    The inner search algorithm (e.g. PatternSearch, LFBOPatternSearch) generates
    candidate configs as usual. MultiFidelitySearch intercepts the benchmarking
    step to apply the staged evaluation.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        inner_cls: type[PopulationBasedSearch] | None = None,
        schedule: list[FidelityLevel] | None = None,
        min_candidates_for_filtering: int = 4,
        correlation_threshold: float = MIN_RANK_CORRELATION,
        **inner_kwargs: object,
    ) -> None:
        """Create a MultiFidelitySearch autotuner.

        Args:
            kernel: The kernel to be autotuned.
            args: The arguments to be passed to the kernel.
            inner_cls: The inner search algorithm class to wrap.
                Defaults to PatternSearch if not specified.
            schedule: The fidelity schedule. Defaults to DEFAULT_SCHEDULE.
            min_candidates_for_filtering: Minimum number of candidates
                needed to apply filtering. Below this, skip to full fidelity.
            correlation_threshold: Minimum Spearman rank correlation between
                low-fidelity and high-fidelity rankings to trust filtering.
                If correlation drops below this, fall back to full fidelity.
            **inner_kwargs: Additional keyword arguments passed to the inner
                search algorithm constructor.
        """
        super().__init__(kernel, args)
        self.schedule = schedule or list(DEFAULT_SCHEDULE)
        _validate_schedule(self.schedule)
        self.min_candidates_for_filtering = min_candidates_for_filtering
        self.correlation_threshold = correlation_threshold
        self._inner_kwargs = inner_kwargs

        if inner_cls is None:
            from .pattern_search import PatternSearch

            inner_cls = PatternSearch
        self._inner_cls = inner_cls

        # Statistics tracking
        self._stats: dict[str, int] = {
            "total_candidates": 0,
            "full_fidelity_evals": 0,
            "filtered_out": 0,
        }
        self._level_counts: dict[int, int] = {}
        self._fallback_count = 0
        self._time_in_low_fidelity = 0.0

    def _autotune(self) -> Config:
        """Run the inner search algorithm with multi-fidelity evaluation."""
        # pyrefly: ignore[bad-argument-type]
        inner = self._inner_cls(self.kernel, self.args, **self._inner_kwargs)

        # Share state from our prepared base
        inner._prepared = self._prepared
        inner._baseline_output = self._baseline_output
        inner._mutated_arg_indices = self._mutated_arg_indices
        inner._baseline_post_args = self._baseline_post_args
        inner._effective_atol = self._effective_atol
        inner._effective_rtol = self._effective_rtol
        inner._jobs = self._jobs
        inner._autotune_metrics = self._autotune_metrics
        inner._precompile_tmpdir = self._precompile_tmpdir
        inner._precompile_args_path = self._precompile_args_path
        inner.log = self.log
        inner.best_perf_so_far = self.best_perf_so_far

        # Patch the inner search's parallel_benchmark_population to use
        # multi-fidelity filtering
        original_parallel_benchmark_population = inner.parallel_benchmark_population

        def multi_fidelity_parallel_benchmark(
            members: list[PopulationMember], *, desc: str = "Benchmarking"
        ) -> list[PopulationMember]:
            return self._multi_fidelity_benchmark(
                inner,
                members,
                original_parallel_benchmark_population,
                desc=desc,
            )

        inner.parallel_benchmark_population = (  # type: ignore[assignment]
            multi_fidelity_parallel_benchmark
        )

        result = inner._autotune()

        # Update our state from inner
        self.best_perf_so_far = inner.best_perf_so_far
        self.population = inner.population

        self._log_statistics()
        return result

    def _multi_fidelity_benchmark(
        self,
        inner: PopulationBasedSearch,
        members: list[PopulationMember],
        original_fn: Callable[..., list[PopulationMember]],
        *,
        desc: str = "Benchmarking",
    ) -> list[PopulationMember]:
        """Apply staged multi-fidelity evaluation to a batch of candidates."""
        n = len(members)
        self._stats["total_candidates"] += n

        # Too few candidates to filter — just benchmark at full fidelity
        if n < self.min_candidates_for_filtering:
            self._stats["full_fidelity_evals"] += n
            self._level_counts.setdefault(len(self.schedule) - 1, 0)
            self._level_counts[len(self.schedule) - 1] += n
            return original_fn(members, desc=desc)

        # Apply multi-fidelity stages (skip the last level = full fidelity)
        survivors = members
        for level_idx, level in enumerate(self.schedule[:-1]):
            if len(survivors) < self.min_candidates_for_filtering:
                break

            t0 = time.perf_counter()
            low_fi_perfs = self._benchmark_at_fidelity(
                inner, survivors, level.scale_factor
            )
            self._time_in_low_fidelity += time.perf_counter() - t0

            self._level_counts.setdefault(level_idx, 0)
            self._level_counts[level_idx] += len(survivors)

            # Filter: keep top keep_ratio fraction
            n_keep = max(
                self.min_candidates_for_filtering,
                int(math.ceil(len(survivors) * level.keep_ratio)),
            )
            n_keep = min(n_keep, len(survivors))

            # Sort by low-fidelity performance (lower is better)
            indexed = sorted(range(len(survivors)), key=lambda i: low_fi_perfs[i])
            survivor_indices = set(indexed[:n_keep])
            new_survivors = [
                survivors[i] for i in range(len(survivors)) if i in survivor_indices
            ]
            filtered_count = len(survivors) - len(new_survivors)
            self._stats["filtered_out"] += filtered_count

            self.log(
                f"Multi-fidelity level {level_idx} "
                f"(scale={level.scale_factor:.0%}): "
                f"{len(survivors)} -> {len(new_survivors)} candidates "
                f"(filtered {filtered_count})"
            )

            survivors = new_survivors

        # Final stage: full-fidelity benchmark on survivors
        self._stats["full_fidelity_evals"] += len(survivors)
        self._level_counts.setdefault(len(self.schedule) - 1, 0)
        self._level_counts[len(self.schedule) - 1] += len(survivors)

        result = original_fn(survivors, desc=desc)

        # For members that were filtered out, set perf to inf
        survivor_set = {id(m) for m in survivors}
        for m in members:
            if id(m) not in survivor_set and len(m.perfs) == 0:
                m.perfs.append(math.inf)

        return result

    def _benchmark_at_fidelity(
        self,
        inner: PopulationBasedSearch,
        members: list[PopulationMember],
        scale_factor: float,
    ) -> list[float]:
        """Benchmark candidates at a reduced fidelity level.

        Uses fewer benchmark repetitions proportional to scale_factor
        for quick filtering. Configs that crash or are clearly slow
        get filtered out before the expensive full benchmark.

        Returns a list of performance values (lower is better, inf on failure).
        """
        # Scale repetitions: scale_factor=0.25 -> rep=5, scale_factor=0.5 -> rep=10
        rep = max(3, int(20 * scale_factor))
        warmup = max(1, int(5 * scale_factor))

        perfs: list[float] = []
        for member in members:
            try:
                fn = self.kernel.compile_config(member.config, allow_print=False)

                _backend = getattr(getattr(self, "config_spec", None), "backend", None)
                _bench_fn = (
                    _backend.get_do_bench() if _backend is not None else None
                ) or do_bench
                res = _bench_fn(
                    functools.partial(fn, *list(self.args)),
                    return_mode="median",
                    warmup=warmup,
                    rep=rep,
                )
                res = sync_object(res)
                assert isinstance(res, float)
                perfs.append(res)
            except Exception:
                perfs.append(math.inf)

        return perfs

    def _log_statistics(self) -> None:
        """Log multi-fidelity search statistics."""
        total = self._stats["total_candidates"]
        full = self._stats["full_fidelity_evals"]
        filtered = self._stats["filtered_out"]

        if total == 0:
            return

        pct_full = 100.0 * full / total

        self.log(
            f"Multi-fidelity statistics:\n"
            f"  Total candidates seen: {total}\n"
            f"  Full-fidelity evals: {full} ({pct_full:.1f}%)\n"
            f"  Filtered out early: {filtered}\n"
            f"  Time in low-fidelity benchmarks: "
            f"{self._time_in_low_fidelity:.1f}s\n"
            f"  Fallbacks to full fidelity: {self._fallback_count}",
            level=logging.INFO,
        )

        for level_idx in sorted(self._level_counts):
            level_count = self._level_counts[level_idx]
            if level_idx < len(self.schedule):
                scale = self.schedule[level_idx].scale_factor
                self.log(
                    f"  Level {level_idx} (scale={scale:.0%}): {level_count} evals",
                    level=logging.DEBUG,
                )
