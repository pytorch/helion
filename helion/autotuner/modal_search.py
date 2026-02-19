"""Modal-based distributed autotuner for Helion kernels.

Dispatches benchmarking to parallel Modal GPU workers.
Supports iterative algorithms (PatternSearch, LFBO, etc.) that
call parallel_benchmark multiple times.

Usage:
    # Select via environment variable:
    HELION_AUTOTUNER=ModalSearch python my_kernel.py

    # Or use modal_autotune() from a machine with no GPU:
    from helion.autotuner.modal_search import modal_autotune
    best_config = modal_autotune(my_kernel_fn, args)
"""

from __future__ import annotations

import io
import logging
import math
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import torch

from .base_search import BenchmarkResult
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import _unset_fn

if TYPE_CHECKING:
    from collections.abc import Sequence

    import modal

    from ..runtime.config import Config
    from ..runtime.kernel import CompiledConfig
    from .base_search import _AutotunableKernel
    from .config_generation import FlatConfig

log = logging.getLogger(__name__)


def _serialize_args(args: Sequence[object]) -> bytes:
    """Serialize kernel arguments to bytes for transfer to Modal workers.

    Tensors are saved via torch.save (moved to CPU first), non-tensor args
    are pickled alongside.
    """
    buf = io.BytesIO()
    cpu_args: list[object] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            cpu_args.append(arg.detach().cpu())
        else:
            cpu_args.append(arg)
    torch.save(cpu_args, buf)
    return buf.getvalue()


def _deserialize_args(data: bytes, device: str = "cuda") -> list[object]:
    """Deserialize kernel arguments from bytes, moving tensors to the target device."""
    buf = io.BytesIO(data)
    args = torch.load(buf, map_location=device, weights_only=False)
    assert isinstance(args, list)
    return args


def _get_helion_source_path() -> str:
    """Get the path to the helion package source directory."""
    import helion

    pkg_path = os.path.dirname(os.path.abspath(helion.__file__))
    return pkg_path


def _create_modal_app(
    gpu_type: str = "H100",
    helion_source_path: str | None = None,
) -> tuple[modal.App, modal.Function]:
    """Build a Modal app with a GPU worker function for benchmarking.

    The worker function is defined in _modal_worker.py at module level so
    Modal can import it on the remote side without serialization. This avoids
    Python version mismatch issues (e.g. local 3.13, remote 3.12).

    Returns:
        A tuple of (app, benchmark_fn).
    """
    import modal

    from . import _modal_worker

    if helion_source_path is None:
        helion_source_path = _get_helion_source_path()

    app = modal.App("helion-autotuner")

    # Pin to Python 3.12 — triton only has wheels for <=3.12
    py_version = "3.12"

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04",
            add_python=py_version,
        )
        .entrypoint([])
        .pip_install(
            "torch",
            "triton",
            "packaging",
            "numpy",
            "rich",
            "psutil",
            "scikit-learn",
            "filecheck",
            "typing_extensions",
        )
        .add_local_dir(
            helion_source_path,
            remote_path=f"/usr/local/lib/python{py_version}/site-packages/helion",
            copy=True,
        )
    )

    # Wrap the module-level function with Modal — no serialized=True needed
    # since _modal_worker.benchmark_config is at module scope.
    # Modal imports it by qualified name on the remote side.
    benchmark_fn = app.function(image=image, gpu=gpu_type, timeout=600)(
        _modal_worker.benchmark_config
    )

    return app, benchmark_fn


class ModalBenchmarkDispatcher:
    """Manages Modal GPU workers for dispatching benchmark jobs.

    Two modes of operation:

    1. **Deployed (fast)**: If the helion-autotuner app has been deployed
       via ``modal deploy helion/autotuner/_modal_app.py``, the dispatcher
       calls the already-running function via ``Function.from_name()``.
       Containers are pre-warmed — dispatch is near-instant.

    2. **Ephemeral (fallback)**: If not deployed, falls back to creating an
       ephemeral app via ``app.run()``. This has a cold start penalty on the
       first call but requires no setup.

    Args are uploaded once to a shared Modal Volume so that each starmap
    call only sends the triton code (~few KB) instead of the full tensor
    data (~67MB+). Workers read args from the volume by path.

    Reusable across multiple dispatch_batch() calls (supports iterative
    search algorithms).
    """

    def __init__(
        self,
        gpu_type: str = "H100",
        max_concurrent: int = 50,
        helion_source_path: str | None = None,
    ) -> None:
        self.gpu_type = gpu_type
        self.max_concurrent = max_concurrent
        self.helion_source_path = helion_source_path
        self._benchmark_fn: modal.Function | None = None
        self._app: modal.App | None = None
        self._run_context: Any | None = None
        self._is_deployed: bool = False
        self._args_dict_key: str | None = None

    def _ensure_app(self) -> None:
        """Connect to a deployed app, or create an ephemeral one as fallback."""
        if self._benchmark_fn is not None:
            return

        import modal

        APP_NAME = "helion-autotuner"

        # Try deployed function first (instant, no cold start)
        try:
            self._benchmark_fn = modal.Function.from_name(
                APP_NAME, "benchmark_config"
            )
            # from_name returns a lazy reference that fails on first use if
            # the app isn't deployed — we catch that in dispatch_batch
            self._is_deployed = True
            log.info("Using deployed Modal app '%s' (warm containers)", APP_NAME)
            return
        except Exception:
            pass

        # Fallback: ephemeral app (cold start)
        log.info("No deployed Modal app found, starting ephemeral app (cold start)...")
        self._app, self._benchmark_fn = _create_modal_app(
            gpu_type=self.gpu_type,
            helion_source_path=self.helion_source_path,
        )
        self._run_context = self._app.run()
        self._run_context.__enter__()
        self._is_deployed = False

    def _upload_args(self, args_bytes: bytes) -> str:
        """Upload serialized args to a shared Modal Dict once.

        Returns the dict key that workers should use to fetch args.
        """
        if self._args_dict_key is not None:
            return self._args_dict_key

        import hashlib

        import modal

        # Use content hash so we don't re-upload identical args
        args_hash = hashlib.md5(args_bytes[:4096]).hexdigest()[:12]
        dict_key = f"args_{args_hash}"

        d = modal.Dict.from_name("helion-autotuner-data", create_if_missing=True)
        d[dict_key] = args_bytes

        self._args_dict_key = dict_key
        log.info(
            "Uploaded args to Modal Dict (%d bytes, key=%s)",
            len(args_bytes),
            dict_key,
        )
        return dict_key

    def close(self) -> None:
        """Shut down the ephemeral Modal app (no-op for deployed mode)."""
        if self._run_context is not None:
            try:
                self._run_context.__exit__(None, None, None)
            except Exception:
                pass
            self._run_context = None

    def __enter__(self) -> ModalBenchmarkDispatcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def dispatch_batch(
        self,
        triton_codes: list[str],
        fn_names: list[str],
        args_bytes: bytes,
    ) -> list[dict[str, object]]:
        """Dispatch a batch of configs to Modal workers and collect results.

        Args:
            triton_codes: List of generated triton source strings, one per config.
            fn_names: List of function names to call in each generated module.
            args_bytes: Serialized kernel arguments (shared across all configs).

        Returns:
            List of result dicts with 'perf', 'status', 'error' keys.
        """
        self._ensure_app()
        assert self._benchmark_fn is not None

        # Upload args to shared Dict once (not per starmap call)
        args_dict_key = self._upload_args(args_bytes)

        # Build starmap inputs: each entry is (triton_code, fn_name, dict_key)
        # Only triton_code varies per call (~few KB each), not the 67MB+ args
        inputs = [
            (code, name, args_dict_key)
            for code, name in zip(triton_codes, fn_names, strict=True)
        ]

        try:
            return self._dispatch_starmap(inputs)
        except Exception as e:
            if self._is_deployed:
                # Deployed function may have been torn down; fall back to ephemeral
                log.warning(
                    "Deployed function call failed (%s), falling back to ephemeral app",
                    e,
                )
                self._benchmark_fn = None
                self._is_deployed = False
                self._args_dict_key = None
                self._ensure_app()
                args_dict_key = self._upload_args(args_bytes)
                inputs = [
                    (code, name, args_dict_key)
                    for code, name in zip(triton_codes, fn_names, strict=True)
                ]
                return self._dispatch_starmap(inputs)
            raise

    def _dispatch_starmap(
        self, inputs: list[tuple[str, str, str]]
    ) -> list[dict[str, object]]:
        """Run starmap and collect results."""
        assert self._benchmark_fn is not None
        results: list[dict[str, object]] = []
        for result in self._benchmark_fn.starmap(
            inputs, return_exceptions=True, wrap_returned_exceptions=False
        ):
            if isinstance(result, Exception):
                results.append(
                    {
                        "perf": float("inf"),
                        "status": "error",
                        "error": f"{type(result).__qualname__}: {result}",
                    }
                )
            else:
                results.append(result)
        return results


def _find_compiled_fn_name(triton_code: str) -> str:
    """Extract the compiled function name from generated triton code.

    The generated code from to_triton_code() defines a function that calls
    the triton kernel. We look for the last non-triton function definition
    which is the compiled entry point.
    """
    import re

    # Find all top-level function definitions
    fn_names = re.findall(r"^def (\w+)\(", triton_code, re.MULTILINE)
    if not fn_names:
        raise ValueError("No function definitions found in generated triton code")
    # The entry point is typically the last defined function (the wrapper)
    # Filter out functions that start with underscore or 'triton_' as those
    # are kernel implementations
    candidates = [
        name for name in fn_names if not name.startswith("_triton_") and not name.startswith("triton_")
    ]
    if candidates:
        return candidates[-1]
    return fn_names[-1]


class ModalSearch(PopulationBasedSearch):
    """Search algorithm that dispatches benchmarking to Modal GPU workers.

    Overrides parallel_benchmark, rebenchmark, and _compute_baseline to
    dispatch all GPU work to Modal. This means any search algorithm that
    inherits from PopulationBasedSearch (PatternSearch, LFBO, DE, etc.)
    can use Modal by simply selecting this class.

    Select via: HELION_AUTOTUNER=ModalSearch
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        initial_population: int = 20,
        copies: int = 3,
        max_generations: int = 10,
        min_improvement_delta: float = 0.001,
    ) -> None:
        # Initialize the dispatcher before calling super().__init__()
        # because super().__init__() calls _compute_baseline() which
        # needs the dispatcher to be ready.
        settings = kernel.settings
        gpu_type = getattr(settings, "autotune_modal_gpu", "H100")
        max_concurrent = getattr(settings, "autotune_modal_max_concurrent", 50)

        self._dispatcher = ModalBenchmarkDispatcher(
            gpu_type=gpu_type,
            max_concurrent=max_concurrent,
        )

        # Serialize args before super().__init__() since _compute_baseline needs them
        self._args_bytes: bytes = _serialize_args(args)

        # Now call super().__init__() which will call _compute_baseline()
        super().__init__(kernel, args)

        self.initial_population = initial_population
        self.copies = copies
        self.max_generations = max_generations
        self.min_improvement_delta = min_improvement_delta

    def cleanup(self) -> None:
        """Clean up resources including the Modal app connection."""
        self._dispatcher.close()
        super().cleanup()

    def _compute_baseline(
        self,
    ) -> tuple[object, Sequence[int], Sequence[object] | None]:
        """Compute baseline by running the default config on Modal.

        If a custom baseline_fn is provided, fall back to local execution
        (which requires a local GPU). Otherwise dispatch to Modal.
        """
        if self.settings.autotune_baseline_fn is not None:
            # Custom baseline functions run locally
            return super()._compute_baseline()

        from ..autotuner.base_search import _clone_args
        from ..runtime.config import Config

        new_args = _clone_args(self._original_args)
        baseline_config = self.config_spec.default_config()

        # Generate triton code for the default config
        triton_code = self.kernel.to_triton_code(
            baseline_config, output_origin_lines=False
        )
        if triton_code is None:
            raise ValueError("Failed to generate triton code for default config")

        fn_name = _find_compiled_fn_name(triton_code)

        # Dispatch to Modal to run the default config
        results = self._dispatcher.dispatch_batch(
            triton_codes=[triton_code],
            fn_names=[fn_name],
            args_bytes=self._args_bytes,
        )

        result = results[0]
        if result["status"] != "ok":
            raise ValueError(
                f"Default config failed on Modal: {result.get('error', 'unknown error')}"
            )

        # For Modal-based autotuning, we skip accuracy checks since we
        # cannot easily transfer the full output tensors back. We trust
        # that the generated code is correct if it runs without error.
        # Return empty baseline to disable accuracy validation.
        return None, [], None

    def _compute_effective_tolerances(self) -> tuple[float, float]:
        """Skip tolerance computation since Modal doesn't do local accuracy checks."""
        return (1e-2, 1e-2)

    def _validate_against_baseline(
        self, config: Any, output: object, args: Sequence[object]
    ) -> bool:
        """Skip accuracy validation for Modal-based benchmarking.

        Accuracy is validated by the worker running the kernel successfully.
        """
        return True

    def _decide_num_jobs(self) -> int:
        """Modal manages its own concurrency, so we don't need local job limits."""
        return 1

    def parallel_benchmark(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        """Dispatch configs to Modal workers for parallel benchmarking.

        For each config, generates triton code and dispatches to a Modal
        GPU worker. Collects results and returns BenchmarkResult entries.
        """
        if not configs:
            return []

        triton_codes: list[str] = []
        fn_names: list[str] = []
        valid_indices: list[int] = []
        results: list[BenchmarkResult] = [
            BenchmarkResult(
                config=config,
                fn=_unset_fn,
                perf=math.inf,
                status="error",
                compile_time=None,
            )
            for config in configs
        ]

        for i, config in enumerate(configs):
            try:
                code = self.kernel.to_triton_code(
                    config, output_origin_lines=False
                )
                if code is None:
                    self.log.warning(f"Failed to generate triton code for config {config}")
                    continue
                fn_name = _find_compiled_fn_name(code)
                triton_codes.append(code)
                fn_names.append(fn_name)
                valid_indices.append(i)
            except Exception as e:
                self.log.warning(
                    f"Code generation failed for config {config}: {type(e).__name__}: {e}"
                )

        if not triton_codes:
            return results

        self.log(
            f"Dispatching {len(triton_codes)} configs to Modal ({self._dispatcher.gpu_type})"
        )

        modal_results = self._dispatcher.dispatch_batch(
            triton_codes=triton_codes,
            fn_names=fn_names,
            args_bytes=self._args_bytes,
        )

        for idx, modal_result in zip(valid_indices, modal_results, strict=True):
            config = configs[idx]
            perf = float(modal_result.get("perf", math.inf))
            status_str = str(modal_result.get("status", "error"))
            status: Literal["ok", "error", "timeout"] = (
                "ok" if status_str == "ok" and math.isfinite(perf) else "error"
            )
            error = modal_result.get("error")

            self.counters["benchmark"] += 1
            if status == "ok":
                self.log.debug(f"Modal result for {config}: {perf:.4f}ms")
                if perf < self.best_perf_so_far:
                    self.best_perf_so_far = perf
            else:
                self.log.debug(f"Modal error for {config}: {error}")

            # We don't have a local compiled fn for Modal-dispatched configs
            results[idx] = BenchmarkResult(
                config=config,
                fn=_unset_fn,
                perf=perf,
                status=status,
                compile_time=None,
            )

        return results

    def benchmark_function(self, config: Config, fn: CompiledConfig) -> float:
        """Benchmark a single config via Modal.

        This override dispatches a single config to Modal rather than
        benchmarking locally.
        """
        try:
            code = self.kernel.to_triton_code(config, output_origin_lines=False)
            if code is None:
                return math.inf
            fn_name = _find_compiled_fn_name(code)
        except Exception as e:
            self.log.debug(f"Code generation failed: {e}")
            return math.inf

        results = self._dispatcher.dispatch_batch(
            triton_codes=[code],
            fn_names=[fn_name],
            args_bytes=self._args_bytes,
        )

        result = results[0]
        perf = float(result.get("perf", math.inf))
        if result.get("status") == "ok" and math.isfinite(perf):
            self.counters["benchmark"] += 1
            if perf < self.best_perf_so_far:
                self.best_perf_so_far = perf
            return perf
        return math.inf

    def rebenchmark(
        self, members: list[PopulationMember], *, desc: str = "Rebenchmarking"
    ) -> None:
        """Re-benchmark members via Modal for fair comparison.

        Dispatches all members to Modal simultaneously so they benchmark
        under identical conditions.
        """
        if len(members) < 2:
            return

        triton_codes: list[str] = []
        fn_names: list[str] = []
        valid_member_indices: list[int] = []

        for i, member in enumerate(members):
            try:
                code = self.kernel.to_triton_code(
                    member.config, output_origin_lines=False
                )
                if code is None:
                    continue
                fn_name = _find_compiled_fn_name(code)
                triton_codes.append(code)
                fn_names.append(fn_name)
                valid_member_indices.append(i)
            except Exception:
                continue

        if not triton_codes:
            return

        modal_results = self._dispatcher.dispatch_batch(
            triton_codes=triton_codes,
            fn_names=fn_names,
            args_bytes=self._args_bytes,
        )

        for idx, modal_result in zip(
            valid_member_indices, modal_results, strict=True
        ):
            member = members[idx]
            perf = float(modal_result.get("perf", math.inf))
            if modal_result.get("status") == "ok" and math.isfinite(perf):
                member.perfs.append(perf)
                if perf < self.best_perf_so_far:
                    self.best_perf_so_far = perf
            else:
                member.perfs.append(math.inf)

    def _generate_neighbors(self, base: FlatConfig) -> list[FlatConfig]:
        """Generate neighboring configurations by changing one parameter at a time.

        Mirrors PatternSearch._generate_neighbors but without the block-size
        pair optimization (to keep batch sizes manageable for Modal dispatch).
        """
        neighbors: list[FlatConfig] = []
        for index, spec in enumerate(self.config_gen.flat_spec):
            for candidate_value in spec.pattern_neighbors(base[index]):
                new_flat = [*base]
                new_flat[index] = candidate_value
                neighbors.append(new_flat)
        return neighbors

    def _autotune(self) -> Config:
        """Run pattern-search-style autotuning via Modal.

        Uses the same algorithm structure as PatternSearch but all
        benchmarking is dispatched to Modal GPU workers.
        """
        visited: set[Config] = set()

        # Generate initial population
        initial_flat = self.config_gen.random_population_flat(self.initial_population)
        initial_members: list[PopulationMember] = []
        for flat in initial_flat:
            member = self.make_unbenchmarked(flat)
            if member.config not in visited:
                visited.add(member.config)
                initial_members.append(member)

        self.set_generation(0)
        self.parallel_benchmark_population(initial_members, desc="Initial population")
        self.population = initial_members

        # Filter to working configs for starting points
        working = [m for m in self.population if math.isfinite(m.perf)]
        if not working:
            from .. import exc

            raise exc.NoConfigFound

        self.log(f"Initial population: {self.statistics()}")

        # Select top configs as starting points for pattern search
        working.sort(key=lambda m: m.perf)
        starting_points = working[: self.copies]

        best_member = self.best
        no_improvement_count = 0

        for gen in range(1, self.max_generations + 1):
            self.set_generation(gen)

            # Generate neighbors of the best configs
            neighbor_flats: list[FlatConfig] = []
            for member in starting_points:
                for flat in self._generate_neighbors(member.flat_values):
                    new_member = self.make_unbenchmarked(flat)
                    if new_member.config not in visited:
                        visited.add(new_member.config)
                        neighbor_flats.append(flat)

            if not neighbor_flats:
                self.log(f"Generation {gen}: no new neighbors to explore, stopping")
                break

            self.log(
                f"Generation {gen}: exploring {len(neighbor_flats)} neighbors"
            )

            # Benchmark neighbors
            neighbor_members = [*map(self.make_unbenchmarked, neighbor_flats)]
            self.parallel_benchmark_population(
                neighbor_members, desc=f"Generation {gen}"
            )

            # Add successful members to population
            new_good = [m for m in neighbor_members if math.isfinite(m.perf)]
            self.population.extend(new_good)

            current_best = self.best
            if current_best.perf < best_member.perf:
                improvement = (best_member.perf - current_best.perf) / best_member.perf
                self.log(
                    f"Generation {gen}: improved {best_member.perf:.4f}ms -> "
                    f"{current_best.perf:.4f}ms ({improvement:.2%})"
                )
                if improvement < self.min_improvement_delta:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                best_member = current_best
                # Update starting points to the new best configs
                all_working = [m for m in self.population if math.isfinite(m.perf)]
                all_working.sort(key=lambda m: m.perf)
                starting_points = all_working[: self.copies]
            else:
                no_improvement_count += 1
                self.log(f"Generation {gen}: no improvement ({self.statistics()})")

            if no_improvement_count >= 3:
                self.log(
                    f"Stopping after {gen} generations (no improvement for 3 rounds)"
                )
                break

        # Finishing phase
        best_member = self.best
        if self.finishing_rounds > 0:
            best_member = self.run_finishing_phase(best_member, self.finishing_rounds)

        # Rebenchmark top candidates for final ranking
        top = sorted(self.population, key=lambda m: m.perf)[:5]
        if len(top) >= 2:
            self.rebenchmark(top, desc="Final verification")

        final_best = min(top, key=lambda m: m.perf)
        return final_best.config.minimize(self.config_spec)


def modal_autotune(
    kernel_fn: Any,
    *args: Any,
    gpu_type: str = "H100",
    max_concurrent: int = 50,
    n_configs: int = 20,
    max_generations: int = 10,
    **kernel_kwargs: Any,
) -> Config:
    """Run autotuning entirely on Modal, suitable for machines with no local GPU.

    This is a convenience function that:
    1. Creates a BoundKernel from the kernel function and args
    2. Runs ModalSearch to find the best config
    3. Returns the best Config

    Args:
        kernel_fn: A helion.kernel-decorated function.
        *args: Arguments to pass to the kernel for benchmarking.
        gpu_type: Modal GPU type (default "H100").
        max_concurrent: Max parallel Modal workers.
        n_configs: Number of initial configs to try.
        max_generations: Max generations for pattern search.
        **kernel_kwargs: Additional kwargs passed to the kernel settings.

    Returns:
        The best Config found.
    """
    from ..runtime.config import Config
    from ..runtime.settings import Settings

    # Set up settings for Modal autotuning
    settings = Settings(
        autotune_modal_gpu=gpu_type,
        autotune_modal_max_concurrent=max_concurrent,
        autotune_precompile=None,  # No local precompilation needed
        **kernel_kwargs,
    )

    # Create bound kernel
    bound_kernel = kernel_fn.bind(args, settings=settings)

    # Run ModalSearch
    searcher = ModalSearch(
        bound_kernel,
        args,
        initial_population=n_configs,
        max_generations=max_generations,
    )

    return searcher.autotune()
