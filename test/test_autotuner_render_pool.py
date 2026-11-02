from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.context
import operator
import os
import tempfile
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Callable
from unittest.mock import Mock
from unittest.mock import patch
import zlib

import torch

import helion
from helion._compiler.backend import autotune_precompile_mode
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner import DifferentialEvolutionSearch
from helion.autotuner import PatternSearch
from helion.autotuner import benchmark_provider as benchmark_provider_module
from helion.autotuner import render_pool
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.render_pool import CudaDeviceFacts
from helion.autotuner.render_pool import RenderAhead
import helion.language as hl

if TYPE_CHECKING:
    from typing_extensions import Self


def _make_bound(**settings: object) -> tuple[object, tuple[torch.Tensor, torch.Tensor]]:
    kwargs: dict[str, object] = {
        "autotune_log_level": logging.CRITICAL,
        "autotune_benchmark_subprocess": False,
        "autotune_baseline_fn": operator.add,
        "autotune_random_seed": 1234,
        "autotune_progress_bar": False,
        **settings,
    }

    @helion.kernel(**kwargs)  # pyrefly: ignore[bad-argument-type]
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(a)
        for tile in hl.tile(out.size()):
            out[tile] = a[tile] + b[tile]
        return out

    args = (
        torch.randn([1024], device=DEVICE),
        torch.randn([1024], device=DEVICE),
    )
    return add.bind(args), args


def _configs(bound: object, count: int) -> list[helion.Config]:
    gen = bound.config_spec.create_config_generation()  # pyrefly: ignore[missing-attribute]
    return [gen.unflatten(flat) for flat in gen.random_population_flat(count)]


def _normalized(bound: object, config: helion.Config) -> helion.Config:
    return bound._normalized_config_copy(bound._normalize_config(config))  # pyrefly: ignore[missing-attribute]


def _probe_device_facts(facts: CudaDeviceFacts, conn: object) -> None:
    """Fork target: answer device queries from the facts, then try the driver."""
    facts.install()
    conn.send(  # pyrefly: ignore[missing-attribute]
        (
            torch.cuda.current_device(),
            torch.cuda.get_device_properties(None).multi_processor_count,
            torch.cuda.get_device_capability(torch.device("cuda", facts.current)),
            torch.cuda.get_device_name(facts.current),
        )
    )
    try:
        torch.cuda.synchronize()
        conn.send("synchronized")  # pyrefly: ignore[missing-attribute]
    except RuntimeError as error:
        conn.send(f"refused: {error}")  # pyrefly: ignore[missing-attribute]
    os._exit(0)


class _RecordedPools:
    """``patch`` the provider's pool class to record every pool it builds."""

    def __init__(self) -> None:
        self.created: list[RenderAhead] = []
        created = self.created

        class Recording(RenderAhead):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]
                created.append(self)

        self._patch = patch.object(benchmark_provider_module, "RenderAhead", Recording)

    def __enter__(self) -> Self:
        self._patch.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._patch.stop()


def _record_pools() -> _RecordedPools:
    return _RecordedPools()


@skipIfRefEager("Autotuning requires compilation, not supported in ref eager mode")
@onlyBackends(["cute"])
class TestRenderPool(TestCase):
    """CuTe candidates render in forked workers; the parent loads the source."""

    def test_forked_renders_match_the_parent(self) -> None:
        bound, _args = _make_bound()
        configs = _configs(bound, 6)
        facts = CudaDeviceFacts.capture(bound.env.device)  # pyrefly: ignore[missing-attribute]
        self.assertIsNotNone(facts)
        with tempfile.TemporaryDirectory() as tmp:
            counter = iter(range(10**6))
            pool = RenderAhead(
                bound,  # pyrefly: ignore[bad-argument-type]
                configs,
                jobs=3,
                timeout=120,
                facts=facts,
                result_path=lambda: os.path.join(tmp, f"render_{next(counter)}.pkl"),
                log=AutotuningLogger(bound.settings),  # pyrefly: ignore[missing-attribute]
            )
            try:
                for index, config in enumerate(configs):
                    future = pool.take(index)
                    self.assertEqual(future.status, "ok", future.error)
                    self.assertEqual(
                        future.code,
                        bound.to_triton_code(config),  # pyrefly: ignore[missing-attribute]
                    )
                    self.assertGreater(future.elapsed, 0.0)
            finally:
                pool.close()
            self.assertFalse(os.listdir(tmp))

    def test_device_facts_answer_cuda_queries_in_a_forked_child(self) -> None:
        facts = CudaDeviceFacts.capture(DEVICE)
        assert facts is not None
        ctx = mp.get_context("fork")
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(target=_probe_device_facts, args=(facts, child_conn))
        process.start()
        try:
            answers = parent_conn.recv()
            driver = parent_conn.recv()
        finally:
            process.join()
        self.assertEqual(
            answers,
            (
                facts.current,
                torch.cuda.get_device_properties(facts.current).multi_processor_count,
                torch.cuda.get_device_capability(facts.current),
                torch.cuda.get_device_name(facts.current),
            ),
        )
        # Only the captured facts are served; the driver stays off limits.
        self.assertIn("refused", driver)
        with self.assertRaisesRegex(RuntimeError, "not captured"):
            facts.index(facts.current + 1000)

    def test_provider_loads_worker_renders_and_rerenders_before_execution(
        self,
    ) -> None:
        bound, args = _make_bound()
        search = PatternSearch(bound, args, initial_population=6, copies=1)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        self.assertEqual(bound.settings.autotune_precompile, "fork")  # pyrefly: ignore[missing-attribute]
        self.assertTrue(provider._render_pool_enabled())
        self.assertFalse(provider.compiles_ahead)
        configs = _configs(bound, 6)
        parent_renders = Mock(wraps=bound.to_triton_code)  # pyrefly: ignore[missing-attribute]
        with patch.object(bound, "to_triton_code", parent_renders):
            with patch.object(
                LocalBenchmarkProvider, "_benchmark_function", return_value=1.0
            ):
                results = provider.benchmark(configs)
            self.assertEqual([r.status for r in results], ["ok"] * 6)
            # Every render happened in a worker; compile_time is the worker's.
            parent_renders.assert_not_called()
            self.assertTrue(
                all(r.compile_time is not None and r.compile_time > 0 for r in results)
            )
            normalized = [_normalized(bound, config) for config in configs]
            for config in normalized:
                self.assertIn(config, bound._compile_cache)  # pyrefly: ignore[missing-attribute]
                self.assertIn(config, bound._render_only_configs)  # pyrefly: ignore[missing-attribute]
                self.assertNotIn(config, bound._cute_reduction_extents)  # pyrefly: ignore[missing-attribute]

            # Outside the search the config about to run is rendered here,
            # once, so code generation records its facts on this kernel.
            fn = bound.compile_config(configs[0])  # pyrefly: ignore[missing-attribute]
            parent_renders.assert_called_once()
            self.assertNotIn(normalized[0], bound._render_only_configs)  # pyrefly: ignore[missing-attribute]
            self.assertIn(normalized[0], bound._cute_reduction_extents)  # pyrefly: ignore[missing-attribute]
            self.assertIs(fn, bound._compile_cache[normalized[0]])  # pyrefly: ignore[missing-attribute]
            self.assertIs(fn, bound.compile_config(configs[0]))  # pyrefly: ignore[missing-attribute]
            parent_renders.assert_called_once()

    def test_pool_falls_back_to_inline_rendering_when_fork_fails(self) -> None:
        bound, args = _make_bound()
        search = PatternSearch(bound, args, initial_population=4, copies=1)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        configs = _configs(bound, 4)
        parent_renders = Mock(wraps=bound.to_triton_code)  # pyrefly: ignore[missing-attribute]
        log = Mock(wraps=provider.log)
        with (
            patch.object(bound, "to_triton_code", parent_renders),
            patch.object(provider, "log", log),
            patch.object(
                multiprocessing.context.ForkProcess,
                "start",
                side_effect=OSError("fork refused"),
            ),
            patch.object(
                LocalBenchmarkProvider, "_benchmark_function", return_value=1.0
            ),
        ):
            results = provider.benchmark(configs)
        self.assertEqual([r.status for r in results], ["ok"] * 4)
        self.assertEqual(parent_renders.call_count, 4)
        warnings = [str(call.args[0]) for call in log.warning.call_args_list]
        expected = (
            "Could not fork a render worker (OSError: fork refused); rendering "
            "autotune candidates in-process."
        )
        self.assertEqual(
            [m for m in warnings if m.startswith("Could not fork a render worker")],
            [expected],
        )
        # Later batches do not try again.
        self.assertFalse(provider._render_pool_enabled())
        self.assertFalse([m for m in warnings if m.startswith("Could not fork")][1:])

    def test_render_timeout_falls_back_to_the_parent(self) -> None:
        # One worker never finishes; its config is rendered in-process after
        # the timeout and is not failed. Other workers may also run past the
        # short timeout under load; they fall back the same way.
        bound, args = _make_bound(autotune_compile_timeout=2)
        search = PatternSearch(bound, args, initial_population=3, copies=1)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        configs = _configs(bound, 3)
        slow = configs[0]
        real_render = render_pool._render_in_child

        def stalling_render(
            kernel: object, config: helion.Config, facts: object, result_path: str
        ) -> None:
            if config == slow:
                time.sleep(3600)
            real_render(kernel, config, facts, result_path)  # pyrefly: ignore[bad-argument-type]

        parent_renders = Mock(wraps=bound.to_triton_code)  # pyrefly: ignore[missing-attribute]
        log = Mock(wraps=provider.log)
        pools = _record_pools()
        started = time.perf_counter()
        with (
            patch.object(render_pool, "_render_in_child", stalling_render),
            patch.object(bound, "to_triton_code", parent_renders),
            patch.object(provider, "log", log),
            pools,
            patch.object(
                LocalBenchmarkProvider, "_benchmark_function", return_value=1.0
            ),
        ):
            results = provider.benchmark(configs)
        self.assertEqual([r.status for r in results], ["ok", "ok", "ok"])
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 0)
        (pool,) = pools.created
        self.assertEqual(pool.futures[0].status, "timeout")
        self.assertIn(slow, [call.args[0] for call in parent_renders.call_args_list])
        messages = [str(call.args[0]) for call in log.call_args_list]
        self.assertEqual(
            len([m for m in messages if m.startswith("Render worker exceeded 2s")]),
            1,
            messages,
        )
        self.assertFalse([f for f in pool.futures if f.running()])
        self.assertLess(time.perf_counter() - started, 60)

    def test_worker_errors_fall_back_and_disable_the_workers_after_a_streak(
        self,
    ) -> None:
        bound, args = _make_bound()
        search = PatternSearch(bound, args, initial_population=6, copies=1)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        configs = _configs(bound, 6)

        def failing_render(
            kernel: object, config: helion.Config, facts: object, result_path: str
        ) -> None:
            render_pool._write_result_file(
                result_path,
                {
                    "status": "error",
                    "error": "RuntimeError: tensor data is not readable here",
                    "traceback": "Traceback\nRuntimeError: tensor data is not readable here",
                },
            )
            os._exit(1)

        parent_renders = Mock(wraps=bound.to_triton_code)  # pyrefly: ignore[missing-attribute]
        log = Mock(wraps=provider.log)
        pools = _record_pools()
        with (
            patch.object(render_pool, "_render_in_child", failing_render),
            patch.object(bound, "to_triton_code", parent_renders),
            patch.object(provider, "log", log),
            pools,
            patch.object(
                LocalBenchmarkProvider, "_benchmark_function", return_value=1.0
            ),
        ):
            results = provider.benchmark(configs)
        self.assertEqual([r.status for r in results], ["ok"] * 6)
        # Every config was rendered in-process: three after a failed worker,
        # the rest directly once the workers were given up.
        self.assertEqual(parent_renders.call_count, 6)
        (pool,) = pools.created
        self.assertEqual([f.status for f in pool.futures[:3]], ["error"] * 3)
        self.assertTrue(
            all(f.status in ("cancelled", "unavailable") for f in pool.futures[3:]),
            [f.status for f in pool.futures],
        )
        self.assertFalse(provider._render_pool_enabled())
        warnings = [str(call.args[0]) for call in log.warning.call_args_list]
        streak = [m for m in warnings if m.startswith("Render workers fail")]
        self.assertEqual(len(streak), 1, warnings)
        self.assertIn("3 in a row", streak[0])
        self.assertIn("tensor data is not readable here", streak[0])

    def test_fork_failure_mid_batch_falls_back_for_the_rest(self) -> None:
        bound, args = _make_bound()
        search = PatternSearch(bound, args, initial_population=4, copies=1)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        configs = _configs(bound, 4)
        real_start = multiprocessing.context.ForkProcess.start
        starts = [0]

        def flaky_start(process: multiprocessing.context.ForkProcess) -> None:
            starts[0] += 1
            if starts[0] > 2:
                raise AssertionError(
                    "daemonic processes are not allowed to have children"
                )
            real_start(process)

        parent_renders = Mock(wraps=bound.to_triton_code)  # pyrefly: ignore[missing-attribute]
        log = Mock(wraps=provider.log)
        pools = _record_pools()
        with (
            patch.object(bound, "to_triton_code", parent_renders),
            patch.object(provider, "log", log),
            patch.object(multiprocessing.context.ForkProcess, "start", flaky_start),
            pools,
            patch.object(
                LocalBenchmarkProvider, "_benchmark_function", return_value=1.0
            ),
        ):
            results = provider.benchmark(configs)
        self.assertEqual([r.status for r in results], ["ok"] * 4)
        (pool,) = pools.created
        self.assertEqual(
            [f.status for f in pool.futures], ["ok", "ok", "unavailable", "unavailable"]
        )
        self.assertEqual(parent_renders.call_count, 2)
        warnings = [str(call.args[0]) for call in log.warning.call_args_list]
        expected = (
            "Could not fork a render worker (AssertionError: daemonic processes "
            "are not allowed to have children); rendering autotune candidates "
            "in-process."
        )
        self.assertEqual(
            [m for m in warnings if m.startswith("Could not fork a render worker")],
            [expected],
        )
        self.assertFalse(provider._render_pool_enabled())

    def test_budget_cut_cancels_workers_in_flight(self) -> None:
        bound, args = _make_bound()
        search = PatternSearch(bound, args, initial_population=8, copies=1)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        configs = _configs(bound, 8)
        # The budget runs out once the first config has been timed.
        provider.set_budget_exceeded_fn(
            lambda: provider._autotune_metrics.num_configs_tested >= 1
        )

        def benchmark_function(
            config: helion.Config, fn: object, **kwargs: object
        ) -> float:
            provider._autotune_metrics.num_configs_tested += 1
            return 1.0

        pools = _record_pools()
        with (
            pools,
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark_function,
            ),
        ):
            results = provider.benchmark(configs)
        self.assertEqual([r.status for r in results], ["ok"] + ["skipped"] * 7)
        (pool,) = pools.created
        self.assertTrue(pool.closed)
        self.assertFalse([f for f in pool.futures if f.running()])
        self.assertEqual(pool.futures[0].status, "ok")
        self.assertTrue(
            all(f.status in ("cancelled", "ok") for f in pool.futures[1:]),
            [f.status for f in pool.futures],
        )

    def test_exception_in_the_batch_still_stops_the_workers(self) -> None:
        bound, args = _make_bound()
        search = PatternSearch(bound, args, initial_population=6, copies=1)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        configs = _configs(bound, 6)
        pools = _record_pools()
        with (
            pools,
            patch.object(bound, "compile_config", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            provider.benchmark(configs)
        (pool,) = pools.created
        self.assertTrue(pool.closed)
        self.assertFalse([f for f in pool.futures if f.running()])
        self.assertFalse([f for f in pool.futures if f.status == "pending"])

    def test_cached_configs_get_no_worker(self) -> None:
        bound, args = _make_bound()
        search = PatternSearch(bound, args, initial_population=4, copies=1)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        configs = _configs(bound, 4)
        bound.compile_config(configs[1])  # pyrefly: ignore[missing-attribute]
        self.assertTrue(bound.has_compiled_config(configs[1]))  # pyrefly: ignore[missing-attribute]
        parent_renders = Mock(wraps=bound.to_triton_code)  # pyrefly: ignore[missing-attribute]
        pools = _record_pools()
        with (
            pools,
            patch.object(bound, "to_triton_code", parent_renders),
            patch.object(
                LocalBenchmarkProvider, "_benchmark_function", return_value=1.0
            ),
        ):
            results = provider.benchmark(configs)
        self.assertEqual([r.status for r in results], ["ok"] * 4)
        (pool,) = pools.created
        self.assertEqual([f.status for f in pool.futures], ["ok", "cached", "ok", "ok"])
        self.assertIsNone(pool.futures[1].process)
        parent_renders.assert_not_called()

    def test_render_jobs_follow_affinity_and_local_ranks(self) -> None:
        bound, args = _make_bound()
        search = PatternSearch(bound, args, initial_population=2)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        cpus = len(os.sched_getaffinity(0))
        with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": ""}):
            self.assertEqual(provider._render_jobs(), max(1, min(cpus, 32)))
        with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "8"}):
            self.assertEqual(provider._render_jobs(), max(1, min(cpus // 8, 32)))
        provider.settings.autotune_precompile_jobs = 5
        with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "8"}):
            self.assertEqual(provider._render_jobs(), 5)

    def test_adaptive_timeout_and_cache_key_ignore_the_workers(self) -> None:
        bound, args = _make_bound(
            autotune_baseline_fn=None,
            autotune_adaptive_timeout=True,
            autotune_precompile_jobs=5,
        )
        search = PatternSearch(bound, args, initial_population=2)
        search._prepare()
        members = [
            PopulationMember(
                lambda: None, [1.0], (), config, status="ok", compile_time=20.0
            )
            for config in _configs(bound, 3)
        ]
        search.set_adaptive_compile_timeout(members, min_seconds=30, quantile=0.9)
        # Render times do not fit a timeout into the kernel's shared settings.
        self.assertEqual(bound.settings.autotune_compile_timeout, 60)  # pyrefly: ignore[missing-attribute]
        policy = search.cache_policy()
        assert policy is not None
        self.assertIsNone(policy["settings"]["autotune_precompile_jobs"])  # pyrefly: ignore[index-error]

    def test_pool_keeps_the_search_trajectory(self) -> None:
        trajectories: dict[str | None, tuple[list[str], list[str], str]] = {}
        parent_render_calls: dict[str | None, int] = {}
        for mode in ("fork", None):
            bound, args = _make_bound(autotune_precompile=mode)
            search = PatternSearch(
                bound, args, initial_population=8, copies=1, max_generations=2
            )
            search.final_selection = None
            compiled: list[str] = []
            benchmarked: list[str] = []
            real_compile = bound.compile_config  # pyrefly: ignore[missing-attribute]

            def compile_config(
                config: helion.Config,
                *,
                allow_print: bool = True,
                real_compile: Callable[..., object] = real_compile,
                compiled: list[str] = compiled,
            ) -> object:
                compiled.append(repr(config))
                return real_compile(config, allow_print=allow_print)

            def benchmark_function(
                config: helion.Config,
                fn: object,
                search: PatternSearch = search,
                benchmarked: list[str] = benchmarked,
                **kwargs: object,
            ) -> float:
                search._autotune_metrics.num_configs_tested += 1
                benchmarked.append(repr(config))
                return 1.0 + (zlib.crc32(repr(config).encode()) % 1000) / 1000

            parent_renders = Mock(wraps=bound.to_triton_code)  # pyrefly: ignore[missing-attribute]
            with (
                patch.object(bound, "to_triton_code", parent_renders),
                patch.object(bound, "compile_config", side_effect=compile_config),
                patch.object(
                    LocalBenchmarkProvider,
                    "_benchmark_function",
                    side_effect=benchmark_function,
                ),
                patch.object(search, "rebenchmark", lambda *args, **kwargs: None),
            ):
                best = search.autotune()
            trajectories[mode] = (compiled, benchmarked, repr(best))
            parent_render_calls[mode] = parent_renders.call_count
        self.assertEqual(trajectories["fork"], trajectories[None])
        self.assertGreater(len(trajectories[None][0]), 8)
        # With workers the parent rendered nothing; without, everything.
        self.assertEqual(parent_render_calls["fork"], 0)
        self.assertEqual(parent_render_calls[None], len(trajectories[None][0]))

    def test_precompile_mode_cache_policy_and_de_update_with_workers(self) -> None:
        # The default baseline (the kernel itself) keeps the cache policy
        # defined; a Python baseline callable makes the search uncacheable.
        bound, args = _make_bound(autotune_baseline_fn=None)
        backend = bound.config_spec.backend  # pyrefly: ignore[missing-attribute]
        self.assertTrue(backend.supports_parallel_render())
        self.assertFalse(backend.supports_precompile())
        self.assertEqual(autotune_precompile_mode(backend, "fork"), "fork")
        self.assertIsNone(autotune_precompile_mode(backend, "spawn"))
        self.assertIsNone(autotune_precompile_mode(backend, None))
        # Any object with a precompile pool stands in for a backend.
        fake = SimpleNamespace(supports_precompile=lambda: True)
        self.assertEqual(autotune_precompile_mode(fake, "spawn"), "spawn")  # pyrefly: ignore[bad-argument-type]
        search = PatternSearch(bound, args, initial_population=2)
        self.assertEqual(bound.settings.autotune_precompile, "fork")  # pyrefly: ignore[missing-attribute]
        # Rendering in workers changes no result: the cache key ignores it.
        policy = search.cache_policy()
        assert policy is not None
        self.assertIsNone(policy["settings"]["autotune_precompile"])  # pyrefly: ignore[index-error]
        de = DifferentialEvolutionSearch(
            bound, args, population_size=2, max_generations=1
        )
        self.assertTrue(de.immediate_update)

        spawn_bound, spawn_args = _make_bound(autotune_precompile="spawn")
        PatternSearch(spawn_bound, spawn_args, initial_population=2)
        self.assertIsNone(spawn_bound.settings.autotune_precompile)  # pyrefly: ignore[missing-attribute]

        serial_bound, serial_args = _make_bound(autotune_precompile=None)
        serial = PatternSearch(serial_bound, serial_args, initial_population=2)
        serial._prepare()
        provider = serial.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        self.assertFalse(provider._render_pool_enabled())
        self.assertFalse(provider.compiles_ahead)


@skipIfRefEager("Autotuning requires compilation, not supported in ref eager mode")
@onlyBackends(["triton"])
class TestRenderPoolTriton(TestCase):
    def test_triton_keeps_its_precompile_pool_and_no_render_workers(self) -> None:
        # The default baseline (the kernel itself) keeps the cache policy
        # defined; a Python baseline callable makes the search uncacheable.
        bound, args = _make_bound(autotune_baseline_fn=None)
        backend = bound.config_spec.backend  # pyrefly: ignore[missing-attribute]
        self.assertFalse(backend.supports_parallel_render())
        self.assertEqual(autotune_precompile_mode(backend, "spawn"), "spawn")
        search = PatternSearch(bound, args, initial_population=2)
        self.assertEqual(bound.settings.autotune_precompile, "fork")  # pyrefly: ignore[missing-attribute]
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        self.assertFalse(provider._render_pool_enabled())
        self.assertTrue(provider.compiles_ahead)
        policy = search.cache_policy()
        assert policy is not None
        self.assertEqual(policy["settings"]["autotune_precompile"], "fork")  # pyrefly: ignore[index-error]
