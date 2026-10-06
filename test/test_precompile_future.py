from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock
from unittest.mock import patch
import weakref

import torch

from helion import exc
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.precompile_future import PrecompileContext
from helion.autotuner.precompile_future import PrecompileFuture
from helion.autotuner.precompile_future import SerializedCompiledFunction
from helion.autotuner.precompile_future import _ExtractedLaunchArgs
from helion.autotuner.precompile_future import _prepare_precompiler_for_fork
from helion.runtime.config import Config
from helion.runtime.settings import Settings


class TestPrecompileCapability(unittest.TestCase):
    def _provider(
        self, settings: Settings, *, supported: bool
    ) -> LocalBenchmarkProvider:
        backend = SimpleNamespace(
            name=settings.backend,
            supports_precompile=lambda: supported,
            get_do_bench=lambda: None,
            probe_long_autotune_kernels=lambda _config_spec: False,
        )
        config_spec = SimpleNamespace(backend=backend, cute_flash_search_enabled=False)
        kernel = SimpleNamespace(
            settings=settings,
            config_spec=config_spec,
            supports_subprocess_benchmark=lambda: True,
            env=SimpleNamespace(device=torch.device("cpu"), process_group_name=None),
        )
        with patch.object(
            LocalBenchmarkProvider, "_compute_baseline", return_value=(None, [], None)
        ):
            provider = LocalBenchmarkProvider(
                kernel, settings, config_spec, (torch.ones(1),), Mock(), Mock()
            )
        self.addCleanup(provider.cleanup)
        return provider

    def test_unsupported_precompile_preserves_benchmark_worker_and_settings(
        self,
    ) -> None:
        for mode in ("fork", "spawn"):
            with self.subTest(mode=mode):
                settings = Settings(
                    backend="cute",
                    autotune_precompile=mode,
                    autotune_benchmark_subprocess=True,
                    autotune_benchmark_timeout=47,
                    autotune_compile_timeout=13,
                )
                provider = self._provider(settings, supported=False)
                self.assertIsNone(provider.settings.autotune_precompile)
                self.assertIsNot(provider.settings, settings)
                self.assertIs(provider.kernel.settings, settings)
                self.assertEqual(settings.autotune_precompile, mode)
                self.assertTrue(provider._subprocess_benchmark_enabled())
                self.assertEqual(provider.settings.autotune_benchmark_timeout, 47)
                self.assertEqual(provider.settings.autotune_compile_timeout, 13)
                self.assertIn("does not support", provider.log.call_args.args[0])

                provider.setup()
                self.assertTrue(Path(provider._precompile_args_path).is_file())
                with patch(
                    "helion.autotuner.precompile_future.make_precompiler"
                ) as triton_precompiler:
                    self.assertTrue(
                        provider._create_precompile_future(Config(), Mock()).ok
                    )
                triton_precompiler.assert_not_called()

                worker = Mock()
                worker.run.return_value = 0.25
                provider._benchmark_worker = worker
                spec = SerializedCompiledFunction("kernel", "", None, None)
                with patch(
                    "helion.autotuner.benchmark_provider._serialize_compiled_fn",
                    return_value=spec,
                ):
                    latency = provider._run_subprocess_benchmark_job(
                        Mock(), warmup=1, rep=50
                    )
                self.assertEqual(latency, 0.25)
                self.assertEqual(worker.run.call_args.kwargs, {"timeout": 47.0})

    def test_supported_modes_and_explicit_disable_are_unchanged(self) -> None:
        for mode, supported in (("fork", True), ("spawn", True), (None, False)):
            with self.subTest(mode=mode, supported=supported):
                settings = Settings(
                    backend="triton",
                    autotune_precompile=mode,
                    autotune_precompile_jobs=2,
                )
                provider = self._provider(settings, supported=supported)
                self.assertIs(provider.settings, settings)
                self.assertEqual(provider.settings.autotune_precompile, mode)
                provider.log.assert_not_called()

    def test_fork_helper_rejects_unsupported_backend_before_launch_extraction(
        self,
    ) -> None:
        provider = self._provider(Settings(backend="cute"), supported=False)
        fn = Mock()
        with (
            patch("helion.autotuner.precompile_future.make_precompiler") as precompiler,
            self.assertRaisesRegex(exc.InvalidAPIUsage, "does not support"),
        ):
            _prepare_precompiler_for_fork(
                fn, (), Config(), provider.kernel, "@kernel", provider.log
            )
        fn.assert_not_called()
        precompiler.assert_not_called()


class TestPrecompileDiagnostics(unittest.TestCase):
    def _context(self, *, ignore_errors: bool = False) -> PrecompileContext:
        settings = Settings(
            autotune_precompile="fork", autotune_ignore_errors=ignore_errors
        )
        kernel = SimpleNamespace(format_kernel_decorator=lambda *_args: "@kernel")
        return PrecompileContext(settings, Mock(), kernel, (), 1)

    def _create(self, context: PrecompileContext) -> PrecompileFuture:
        return PrecompileFuture.create(
            context, Config(), Mock(), (), "unused-result.pkl", None
        )

    def test_launch_tensors_are_not_exception_message_arguments(self) -> None:
        class Buffer:
            def __repr__(self) -> str:
                raise AssertionError("Formatting an exception must not inspect buffers")

        buffer = Buffer()
        error = _ExtractedLaunchArgs(object(), (1,), (buffer,), {})
        self.assertEqual(error.args, ())
        self.assertEqual(str(error), "")
        self.assertIs(error.launch_args[0], buffer)

    def test_preparation_error_preserves_text_and_releases_exception_frames(
        self,
    ) -> None:
        refs: list[weakref.ReferenceType[torch.Tensor]] = []

        def preparation_failure(*_args: object) -> None:
            temporary = torch.ones(1)
            refs.append(weakref.ref(temporary))
            try:
                raise RuntimeError("underlying compiler failure")
            except RuntimeError as cause:
                raise ValueError("precompile failed") from cause

        context = self._context()
        error: Any = None
        with patch(
            "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
            side_effect=preparation_failure,
        ):
            try:
                self._create(context)
            except ValueError as caught:
                error = caught
        self.assertIsNotNone(error)
        self.assertIsNone(error.__context__)
        self.assertIsNone(error.__cause__)
        self.assertIn("underlying compiler failure", error.remote_traceback)
        self.assertIn("preparation_failure", error.remote_traceback)
        self.assertIn("precompile failed", error.remote_traceback)
        gc.collect()
        self.assertIsNone(refs[0]())

    def test_skipped_preparation_error_retains_string_diagnostic(self) -> None:
        context = self._context()
        with patch(
            "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
            side_effect=RuntimeError("out of resource: shared memory"),
        ):
            future = self._create(context)
        self.assertFalse(future.ok)
        self.assertIsNotNone(future.remote_error)
        self.assertIn("out of resource", future.remote_error.traceback)
        self.assertEqual(future.remote_error.classification, "debug")
        self.assertEqual(
            future.remote_error.exc_args, ("out of resource: shared memory",)
        )


if __name__ == "__main__":
    unittest.main()
