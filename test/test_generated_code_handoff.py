from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

import helion
from helion._compiler.backend import Backend
from helion._testing import TestCase
from helion.autotuner import HandoffBundle
from helion.autotuner import build_handoff
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.handoff import HandoffPoint
from helion.autotuner.handoff import HandoffProgress
from helion.autotuner.handoff_evaluation import HandoffEvaluation
import helion.autotuner.handoff_pipeline as pipeline
from helion.autotuner.logger import AutotuningLogger
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import KernelCompiler
from helion.runtime.kernel import PyCodeCache
from helion.runtime.ref_mode import is_ref_mode_enabled


def _cached_handoff_increment(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@unittest.skipIf(
    is_ref_mode_enabled(helion.Settings()),
    "generated source caching requires frontend compilation",
)
class TestGeneratedCodeHandoff(TestCase):
    def setUp(self) -> None:
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(
            patch.dict(
                "os.environ",
                {
                    "HELION_CACHE_DIR": str(self.root),
                    "HELION_SKIP_CACHE": "0",
                    "HELION_AUTOTUNE_HANDOFF": "0",
                },
            )
        )
        for name in ("helion_key", "torch_key_wrapper", "triton_key_wrapper"):
            stack.enter_context(patch.object(cache, name, return_value=name))
        stack.enter_context(
            patch.object(cache, "supports_torch_compile_fusion", return_value=True)
        )
        self.config = helion.Config(block_sizes=[32])
        self.sources: list[str] = []
        self.run = Mock(side_effect=lambda x: x + 1)
        stack.enter_context(
            patch.object(PyCodeCache, "load", side_effect=self._source_loader)
        )
        self.searches: list[Mock] = []
        self.selector = Mock(side_effect=self._select)

    def _source_loader(self, source: str, *, extra: str = ""):
        self.sources.append(source)
        path = self.root / f"module_{len(self.sources)}.py"
        path.write_text(source)
        # CPU tests exercise the real frontend and export; only launch is mocked.
        return SimpleNamespace(__file__=str(path), _cached_handoff_increment=self.run)

    def _select(self, bound, args):
        search = Mock(spec=BaseSearch)
        search.kernel = bound
        search.args = tuple(args)
        search.settings = bound.settings
        search.log = AutotuningLogger(bound.settings)
        search.autotune.return_value = self.config
        self.searches.append(search)
        return search

    def _kernel(self, *, handoff: bool = False):
        return helion.kernel(
            backend="triton",
            generated_code_cache=True,
            autotune_effort="quick",
            autotuner_fn=self.selector,
            autotune_handoff=handoff,
            autotune_log=str(self.root / "tuning.log"),
        )(_cached_handoff_increment)

    def _warm_binding(self, x: torch.Tensor):
        self._kernel().bind((x,)).set_config(self.config)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            bound = self._kernel().bind((x,))
        self.assertNotIsInstance(bound, BoundKernel)
        return bound

    def test_custom_selector_cache_hit_and_enabled_handoff_use_backend(self) -> None:
        x = torch.ones(16)
        first = self._kernel().bind((x,))
        self.assertEqual(first.autotune((x,), force=False), self.config)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            warm = self._kernel().bind((x,))
            self.assertEqual(warm.autotune((x,), force=False), self.config)
            torch.testing.assert_close(warm(x), x + 1)
        self.assertNotIsInstance(warm, BoundKernel)
        self.assertEqual(self.selector.call_count, 2)
        for search in self.searches:
            search.autotune.assert_called_once_with(skip_cache=False)

        with (
            patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn,
            patch.object(
                Backend, "autotune", autospec=True, side_effect=Backend.autotune
            ) as backend_autotune,
            patch.object(
                pipeline, "autotune_with_handoff", return_value=self.config
            ) as automatic,
        ):
            enabled = self._kernel(handoff=True).bind((x,))
            self.assertIsInstance(enabled, BoundKernel)
            self.assertEqual(enabled.autotune((x,), force=False), self.config)
        self.assertEqual(compile_fn.call_count, 1)
        self.assertIs(backend_autotune.call_args.args[1], enabled)
        automatic.assert_called_once_with(self.searches[-1], skip_cache=False)
        self.searches[-1].autotune.assert_not_called()

    def test_live_handoff_toggle_and_manual_cached_export(self) -> None:
        x = torch.ones(16)
        warm = self._warm_binding(x)
        warm.settings.autotune_handoff = True
        with (
            patch.object(
                Backend, "autotune", autospec=True, side_effect=Backend.autotune
            ) as backend_autotune,
            patch.object(
                pipeline, "autotune_with_handoff", return_value=self.config
            ) as automatic,
        ):
            self.assertEqual(warm.autotune((x,), force=False), self.config)
        normal = backend_autotune.call_args.args[1]
        self.assertIsInstance(normal, BoundKernel)
        self.assertIsNot(normal, warm)
        automatic.assert_called_once_with(self.searches[-1], skip_cache=False)

        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            cached = self._kernel().bind((x,))
        self.assertNotIsInstance(cached, BoundKernel)
        search = self._select(cached, (x,))
        point = HandoffPoint(
            "completed",
            self.config,
            self.run,
            HandoffProgress(1, 1, 0.1, 1.0),
            (),
            (),
            "ms",
            None,
        )
        directory = self.root / "manual_handoff"
        evaluation = HandoffEvaluation(
            True, 1.0, "ms", (), str(directory), "2026-10-09T00:00:00Z"
        )
        with (
            patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn,
            patch.object(
                HandoffBundle, "evaluate", return_value=evaluation
            ) as evaluate,
        ):
            bundle = build_handoff(search, point, directory, repetitions=1)
        self.assertEqual(compile_fn.call_count, 1)
        self.assertIs(search.kernel, cached)
        evaluate.assert_called_once_with(repetitions=1, timeout=120)
        case = bundle.manifest["cases"][0]
        self.assertEqual(case["backend"], "triton")
        self.assertEqual(case["entrypoint"], "_cached_handoff_increment")
        self.assertEqual(case["reference_kind"], "confirmed_original")
        self.assertTrue((directory / case["source"]).is_file())
        self.assertTrue((directory / case["reference"]).is_file())
        search.autotune.assert_not_called()

    def test_live_toggle_propagates_preflight_error_before_tuning(self) -> None:
        x = torch.ones(16)
        warm = self._warm_binding(x)
        warm.set_config(self.config)
        loaded = len(self.sources)
        self.run.reset_mock()
        warm.settings.autotune_handoff = True
        with (
            patch.object(
                pipeline.CLISourceAgent,
                "validate_available",
                side_effect=FileNotFoundError("Codex CLI unavailable"),
            ) as preflight,
            patch.object(pipeline, "find_handoff") as find,
            patch.object(pipeline, "build_handoff") as build,
            self.assertRaisesRegex(FileNotFoundError, "Codex CLI unavailable"),
        ):
            warm.autotune((x,), force=False)
        preflight.assert_called_once_with()
        find.assert_not_called()
        build.assert_not_called()
        self.searches[-1].autotune.assert_not_called()
        self.run.assert_not_called()
        self.assertEqual(len(self.sources), loaded)
        self.assertFalse((self.root / "tuning.handoff").exists())
