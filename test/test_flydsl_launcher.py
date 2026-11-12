"""Launcher tests that stub flydsl, so they need neither a GPU nor flydsl.

They run in every backend lane on purpose: the launcher is covered without ROCm.
"""

from __future__ import annotations

import linecache
import sys
import types
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from helion._compiler.flydsl import launcher
from helion._compiler.flydsl.backend import FlyDSLBackend
import helion.runtime

pytestmark = pytest.mark.cpu_only


class TestFlydslLauncher(unittest.TestCase):
    def test_compile_only_does_not_launch_or_cache(self) -> None:
        # flyc.compile returns None under FlyDSL's COMPILE_ONLY mode.
        fx = types.ModuleType("flydsl.expr")
        fx.Stream = Mock(return_value="stream")  # pyrefly: ignore[missing-attribute]
        fx.Tensor = object  # pyrefly: ignore[missing-attribute]
        flyc = types.ModuleType("flydsl.compiler")
        flyc.jit = lambda fn: fn  # pyrefly: ignore[missing-attribute]
        flyc.compile = Mock(return_value=None)  # pyrefly: ignore[missing-attribute]
        flydsl = types.ModuleType("flydsl")
        flydsl.expr = fx  # pyrefly: ignore[missing-attribute]
        flydsl.compiler = flyc  # pyrefly: ignore[missing-attribute]
        modules = {"flydsl": flydsl, "flydsl.expr": fx, "flydsl.compiler": flyc}
        kernel = Mock()
        with (
            patch.dict(sys.modules, modules),
            patch.dict(linecache.cache),
            patch.object(launcher, "_flydsl_jit_cache", {}),
            patch.object(launcher, "_flydsl_compiled_cache", {}) as compiled_cache,
            patch.object(launcher, "_flydsl_stream", None),
        ):
            for _ in range(2):
                self.assertIsNone(launcher.default_flydsl_launcher(kernel, (1,), 1, 2))
            self.assertEqual(compiled_cache, {})
        self.assertEqual(flyc.compile.call_count, 2)

    def test_old_runtime_import_path_forwards(self) -> None:
        # Code saved by v1.5.0-v1.5.2 imports the launcher from helion.runtime.
        with self.assertWarnsRegex(DeprecationWarning, "moved to"):
            old = helion.runtime.default_flydsl_launcher  # pyrefly: ignore
        self.assertIs(old, launcher.default_flydsl_launcher)
        with self.assertRaises(AttributeError):
            helion.runtime.no_such_name  # noqa: B018  # pyrefly: ignore

    def test_core_test_stand_ins_match_flydsl(self) -> None:
        # test_reduction_strategy and test_cute_reduction_chunk_seeds use
        # stand-in backends with these caps instead of importing FlyDSLBackend.
        backend = FlyDSLBackend()
        self.assertEqual(backend.max_reduction_threads(), 64)
        self.assertEqual(backend.max_reduction_loop(), 8192)


if __name__ == "__main__":
    unittest.main()
