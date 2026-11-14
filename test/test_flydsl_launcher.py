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
import warnings

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

    def test_runtime_name_is_the_launcher(self) -> None:
        # Generated and saved FlyDSL code imports the launcher from
        # helion.runtime; the name stays stable (no warning) while the code moves.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            old = helion.runtime.default_flydsl_launcher
        self.assertIs(old, launcher.default_flydsl_launcher)
        with self.assertRaises(AttributeError):
            helion.runtime.no_such_name  # noqa: B018  # pyrefly: ignore

    def test_generated_code_imports_the_stable_name(self) -> None:
        # Not the module path, which moves when FlyDSL leaves the tree.
        imports = FlyDSLBackend().library_imports
        self.assertEqual(
            imports["_default_flydsl_launcher"],
            "from helion.runtime import default_flydsl_launcher"
            " as _default_flydsl_launcher",
        )

    def test_core_test_stand_ins_match_flydsl(self) -> None:
        # test_reduction_strategy and test_cute_reduction_chunk_seeds use
        # stand-in backends with these caps instead of importing FlyDSLBackend.
        backend = FlyDSLBackend()
        self.assertEqual(backend.max_reduction_threads(), 64)
        self.assertEqual(backend.max_reduction_loop(), 8192)


if __name__ == "__main__":
    unittest.main()
