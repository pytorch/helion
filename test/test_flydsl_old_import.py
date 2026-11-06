"""Saved FlyDSL code imports its launcher from helion.runtime; FlyDSL now ships
as the helion-flydsl plugin, so the old name forwards there."""

from __future__ import annotations

import importlib.machinery
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pytest

import helion.runtime

pytestmark = pytest.mark.cpu_only


class TestOldLauncherImport(unittest.TestCase):
    def test_forwards_to_the_plugin_with_a_warning(self) -> None:
        plugin = types.ModuleType("helion_flydsl.launcher")
        plugin.default_flydsl_launcher = object()  # pyrefly: ignore[missing-attribute]
        modules = {
            "helion_flydsl": types.ModuleType("helion_flydsl"),
            "helion_flydsl.launcher": plugin,
        }
        with patch.dict(sys.modules, modules):
            with self.assertWarnsRegex(DeprecationWarning, "helion_flydsl.launcher"):
                old = helion.runtime.default_flydsl_launcher  # pyrefly: ignore
            self.assertIs(old, plugin.default_flydsl_launcher)  # pyrefly: ignore

    def test_without_the_plugin_says_how_to_install_it(self) -> None:
        # helion_flydsl cannot be found at all (the plain path finder, an empty
        # path). Saved code uses the from-import, so the message must survive it.
        with (
            tempfile.TemporaryDirectory() as empty,
            patch.dict(sys.modules),
            patch.object(sys, "path", [empty]),
            patch.object(sys, "meta_path", [importlib.machinery.PathFinder]),
        ):
            for name in [m for m in sys.modules if m.startswith("helion_flydsl")]:
                del sys.modules[name]
            with self.assertRaisesRegex(ImportError, "pip install helion-flydsl"):
                exec("from helion.runtime import default_flydsl_launcher")
            with self.assertRaisesRegex(ImportError, "pip install helion-flydsl"):
                helion.runtime.default_flydsl_launcher  # noqa: B018  # pyrefly: ignore

    def test_an_outdated_plugin_is_not_reported_as_missing(self) -> None:
        # Installed, but without the launcher module: not "pip install".
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "helion_flydsl"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            with patch.dict(sys.modules), patch.object(sys, "path", [tmp, *sys.path]):
                for name in [m for m in sys.modules if m.startswith("helion_flydsl")]:
                    del sys.modules[name]
                with self.assertRaises(ModuleNotFoundError) as ctx:
                    helion.runtime.default_flydsl_launcher  # noqa: B018  # pyrefly: ignore
        self.assertEqual(ctx.exception.name, "helion_flydsl.launcher")

    def test_a_broken_plugin_is_not_reported_as_missing(self) -> None:
        # Installed, but its own import fails: that error must come through.
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "helion_flydsl"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "launcher.py").write_text("import _hnot_installed_dependency\n")
            with patch.dict(sys.modules), patch.object(sys, "path", [tmp, *sys.path]):
                for name in [m for m in sys.modules if m.startswith("helion_flydsl")]:
                    del sys.modules[name]
                with self.assertRaises(ModuleNotFoundError) as ctx:
                    helion.runtime.default_flydsl_launcher  # noqa: B018  # pyrefly: ignore
        self.assertEqual(ctx.exception.name, "_hnot_installed_dependency")

    def test_other_names_still_raise(self) -> None:
        with self.assertRaises(AttributeError):
            helion.runtime.no_such_name  # noqa: B018  # pyrefly: ignore


if __name__ == "__main__":
    unittest.main()
