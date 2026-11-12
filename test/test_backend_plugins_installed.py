"""Tests that install the reference plugin (test/data/helion_dummy_backend)
and run fresh interpreters, so they are slow."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import pytest

import helion
from helion._testing import onlyBackends

_REPO_ROOT = Path(helion.__file__).resolve().parent.parent
_FIXTURE = _REPO_ROOT / "test" / "data" / "helion_dummy_backend"


def _run_with_fixture(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    full_env.pop("HELION_DISABLE_BACKEND_PLUGINS", None)
    full_env["PYTHONPATH"] = os.pathsep.join(
        [str(_FIXTURE), str(_REPO_ROOT), full_env.get("PYTHONPATH", "")]
    )
    full_env.update(env)
    # A real file, since Helion reads kernel source with inspect.getsource.
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "script.py"
        script.write_text(textwrap.dedent(code))
        return subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            env=full_env,
            timeout=300,
        )


@pytest.mark.slow
@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestInstalledPlugin(unittest.TestCase):
    # The fixture is a package plus a .dist-info on PYTHONPATH, which
    # importlib.metadata discovers like a pip-installed plugin.
    def test_not_imported_until_requested(self) -> None:
        result = _run_with_fixture(
            """
            import sys
            import helion
            from helion._compiler.backend_registry import list_backends

            assert "triton_oot" in list_backends(), list_backends()
            assert "helion_dummy_backend.backend" not in sys.modules
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plugin_imported_before_helion(self) -> None:
        result = _run_with_fixture(
            """
            import helion_dummy_backend.backend
            import helion
            from helion._compiler.backend_registry import get_backend_class

            assert get_backend_class("triton_oot").__name__ == "TritonOOTBackend"
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selected_by_env(self) -> None:
        result = _run_with_fixture(
            """
            import helion
            assert helion.Settings().backend == "triton_oot"
            """,
            HELION_BACKEND="triton_oot",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_spawned_worker_loads_plugin(self) -> None:
        result = _run_with_fixture(
            """
            import multiprocessing
            import sys

            def lookup():
                from helion._compiler.backend_registry import get_backend_class

                name = get_backend_class("triton_oot").__name__
                return name, "helion_dummy_backend.backend" in sys.modules

            if __name__ == "__main__":
                with multiprocessing.get_context("spawn").Pool(1) as pool:
                    assert pool.apply(lookup) == ("TritonOOTBackend", True)
                # The worker loaded it itself; this process never looked it up.
                assert "helion_dummy_backend.backend" not in sys.modules
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@pytest.mark.slow
@onlyBackends(["triton"])
class TestInstalledPluginCompile(unittest.TestCase):
    def test_compiles_and_runs(self) -> None:
        result = _run_with_fixture(
            """
            import torch
            import helion
            import helion.language as hl
            from helion._testing import DEVICE

            @helion.kernel(backend="triton_oot")
            def add(x, y):
                out = torch.empty_like(x)
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] + y[tile]
                return out

            x = torch.randn(64, 64, device=DEVICE)
            bound = add.bind((x, x))
            out = bound.compile_config(bound.config_spec.default_config())(x, x)
            torch.testing.assert_close(out, x + x)
            """,
            HELION_BACKEND="triton",
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
