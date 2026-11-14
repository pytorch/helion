"""Run Helion-free generated code (``to_code(allow_helion_deps=False)``)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

from . import DEVICE


def run_add_without_helion(code: str, entrypoint: str, shape: tuple[int, int]) -> None:
    """Run an **add** standalone in a subprocess where importing ``helion`` is
    blocked, proving zero Helion runtime dependency and the right result. Specific
    to add kernels -- it hardcodes the ``x + y`` reference."""
    m, n = shape
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "generated.py"
        path.write_text(code)
        script = textwrap.dedent(f"""
            import sys
            # Poison the helion package so any leftover dependency fails loudly.
            sys.modules["helion"] = None
            import importlib.util
            import torch

            spec = importlib.util.spec_from_file_location("_standalone", {str(path)!r})
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            torch.manual_seed(0)
            x = torch.randn([{m}, {n}], device="{DEVICE}", dtype=torch.bfloat16)
            y = torch.randn([{m}, {n}], device="{DEVICE}", dtype=torch.bfloat16)
            out = mod.{entrypoint}(x, y)
            torch.testing.assert_close(out, x + y)
            print("STANDALONE_OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
    if "STANDALONE_OK" not in result.stdout:
        raise AssertionError(
            f"standalone run failed:\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
