from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "script",
    ("compare_kda_decode_backends.py", "compare_kda_recurrent_backends.py"),
)
def test_checkout_benchmarks_precede_installed_regular_package(
    script: str, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[1]
    installed = tmp_path / "installed"
    foreign = installed / "benchmarks"
    foreign.mkdir(parents=True)
    (foreign / "__init__.py").write_text(
        'raise AssertionError("unrelated installed benchmarks imported")\n'
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(root), str(installed), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "cute" / script),
            "--help",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout
