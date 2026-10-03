from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_cpu_codegen_does_not_import_experimental_device_primitives() -> None:
    pytest.importorskip("cutlass")
    # A fresh interpreter prevents another device test's imports from hiding
    # an optional SDK dependency in the host-side planner path.
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root))
    environment.pop("HELION_BACKEND", None)
    environment.pop("HELION_AUTOTUNE_EFFORT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            textwrap.dedent(
                """
                import builtins
                import cutlass

                # New SDKs eagerly expose experimental modules while importing
                # cutlass itself. Block Helion's imports after SDK bootstrap;
                # intercept cached imports too, so SDK version cannot hide them.
                original_import = builtins.__import__
                def import_without_experimental(name, *args, **kwargs):
                    if name == "cutlass.experimental" or name.startswith(
                        "cutlass.experimental."
                    ):
                        raise ModuleNotFoundError(
                            "CPU codegen imported " + name, name=name
                        )
                    return original_import(name, *args, **kwargs)

                builtins.__import__ = import_without_experimental

                from test._cute_aux import _config, _rank_two_code, _rank_two_inputs
                import torch

                code = _rank_two_code(_rank_two_inputs(), _config())
                assert "@cute.kernel" in code
                assert not torch.cuda.is_initialized()
                """
            ),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
