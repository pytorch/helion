"""Deployable Modal app for helion autotuner.

Deploy once:
    modal deploy helion/autotuner/_modal_app.py

Then all ModalSearch calls use the already-warm containers — no cold start.

To undeploy:
    modal app stop helion-autotuner
"""

from __future__ import annotations

import os
import sys

# Ensure helion is importable when modal runs this file directly
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_this_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import modal

from helion.autotuner import _modal_worker

APP_NAME = "helion-autotuner"

_helion_pkg = os.path.dirname(_this_dir)
_py_version = "3.12"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04",
        add_python=_py_version,
    )
    .entrypoint([])
    .pip_install(
        "torch",
        "triton",
        "packaging",
        "numpy",
        "rich",
        "psutil",
        "scikit-learn",
        "filecheck",
        "typing_extensions",
    )
    .add_local_dir(
        _helion_pkg,
        remote_path=f"/usr/local/lib/python{_py_version}/site-packages/helion",
        copy=True,
    )
)

benchmark_config = app.function(
    image=image,
    gpu="H100",
    timeout=600,
    max_containers=10,
)(_modal_worker.benchmark_config)
