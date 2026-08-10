"""Physical TorchTPU validation for Helion remote copies."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(
    os.environ.get("HELION_TEST_TORCH_TPU_MULTIPROCESS") == "1"
    and os.environ.get("HELION_BACKEND") == "pallas"
    and importlib.util.find_spec("torch_tpu") is not None,
    "requires a dedicated TorchTPU multiprocess test invocation",
)
class TestRemoteCopyTorchTpuRuntime(unittest.TestCase):
    def test_one_process_per_device_peer_exchange(self) -> None:
        env = os.environ.copy()
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        runner = Path(__file__).with_name("remote_copy_torch_tpu_runner.py")
        subprocess.run(
            [sys.executable, str(runner)],
            check=True,
            env=env,
            timeout=180,
        )
