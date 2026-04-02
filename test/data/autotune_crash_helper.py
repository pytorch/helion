"""Helper script for bash crash recovery tests.

Run via:
    scripts/autotune_with_crash_recovery.sh --checkpoint-dir DIR -- python test/data/autotune_crash_helper.py

On first run (when _CRASH_ON_FIRST_BENCHMARK is set and no counter file
exists): patches benchmark_function to crash after writing the pending file
via the real code path. On subsequent runs: autotuning resumes from
checkpoint normally, skipping the bad config.

Without _CRASH_ON_FIRST_BENCHMARK: runs autotuning normally (used to test
that the bash script passes through a successful run).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

checkpoint_dir = os.environ["HELION_AUTOTUNE_CHECKPOINT_DIR"]
crash_on_first = os.environ.get("_CRASH_ON_FIRST_BENCHMARK", "")
counter_file = Path(checkpoint_dir) / "_benchmark_counter"

if crash_on_first and not counter_file.exists():
    from helion.autotuner.base_search import BaseSearch

    def _crashing_benchmark(self, config, fn):  # type: ignore[no-untyped-def]
        counter_file.write_text("done")
        # Write pending file via the real code path
        self._write_pending_config(str(config))
        # Crash without clearing the pending file — simulates CUDA kill
        os._exit(1)

    BaseSearch.benchmark_function = _crashing_benchmark  # type: ignore[assignment]

# Import and run real autotuning
from helion._testing import import_path  # noqa: E402

datadir = Path(__file__).parent
basic_kernels = import_path(datadir / "basic_kernels.py")

args = (torch.randn([8, 32], device="cuda"), torch.randn([8, 32], device="cuda"))
bound = basic_kernels.add.bind(args)
bound.settings.autotune_checkpoint_dir = checkpoint_dir
bound.settings.autotune_effort = "quick"
config = bound.autotune(args, force=True)
result = bound(*args)
torch.testing.assert_close(result, args[0] + args[1])
