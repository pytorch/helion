"""Helper script for bash crash recovery tests.

Run via:
    HELION_AUTOTUNE_CHECKPOINT_DIR=DIR \
        scripts/autotune_with_crash_recovery.sh -- python test/data/autotune_crash_helper.py

On first run (when _CRASH_ON_FIRST_BENCHMARK is set and no counter file
exists): patches do_bench to trigger a real CUDA illegal memory access,
which exercises the real _pending_config context manager and
TritonUnrecoverableRuntimeError code path.  On subsequent runs: autotuning
resumes from checkpoint normally, skipping the bad config.

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
    import triton
    import triton.language as tl

    import helion.autotuner.base_search as _bs

    @triton.jit
    def _ima_kernel(ptr):
        """Triton kernel that triggers illegal memory access."""
        bad_ptr = ptr + (1 << 40)
        tl.store(bad_ptr, tl.full([], 42.0, dtype=tl.float32))

    _original_do_bench = _bs.do_bench

    def _ima_do_bench(*args, **kwargs):  # type: ignore[no-untyped-def]
        counter_file.write_text("done")
        # Restore original so this only fires once
        _bs.do_bench = _original_do_bench
        # Trigger real CUDA illegal memory access
        x = torch.zeros(1, device="cuda")
        _ima_kernel[(1,)](x)
        torch.cuda.synchronize()
        # Should not reach here — IMA raises an exception
        return _original_do_bench(*args, **kwargs)

    _bs.do_bench = _ima_do_bench

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
