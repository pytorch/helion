"""Helper script for crash recovery tests.

Run via:
    HELION_AUTOTUNE_CHECKPOINT_DIR=DIR \
        python -m helion.experimental.crash_recovery -- python test/data/autotune_crash_helper.py

On first run (when _CRASH_ON_FIRST_BENCHMARK or _CRASH_ON_FIRST_COMPILE is
set and no counter file exists): patches do_bench / compile_config to trigger
a hard crash, which exercises the pending_config sentinel and the crash
recovery script.  On subsequent runs: autotuning resumes from checkpoint
normally, skipping the crashed config.

Without the crash env vars: runs autotuning normally (used to test that the
crash recovery script passes through a successful run).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from helion._testing import DEVICE

checkpoint_dir = os.environ["HELION_AUTOTUNE_CHECKPOINT_DIR"]
crash_on_first_benchmark = os.environ.get("_CRASH_ON_FIRST_BENCHMARK", "")
crash_on_first_compile = os.environ.get("_CRASH_ON_FIRST_COMPILE", "")
counter_file = Path(checkpoint_dir) / "_crash_counter"

if crash_on_first_benchmark and not counter_file.exists():
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
        x = torch.zeros(1, device=DEVICE)
        _ima_kernel[(1,)](x)
        torch.cuda.synchronize()
        # Should not reach here — IMA raises an exception
        return _original_do_bench(*args, **kwargs)

    _bs.do_bench = _ima_do_bench

if crash_on_first_compile and not counter_file.exists():
    import triton
    import triton.language as tl

    import helion.autotuner.base_search as _bs

    @triton.jit
    def _ima_kernel_compile(ptr):
        """Triton kernel that triggers illegal memory access."""
        bad_ptr = ptr + (1 << 40)
        tl.store(bad_ptr, tl.full([], 42.0, dtype=tl.float32))

    # Wrap _benchmark so the real sentinel-writing code runs, but
    # compile_config triggers a real CUDA IMA on first call.
    # base_search._benchmark now detects unrecoverable errors and
    # preserves the sentinel instead of cleaning it up.
    _original_benchmark = _bs.BaseSearch._benchmark

    def _crashing_benchmark(self, configs, **kwargs):  # type: ignore[no-untyped-def]
        def _crash_compile(*args, **kw):  # type: ignore[no-untyped-def]
            counter_file.write_text("done")
            # Trigger real CUDA illegal memory access during compile
            x = torch.zeros(1, device=DEVICE)
            _ima_kernel_compile[(1,)](x)
            torch.cuda.synchronize()

        self.kernel.compile_config = _crash_compile
        return _original_benchmark(self, configs, **kwargs)

    _bs.BaseSearch._benchmark = _crashing_benchmark  # type: ignore[assignment]

# Import and run real autotuning
from helion._testing import import_path  # noqa: E402

datadir = Path(__file__).parent
basic_kernels = import_path(datadir / "basic_kernels.py")

args = (torch.randn([8, 32], device=DEVICE), torch.randn([8, 32], device=DEVICE))
bound = basic_kernels.add.bind(args)
bound.settings.autotune_checkpoint_dir = checkpoint_dir
bound.settings.autotune_effort = "quick"
config = bound.autotune(args, force=True)
result = bound(*args)
torch.testing.assert_close(result, args[0] + args[1])
