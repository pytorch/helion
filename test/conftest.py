from __future__ import annotations

import glob
import os
from pathlib import Path
import warnings


def pytest_sessionfinish(session: object, exitstatus: object) -> None:
    # TEMPORARY DEBUG (DO NOT MERGE): surface the per-worker softmax crash logs
    # written by test/test_autodiff.py so they appear in CI output even when an
    # xdist worker dies without returning its captured output. Remove with the
    # debug block in test_autodiff.py once the flaky crash is root-caused.
    debug_dir = os.environ.get("HELION_SOFTMAX_DEBUG_DIR", "/tmp")
    for pattern in ("helion_softmax_debug_*.log", "helion_softmax_fault_*.log"):
        for path in sorted(glob.glob(os.path.join(debug_dir, pattern))):
            try:
                data = Path(path).read_text().strip()
            except OSError:
                continue
            if data:
                print(
                    f"\n===== {path} =====\n{data}\n===== end {path} =====", flush=True
                )


def pytest_configure() -> None:
    # The final-verification rebench re-times the top configs for a full 5s each
    # by default (HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS=5000), which alone
    # pushes the autotuner tests well past the suite's 60s per-test timeout and
    # gets their xdist worker killed. Clamp it to the floor (200ms) so the step
    # still runs (and stays covered) without dominating the test runtime.
    os.environ.setdefault("HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS", "200")

    # TODO(tcombes): remove this once Pallas RNG generation avoids int64.
    # JAX x64 is disabled on TPU, so RNG-generated int64s are truncated and
    # spam Pallas test logs with one warning per generated statement.
    warnings.filterwarnings(
        "ignore",
        message=(
            "Explicitly requested dtype int64 requested in .* is not available, "
            "and will be truncated to dtype int32.*"
        ),
        category=UserWarning,
    )
