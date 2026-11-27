"""Launch stress of the per-key attention schedule at two CTAs per SM (GPU only).

In the per-key paged attention pipeline compute warp 0 issues the PV MMA. The
converters stage the next V as soon as that MMA commits (``value_free``),
which needs warp 0 alone, so a value-ready barrier that the other compute
warps also wait on can complete its next phase before one of them has
observed the previous one. A parity wait cannot tell a phase from the one two
later: that warp then waits for a phase that needs the softmax's 128-thread
barrier, which needs that warp, and one stuck CTA hangs the launch. Only the
issuing warp may therefore wait there, which
``test_cute_paged_attention_handoff.py`` pins in the templates. This test
launches the configuration that hangs when all four compute warps wait.

The hang needs this kernel at this occupancy: 256 CTAs of the windowed
decode-step fixture at two per SM (``cute_min_blocks_per_mp=2``). One CTA per
SM, full attention, a five-token query, an L2 flush kernel between launches
and the other paged fixtures (reverse page order, post-scaled operands) do
not show it, and which property of the generated code opens the window is
not known, so the test keeps the configuration that reproduces.

The test compiles the configuration in a fresh subprocess and launches it
``HELION_CUTE_PAGED_STRESS_LAUNCHES`` times (default 30,000), with a
synchronize after every third launch and after the last, under a no-progress
watchdog that is armed before the first call; a hang kills the subprocess
and fails the test. The first call also JIT-compiles the kernel, so it has a
longer bound of its own: a slow compile is not reported as a hang, a first
launch that never returns still is. K and V are zero, so the only numerical
check is that every output element was stored as zero.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_attention_handoff import LaunchWatchdog
from helion._testing.cute_paged_attention_handoff import paged_decode_args
from helion._testing.cute_paged_attention_handoff import paged_decode_attention
from helion._testing.cute_paged_attention_handoff import per_key_two_cta_config

# The test compiles and launches in a fresh subprocess under a watchdog.
pytestmark = pytest.mark.slow

_LAUNCHES_ENV = "HELION_CUTE_PAGED_STRESS_LAUNCHES"
_DEFAULT_LAUNCHES = 30_000
# Launches between two synchronizes.
_BURST = 3
_WATCHDOG_SECONDS = 30.0
# The first call compiles before it launches. A cold compile cache on a
# loaded machine can take longer than a launch may stall, so that call is
# bounded on its own (``LaunchWatchdog``).
_FIRST_CALL_SECONDS = 300.0
# The subprocess bound covers what the watchdog does not see (imports,
# inputs, bind and codegen) and allows each launch many times its duration.
_STARTUP_SECONDS = 90.0
_SECONDS_PER_LAUNCH = 0.001
# Two heads per request: 256 CTAs of the per-key schedule.
_BATCH = 128
_QUERY_LEN = 4


def _stress(launches: int, stall: float, first_call: float) -> None:
    """Compile the schedule and launch it exactly ``launches`` times."""
    args = paged_decode_args(_BATCH, _QUERY_LEN, DEVICE)
    out = args[7]
    assert isinstance(out, torch.Tensor)
    bound = paged_decode_attention.bind(args)
    with bound.env:
        config = bound.config_spec.normalized_config(per_key_two_cta_config())
    assert config.config["cute_mma_dag_pass_warps"] == "attention"
    assert config.config["cute_min_blocks_per_mp"] == 2
    assert "paged_attention_partial_body(" in bound.to_code(config)
    fn = bound.compile_config(config)
    out.fill_(float("nan"))

    watchdog = LaunchWatchdog(first_call=first_call, stall=stall)

    def watch() -> None:
        while watchdog.launches < launches:
            time.sleep(0.5)
            report = watchdog.expired()
            if report is not None:
                sys.stdout.write(report + "\n")
                sys.stdout.flush()
                os._exit(3)

    # Armed before the first call, which JIT-compiles and then launches: a
    # hang of the very first launch is reported too, under that call's bound.
    threading.Thread(target=watch, daemon=True).start()
    started = time.monotonic()
    fn(*args)
    torch.cuda.synchronize()
    watchdog.progress(0)
    sys.stdout.write(f"first call took {time.monotonic() - started:.1f}s\n")
    assert bool((out == 0).all().item())
    for done in range(1, launches + 1):
        fn(*args)
        if done % _BURST == 0 or done == launches:
            torch.cuda.synchronize()
            watchdog.progress(done)
    assert bool((out == 0).all().item())
    sys.stdout.write(f"no hang in {watchdog.launches} launches\n")


@skipIfNotCUDA()
def test_per_key_schedule_survives_repeated_launches() -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    if torch.cuda.get_device_properties(DEVICE).multi_processor_count >= 2 * _BATCH:
        pytest.skip("the grid would leave at most one CTA per SM")
    launches = int(os.environ.get(_LAUNCHES_ENV, str(_DEFAULT_LAUNCHES)))
    tree = os.path.dirname(os.path.dirname(os.path.abspath(helion.__file__)))
    env = {
        **os.environ,
        "HELION_BACKEND": "cute",
        "PYTHONPATH": os.pathsep.join(
            path for path in (tree, os.environ.get("PYTHONPATH", "")) if path
        ),
    }
    proc = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--launches",
            str(launches),
            "--watchdog",
            str(_WATCHDOG_SECONDS),
            "--first-call",
            str(_FIRST_CALL_SECONDS),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=(
            _STARTUP_SECONDS + _FIRST_CALL_SECONDS + launches * _SECONDS_PER_LAUNCH
        ),
        check=False,
    )
    assert proc.returncode == 0, (
        f"stress subprocess exited with {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr ---\n{proc.stderr[-3000:]}"
    )
    # The subprocess reports the launches it made: exactly the count asked
    # for, also when that is not a multiple of the burst.
    assert f"no hang in {launches} launches" in proc.stdout


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launches", type=int, default=_DEFAULT_LAUNCHES)
    parser.add_argument("--watchdog", type=float, default=_WATCHDOG_SECONDS)
    parser.add_argument("--first-call", type=float, default=_FIRST_CALL_SECONDS)
    args = parser.parse_args()
    _stress(args.launches, args.watchdog, args.first_call)


if __name__ == "__main__":
    _main()
