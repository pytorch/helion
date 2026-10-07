"""Flushed-launch stress of the FA4 statistics handoff (GPU only).

The softmax warpgroups hand the correction warps one statistic per KV step
(the rescale factor) plus the final row sum through a two-slot ring whose
"full" side is a 64-thread named barrier per warp pair and whose "empty" side
is an mbarrier per slot.  A named barrier counts arrivals: two ``bar.arrive``
from the same softmax warp complete it without the correction warp, whose
later ``bar.sync`` then waits for a phase nobody fills.  The MMA warp's
PV-before-next-QK order keeps every rescale publish behind the consumption
of the previous one, but the row sum has no score dependency, so before
``c8648353`` it could follow the last rescale publish while the correction
warp was still parked on the other warpgroup's barrier: an intermittent hang
whenever an L2 flush separated the launches (``do_bench`` style; back-to-back
launches never hung) and the GPU was shared with another process.  On the
tree before the fix the whole-row configuration below hung within the first
200 to 1,000 flushed launches in every run next to a matmul loop in a second
process (nine of nine implementer runs, eight of eight reviewer runs), but in
only one of twelve runs of 20,000 launches on an otherwise idle GPU.  The
test therefore keeps a second process busy with large matmuls while it runs.
After the fix, 100,000 flushed launches ran clean.

The test compiles the configuration in a fresh subprocess and launches it
``HELION_CUTE_FLASH_STRESS_LAUNCHES`` times (default 3,000; about 15 s on an
idle B200 including the compile) with a 256 MB flush between launches
under a watchdog; a hang kills the subprocess and fails the test.
Setting the variable marks a soak run: it also enables the XSA case on the
chunked body (the autotune candidate whose benchmark hung, which the fix
covers as well but which hung far more rarely before it), and
``HELION_CUTE_FLASH_STRESS_PROCESSES`` repeats each case in that many fresh
subprocesses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import skipIfFn
from helion._testing import skipIfNotCUDA

_SHAPE = (2, 16, 2048, 128)
_LAUNCHES_ENV = "HELION_CUTE_FLASH_STRESS_LAUNCHES"
_PROCESSES_ENV = "HELION_CUTE_FLASH_STRESS_PROCESSES"
_DEFAULT_LAUNCHES = 3_000
_DEFAULT_PROCESSES = 1
_BATCH = 200
_WATCHDOG_SECONDS = 60.0

# Plain attention on the whole-row FA4 body with the two-slot ring (a searched
# plain-attention configuration; the fused-row-epilogue guard does not apply).
_ATTENTION_WHOLE_ROW_RING2: dict[str, object] = {
    "block_sizes": [1, 128, 128],
    "cute_flash_pipeline_family": "fa4_2cta",
    "cute_flash_persistent": True,
    "cute_flash_kv_stage": 2,
    "cute_flash_s_stage": 2,
    "cute_flash_softmax_disc": False,
    "cute_flash_epi_tma": True,
    "cute_flash_stat_transport": "ring2",
    "cute_flash_e2e_schedule": "8/2",
}
# XSA on the chunked body, softmax-warpgroup row epilogue: the autotune
# candidate whose benchmark hung (a deep disc pipe lets the two warpgroups
# drift apart at the end of a work item).  Soak runs only: before the fix it
# hung in a minority of 15,000-launch runs, so a short run proves little.
_XSA_CHUNKED_SOFTMAX_ROUTE: dict[str, object] = {
    "block_sizes": [1, 128, 128],
    "cute_flash_pipeline_family": "fa4_2cta",
    "cute_flash_persistent": True,
    "cute_flash_kv_stage": 3,
    "cute_flash_s_stage": 2,
    "cute_flash_softmax_disc": True,
    "cute_flash_disc_pipe": 4,
    "cute_flash_e2e_schedule": "16/6",
    "cute_flash_e2e_offset0": 4,
    "cute_flash_exp2_packet": "deg2_16x6",
    "cute_flash_first_load_order": 4,
    "cute_flash_epi_tma": True,
    "cute_flash_row_epilogue_warps": "softmax",
}


def _stress(
    module: str, config: dict[str, object], launches: int, watchdog: float
) -> None:
    """Compile ``config`` and launch it ``launches`` times with an L2 flush in between."""
    import helion

    if module == "attention":
        from examples.attention import attention as example

        def call(fn: object) -> torch.Tensor:
            return fn(q, k, v)[0]  # pyrefly: ignore [not-callable]

        def reference() -> torch.Tensor:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    else:
        from examples.xsa import ref_xsa
        from examples.xsa import xsa_kernel as example

        def call(fn: object) -> torch.Tensor:
            return fn(q, k, v)  # pyrefly: ignore [not-callable]

        def reference() -> torch.Tensor:
            return ref_xsa(q, k, v)

    torch.manual_seed(0)
    q, k, v = (
        torch.randn(*_SHAPE, dtype=torch.bfloat16, device=DEVICE) for _ in range(3)
    )
    kernel = helion.kernel(
        example.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel.bind((q, k, v))
    normalized = bound._normalized_config_copy(helion.Config(**config))
    assert (
        normalized.config["cute_flash_softmax_disc"]
        is config["cute_flash_softmax_disc"]
    )
    fn = bound.compile_config(normalized)
    out = call(fn)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference(), atol=2e-2, rtol=2e-2)

    progress = {"time": time.time(), "launches": 0}

    def watch() -> None:
        while progress["launches"] < launches:
            time.sleep(0.5)
            if time.time() - progress["time"] > watchdog:
                sys.stdout.write(
                    f"HANG: no progress for {watchdog}s after "
                    f"{progress['launches']} launches\n"
                )
                sys.stdout.flush()
                os._exit(3)

    threading.Thread(target=watch, daemon=True).start()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE)
    while progress["launches"] < launches:
        for _ in range(_BATCH):
            flush.zero_()
            call(fn)
        torch.cuda.synchronize()
        progress["time"] = time.time()
        progress["launches"] += _BATCH
    sys.stdout.write(f"no hang in {progress['launches']} flushed launches\n")


# Prints ``busy`` once its first matmul has run, then keeps the GPU time
# sliced until killed (the deadline is a backstop).
_BUSY_NEIGHBOUR = """
import sys
import time
import torch
a = torch.randn(8192, 8192, dtype=torch.float16, device={device!r})
b = torch.randn_like(a)
c = torch.empty_like(a)
torch.mm(a, b, out=c)
torch.cuda.synchronize()
sys.stdout.write("busy\\n")
sys.stdout.flush()
deadline = time.time() + {seconds}
while time.time() < deadline:
    for _ in range(20):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
"""


def _soak_requested() -> bool:
    return _LAUNCHES_ENV in os.environ


def _run_in_subprocesses(module: str, config: dict[str, object]) -> None:
    import helion

    launches = int(os.environ.get(_LAUNCHES_ENV, str(_DEFAULT_LAUNCHES)))
    processes = int(os.environ.get(_PROCESSES_ENV, str(_DEFAULT_PROCESSES)))
    tree = os.path.dirname(os.path.dirname(os.path.abspath(helion.__file__)))
    env = {
        **os.environ,
        "HELION_BACKEND": "cute",
        "PYTHONPATH": os.pathsep.join(
            path for path in (tree, os.environ.get("PYTHONPATH", "")) if path
        ),
    }
    neighbour = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _BUSY_NEIGHBOUR.format(device=str(DEVICE), seconds=1800),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert neighbour.stdout is not None
        assert neighbour.stdout.readline().strip() == "busy", (
            "the busy neighbour process did not start"
        )
        for attempt in range(processes):
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--module",
                    module,
                    "--config",
                    json.dumps(config),
                    "--launches",
                    str(launches),
                    "--watchdog",
                    str(_WATCHDOG_SECONDS),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            assert proc.returncode == 0, (
                f"stress subprocess {attempt + 1}/{processes} exited with "
                f"{proc.returncode}\n--- stdout ---\n{proc.stdout[-3000:]}\n"
                f"--- stderr ---\n{proc.stderr[-3000:]}"
            )
            assert f"no hang in {launches} flushed launches" in proc.stdout
    finally:
        neighbour.kill()
        neighbour.wait()


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_whole_row_ring2_attention_survives_flushed_launches() -> None:
    _run_in_subprocesses("attention", _ATTENTION_WHOLE_ROW_RING2)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@skipIfFn(
    lambda: not _soak_requested(),
    reason=f"soak run only: set {_LAUNCHES_ENV} (the chunked-body hang was rare)",
)
def test_chunked_softmax_route_xsa_survives_flushed_launches() -> None:
    _run_in_subprocesses("xsa", _XSA_CHUNKED_SOFTMAX_ROUTE)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=("attention", "xsa"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--launches", type=int, default=_DEFAULT_LAUNCHES)
    parser.add_argument("--watchdog", type=float, default=_WATCHDOG_SECONDS)
    args = parser.parse_args()
    _stress(args.module, json.loads(args.config), args.launches, args.watchdog)


if __name__ == "__main__":
    _main()
