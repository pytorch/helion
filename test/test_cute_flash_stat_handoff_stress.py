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

The flex case runs the FlexAttention envelope's BlockMask list walk
(``examples/flex_attention.py`` on the fa4 pair body) over random partial /
full lists: counts from zero to every tile, rows without keys, an empty pair
and an asymmetric pair, so every role's per-item trip count comes from the
tables and the zero-P path is exercised; its reference is block-level
visibility (the noop mask_mod keeps every key of a listed block).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._compiler.cute import cute_flash
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import skipIfFn
from helion._testing import skipIfNotCUDA

_SHAPE = (2, 16, 2048, 128)
# The flex list walk: 16 query rows x 16 KV tiles per (batch, head).
_FLEX_SHAPE = (2, 16, 2048, 64)
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


# The flex list walk on the chunked fa4 body with the two-slot ring (the
# pruned-fa4 campaign structure the flex GPU suite pins as well).
_FLEX_LIST_WALK: dict[str, object] = {
    "block_sizes": [128, 128],
    "cute_flash_pipeline_family": "fa4",
    "cute_flash_exp2_packet": "8x2",
    "cute_flash_e2e_schedule": "16/8",
    "cute_flash_s_load_rep": 16,
    "cute_flash_p_store_rep": 16,
    "cute_flash_disc_pipe": 2,
    "cute_flash_split_p_arrive": True,
    "cute_flash_causal_kv_order": "descending",
    "cute_flash_epi_tma": True,
}


def _random_block_mask(batch: int, heads: int, rows: int, tiles: int) -> object:
    """Random partial / full lists per (batch, head, row): a total from zero
    to every tile, a random partial / full split, plus an empty pair (rows
    0 and 1 of head 0) and an asymmetric pair (row 2 empty, row 3 full)."""
    from torch.nn.attention.flex_attention import BlockMask
    from torch.nn.attention.flex_attention import noop_mask

    generator = torch.Generator().manual_seed(1)
    part_cnt = torch.zeros(batch, heads, rows, dtype=torch.int32)
    full_cnt = torch.zeros(batch, heads, rows, dtype=torch.int32)
    part_idx = torch.zeros(batch, heads, rows, tiles, dtype=torch.int32)
    full_idx = torch.zeros(batch, heads, rows, tiles, dtype=torch.int32)
    for b in range(batch):
        for h in range(heads):
            for r in range(rows):
                total = int(torch.randint(0, tiles + 1, (1,), generator=generator))
                n_part = int(torch.randint(0, total + 1, (1,), generator=generator))
                perm = torch.randperm(tiles, generator=generator)
                part = perm[:n_part].sort().values
                full = perm[n_part:total].sort().values
                part_cnt[b, h, r] = n_part
                part_idx[b, h, r, :n_part] = part.to(torch.int32)
                full_cnt[b, h, r] = total - n_part
                full_idx[b, h, r, : total - n_part] = full.to(torch.int32)
    for r in (0, 1, 2):
        part_cnt[0, 0, r] = 0
        full_cnt[0, 0, r] = 0
    part_cnt[0, 0, 3] = 0
    full_cnt[0, 0, 3] = tiles
    full_idx[0, 0, 3] = torch.arange(tiles, dtype=torch.int32)
    return BlockMask.from_kv_blocks(
        part_cnt.to(DEVICE),
        part_idx.to(DEVICE),
        full_cnt.to(DEVICE),
        full_idx.to(DEVICE),
        BLOCK_SIZE=128,
        mask_mod=noop_mask,
    )


def _block_visibility_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: object,
    scale: float,
) -> torch.Tensor:
    """Softmax over the keys of the listed (partial | full) blocks of each
    query row's block; a row without a listed block gives 0."""
    visible = block_mask.to_dense().bool()  # pyrefly: ignore [missing-attribute]
    rows, tiles = visible.shape[-2:]
    mask = visible.repeat_interleave(q.size(2) // rows, dim=2)
    mask = mask.repeat_interleave(k.size(2) // tiles, dim=3)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    p = torch.where(mask.any(dim=-1, keepdim=True), p, torch.zeros_like(p))
    return torch.matmul(p, v.float()).to(q.dtype)


def _stress(
    module: str, config: dict[str, object], launches: int, watchdog: float
) -> None:
    """Compile ``config`` and launch it ``launches`` times with an L2 flush in between."""
    import helion

    torch.manual_seed(0)
    shape = _FLEX_SHAPE if module == "flex" else _SHAPE
    q, k, v = (
        torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE) for _ in range(3)
    )
    bind_args: tuple[object, ...] = (q, k, v)
    if module == "attention":
        from examples.attention import attention as example

        def call(fn: object) -> torch.Tensor:
            return fn(q, k, v)[0]  # pyrefly: ignore [not-callable]

        def reference() -> torch.Tensor:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    elif module == "flex":
        from examples.flex_attention import _identity
        from examples.flex_attention import helion_flex_attention_kernel as example

        batch, heads, seq, head_dim = shape
        block_mask = _random_block_mask(batch, heads, seq // 128, seq // 128)
        scale = 1.0 / math.sqrt(head_dim)
        bind_args = (
            q,
            k,
            v,
            _identity,
            block_mask.kv_num_blocks,  # pyrefly: ignore [missing-attribute]
            block_mask.kv_indices,  # pyrefly: ignore [missing-attribute]
            block_mask.full_kv_num_blocks,  # pyrefly: ignore [missing-attribute]
            block_mask.full_kv_indices,  # pyrefly: ignore [missing-attribute]
            block_mask.mask_mod,  # pyrefly: ignore [missing-attribute]
            128,
            128,
            scale,
            False,
        )

        def call(fn: object) -> torch.Tensor:
            return fn(*bind_args)[0]  # pyrefly: ignore [not-callable]

        def reference() -> torch.Tensor:
            return _block_visibility_reference(q, k, v, block_mask, scale)
    else:
        from examples.xsa import ref_xsa
        from examples.xsa import xsa_kernel as example

        def call(fn: object) -> torch.Tensor:
            return fn(q, k, v)  # pyrefly: ignore [not-callable]

        def reference() -> torch.Tensor:
            return ref_xsa(q, k, v)

    # The polynomial exp2 packets (the XSA route's ``deg2_16x6``) are
    # approximations and reach the surface only under the fast_math setting.
    kernel = helion.kernel(
        example.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        fast_math=config.get("cute_flash_exp2_packet")
        in cute_flash._FLASH_FAST_MATH_EXP2_PACKETS,
    )
    bound = kernel.bind(bind_args)
    normalized = bound._normalized_config_copy(helion.Config(**config))
    if "cute_flash_softmax_disc" in config:
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
def test_flex_list_walk_survives_flushed_launches() -> None:
    """The BlockMask list walk over random tables (zero counts, an empty
    pair, an asymmetric pair) on the chunked fa4 body with the two-slot
    ring: the per-item trip count and the zero-P path under flushed launches
    next to a busy neighbour."""
    _run_in_subprocesses("flex", _FLEX_LIST_WALK)


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
    parser.add_argument("--module", choices=("attention", "flex", "xsa"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--launches", type=int, default=_DEFAULT_LAUNCHES)
    parser.add_argument("--watchdog", type=float, default=_WATCHDOG_SECONDS)
    args = parser.parse_args()
    _stress(args.module, json.loads(args.config), args.launches, args.watchdog)


if __name__ == "__main__":
    _main()
