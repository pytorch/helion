"""Batched tcgen05 schedules under both ``tcgen05_batch_raster`` walks (GPU only).

``tcgen05_batch_raster=batch_slowest`` moves a batched GEMM's batch axis to
the scheduler's slowest dim (``_tcgen05_scheduler_axes``).  Two pieces of the
persistent machinery decode scheduler coordinates on their own, outside the
monolithic role loops, and each used to assume the default dim order:

- the L2-swizzle padding guard decided whether padding exists from the PID
  axis the *default* order puts on dim 1, so an even M-tile count skipped the
  guard while dim 1 (N under ``batch_slowest``) was padded: the padding slots
  ran as work and the kernel never finished;
- the scheduler-warp mailbox (``tcgen05_strategy=role_local_with_scheduler``)
  publishes ``tile_idx`` slot for slot, and its consumers linearized the slots
  in PID order, reading (m slots, n, batch) as (batch, m, n): a hang under
  ``batch_slowest`` and wrong results for ``cluster_n=2`` under the default
  walk, whose dim 1 carries N as well.

The render tests pin the guard and the mailbox decode.  The run tests compile
and launch each case in a fresh subprocess under a timeout, so a regression
fails the test instead of hanging the suite; every case is checked against
``torch.bmm`` and the cases of one test are compared bit for bit, since the
raster, the cluster shape and the scheduler strategy only reorder the tiles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import helion
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl

get_cute_mma_support = importlib.import_module(
    "helion._compiler.cute.mma_support"
).get_cute_mma_support

_TIMEOUT_ENV = "HELION_CUTE_BATCH_RASTER_TIMEOUT"
_DEFAULT_TIMEOUT_SECONDS = 900.0

# 6 batches of 2 x 3 output tiles (bm=256) or 4 x 3 (bm=128): an even M-tile
# count and an odd N-tile count, which a swizzle of 2 pads on dim 1.
_ODD_N_SHAPE = (6, 512, 256, 768)
# 12 batches of 2 x 4 tiles: 48 CtaGroup.TWO pair tiles, more than a B200
# co-schedules as four-CTA clusters (33), so the persistent clusters loop.
_LOOPING_SHAPE = (12, 512, 256, 1024)
# The odd-N tile grid with enough batches (26 x 4 x 3 = 312 CTAs, two-CTA pairs
# counted as two) to exceed one wave of SMs: the persistent raster, whose
# swizzle padding the guard exists for.  One wave of tiles takes the one-shot
# scheduler, which drops the swizzle and needs no guard.
_ODD_N_LOOPING_SHAPE = (26, 512, 256, 768)

_TWO_CTA: dict[str, object] = {
    "block_sizes": [1, 256, 256, 64],
    "pid_type": "persistent_interleaved",
    "tcgen05_cluster_m": 2,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 4,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 2,
    "l2_groupings": [1],
}
_ONE_CTA: dict[str, object] = {
    **_TWO_CTA,
    "block_sizes": [1, 128, 256, 64],
    "tcgen05_cluster_m": 1,
}
_WITH_SCHEDULER: dict[str, object] = {
    "tcgen05_strategy": "role_local_with_scheduler",
    "tcgen05_warp_spec_scheduler_warps": 1,
}
_RASTERS = ("batch_fastest", "batch_slowest")


@helion.kernel(backend="cute")
def batched_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc,
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def batched_matmul_rowvec_bias(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # The trailing-axis bias classifies as the rowvec aux form and is fused
    # into the tcgen05 epilogue (the C-input warp reads the mailbox too).
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc,
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
            )
        out[tile_b, tile_m, tile_n] = (acc + bias[tile_n]).to(torch.bfloat16)
    return out


_KERNELS = {
    "batched_matmul": batched_matmul,
    "batched_matmul_rowvec_bias": batched_matmul_rowvec_bias,
}


def _skip_without_tcgen05() -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")


def _inputs(
    kernel_name: str, shape: tuple[int, int, int, int], device: torch.device | str
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Seeded inputs and the fp32 reference of ``kernel_name`` on ``shape``."""
    torch.manual_seed(0)
    b, m, k, n = shape
    x = torch.randn(b, m, k, device=device, dtype=torch.float16)
    y = torch.randn(b, k, n, device=device, dtype=torch.float16)
    reference = torch.bmm(x.float(), y.float())
    if kernel_name == "batched_matmul_rowvec_bias":
        bias = torch.randn(n, device=device, dtype=torch.bfloat16)
        return (x, y, bias), reference + bias.float()
    return (x, y), reference


def _render(
    kernel_name: str, shape: tuple[int, int, int, int], **config: object
) -> str:
    args, _reference = _inputs(kernel_name, shape, DEVICE)
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        return _KERNELS[kernel_name].bind(args).to_triton_code(helion.Config(**config))


def _guard_lines(code: str) -> list[str]:
    return [line.strip() for line in code.splitlines() if "tile_idx[1] <" in line]


def _virtual_pid_lines(code: str) -> set[str]:
    return {
        line.strip()
        for line in code.splitlines()
        if line.strip().startswith("virtual_pid = ")
    }


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_padding_guard_follows_the_scheduler_dims() -> None:
    """The swizzle guard is decided by the tile count on scheduler dim 1."""
    _skip_without_tcgen05()
    n_tiles = "(768 + _BLOCK_SIZE_2 - 1) // _BLOCK_SIZE_2"
    for grid in (_TWO_CTA, _ONE_CTA):
        slowest = _render(
            "batched_matmul",
            _ODD_N_LOOPING_SHAPE,
            **grid,
            tcgen05_batch_raster="batch_slowest",
            tcgen05_l2_swizzle_size=2,
        )
        # Dim 1 carries the three N tiles, padded to four: every role loop
        # admits only the logical ones.
        assert (
            f"{n_tiles}, 26), ({grid['tcgen05_cluster_m']}, 1, 1), swizzle_size=2"
            in slowest
        )
        guards = _guard_lines(slowest)
        assert len(guards) == 3, guards
        assert all(guard.endswith(f"tile_idx[1] < {n_tiles}:") for guard in guards)
        fastest = _render(
            "batched_matmul",
            _ODD_N_LOOPING_SHAPE,
            **grid,
            tcgen05_batch_raster="batch_fastest",
            tcgen05_l2_swizzle_size=2,
        )
        # Dim 1 carries the M tiles, a multiple of the swizzle: no padding.
        assert "swizzle_size=2" in fastest
        assert _guard_lines(fastest) == []


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_scheduler_mailbox_decodes_the_raster() -> None:
    """Mailbox consumers read each PID axis from the slot of its scheduler dim."""
    _skip_without_tcgen05()
    m_tiles = "((512 + _BLOCK_SIZE_1 - 1) // _BLOCK_SIZE_1)"

    def slot(index: int) -> str:
        return f"tcgen05_work_tile_smem[cutlass.Int32({index})]"

    for cluster_n in (1, 2):
        # batch_slowest: dims (m slots, n, batch) for either cluster shape.
        config = {
            **_TWO_CTA,
            **_WITH_SCHEDULER,
            "tcgen05_cluster_n": cluster_n,
            "tcgen05_batch_raster": "batch_slowest",
        }
        code = _render("batched_matmul", _LOOPING_SHAPE, **config)
        expected = (
            f"virtual_pid = {slot(2)} + {slot(0)} // cutlass.Int32(2) * 12 "
            f"+ {slot(1)} * (12 * {m_tiles})"
        )
        assert _virtual_pid_lines(code) == {expected}
        config["tcgen05_batch_raster"] = "batch_fastest"
        code = _render("batched_matmul", _LOOPING_SHAPE, **config)
        # batch_fastest: dims (batch slots, m, n), or (batch slots, n, m)
        # when cluster_n=2 pairs the N tiles on dim 1.
        m_slot, n_slot = (1, 2) if cluster_n == 1 else (2, 1)
        expected = (
            f"virtual_pid = {slot(0)} // cutlass.Int32(2) + {slot(m_slot)} * 12 "
            f"+ {slot(n_slot)} * (12 * {m_tiles})"
        )
        assert _virtual_pid_lines(code) == {expected}


def _digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _run_case(case: dict[str, object]) -> dict[str, object]:
    """Compile ``case`` on the current device, run it twice and check it."""
    kernel_name = str(case["kernel"])
    shape = tuple(int(v) for v in case["shape"])  # pyrefly: ignore [not-iterable]
    assert len(shape) == 4
    args, reference = _inputs(kernel_name, shape, "cuda")
    config = helion.Config(**case["config"])  # pyrefly: ignore [bad-argument-type]
    fn = _KERNELS[kernel_name].bind(args).compile_config(config)
    out = fn(*args)
    torch.cuda.synchronize()
    again = fn(*args)
    torch.cuda.synchronize()
    if out.dtype == torch.bfloat16:
        # One bf16 ulp at the output's magnitude against the fp32 reference.
        tolerance = {"rtol": 1e-2, "atol": 0.5}
    else:
        tolerance = {"rtol": 1e-3, "atol": 1e-2}
    torch.testing.assert_close(out.float(), reference, **tolerance)
    assert torch.equal(out, again)
    return {"digest": _digest(out)}


def _run_cases(cases: dict[str, dict[str, object]]) -> dict[str, str]:
    """Run every case in its own subprocess; return their output digests."""
    timeout = float(os.environ.get(_TIMEOUT_ENV, str(_DEFAULT_TIMEOUT_SECONDS)))
    tree = os.path.dirname(os.path.dirname(os.path.abspath(helion.__file__)))
    env = {
        **os.environ,
        "HELION_BACKEND": "cute",
        "HELION_CUTE_MMA_IMPL": "tcgen05",
        "PYTHONPATH": os.pathsep.join(
            path for path in (tree, os.environ.get("PYTHONPATH", "")) if path
        ),
    }
    procs = {
        name: subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--case", json.dumps(case)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for name, case in cases.items()
    }
    digests: dict[str, str] = {}
    try:
        for name, proc in procs.items():
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                pytest.fail(
                    f"{name} did not finish within {timeout:.0f}s: the kernel hung\n"
                    f"--- stderr ---\n{stderr[-3000:]}"
                )
            assert proc.returncode == 0, (
                f"{name} exited with {proc.returncode}\n"
                f"--- stdout ---\n{stdout[-3000:]}\n--- stderr ---\n{stderr[-3000:]}"
            )
            results = [
                line for line in stdout.splitlines() if line.startswith("RESULT ")
            ]
            assert len(results) == 1, stdout[-3000:]
            digests[name] = json.loads(results[0][len("RESULT ") :])["digest"]
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
    return digests


def _assert_one_digest(digests: dict[str, str]) -> None:
    assert len(set(digests.values())) == 1, digests


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_swizzled_odd_n_tiles_complete_under_both_rasters() -> None:
    """Padded dim-1 slots are skipped on the two-CTA and one-CTA batched grids."""
    _skip_without_tcgen05()
    cases = {
        f"{grid_name}_{raster}": {
            "kernel": "batched_matmul",
            "shape": _ODD_N_SHAPE,
            "config": {
                **grid,
                "tcgen05_batch_raster": raster,
                "tcgen05_l2_swizzle_size": 2,
            },
        }
        for grid_name, grid in (("two_cta", _TWO_CTA), ("one_cta", _ONE_CTA))
        for raster in _RASTERS
    }
    _assert_one_digest(_run_cases(cases))


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_scheduler_warp_bmm_completes_under_both_rasters() -> None:
    """The scheduler-warp mailbox feeds the right tiles under either walk."""
    _skip_without_tcgen05()
    cases = {
        f"cluster_n{cluster_n}_{raster}": {
            "kernel": "batched_matmul",
            "shape": _LOOPING_SHAPE,
            "config": {
                **_TWO_CTA,
                **_WITH_SCHEDULER,
                "tcgen05_cluster_n": cluster_n,
                "tcgen05_batch_raster": raster,
            },
        }
        for cluster_n in (1, 2)
        for raster in _RASTERS
    }
    _assert_one_digest(_run_cases(cases))


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_scheduler_warp_rowvec_bias_completes_under_both_rasters() -> None:
    """The aux C-input warp decodes the mailbox like the MMA and store roles."""
    _skip_without_tcgen05()
    cases = {
        f"cluster_n{cluster_n}_{raster}": {
            "kernel": "batched_matmul_rowvec_bias",
            "shape": _LOOPING_SHAPE,
            "config": {
                **_TWO_CTA,
                **_WITH_SCHEDULER,
                "tcgen05_cluster_n": cluster_n,
                "tcgen05_batch_raster": raster,
            },
        }
        for cluster_n in (1, 2)
        for raster in _RASTERS
    }
    _assert_one_digest(_run_cases(cases))


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    result = _run_case(json.loads(parser.parse_args().case))
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    _main()
