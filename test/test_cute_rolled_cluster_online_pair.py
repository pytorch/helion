"""The rolled (``reduction_loops``) online-softmax pair and its seeds.

``torch.nn.functional.softmax(x[tile, :])`` lowers to one ``for roffset``
sweep per reduction with the cluster reduce emitted after the sweep.  With
``cute_cluster_warp_online`` the two DSM exchanges become one: under
``fast_math`` the CTA-level packed pair with sweep B's exponentials cached
for the write sweep (``cluster_online_pair`` run on the owner-wrapped rolled
body); otherwise the exact warp pair that keeps every exponential as
written.  The knob is honoured whatever ``cute_register_row`` says, and the
masked rolled-cluster seeds only set the knobs a kernel's sweep structure
can use.
"""

from __future__ import annotations

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._compiler.autotuner_heuristics.cute import CuteMaskedRolledClusterHeuristic
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl

PAIR = "_cute_grouped_reduce_cluster_online_pair("
WARP_PAIR = "_cute_grouped_reduce_cluster_warp_online_pair("
CLUSTER = "_cute_grouped_reduce_cluster("
BLOCK = "_cute_grouped_reduce_block("


def _softmax(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out


def _logsumexp(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    for row in hl.tile(x.shape[0]):
        values = x[row, :].float()
        maximum = values.amax(-1)
        denominator = torch.exp(values - maximum[:, None]).sum(-1)
        out[row] = denominator.log() + maximum
    return out


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :].float()
        scale = torch.rsqrt(torch.mean(values * values, dim=-1, keepdim=True) + 1e-6)
        out[tile_n, :] = (values * scale).to(x.dtype)
    return out


softmax_exact = helion.kernel(backend="cute", static_shapes=True)(_softmax)
softmax_fast_math = helion.kernel(backend="cute", static_shapes=True, fast_math=True)(
    _softmax
)
logsumexp_exact = helion.kernel(backend="cute", static_shapes=True)(_logsumexp)
rms_norm_exact = helion.kernel(backend="cute", static_shapes=True)(_rms_norm)

ROLLED = {
    "block_sizes": [1],
    "num_threads": [1, 128],
    "reduction_loops": [4096],
    "cute_vector_widths": [8, 1],
    "cute_lane_layouts": ["strided", "blocked"],
    "cute_reduction_reloads": ["register"],
    "cute_cluster_n": 4,
    "cute_cluster_warp_online": True,
    "cute_warp_fp32_redux": True,
}


def _render(kernel: object, **patch: object) -> str:
    x = torch.empty((4096, 32768), dtype=torch.float16)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, (x,))
        return bound.to_code(helion.Config(**(ROLLED | patch)))


def _seed_values(kernel: object) -> dict[str, set[object]]:
    x = torch.empty((4096, 32768), dtype=torch.float16)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, (x,))
        with bound.env:
            seeds = CuteMaskedRolledClusterHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
    assert seeds
    return {
        key: {seed.config.get(key) for seed in seeds}
        for key in (
            "cute_cluster_warp_online",
            "cute_register_row",
            "cute_scalar_epilogue",
            "cute_scalar_epilogue_keep_inputs",
            "cute_warp_fp32_redux",
        )
    }


@pytest.mark.parametrize("register_row", [False, True])
def test_fast_math_rolled_pair_caches_the_exponentials(register_row: bool) -> None:
    """One CTA-level packed exchange; sweep C reuses sweep B's exponentials
    (times the CTA rescale) instead of recomputing ``exp2`` in the global
    frame -- and the knob is reached with ``cute_register_row`` on, whose
    lowering declines the three-sweep softmax."""
    code = _render(softmax_fast_math, cute_register_row=register_row)
    assert code.count(PAIR) == 1
    assert code.count(BLOCK) == 1
    assert CLUSTER not in code
    assert WARP_PAIR not in code
    assert "_pair_exp_cache_0 = cute.make_rmem_tensor(64, cutlass.Float32)" in code
    assert "_pair_rescale_0" in code
    assert "cute.arch.alloc_smem(cutlass.Int64, 4)" in code
    # Sweep B's exp2 and the rescale only: the write sweep's recompute is gone.
    assert code.count("cute.math.exp2(") == 2
    assert "_pair_exp_cache_0[" in code.split("_pair_rescale_0 = ")[1]


@pytest.mark.parametrize("register_row", [False, True])
def test_exact_rolled_pair_keeps_every_exponential(register_row: bool) -> None:
    """Without fast_math the warp pair exchanges once but both sweeps keep
    their ``exp2`` as written (sweep B in the warp's own frame)."""
    code = _render(softmax_exact, cute_register_row=register_row)
    assert code.count(WARP_PAIR) == 1
    assert code.count("_cute_grouped_reduce_warp(") == 1
    assert CLUSTER not in code
    assert PAIR not in code
    assert "_pair_exp_cache" not in code
    assert code.count("cute.math.exp2(") == 2
    assert "_rolled_pair_frame" in code


@pytest.mark.parametrize("kernel", [softmax_exact, softmax_fast_math])
def test_knob_off_keeps_the_two_exchange_form(kernel: object) -> None:
    code = _render(kernel, cute_cluster_warp_online=False)
    assert code.count(CLUSTER) == 2
    assert PAIR not in code
    assert WARP_PAIR not in code


def test_seeds_set_only_the_knobs_the_sweep_structure_admits() -> None:
    """Exact softmax: the pair would un-share the two sweeps' exponentials
    and the register row / scalar epilogue never act on a row-storing
    sweep, so no seed turns them on (the knobs are not searched).  Under
    fast_math the cached pair pays, so the seeds carry it.  A scalar-output
    pair (logsumexp) seeds all three; a kernel without the pair (RMS norm)
    seeds none of them.  ``cute_warp_fp32_redux`` stays seeded throughout."""
    exact = _seed_values(softmax_exact)
    assert exact["cute_cluster_warp_online"] == {False}
    assert True not in exact["cute_register_row"]
    assert exact["cute_scalar_epilogue"] == {False}
    assert True not in exact["cute_scalar_epilogue_keep_inputs"]
    assert True in exact["cute_warp_fp32_redux"]

    fast = _seed_values(softmax_fast_math)
    assert fast["cute_cluster_warp_online"] == {True}
    assert True not in fast["cute_register_row"]
    assert fast["cute_scalar_epilogue"] == {False}
    assert True in fast["cute_warp_fp32_redux"]

    lse = _seed_values(logsumexp_exact)
    assert lse["cute_cluster_warp_online"] == {True}
    assert True in lse["cute_register_row"]
    assert lse["cute_scalar_epilogue"] == {True}
    assert True in lse["cute_scalar_epilogue_keep_inputs"]
    assert True in lse["cute_warp_fp32_redux"]

    rms = _seed_values(rms_norm_exact)
    assert rms["cute_cluster_warp_online"] == {False}
    assert True not in rms["cute_register_row"]
    assert rms["cute_scalar_epilogue"] == {False}
    assert True in rms["cute_warp_fp32_redux"]


@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestCuteRolledClusterOnlinePairCuda(TestCase):
    def test_fast_math_cached_pair_matches_torch(self) -> None:
        """The cached single exchange, including a CTA whose whole slice is
        -inf (zero frame, zero rescale) and an all--inf row (NaN like
        torch)."""
        x = torch.randn(64, 32768, device=DEVICE, dtype=torch.float16)
        x[3, :8192] = float("-inf")
        x[5, :] = float("-inf")
        x[7, 8192:] = float("-inf")
        code, out = code_and_output(softmax_fast_math, (x,), **ROLLED)
        self.assertEqual(code.count(PAIR), 1)
        self.assertNotIn(CLUSTER, code)
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3, equal_nan=True)
        self.assertTrue(torch.isnan(out[5]).all())
        self.assertEqual(out[3, :8192].abs().max().item(), 0.0)

    def test_exact_warp_pair_matches_torch(self) -> None:
        x = torch.randn(64, 32768, device=DEVICE, dtype=torch.float16)
        x[3, :8192] = float("-inf")
        code, out = code_and_output(softmax_exact, (x,), **ROLLED)
        self.assertEqual(code.count(WARP_PAIR), 1)
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)
