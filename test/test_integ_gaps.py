"""Cross-loop schedule paths the other suites do not reach (from the
integration review's mutation testing: each test kills a mutant that passed
every other suite).

* grouped source root whose task count is not a multiple of the group (port:moe F2): the last ticket runs fewer tasks,
  publishes ``tl.minimum(g, n - first)`` arrivals, and its out-of-range group slot is skipped.  The in-tree test uses 64
  tasks with groups of 4 and 2 only, so neither the partial count nor the range guard is exercised there.
* a root-barrier publisher under a transitive cover (port:moe F3): its counter waits must stay unconditional when a
  downstream consumer relies on them through the cover.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

import helion
from helion._compiler import cross_loop_scheduler
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl

_GLOBALTIMER_SPIN = (
    "{ .reg .u64 t0, t1; .reg .pred p; mov.u64 t0, %globaltimer; "
    "SPIN: mov.u64 t1, %globaltimer; sub.u64 t1, t1, t0; "
    "setp.lt.u64 p, t1, $1; @p bra SPIN; mov.u32 $0, 0; }"
)


def _grouped_source_impl(
    w: torch.Tensor, x: torch.Tensor, parts: torch.Tensor, delay: torch.Tensor
) -> torch.Tensor:
    # S (n tasks, a 64 KiB fp32 tile each, no producer) writes one partial per task after spinning `delay` ns; C reads
    # every partial (a root barrier for all of S).
    out = torch.empty_like(x)
    for tile in hl.tile(w.size(0), block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [delay[0]],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        parts[tile] = torch.sum(torch.sum(w[tile, :, :], dim=-1), dim=-1) + spun.to(
            torch.float32
        )
    for tile in hl.tile(x.size(0), block_size=1):
        out[tile] = x[tile] + torch.sum(parts[:], dim=0)
    return out


# 1 MiB + 4 KiB budget, default cap 2: n=63 tasks of 64 KiB -> groups of 2 -> 32 tickets, the last one with 1 task.
grouped_rem = helion.kernel(
    _grouped_source_impl,
    static_shapes=True,
    autotune_effort="none",
    cross_loop_source_inflight_budget=(1 << 20) + (1 << 12),
)
# cap 4: 4 + 4 ... + 3 (63 = 15 * 4 + 3)
grouped_rem4 = helion.kernel(
    _grouped_source_impl,
    static_shapes=True,
    autotune_effort="none",
    cross_loop_source_inflight_budget=(1 << 20) + (1 << 12),
    cross_loop_source_max_group=4,
)


def _f3_chain_impl(
    x: torch.Tensor,
    keep: torch.Tensor,
    small: torch.Tensor,
    r_out: torch.Tensor,
    delay: torch.Tensor,
) -> torch.Tensor:
    # S writes small[i] after spinning delay[i] ns; R (publishes only a root barrier: P reads all of r_out) reads
    # small[i] only where keep[i] > 0; P reads all of r_out and publishes keyed rows; C reads its keyed row and ALL of
    # small.  The scheduler covers C's S obligation transitively through P's keyed counter (P waited on every R task,
    # each of which waited on its row of S), so R's counter waits must stay unconditional (F3's certifies_downstream).
    keyed = torch.empty_like(x)
    out = torch.empty_like(x)
    for t in hl.tile(x.size(0), block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [delay[t.begin]],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        small[t] = x[t] + 1 + spun.to(torch.float32)
    for t in hl.tile(x.size(0), block_size=1):
        if keep[t.begin] > 0:
            r_out[t] = small[t] * 2
    for t in hl.tile(x.size(0), block_size=1):
        keyed[t] = x[t] + torch.sum(r_out[:], dim=0)
    for t in hl.tile(x.size(0), block_size=1):
        out[t] = keyed[t] + torch.sum(small[:], dim=0)
    return out


CANARY = 12345.0


class TestIntegGaps(TestCase):
    def _run_grouped(self, kernel, n, group):
        w = torch.randn(n, 128, 128, device=DEVICE, dtype=torch.float32)
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        buf = torch.empty(n + 64, device=DEVICE, dtype=torch.float32)
        parts = buf[:n]
        delay = torch.full((1,), 20000, device=DEVICE, dtype=torch.int64)
        code = None
        for launch in range(4):
            buf[:n].fill_(float("nan"))
            buf[n:].fill_(CANARY)
            code, out = code_and_output(
                kernel,
                (w + launch, x, parts, delay),
                pid_type="persistent_blocked",
                cross_loop_pipeline="dynamic",
                num_sm_multiplier=1,
                num_warps=4,
            )
            torch.cuda.synchronize()
            reference = (w + launch).sum(dim=(1, 2))
            torch.testing.assert_close(parts, reference, rtol=1e-4, atol=1e-2)
            torch.testing.assert_close(out, x + parts.sum(), rtol=1e-5, atol=1e-3)
            # no task outside 0..n-1 ran (a group slot past the end would write buf[n])
            self.assertTrue(bool((buf[n:] == CANARY).all()), buf[n : n + 4])
        assert code is not None
        packets = -(-n // group)
        self.assertIn("tile_dependency_group_task", code)
        self.assertIn(f"in tl.range(0, {group}, 1):", code)
        self.assertIn(f"tl.minimum({group}, {n} - ", code)
        self.assertIn(f"< {n}:", code)
        self.assertIn(f"tile_dependency_dispatch_ticket_1 < {packets}", code)
        self.assertIn(f"tl.cast({n}, tl.uint32)", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_grouped_source_with_a_partial_last_group(self) -> None:
        self._run_grouped(grouped_rem, 63, 2)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_grouped_source_with_a_partial_last_group_of_four(self) -> None:
        self._run_grouped(grouped_rem4, 63, 4)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_barrier_publisher_keeps_its_wait_under_a_transitive_cover(self) -> None:
        # Kills MC3 (certifies_downstream always False): R would sink its S wait into the keep branch, rows with
        # keep == 0 would be waited by nobody, and C (covered only through P) would read them while S still spins
        # (only those rows are delayed, 20 us; small is NaN-poisoned).
        n = 8
        x = torch.randn(n, device=DEVICE, dtype=torch.float32)
        keep = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0], device=DEVICE, dtype=torch.int32)
        delay = torch.where(keep > 0, 0, 20000).to(torch.int64)
        small = torch.empty(n, device=DEVICE, dtype=torch.float32)
        r_out = torch.empty(n, device=DEVICE, dtype=torch.float32)
        kernel = helion.kernel(
            _f3_chain_impl, static_shapes=True, autotune_effort="none"
        )
        no_continuations = mock.patch.object(
            cross_loop_scheduler,
            "choose_final_arrival_continuations",
            lambda *args, **kwargs: (),
        )
        bad = 0
        with no_continuations:
            for launch in range(20):
                small.fill_(float("nan"))
                r_out.zero_()
                code, out = code_and_output(
                    kernel,
                    (x + launch, keep, small, r_out, delay),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="dynamic",
                    num_sm_multiplier=1,
                    num_warps=1,
                )
                torch.cuda.synchronize()
                s_ref = x + launch + 1
                ref = (
                    (x + launch)
                    + torch.where(keep > 0, s_ref * 2, 0).sum()
                    + s_ref.sum()
                )
                bad += int(not torch.allclose(out, ref, rtol=1e-5, atol=1e-3))
        self.assertEqual(bad, 0)
        # the structure that makes the test meaningful: C does not wait on S itself (covered through P)
        c_body = code.split("def tile_dependency_root_3(")[1].split("@triton.jit")[0]
        self.assertNotIn("tile_dependency_state + 0 + 0 + 0 +", c_body)
        self.assertIn("readiness_wait", c_body)


if __name__ == "__main__":
    unittest.main()
