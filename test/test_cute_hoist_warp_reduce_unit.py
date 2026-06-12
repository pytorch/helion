"""Pure-AST unit tests for ``hoist_warp_reduce_from_vloop``'s running-sum
handling.

The pass hoists a ``cute.arch.warp_reduction_*`` out of a constexpr V-loop.
How it hoists depends on the reduce input:

  (a) a FRESH per-lane value (softmax / layernorm ``amax``/``sum``): build a
      ``_helion_vfold_acc`` and fold each lane in, then ONE warp reduce.
  (b) a matmul-fallback PER-THREAD RUNNING SUM, flagged authoritatively by the
      producer via ``running_sum_accumulators``: keep the self-accumulating
      ``acc = acc + product`` def in the loop and reduce its FINAL value ONCE
      (folding it again would double-count the already-accumulated sum).
  (c) loop-carried (its in-loop def reads itself) but NOT a flagged running sum
      (a rescaled recurrence, ``di = di * alpha + ...``): neither (a) nor (b)
      is valid, so leave the V-loop untouched.

The (b) decision is driven by the producer's flag — NOT by re-deriving the
shape from the AST — so these tests pass the running-sum set explicitly.

Each test inlines its own kernel and snapshots the full rewritten output via
``assertExpectedJournal`` (run with ``EXPECTTEST_ACCEPT=1`` to update).  These
run the pass directly on parsed ASTs — no CUDA.
"""

from __future__ import annotations

import ast
import textwrap

from helion._compiler.cute.hoist_warp_reduce import hoist_warp_reduce_from_vloop
from helion._testing import TestCase


def _run(src: str, running_sums: set[str] | None = None) -> str:
    body = hoist_warp_reduce_from_vloop(
        ast.parse(textwrap.dedent(src)).body,
        running_sum_accumulators=running_sums,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


class TestHoistWarpReduceRunningSum(TestCase):
    # ---- (b) flagged running sum -> reduce-once, no V-fold ----

    def test_flagged_running_sum_reduced_once(self) -> None:
        # matmul SIMT fallback: ``dot_acc`` is a per-thread running sum flagged
        # by the producer.  Expect the self-accumulating update to stay in the
        # loop and a single warp reduce hoisted to AFTER it (no V-fold).
        src = """
        dot_acc_base = acc
        dot_acc = cutlass.Float32(0)
        for vec_lane_1 in cutlass.range_constexpr(4):
            load = cutlass.Uint8(_v0 >> 8 * vec_lane_1 & 255)
            load_1 = cutlass.Uint8(_v1 >> 8 * vec_lane_1 & 255)
            dot_product_0 = cutlass.Float32(_dec(load) * _dec(load_1))
            dot_acc = dot_acc + dot_product_0
            acc = dot_acc_base + cutlass.Float32(cute.arch.warp_reduction_sum(dot_acc, threads_in_group=32))
        """
        self.assertExpectedJournal(_run(src, running_sums={"dot_acc"}))

    def test_flagged_running_sum_grouped_add(self) -> None:
        # Grouped accumulate ``dot_acc = dot_acc + (a + b)`` (paired-decode
        # product).  Driven by the flag, not an AST shape match, so it is
        # handled identically: reduce-once, no V-fold.
        src = """
        dot_acc_base = acc
        dot_acc = cutlass.Float32(0)
        for vec_lane_1 in cutlass.range_constexpr(4):
            dot_acc = dot_acc + (px0 * py0 + px1 * py1)
            acc = dot_acc_base + cutlass.Float32(cute.arch.warp_reduction_sum(dot_acc, threads_in_group=32))
        """
        self.assertExpectedJournal(_run(src, running_sums={"dot_acc"}))

    # ---- (a) fresh per-lane sum -> keeps the V-fold ----

    def test_fresh_sum_keeps_vfold(self) -> None:
        # Not flagged and not loop-carried: the input ``p`` is recomputed each
        # lane, so the original V-fold path applies (a ``_helion_vfold_acc``
        # accumulator + one warp reduce after the loop).
        src = """
        for vec_lane_1 in cutlass.range_constexpr(4):
            values = cutlass.Float32(_v0[vec_lane_1])
            p = cute.math.exp2(values)
            s = cutlass.Float32(cute.arch.warp_reduction_sum(p, threads_in_group=32))
            di = di + s
        """
        self.assertExpectedJournal(_run(src, running_sums=set()))

    # ---- (c) loop-carried but unflagged -> untouched ----

    def test_loop_carried_rescale_unflagged_untouched(self) -> None:
        # ``di = di * alpha + p`` reads ``di`` in its def (loop-carried) but is
        # NOT a flagged running sum: V-folding would fold the carried value and
        # reduce-once's sum identity doesn't hold, so leave the loop exactly as
        # written.
        src = """
        for vec_lane_1 in cutlass.range_constexpr(4):
            p = cutlass.Float32(_v0[vec_lane_1])
            s = cutlass.Float32(cute.arch.warp_reduction_sum(di, threads_in_group=32))
            di = di * alpha + p
        """
        self.assertExpectedJournal(_run(src, running_sums=set()))

    def test_running_sum_unflagged_untouched(self) -> None:
        # The matmul running-sum kernel but WITHOUT flagging ``dot_acc``: it is
        # loop-carried, so case (c) applies and the loop is left untouched (no
        # double-count V-fold, no reduce-once).  Contrast with
        # ``test_flagged_running_sum_reduced_once`` (same input, flagged).
        src = """
        dot_acc_base = acc
        dot_acc = cutlass.Float32(0)
        for vec_lane_1 in cutlass.range_constexpr(4):
            load = cutlass.Uint8(_v0 >> 8 * vec_lane_1 & 255)
            load_1 = cutlass.Uint8(_v1 >> 8 * vec_lane_1 & 255)
            dot_product_0 = cutlass.Float32(_dec(load) * _dec(load_1))
            dot_acc = dot_acc + dot_product_0
            acc = dot_acc_base + cutlass.Float32(cute.arch.warp_reduction_sum(dot_acc, threads_in_group=32))
        """
        self.assertExpectedJournal(_run(src, running_sums=set()))


if __name__ == "__main__":
    import unittest

    unittest.main()
