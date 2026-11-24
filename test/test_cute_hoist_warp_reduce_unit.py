"""Pure-AST unit tests for ``hoist_warp_reduce_from_vloop``'s running-sum
handling (and for ``validate_cluster_reduce_placement``'s trip counts, at the
end).

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

import pytest

from helion import exc
from helion._compiler.cute.hoist_warp_reduce import hoist_warp_reduce_from_vloop
from helion._compiler.cute.hoist_warp_reduce import validate_cluster_reduce_placement
from helion._testing import TestCase

# Pure-AST tests: they run on every backend row.
pytestmark = pytest.mark.cpu_only


def _run(
    src: str,
    running_sums: set[str] | None = None,
    input_dtypes: dict[str, str | None] | None = None,
) -> str:
    body = hoist_warp_reduce_from_vloop(
        ast.parse(textwrap.dedent(src)).body,
        running_sum_accumulators=running_sums,
        reduce_input_dtypes=input_dtypes,
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
        # accumulator + one warp reduce after the loop).  The emitter says
        # which dtype the direct reduce combines in.
        src = """
        for vec_lane_1 in cutlass.range_constexpr(4):
            values = cutlass.Float32(_v0[vec_lane_1])
            p = cute.math.exp2(values)
            s = cutlass.Float32(cute.arch.warp_reduction_sum(p, threads_in_group=32))
            di = di + s
        """
        self.assertExpectedJournal(
            _run(src, running_sums=set(), input_dtypes={"p": "cutlass.Float32"})
        )

    def test_store_elements_ride_in_the_fold_loop(self) -> None:
        # A where's result stored after the loop and summed: the elements of
        # the store are computed in the fold loop, in their order, and the
        # sum is reduced once after it (it stayed per element, and the config
        # failed in fold_lane_reductions).
        src = """
        _tile_store_vals_1_0 = []
        for vec_lane_1 in cutlass.range_constexpr(4):
            tmp3 = cutlass.Float32(s) if m[vec_lane_1] else cutlass.Float32(_v0[vec_lane_1])
            v_2 = cutlass.BFloat16(tmp3)
            _tile_store_vals_1_0.append(cutlass.BFloat16(v_2).bitcast(cutlass.Uint16))
            _mask_to = cutlass.Float32(tmp3)
            sum_1 = cutlass.Float32(cute.arch.warp_reduction_sum(_mask_to, threads_in_group=32))
            v_3 = acc + sum_1
        _cute_store_u16_vec(out0.iterator, _tile_store_vals_1_0)
        """
        out = _run(src, input_dtypes={"_mask_to": "cutlass.Float32"})
        module = ast.parse(out)
        (loop,) = [n for n in module.body if isinstance(n, ast.For)]
        body = ast.unparse(loop)
        self.assertIn("_tile_store_vals_1_0.append(", body)
        self.assertIn("_helion_vfold_acc_0 = _helion_vfold_acc_0 + ", body)
        self.assertNotIn("warp_reduction_sum", body)
        after = ast.unparse(
            ast.Module(body=module.body[module.body.index(loop) + 1 :], type_ignores=[])
        )
        self.assertIn("warp_reduction_sum(_helion_vfold_acc_0", after)
        self.assertIn("v_3 = acc + sum_1", after)

    def test_store_elements_between_two_reductions_keep_the_loop(self) -> None:
        # An element appended between two reductions belongs to no fold loop;
        # hoisted, the append was dropped and the store wrote an empty list.
        src = """
        _tile_store_vals_1_0 = []
        for vec_lane_1 in cutlass.range_constexpr(4):
            a = cutlass.Float32(_v0[vec_lane_1])
            sum_1 = cutlass.Float32(cute.arch.warp_reduction_sum(a, threads_in_group=32))
            b = cutlass.Float32(_v1[vec_lane_1])
            _tile_store_vals_1_0.append(cutlass.BFloat16(b).bitcast(cutlass.Uint16))
            c = cutlass.Float32(_v2[vec_lane_1])
            sum_2 = cutlass.Float32(cute.arch.warp_reduction_sum(c, threads_in_group=32))
            v_3 = acc + sum_1 + sum_2
        _cute_store_u16_vec(out0.iterator, _tile_store_vals_1_0)
        """
        out = _run(src, input_dtypes={"a": "cutlass.Float32", "c": "cutlass.Float32"})
        self.assertIn("_tile_store_vals_1_0.append(", out)
        self.assertNotIn("_helion_vfold_acc", out)

    def test_store_elements_take_their_values_into_the_fold_loop(self) -> None:
        # The append ahead of both reductions runs in the first fold loop, and
        # so must the value it stores, which only the second reduction reads:
        # without it the first loop appended a ``b`` it had not defined.
        src = """
        _tile_store_vals_1_0 = []
        for vec_lane_1 in cutlass.range_constexpr(4):
            a = cutlass.Float32(_v0[vec_lane_1])
            b = cutlass.Float32(_v1[vec_lane_1])
            _tile_store_vals_1_0.append(cutlass.BFloat16(b).bitcast(cutlass.Uint16))
            sum_1 = cutlass.Float32(cute.arch.warp_reduction_sum(a, threads_in_group=32))
            sum_2 = cutlass.Float32(cute.arch.warp_reduction_sum(b, threads_in_group=32))
            v_3 = acc + sum_1 + sum_2
        _cute_store_u16_vec(out0.iterator, _tile_store_vals_1_0)
        """
        out = _run(src, input_dtypes={"a": "cutlass.Float32", "b": "cutlass.Float32"})
        loops = [n for n in ast.parse(out).body if isinstance(n, ast.For)]
        self.assertEqual(len(loops), 2)
        first = [ast.unparse(stmt) for stmt in loops[0].body]
        append = next(i for i, text in enumerate(first) if ".append(" in text)
        self.assertIn("b = cutlass.Float32(_v1[vec_lane_1])", first[:append])
        self.assertNotIn(".append(", ast.unparse(loops[1]))

    def test_element_stores_ride_in_the_fold_loop_of_one_reduction(self) -> None:
        # An r-vectorized copy and sum over r whose output the sink leaves per
        # element: the stores, masked or not, run in the fold loop after the
        # values they store, and the sum is reduced once after it (it stayed
        # per element, and the config failed in fold_lane_reductions).
        src = """
        for vec_lane_1 in cutlass.range_constexpr(4):
            indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
            mask_1 = indices_1 < 100
            tmp3 = cutlass.Float32(_v0[vec_lane_1])
            v_1 = cutlass.BFloat16(tmp3)
            (out0.iterator + cutlass.Int32(indices_1) * 64).store(cutlass.BFloat16(v_1))
            if mask_1:
                (out1.iterator + cutlass.Int32(indices_1) * 64).store(cutlass.Float32(tmp3))
            else:
                pass
            sum_1 = cutlass.Float32(cute.arch.warp_reduction_sum(tmp3, threads_in_group=32))
            v_3 = acc + sum_1
        """
        out = _run(src, input_dtypes={"tmp3": "cutlass.Float32"})
        module = ast.parse(out)
        (loop,) = [n for n in module.body if isinstance(n, ast.For)]
        body = [ast.unparse(stmt) for stmt in loop.body]
        stores = [i for i, text in enumerate(body) if ".store(" in text]
        self.assertEqual(len(stores), 2)
        self.assertLess(body.index("v_1 = cutlass.BFloat16(tmp3)"), stores[0])
        self.assertTrue(body[stores[1]].startswith("if mask_1:"))
        self.assertNotIn("warp_reduction_sum", ast.unparse(loop))
        after = ast.unparse(
            ast.Module(body=module.body[module.body.index(loop) + 1 :], type_ignores=[])
        )
        self.assertIn("warp_reduction_sum(_helion_vfold_acc_0", after)
        self.assertIn("v_3 = acc + sum_1", after)

    def test_a_store_ahead_of_two_reductions_keeps_the_loop(self) -> None:
        # The second fold loop would load its elements after every store of
        # the first: a buffer updated in place would read the stored values.
        src = """
        for vec_lane_1 in cutlass.range_constexpr(4):
            a = cutlass.Float32(_v0[vec_lane_1])
            (out0.iterator + vec_lane_1).store(cutlass.BFloat16(a))
            sum_1 = cutlass.Float32(cute.arch.warp_reduction_sum(a, threads_in_group=32))
            c = cutlass.Float32((in0.iterator + vec_lane_1).load())
            sum_2 = cutlass.Float32(cute.arch.warp_reduction_sum(c, threads_in_group=32))
            v_3 = acc + sum_1 + sum_2
        """
        out = _run(src, input_dtypes={"a": "cutlass.Float32", "c": "cutlass.Float32"})
        self.assertNotIn("_helion_vfold_acc", out)

    def test_a_store_with_an_effect_in_its_value_keeps_the_loop(self) -> None:
        # Only stores whose address, value and mask make no call with an
        # effect move: an atomic in the stored value keeps the loop.
        src = """
        for vec_lane_1 in cutlass.range_constexpr(4):
            tmp3 = cutlass.Float32(_v0[vec_lane_1])
            (out0.iterator + vec_lane_1).store(cute.arch.atomic_add(ptr, tmp3))
            sum_1 = cutlass.Float32(cute.arch.warp_reduction_sum(tmp3, threads_in_group=32))
            v_3 = acc + sum_1
        """
        out = _run(src, input_dtypes={"tmp3": "cutlass.Float32"})
        self.assertNotIn("_helion_vfold_acc", out)

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


@pytest.mark.parametrize(
    ("spelling", "trips"),
    [
        ("cutlass.range_constexpr(1, 3)", 2),
        ("cutlass.range_constexpr(2, 3)", 1),
        ("cutlass.range_constexpr(N)", 1),
        ("range(1, -1, -1)", 2),
    ],
)
def test_cluster_reduce_placement_counts_any_range_spelling(
    spelling: str, trips: int
) -> None:
    """A cluster reduce must run once per kernel, so the check counts the
    trips of the loops around it the way the lane passes do (``_loop_trips``),
    whatever the range's spelling.  It read a ``range_constexpr``'s first
    argument as its trip count (``(1, 3)`` once, ``(2, 3)`` twice) and took a
    bound named by a constexpr value as unknown."""
    body = ast.parse(f"for i in {spelling}:\n    x = _cute_grouped_reduce_cluster(y)\n")
    if trips > 1:
        with pytest.raises(exc.BackendUnsupported, match="exactly once per kernel"):
            validate_cluster_reduce_placement(body.body, {"N": 1})
    else:
        validate_cluster_reduce_placement(body.body, {"N": 1})


if __name__ == "__main__":
    import unittest

    unittest.main()
