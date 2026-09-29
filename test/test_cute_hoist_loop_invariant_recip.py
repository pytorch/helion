"""Tests for the CuTe ``hoist_loop_invariant_recips`` AST pass.

A multi-sub-pass orchestration that runs after the strategy lowers the
kernel and before the cute DSL trace:

1. **Alias DCE** — inline pure SSA-style ``NAME = ANOTHER_NAME`` chains
   so the per-iter ``mi_copy = mi; mi_copy_0 = mi_copy`` aliases that
   Helion's SSA maintenance leaves behind collapse to direct ``mi``
   reads.
2. **Reciprocal hoist** (the original P16 pass, now outer-in) — walks
   for-loops outer-to-inner and rewrites ``x / scalar`` to
   ``inv = 1.0 / scalar`` hoisted above the OUTERMOST legal scope plus
   ``x * inv`` inside. fp32 divide is ~22 cycles on B200 vs ~2 for
   multiply, and softmax's consume sweep emits one divide per fp16
   element across N elements per row.
3. **FMA-friendly scale hoist** — ``(A - INV) * CONST`` where ``INV``
   is loop-invariant becomes ``A * CONST - HOISTED`` where
   ``HOISTED = INV * CONST`` is hoisted above the loop. Same value,
   one fewer inner-loop multiply, and FMA-friendly for ptx codegen.
4. **DCE for dead pure assigns** — removes the original ``v_X = A - INV``
   subs that became dead after the FMA hoist.
5. **Block-local reciprocal sharing** — ``n >= 2`` divisions by the same
   divisor value inside one straight-line block (a divisor recomputed
   every iteration, which the hoist cannot lift) share one
   ``inv = 1.0 / d`` and become ``x * inv`` each.  Same rounding class as
   the hoist (``fl(x * fl(1/d))``: at most 1 ulp from the IEEE quotient).

The hoist-OUT passes (2 and 3) use a **canonical-aware invariance**
mode that maps names through Helion's rename-group map so ``v_1_0``
(which renames to ``mi`` post-pass) is recognized as the SAME variable
as ``mi``. Without this canonicalization the FMA hoist would lift
``mi * 1.4427`` ABOVE the reduce loop and capture the stale initial
``-inf`` value of ``mi``, producing wildly wrong outputs.

Lives in ``helion/_compiler/cute/hoist_loop_invariant_recip.py``.
"""

from __future__ import annotations

import ast
import os
import textwrap
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.hoist_loop_invariant_recip import hoist_loop_invariant_recips
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@pytest.fixture(autouse=True)
def _disable_online_to_3pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests in this file pin codegen details of the ORIGINAL online
    two-pass form.  The ``online_to_3pass`` rewrite would change them,
    so disable it here; the rewrite itself is covered in
    ``test_cute_online_to_3pass.py``.
    """
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")


@helion.kernel(backend="cute")
def _reduction_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


@onlyBackends(["cute"])
class TestCuteHoistRecip(TestCase):
    """Reciprocal hoist (P16 + outer-in walk from P17)."""

    def test_recip_hoist_fires_on_two_pass_pattern(self) -> None:
        """The consume sweep's per-element divide by ``di`` (a per-row
        scalar) must be rewritten to a single hoisted
        ``_helion_inv_div_*`` reciprocal + multiply.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The hoisted reciprocal declaration must reference the
        # loop-external root name ``di``.
        self.assertIn("_helion_inv_div_", code)
        self.assertIn("= 1.0 / di", code)
        # The inner divide must have been rewritten to a multiply against
        # the hoisted reciprocal name.
        self.assertIn("* _helion_inv_div_", code)
        # The original per-element divide pattern is no longer present.
        self.assertNotIn("/ di_copy_1_0", code)

    def test_recip_hoist_does_not_fire_on_loop_dependent_divisor(self) -> None:
        """When the divisor changes per-iteration (e.g. ``local_sum``
        computed inside the loop), the pass must NOT hoist it.
        """

        @helion.kernel(backend="cute")
        def divide_inside_loop(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                for tile_n in hl.tile(n):
                    values = x[tile_m, tile_n]
                    # The divisor changes per tile_n iteration.
                    local_sum = torch.sum(values, dim=1, keepdim=True)
                    out[tile_m, tile_n] = values / local_sum
            return out

        x = torch.randn(4096, 1024, device=DEVICE, dtype=HALF_DTYPE)
        code, _ = code_and_output(
            divide_inside_loop,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        # No reciprocal hoist — local_sum is recomputed every iter.
        self.assertNotIn("_helion_inv_div_", code)

    def test_recip_hoist_disable_env(self) -> None:
        """``HELION_DISABLE_HOIST_RECIP=1`` disables the pass for
        experimentation.
        """
        old = os.environ.get("HELION_DISABLE_HOIST_RECIP")
        os.environ["HELION_DISABLE_HOIST_RECIP"] = "1"
        try:
            x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
            code, out = code_and_output(
                _reduction_kernel,
                (x,),
                block_sizes=[1, 128],
                num_threads=[0, 32],
                cute_vector_widths=[1, 4],
            )
            ref = torch.nn.functional.softmax(x, dim=1)
            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
            # No reciprocal hoist with the env disabled.
            self.assertNotIn("_helion_inv_div_", code)
            # The original per-element divide is still there.
            self.assertIn("/ di_copy_", code)
        finally:
            if old is None:
                os.environ.pop("HELION_DISABLE_HOIST_RECIP", None)
            else:
                os.environ["HELION_DISABLE_HOIST_RECIP"] = old


@onlyBackends(["cute"])
class TestCuteAliasDCE(TestCase):
    """Alias DCE sub-pass (inline SSA-style ``NAME = ANOTHER_NAME`` chains)."""

    def test_ssa_alias_chain_inlined(self) -> None:
        """The per-iter ``mi_copy_*`` and ``di_copy_*`` alias chains
        Helion's SSA maintenance inserts MUST be inlined so the deepest
        use reads the root name directly.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Neither ``mi_copy`` nor ``di_copy`` should appear in the
        # generated code — both were SSA snapshots whose only purpose
        # was to capture the value at a specific point; with the
        # snapshot removed the inner use reads the root directly.
        self.assertEqual(code.count("_copy"), 0)

    def test_useless_cascade_alias_removed(self) -> None:
        """The original inner-first hoist produced a cascade of useless
        ``_helion_inv_div_N = 1.0 * _helion_inv_div_{N+1}`` aliases when
        the consume loop was 3 levels deep.  The outer-in walk emits a
        single hoist at the outermost legal scope instead.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The useless cascade text MUST be gone.
        self.assertEqual(code.count("1.0 * _helion_inv_div_"), 0)
        # Exactly ONE reciprocal hoist for the consume sweep's ``1.0/di``.
        self.assertEqual(code.count("= 1.0 / di"), 1)


@onlyBackends(["cute"])
class TestCuteFMAScaleHoist(TestCase):
    """FMA-friendly scale hoist sub-pass + DCE for dead Sub assigns."""

    def test_fma_scale_hoist_above_consume(self) -> None:
        """The consume loop's ``exp2((v_9 - mi) * 1.4427)`` pattern with
        ``mi`` loop-invariant MUST get a hoisted
        ``_helion_scaled_K = mi * 1.4426950408889634`` placed BEFORE the
        consume loop, and the inner expression must be rewritten to
        ``v_9 * 1.4427 - _helion_scaled_K``.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # A scaled hoist for ``mi * 1.4426950408889634`` must be emitted.
        self.assertIn("_helion_scaled_", code)
        self.assertIn("= mi * 1.4426950408889634", code)
        # The inner consume body must use the FMA-friendly form.
        self.assertIn("cute.math.exp2(v_9 * 1.4426950408889634 - _helion_scaled_", code)

    def test_fma_scale_hoist_in_reduce_v_loop(self) -> None:
        """In the reduce loop's INNER ``for vec_lane_1`` V-loop, the
        ``(v_5 - v_1) * 1.4427`` pattern has ``v_1`` loop-invariant
        w.r.t. the V-loop (v_1 is the new-max from the warp_reduce
        that runs before the V-loop), so the scale hoist must also
        fire here.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The reduce-loop scale hoist for ``v_1 * 1.4427``.
        self.assertIn("= v_1 * 1.4426950408889634", code)
        # The V-loop body preserves the explicit FP32 FMA via v_5.
        self.assertIn(
            "cute.math.exp2(cute.math.fma(v_5, 1.4426950408889634, -_helion_scaled_",
            code,
        )

    def test_dce_removes_dead_sub_after_fma_hoist(self) -> None:
        """After the FMA hoist rewrites ``cast(v_X) * CONST`` to read
        the underlying ``A`` directly (where ``v_X = A - INV`` was the
        Sub statement), the ``v_X = A - INV`` becomes dead and the
        final DCE pass must remove it.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The consume-loop ``v_10 = v_9 - mi`` MUST be DCE'd.
        self.assertNotIn("v_10 = v_9 - mi", code)
        # Same for the inner reduce V-loop ``v_6 = v_5 - v_1``.
        self.assertNotIn("v_6 = v_5 - v_1", code)
        # But statements that ARE read MUST NOT be DCE'd (the surviving
        # ``di = v_4 + sum_1`` update is subsequently contracted by the
        # fuse_fma pass into a single FMA).
        self.assertIn("di = cute.math.fma(di, v_3, sum_1)", code)

    def test_invariance_canonicalization_does_not_break_consume(self) -> None:
        """The pass must use the post-rename canonical name map for
        invariance analysis on hoist-OUT passes, so that ``mi`` in the
        REDUCE loop body is correctly classified as loop-VARIANT
        (because ``v_1_0 = v_1`` will be renamed to ``mi = v_1``).
        Without this, the FMA hoist would lift the ``mi * 1.4427`` scale
        ABOVE the reduce loop and capture the stale initial ``-inf``
        value of ``mi`` — silently producing wildly wrong outputs.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        # Tight tolerance — this test specifically guards a silent
        # mis-compile (the bench would still "look fast" with wrong
        # outputs if we regressed here).
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The ``mi`` scale hoist for the consume sweep lives BETWEEN
        # the two outer for-loops (after the reduce loop finishes
        # mutating ``mi`` via the rename ``v_1_0 -> mi``), not before
        # the reduce loop.
        reduce_end = code.find("mi = v_1\n")
        self.assertGreaterEqual(reduce_end, 0)
        scaled_consume = code.find("= mi * 1.4426950408889634")
        self.assertGreater(scaled_consume, reduce_end)


# The per-element lane body the CuTe backend emits for ``swiglu_bwd``
# (``examples/swiglu.py``): ``dextra = x1_exp / p + x1 * x1_exp / p / p``
# with ``p = exp(x1) + 1`` recomputed for every element, so the
# loop-invariant hoist cannot fire and each quotient is an IEEE divide.
_SWIGLU_LANE_BODY = """\
for vec_lane_0 in cutlass.range_constexpr(8):
    load = cutlass.Uint16(_vec_0[vec_lane_0]).bitcast(cutlass.Float16)
    v_0 = cutlass.Float32(load)
    v_7 = cute.math.exp2(cutlass.Float32(v_0) * 1.4426950408889634)
    v_8 = 1
    v_9 = v_7 + v_8
    v_10 = v_7 / v_9
    v_11 = v_0 * v_7
    v_12 = v_11 / v_9
    v_13 = v_12 / v_9
    v_14 = v_10 + v_13
    _tile_store_vals_0_1.append(cutlass.Float16(v_14).bitcast(cutlass.Uint16))
"""

_SWIGLU_LANE_BODY_SHARED = """\
for vec_lane_0 in cutlass.range_constexpr(8):
    load = cutlass.Uint16(_vec_0[vec_lane_0]).bitcast(cutlass.Float16)
    v_0 = cutlass.Float32(load)
    v_7 = cute.math.exp2(cutlass.Float32(v_0) * 1.4426950408889634)
    v_8 = 1
    v_9 = v_7 + v_8
    _helion_inv_div_0 = 1.0 / v_9
    v_10 = v_7 * _helion_inv_div_0
    v_11 = v_0 * v_7
    v_12 = v_11 * _helion_inv_div_0
    v_13 = v_12 * _helion_inv_div_0
    v_14 = v_10 + v_13
    _tile_store_vals_0_1.append(cutlass.Float16(v_14).bitcast(cutlass.Uint16))
"""


def _run_pass(source: str, rename_groups: dict[str, str] | None = None) -> str:
    """Run the full ``hoist_loop_invariant_recips`` pass on a kernel-body
    snippet (GPU-free) and return the unparsed result."""
    body = ast.parse(textwrap.dedent(source)).body
    rewritten = hoist_loop_invariant_recips(body, rename_groups)
    return ast.unparse(ast.Module(body=rewritten, type_ignores=[]))


def _normalize(source: str) -> str:
    return ast.unparse(ast.parse(textwrap.dedent(source)))


@onlyBackends(["cute"])
class TestCuteSharedRecip(TestCase):
    """Block-local reciprocal sharing sub-pass (AST-level, no GPU)."""

    def test_three_divisions_by_one_divisor_share_one_reciprocal(self) -> None:
        """``swiglu_bwd``: three IEEE divides by ``v_9`` become one
        IEEE reciprocal placed right after ``v_9`` is defined plus one
        multiply per quotient; nothing else in the body changes."""
        code = _run_pass(_SWIGLU_LANE_BODY)
        self.assertEqual(code, _normalize(_SWIGLU_LANE_BODY_SHARED))
        self.assertEqual(code.count("= 1.0 / v_9"), 1)
        self.assertEqual(code.count("* _helion_inv_div_0"), 3)
        # The reciprocal is the only remaining division by ``v_9``.
        self.assertEqual(code.count("/ v_9"), 1)

    def test_recip_hoist_descends_into_while_bodies(self) -> None:
        """A persistent flash kernel wraps every role body in a ``while``
        tile loop.  A ``for`` loop nested in that ``while`` whose divisor is
        rebound per tile (so it is invariant only relative to the ``for``)
        must still get its reciprocal hoisted -- to just above the ``for``,
        inside the ``while`` body."""
        source = """\
            while flash_tile_id < 128:
                v_7 = cute.math.max(v_5, eps)
                acc = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(16):
                    v_2 = cutlass.Float32(a[j])
                    v_8 = v_2 / v_7
                    acc = acc + v_8
                flash_tile_id = flash_tile_id + 1
            """
        code = _run_pass(source)
        expected = """\
            while flash_tile_id < 128:
                v_7 = cute.math.max(v_5, eps)
                acc = cutlass.Float32(0.0)
                _helion_inv_div_0 = 1.0 / v_7
                for j in cutlass.range_constexpr(16):
                    v_2 = cutlass.Float32(a[j])
                    v_8 = v_2 * _helion_inv_div_0
                    acc = acc + v_8
                flash_tile_id = flash_tile_id + 1
            """
        self.assertEqual(code, _normalize(expected))

    def test_while_loop_itself_is_not_a_hoist_target(self) -> None:
        """The ``while`` body rebinds nothing the divide reads, yet the
        reciprocal must not be lifted above the ``while``: only ``for``
        loops are hoist targets, so a lone divide directly under a
        ``while`` stays a divide."""
        source = """\
            while flash_tile_id < 128:
                v_8 = v_2 / v_7
                acc = acc + v_8
                flash_tile_id = flash_tile_id + 1
            """
        self.assertEqual(_run_pass(source), _normalize(source))

    def test_distinct_divisors_are_left_alone(self) -> None:
        source = """\
            for vec_lane_0 in cutlass.range_constexpr(8):
                v_9 = v_7 + 1
                v_8 = v_7 + 2
                v_10 = v_7 / v_9
                v_12 = v_11 / v_8
                out.append(v_10 + v_12)
            """
        code = _run_pass(source)
        self.assertEqual(code, _normalize(source))
        self.assertNotIn("_helion_inv_div_", code)

    def test_single_division_is_not_rewritten(self) -> None:
        """A reciprocal costs as much as the divide it would replace."""
        source = """\
            for vec_lane_0 in cutlass.range_constexpr(8):
                v_9 = v_7 + 1
                v_10 = v_7 / v_9
                out.append(v_10)
            """
        self.assertEqual(_run_pass(source), _normalize(source))

    def test_each_shared_divisor_gets_its_own_reciprocal(self) -> None:
        source = """\
            for vec_lane_0 in cutlass.range_constexpr(8):
                v_9 = v_7 + 1
                v_8 = v_7 + 2
                v_10 = v_7 / v_9
                v_12 = v_11 / v_8
                v_13 = v_12 / v_9
                v_14 = v_13 / v_8
                out.append(v_10 + v_14)
            """
        expected = """\
            for vec_lane_0 in cutlass.range_constexpr(8):
                v_9 = v_7 + 1
                v_8 = v_7 + 2
                _helion_inv_div_0 = 1.0 / v_9
                v_10 = v_7 * _helion_inv_div_0
                _helion_inv_div_1 = 1.0 / v_8
                v_12 = v_11 * _helion_inv_div_1
                v_13 = v_12 * _helion_inv_div_0
                v_14 = v_13 * _helion_inv_div_1
                out.append(v_10 + v_14)
            """
        self.assertEqual(_run_pass(source), _normalize(expected))

    def test_rebound_divisor_starts_a_new_group(self) -> None:
        """Divisions before and after a rebind of the divisor see
        different values and must not share a reciprocal."""
        source = """\
            for vec_lane_0 in cutlass.range_constexpr(8):
                v_9 = v_7 + 1
                v_10 = v_7 / v_9
                v_9 = v_9 + 1
                v_12 = v_11 / v_9
                out.append(v_10 + v_12)
            """
        self.assertEqual(_run_pass(source), _normalize(source))

    def test_conditional_rebind_in_nested_block_starts_a_new_group(self) -> None:
        source = """\
            for vec_lane_0 in cutlass.range_constexpr(8):
                v_10 = v_7 / v_9
                if v_10 > 0:
                    v_9 = v_9 + 1
                v_12 = v_11 / v_9
                out.append(v_10 + v_12)
            """
        self.assertEqual(_run_pass(source), _normalize(source))

    def test_rebind_through_rename_alias_starts_a_new_group(self) -> None:
        """``v_9_0 = ...`` is a write to ``v_9`` once ``ast_rename``
        collapses the group, so it must end the sharing group exactly
        like a direct rebind of ``v_9``."""
        source = """\
            for vec_lane_0 in cutlass.range_constexpr(8):
                v_9 = v_7 + vec_lane_0
                v_10 = v_7 / v_9
                v_9_0 = v_9 + 1
                v_12 = v_11 / v_9
                out.append(v_10 + v_12 + v_9_0)
            """
        self.assertEqual(_run_pass(source, {"v_9_0": "v_9"}), _normalize(source))
        # Without the rename group ``v_9_0`` is an unrelated name and the
        # two quotients legitimately share ``1.0 / v_9``.
        code = _run_pass(source)
        self.assertEqual(code.count("= 1.0 / v_9"), 1)
        self.assertEqual(code.count("* _helion_inv_div_0"), 2)

    def test_loop_invariant_hoist_output_is_not_rewrapped(self) -> None:
        """When the hoist already lifted ``1.0 / di`` above the loop, the
        sharing sub-pass must neither emit a second reciprocal nor wrap
        the existing one as ``1.0 * inv``."""
        source = """\
            for lane_0 in range(4):
                v_1 = v_0 / di
                v_2 = v_1 / di
                out.append(v_2)
            """
        code = _run_pass(source)
        self.assertEqual(code.count("= 1.0 / di"), 1)
        self.assertEqual(code.count("* _helion_inv_div_0"), 2)
        self.assertNotIn("1.0 *", code)
        self.assertNotIn("_helion_inv_div_1", code)
        self.assertEqual(code.splitlines()[0], "_helion_inv_div_0 = 1.0 / di")

    def test_disable_env_covers_shared_reciprocals(self) -> None:
        with patch.dict(os.environ, {"HELION_DISABLE_HOIST_RECIP": "1"}):
            code = _run_pass(_SWIGLU_LANE_BODY)
        self.assertEqual(code, _normalize(_SWIGLU_LANE_BODY))
