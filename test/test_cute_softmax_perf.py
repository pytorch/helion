"""Tests for the CuTe softmax performance changes (P1, P2, P5, P6, P7, P11, P14).

These tests exercise the codegen paths that lifted Helion CuTe softmax
performance from ~0.45x ATen to ~0.66x ATen on (4096, *) shapes:

- P1: tile-loop vec hoist (PerThreadNDTileStrategy emits ``cute.arch.load``
  with ``ir.VectorType`` so masked reductions can vectorize).
- P2/P5: ``fuse_two_pass_loads`` extended alias map to canonicalize
  per-sweep ``mask_<N>``, ``lane_base_<N>``, ``vec_lane_<N>``, and
  ``_tile_unroll_vec_<N>_<M>`` so the fuser fires across sweeps.
- P6: ``CuteBackend.launcher_keyword_args`` raises ``BackendUnsupported``
  when the launcher would silently truncate joint thread count below
  what codegen committed to (was producing nan-valued kernels that
  showed up as "fast" to the autotuner).
- P7: ``CuteTileVecWarpReduceHeuristic`` autotuner heuristic seeds
  ``block_sizes=[1, V*32]``, ``num_threads=[0, 32]``, ``cute_vector_widths=[1, V]``
  for softmax-shaped reduction kernels with no rolled reduction.
- P11: ``hoist_warp_reduce`` AST pass moves ``cute.arch.warp_reduction_*``
  calls out of constexpr ``range_constexpr(V)`` loops so the reduce
  runs once per outer iter instead of V times.
- P14: ``merge_sibling_v_loops`` AST pass caches the per-V-lane scalar
  shared across two consecutive constexpr V-loops (online softmax max +
  sum passes) into a small ``cute.make_rmem_tensor(V, fp32)`` so V-loop 2
  reads the cached fp32 value rather than re-bitcasting from the
  underlying U16 vec load.  Same pass also elides the redundant
  ``Float32(Float16(warp_reduction(...)))`` round-trip on warp-reduce
  results.
"""

from __future__ import annotations

import os
import re

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.exc import BackendUnsupported
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute")
def softmax_two_pass_kernel(x: torch.Tensor) -> torch.Tensor:
    """Helion two-pass softmax — the kernel under test.

    Mirrors ``examples/softmax.py::softmax_two_pass`` so the perf tests
    here exercise the same codegen path as the bench. Inlined to avoid
    the example's ``softmax_tritonbench`` wrapper.
    """
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


# Cache of compiled softmax artifacts shared across tests.  Many tests in
# this file pin different codegen properties of the byte-identical kernel
# (same shape, config, and env — the autouse fixture below pins
# ``HELION_DISABLE_ONLINE_TO_3PASS=1`` for every class that uses this
# cache), so paying the full CuTe compile once per distinct artifact
# instead of once per test keeps the assertions while cutting most of the
# file's wall time.  Keyed by (N, block_n, vector_width); M is fixed at
# 4096 (it is never the property under test).
_ARTIFACT_CACHE: dict[tuple[int, int, int], tuple[str, torch.Tensor, torch.Tensor]] = {}


def _pipe_shadow_var_count(code: str, kind: str) -> int:
    """Count ``_pipe_<kind>_N = `` occurrences (prologue + per-iter
    prefetch assignments, so 2 per pipelined load site)."""
    return len(re.findall(rf"_pipe_{kind}_\d+ = ", code))


def _softmax_artifact(
    n: int, *, block_n: int = 128, vec_width: int = 4
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Compile ``softmax_two_pass_kernel`` on (4096, n) fp16 once and
    return ``(code, out, ref)`` for shared assertions."""
    key = (n, block_n, vec_width)
    if key not in _ARTIFACT_CACHE:
        x = torch.randn(4096, n, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[1, block_n],
            num_threads=[0, 32],
            cute_vector_widths=[1, vec_width],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        _ARTIFACT_CACHE[key] = (code, out, ref)
    return _ARTIFACT_CACHE[key]


@pytest.fixture(autouse=True)
def _disable_online_to_3pass(request):
    """The 2-loop softmax perf tests in this file pin codegen details of
    the ORIGINAL online-merge form.  P21's online->3-pass AST rewrite
    rewrites the kernel into a 3-loop form before tracing, which by
    design changes the codegen those tests inspect.  This autouse
    fixture sets ``HELION_DISABLE_ONLINE_TO_3PASS=1`` for every test in
    the file EXCEPT those in ``TestCuteOnlineTo3PassRewrite`` (the
    rewrite-specific tests at the bottom of this file).
    """
    cls = request.cls
    if cls is not None and cls.__name__ == "TestCuteOnlineTo3PassRewrite":
        yield
        return
    old = os.environ.get("HELION_DISABLE_ONLINE_TO_3PASS")
    os.environ["HELION_DISABLE_ONLINE_TO_3PASS"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("HELION_DISABLE_ONLINE_TO_3PASS", None)
        else:
            os.environ["HELION_DISABLE_ONLINE_TO_3PASS"] = old


@onlyBackends(["cute"])
class TestCuteSoftmaxVecHoist(TestCase):
    """P1: tile-loop vec hoist for masked reductions."""

    def test_vec_hoist_fires_at_v4_fp16(self) -> None:
        """When ``cute_vector_widths=[1, 4]`` is set on a fp16 reduction
        kernel and the inner tile aligns with V, the codegen must emit a
        ``cute.arch.load(..., ir.VectorType.get([4], cutlass.Uint16.mlir_type))``
        — the new tile-loop vec hoist path that replaces 4 scalar fp16
        loads with one 8-byte vec load per thread per iter.
        """
        code, out, ref = _softmax_artifact(6400)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The vec hoist target var name is _tile_unroll_vec_*.
        self.assertIn("_tile_unroll_vec_", code)
        # The hoisted load wraps the actual ptr in an IfExp for
        # in-bounds guard; the VectorType arg pins the V/dtype.
        self.assertIn(
            "ir.VectorType.get([4], cutlass.Uint16.mlir_type)",
            code,
        )
        # And the per-V-lane extract is a bitcast back to Float16.
        self.assertIn("bitcast(cutlass.Float16)", code)

    def test_vec_hoist_v8_uses_4plus4_split(self) -> None:
        """V=8 fp16/bf16 cannot use a single ``cute.arch.load`` (the CuTe
        DSL's ``nvvm.load.ext`` ICEs at V=8), so the codegen lowers it as
        TWO back-to-back ``cute.arch.load(..., V=4)`` calls (covering vec
        lanes 0-3 and 4-7).  The SASS scheduler is free to overlap the
        two LDGs, so per-thread bytes-per-load still hit the full
        LDG.128 (16 bytes) without invoking the DSL bug.
        """
        x = torch.randn(4096, 8192, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[1, 256],
            num_threads=[0, 32],
            cute_vector_widths=[1, 8],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The single-V=8 load form is still NOT emitted (would ICE).
        self.assertNotIn("ir.VectorType.get([8]", code)
        # The split-2 path emits TWO V=4 vec loads per outer-lane iter
        # with the ``_a`` / ``_b`` suffix on the hoist var names.  When
        # the P18 load-pipeline pass fires, the ``_a`` load assignment
        # text gets rewritten to ``_tile_unroll_vec_1_0_a = _pipe_load_*``
        # with the actual ``cute.arch.load`` hoisted into the prologue
        # and an explicit per-iter prefetch; the ``_b`` load is left as
        # an inline ``cute.arch.load`` call (it's not the first inner
        # load).  Accept either form.
        self.assertTrue(
            "_tile_unroll_vec_1_0_a = cute.arch.load(" in code
            or "_tile_unroll_vec_1_0_a = _pipe_load_" in code,
            "expected _a hoist var either as inline load or pipeline snapshot",
        )
        self.assertIn("_tile_unroll_vec_1_0_b = cute.arch.load(", code)
        # Both halves use V=4.
        self.assertGreaterEqual(
            code.count("ir.VectorType.get([4], cutlass.Uint16.mlir_type)"),
            2,
        )
        # The constexpr V-loop now runs 8 iters (not 4).
        self.assertIn("cutlass.range_constexpr(8)", code)
        # The per-vec-lane extract uses the if-else split selector so the
        # constexpr loop unroller picks the right half per iter.
        self.assertIn("if vec_lane_1 < 4 else", code)

    def test_vec_hoist_bf16(self) -> None:
        """The vec hoist must also fire for bf16 (also a uint16-backed type)."""
        x = torch.randn(4096, 6400, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertIn("ir.VectorType.get([4]", code)
        self.assertIn("cutlass.BFloat16", code)

    def test_scalar_load_when_vec_width_is_one(self) -> None:
        """When V=1 the vec hoist must NOT fire — the codegen falls back to
        the scalar load path so the change is opt-in.  (N is not the
        property here; shares the V=1 artifact with the P11/P14 no-op
        gate tests.)
        """
        code, out, ref = _softmax_artifact(4096, vec_width=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertNotIn("_tile_unroll_vec_", code)


@onlyBackends(["cute"])
class TestCuteFuseTwoPassAlias(TestCase):
    """P2/P5: alias map covers mask/lane_base/vec_lane/_tile_unroll_vec names."""

    def test_fuser_fires_with_vec_hoist(self) -> None:
        """When the vec hoist runs, the two sweeps' loads use names like
        ``lane_base_1`` vs ``lane_base_2``, ``vec_lane_1`` vs ``vec_lane_2``,
        and ``_tile_unroll_vec_1_0`` vs ``_tile_unroll_vec_1_1``. P5's
        alias map canonicalizes these so the fuser matches and emits one
        ``_fuse_cache_*`` fragment instead of re-reading from gmem.

        The cache_size cap is 64; for (4096, 1024) with block_size 128 we
        have trip=8 and lane_trip=1 → cache_size=8 → fits.
        """
        x = torch.randn(4096, 1024, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Cache fragment allocated.
        self.assertIn("_fuse_cache_", code)
        # When the P18 load-pipeline pass is OFF the consume sweep just
        # reads from cache so the kernel has 1 ``cute.arch.load`` total.
        # With pipelining ON the reduce sweep's load is hoisted into a
        # prologue + per-iter prefetch (2 load sites), so the total is
        # 2 — but it's STILL the case that the consume sweep emits NO
        # load (cache hit).  Both forms are correct.  Verify the consume
        # sweep is cache-only by looking at the inner consume loop.
        load_count = code.count("cute.arch.load(")
        self.assertTrue(
            load_count in {1, 2},
            f"expected 1 or 2 cute.arch.load sites, got {load_count}",
        )
        # The consume sweep (second top-level ``for tile_offset`` loop)
        # must not contain a gmem load.
        consume_marker = "for tile_offset_2 in range"
        consume_start = code.rfind(consume_marker)
        self.assertGreater(consume_start, 0)
        self.assertNotIn(
            "cute.arch.load(",
            code[consume_start:],
            "consume sweep must read from _fuse_cache_, not gmem",
        )

    # The trip>64 fuser-bail case on (4096, 12672) is pinned in
    # ``TestCuteCanonicalSoftmaxArtifact`` (shared compile).


@onlyBackends(["cute"])
class TestCuteHoistWarpReduce(TestCase):
    """P11: hoist warp_reduction_* out of constexpr V-loops."""

    def test_warp_reduce_hoisted_out_of_v_loop(self) -> None:
        """The new ``hoist_warp_reduce`` AST pass must remove the
        ``cute.arch.warp_reduction_max`` and ``cute.arch.warp_reduction_sum``
        calls from inside the ``range_constexpr(V)`` loop body.

        Before the pass, the V-loop body contained V calls to each warp
        reduce (one per vec lane). After the pass, the V-loop body has
        zero warp reduces — they live OUTSIDE the loop with a per-thread
        local fold first.
        """
        code, out, ref = _softmax_artifact(6400)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Both reduces still get emitted (correctness), just outside the V-loop.
        self.assertIn("warp_reduction_max", code)
        self.assertIn("warp_reduction_sum", code)
        # Sanity: the V-loop is still present.
        self.assertIn("cutlass.range_constexpr(", code)
        # The hoist creates a fresh accumulator var visible in the body.
        # The pass tags accumulators ``_helion_vfold_acc_*`` (see
        # helion/_compiler/cute/hoist_warp_reduce.py).
        self.assertIn("_helion_vfold_acc_", code)

    def test_no_hoist_when_v_loop_absent(self) -> None:
        """When V=1 there's no constexpr V-loop, so the hoist pass must be
        a no-op — the reduce stays where the strategy emitted it.
        """
        code, out, ref = _softmax_artifact(4096, vec_width=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # No hoist accumulator names (no V-loop to hoist out of).
        self.assertNotIn("_helion_vfold_acc_", code)


@onlyBackends(["cute"])
class TestCuteMergeSiblingVLoops(TestCase):
    """P14: ``merge_sibling_v_loops`` AST pass caches the per-V-lane
    scalar shared across two consecutive constexpr V-loops.

    For online softmax's max + sum passes inside one outer tile iter,
    both V-loops read the SAME ``bitcast(_tile_unroll_vec_*[v])`` value.
    The pass introduces a ``cute.make_rmem_tensor(V, Float32)`` cache so
    V-loop 1 stores fp32 there once, V-loop 2 reads back instead of
    re-running the bitcast/cast chain.  Same pass also elides the
    redundant Float16(...) wrap on warp_reduction results when the
    downstream code immediately re-promotes to fp32.
    """

    # The merge-fires and cast-elision pins on (4096, 12672) live in
    # ``TestCuteCanonicalSoftmaxArtifact`` (shared compile).

    def test_no_merge_when_v_loop_absent(self) -> None:
        """When V=1 there's no constexpr V-loop, so the merge pass must
        be a no-op — no cache fragment should be emitted.
        """
        code, out, ref = _softmax_artifact(4096, vec_width=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertNotIn("_helion_vmerge_cache_", code)


@onlyBackends(["cute"])
class TestCuteHoistLoopInvariantRecip(TestCase):
    """P16: ``hoist_loop_invariant_recips`` AST pass hoists ``x / scalar``
    divisions out of inner loops when the divisor is loop-invariant.

    Softmax's second pass computes ``out[k] = exp(x[k] - mi) / di`` where
    ``di`` is a per-row scalar.  Each inner tile iteration does a
    floating-point divide; B200 fp32 divide is ~22 cycles vs ~2 for
    multiply.  The pass turns the divide into ``inv = 1.0 / di`` hoisted
    above the loop + ``x * inv`` inside, yielding +20% wall-clock on
    (4096, 12672) fp16 and even larger gains on N <= 4096 shapes.

    The pass handles SSA-style alias chains
    (``di_copy = di; di_copy_0 = di_copy; out = x / di_copy_0``) by
    transitively walking the assignments to find the loop-EXTERNAL root
    divisor name, so the hoisted reciprocal references a name visible
    outside the loop.
    """

    # The recip-hoist-fires pin on (4096, 12672) lives in
    # ``TestCuteCanonicalSoftmaxArtifact`` (shared compile).

    def test_recip_hoist_does_not_fire_on_loop_dependent_divisor(self) -> None:
        """When the divisor changes per-iteration (e.g. ``local_sum``
        computed inside the loop), the pass must NOT hoist it.  Pinning
        this guards against accidentally hoisting genuinely loop-dependent
        divides which would produce wrong values.
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
        code, out = code_and_output(
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
        experimentation.  The two-pass kernel must compile + run
        correctly when the pass is disabled, just slower.
        """
        import os

        old = os.environ.get("HELION_DISABLE_HOIST_RECIP")
        os.environ["HELION_DISABLE_HOIST_RECIP"] = "1"
        try:
            x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
            code, out = code_and_output(
                softmax_two_pass_kernel,
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
class TestCuteCanonicalSoftmaxArtifact(TestCase):
    """All codegen pins on the canonical (4096, 12672) fp16 artifact
    (``block_sizes=[1, 128]``, ``num_threads=[0, 32]``,
    ``cute_vector_widths=[1, 4]``), compiled ONCE via
    ``_softmax_artifact`` and shared across the tests below.

    Previously each of these pins was its own test that recompiled the
    byte-identical kernel (11 full CuTe compiles).  The properties are
    unchanged — grouped here by pass:

    * P2/P5 fuser bail: trip = 99 > cache cap 64, so both sweeps load
      from gmem (guards the P8 register-pressure regression).
    * P14 ``merge_sibling_v_loops``: vmerge cache + double-cast elision.
    * P16 reciprocal hoist: ``1.0 / di`` hoisted, inner divide becomes
      a multiply.
    * P17 extended hoists: alias DCE (``*_copy`` chains inlined),
      outer-in reciprocal walk (no ``1.0 * _helion_inv_div_`` cascade),
      FMA-friendly scale hoists in both the consume loop and the reduce
      V-loop, dead-Sub DCE, and the post-rename invariance
      canonicalization that keeps ``mi`` loop-VARIANT in the reduce
      loop (silent-miscompile guard).
    * P18 load pipeline: both sweeps pipelined (prologue snapshot +
      per-iter prefetch).
    """

    def test_numerics_and_fuser_bails_above_cache_cap(self) -> None:
        """End-to-end correctness of the canonical artifact, plus the
        P2/P5 fuser gate: for (4096, 12672) with block_size 128 the
        trip count is 99 > the 64-entry cache cap, so the fuser must
        bail and BOTH sweeps must load from gmem.
        """
        code, out, ref = _softmax_artifact(12672)
        # Tight tolerance — this also guards the P17 invariance
        # canonicalization silent-miscompile (the bench would still
        # "look fast" with wrong outputs if we regressed there).
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Cache fragment NOT allocated (fuser bails on cache_size > 64).
        self.assertNotIn("_fuse_cache_", code)
        # When the P18 load-pipeline pass is OFF the kernel has exactly
        # 2 ``cute.arch.load`` calls (one per sweep).  With pipelining
        # ON each load site is hoisted into a prologue and a per-iter
        # prefetch, so the total is 4 (2 prologue + 2 inner-loop
        # prefetch).  Both forms are correct; assert at least 2.
        self.assertGreaterEqual(code.count("cute.arch.load("), 2)
        # Each sweep must contain at least one gmem load (either
        # inline or as a per-iter prefetch).
        first_sweep_marker = "for tile_offset_2 in range"
        first_sweep_start = code.find(first_sweep_marker)
        second_sweep_start = code.find(first_sweep_marker, first_sweep_start + 1)
        self.assertGreater(second_sweep_start, first_sweep_start)
        first_sweep = code[first_sweep_start:second_sweep_start]
        second_sweep = code[second_sweep_start:]
        self.assertIn("cute.arch.load(", first_sweep)
        self.assertIn("cute.arch.load(", second_sweep)

    def test_vmerge_recip_and_fma_hoists(self) -> None:
        """P14 + P16 + P17 codegen pins on the shared artifact."""
        code, _, _ = _softmax_artifact(12672)
        # P14: vmerge cache fragment allocated, written in V-loop 1 at
        # the vec_lane index, and allocated as Float32 (promotes from
        # fp16 so V-loop 2 doesn't need the redundant Float32 cast).
        self.assertIn("_helion_vmerge_cache_", code)
        self.assertIn("_helion_vmerge_cache_0[vec_lane_1]", code)
        self.assertIn(
            "_helion_vmerge_cache_0 = cute.make_rmem_tensor(4, cutlass.Float32)",
            code,
        )
        # P14 double-cast peephole: ``local_amax =
        # Float16(warp_reduction_max(...))`` must have been elided to
        # just ``local_amax = warp_reduction_max(...)`` (the hoist pass
        # already promoted the V-fold acc to fp32, so the Float16 wrap
        # was a no-op cycle).
        self.assertNotIn(
            "local_amax = cutlass.Float16(cute.arch.warp_reduction_max",
            code,
        )
        self.assertIn(
            "local_amax = cute.arch.warp_reduction_max",
            code,
        )
        # P16: the hoisted reciprocal references the loop-external root
        # name ``di`` (not an inside-loop ``di_copy_*`` alias) and the
        # inner divide is rewritten to a multiply.
        self.assertIn("_helion_inv_div_", code)
        self.assertIn("= 1.0 / di", code)
        self.assertIn("* _helion_inv_div_", code)
        self.assertNotIn("/ di_copy_1_0", code)
        # P17 outer-in walk: no ``_helion_inv_div_N = 1.0 *
        # _helion_inv_div_{N+1}`` cascade, exactly ONE reciprocal hoist
        # for the consume sweep's ``1.0/di``.
        self.assertEqual(code.count("1.0 * _helion_inv_div_"), 0)
        self.assertEqual(code.count("= 1.0 / di"), 1)
        # P17 alias DCE: neither ``mi_copy`` nor ``di_copy`` appears —
        # both were SSA snapshots; with the snapshot removed the inner
        # use reads the root directly.
        self.assertEqual(code.count("_copy"), 0)
        # P17 FMA scale hoist above the consume loop: ``mi * log2(e)``
        # hoisted, inner expression rewritten to the FMA-friendly form
        # (the outer redundant ``cutlass.Float32(v_9)`` cast is
        # stripped because ``v_9`` is already fp32).
        self.assertIn("_helion_scaled_", code)
        self.assertIn("= mi * 1.4426950408889634", code)
        self.assertIn("cute.math.exp2(v_9 * 1.4426950408889634 - _helion_scaled_", code)
        # P17 FMA scale hoist in the reduce V-loop: ``v_1`` (the
        # new-max from the warp_reduce before the V-loop) is
        # V-loop-invariant, so the hoist fires there too.
        self.assertIn("= v_1 * 1.4426950408889634", code)
        self.assertIn("cute.math.exp2(v_5 * 1.4426950408889634 - _helion_scaled_", code)
        # P17 DCE: the consume-loop ``v_10 = v_9 - mi`` and the reduce
        # V-loop ``v_6 = v_5 - v_1`` are dead after the FMA hoists and
        # must be removed; statements that ARE read (``v_4`` feeds
        # ``di``) must survive.
        self.assertNotIn("v_10 = v_9 - mi", code)
        self.assertNotIn("v_6 = v_5 - v_1", code)
        self.assertIn("di = v_4 + sum_1", code)
        # P17 invariance canonicalization: the ``mi`` scale hoist for
        # the consume sweep lives BETWEEN the two outer for-loops
        # (after the reduce loop finishes mutating ``mi`` via the
        # rename ``v_1_0 -> mi``), not before the reduce loop — else
        # it would capture the stale initial ``-inf`` value of ``mi``.
        reduce_end = code.find("mi = v_1\n")
        self.assertGreaterEqual(reduce_end, 0)
        scaled_consume = code.find("= mi * 1.4426950408889634")
        self.assertGreater(scaled_consume, reduce_end)

    def test_load_pipeline_fires_on_both_sweeps(self) -> None:
        """P18: both the reduce and consume sweeps match the (single
        inner-lane loop, single load) shape so the pass rewrites both.
        Each rewrite emits a prologue snapshot and a per-iter prefetch
        — 2 ``_pipe_lane_base_*`` assigns and 2 ``_pipe_load_*``
        assigns per pipelined site, for 4 total of each.
        """
        code, _, _ = _softmax_artifact(12672)
        self.assertEqual(_pipe_shadow_var_count(code, "lane_base"), 4)
        self.assertEqual(_pipe_shadow_var_count(code, "load"), 4)
        # The snapshot ``lane_base_<N> = _pipe_lane_base_<M>`` form must
        # appear inside the loop body — that's the substitution that
        # gives the SASS scheduler one full iter of slack to issue the
        # next load.
        self.assertRegex(code, r"lane_base_\d+ = _pipe_lane_base_\d+")
        self.assertRegex(code, r"_tile_unroll_vec_\d+_\d+ = _pipe_load_\d+")


@onlyBackends(["cute"])
class TestCuteLoadPipelineP18(TestCase):
    """P18: software-pipeline the per-iteration vec load by one stage.

    The pass pre-issues the iter 0 vec load above the inner loop and
    inside the body issues the NEXT iter's load BEFORE the current
    iter's compute runs.  The SASS scheduler then has multiple
    ``ld.global`` instructions in flight per warp, hiding HBM round-trip
    latency.  Measured ~18% drop in
    ``smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio``
    on the (4096, 12672) softmax shape (NCU) and +10-20% wall-clock
    GB/s gain on most softmax shapes.
    """

    # The pipeline-fires pin on (4096, 12672) lives in
    # ``TestCuteCanonicalSoftmaxArtifact`` (shared compile).

    def test_pipeline_skips_when_lane_reps_greater_than_one(self) -> None:
        """V=4 with block_size=256 forces the inner lane loop to
        ``for lane in range(2):`` (block=256, V=4 → 2 lanes per
        thread).  The pipeline transform must SKIP this case: the
        per-iter prefetch advances ``_pipe_lane_base`` to the next
        outer iter but uses the CURRENT ``lane`` index, which would
        corrupt the snapshot read by the NEXT lane index in the same
        outer iter.  Verify the rewrite did not fire.
        """
        x = torch.randn(4096, 8192, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[1, 256],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # lane_reps == 2, so the pass bails -- no _pipe_* vars.
        self.assertNotIn("_pipe_lane_base_", code)
        self.assertNotIn("_pipe_load_", code)

    def test_pipeline_disable_env(self) -> None:
        """``HELION_DISABLE_LOAD_PIPELINE=1`` disables the pass for
        experimentation.  The kernel must still compile and produce
        correct output without the pipeline rewrite.
        """
        import os

        old = os.environ.get("HELION_DISABLE_LOAD_PIPELINE")
        os.environ["HELION_DISABLE_LOAD_PIPELINE"] = "1"
        try:
            x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
            code, out = code_and_output(
                softmax_two_pass_kernel,
                (x,),
                block_sizes=[1, 128],
                num_threads=[0, 32],
                cute_vector_widths=[1, 4],
            )
            ref = torch.nn.functional.softmax(x, dim=1)
            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
            # No pipeline vars when the pass is disabled.
            self.assertNotIn("_pipe_lane_base_", code)
            self.assertNotIn("_pipe_load_", code)
        finally:
            if old is None:
                os.environ.pop("HELION_DISABLE_LOAD_PIPELINE", None)
            else:
                os.environ["HELION_DISABLE_LOAD_PIPELINE"] = old


@onlyBackends(["cute"])
class TestCuteLoadPipelineCarriedOnlyP19(TestCase):
    """P19: opt-in loop-carried-scalar-write gate for the load
    pipeline pass.

    The default behavior (P18) pipelines BOTH sweeps of two-pass
    softmax: the reduce sweep (which writes the ``mi``/``di``
    accumulator) and the consume sweep (a pure stream-store).  In
    microbench across the standard 4 shapes (5248/10240/12672/16384)
    the consume-sweep pipeline still measures faster at the GPU
    level so the default is to pipeline both.

    However, autotuner sweeps that want to explore a more
    conservative alternative (only pipeline loops with an
    inter-iter scalar dep) can set
    ``HELION_LOAD_PIPELINE_CARRIED_ONLY=1`` to enable the gate.
    These tests pin both the gate semantics and the env-var plumbing.
    """

    def setUp(self) -> None:
        super().setUp()
        self._env_to_restore: dict[str, str | None] = {}

    def tearDown(self) -> None:
        for name, prev in self._env_to_restore.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev
        super().tearDown()

    def _set_env(self, name: str, value: str | None) -> None:
        """Restore-aware env var setter."""
        if name not in self._env_to_restore:
            self._env_to_restore[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def test_carried_only_pipelines_reduce_sweep_but_not_consume(self) -> None:
        """With the gate enabled, the reduce sweep (which writes ``mi``
        and ``di`` — both defined in the outer function-body scope
        before the loop) MUST still be pipelined, while the consume
        sweep (which only stores into ``out[k]`` and never writes a
        scalar accumulator in the outer scope) MUST NOT be.

        The reduce-sweep pipeline writes ``_pipe_lane_base_0`` and
        ``_pipe_load_0``; the consume sweep would have written
        ``_pipe_lane_base_1`` and ``_pipe_load_1`` under the default
        behavior.  Under the gate, only the ``_0`` indices appear.
        (Was two tests compiling the identical artifact twice.)
        """
        self._set_env("HELION_LOAD_PIPELINE_CARRIED_ONLY", "1")
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The reduce sweep MUST still be pipelined — there's a
        # loop-carried scalar write to ``mi`` (pre-rename: ``v_1_0
        # = v_1``) so the gate fires positive.  ONLY the reduce-sweep
        # pipe (index 0) — no index 1.
        self.assertIn("_pipe_lane_base_", code)
        self.assertIn("_pipe_load_0", code)
        self.assertNotIn("_pipe_load_1", code)
        self.assertNotIn("_pipe_lane_base_1", code)
        # The per-iter snapshot ``lane_base_<N> = _pipe_lane_base_<M>``
        # form appears inside the reduce loop body.
        self.assertRegex(code, r"lane_base_\d+ = _pipe_lane_base_\d+")
        # And the consume sweep retains its original inline
        # ``cute.arch.load`` (no snapshot/prefetch rewrite).  The
        # consume sweep's load assigns to ``_tile_unroll_vec_1_1``.
        self.assertIn("_tile_unroll_vec_1_1 = cute.arch.load(", code)

    def test_pipeline_all_overrides_carried_only(self) -> None:
        """``HELION_LOAD_PIPELINE_ALL=1`` must explicitly override the
        gate even when ``HELION_LOAD_PIPELINE_CARRIED_ONLY`` is also
        set, restoring the pre-P19 "pipeline every qualifying loop"
        behavior.
        """
        self._set_env("HELION_LOAD_PIPELINE_CARRIED_ONLY", "1")
        self._set_env("HELION_LOAD_PIPELINE_ALL", "1")
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, _ = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        # Both sweeps pipelined → both _pipe_load_0 and _pipe_load_1
        # appear (4 total ``_pipe_load_*`` write sites, same as the
        # P18 default).
        self.assertIn("_pipe_load_0", code)
        self.assertIn("_pipe_load_1", code)


@onlyBackends(["cute"])
class TestCuteThreadBudgetRejection(TestCase):
    """P6: reject configs where the launcher would silently truncate
    joint thread count below what codegen committed to.
    """

    def test_joint_thread_overflow_rejected(self) -> None:
        """For ``block_sizes=[8, 1024], num_threads=[0, 256]`` the codegen
        commits to a 8 * 256 = 2048-thread layout, but the launcher caps
        at MAX_THREADS_PER_BLOCK = 1024 → axis is silently truncated and
        the kernel writes nan. P6's guard raises ``BackendUnsupported``
        instead.
        """
        x = torch.randn(4096, 1024, device=DEVICE, dtype=HALF_DTYPE)
        with pytest.raises(BackendUnsupported):
            code_and_output(
                softmax_two_pass_kernel,
                (x,),
                block_sizes=[8, 1024],
                num_threads=[0, 256],
                cute_vector_widths=[1, 4],
            )

    def test_in_budget_multi_row_passes(self) -> None:
        """A multi-row config that DOES fit in 1024 threads must still
        compile & run cleanly (the rejection must be precise).
        """
        x = torch.randn(4096, 256, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[2, 256],
            num_threads=[1, 32],  # 2 * 32 = 64 threads — within budget
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@onlyBackends(["cute"])
class TestCuteWarpReduceHeuristic(TestCase):
    """P7: ``CuteTileVecWarpReduceHeuristic`` seeds a config that the
    autotuner's quick-effort path can actually pick for fp16 softmax.
    """

    def test_warp_reduce_heuristic_seed_is_picked(self) -> None:
        """The seed config block_sizes=[1, V*32], num_threads=[0, 32],
        cute_vector_widths=[1, V] must produce a working kernel that
        uses ``cute.arch.warp_reduction_*`` and not the shared-memory
        two-stage reduce.
        """
        code, out, ref = _softmax_artifact(6400)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertIn("cute.arch.warp_reduction_max", code)
        self.assertIn("cute.arch.warp_reduction_sum", code)
        # Block of 32 threads on the reduction axis — exactly one warp.
        self.assertIn("block=(32, 1, 1)", code)
        # Should NOT use the shared-memory two-stage reduce at this size.
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)

    def test_warp_reduce_heuristic_class_is_registered(self) -> None:
        """The CuteTileVecWarpReduceHeuristic class must be discoverable
        and registered for the cute backend so the autotuner can use it.
        """
        from helion._compiler.autotuner_heuristics import HEURISTICS_BY_BACKEND
        from helion._compiler.autotuner_heuristics.cute import (
            CuteTileVecWarpReduceHeuristic,
        )

        self.assertIn(
            CuteTileVecWarpReduceHeuristic, HEURISTICS_BY_BACKEND.get("cute", ())
        )


@onlyBackends(["cute"])
class TestCuteSoftmaxCorrectness(TestCase):
    """End-to-end correctness across the configs the autotuner actually
    explores, to guard against silent miscompiles introduced by the new
    codegen paths.
    """

    def _check_softmax(self, shape: tuple[int, int], cfg: dict[str, object]) -> None:
        x = torch.randn(*shape, device=DEVICE, dtype=HALF_DTYPE)
        _, out = code_and_output(softmax_two_pass_kernel, (x,), **cfg)
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_softmax_4096x256(self) -> None:
        self._check_softmax(
            (4096, 256),
            {
                "block_sizes": [1, 32],
                "num_threads": [0, 16],
                "cute_vector_widths": [1, 2],
            },
        )

    def test_softmax_4096x6400(self) -> None:
        _, out, ref = _softmax_artifact(6400)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    # (4096, 12672) numerics are pinned by
    # ``TestCuteCanonicalSoftmaxArtifact`` and (4096, 8192) with
    # ``block_sizes=[1, 256]`` (the power-of-two divisible case) by
    # ``TestCuteLoadPipelineP18.test_pipeline_skips_when_lane_reps_greater_than_one``
    # — both on the identical artifact these entries used to recompile.


@onlyBackends(["cute"])
class TestCuteMultiRowInvestigation(TestCase):
    """P13: multi-row CTAs were investigated as a way to lift small-N
    softmax shapes (e.g. (4096, 256)) toward ATen.  The investigation
    concluded that within the current Helion CuTe launcher architecture,
    multi-row CTAs do NOT improve wall-clock perf — the bottleneck is
    the per-launch Python overhead in ``default_cute_launcher`` (~25us
    per call), not the per-CTA scheduling overhead.

    Findings (CUDA_VISIBLE_DEVICES=7, B200, fp16):

    * Single-row (autotuned: ``block_sizes=[1, 128]``, ``num_threads=[0, 32]``,
      ``cute_vector_widths=[1, 4]``): 31us wall / 134 GB/s
      (4.3us GPU kernel time, ~25us Python launcher overhead).
    * Multi-row 2/4/8/16/32 rows + serialized M (``num_threads=[1, 32]``):
      31-36us wall / 116-133 GB/s — no improvement.
    * Multi-row 8 rows + threaded M (``num_threads=[0, 32]``,
      shared-mem reduce): 48us wall / 87 GB/s — WORSE due to
      ``_cute_grouped_reduce_shared_two_stage`` cost.
    * Hand-coded warp-per-row kernel with ``block=(32, M_block, 1)``:
      kernel time drops to ~2us (NCU) but wall-clock stays at ~34us
      because the Helion ``_CompiledCuteLauncher.__call__`` +
      cutlass DSL ``generate_execution_args`` dominate (~30us per call).

    The "warp-per-row" layout (one CUDA warp per row, with
    ``thread_idx[0]`` = lane in row, ``thread_idx[1]`` = row index)
    would require structurally swapping the thread-axis assignment
    between the M-grid loop and the inner N-tile loop in
    ``PerThreadNDTileStrategy``.  Even if implemented, the wall-clock gain
    is bounded by the launcher overhead.

    Future work to close the (4096, 256) gap should target either:
    (a) reducing the per-call Helion CuTe launcher overhead, or
    (b) using CUDA graphs / static launcher (not enabled in the
    autotuner bench mode), or
    (c) batching multiple shapes into one launch (kernel-level fusion).

    These tests pin the multi-row configs as compilable / correct so
    that a future architectural change does not silently regress them.
    """

    def test_multi_row_serial_8_compiles_and_is_correct(self) -> None:
        """``block_sizes=[8, 128], num_threads=[1, 32]`` — M serialized
        via a ``for lane_0 in range(8)`` loop inside the kernel.  This
        is the path the prompt's example targeted; it does compile and
        produce correct output, but is slower than single-row.
        """
        x = torch.randn(4096, 128, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[8, 128],
            num_threads=[1, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The M-axis is serialized — a ``for lane_0 in range(8):`` loop
        # wraps the per-row body. (8 = block_sizes[0].)
        self.assertIn("for lane_0 in range(8):", code)
        # Warp-reduce IS used (since the M-axis doesn't contribute to
        # group_span — it's serialized, so group_span = reduce_extent =
        # 32 → warp-reduce path fires).
        self.assertIn("cute.arch.warp_reduction_max", code)
        # The launch is still a single-warp block; only the grid count
        # is reduced (4096 / 8 = 512 CTAs vs 4096).
        self.assertIn("block=(32, 1, 1)", code)

    def test_multi_row_threaded_8_uses_warp_per_row(self) -> None:
        """``block_sizes=[8, 128], num_threads=[0, 32]`` — M is threaded.

        With the warp-per-row layout (P15), the codegen swaps the
        thread-axis assignment so N (the reduction axis) lands on
        ``thread_idx[0]`` (32 contiguous threads = one warp per row)
        and M lands on ``thread_idx[1]`` (warp index = row index).
        Each warp's per-row reduction uses ``_cute_grouped_reduce_warp``
        with ``pre=1, group_span=32`` (one warp shuffle per warp), NOT
        the cross-warp ``_cute_grouped_reduce_shared_two_stage`` path.

        The launch becomes ``block=(32, 8, 1)`` (was ``(8, 32, 1)``
        before P15) and the joint thread count is 8 × 32 = 256, giving
        8 warps per CTA for higher occupancy on softmax-shaped kernels.
        """
        x = torch.randn(4096, 128, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[8, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Warp-per-row launch: N (the reduction axis with 32 threads)
        # lands on axis 0, M (8 rows) lands on axis 1.
        self.assertIn("block=(32, 8, 1)", code)
        # Each warp reduces its own row via _cute_grouped_reduce_warp
        # — NO cross-warp SMEM reduce.
        self.assertIn("_cute_grouped_reduce_warp", code)
        self.assertIn("pre=1, group_span=32", code)
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)

    def test_warp_per_row_axis_swap_emits_warp_reduce(self) -> None:
        """P15: when M is threaded (``num_threads=[0, 32]``) the warp-per-row
        plan swaps the thread-axis assignment so:

        * N (inner reduction axis) lands on ``thread_idx[0]``
        * M (outer grid row axis) lands on ``thread_idx[1]``

        And the reduction dispatcher picks the per-warp
        ``_cute_grouped_reduce_warp`` path with ``pre=1, group_span=32``
        (each warp does ONE warp shuffle for its own row independently)
        instead of ``_cute_grouped_reduce_shared_two_stage`` (the
        cross-warp SMEM path that would dominate without the axis swap).
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[2, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The launch is ``block=(32, 2, 1)`` — N on axis 0 (32 threads
        # per warp = one row), M on axis 1 (2 warps = 2 rows).
        self.assertIn("block=(32, 2, 1)", code)
        # M uses ``thread_idx[1]`` for the row index.
        self.assertIn(
            "indices_0 = tile_offset_0 + cutlass.Int32(cute.arch.thread_idx()[1])",
            code,
        )
        # N uses ``thread_idx[0]`` for the lane (32 contiguous lanes).
        self.assertIn("cute.arch.thread_idx()[0]) * 4", code)
        # Per-warp reduce via ``_cute_grouped_reduce_warp`` (with pre=1
        # the helper does ONE warp shuffle per warp -- effectively the
        # direct cute.arch.warp_reduction_* path).
        self.assertIn("_cute_grouped_reduce_warp", code)
        self.assertIn("pre=1, group_span=32", code)
        # NO shared-memory two-stage reduce.
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)

    def test_single_row_warp_reduce_is_baseline(self) -> None:
        """Pin the single-row + warp-reduce config that ``CuteTileVecWarpReduceHeuristic``
        seeds — this is the autotuner's pick for (4096, 256) and the
        baseline that multi-row attempts had to beat.  Documents the
        baseline shape so a future regression in heuristic seed coverage
        would be caught.
        """
        x = torch.randn(4096, 128, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # One warp per CTA.
        self.assertIn("block=(32, 1, 1)", code)
        # Warp reduce, no shared-mem reduction.
        self.assertIn("cute.arch.warp_reduction_max", code)
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)
        # The two-pass fuser must fire here so the consume sweep
        # reads x from a register cache (only one ``cute.arch.load``).
        # For (4096, 128) with block_size 128, trip = 1 → fuser bails;
        # but the larger N=256 path with block_size=128 has trip=2
        # which fires the fuser.  Document the trip-1 case here so the
        # fuser path stays explicit.
        self.assertEqual(code.count("cute.arch.load("), 2)


# A second, separately-decorated kernel so the P21 tests below can
# trigger the rewrite without inheriting the autouse fixture that
# disables it for the rest of this file.
@helion.kernel(backend="cute")
def softmax_two_pass_kernel_for_3pass(x: torch.Tensor) -> torch.Tensor:
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
class TestCuteOnlineTo3PassRewrite(TestCase):
    """P21: rewrite the online two-pass softmax body into the 3-pass form.

    The CuTe backend pre-processes the user's AST before tracing.  When
    the body matches the canonical online softmax pattern AND the
    reduction-axis extent is at or above the cutoff (default 2048), the
    pass rewrites the body into THREE inner ``for tile_n`` loops:

      1. max-only sweep
      2. sum-only sweep
      3. consume sweep (unchanged)

    The 3-pass form's two reductions are independent (no rescale data
    dependency between them) and compile to materially faster code on
    CuTe for large-N softmax shapes.

    Tests below pin:
      * detection fires for the canonical pattern,
      * shape gate skips small-N shapes where the rewrite regresses,
      * correctness end-to-end after the rewrite,
      * disable env-var skips the pass entirely.
    """

    def setUp(self) -> None:
        super().setUp()
        self._env_to_restore: dict[str, str | None] = {}
        # The Kernel decorator caches bound kernels by input signature
        # (shape + dtype + stride).  Each test in this class flips env
        # vars that change the compiler pipeline behavior, so we must
        # invalidate the cache before every test or stale compilations
        # leak across cases.
        softmax_two_pass_kernel_for_3pass.reset()

    def tearDown(self) -> None:
        for name, prev in self._env_to_restore.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev
        softmax_two_pass_kernel_for_3pass.reset()
        super().tearDown()

    def _set_env(self, name: str, value: str | None) -> None:
        if name not in self._env_to_restore:
            self._env_to_restore[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def _generated_loop_count(self, code: str) -> int:
        """Count the number of inner ``for tile_offset_3 in range(`` sites in
        the generated CuTe source.  Each inner sweep yields one of these
        loops, so this is the number of inner sweeps the rewrite emitted.
        """
        import re

        return len(re.findall(r"for tile_offset_\d+ in range\(", code))

    def test_rewrite_fires_on_canonical_pattern(self) -> None:
        """For (4096, 12672) the rewrite MUST fire and the generated
        kernel MUST have THREE inner ``for tile_offset`` loops (max,
        sum, consume) instead of TWO.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel_for_3pass,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # 3 inner sweeps now (was 2 in the original online form).
        self.assertEqual(self._generated_loop_count(code), 3)

    def test_shape_gate_skips_small_n(self) -> None:
        """For (4096, 256) the rewrite MUST be skipped by the
        reduction-axis-extent gate (256 < default cutoff 2048), so the
        generated kernel still has TWO inner sweeps (the original
        online form).
        """
        x = torch.randn(4096, 256, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel_for_3pass,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(self._generated_loop_count(code), 2)

    def test_disable_env_skips_rewrite(self) -> None:
        """``HELION_DISABLE_ONLINE_TO_3PASS=1`` MUST skip the pass even
        when the shape and pattern would otherwise match — the kernel
        falls back to the original 2-loop online form.
        """
        self._set_env("HELION_DISABLE_ONLINE_TO_3PASS", "1")
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel_for_3pass,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(self._generated_loop_count(code), 2)

    def test_min_n_env_override(self) -> None:
        """``HELION_ONLINE_TO_3PASS_MIN_N=0`` MUST always fire the rewrite
        regardless of reduction-axis extent (useful for A/B sweeps).
        Verify by running the (4096, 256) shape — which the default gate
        would skip — and observe the 3-loop form.
        """
        self._set_env("HELION_ONLINE_TO_3PASS_MIN_N", "0")
        x = torch.randn(4096, 256, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            softmax_two_pass_kernel_for_3pass,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(self._generated_loop_count(code), 3)

    def test_correctness_after_rewrite(self) -> None:
        """Tight-tolerance correctness check on the canonical large-N
        shape.  Guards against any silent numeric drift introduced by
        the rewrite (the 3-pass and online forms have different
        rounding paths in fp16 — the diff must stay inside softmax's
        existing tolerance).

        One power-of-two and one non-power-of-two N; each N is a full
        recompile.  (4096, 12672) is already numerics-checked by
        ``test_rewrite_fires_on_canonical_pattern`` above.
        """
        for N in (4096, 6400):
            x = torch.randn(4096, N, device=DEVICE, dtype=HALF_DTYPE)
            _, out = code_and_output(
                softmax_two_pass_kernel_for_3pass,
                (x,),
                block_sizes=[1, 128],
                num_threads=[0, 32],
                cute_vector_widths=[1, 4],
            )
            ref = torch.nn.functional.softmax(x, dim=1)
            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
