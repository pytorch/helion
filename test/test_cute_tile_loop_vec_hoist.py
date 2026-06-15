"""Tests for the CuTe tile-loop vec hoist codegen.

The hoist emits a single ``cute.arch.load(..., ir.VectorType.get([V], ...))``
above the constexpr V-loop in ``CuteNDTileStrategy``; the V-loop body then
reads per-lane scalars via bitcast extracts from the hoist var. This
replaces V scalar fp16 loads with one V*2-byte vec load per thread per
outer iter.

The hoist is gated to V<=4 for fp16/bf16 (the CuTe DSL's
``nvvm.load.ext`` ICEs at V=8); V=8 falls back to a 2x V=4 split with
``_a`` / ``_b`` suffixed hoist vars covering vec lanes 0-3 and 4-7.

Lives in ``helion/_compiler/tile_strategy.py``
(``CuteNDTileStrategy._cute_*_by_block`` hooks) and
``helion/language/memory_ops.py`` (``_cute_register_tile_unroll_vec_hoist``
and ``_cute_vector_load_ctx`` ``"tile_unroll"`` / ``"tile_unroll_split2"``
modes).
"""

from __future__ import annotations

import pytest
import torch

import helion
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
    two-pass form.  The ``online_to_3pass`` rewrite (in
    ``helion/_compiler/cute/online_to_3pass.py``) rewrites the kernel
    into a 3-loop form for N >= 2048, which by design changes the
    codegen those tests inspect.  Disable the rewrite globally for
    this file's tests; the rewrite itself is covered in
    ``test_cute_online_to_3pass.py``.
    """
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")


@helion.kernel(backend="cute")
def _reduction_kernel(x: torch.Tensor) -> torch.Tensor:
    """A two-pass reduction kernel that exercises the vec hoist path.

    Mirrors the structure ``examples/softmax.py::softmax_two_pass`` uses
    (an outer M-grid tile, an inner N-tile loop) so the inner load goes
    through ``CuteNDTileStrategy``'s tile-loop vec hoist when
    ``cute_vector_widths`` is set.
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


@helion.kernel(backend="cute", static_shapes=True)
def _fp8_matmul_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sa2d: torch.Tensor,
    sb1d: torch.Tensor,
) -> torch.Tensor:
    """A scaled fp8 matmul whose ``hl.dot`` lowers to the SIMT scalar
    fallback for skinny M, exercising the fp8 tile-loop vec hoist for both
    the row-major lhs (``x``) and the K-major rhs (``y``)."""
    m, k = x.size()
    k2, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = (acc * sa2d[tm, tn] * sb1d[tn]).to(torch.bfloat16)
    return out


@onlyBackends(["cute"])
class TestCuteTileLoopVecHoist(TestCase):
    def test_vec_hoist_fires_at_v4_fp16(self) -> None:
        """When ``cute_vector_widths=[1, 4]`` is set on a fp16 reduction
        kernel and the inner tile aligns with V, the codegen must emit a
        ``cute.arch.load(..., ir.VectorType.get([4], cutlass.Uint16.mlir_type))``
        — the new tile-loop vec hoist path that replaces 4 scalar fp16
        loads with one 8-byte vec load per thread per iter.
        """
        x = torch.randn(4096, 6400, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
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
        lanes 0-3 and 4-7).
        """
        x = torch.randn(4096, 8192, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
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
        # the load-pipeline pass fires, the ``_a`` load assignment text
        # gets rewritten to ``_tile_unroll_vec_1_0_a = _pipe_load_*``
        # with the actual ``cute.arch.load`` hoisted into the prologue;
        # the ``_b`` load is left as an inline ``cute.arch.load`` call.
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
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertIn("ir.VectorType.get([4]", code)
        self.assertIn("cutlass.BFloat16", code)

    def test_vec_hoist_fp8_matmul_both_operands(self) -> None:
        """fp8 matmul operands must vectorize through the tile-loop hoist as
        raw ``Uint8`` vectors (not bf16/fp16's ``Uint16``).  Both the
        row-major lhs and the K-major rhs (lane axis at index position 0)
        must emit a wide ``cute.arch.load(..., VectorType([4], Uint8))``.
        """
        m, k, n = 16, 4096, 4096
        x = (torch.randn(m, k, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (
            (torch.randn(k, n, device=DEVICE) * 0.4)
            .to(torch.float8_e4m3fn)
            .T.contiguous()
            .T
        )
        sa = (torch.rand(m, 1, device=DEVICE) + 0.5).float()
        sb = (torch.rand(1, n, device=DEVICE) + 0.5).float()
        code, out = code_and_output(
            _fp8_matmul_kernel,
            (x, y, sa.expand(m, n), sb.reshape(n)),
            block_sizes=[16, 64, 1024],
            num_threads=[1, 64, 0],
            cute_vector_widths=[1, 1, 4],
            indexing=[
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
        )
        ref = torch._scaled_mm(
            x, y, sa, sb, use_fast_accum=False, out_dtype=torch.bfloat16
        )
        torch.testing.assert_close(out.float(), ref.float(), atol=1.0, rtol=0.1)
        # fp8 V=4 loads a packed Uint32 (one LDG.32), NOT a VectorType nor the
        # bf16/fp16 Uint16 form; lane bytes come out via shift+mask.
        self.assertIn("cutlass.Uint32)", code)
        self.assertNotIn("ir.VectorType", code)
        self.assertIn("& 255", code)
        # Both operands must be hoisted (two distinct packed-load vars).
        self.assertIn("_tile_unroll_vec_", code)
        self.assertGreaterEqual(code.count("cutlass.Uint32)"), 2)

    def test_fp8_matmul_split_k_warp_reduce_vec(self) -> None:
        """fp8 matmul with K-axis threading (split-K within a warp) MUST stay
        numerically correct when combined with vectorized fp8 loads.

        Regression for the hoist_warp_reduce double-count bug: the matmul
        fallback feeds a loop-carried ``dot_acc`` running sum to
        ``warp_reduction_sum``.  The pass must reduce ``dot_acc``'s FINAL
        value once after the V-loop, not create a fresh V-fold accumulator
        (which would sum the running ``dot_acc`` every V iter and over-count).
        """
        m, k, n = 1, 4096, 4096
        x = (torch.randn(m, k, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (
            (torch.randn(k, n, device=DEVICE) * 0.4)
            .to(torch.float8_e4m3fn)
            .T.contiguous()
            .T
        )
        sa = (torch.rand(m, 1, device=DEVICE) + 0.5).float()
        sb = (torch.rand(1, n, device=DEVICE) + 0.5).float()
        code, out = code_and_output(
            _fp8_matmul_kernel,
            (x, y, sa.expand(m, n), sb.reshape(n)),
            # N-tile=1, 32 threads on the K reduction axis -> one warp per
            # output column, warp_reduction_sum (group_span=32, pre=1).
            block_sizes=[1, 1, 1024],
            num_threads=[1, 1, 32],
            cute_vector_widths=[1, 1, 4],
            indexing=[
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
        )
        ref = torch._scaled_mm(
            x, y, sa, sb, use_fast_accum=False, out_dtype=torch.bfloat16
        )
        torch.testing.assert_close(out.float(), ref.float(), atol=1.0, rtol=0.1)
        # The K reduction must use a single warp shuffle (not the 256-thread
        # shared-memory two-stage path), and the loaded fp8 must vectorize
        # (packed Uint32 for V=4).
        self.assertIn("warp_reduction_sum", code)
        self.assertIn("cutlass.Uint32)", code)
        # The warp reduce must NOT sit inside the constexpr V-loop (it would
        # be re-issued V times) — exactly one reduce per outer K iter.
        self.assertNotIn("_helion_vfold_acc", code)

    def test_scalar_load_when_vec_width_is_one(self) -> None:
        """When V=1 the vec hoist must NOT fire — the codegen falls back to
        the scalar load path so the change is opt-in.
        """
        x = torch.randn(4096, 6400, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 1],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertNotIn("_tile_unroll_vec_", code)
