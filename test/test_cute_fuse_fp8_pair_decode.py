"""Unit tests for the ``fuse_fp8_pair_decode`` AST peephole.

The pass rewrites the SIMT matmul fallback's per-lane scalar fp8 decode
(`load = Uint8(PK >> 8*lane & 255); ...; dot_acc = dot_acc + Float32(
decode(x) * decode(y))`) in a constexpr V-loop into a step-2 loop that
decodes two e4m3 bytes per operand per iter with one
`_cute_fp8e4m3fn_x2_to_float32` call.

These are pure-AST tests — the pass does not touch CUDA — so they assert
the rewrite contract directly (no device, no codegen).  End-to-end codegen +
numerics are covered by
``test_cute_tile_loop_vec_hoist.py::...test_fp8_paired_decode_in_warp_per_row_gemv``.
"""

from __future__ import annotations

import ast
import textwrap

from helion._compiler.cute.fuse_fp8_pair_decode import fuse_fp8_pair_decode
from helion._testing import TestCase


def _run(src: str) -> str:
    """Run the pass on dedented source and return the unparsed result."""
    body = ast.parse(textwrap.dedent(src)).body
    out = fuse_fp8_pair_decode(body)
    return ast.unparse(ast.Module(body=out, type_ignores=[]))


# The canonical matmul-fallback V=8 loop the pass targets.
_FUSIBLE_V8 = """
for vec_lane_1 in cutlass.range_constexpr(8):
    indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
    mask_1 = indices_1 < 4096
    load = cutlass.Uint8(_tile_unroll_vec_1_0 >> 8 * vec_lane_1 & 255) if mask_1 else cutlass.Uint8(0)
    load_1 = cutlass.Uint8(_tile_unroll_vec_1_1 >> 8 * vec_lane_1 & 255) if mask_1 and mask_0 else cutlass.Uint8(0)
    dot_product_0 = cutlass.Float32(_cute_fp8e4m3fn_to_float32(load) * _cute_fp8e4m3fn_to_float32(load_1))
    dot_acc = dot_acc + dot_product_0
"""


class TestFuseFp8PairDecode(TestCase):
    # ---- positive: the rewrite fires and is well-formed ----

    def test_v8_fuses_to_paired_decode(self) -> None:
        # Snapshot the full rewritten loop (run with EXPECTTEST_ACCEPT=1 to
        # update the .expected journal).
        self.assertExpectedJournal(_run(_FUSIBLE_V8))

    def test_v4_fuses_to_two_iters(self) -> None:
        out = _run(_FUSIBLE_V8.replace("range_constexpr(8)", "range_constexpr(4)"))
        self.assertExpectedJournal(out)

    def test_output_is_valid_python(self) -> None:
        # The rewritten loop must re-parse (no malformed f-string fragments).
        ast.parse(_run(_FUSIBLE_V8))

    def test_idempotent(self) -> None:
        # Running twice changes nothing the second time (no scalar decode to
        # match after the first pass).
        once = _run(_FUSIBLE_V8)
        twice = ast.unparse(
            ast.Module(body=fuse_fp8_pair_decode(ast.parse(once).body), type_ignores=[])
        )
        self.assertEqual(once, twice)

    def test_fires_inside_nested_loops(self) -> None:
        nested = "for tile_offset_1 in range(0, 4096, 1024):\n" + textwrap.indent(
            _FUSIBLE_V8, "    "
        )
        out = _run(nested)
        self.assertIn("for tile_offset_1 in range(0, 4096, 1024)", out)
        self.assertIn("_cute_fp8e4m3fn_x2_to_float32", out)
        self.assertIn("cutlass.range_constexpr(4)", out)

    # ---- negative: the pass must leave non-matching loops untouched ----

    def _assert_unchanged(self, src: str) -> None:
        expected = ast.unparse(ast.parse(textwrap.dedent(src)))
        self.assertEqual(_run(src), expected)

    def test_odd_trip_count_not_fused(self) -> None:
        self._assert_unchanged(
            _FUSIBLE_V8.replace("range_constexpr(8)", "range_constexpr(3)")
        )

    def test_v1_not_fused(self) -> None:
        self._assert_unchanged(
            _FUSIBLE_V8.replace("range_constexpr(8)", "range_constexpr(1)")
        )

    def test_non_constexpr_loop_not_fused(self) -> None:
        # A plain ``range`` (lane) loop, not ``cutlass.range_constexpr``.
        self._assert_unchanged(
            _FUSIBLE_V8.replace("cutlass.range_constexpr(8)", "range(8)")
        )

    def test_2d_inloop_acc_not_fused(self) -> None:
        # The 2D matmul form has a trailing ``acc = dot_acc_base + dot_acc``
        # inside the loop (not the hoisted warp-per-row form).  The pass must
        # NOT fire — `acc` is not a self-accumulating `dot_acc += product`.
        src = _FUSIBLE_V8 + "    acc = dot_acc_base + dot_acc\n"
        self._assert_unchanged(src)

    def test_non_self_accumulating_not_fused(self) -> None:
        # `total = dot_acc + dot_product_0` is not `x = x + ...`.
        src = _FUSIBLE_V8.replace(
            "dot_acc = dot_acc + dot_product_0", "total = dot_acc + dot_product_0"
        )
        self._assert_unchanged(src)

    def test_non_fp8_decode_not_fused(self) -> None:
        # bf16/fp16 loads bitcast a Uint16 vector — not the Uint8 byte-extract
        # the pass keys on — so they must be left alone.
        src = """
        for vec_lane_1 in cutlass.range_constexpr(8):
            load = cutlass.Uint16(_tile_unroll_vec_1_0[vec_lane_1]).bitcast(cutlass.BFloat16) if mask_1 else cutlass.BFloat16(0)
            load_1 = cutlass.Uint16(_tile_unroll_vec_1_1[vec_lane_1]).bitcast(cutlass.BFloat16) if mask_1 else cutlass.BFloat16(0)
            dot_product_0 = cutlass.Float32(cutlass.Float32(load) * cutlass.Float32(load_1))
            dot_acc = dot_acc + dot_product_0
        """
        self._assert_unchanged(src)

    def test_single_operand_not_fused(self) -> None:
        # Only one packed load present — not a 2-operand product.
        src = """
        for vec_lane_1 in cutlass.range_constexpr(8):
            load = cutlass.Uint8(_tile_unroll_vec_1_0 >> 8 * vec_lane_1 & 255) if mask_1 else cutlass.Uint8(0)
            dot_product_0 = cutlass.Float32(_cute_fp8e4m3fn_to_float32(load))
            dot_acc = dot_acc + dot_product_0
        """
        self._assert_unchanged(src)

    def test_unexpected_extra_stmt_not_fused(self) -> None:
        # An unrecognized statement in the loop body bails the whole rewrite.
        src = _FUSIBLE_V8 + "    side_effect = some_call(load)\n"
        self._assert_unchanged(src)


if __name__ == "__main__":
    import unittest

    unittest.main()
