"""CPU codegen of MPP's register operands (``mpp_registers``): a launch of
one SIMD group builds a matmul's computed operands as cooperative tensors
in registers, the lanes' elements at layouts factored into lane bases
(``coop_layout``)."""

from __future__ import annotations

import dataclasses
import re
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from helion._compiler.metal import coop_layout
from helion._compiler.metal import mpp_registers
from helion._testing import TestCase
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_fragment import k_from_16
from helion._testing.metal_fragment import nvfp4_cpu_args
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_registers import inner_tiled_mm
from helion._testing.metal_registers import register_config

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


def _k_loop(msl: str) -> str:
    """The body of the kernel's K loop."""
    start = msl.index("for (int offset_2 = ")
    return msl[start : msl.index("\n    }\n", start)]


class TestLayout(TestCase):
    def test_blocked_layout_factors(self) -> None:
        """The M4's layout of each operand of each op shape is a lane base
        plus a constant along both coordinates, the bases sums of the lane
        index's bits (the 8x8 fragments' ``fm`` and ``fn``); a device
        input has no elements."""
        for shape in ((16, 32, 16), (32, 32, 16), (16, 16, 32), (8, 32, 16)):
            device = shape[0] == 8
            atom = coop_layout.Atom(*shape, "half", "float", "float", device)
            layout = coop_layout.blocked_layout(atom)
            for operand in coop_layout.OPERANDS:
                coords = layout.of(operand)
                if operand == "left" and device:
                    self.assertEqual(coords, tuple(() for _ in range(32)))
                    continue
                rows, cols = {
                    "left": shape[::2],
                    "right": shape[:0:-1],
                    "dest": shape[:2],
                }[operand]
                # Every element of the operand once.
                self.assertEqual(
                    sorted(c for lane in coords for c in lane),
                    [(r, c) for r in range(rows) for c in range(cols)],
                )
                for d in (0, 1):
                    f = coop_layout.factor(coords, d)
                    assert f is not None
                    weights = coop_layout.lane_bits(f.base)
                    self.assertEqual(weights, ((0, 1, 2, 0, 4), (2, 0, 0, 4, 0))[d])
        self.assertEqual(
            coop_layout.lane_expr("l", (2, 0, 0, 4, 0)),
            "(l & 1) * 2 + ((l >> 3) & 1) * 4",
        )

    def test_layout_that_does_not_factor(self) -> None:
        """A layout whose coordinates aren't a lane base plus a constant per
        element has no factoring; one with an invalid element neither."""
        coords = tuple(((lane, 0), (lane * lane % 7, 1)) for lane in range(32))
        self.assertIsNone(coop_layout.factor(coords, 0))
        invalid = tuple(((0, lane), coop_layout.INVALID) for lane in range(32))
        self.assertIsNone(coop_layout.factor(invalid, 1))

    def test_k_order(self) -> None:
        """Between two register operands the K positions are relabelled so
        each lane's elements of a row are adjacent: an M4's lanes hold K
        ``fm`` and ``fm + 8`` of the right input, which the order
        interleaves, while an M5's (MLX's M5 GEMM fragments: rows ``fm`` and
        ``fm + 8`` by K ``fn .. fn + 3``) are adjacent already and keep the
        identity; both factor."""
        atom = coop_layout.Atom(16, 32, 16, "half", "half", "float")
        order = coop_layout.k_order(coop_layout.blocked_layout(atom), 16)
        self.assertEqual(order, tuple(2 * (k % 8) + k // 8 for k in range(16)))

        def m5(lane: int) -> tuple[int, int]:
            qid = lane >> 2
            return (qid & 4) | ((lane >> 1) & 3), ((qid & 2) | (lane & 1)) * 4

        rows4 = [(i, j) for i in range(2) for j in range(4)]
        layout = coop_layout.Layout(
            tuple(
                tuple((m5(lane)[0] + 8 * i, m5(lane)[1] + j) for i, j in rows4)
                for lane in range(32)
            ),
            tuple(
                tuple(
                    (m5(lane)[1] + j, 16 * h + m5(lane)[0] + 8 * i)
                    for h in range(2)
                    for i, j in rows4
                )
                for lane in range(32)
            ),
            tuple(
                tuple(
                    (m5(lane)[0] + 8 * i, 16 * h + m5(lane)[1] + j)
                    for h in range(2)
                    for i, j in rows4
                )
                for lane in range(32)
            ),
        )
        self.assertEqual(coop_layout.k_order(layout, 16), tuple(range(16)))
        for operand in coop_layout.OPERANDS:
            for d in (0, 1):
                f = coop_layout.factor(layout.of(operand), d)
                assert f is not None
                self.assertIsNotNone(coop_layout.lane_bits(f.base))


class TestRegisterCodegen(TestCase):
    def _assert_registers(self, msl: str) -> str:
        """The op runs at SIMD-group scope, with no threadgroup tile or
        barrier in its K loop; returns the loop."""
        self.assertIn("execution_simdgroup> _mpp_op_0;", msl)
        self.assertNotIn("execution_simdgroups<", msl)
        loop = _k_loop(msl)
        self.assertNotIn("threadgroup", loop)
        return loop

    def test_two_register_operands(self) -> None:
        """At 16 rows on an M4 both operands are built in registers: per
        16-deep step every lane writes its elements of the left tensor (x,
        exact in half) and of the right (the weight in float), and the op
        runs.  Each element's word, shift and group fold over the lane
        bases into one expression: a 4-bit element's one word, a 6-bit run
        of 8 K positions three (the K order keeps a lane's positions
        adjacent)."""
        for bits in (4, 5, 6):
            with self.subTest(bits=bits):
                args = affine_args(bits, 32, 16, 64, 512)
                msl = render_msl(affine_mm, args, register_config([16, 32, 64]))
                loop = self._assert_registers(msl)
                self.assertIn(
                    "matmul2d<matmul2d_descriptor(16, 32, 16, false, true, false, "
                    "matmul2d_descriptor::mode::multiply_accumulate), "
                    "execution_simdgroup> _mpp_op_0;",
                    msl,
                )
                self.assertIn(
                    "auto _mr_a = _mpp_op_0.get_left_input_cooperative_tensor"
                    "<half, float, float>();",
                    loop,
                )
                self.assertEqual(
                    loop.count("_mpp_op_0.run(_mr_a, _mr_b, acc_copy_0);"), 4
                )
                self.assertIn(
                    "const short _mrl0 = ((_mpp_lane >> 1) & 1) + ((_mpp_lane >> 2) "
                    "& 1) * 2 + ((_mpp_lane >> 4) & 1) * 4;",
                    msl,
                )
                self.assertIsNone(re.search(r"_mrl\d+\)? *%", loop))
                if bits == 4:
                    self.assertRegex(
                        loop,
                        r"auto v_3 = static_cast<int>\(\(_mrl\d+ \+ \(offset_2 / 4\)\)\);",
                    )
                if bits == 6:
                    self.assertRegex(
                        loop,
                        r"auto v_3 = static_cast<int>\(\(\(\(_mrl\d+ / 2\) \* 3\) "
                        r"\+ \(\(offset_2 / 8\) \* 3\)\)\);",
                    )

    def test_view_operand(self) -> None:
        """At 8 rows x stays a device view of its half tensor, read one
        16-deep step at a time beside the weight's cooperative tensor, so
        the op takes 8 rows where two cooperative inputs would take 16; a
        step past the tile's live K is skipped.  At 16 rows x is built in
        registers on the M4's layout, and stays a view on one that keeps
        each lane's K positions of a row adjacent in order (the M5's;
        mocked)."""
        args = affine_args(4, 32, 8, 64, 512)
        config = register_config([16, 32, 64])
        msl = render_msl(affine_mm, args, config)
        loop = self._assert_registers(msl)
        self.assertIn("matmul2d_descriptor(8, 32, 16, false, true, false,", msl)
        self.assertNotIn("_mr_a", msl)
        self.assertEqual(loop.count("auto _mpp_a = tensor<device half"), 4)
        self.assertIn("(long)((offset_2) + 16) * 1)", loop)
        # Every K tile is whole, so every step is live.
        self.assertNotIn("< (long)(512)) {", loop)
        self.assertEqual(loop.count("_mpp_op_0.run(_mpp_a, _mr_b, acc_copy_0);"), 4)
        # The destination is typed by the device input.
        self.assertIn(
            "get_destination_cooperative_tensor<tensor<device half, extents<int32_t, "
            "16, 8>, tensor_inline>, ",
            msl,
        )
        args = affine_args(4, 32, 16, 64, 512)
        msl = render_msl(affine_mm, args, config)
        self.assertIn("matmul2d_descriptor(16, 32, 16, false, true, false,", msl)
        self.assertIn("_mr_a", msl)
        with mock.patch.object(
            mpp_registers, "k_order", lambda layout, k: tuple(range(k))
        ):
            msl = render_msl(affine_mm, args, config)
        self.assertIn("matmul2d_descriptor(16, 32, 16, false, true, false,", msl)
        self.assertNotIn("_mr_a", msl)

    def test_op_shapes(self) -> None:
        """With two cooperative inputs MPP takes M, N and K of 16 or 32, one
        of them 32: a 16 x 16 tile takes a K of 32 and a 32-row tile ops 32
        wide; a tile wider than 32 keeps x a device view, whose op covers
        it (one cooperative input)."""
        args = affine_args(4, 32, 16, 64, 256)
        msl = render_msl(affine_mm, args, register_config([16, 16, 64]))
        self.assertIn("matmul2d_descriptor(16, 16, 32,", msl)
        self.assertEqual(_k_loop(msl).count("_mpp_op_0.run(_mr_a, _mr_b,"), 2)
        args = affine_args(4, 32, 32, 64, 256)
        msl = render_msl(affine_mm, args, register_config([32, 32, 64]))
        self.assertIn("matmul2d_descriptor(32, 32, 16,", msl)
        msl = render_msl(affine_mm, args, register_config([32, 64, 64]))
        self.assertIn("matmul2d_descriptor(32, 64, 16,", msl)
        self.assertIn("auto _mpp_a = tensor<device half", msl)

    def test_more_simdgroups_stage(self) -> None:
        """A launch of two SIMD groups stages the weight through threadgroup
        memory, as before: cooperative inputs need one SIMD group, and the
        two share each tile, whether their threads split K or N."""
        args = affine_args(4, 32, 16, 64, 256)
        for block_sizes, threads in (
            ([16, 32, 64], [4, 8, 2]),
            ([16, 64, 64], [4, 16, 1]),
        ):
            with self.subTest(threads=threads):
                msl = render_msl(affine_mm, args, register_config(block_sizes, threads))
                self.assertIn("threadgroup float _mpp_tile", msl)
                self.assertNotIn("_mr_b", msl)
                self.assertIn("execution_simdgroups<", msl)

    def test_operand_types(self) -> None:
        """A register operand exactly representable in half or bfloat takes
        it, widened or not, another is float; a view is its tensor's."""
        x, wq, s = nvfp4_cpu_args(16)
        msl = render_msl(nvfp4_rows, (x, wq, s), register_config([16, 32, 64]))
        self.assertIn("get_right_input_cooperative_tensor<bfloat, bfloat, float>", msl)
        self.assertIn("auto _mpp_a = tensor<device bfloat", msl)
        # Every conversion to bfloat in the element loops rounds as lane
        # copies round (helion_bf16_cast), keeping bfloat subnormals that
        # Metal's own cast flushes.
        self.assertRegex(msl, r"_mr_b\[\d+\] = helion_bf16_cast\(")
        self.assertNotIn("static_cast<bfloat>", _k_loop(msl))
        args = affine_args(4, 32, 16, 64, 256)
        msl = render_msl(
            affine_mm, (args[0].float(), *args[1:]), register_config([16, 32, 64])
        )
        self.assertIn("get_right_input_cooperative_tensor<float, float, float>", msl)
        self.assertIn("auto _mpp_a = tensor<device float", msl)

    def test_narrowed_operands(self) -> None:
        """Operands converted to half from float are both built in registers
        (no device view holds them), over a K loop from 16 with tails along
        M, N and K: elements past the K tile's live length are 0."""
        x, y = torch.randn(13, 40), torch.randn(40, 20)
        msl = render_msl(k_from_16, (x, y), register_config([16, 32, 16]))
        loop = self._assert_registers(msl)
        self.assertIn(
            "auto _mr_a = _mpp_op_0.get_left_input_cooperative_tensor<half, half, "
            "float>();",
            loop,
        )
        self.assertRegex(loop, r"_e_k < \(\w+ - offset_2\)")

    def test_view_span(self) -> None:
        """A view whose handles would pass the int32 offsets MPP tensor
        handles index with isn't read as one: a widened activation is built
        in registers instead, and a device view keeps the staged path, which
        refuses it as before."""
        self.assertFalse(
            mpp_registers._view_fits(SimpleNamespace(strides=(2**29, 1)), (8, 16))
        )
        self.assertTrue(
            mpp_registers._view_fits(SimpleNamespace(strides=(4096, 1)), (8, 16))
        )
        config = register_config([16, 32, 64])
        with mock.patch.object(mpp_registers, "_view_fits", lambda view, tile: False):
            msl = render_msl(affine_mm, affine_args(4, 32, 8, 64, 512), config)
            self.assertNotIn("tensor<device half", msl)
            self.assertIn("_mr_a", msl)
            msl = render_msl(nvfp4_rows, nvfp4_cpu_args(), config)
            self.assertNotIn("_mr_b", msl)
            self.assertIn("threadgroup bfloat _mpp_tile[", msl)

    def test_relabel_needs_both_inputs(self) -> None:
        """Two register inputs relabel their K positions only when both
        inputs' coordinates factor: an input whose coordinates are read at
        run time keeps the op's K order, so the other keeps it too (a layout
        whose right input doesn't factor, mocked)."""

        def scrambled(atom: coop_layout.Atom, on_gpu: bool) -> coop_layout.Layout:
            probed = coop_layout.blocked_layout(atom)
            right = [list(lane) for lane in probed.right]
            right[0][0], right[1][0] = right[1][0], right[0][0]
            return dataclasses.replace(
                probed, right=tuple(tuple(lane) for lane in right)
            )

        args = affine_args(4, 32, 16, 64, 512)
        with mock.patch.object(mpp_registers, "layout", scrambled):
            msl = render_msl(affine_mm, args, register_config([16, 32, 64]))
        self.assertIn("_mr_a", msl)
        self.assertIn("get_multidimensional_index", msl)
        k_positions = re.findall(r"int _e_k = ([^;]+);", msl)
        self.assertTrue(k_positions)
        for k_pos in k_positions:
            self.assertTrue(k_pos.startswith("16 * _fs + "), k_pos)

    def test_view_too_deep(self) -> None:
        """Where the view's 8-row op would take too many steps (a K tile of
        512 is 32 steps of 16), x goes into registers beside the weight:
        the two-register op takes 16 steps of 32."""
        args = affine_args(4, 32, 8, 64, 1024)
        msl = render_msl(affine_mm, args, register_config([16, 16, 512]))
        loop = self._assert_registers(msl)
        self.assertIn("matmul2d_descriptor(16, 16, 32, false, true, false,", msl)
        self.assertEqual(loop.count("_mpp_op_0.run(_mr_a, _mr_b, acc_copy_0);"), 16)

    def test_view_tail(self) -> None:
        """Over a K tail a view's step past the tile's live K is skipped."""
        args = affine_args(4, 32, 8, 64, 544)
        msl = render_msl(affine_mm, args, register_config([16, 32, 64]))
        self.assertIn("if ((long)((offset_2) + 48) < (long)(544)) {", _k_loop(msl))

    def test_one_decision(self) -> None:
        """The accumulator and its run take one form where the launch's
        thread axes codegen sees differ inside the K loop (M and N tiled in
        a loop of a one-program grid, K's block over 16 threads): built in
        registers on the launch's one SIMD group, not an op declared for
        registers running staged tiles."""
        x, y = torch.randn(16, 128), torch.randn(128, 32)
        config = register_config([16, 32, 64], [1, 2, 16])
        msl = render_msl(inner_tiled_mm, (x, y), config)
        self.assertIn("execution_simdgroup> _mpp_op_0;", msl)
        self.assertNotIn("_mpp_tile", msl)
        self.assertIn("_mpp_op_0.run(_mr_a, ", msl)

    def test_lane_table(self) -> None:
        """A lane base that factors but isn't a sum of the lane index's bits'
        weights is read from a table (a layout with three lanes rotated,
        mocked)."""

        def rotated(atom: coop_layout.Atom, on_gpu: bool) -> coop_layout.Layout:
            probed = coop_layout.blocked_layout(atom)

            def rotate(coords: coop_layout.Coords) -> coop_layout.Coords:
                lanes = list(coords)
                lanes[1], lanes[2], lanes[3] = lanes[3], lanes[1], lanes[2]
                return tuple(lanes)

            return coop_layout.Layout(
                rotate(probed.left), rotate(probed.right), rotate(probed.dest)
            )

        args = affine_args(4, 32, 16, 64, 384)
        with mock.patch.object(mpp_registers, "layout", rotated):
            msl = render_msl(affine_mm, args, register_config([16, 32, 64]))
        self.assertRegex(msl, r"const short _mrl\d+ = _mrl_table\d+\[_mpp_lane\];")

    def test_runtime_coordinates(self) -> None:
        """A layout that doesn't factor reads each element's coordinates and
        validity from the op in the preamble; the elements fold over their
        ranges only, and an invalid element writes 0."""
        args = affine_args(4, 32, 16, 64, 256)
        with mock.patch.object(mpp_registers, "factor", lambda coords, d: None):
            msl = render_msl(affine_mm, args, register_config([16, 32, 64]))
        self.assertIn(
            "const short _mrl0_r0 = (short)_mrl_t0.get_multidimensional_index("
            "(ushort)0)[1];",
            msl,
        )
        self.assertIn("if (_mrl0_v0) {", msl)
        self.assertIn("} else _mr_a[0] = 0;", msl)


if __name__ == "__main__":
    import unittest

    unittest.main()
