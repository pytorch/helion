"""CPU codegen of MPP's register operands (``mpp_registers``): a launch of
one SIMD group builds a matmul's computed operands as cooperative tensors
in registers, the lanes' elements at layouts factored into lane bases
(``coop_layout``)."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
import re
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import helion
from helion import exc
from helion._compiler.metal import bit_windows
from helion._compiler.metal import coop_layout
from helion._compiler.metal import mpp_registers
from helion._compiler.metal.affine_fold import Affine
from helion._compiler.metal.affine_fold import Fact
from helion._testing import TestCase
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_fragment import affine_mm_f16
from helion._testing.metal_fragment import k_from_16
from helion._testing.metal_fragment import nvfp4_cpu_args
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_fragment import seeded_rhs
from helion._testing.metal_registers import affine_gated_mm
from helion._testing.metal_registers import apart_mm
from helion._testing.metal_registers import bounded_k_mm
from helion._testing.metal_registers import copying_mm
from helion._testing.metal_registers import counted_mm
from helion._testing.metal_registers import dequant_forms_mm
from helion._testing.metal_registers import early_a_mm
from helion._testing.metal_registers import early_b_mm
from helion._testing.metal_registers import filled_mm
from helion._testing.metal_registers import free_cols_mm
from helion._testing.metal_registers import gated_mm
from helion._testing.metal_registers import inner_tiled_mm
from helion._testing.metal_registers import inner_tiled_weight_mm
from helion._testing.metal_registers import long_k_mm
from helion._testing.metal_registers import mixed_gated_mm
from helion._testing.metal_registers import mm_and_double
from helion._testing.metal_registers import narrowed_gated_mm
from helion._testing.metal_registers import offset_k_mm
from helion._testing.metal_registers import register_config
from helion._testing.metal_registers import rewriting_mm
from helion._testing.metal_registers import row_scaled_mm
from helion._testing.metal_registers import rowsum_mm
from helion._testing.metal_registers import scaled_gated_mm
from helion._testing.metal_registers import seed_join_mm
from helion._testing.metal_registers import seed_read_mm
from helion._testing.metal_registers import shifted_x_mm
from helion._testing.metal_registers import silu_gated_mm
from helion._testing.metal_registers import staged_apart_mm
from helion._testing.metal_registers import transposed_out_mm
from helion._testing.metal_registers import two_acc_mm
from helion._testing.metal_registers import unread_x
from helion.autotuner.config_spec import DEFAULT_NUM_WARPS
from helion.runtime import capture_metal_launches

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


def _k_loop(msl: str) -> str:
    """The body of the kernel's K loop."""
    start = msl.index("for (int offset_2 = ")
    return msl[start : msl.index("\n    }\n", start)]


def _runs(loop: str, left: str, right: str) -> list[int]:
    """The steps of the ops on ``left``/``right`` operands in ``loop``, in
    order: each step's operands carry its number (``_mr_a1``, ``_mpp_a1``),
    and the loop runs no other op."""
    steps = [
        int(m.group(1))
        for m in re.finditer(
            rf"_mpp_op_0\.run\({left}(\d+), {right}\1, acc_copy_0\);", loop
        )
    ]
    assert loop.count("_mpp_op_0.run(") == len(steps), loop
    return steps


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
                    "auto _mr_a0 = _mpp_op_0.get_left_input_cooperative_tensor"
                    "<half, float, float>();",
                    loop,
                )
                self.assertEqual(_runs(loop, "_mr_a", "_mr_b"), [0, 1, 2, 3])
                # With both operands in registers the steps keep their order.
                self.assertLess(
                    loop.index("_mpp_op_0.run(_mr_a0, _mr_b0, acc_copy_0);"),
                    loop.index("auto _mr_a1 = "),
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
        self.assertEqual(
            len(re.findall(r"auto _mpp_a\d+ = tensor<device half", loop)), 4
        )
        self.assertIn("(long)((offset_2) + 16) * 1)", loop)
        # Every K tile is whole, so every step is live.
        self.assertNotIn("< (long)(512)) {", loop)
        self.assertEqual(_runs(loop, "_mpp_a", "_mr_b"), [0, 1, 2, 3])
        # Beside a view, each step's operands are built before the previous
        # step's op, so their reads overlap it.
        self.assertLess(
            loop.index("auto _mr_b1 = "),
            loop.index("_mpp_op_0.run(_mpp_a0, _mr_b0, acc_copy_0);"),
        )
        self.assertLess(
            loop.index("_mpp_op_0.run(_mpp_a0, _mr_b0, acc_copy_0);"),
            loop.index("auto _mr_b2 = "),
        )
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

    def test_accumulator_indexed(self) -> None:
        """A register op's accumulator is indexed by constants only: filled
        by an unrolled loop over its capacity and stored by an element loop
        unrolled over its slots (16 of a 16 x 32 op on one SIMD group),
        never by a loop over its iterator."""
        args = affine_args(4, 32, 16, 64, 512)
        msl = render_msl(affine_mm, args, register_config([16, 32, 64]))
        self.assertIn(
            "for (ushort _e = 0; _e < acc.get_capacity(); _e++) acc[_e] = 0.0f;", msl
        )
        self.assertIn("constexpr ushort _mpp_slots = ((512u + 32 - 1) / 32)", msl)
        self.assertNotIn(".begin(); _it != ", msl)

    def test_op_shapes(self) -> None:
        """With two cooperative inputs MPP takes M, N and K of 16 or 32, one
        of them 32: a 16 x 16 tile takes a K of 32 and a 32-row tile ops 32
        wide; a tile wider than 32 keeps x a device view, whose op covers
        it (one cooperative input)."""
        args = affine_args(4, 32, 16, 64, 256)
        msl = render_msl(affine_mm, args, register_config([16, 16, 64]))
        self.assertIn("matmul2d_descriptor(16, 16, 32,", msl)
        self.assertEqual(_runs(_k_loop(msl), "_mr_a", "_mr_b"), [0, 1])
        args = affine_args(4, 32, 32, 64, 256)
        msl = render_msl(affine_mm, args, register_config([32, 32, 64]))
        self.assertIn("matmul2d_descriptor(32, 32, 16,", msl)
        msl = render_msl(affine_mm, args, register_config([32, 64, 64]))
        self.assertIn("matmul2d_descriptor(32, 64, 16,", msl)
        self.assertRegex(msl, r"auto _mpp_a\d+ = tensor<device half")

    def test_grid(self) -> None:
        """A launch of several SIMD groups splits a whole tile over them, each
        running an op with two cooperative inputs over its sub-tile: every
        thread stages a run of one row's K positions of the weight, its
        masks true, into a tile of rows of N positions padded by 16 bytes
        (4 floats); after a barrier each SIMD group builds x from its view
        and the weight from the tile, and its sub-tile's origin offsets its
        accumulator's elements.  The sub-tiles prefer 32 x 32, then the
        most rows."""
        for rows, block_sizes, threads, (am, an, ak) in (
            # 2 SIMD groups, their threads splitting K: 16 x 16 sub-tiles.
            (16, [16, 32, 64], [4, 8, 2], (16, 16, 32)),
            (16, [16, 64, 64], [4, 16, 1], (16, 32, 16)),
            (64, [64, 64, 64], [8, 16, 1], (32, 32, 16)),
            # 32 x 16 sub-tiles over 16 x 32.
            (32, [32, 64, 64], [8, 16, 1], (32, 16, 16)),
        ):
            with self.subTest(block_sizes=block_sizes, threads=threads):
                args = affine_args(4, 32, rows, 128, 256)
                msl = render_msl(affine_mm, args, register_config(block_sizes, threads))
                bn, bk = block_sizes[1:]
                sn = bn // an
                self.assertIn(
                    f"matmul2d_descriptor({am}, {an}, {ak}, false, true, false,", msl
                )
                self.assertNotIn("execution_simdgroups<", msl)
                self.assertIn(
                    f"threadgroup float _mst_tile_float[{bn * (bk + 4)}];", msl
                )
                self.assertIn(f"constexpr int _mst0_ld = {bk + 4};", msl)
                self.assertIn(f"const int _mst0_m0 = int(_frag_sg / {sn}) * {am};", msl)
                self.assertIn(f"const int _mst0_n0 = int(_frag_sg % {sn}) * {an};", msl)
                self.assertIn("uint _frag_sg [[simdgroup_index_in_threadgroup]]", msl)
                loop = _k_loop(msl)
                self.assertEqual(loop.count("threadgroup_barrier("), 2)
                self.assertIn(
                    "_mst_tile_float[_mst0_r * _mst0_ld + _mst0_k + 0] = ", loop
                )
                self.assertIn("auto mask_1 = true;", loop)
                self.assertIn("auto mask_2 = true;", loop)
                self.assertRegex(
                    loop, r"_mr_b0\[0\] = _mst_tile_float\[\(_mst0_n0 \+ 0 \+ _mrl\d+\)"
                )
                self.assertIn("_mpp_op_0.run(_mr_a0, _mr_b0, acc_copy_0);", loop)
                self.assertIn("int _e_n = (int)_md[0] + _mst0_n0;", msl)
                self.assertIn("int _e_m = (int)_md[1] + _mst0_m0;", msl)
                # The sub-tiles cover the tile: no element lies past it.
                self.assertNotIn("if (_e_m >= ", msl)

    def test_grid_tiles(self) -> None:
        """The grid runs of one type share one tile, which counts against the
        threadgroup memory: two accumulators' runs over [32, 64, 64] take one
        tile of 64 x 68 floats.  Rows are padded only where every padded tile
        fits: 128 x 64 floats and 128 x 128 halves fill 32 KB unpadded, and a
        float tile beside a half one over 32 KB padded fits unpadded; a tile
        past it even unpadded (128 x 128 floats) is refused before the
        pipeline is built."""
        x = torch.randn(32, 256).to(torch.float16)
        y, z = torch.randn(256, 64), torch.randn(256, 64)
        msl = render_msl(
            two_acc_mm, (x, y, z), register_config([32, 64, 64], [8, 16, 1])
        )
        self.assertEqual(msl.count("threadgroup float _mst_tile_float[4352];"), 1)
        self.assertEqual(
            set(re.findall(r"_mst_tile_float\[_mst(\d)_r \*", msl)), {"0", "1"}
        )
        args = affine_args(4, 32, 32, 128, 256)
        msl = render_msl(affine_mm, args, register_config([32, 128, 64], [8, 16, 1]))
        self.assertIn("threadgroup float _mst_tile_float[8192];", msl)
        self.assertIn("constexpr int _mst0_ld = 64;", msl)
        args = affine_args(4, 32, 32, 256, 512)
        msl = render_msl(
            dequant_forms_mm,
            (*args, "half"),
            register_config([32, 128, 128], [8, 16, 1]),
        )
        self.assertIn("threadgroup half _mst_tile_half[16384];", msl)
        self.assertIn("constexpr int _mst0_ld = 128;", msl)
        self.assertEqual(
            mpp_registers.grid_tiles({0: ("float", 64, 64), 1: ("half", 64, 64)}, 8192),
            ({"float": 4096, "half": 4096}, {0: 64, 1: 64}, 32768),
        )
        self.assertEqual(
            mpp_registers.grid_tiles({0: ("float", 64, 64)}, 0),
            ({"float": 4352}, {0: 68}, 17408),
        )
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "grid runs' tiles need 65536 bytes"
        ):
            render_msl(affine_mm, args, register_config([32, 128, 128], [8, 16, 1]))

    def test_grid_runtime_lanes(self) -> None:
        """Where a layout doesn't factor (its coordinates read at run time), a
        grid run's elements a lane doesn't hold read nothing and hold 0."""
        args = affine_args(4, 32, 32, 128, 256)
        with mock.patch.object(mpp_registers, "factor", lambda coords, d: None):
            msl = render_msl(affine_mm, args, register_config([32, 64, 64], [8, 16, 1]))
        self.assertRegex(
            msl, r"if \(_mrl\d+_v0\) _mr_a0\[0\] = _mst0_x\[.*\]; else _mr_a0\[0\] = 0;"
        )
        self.assertRegex(
            msl,
            r"if \(_mrl\d+_v0\) _mr_b0\[0\] = _mst_tile_float\[.*\]; "
            r"else _mr_b0\[0\] = 0;",
        )

    def test_grid_inner_tiled(self) -> None:
        """M and N tiled in a loop of a one-program grid: the thread axes seen
        at the accumulator aren't the launch's, so a launch of several SIMD
        groups keeps the staged op."""
        x = torch.randn(32, 128).to(torch.float16)
        y = torch.randn(128, 64)
        for block_sizes, threads in (
            ([16, 32, 64], [2, 4, 4]),
            ([32, 64, 32], [4, 8, 2]),
        ):
            with self.subTest(block_sizes=block_sizes, threads=threads):
                msl = render_msl(
                    inner_tiled_weight_mm, (x, y), register_config(block_sizes, threads)
                )
                self.assertNotIn("_mst_tile", msl)
                self.assertIn("execution_simdgroups<", msl)

    def test_grid_unread_x(self) -> None:
        """A widened x the lanes may not read at the op (``registers`` unset,
        as for a load of a tensor the kernel writes) keeps the staged op."""
        args = affine_args(4, 32, 16, 128, 256)
        config = register_config([16, 64, 64], [4, 16, 1])
        self.assertIn("_mst_tile", render_msl(affine_mm, args, config))
        with unread_x():
            msl = render_msl(affine_mm, args, config)
        self.assertNotIn("_mst_tile", msl)
        self.assertIn("threadgroup float _mpp_tile", msl)

    def test_grid_views(self) -> None:
        """A grid run reads x's elements from the tensor unmasked, so each
        read must lie inside it: a view of x 8 rows past its tile's offset, a
        K loop whose begin is a run-time value, and one past x's K (64 of
        128, or 1, a size-1 dim) keep the staged op."""
        config = register_config([32, 64, 64], [8, 16, 1])
        cases = [
            (shifted_x_mm, (torch.randn(40, 128).half(), torch.randn(128, 64))),
            (offset_k_mm, (torch.randn(32, 256).half(), torch.randn(256, 64), 64)),
            (offset_k_mm, (torch.randn(32, 1).half(), torch.randn(128, 64), 0)),
            (long_k_mm, (torch.randn(32, 64).half(), torch.randn(128, 64))),
            (long_k_mm, (torch.randn(32, 1).half(), torch.randn(128, 64))),
        ]
        for kernel, args in cases:
            with self.subTest(kernel=kernel.name, x=tuple(args[0].shape)):
                msl = render_msl(kernel, args, config)
                self.assertNotIn("_mst_tile", msl)
                self.assertIn("execution_simdgroups<", msl)

    def test_grid_bfloat(self) -> None:
        """A bfloat x beside weights dequantized in bfloat takes a grid run:
        the lanes read x as bfloat, and every thread rounds its float
        weights into a bfloat tile."""
        x, w, scales, biases = affine_args(4, 32, 32, 128, 512)
        args = (x.to(torch.bfloat16), w, scales, biases, "bfloat")
        config = register_config([32, 64, 64], [8, 8, 1])
        msl = render_msl(dequant_forms_mm, args, config)
        self.assertIn("threadgroup bfloat _mst_tile_bfloat[4608]", msl)
        self.assertIn("const device bfloat* _mst0_x = x + ", msl)
        self.assertIn("= helion_bf16_cast(static_cast<float>(", msl)
        self.assertIn("cooperative_tensor<bfloat, bfloat, float>", msl)

    def test_grid_all_or_none(self) -> None:
        """Where one group stays on the staged op (a staged vector), the
        kernel's other groups that stage an operand do too, whose tiles share
        threadgroup memory with its own: configs its staged ops fit in, over
        half of it each, still compile."""
        x = torch.randn(64, 512).to(torch.float16)
        y, z, v = torch.randn(512, 128), torch.randn(512, 128), torch.randn(4, 128)
        for block_sizes in ([32, 64, 64], [32, 128, 64], [64, 64, 128]):
            with self.subTest(block_sizes=block_sizes):
                config = register_config(block_sizes, [8, 16, 1])
                msl = render_msl(staged_apart_mm, (x, y, z, v), config)
                self.assertNotIn("_mst_tile", msl)
                self.assertIn("_mpp_vec", msl)

    def test_grid_refusals(self) -> None:
        """A launch of several SIMD groups stages the weight through
        threadgroup memory for one op over the whole tile, as before, when a
        tile is partial (12 rows of a 16-row tile; a K of 288 over tiles of
        64) or no sub-tile fits two cooperative inputs (16 x 16 over 2 SIMD
        groups)."""
        for rows, k, block_sizes, threads in (
            (12, 256, [16, 64, 64], [4, 16, 1]),
            (16, 288, [16, 64, 64], [4, 16, 1]),
            (16, 256, [16, 16, 64], [4, 4, 4]),
        ):
            with self.subTest(rows=rows, k=k, block_sizes=block_sizes):
                args = affine_args(4, 32, rows, 128, k)
                msl = render_msl(affine_mm, args, register_config(block_sizes, threads))
                self.assertIn("threadgroup float _mpp_tile", msl)
                self.assertIn("execution_simdgroups<", msl)
                self.assertNotIn("_mst0", msl)

    def test_operand_types(self) -> None:
        """A register operand exactly representable in half or bfloat takes
        it, widened or not, another is float; a view is its tensor's."""
        x, wq, s = nvfp4_cpu_args(16)
        msl = render_msl(nvfp4_rows, (x, wq, s), register_config([16, 32, 64]))
        self.assertIn("get_right_input_cooperative_tensor<bfloat, bfloat, float>", msl)
        self.assertRegex(msl, r"auto _mpp_a\d+ = tensor<device bfloat")
        # Every conversion to bfloat in the element loops rounds as lane
        # copies round (helion_bf16_cast), keeping bfloat subnormals that
        # Metal's own cast flushes.
        self.assertRegex(msl, r"_mr_b\d+\[\d+\] = helion_bf16_cast\(")
        self.assertNotIn("static_cast<bfloat>", _k_loop(msl))
        args = affine_args(4, 32, 16, 64, 256)
        msl = render_msl(
            affine_mm, (args[0].float(), *args[1:]), register_config([16, 32, 64])
        )
        self.assertIn("get_right_input_cooperative_tensor<float, float, float>", msl)
        self.assertRegex(msl, r"auto _mpp_a\d+ = tensor<device float")

    def test_narrowed_operands(self) -> None:
        """Operands converted to half from float are both built in registers
        (no device view holds them), over a K loop from 16 with tails along
        M, N and K: elements past the K tile's live length are 0."""
        x, y = torch.randn(13, 40), torch.randn(40, 20)
        msl = render_msl(k_from_16, (x, y), register_config([16, 32, 16]))
        loop = self._assert_registers(msl)
        self.assertIn(
            "auto _mr_a0 = _mpp_op_0.get_left_input_cooperative_tensor<half, half, "
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
        the two-register op takes 16 steps of 32, in two chunks of 8
        (``MAX_UNROLLED_STEPS``)."""
        args = affine_args(4, 32, 8, 64, 1024)
        msl = render_msl(affine_mm, args, register_config([16, 16, 512]))
        loop = self._assert_registers(msl)
        self.assertIn("matmul2d_descriptor(16, 16, 32, false, true, false,", msl)
        self.assertIn("for (short _fc = 0; _fc < 2; _fc++) {", loop)
        self.assertEqual(_runs(loop, "_mr_a", "_mr_b"), list(range(8)))

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
        self.assertRegex(msl, r"_mpp_op_0\.run\(_mr_a\d+, ")

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
        self.assertRegex(msl, r"\} else _mr_a\d+\[0\] = 0;")

    def test_fast_half_families(self) -> None:
        """The affine split's GPU table, read from the device's name: on for
        the M5 family on macOS 26.2 or newer, off for the M1 to M4 and for a
        device whose family isn't listed."""
        for name, macos, fast in (
            ("Apple M5", True, True),
            ("Apple M5 Max", True, True),
            ("Apple M5 Max", False, False),
            ("Apple M4 Max", True, False),
            ("Apple M1", True, False),
            ("Apple M6 Pro", True, False),
            ("Apple Paravirtual device", True, False),
        ):
            with (
                self.subTest(name=name, macos=macos),
                mock.patch.object(
                    torch._C, "_mps_get_name", return_value=name, create=True
                ),
                mock.patch.object(
                    torch.backends.mps, "is_macos_or_newer", return_value=macos
                ),
            ):
                self.assertEqual(coop_layout.fast_half_inputs(), fast)

    def test_affine_split(self) -> None:
        """Where half inputs run faster (forced here), a K tile within one
        group has one scale and bias per weight row: x times the codes in
        half, x's row sums from a product with ones, and each accumulator
        element adds the tile's scale times its product plus the bias times
        its row's sum.  Off where no GPU runs the kernel, without
        fast_math, and over a K tile of two groups."""
        for bits, group, k_block, rows in (
            (4, 32, 32, 16),
            (5, 128, 128, 8),
            (6, 128, 64, 16),
        ):
            with self.subTest(bits=bits, rows=rows):
                args = affine_args(bits, group, rows, 64, 512)
                config = register_config([16, 32, k_block])
                with mock.patch.object(
                    mpp_registers, "affine_split_enabled", return_value=True
                ):
                    msl = render_msl(affine_mm, args, config)
                loop = _k_loop(msl)
                self.assertIn("_mr_ones[_e] = static_cast<half>(1.0f);", loop)
                self.assertIn(
                    "get_right_input_cooperative_tensor<half, half, float>", loop
                )
                self.assertEqual(loop.count("_mpp_op_0.run("), 2 * (k_block // 16))
                # The split's steps keep their order: each step's ops run
                # before the next step's operands are built.
                self.assertLess(
                    loop.index("_mr_ones, _mr_x);"), loop.index("auto _mr_b1 = ")
                )
                self.assertRegex(
                    loop,
                    r"acc_copy_0\[0\] = \(_mr_av \+ \(\(_mpp_rs_0 \* _mr_pv\) "
                    r"\+ \(_mpp_rb_0 \* _mr_xv\)\)\);",
                )
                self.assertRegex(
                    loop, rf"auto v_\d+ = static_cast<int>\(\(offset_2 / {group}\)\);"
                )
                self.assertNotIn("_mr_ones", render_msl(affine_mm, args, config))
        args = affine_args(4, 32, 16, 64, 512)
        with mock.patch.object(
            mpp_registers, "affine_split_enabled", return_value=True
        ):
            msl = render_msl(affine_mm, args, register_config([16, 32, 64]))
        self.assertNotIn("_mr_ones", msl)
        # Where half inputs pay, only under fast_math: the split
        # reassociates the sum.
        fast = helion.kernel(backend="metal", static_shapes=True, fast_math=True)(
            affine_mm.fn
        )
        config = register_config([16, 32, 32])
        with mock.patch.object(mpp_registers, "_half_pays", lambda: True):
            self.assertNotIn("_mr_ones", render_msl(affine_mm, args, config))
            self.assertIn("_mr_ones", render_msl(fast, args, config))
        # Nor, under fast_math, where no GPU runs the kernel.
        self.assertNotIn("_mr_ones", render_msl(fast, args, config))
        # Nor where it wouldn't pay or could overflow: a weight already
        # half, whose op takes half inputs as it is, and a float32 scale and
        # bias, whose products can pass float's range where the weights
        # don't.
        x, w, scales, biases = affine_args(4, 32, 16, 64, 384)
        with mock.patch.object(
            mpp_registers, "affine_split_enabled", return_value=True
        ):
            msl = render_msl(
                dequant_forms_mm,
                (x, w, scales, biases, "half"),
                register_config([16, 32, 32]),
            )
            self.assertIn("get_right_input_cooperative_tensor<half, half, float>", msl)
            self.assertNotIn("_mr_ones", msl)
            wide = (x, w, scales.float(), biases.float())
            msl = render_msl(affine_mm, wide, register_config([16, 32, 32]))
            self.assertIn("_mr_b", msl)
            self.assertNotIn("_mr_ones", msl)
            # An fp16 scale and bias the float32 codes promote split; a
            # product rounded to bf16 on the way, or a group index that
            # isn't constant over an aligned tile (k // -32), doesn't.
            x, w, scales, biases = affine_args(4, 32, 16, 64, 288)
            for form, split in (
                ("promoted", True),
                ("bf16_product", False),
                ("negative_group", False),
            ):
                with self.subTest(form=form):
                    wide_scales = torch.randn(64, 10).to(torch.float16)
                    msl = render_msl(
                        dequant_forms_mm,
                        (x, w, wide_scales, wide_scales.clone(), form),
                        register_config([16, 32, 32]),
                    )
                    self.assertIn("_mr_b", msl)
                    self.assertEqual("_mr_ones" in msl, split)
        # On a layout read at run time, an accumulator element the lane
        # doesn't hold keeps its value.
        args = affine_args(4, 32, 16, 64, 256)
        with (
            mock.patch.object(mpp_registers, "affine_split_enabled", return_value=True),
            mock.patch.object(mpp_registers, "factor", lambda coords, d: None),
        ):
            msl = render_msl(affine_mm, args, register_config([16, 32, 32]))
        self.assertRegex(
            msl,
            r"if \(_mrl\d+_v0\) \{\n\s+\{\n\s+int _e_m = 0 \+ _mrl\d+_r0;\n"
            r"\s+int _e_n = 0 \+ _mrl\d+_c0;\n\s+int _e_k = 0;\n\s+float _mr_av",
        )


class TestRegisterSplit(TestCase):
    """An MPP config whose tiles take one SIMD group and whose ``num_warps``
    is above 1 splits its register runs' K loops over that many SIMD groups
    (``mpp_registers.register_split``); where the split can't apply, the
    config keeps its one SIMD group, as before."""

    def test_split(self) -> None:
        """Eight SIMD groups each run the tile's 32-thread layout over every
        eighth K tile from their own register operands; after the loop the
        first adds the others' accumulators through threadgroup memory, in
        SIMD-group order, and alone stores."""
        args = affine_args(4, 32, 16, 64, 4096)
        msl = render_msl(affine_mm, args, register_config([16, 16, 64], num_warps=8))
        self.assertIn("[[required_threads_per_threadgroup(4, 8, 8)]]", msl)
        self.assertIn("uint _frag_sg [[simdgroup_index_in_threadgroup]]", msl)
        self.assertIn(
            "const uint3 tid = uint3(_mpp_lane % 4, (_mpp_lane / 4) % 8, "
            "_mpp_lane / 32);",
            msl,
        )
        self.assertIn("threadgroup float _frag_split[1792];", msl)
        self.assertIn(
            "for (int offset_2 = _frag_sg * _BLOCK_SIZE_2; offset_2 < 4096; "
            "offset_2 += 8 * _BLOCK_SIZE_2) {",
            msl,
        )
        self.assertRegex(msl, r"_mpp_op_0\.run\(_mr_a(\d+), _mr_b\1, acc_copy_0\);")
        tail = msl[msl.index("    threadgroup_barrier(") :]
        for e in range(8):
            self.assertIn(
                f"_frag_split[((_frag_sg - 1) * 8 + {e}) * 32 + _mpp_lane] = acc[{e}];",
                tail,
            )
            self.assertIn(f"acc[{e}] = 0.0f;", tail)
            self.assertIn(
                f"acc[{e}] += _frag_split[(_s * 8 + {e}) * 32 + _mpp_lane];", tail
            )
        self.assertIn("for (uint _s = 0; _s < 7; _s++) {", tail)
        self.assertIn("if (_frag_sg == 0 && ((mask_0 & mask_1))) {", tail)
        # x read as a device view at 8 rows: each SIMD group's tile offset.
        args = affine_args(6, 128, 8, 64, 4096)
        msl = render_msl(affine_mm, args, register_config([16, 16, 128], num_warps=16))
        self.assertIn("offset_2 += 16 * _BLOCK_SIZE_2) {", msl)
        self.assertRegex(msl, r"auto _mpp_a\d+ = tensor<device half")
        self.assertRegex(msl, r"_mpp_op_0\.run\(_mpp_a(\d+), _mr_b\1, acc_copy_0\);")
        # With the affine split, each SIMD group's epilogue adds its own
        # tiles' terms before the combine.
        args = affine_args(4, 32, 16, 64, 4096)
        config = register_config([16, 32, 32], num_warps=4)
        with mock.patch.object(
            mpp_registers, "affine_split_enabled", return_value=True
        ):
            msl = render_msl(affine_mm, args, config)
        self.assertIn("offset_2 += 4 * _BLOCK_SIZE_2) {", msl)
        self.assertIn("get_right_input_cooperative_tensor<half, half, float>", msl)
        self.assertIn("_mr_ones", msl)
        tail = msl[msl.index("    threadgroup_barrier(") :]
        self.assertIn("for (uint _s = 0; _s < 3; _s++) {", tail)

    def test_normalize(self) -> None:
        """An MPP config whose counts launch one SIMD group and that leaves
        num_warps out gets 1, as such configs ran before the split, rather
        than the default; one that names it keeps it, and a launch of more
        SIMD groups keeps the default, as fragment configs do."""
        bound = affine_mm.bind(affine_args(4, 32, 16, 64, 1024))
        nested = {"block_sizes": [[16, 16], 64], "num_threads": [4, 8, 1]}
        grouped = {"block_sizes": [16, 16, 64], "num_threads": [[4, 8], 1]}
        for config, warps in (
            (register_config([16, 16, 64]), 1),
            (register_config([16, 16, 64], num_warps=8), 8),
            (register_config([16, 64, 64], [4, 16, 1]), DEFAULT_NUM_WARPS),
            # Grouped sequences read as their flat form.
            (helion.Config.from_dict(nested), 1),
            (helion.Config.from_dict(grouped), 1),
        ):
            with self.subTest(config=config):
                normalized = helion.Config.from_dict(dict(config.config))
                bound.config_spec.normalize(normalized)
                self.assertEqual(normalized.config["num_warps"], warps)
        # A reduction's threads come from the num_warps budget too, and it
        # refuses the split: the default stays.
        x = torch.randn(16, 256).to(torch.float16)
        bound = rowsum_mm.bind((x, torch.randn(256, 32), torch.randn(16, 8)))
        normalized = helion.Config.from_dict(dict(register_config([16, 16, 32]).config))
        bound.config_spec.normalize(normalized)
        self.assertEqual(normalized.config["num_warps"], DEFAULT_NUM_WARPS)
        # Counts that launch one SIMD group only once a bounded block is
        # clamped (K to the N block's 16, its 32 threads to 16) get 1 too,
        # so codegen doesn't split them.
        args = (x, torch.randn(256, 256))
        given = {"block_sizes": [16, 16, 64], "num_threads": [1, 2, 32]}
        normalized = helion.Config.from_dict(dict(given))
        bounded_k_mm.bind(args).config_spec.normalize(normalized)
        self.assertEqual(normalized.config["num_warps"], 1)
        msl = render_msl(bounded_k_mm, args, helion.Config.from_dict(dict(given)))
        self.assertNotIn("_frag_sg", msl)

    def test_unmodelled_launch(self) -> None:
        """No split where normalization doesn't see the launch codegen
        builds: beside a free hl.arange axis, whose threads come from the
        num_warps budget, and in a kernel of two roots, where a config that
        leaves num_warps out keeps the default 4.  Both keep one SIMD
        group's register run."""
        x = torch.randn(64, 256).to(torch.float16)
        config = helion.Config.from_dict(
            {"block_sizes": [16, 16, 32], "num_threads": [16, 2, 1], "num_warps": 8}
        )
        msl = render_msl(free_cols_mm, (x, torch.randn(256, 16)), config)
        self.assertIn("[[required_threads_per_threadgroup(2, 16, 1)]]", msl)
        self.assertIn("_mr_b", msl)
        self.assertNotIn("_frag_split", msl)
        args = (x[:16], torch.randn(256, 32), torch.randn(64))
        config = helion.Config.from_dict(
            {"block_sizes": [16, 32, 32, 32], "num_threads": [4, 8, 1, 4]}
        )
        normalized = helion.Config.from_dict(dict(config.config))
        mm_and_double.bind(args).config_spec.normalize(normalized)
        self.assertEqual(normalized.config["num_warps"], DEFAULT_NUM_WARPS)
        msl = render_msl(mm_and_double, args, config)
        self.assertIn("execution_simdgroup> _mpp_op_0;", msl)
        self.assertNotIn("_frag_split", msl)
        # Nor at counts partly on auto, which normalization doesn't take
        # for one SIMD group, so num_warps keeps the default 4.
        config = helion.Config.from_dict(
            {"block_sizes": [16, 16, 64], "num_threads": [2, 0, 1]}
        )
        affine = affine_args(4, 32, 16, 64, 1024)
        msl = render_msl(affine_mm, affine, config)
        self.assertIn("execution_simdgroup> _mpp_op_0;", msl)
        self.assertNotIn("_frag_split", msl)

    def test_fill_and_accumulators(self) -> None:
        """Under a split only the first SIMD group starts from a fill other
        than 0, and after the K loop one combine per accumulator adds it up,
        each through one accumulator's slots."""
        x = torch.randn(16, 256).to(torch.float16)
        y = torch.randn(256, 32)
        config = register_config([16, 32, 32], num_warps=4)
        msl = render_msl(filled_mm, (x, y), config)
        self.assertIn("acc[_e] = (_frag_sg == 0 ? 0.5f : 0.0f);", msl)
        msl = render_msl(two_acc_mm, (x, y, torch.randn(256, 32)), config)
        self.assertEqual(msl.count("threadgroup_barrier("), 4)
        self.assertIn("threadgroup float _frag_split[1536];", msl)
        for line in (
            "_frag_split[((_frag_sg - 1) * 16 + 0) * 32 + _mpp_lane] = a[0];",
            "_frag_split[((_frag_sg - 1) * 16 + 0) * 32 + _mpp_lane] = b[0];",
            "b[15] += _frag_split[(_s * 16 + 15) * 32 + _mpp_lane];",
        ):
            self.assertIn(line, msl)

    def test_fallback(self) -> None:
        """Where the split can't apply the config keeps the code it had: as
        on fewer SIMD groups, which a launch whose counts it takes doesn't
        change.  An accumulator seeded from a tile, a K loop that carries a
        count beside its accumulator or writes memory
        (``operand_elements.split_refusal``), an accumulator read in the
        Scalar layout or beside a staged row vector (both through
        threadgroup memory) and a combine over more threadgroup memory than
        there is (a 32x32 tile on 16 SIMD groups) keep one SIMD group's
        register run; a tile whose threads span two SIMD groups splits the
        tile over them (``mpp_registers._grid_run``), not K."""
        x = torch.randn(16, 256).to(torch.float16)
        y = torch.randn(256, 32)
        affine = affine_args(4, 32, 32, 64, 1024)
        for name, kernel, args, block_sizes, threads, warps in (
            ("seeded", seeded_rhs, (x, y, torch.randn(16, 32)), [16, 32, 32], None, 1),
            ("counted", counted_mm, (x, y), [16, 32, 32], None, 1),
            ("copying", copying_mm, (x, y, torch.empty_like(x)), [16, 32, 32], None, 1),
            ("materialized", transposed_out_mm, (x, y), [16, 32, 32], None, 1),
            (
                "row vector",
                row_scaled_mm,
                (x, y, torch.randn(4, 32)),
                [16, 32, 32],
                None,
                1,
            ),
            ("combine", affine_mm, affine, [32, 32, 64], None, 1),
            ("two simdgroups", affine_mm, affine, [16, 64, 64], [4, 16, 1], 2),
        ):
            with self.subTest(name):
                msl = render_msl(
                    kernel, args, register_config(block_sizes, threads, num_warps=16)
                )
                self.assertNotIn("offset_2 = _frag_sg", msl)
                self.assertEqual("_mst_tile" in msl, name == "two simdgroups")
                self.assertEqual(
                    msl,
                    render_msl(
                        kernel,
                        args,
                        register_config(block_sizes, threads, num_warps=warps),
                    ),
                )
                self.assertIn(
                    "threadgroup float _mst_tile_float" if threads else "_mr_b", msl
                )


def _nax_right(atom: coop_layout.Atom, on_gpu: bool) -> coop_layout.Layout:
    """The M4's layout with the right input of an M5 Max's, as probed there
    (MLX's M5 GEMM fragments): lane ``l`` holds K positions ``kb .. kb + 3`` of
    N ``nb + 8 q``, ``kb = 4 (l & 1) + 8 ((l >> 3) & 1)`` and ``nb = ((l >> 1) &
    1) + 2 ((l >> 2) & 1) + 4 ((l >> 4) & 1)``."""
    probed = coop_layout.blocked_layout(atom)
    if atom.right_device:
        return probed
    lanes = []
    for lane in range(32):
        kb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1)
        nb = ((lane >> 1) & 1) + 2 * ((lane >> 2) & 1) + 4 * ((lane >> 4) & 1)
        lanes.append(
            tuple((kb + j, 8 * q + nb) for q in range(atom.n // 8) for j in range(4))
        )
    return dataclasses.replace(probed, right=tuple(lanes))


class TestCodeWindows(TestCase):
    """A lane's run of packed codes read from one window of words
    (``bit_windows``)."""

    def test_windows(self) -> None:
        """On an M5's layout (mocked), each lane's 4 adjacent K positions of a
        weight row read their codes from one window: the words its first
        code's bit lies in, shifted right by that bit's offset in its word,
        the codes its bits at 0, b, 2b and 3b.  A word that may be the last
        code's next word is read clamped as that code reads it (3, 5 and 7
        bits; 6-bit runs end in their last code's word); 7-bit runs take a
        ``ulong`` of three words.  The elements keep no division, remainder
        or word load of their own."""
        for bits, window, clamped in (
            (3, "uint", "_mbw0_w + 1"),
            (5, "uint", "_mbw0_w + 1"),
            (6, "uint", None),
            (7, "ulong", "_mbw0_w + 2"),
        ):
            with self.subTest(bits=bits):
                args = affine_args(bits, 128, 32, 64, 1024)
                config = register_config([32, 32, 128], num_warps=8)
                with mock.patch.object(mpp_registers, "layout", _nax_right):
                    loop = _k_loop(render_msl(affine_mm_f16, args, config))
                # 8 steps of 16, each a window per row of the lane's 4.
                names = re.findall(rf"const {window} (_mbw\d+) = \(", loop)
                self.assertEqual(len(names), 32)
                codes = re.findall(
                    r"static_cast<int>\(\(\((_mbw\d+) >> (\d+)\) & (\d+)\)\)", loop
                )
                self.assertEqual(len(codes), 8 * 16)
                for name in names:
                    self.assertEqual(
                        [(int(o), int(m)) for n, o, m in codes if n == name],
                        [(bits * j, (1 << bits) - 1) for j in range(4)],
                    )
                for absent in ("floor_divide", "remainder", "w["):
                    self.assertNotIn(absent, loop)
                self.assertIn(
                    f"const int _mbw0_x = ((_mrl0 * {bits}) + (offset_2 * {bits}));",
                    loop,
                )
                self.assertEqual(
                    clamped is not None,
                    f"metal::min({clamped}, {1024 * bits // 16 - 1})" in loop,
                )
                if bits == 6:
                    self.assertIn(
                        "const uint _mbw0 = (uint(_mbw0_p[_mbw0_w]) | "
                        "(uint(_mbw0_p[_mbw0_w + 1]) << 16)) >> (_mbw0_x & 15);",
                        loop,
                    )

    def test_windows_read_through_a_device_pointer(self) -> None:
        """A window loads its words through a ``const device`` pointer, which
        MLX's ``constant`` binding of a small input cannot take: the packed
        weight is among the kernel's device reads, which export checks
        (``export._check_mlx_buffers``)."""
        args = affine_args(3, 128, 32, 64, 1024)
        config = register_config([32, 32, 128], num_warps=8)
        with mock.patch.object(mpp_registers, "layout", _nax_right):
            (launch,) = capture_metal_launches(affine_mm_f16, args, config)
        source = launch.msl.render()
        self.assertTrue("const device ushort* _mbw0_p" in source, "no window")
        self.assertIn("w", launch.msl.device_views)

    def test_word_forms(self) -> None:
        """On an M5's layout (mocked), windows over 8-bit words, three bytes
        whose last is read clamped to the row as the last code reads its next
        word, and over 16-bit words a row holds one more of, whose next word
        reads unclamped."""
        for words, padded, line in (
            (
                torch.uint8,
                False,
                (
                    "const uint _mbw0 = (uint(_mbw0_p[_mbw0_w]) | (uint(_mbw0_p[_mbw0_w "
                    "+ 1]) << 8) | (uint(_mbw0_p[metal::min(_mbw0_w + 2, 639)]) << 16)) "
                    ">> (_mbw0_x & 7);"
                ),
            ),
            (
                torch.int16,
                True,
                (
                    "const uint _mbw0 = (uint(_mbw0_p[_mbw0_w]) | (uint(_mbw0_p[_mbw0_w "
                    "+ 1]) << 16)) >> (_mbw0_x & 15);"
                ),
            ),
        ):
            with self.subTest(words=words):
                args = affine_args(5, 64, 32, 64, 1024, words=words, padded=padded)
                config = register_config([32, 32, 128], num_warps=8)
                with mock.patch.object(mpp_registers, "layout", _nax_right):
                    loop = _k_loop(render_msl(affine_mm_f16, args, config))
                self.assertEqual(len(re.findall(r"const uint _mbw\d+ = \(", loop)), 32)
                self.assertIn(line, loop)
                for absent in ("floor_divide", "remainder", "w["):
                    self.assertNotIn(absent, loop)

    def test_kept_reads(self) -> None:
        """Codes that no window serves keep their elements' reads: on the
        M4's layout a lane's codes of a row lie 8 K positions apart, a
        5-word window for two 6-bit codes; 4-bit codes don't straddle and
        their shifts fold to constants already."""
        for bits in (4, 6):
            with self.subTest(bits=bits):
                args = affine_args(bits, 128, 32, 64, 1024)
                loop = _k_loop(
                    render_msl(
                        affine_mm_f16, args, register_config([32, 32, 128], num_warps=8)
                    )
                )
                self.assertNotIn("_mbw", loop)
                self.assertIn("w[", loop)
                self.assertEqual("floor_divide" in loop, bits == 6)

    def test_plan(self) -> None:
        """A window loads fewer words than its codes would, two each, and
        fits a ``ulong``; a word it clamps, which another lane's code may
        read unclamped, takes the facts' proof that no code's first word
        lies past the clamp."""
        facts = {
            "r": Fact(1, 0, 63),
            "_mrl0": Fact(4, 0, 12),
            "offset_2": Fact(128, 0, 896),
        }

        def run(
            bits: int, offsets: list[int], clamp: int | None = 319, **extra: Fact
        ) -> list[bit_windows.Window]:
            codes = [
                [
                    bit_windows.Code(
                        f"q{i}",
                        "w",
                        "ushort",
                        16,
                        Affine.term("r", 1024 * bits // 16),
                        Affine(
                            Fraction(offset),
                            (("_mrl0", Fraction(bits)), ("offset_2", Fraction(bits))),
                        ),
                        bits,
                        None if clamp is None else Affine.of(clamp),
                    )
                ]
                for i, offset in enumerate(offsets)
            ]
            names = iter(f"_w{i}" for i in range(99))
            return bit_windows.plan_windows(
                codes, {**facts, **extra}, lambda: next(names)
            )

        (window,) = run(5, [0, 5, 10, 15])
        self.assertEqual((window.count, window.clamped), (2, (False, True)))
        self.assertEqual(
            window.members, ((0, "q0", 0), (1, "q1", 5), (2, "q2", 10), (3, "q3", 15))
        )
        (window,) = run(6, [0, 6, 12, 18], clamp=383)
        self.assertEqual((window.count, window.clamped), (2, (False, False)))
        # Unbounded offsets: the clamped word goes unproven, an unclamped
        # one needs no proof.
        unbounded = {"offset_2": Fact(128, 0, None)}
        self.assertEqual(run(5, [0, 5, 10, 15], **unbounded), [])
        (window,) = run(5, [0, 5, 10, 15], clamp=None, **unbounded)
        self.assertEqual(window.clamped, (False, False))
        # A 5-bit code that may straddle takes its own two words; a 6-bit
        # one whose first bit is 0 or 8 of a word never straddles: one.
        self.assertEqual(run(5, [0]), [])
        (window,) = run(6, [0], clamp=383)
        self.assertEqual(window.count, 1)
        # Two codes 48 bits apart, as an M4's lane holds them, from any even
        # bit of a word: five words.
        m4 = {"_mrl0": Fact(1, 0, 7)}
        self.assertEqual(run(6, [0, 48], clamp=383, **m4), [])


class TestRolledSteps(TestCase):
    """A K iteration unrolls at most ``MAX_UNROLLED_STEPS`` steps of its
    register runs; past it a run loops over chunks of its steps
    (``mpp_registers.emit_register_run``)."""

    def test_unrolled_chunk(self) -> None:
        """Up to 8 steps of a K iteration's runs unroll; past them each run
        unrolls the largest divisor of its steps within its share of 8, in
        proportion to its steps."""
        self.assertEqual(
            [mpp_registers.unrolled_chunk(s, 8) for s in (2, 2, 4)], [2, 2, 4]
        )
        self.assertEqual(mpp_registers.unrolled_chunk(16, 16), 8)
        self.assertEqual(mpp_registers.unrolled_chunk(8, 16), 4)
        self.assertEqual(mpp_registers.unrolled_chunk(6, 12), 3)
        self.assertEqual(mpp_registers.unrolled_chunk(4, 32), 1)

    def test_chunks(self) -> None:
        """Two runs of 4 steps stay unrolled; two runs of 8 loop over two
        chunks of 4 each, each step's ``_fs`` its chunk's start plus its
        place; one run of 16 steps loops over two chunks of 8."""
        x = torch.randn(32, 512).to(torch.float16)
        y, z = torch.randn(512, 32), torch.randn(512, 32)
        msl = render_msl(two_acc_mm, (x, y, z), register_config([32, 32, 64]))
        self.assertNotIn("_fc", msl)
        msl = render_msl(two_acc_mm, (x, y, z), register_config([32, 32, 128]))
        self.assertEqual(msl.count("for (short _fc = 0; _fc < 2; _fc++) {"), 2)
        self.assertIn("const short _fs = _fc * 4 + 3;", msl)
        self.assertNotIn("_fc * 4 + 4", msl)
        args = affine_args(4, 32, 32, 64, 1024)
        msl = render_msl(affine_mm, args, register_config([32, 32, 256]))
        self.assertEqual(msl.count("#pragma clang loop unroll(disable)"), 1)
        self.assertIn("const short _fs = _fc * 8 + 7;", msl)

    def test_windows(self) -> None:
        """A rolled chunk's steps still read their packed codes from windows
        (``bit_windows``), the chunk index folded into the first code's bit:
        6-bit codes over 16 steps on an M5's layout (mocked)."""
        args = affine_args(6, 128, 32, 64, 1024)
        config = register_config([32, 32, 256], num_warps=4)
        with mock.patch.object(mpp_registers, "layout", _nax_right):
            loop = _k_loop(render_msl(affine_mm_f16, args, config))
        self.assertEqual(loop.count("for (short _fc = 0; _fc < 2; _fc++) {"), 1)
        self.assertEqual(len(re.findall(r"const uint _mbw\d+ = \(", loop)), 32)
        self.assertIn(
            "const int _mbw0_x = (((_fc * 768) + (_mrl0 * 6)) + (offset_2 * 6));", loop
        )
        for absent in ("floor_divide", "remainder", "w["):
            self.assertNotIn(absent, loop)


class TestPartners(TestCase):
    """An element-wise epilogue over the results of two ops of one type
    (``mpp_matmul._classify_partner_uses``)."""

    def _args(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.randn(32, 128).to(torch.float16),
            torch.randn(128, 32),
            torch.randn(128, 32),
        )

    def test_read_at_the_slot(self) -> None:
        """The element loop runs over the first accumulator and reads the
        second at its own slot, with no materialization: in a register run,
        unsplit and split over 2 SIMD groups (which the kernel's
        ``register_splits`` fact allows, though the fragment lowering refuses
        the kernel), and on the staged op."""
        for name, config, threads in (
            ("register", register_config([32, 32, 32]), "(4, 8, 1)"),
            (
                "split",
                register_config([32, 32, 32], [1, 1, 32], num_warps=2),
                "(32, 1, 2)",
            ),
            ("staged", helion.Config(block_sizes=[32, 32, 32], num_warps=4), None),
        ):
            with self.subTest(name):
                msl = render_msl(gated_mm, self._args(), config)
                self.assertNotIn("_mpp_mat", msl)
                self.assertIn("static_cast<half>(b[_ei])", msl)
                if threads is not None:
                    self.assertIn(f"required_threads_per_threadgroup{threads}", msl)
        bound = gated_mm.bind(self._args())
        self.assertTrue(bound.config_spec.backend_facts.register_splits)

    def test_one_layout(self) -> None:
        """Partners take one run, which keeps their ops of one type: a staged
        vector on the first keeps both on the staged op on a launch of
        several SIMD groups (grid runs for all or none), and on one SIMD
        group both take register runs; where the affine split takes the
        first's float op, neither takes it."""
        x = torch.randn(32, 256).to(torch.float16)
        y, z, v = torch.randn(256, 64), torch.randn(256, 64), torch.randn(4, 64)
        for name, config, run in (
            ("grid", register_config([32, 64, 64], [8, 16, 1]), "execution_simdgroups"),
            ("register", register_config([32, 32, 32]), "execution_simdgroup>"),
        ):
            with self.subTest(name):
                msl = render_msl(scaled_gated_mm, (x, y, z, v), config)
                self.assertIn("_mpp_vec", msl)
                self.assertNotIn("_mst_tile", msl)
                self.assertIn(run, msl)
                self.assertIn("static_cast<half>(b[_ei])", msl)
                self.assertNotIn("_mpp_mat", msl)
        args = (*affine_args(4, 32, 16, 64, 512), torch.randn(512, 64))
        for name, config in (
            ("register", register_config([16, 32, 32])),
            ("split", register_config([16, 32, 32], [1, 1, 32], num_warps=2)),
        ):
            with (
                self.subTest(name),
                mock.patch.object(
                    mpp_registers, "affine_split_enabled", return_value=True
                ),
            ):
                msl = render_msl(affine_gated_mm, args, config)
                self.assertNotIn("_mr_ones", msl)
                self.assertIn("static_cast<half>(b[_ei])", msl)

    def test_unread_together(self) -> None:
        """Two ops of one layout key whose results no op reads together aren't
        partners: the first keeps its affine split beside the second's float
        op."""
        args = (
            *affine_args(4, 32, 32, 64, 256),
            torch.randn(256, 64),
            torch.randn(4, 64),
        )
        with mock.patch.object(
            mpp_registers, "affine_split_enabled", return_value=True
        ):
            msl = render_msl(apart_mm, args, register_config([32, 32, 32]))
        self.assertIn("_mr_ones", msl)
        self.assertNotIn("[_ei]", msl)

    def test_register_types_apart(self) -> None:
        """Float operands whose register runs take two types (the gate's
        weight rounded to half) aren't partners: the epilogue reads both
        results through materializations, on one SIMD group and with the K
        loop split over two."""
        x = torch.randn(32, 256).to(torch.float16)
        y, z = torch.randn(256, 32), torch.randn(256, 32)
        for config in (
            register_config([32, 32, 32]),
            register_config([32, 32, 32], [1, 1, 32], num_warps=2),
        ):
            with self.subTest(config=config):
                msl = render_msl(narrowed_gated_mm, (x, y, z), config)
                self.assertIn(
                    "get_right_input_cooperative_tensor<half, half, float>", msl
                )
                self.assertIn(
                    "get_right_input_cooperative_tensor<half, float, float>", msl
                )
                self.assertNotIn("[_ei]", msl)
                self.assertIn("_mpp_mat", msl)

    def test_partner_seed(self) -> None:
        """A value one partner's op updates in place, its seed, reaches the
        other's element loop through its snapshot, never the accumulator
        read at its slot after the op: a + b0 beside a b reads b0's
        materialization, which an element loop refuses, as before the
        partners."""
        x = torch.randn(32, 64).to(torch.float16)
        y, z = torch.randn(64, 32), torch.randn(64, 32)
        for config in (
            register_config([32, 32, 32]),
            helion.Config(block_sizes=[32, 32, 32], num_warps=4),
        ):
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(
                    exc.BackendUnsupported, "materialized MPP result"
                ),
            ):
                render_msl(seed_read_mm, (x, y, z), config)

    def test_seed_beside_partners(self) -> None:
        """An op reading partners and one's seed, directly or through a value
        computed from it, stays apart from them (its readers read a and b's
        materializations: a b + b0, a b + (b0 + 1)), and a partner
        read at its slot inside a cast takes its value type (decltype(+g[_ei])),
        not the reference."""
        x = torch.randn(32, 64).to(torch.float16)
        y, z = torch.randn(64, 32), torch.randn(64, 32)
        for config in (
            register_config([32, 32, 32]),
            helion.Config(block_sizes=[32, 32, 32], num_warps=4),
        ):
            with self.subTest(config=config):
                msl = render_msl(seed_join_mm, (x, y, z), config)
                self.assertIn("_mpp_mat", msl)
                self.assertNotIn("[_ei]", msl)
                msl = render_msl(silu_gated_mm, (x, y, z), config)
                self.assertIn("decltype(+g[_ei])", msl)
                self.assertIsNone(re.search(r"decltype\(\w+\[_ei\]\)", msl))

    def test_kept_apart(self) -> None:
        """Ops of one type whose results an op reads together stay apart where
        the owner's element loop, after every op's write, can't evaluate it:
        a value read from either partner's carry before its op writes it
        (its snapshot read instead), and an epilogue reading a tensor the
        kernel writes after it (its results materialized)."""
        x = torch.randn(32, 64).to(torch.float16)
        y, z = torch.randn(64, 32), torch.randn(64, 32)
        config = helion.Config(block_sizes=[32, 32, 32], num_warps=4)
        for kernel, args in (
            (early_a_mm, (x, y, z)),
            (early_b_mm, (x, y, z)),
            (rewriting_mm, (x, y, z, torch.randn(32, 32))),
        ):
            with self.subTest(kernel.name):
                msl = render_msl(kernel, args, config)
                self.assertNotIn("[_ei]", msl)
                self.assertIn("_mpp_mat", msl)

    def test_other_types(self) -> None:
        """Ops of two types may lay their elements out differently: the
        epilogue reads both results through materializations."""
        msl = render_msl(mixed_gated_mm, self._args(), register_config([32, 32, 32]))
        self.assertIn("_mpp_mat", msl)
        self.assertNotIn("[_ei])", msl)


if __name__ == "__main__":
    import unittest

    unittest.main()
