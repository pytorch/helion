"""MPP's register operands (``mpp_registers``) on the GPU: the probed
layouts, and numerics against float64 for 4-, 5- and 6-bit affine weights
at 4 to 32 rows, with x built in registers or read as a device view, with
and without the affine split, and the register run's other forms."""

from __future__ import annotations

import re
import sys
from typing import cast
import unittest
from unittest import mock

import pytest
import torch

import helion
from helion._compiler.metal import coop_layout
from helion._compiler.metal import mpp_registers
from helion._testing import pairwise
from helion._testing import skipUnlessBackends
from helion._testing.metal import run_with_msl
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_fragment import affine_mm_f16
from helion._testing.metal_fragment import affine_reference
from helion._testing.metal_fragment import k_from_16
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_fragment import seeded_rhs
from helion._testing.metal_registers import affine_gated_mm
from helion._testing.metal_registers import apart_mm
from helion._testing.metal_registers import dequant_forms_mm
from helion._testing.metal_registers import dequant_reference
from helion._testing.metal_registers import early_a_mm
from helion._testing.metal_registers import early_b_mm
from helion._testing.metal_registers import filled_mm
from helion._testing.metal_registers import gated_mm
from helion._testing.metal_registers import inner_tiled_mm
from helion._testing.metal_registers import inner_tiled_weight_mm
from helion._testing.metal_registers import narrowed_gated_mm
from helion._testing.metal_registers import offset_k_mm
from helion._testing.metal_registers import register_config
from helion._testing.metal_registers import rewriting_mm
from helion._testing.metal_registers import scaled_gated_mm
from helion._testing.metal_registers import seed_join_mm
from helion._testing.metal_registers import shifted_x_mm
from helion._testing.metal_registers import silu_gated_mm
from helion._testing.metal_registers import staged_apart_mm
from helion._testing.metal_registers import two_acc_mm
from helion._testing.metal_registers import unread_x

DEVICE = "mps"

# Metal kernels on macOS only: other backends' suites skip the module.
pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests require macOS"),
    skipUnlessBackends(["metal"]),
]
_requires_mps = unittest.skipUnless(
    sys.platform == "darwin" and torch.backends.mps.is_available(),
    "requires a Metal GPU",
)

#: The affine weights' cases, all pairs of: the bit width (4 divides a
#: word, 5 straddles words 4 times in 16, 6 twice in 8), x's rows (at 8 or
#: fewer x is a device view and the op takes 8 rows; at 16 and 32 both
#: operands are in registers on an M4), and the K tile's form: two groups of
#: 32, which the affine split doesn't take, or one group, unsplit (the M4's
#: default) or split (forced on).
_AFFINE = pairwise(
    bits=(4, 5, 6), rows=(4, 8, 16, 32), mode=("two groups", "one group", "split")
)


#: The K split's cases, all pairs of: the bit width, x's rows (8, where x
#: is a device view, 16 and 32) and the SIMD groups that split K (2, 8 and
#: 16).
_SPLIT = pairwise(bits=(4, 5, 6), rows=(8, 16, 32), num_warps=(2, 8, 16))

#: The grid runs' cases (``mpp_registers._grid_run``): the bit width, x's
#: rows (every tile whole) and the tile with its threads, over 2 to 8 SIMD
#: groups in sub-tiles of 16 x 16 (a K of 32), 16 x 32, 32 x 16 and 32 x 32.
_GRID = (
    (4, 16, [16, 32, 64], [4, 8, 2]),
    (5, 16, [16, 64, 64], [4, 16, 1]),
    (6, 32, [32, 64, 64], [8, 16, 1]),
    (4, 64, [64, 64, 64], [8, 16, 1]),
    (5, 64, [64, 128, 32], [8, 16, 2]),
)


def _run(
    kernel: helion.Kernel, args: tuple[object, ...], config: helion.Config
) -> tuple[torch.Tensor, str]:
    out, _, msl = run_with_msl(kernel, args, config, module_key=repr(config))
    return cast("torch.Tensor", out), msl


@_requires_mps
class TestMppRegisters(unittest.TestCase):
    def _close(self, out: torch.Tensor, ref: torch.Tensor, tol: float = 2e-3) -> None:
        """Within fp16 rounding of the largest output."""
        scale = ref.abs().max().item()
        torch.testing.assert_close(
            out.cpu().double(), ref.double(), atol=tol * scale, rtol=tol
        )

    def test_probe(self) -> None:
        """The probed layout holds every element of each cooperative tensor
        on exactly one lane, factors, and is the M4's on an M4; a device
        input has none."""
        for atom in (
            coop_layout.Atom(16, 32, 16, "half", "float", "float"),
            coop_layout.Atom(32, 32, 16, "half", "half", "float"),
            coop_layout.Atom(8, 32, 16, "half", "float", "float", True),
        ):
            probed = coop_layout.probe(atom)
            for operand in coop_layout.OPERANDS:
                coords = probed.of(operand)
                if operand == "left" and atom.left_device:
                    self.assertEqual(coords, tuple(() for _ in range(32)))
                    continue
                rows, cols = {
                    "left": (atom.m, atom.k),
                    "right": (atom.k, atom.n),
                    "dest": (atom.m, atom.n),
                }[operand]
                self.assertEqual(
                    sorted(c for lane in coords for c in lane),
                    [(r, c) for r in range(rows) for c in range(cols)],
                )
                for d in (0, 1):
                    self.assertIsNotNone(coop_layout.factor(coords, d))
            if "M4" in torch.backends.mps.get_name():  # pyrefly: ignore[missing-attribute]
                self.assertEqual(probed, coop_layout.blocked_layout(atom))

    def test_affine(self) -> None:
        """4-, 5- and 6-bit affine weights at 4 to 32 rows against float64,
        with and without the affine split."""
        for bits, rows, mode in _AFFINE:
            group = 32 if bits == 4 or mode == "two groups" else 128
            k_block = 64 if mode == "two groups" else group
            split = mode == "split"
            with self.subTest(bits=bits, rows=rows, mode=mode):
                args = affine_args(bits, group, rows, 96, 1024, DEVICE)
                config = register_config([16 if rows <= 16 else 32, 32, k_block])
                with mock.patch.object(
                    mpp_registers, "affine_split_enabled", return_value=split
                ):
                    out, msl = _run(affine_mm, args, config)
                self._close(out, affine_reference(args))
                self.assertIn("execution_simdgroup> _mpp_op_0;", msl)
                self.assertEqual("_mr_ones" in msl, split)
                self._assert_view(msl, rows)

    def _assert_view(self, msl: str, rows: int) -> None:
        """x is read as a device view at 8 rows or fewer; past that on an M4,
        whose layout relabels K, it is built in registers (on a GPU whose
        layout keeps a lane's K positions in order it stays a view)."""
        view = re.search(r"auto _mpp_a\d+ = tensor<device half", msl) is not None
        if rows <= 8:
            self.assertTrue(view)
        elif "M4" in torch.backends.mps.get_name():  # pyrefly: ignore[missing-attribute]
            self.assertFalse(view)

    def test_code_windows(self) -> None:
        """Packed codes read from windows of words (``bit_windows``, on an
        M5's layout) give the elementwise reads' outputs exactly, and
        float64's of the half-rounded weights within fp16 rounding: 3- to
        7-bit codes at 16 and 32 rows, K split over 2 to 8 SIMD groups, and
        a K tail, whose steps that may pass K keep their masked reads."""
        windows = coop_layout.gpu_family() == "M5"
        for bits, rows, block, warps, k in (
            (3, 16, [16, 32, 128], 8, 1024),
            (5, 32, [32, 32, 64], 2, 1024),
            (6, 32, [32, 32, 128], 8, 1024),
            (7, 16, [16, 32, 64], 4, 1024),
            (6, 16, [16, 32, 128], 4, 1088),
        ):
            with self.subTest(bits=bits, rows=rows, k=k):
                args = affine_args(bits, 64, rows, 96, k, DEVICE)
                config = register_config(block, num_warps=warps)
                out, msl = _run(affine_mm_f16, args, config)
                with mock.patch.object(mpp_registers, "plan_windows", return_value=[]):
                    plain, _, plain_msl = run_with_msl(
                        affine_mm_f16, args, config, module_key=f"plain {config!r}"
                    )
                windowed = "_mbw" in msl
                if windows:
                    # On an M5's layout every run's codes take a window; on
                    # others only those whose window saves loads do (3-bit
                    # codes 8 K positions apart on an M4's).
                    self.assertTrue(windowed)
                self.assertEqual("_helion_masked(w[" in msl, k != 1024 or not windowed)
                self.assertNotIn("_mbw", plain_msl)
                self.assertTrue(torch.equal(out, cast("torch.Tensor", plain)))
                self._close(out, affine_reference(args, torch.float16))

    def test_word_windows(self) -> None:
        """Windows over 8-bit words and over 16-bit words a row holds one more
        of (the next word read unclamped) give the elements' reads' outputs
        exactly, and float64's of the half-rounded weights within fp16
        rounding."""
        windows = coop_layout.gpu_family() == "M5"
        config = register_config([32, 32, 128], num_warps=8)
        for bits, words, padded in (
            (5, torch.uint8, False),
            (7, torch.uint8, False),
            (5, torch.int16, True),
        ):
            with self.subTest(bits=bits, words=words):
                args = affine_args(bits, 64, 32, 96, 1024, DEVICE, words, padded)
                out, msl = _run(affine_mm_f16, args, config)
                with mock.patch.object(mpp_registers, "plan_windows", return_value=[]):
                    plain, _, plain_msl = run_with_msl(
                        affine_mm_f16, args, config, module_key=f"plain {bits} {words}"
                    )
                if windows:
                    self.assertIn("_mbw", msl)
                self.assertNotIn("_mbw", plain_msl)
                self.assertTrue(torch.equal(out, cast("torch.Tensor", plain)))
                self._close(out, affine_reference(args, torch.float16))

    def test_split(self) -> None:
        """K loops split over 2 to 16 SIMD groups, with x in registers or
        read as a device view and a K tail, and with the affine split,
        against float64."""
        for bits, rows, num_warps in _SPLIT:
            with self.subTest(bits=bits, rows=rows, num_warps=num_warps):
                group = 32 if bits == 4 else 128
                args = affine_args(bits, group, rows, 96, 4096 + group, DEVICE)
                config = register_config(
                    [16 if rows <= 16 else 32, 16, 64], num_warps=num_warps
                )
                out, msl = _run(affine_mm, args, config)
                self._close(out, affine_reference(args))
                self.assertIn(f"offset_2 += {num_warps} * _BLOCK_SIZE_2", msl)
                self._assert_view(msl, rows)
        # With the affine split forced on, over 4 SIMD groups.
        args = affine_args(4, 32, 16, 96, 4096 + 32, DEVICE)
        config = register_config([16, 32, 32], num_warps=4)
        with mock.patch.object(
            mpp_registers, "affine_split_enabled", return_value=True
        ):
            out, msl = _run(affine_mm, args, config)
        self.assertIn("_mr_ones", msl)
        self.assertIn("offset_2 += 4 * _BLOCK_SIZE_2", msl)
        self._close(out, affine_reference(args))

    def test_split_fill_and_accumulators(self) -> None:
        """Split over 4 SIMD groups, an accumulator filled with 0.5 and two
        accumulators of one K loop, against float64."""
        x = torch.randn(16, 256, device=DEVICE).to(torch.float16)
        y = torch.randn(256, 32, device=DEVICE)
        z = torch.randn(256, 32, device=DEVICE)
        config = register_config([16, 32, 32], num_warps=4)
        x64, y64, z64 = (t.cpu().double() for t in (x, y, z))
        out, msl = _run(filled_mm, (x, y), config)
        self.assertIn("_frag_sg == 0 ? 0.5f : 0.0f", msl)
        self._close(out, 0.5 + x64 @ (y64 * 0.5))
        outs, _, msl = run_with_msl(two_acc_mm, (x, y, z), config, module_key="two")
        self.assertIn("b[0];", msl)
        a, b = cast("tuple[torch.Tensor, torch.Tensor]", outs)
        self._close(a, x64 @ (y64 * 0.5))
        self._close(b, x64 @ (z64 * 0.25))

    def test_affine_cancelling_and_tail(self) -> None:
        """The affine split, forced on, against float64: with MLX's biases
        (b = -7.5 s for 4 bits), whose terms cancel, and over a K tail whose
        last group is partial (1040 = 32.5 groups of 32)."""
        config = register_config([16, 32, 32])
        x, w, scales, _ = affine_args(4, 32, 16, 96, 1024, DEVICE)
        biases = (-7.5 * scales.float()).to(torch.float16)
        with mock.patch.object(
            mpp_registers, "affine_split_enabled", return_value=True
        ):
            out, msl = _run(affine_mm, (x, w, scales, biases), config)
        self.assertIn("_mr_ones", msl)
        self._close(out, affine_reference((x, w, scales, biases)))
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(16, 1040, generator=gen).to(torch.float16)
        w = torch.randint(-(2**15), 2**15, (96, 260), generator=gen, dtype=torch.int16)
        scales = (torch.rand(96, 33, generator=gen) * 0.02).to(torch.float16)
        biases = (torch.randn(96, 33, generator=gen) * 0.02).to(torch.float16)
        args = tuple(t.to(DEVICE) for t in (x, w, scales, biases))
        with mock.patch.object(
            mpp_registers, "affine_split_enabled", return_value=True
        ):
            out, msl = _run(dequant_forms_mm, (*args, "promoted"), config)
        self.assertIn("_mr_ones", msl)
        self._close(out, dequant_reference(*args))

    def test_runtime_coordinates(self) -> None:
        """Elements read at run time from the op compute the same."""
        args = affine_args(5, 128, 16, 96, 1024, DEVICE)
        config = register_config([16, 32, 64])
        with mock.patch.object(mpp_registers, "factor", lambda coords, d: None):
            out, msl = _run(affine_mm, args, config)
        self.assertIn("get_multidimensional_index((ushort)0)", msl)
        self._close(out, affine_reference(args))

    def test_grid(self) -> None:
        """Grid runs, the SIMD groups of a launch splitting the tile, against
        float64: 4-, 5- and 6-bit affine weights in sub-tiles of each shape,
        and 4-bit weights dequantized in half, whose tile holds halves."""
        for bits, rows, block_sizes, threads in _GRID:
            with self.subTest(bits=bits, rows=rows, block_sizes=block_sizes):
                group = 32 if bits == 4 else 128
                args = affine_args(bits, group, rows, 256, 512, DEVICE)
                out, msl = _run(affine_mm, args, register_config(block_sizes, threads))
                self.assertIn("threadgroup float _mst_tile_float[", msl)
                self._close(out, affine_reference(args))
        x, w, scales, biases = affine_args(4, 32, 32, 256, 512, DEVICE)
        config = register_config([32, 64, 64], [8, 16, 1])
        out, msl = _run(dequant_forms_mm, (x, w, scales, biases, "half"), config)
        self.assertIn("threadgroup half _mst_tile_half[", msl)
        self._close(out, dequant_reference(x, w, scales, biases, torch.float16))

    def test_grid_bfloat(self) -> None:
        """Grid runs of a bfloat x beside weights dequantized in bfloat over 2
        and 4 SIMD groups, against float64 of the rounded weights."""
        for rows, block_sizes, threads in (
            (32, [32, 64, 64], [8, 8, 1]),
            (64, [64, 64, 64], [8, 16, 1]),
        ):
            with self.subTest(block_sizes=block_sizes, threads=threads):
                x, w, scales, biases = affine_args(4, 32, rows, 256, 512, DEVICE)
                xb = x.to(torch.bfloat16)
                config = register_config(block_sizes, threads)
                out, msl = _run(
                    dequant_forms_mm, (xb, w, scales, biases, "bfloat"), config
                )
                self.assertIn("threadgroup bfloat _mst_tile_bfloat[", msl)
                ref = dequant_reference(xb, w, scales, biases, torch.bfloat16)
                self._close(out, ref)

    def test_grid_views(self) -> None:
        """Against float64: a view of x 8 rows past its tile's offset and a K
        loop beginning at a run-time 64 (both the staged op), and a half
        weight tile of 128 x 128 (a grid run on an unpadded tile)."""
        config = register_config([32, 64, 64], [8, 16, 1])
        x = torch.randn(40, 128, device=DEVICE).to(torch.float16)
        y = torch.randn(128, 64, device=DEVICE)
        w64 = (y.cpu().double() * 0.5).half().double()
        out, msl = _run(shifted_x_mm, (x, y), config)
        self.assertNotIn("_mst_tile", msl)
        self._close(out, x.cpu().double()[8:] @ w64)
        x = torch.randn(32, 256, device=DEVICE).to(torch.float16)
        y = torch.randn(256, 64, device=DEVICE)
        w64 = (y.cpu().double()[64:192] * 0.5).half().double()
        out, msl = _run(offset_k_mm, (x, y, 64), config)
        self.assertNotIn("_mst_tile", msl)
        self._close(out, x.cpu().double()[:, 64:192] @ w64)
        x, w, scales, biases = affine_args(4, 32, 32, 256, 512, DEVICE)
        out, msl = _run(
            dequant_forms_mm,
            (x, w, scales, biases, "half"),
            register_config([32, 128, 128], [8, 16, 1]),
        )
        self.assertIn("threadgroup half _mst_tile_half[16384];", msl)
        self._close(out, dequant_reference(x, w, scales, biases, torch.float16))

    def test_grid_all_or_none(self) -> None:
        """Two groups of one K loop, one with a staged vector, both on the
        staged op over [32, 128, 64] on 4 SIMD groups, against float64."""
        x = torch.randn(64, 512, device=DEVICE).to(torch.float16)
        y = torch.randn(512, 128, device=DEVICE)
        z = torch.randn(512, 128, device=DEVICE)
        v = torch.randn(4, 128, device=DEVICE)
        config = register_config([32, 128, 64], [8, 16, 1])
        outs, _, msl = run_with_msl(
            staged_apart_mm, (x, y, z, v), config, module_key=f"staged apart {config!r}"
        )
        out_a, out_b = cast("tuple[torch.Tensor, torch.Tensor]", outs)
        self.assertNotIn("_mst_tile", msl)
        x64 = x.cpu().double()
        wy = (y.cpu().double() * 0.5).half().double()
        wz = (z.cpu().double() * 0.25).half().double()
        self._close(out_a, (x64 @ wy) * v.cpu().double().sum(0))
        self._close(out_b, x64 @ wz)

    def test_grid_unread_x(self) -> None:
        """A widened x the lanes may not read at the op keeps the staged op,
        against float64."""
        args = affine_args(4, 32, 16, 128, 256, DEVICE)
        with unread_x():
            out, msl = _run(affine_mm, args, register_config([16, 64, 64], [4, 16, 1]))
        self.assertNotIn("_mst_tile", msl)
        self._close(out, affine_reference(args))

    def test_grid_inner_tiled(self) -> None:
        """M and N tiled in a loop of a one-program grid over a launch of 2
        SIMD groups (the staged op), against float64."""
        x = torch.randn(32, 128, device=DEVICE).to(torch.float16)
        y = torch.randn(128, 64, device=DEVICE)
        out, _ = _run(
            inner_tiled_weight_mm, (x, y), register_config([32, 64, 32], [4, 8, 2])
        )
        ref = x.cpu().double() @ (y.cpu().double() * 0.5).half().double()
        self._close(out, ref)

    def test_rolled_steps(self) -> None:
        """Runs past ``MAX_UNROLLED_STEPS`` steps a K iteration, looping over
        chunks of their steps, against float64: 4- and 6-bit weights over 16
        steps of one run, and two runs of 8 steps each beside each other."""
        for bits, group in ((4, 32), (6, 128)):
            with self.subTest(bits=bits):
                args = affine_args(bits, group, 32, 256, 1024, DEVICE)
                out, msl = _run(affine_mm, args, register_config([32, 32, 256]))
                self.assertIn("for (short _fc = 0; _fc < 2; _fc++)", msl)
                self._close(out, affine_reference(args))
        x = torch.randn(32, 512, device=DEVICE).to(torch.float16)
        y = torch.randn(512, 64, device=DEVICE)
        z = torch.randn(512, 64, device=DEVICE)
        x64, y64, z64 = (t.cpu().double() for t in (x, y, z))
        config = register_config([32, 32, 128], [1, 1, 32], num_warps=2)
        outs, _, msl = run_with_msl(two_acc_mm, (x, y, z), config, module_key="rolled")
        self.assertIn("for (short _fc = 0; _fc < 2; _fc++)", msl)
        a, b = cast("tuple[torch.Tensor, torch.Tensor]", outs)
        self._close(a, x64 @ (y64 * 0.5))
        self._close(b, x64 @ (z64 * 0.25))

    def test_partners(self) -> None:
        """An epilogue over two accumulators of one op type, each read at the
        first's slot (``mpp_matmul._classify_partner_uses``), against float64:
        in a register run, split over 2 SIMD groups, as a grid run over 4 and
        on the staged op."""
        x = torch.randn(32, 256, device=DEVICE).to(torch.float16)
        y = torch.randn(256, 64, device=DEVICE)
        z = torch.randn(256, 64, device=DEVICE)
        x64, y64, z64 = (t.cpu().double() for t in (x, y, z))
        g = (x64 @ (y64 * 0.5).half().double()).half().double()
        u = (x64 @ (z64 * 0.25).half().double()).half().double()
        ref = g * torch.sigmoid(g) * u
        for name, config in (
            ("register", register_config([32, 32, 32])),
            ("split", register_config([32, 32, 32], [1, 1, 32], num_warps=2)),
            ("grid", register_config([32, 64, 64], [8, 16, 1])),
            ("staged", helion.Config(block_sizes=[32, 32, 32], num_warps=4)),
        ):
            with self.subTest(name):
                out, msl = _run(gated_mm, (x, y, z), config)
                self.assertNotIn("_mpp_mat", msl)
                self.assertIn("b[_ei]", msl)
                self._close(out, ref, tol=4e-3)

    def test_partners_one_layout(self) -> None:
        """Partners whose own runs differ take one, against float64: a staged
        vector on the first under a launch of 4 SIMD groups (both staged)
        and on one SIMD group (both register runs); the affine split forced
        on the first's float op (taken by neither).  Register runs of two
        types aren't partners."""
        x = torch.randn(32, 256, device=DEVICE).to(torch.float16)
        y = torch.randn(256, 64, device=DEVICE)
        z = torch.randn(256, 64, device=DEVICE)
        v = torch.randn(4, 64, device=DEVICE)
        x64, y64, z64, v64 = (t.cpu().double() for t in (x, y, z, v))
        u = (x64 @ (z64 * 0.25).half().double()).half().double()
        g = ((x64 @ (y64 * 0.5).half().double()) * v64.sum(0)).half().double()
        ref = g * torch.sigmoid(g) * u
        for name, config in (
            ("grid", register_config([32, 64, 64], [8, 16, 1])),
            ("register", register_config([32, 32, 32])),
        ):
            with self.subTest(name):
                out, msl = _run(scaled_gated_mm, (x, y, z, v), config)
                self.assertIn("b[_ei]", msl)
                self._close(out, ref, tol=4e-3)
        args = affine_args(4, 32, 16, 64, 512, DEVICE)
        z = torch.randn(512, 64, device=DEVICE)
        g = affine_reference(args).half().double()
        u = (args[0].cpu().double() @ (z.cpu().double() * 0.25)).half().double()
        ref = g * torch.sigmoid(g) * u
        with mock.patch.object(
            mpp_registers, "affine_split_enabled", return_value=True
        ):
            out, msl = _run(affine_gated_mm, (*args, z), register_config([16, 32, 32]))
        self.assertNotIn("_mr_ones", msl)
        self._close(out, ref, tol=4e-3)
        # Register runs of two types aren't partners: both results are
        # materialized, the K loop split over two SIMD groups.
        x = torch.randn(32, 256, device=DEVICE).to(torch.float16)
        y = torch.randn(256, 32, device=DEVICE)
        z = torch.randn(256, 32, device=DEVICE)
        x64, y64, z64 = (t.cpu().double() for t in (x, y, z))
        g = (x64 @ (y64 * 0.5).half().double()).half().double()
        u = (x64 @ (z64 * 0.25)).half().double()
        config = register_config([32, 32, 32], [1, 1, 32], num_warps=2)
        out, msl = _run(narrowed_gated_mm, (x, y, z), config)
        self.assertNotIn("[_ei]", msl)
        self._close(out, g * torch.sigmoid(g) * u, tol=4e-3)
        # Ops of one key that no op reads together: the first's affine split
        # beside the second's float op.
        quant = affine_args(4, 32, 32, 64, 256, DEVICE)
        z = torch.randn(256, 64, device=DEVICE)
        v = torch.randn(4, 64, device=DEVICE)
        config = register_config([32, 32, 32])
        with mock.patch.object(
            mpp_registers, "affine_split_enabled", return_value=True
        ):
            outs, _, msl = run_with_msl(
                apart_mm, (*quant, z, v), config, module_key=f"apart {config!r}"
            )
        out_a, out_b = cast("tuple[torch.Tensor, torch.Tensor]", outs)
        self.assertIn("_mr_ones", msl)
        self.assertNotIn("[_ei]", msl)
        s64 = v.cpu().double().sum(0)
        self._close(out_a, (affine_reference(quant) * s64).half().double())
        u = quant[0].cpu().double() @ (z.cpu().double() * 0.25)
        self._close(out_b, u.half().double())

    def test_partners_seed_and_casts(self) -> None:
        """Against float64: a b + b0 and a b + (b0 + 1) beside b's seed of 1
        (a and b materialized), and silu(g) u in float, g read at u's slot
        inside silu's casts, on one SIMD group and over 4."""
        x = torch.randn(32, 64, device=DEVICE).to(torch.float16)
        y = torch.randn(64, 32, device=DEVICE)
        z = torch.randn(64, 32, device=DEVICE)
        x64 = x.cpu().double()
        g = x64 @ (y.cpu().double() * 0.5).half().double()
        u = x64 @ (z.cpu().double() * 0.25).half().double()
        for config in (
            register_config([32, 32, 32]),
            helion.Config(block_sizes=[32, 32, 32], num_warps=4),
        ):
            with self.subTest(config=config):
                outs, _, _ = run_with_msl(
                    seed_join_mm, (x, y, z), config, module_key=f"seed {config!r}"
                )
                out1, out2 = cast("tuple[torch.Tensor, torch.Tensor]", outs)
                self._close(out1, g * (u + 1) + 1)
                self._close(out2, g * (u + 1) + 2)
                out, msl = _run(silu_gated_mm, (x, y, z), config)
                self.assertIn("decltype(+g[_ei])", msl)
                self._close(out, g * torch.sigmoid(g) * u)

    def test_partners_kept_apart(self) -> None:
        """Ops of one type kept apart, against float64: a value read from
        either accumulator's carry before its op writes it, and an epilogue
        reading a tensor the kernel then zeroes."""
        x = torch.randn(32, 64, device=DEVICE).to(torch.float16)
        y = torch.randn(64, 32, device=DEVICE)
        z = torch.randn(64, 32, device=DEVICE)
        x64 = x.cpu().double()
        wy = (y.cpu().double() * 0.5).half().double()
        wz = (z.cpu().double() * 0.25).half().double()
        a, b = x64 @ wy, x64 @ wz
        a0, b0 = x64[:, :32] @ wy[:32], x64[:, :32] @ wz[:32]
        config = helion.Config(block_sizes=[32, 32, 32], num_warps=4)
        for kernel, ref in ((early_a_mm, 2 * a0 + b), (early_b_mm, a + 2 * b0)):
            with self.subTest(kernel.name):
                out, _ = _run(kernel, (x, y, z), config)
                self._close(out, ref)
        t = torch.randn(32, 32, device=DEVICE)
        t64 = t.cpu().double()
        out, _ = _run(rewriting_mm, (x, y, z, t), config)
        self._close(out, a + b * t64)
        self.assertEqual(int(t.count_nonzero()), 0)

    def test_one_decision(self) -> None:
        """A register run where the launch's thread axes codegen sees differ
        inside the K loop (M and N tiled in a loop of a one-program grid, K's
        block over 16 threads) computes (2 x) @ y."""
        x = torch.randn(16, 128, device=DEVICE)
        y = torch.randn(128, 32, device=DEVICE)
        config = register_config([16, 32, 64], [1, 2, 16])
        out, msl = _run(inner_tiled_mm, (x, y), config)
        self.assertRegex(msl, r"_mpp_op_0\.run\(_mr_a\d+, ")
        self._close(out, (2 * x).cpu().double() @ y.cpu().double())

    def test_forms(self) -> None:
        """The register run's other forms: a computed rhs beside an
        accumulator seeded from a tile, a bfloat view beside an nvfp4
        weight, and two operands converted to half over a K loop from 16
        with tails along M, N and K (positions past the ends read at the
        tile's first or as 0)."""
        x = torch.randn(24, 64, device=DEVICE)
        y = torch.randn(64, 40, device=DEVICE)
        c = torch.randn(24, 40, device=DEVICE)
        xh = x.half()
        b = (y.cpu().double() * 2.0).half().double()
        out, msl = _run(seeded_rhs, (xh, y, c), register_config([16, 32, 32]))
        self._close(out, c.cpu().double() + xh.cpu().double() @ b)
        self.assertIn("_mr_b", msl)
        xt = torch.randn(13, 40, device=DEVICE)
        yt = torch.randn(40, 20, device=DEVICE)
        ref = xt.half().cpu().double()[:, 16:] @ yt.half().cpu().double()[16:]
        out, msl = _run(k_from_16, (xt, yt), register_config([16, 32, 16]))
        self._close(out, ref)
        self.assertIn("_mr_a", msl)
        gen = torch.Generator().manual_seed(0)
        xb = torch.randn(8, 256, generator=gen).to(torch.bfloat16)
        wq = torch.randint(
            -(2**31), 2**31 - 1, (48, 32), generator=gen, dtype=torch.int32
        )
        s = torch.randint(0x30, 0x40, (48, 16), generator=gen, dtype=torch.uint8)
        args = tuple(t.to(DEVICE) for t in (xb, wq, s))
        out, msl = _run(nvfp4_rows, args, register_config([16, 16, 64]))
        staged, _ = _run(nvfp4_rows, args, register_config([16, 16, 64], [4, 8, 2]))
        self.assertIn("_mr_b", msl)
        self._close(out.float(), staged.float().cpu().double(), tol=1e-2)


if __name__ == "__main__":
    unittest.main()
