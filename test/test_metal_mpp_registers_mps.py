"""MPP's register operands (``mpp_registers``) on the GPU: the probed
layouts, and numerics against float64 for 4-, 5- and 6-bit affine weights
at 4 to 32 rows, with x built in registers or read as a device view, and
the register run's other forms."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest import mock

import pytest
import torch

from helion._compiler.metal import coop_layout
from helion._compiler.metal import mpp_registers
from helion._testing import pairwise
from helion._testing import skipUnlessBackends
from helion._testing.metal import run_with_msl
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_fragment import affine_reference
from helion._testing.metal_fragment import k_from_16
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_fragment import seeded_rhs
from helion._testing.metal_registers import inner_tiled_mm
from helion._testing.metal_registers import register_config

if TYPE_CHECKING:
    import helion

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
#: word, 5 straddles words 3 times in 16, 6 twice in 8), x's rows (at 8 or
#: fewer x is a device view and the op takes 8 rows; at 16 and 32 both
#: operands are in registers on an M4), and the K tile (within one group,
#: or over several).
_AFFINE = pairwise(bits=(4, 5, 6), rows=(4, 8, 16, 32), one_group=(True, False))


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
        """4-, 5- and 6-bit affine weights at 4 to 32 rows against float64."""
        for bits, rows, one_group in _AFFINE:
            group = 32 if bits == 4 else 128
            k_block = group if one_group else 64
            if not one_group:
                group = 32
            with self.subTest(bits=bits, rows=rows, one_group=one_group):
                args = affine_args(bits, group, rows, 96, 1024, DEVICE)
                config = register_config([16 if rows <= 16 else 32, 32, k_block])
                out, msl = _run(affine_mm, args, config)
                self._close(out, affine_reference(args))
                self.assertIn("execution_simdgroup> _mpp_op_0;", msl)
                self._assert_view(msl, rows)

    def _assert_view(self, msl: str, rows: int) -> None:
        """x is read as a device view at 8 rows or fewer; past that on an M4,
        whose layout relabels K, it is built in registers (on a GPU whose
        layout keeps a lane's K positions in order it stays a view)."""
        view = "auto _mpp_a = tensor<device half" in msl
        if rows <= 8:
            self.assertTrue(view)
        elif "M4" in torch.backends.mps.get_name():  # pyrefly: ignore[missing-attribute]
            self.assertFalse(view)

    def test_runtime_coordinates(self) -> None:
        """Elements read at run time from the op compute the same."""
        args = affine_args(5, 128, 16, 96, 1024, DEVICE)
        config = register_config([16, 32, 64])
        with mock.patch.object(mpp_registers, "factor", lambda coords, d: None):
            out, msl = _run(affine_mm, args, config)
        self.assertIn("get_multidimensional_index((ushort)0)", msl)
        self._close(out, affine_reference(args))

    def test_one_decision(self) -> None:
        """A register run where the launch's thread axes codegen sees differ
        inside the K loop (M and N tiled in a loop of a one-program grid, K's
        block over 16 threads) computes (2 x) @ y."""
        x = torch.randn(16, 128, device=DEVICE)
        y = torch.randn(128, 32, device=DEVICE)
        config = register_config([16, 32, 64], [1, 2, 16])
        out, msl = _run(inner_tiled_mm, (x, y), config)
        self.assertIn("_mpp_op_0.run(_mr_a, ", msl)
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
