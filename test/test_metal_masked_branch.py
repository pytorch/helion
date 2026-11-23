"""One branch around the statements of a loop step that a lane predicate
masks (``metal/masked_branch.py``): its rules on hand-written statements,
and a masked tail in rendered MSL."""

from __future__ import annotations

import ast
import unittest

import pytest
import torch

import helion
from helion._compiler.metal.masked_branch import branch_masked_steps
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_split import reduced_in_loop
from helion._testing.metal_split import split_gemv
from helion._testing.metal_split import split_gemv_args
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

#: A K loop's step as the walker gets it: a load under the step's mask, its
#: value selected into a carried sum.
_STEP = """
acc = float(0.0)
for offset_0 in tl.range(0, 416, 256):
    indices_0 = offset_0 + tid[0]
    mask_0 = indices_0 < 416
    load = tl.load(x + indices_0, mask_0, other=0)
    v_0 = static_cast(float, load)
    _mask_select = v_0 if mask_0 else static_cast(float, 0)
    acc = acc + _mask_select
"""

_BRANCHED = """
acc = float(0.0)
for offset_0 in tl.range(0, 416, 256):
    indices_0 = offset_0 + tid[0]
    mask_0 = indices_0 < 416
    if mask_0:
        load = tl.load(x + indices_0)
        v_0 = static_cast(float, load)
        _mask_select = v_0
        acc = acc + _mask_select
    else:
        acc = acc + static_cast(decltype(acc), 0)
"""


def _branch(src: str) -> str:
    return "\n".join(ast.unparse(s) for s in branch_masked_steps(ast.parse(src).body))


def _same(src: str) -> str:
    return "\n".join(ast.unparse(s) for s in ast.parse(src).body)


class TestRules(unittest.TestCase):
    def test_step(self) -> None:
        """The statements after the mask go under one ``if`` on it, the load
        unmasked and the select its value; a skipped lane adds +0 to the sum,
        as the select did."""
        self.assertEqual(_branch(_STEP), _same(_BRANCHED))

    def test_fill(self) -> None:
        """A masked load's fill of 0 added to the sum straight: the branch
        drops the mask; a fill of 1, which a skipped lane would not add,
        keeps the masked form."""
        select = (
            "    _mask_select = v_0 if mask_0 else static_cast(float, 0)\n"
            "    acc = acc + _mask_select\n"
        )
        direct = _STEP.replace(select, "    acc = acc + load\n")
        self.assertIn("load = tl.load(x + indices_0)", _branch(direct))
        ones = direct.replace("other=0", "other=1")
        self.assertEqual(_branch(ones), _same(ones))

    def test_products(self) -> None:
        """A product the mask zeroes: the square of a masked load, the load
        times or over a positive literal, and the two loads' fused
        multiply-add onto the sum branch; the load times a negative literal
        (-0.0) or an unknown value (inf or NaN times 0 is NaN) keeps the
        masked form."""
        select = (
            "    v_0 = static_cast(float, load)\n"
            "    _mask_select = v_0 if mask_0 else static_cast(float, 0)\n"
            "    acc = acc + _mask_select\n"
        )
        square = "    v_0 = load * load\n    acc = acc + static_cast(float, v_0)\n"
        updates = {
            "load * load": True,
            "load * 2.0": True,
            "load / 4.0": True,
            "load * -2.0": False,
            "load * x0": False,
        }
        for product, branched in updates.items():
            with self.subTest(product):
                src = _STEP.replace(select, square.replace("load * load", product))
                self.assertEqual("if mask_0:" in _branch(src), branched)
        for fma, branched in (("load, load", True), ("load, x0", False)):
            with self.subTest(fma=fma):
                src = _STEP.replace(select, f"    acc = metal.fma({fma}, acc)\n")
                self.assertEqual("if mask_0:" in _branch(src), branched)

    def test_negative_zero(self) -> None:
        """A sum that may hold -0.0, which a masked lane's +0 turned into
        +0.0: one that starts at -0.0, one whose start the block doesn't show,
        one the step scales before the mask, and one a subtraction follows the
        addition into, branch with the +0 on the skipped lanes."""
        changes = {
            "starts at -0.0": ("acc = float(0.0)", "acc = float(-0.0)"),
            "no start in the block": ("acc = float(0.0)\n", ""),
            "scaled before the step": (
                "    mask_0 = indices_0 < 416\n",
                "    acc = acc * 2.0\n    mask_0 = indices_0 < 416\n",
            ),
            # (acc + 0) - 0 is +0.0 for a -0.0 acc too.
            "added, then subtracted": (
                "    acc = acc + _mask_select\n",
                "    t = acc + _mask_select\n    acc = t - _mask_select\n",
            ),
        }
        for name, (old, new) in changes.items():
            with self.subTest(name):
                out = _branch(_STEP.replace(old, new))
                self.assertIn(
                    "    else:\n        acc = acc + static_cast(decltype(acc), 0)", out
                )

    def test_kept_carry(self) -> None:
        """A select that keeps the carry when the mask is false keeps it
        exactly: the branch, whatever the start, needs no +0."""
        select = (
            "    _mask_select = v_0 if mask_0 else static_cast(float, 0)\n"
            "    acc = acc + _mask_select\n"
        )
        src = _STEP.replace(select, "    acc = v_0 if mask_0 else acc\n")
        src = src.replace("acc = float(0.0)", "acc = float(-0.0)")
        out = _branch(src)
        self.assertIn("    if mask_0:\n        load = tl.load(x + indices_0)", out)
        self.assertNotIn("else:", out)

    def test_bitwise_and(self) -> None:
        """A mask joined to a row mask by ``&`` between booleans: the branch
        drops the step's mask and the load keeps the row's."""
        src = "mask_1 = r < 8\n" + _STEP.replace(
            "tl.load(x + indices_0, mask_0, other=0)",
            "tl.load(x + indices_0, mask_1 & mask_0, other=0)",
        ).replace("v_0 if mask_0 else", "v_0 if mask_1 & mask_0 else")
        out = _branch(src)
        self.assertIn(
            "    if mask_0:\n        load = tl.load(x + indices_0, mask_1, other=0)",
            out,
        )
        self.assertIn("_mask_select = v_0 if mask_1 else", out)

    def test_store(self) -> None:
        """A store under the mask moves into the branch without it; one with
        the mask and another predicate keeps the other."""
        for mask, kept in (("mask_0", "None"), ("mask_0 and mask_1", "mask_1")):
            with self.subTest(mask=mask):
                src = _STEP + f"    tl.store(y + indices_0, v_0, {mask})\n"
                out = _branch(src)
                self.assertIn("if mask_0:", out)
                self.assertIn(f"tl.store(y + indices_0, v_0, {kept})", out)

    def test_limits(self) -> None:
        """Each keeps the masked form, which the analysis can't prove the
        branch equal to: a store without the mask, which skipped lanes would
        drop; an overwrite of the carried sum, and a max into it; a SIMD-group
        call, which a divergent branch breaks; the mask reassigned, or read
        other than as a predicate.  A step that accumulates nothing (a
        pointwise lane loop's) keeps it too."""
        changes = {
            "unmasked store": (
                "acc = acc + _mask_select",
                "acc = acc + _mask_select\n    tl.store(y + indices_0, v_0, None)",
            ),
            "overwrite": ("acc = acc + _mask_select", "acc = _mask_select"),
            "max": ("acc = acc + _mask_select", "acc = metal.max(acc, _mask_select)"),
            "simd": (
                "v_0 = static_cast(float, load)",
                "v_0 = metal.simd_sum(static_cast(float, load))",
            ),
            "reassigned": (
                "acc = acc + _mask_select",
                "acc = acc + _mask_select\n    mask_0 = indices_0 < 100",
            ),
            "bare read": (
                "acc = acc + _mask_select",
                "acc = acc + _mask_select\n    flag = mask_0",
            ),
            "no accumulation": (
                "acc = acc + _mask_select",
                "tl.store(y + indices_0, _mask_select, mask_0)",
            ),
        }
        for name, (old, new) in changes.items():
            with self.subTest(name):
                src = _STEP.replace(old, new)
                self.assertEqual(_branch(src), _same(src))


@helion.kernel(backend="metal", static_shapes=True)
def carry_exp_max(x: torch.Tensor) -> torch.Tensor:
    """A per-lane running max of each tile's exp, maxed after the loop."""
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, bn], float("-inf"), dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = torch.maximum(acc, torch.exp(x[tile_m, tile_n]))
        out[tile_m] = acc.amax(-1)
    return out


def _loop(msl: str) -> str:
    """*msl* from its K loop to the output's store."""
    start = msl.index("for (int offset_")
    return msl[start : msl.index("out[", start)]


class TestRendered(unittest.TestCase):
    _CONFIG = helion.Config(
        block_sizes=[256, 1, 1], num_threads=[256, 1, 1], num_warps=8
    )

    def test_tail(self) -> None:
        """A GEMV over K = 2080 = 8 * 256 + 32 on 256 threads: the step runs
        under its mask, its loads plain and its sum without selects."""
        msl = render_msl(split_gemv, split_gemv_args(1, 64, 2080), self._CONFIG)
        loop = _loop(msl)
        self.assertIn("if (mask_", loop)
        self.assertNotIn("_helion_masked", loop)
        self.assertNotIn(": static_cast<float>(0)", loop)

    def test_max_keeps_its_select(self) -> None:
        """A max carry, which the analysis leaves masked: the 12 padded
        lanes of 500 columns' last 16-column tile take -inf into the max
        through the operand's select."""
        msl = render_msl(
            carry_exp_max, (torch.randn(64, 500),), helion.Config(block_sizes=[16, 4])
        )
        loop = msl[msl.index("for (int offset_") : msl.index("helion_red::")]
        self.assertRegex(
            loop,
            r"_mask_select\w* = \(mask_\d+ \? \w+ : "
            r"static_cast<float>\(\(-INFINITY\)\)\);",
        )
        self.assertNotIn("if (mask_", loop)

    def test_collective_in_step(self) -> None:
        """A step that sums across its threads keeps its masked loads."""
        args = (torch.randn(1, 2080), torch.randn(64, 2080))
        loop = _loop(render_msl(reduced_in_loop, args, self._CONFIG))
        self.assertIn("helion_red::group_sum", loop)
        self.assertIn("_helion_masked", loop)
        self.assertNotIn("if (mask_", loop)


if __name__ == "__main__":
    unittest.main()
