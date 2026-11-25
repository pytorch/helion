"""Helion's copies of c10 helpers in generated MSL
(``helion._compiler.metal.c10_helpers``), from CPU renders: a kernel that
names only copied helpers carries those and no c10 include; one that names
another c10 helper, or reduces 64-bit integers, includes the headers as
before; the table's dependencies resolve."""

from __future__ import annotations

import re
import unittest

import pytest
import torch

from helion._compiler.metal.c10_helpers import C10_INCLUDES
from helion._compiler.metal.c10_helpers import HELPERS
from helion._compiler.metal.c10_helpers import c10_names
from helion._compiler.metal.c10_helpers import helpers_source
from helion._compiler.metal.msl_reduction import REDUCTION_C10
from helion._compiler.metal.msl_reduction import REDUCTION_INCLUDES
from helion._compiler.metal.msl_reduction import REDUCTION_PREAMBLE
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_preamble import bfloat_mul_add
from helion._testing.metal_preamble import erf_of
from helion._testing.metal_preamble import row_sums

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


def _render(kernel: object, args: tuple[object, ...]) -> str:
    return render_msl(kernel, args, kernel.bind(args).config_spec.default_config())


def _carried(msl: str) -> set[str]:
    """The copied helpers *msl* carries."""
    return {name for name, (source, _) in HELPERS.items() if source in msl}


def _closure(names: set[str]) -> set[str]:
    return {
        name
        for name, (source, _) in HELPERS.items()
        if source in "".join(helpers_source(names))
    }


class TestC10Helpers(unittest.TestCase):
    def test_copied(self) -> None:
        """An affine matmul (index math: floor_divide, remainder) and a row
        sum (the reduction preamble: SIMD reductions, max, min,
        simdgroup_size) carry just the helpers they name, with what those use,
        and no c10 include; a kernel that names none carries nothing."""
        matmul = _render(affine_mm, affine_args(4, 32, 8, 256, 1024))
        reduction = _render(row_sums, (torch.randn(64, 256),))
        for msl, named in ((matmul, {"floor_divide"}), (reduction, REDUCTION_C10)):
            with self.subTest(named=sorted(named)):
                self.assertNotIn("#include <c10/", msl)
                using = "} // namespace c10\nusing namespace c10::metal;\n"
                end = msl.index(using) + len(using)
                self.assertLess(end, msl.index("kernel void"))
                rest = msl[end:].replace(REDUCTION_PREAMBLE, "")
                names = c10_names(rest)
                assert names is not None
                if REDUCTION_PREAMBLE in msl:
                    names |= REDUCTION_C10
                self.assertLessEqual(named, names)
                self.assertEqual(_carried(msl), _closure(names))
        # bfloat math names no c10 helper, and carries none.
        x = torch.randn(1024, dtype=torch.bfloat16)
        msl = _render(bfloat_mul_add, (x, x))
        self.assertNotIn("c10", msl)

    def test_headers(self) -> None:
        """erf, which Helion has no copy of, and a sum of int64 values (c10's
        64-bit SIMD reductions) take the c10 includes as before, and no
        copies."""
        erf = _render(erf_of, (torch.randn(1024),))
        self.assertTrue(
            erf.startswith("\n".join(["#include <metal_stdlib>", *C10_INCLUDES]))
        )
        long_sum = _render(row_sums, (torch.zeros(64, 256, dtype=torch.int64),))
        for msl in (erf, long_sum):
            self.assertNotIn("namespace c10 {", msl)
        self.assertIn(REDUCTION_INCLUDES[0], long_sum)
        self.assertIn(C10_INCLUDES[0], long_sum)

    def test_names(self) -> None:
        """The names read past comments, with the unqualified calls the
        headers' ``using namespace c10::metal`` resolves into c10 (a tile's
        end, ``min(offset, end)``, but not ``metal::min``); another c10
        namespace or a macro of the headers sends a kernel to them."""
        self.assertEqual(
            c10_names("x = c10::metal::floor_divide(a, b); // c10::metal::erf"),
            {"floor_divide"},
        )
        self.assertEqual(
            c10_names("auto end = min(offset + 16, 64); y = metal::max(a, b);"),
            {"min"},
        )
        self.assertEqual(c10_names("y = fmod(a, b);"), {"fmod"})
        self.assertEqual(
            c10_names("/* c10::metal::erf */ y = ::c10::metal::max(a, b);"), {"max"}
        )
        for code in (
            "x = c10::metal::detail::simd_type_t<float>(a);",
            "x = c10::util::f(a);",
            "#if C10_METAL_HAS_MPP\n#endif",
        ):
            with self.subTest(code=code):
                self.assertIsNone(c10_names(code))

    def test_table(self) -> None:
        """Every helper's helpers come before it in the table, and the
        reduction preamble names only ``REDUCTION_C10``, all in the table:
        its macros paste ``simd_##NAME`` for the reductions they're invoked
        with."""
        order = list(HELPERS)
        for name, (_, uses) in HELPERS.items():
            for used in uses:
                self.assertLess(order.index(used), order.index(name), (name, used))
        self.assertLessEqual(REDUCTION_C10, HELPERS.keys())
        preamble = re.sub(r"//[^\n]*", "", REDUCTION_PREAMBLE)
        pasted = re.findall(r"c10::metal::(\w+)##", preamble)
        self.assertEqual(set(pasted), {"simd_"})
        reductions = set(
            re.findall(r"^HELION_\w+_REDUCE\((\w+)", preamble, re.MULTILINE)
        )
        self.assertEqual(reductions, {"sum", "prod", "max", "min"})
        names = {
            name
            for name in re.findall(r"c10::metal::(\w+)(?!\w*##)", preamble)
            if name != "simd_"
        } | {f"simd_{r}" for r in reductions}
        self.assertEqual(names, REDUCTION_C10)


if __name__ == "__main__":
    unittest.main()
