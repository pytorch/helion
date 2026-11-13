from __future__ import annotations

import ast
import textwrap
from types import SimpleNamespace
from typing import ClassVar
import unittest

import pytest

from helion._compiler.metal.fma_fusion import fuse_multiply_adds
from helion._testing import skipUnlessBackends

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


class TestMetalFmaFusionRewrite(unittest.TestCase):
    """``fuse_multiply_adds`` on hand-written device bodies."""

    TYPES: ClassVar[dict[str, str]] = {
        "x": "float",
        "w": "float",
        "r": "float",
        "h": "half",
        "n": "int",
    }

    def _fuse(self, source: str) -> str:
        body = ast.parse(textwrap.dedent(source)).body
        return ast.unparse(
            ast.Module(body=fuse_multiply_adds(body, self.TYPES), type_ignores=[])
        )

    def test_float_product_fuses(self) -> None:
        """A float product used once by an add in its block, here a loop
        carry, becomes an fma."""
        got = self._fuse(
            """
            acc = float(0.0)
            for k in range(4):
                a = tl.load(x + k, mask, other=0)
                b = w[k]
                t = a * b
                acc = acc + t
            """
        )
        expected = """
            acc = float(0.0)
            for k in range(4):
                a = tl.load(x + k, mask, other=0)
                b = w[k]
                acc = metal.fma(a, b, acc)
            """
        self.assertEqual(got, ast.unparse(ast.parse(textwrap.dedent(expected))))

    def test_viewed_float_fuses(self) -> None:
        """A float reinterpreted from int bits (``as_type``) is a float."""
        got = self._fuse("a = as_type(float, n)\nb = w[0]\nt = a * b\nu = t + r[0]\n")
        self.assertEqual(
            got, "a = as_type(float, n)\nb = w[0]\nu = metal.fma(a, b, r[0])"
        )

    def test_interleaved_independent_products_both_fuse(self) -> None:
        got = self._fuse("t = x * w\ns = r * 2.0\nu = t + 1.0\nv = s + 3.0\n")
        self.assertEqual(got, "u = metal.fma(x, w, 1.0)\nv = metal.fma(r, 2.0, 3.0)")

    def test_product_assignment_invalidates_earlier_operands(self) -> None:
        source = "t = x * w\nx = r * 5.0\nu = t + 1.0\n"
        inputs = {"x": 2.0, "w": 3.0, "r": 4.0}
        expected = dict(inputs)
        exec(source, expected)
        self.assertEqual(expected["u"], 7.0)

        actual = {
            **inputs,
            # These exact integer-valued floats do not depend on FMA rounding.
            "metal": SimpleNamespace(fma=lambda a, b, c: a * b + c),
        }
        exec(self._fuse(source), actual)
        self.assertEqual(actual["u"], expected["u"])
        self.assertEqual(actual["x"], expected["x"])

    def test_kept(self) -> None:
        for name, source in (
            ("int index math", "t = n * 4\nu = t + n\n"),
            ("half operands", "a = h[0]\nb = h[1]\nc = r[0]\nt = a * b\nu = t + c\n"),
            (
                "addend rounded to half",
                "a = x[0]\nb = w[0]\nc = static_cast(half, r[0])\nt = a * b\nu = t + c\n",
            ),
            (
                "product updated in place",
                "a = x[0]\nb = w[0]\nc = r[0]\nt = a * b\nt += c\nu = t + c\n",
            ),
            (
                "operand reassigned before the add",
                "a = x[0]\nb = w[0]\nt = a * b\nb = r[0]\nu = t + b\n",
            ),
            ("product used twice", "a = x[0]\nt = a * a\nu = t + t\n"),
            (
                "name of two types",
                "a = x[0]\na = h[0]\nb = w[0]\nc = r[0]\nt = a * b\nu = t + c\n",
            ),
            (
                "add in another block",
                (
                    "a = x[0]\nb = w[0]\nc = r[0]\nt = a * b\n"
                    "for k in range(2):\n    u = t + c\n"
                ),
            ),
        ):
            with self.subTest(name):
                source = textwrap.dedent(source)
                self.assertEqual(self._fuse(source), ast.unparse(ast.parse(source)))
