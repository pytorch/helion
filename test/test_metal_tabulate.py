"""Values computed from one small integer read from a table
(``metal/tabulate.py``): pure Python, no Metal device."""

from __future__ import annotations

import ast
import math
import textwrap
import unittest

import pytest
import torch

from helion._compiler.metal.tabulate import tabulate
from helion._testing import skipUnlessBackends

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


def _tabulate(
    src: str, constants: dict[str, int | float] | None = None, max_entries: int = 16
) -> tuple[str, dict[tuple[str, tuple[str, ...]], str]]:
    tables: dict[tuple[str, tuple[str, ...]], str] = {}
    stmts = ast.parse(textwrap.dedent(src)).body
    out = tabulate(stmts, constants or {}, tables, max_entries)
    return "\n".join(ast.unparse(s) for s in out), tables


def _entries(tables: dict[tuple[str, tuple[str, ...]], str], name: str) -> list[float]:
    ((ctype, entries),) = [key for key, n in tables.items() if n == name]
    return [float(e.rstrip("fh")) for e in entries]


#: An nvfp4 code's decode (E2M1 through half bits), as an element runs it.
NIBBLE = """\
    q = (w >> s) & m
    hi = (q & 7) << 9
    sign = (q & 8) << 12
    code = as_type(half, static_cast(short, hi | sign))
    value = static_cast(float, code)
    out = value * scale"""


class TestTabulate(unittest.TestCase):
    def test_nibble(self) -> None:
        """The chain from a 4-bit field to a float is one lookup; the
        statements only it read are gone, and the masked field stays."""
        out, tables = _tabulate(NIBBLE, {"m": 15})
        self.assertEqual(
            out,
            "q = w >> s & m\nvalue = _helion_table_0[q]\nout = value * scale",
        )
        q = torch.arange(16, dtype=torch.int32)
        ref = ((q & 7) << 9 | (q & 8) << 12).to(torch.int16).view(torch.float16)
        self.assertEqual(_entries(tables, "_helion_table_0"), ref.float().tolist())
        self.assertEqual(next(iter(tables))[0], "float")

    def test_one_op(self) -> None:
        """A chain of one op is tabulated too: whether a lookup pays is the
        search's call (``metal_tabulate_max_entries``), not a length rule."""
        out, tables = _tabulate("q = x & 15\nv = static_cast(float, q)\nr = v * w")
        self.assertEqual(out, "q = x & 15\nv = _helion_table_0[q]\nr = v * w")
        self.assertEqual(
            _entries(tables, "_helion_table_0"), [float(i) for i in range(16)]
        )

    def test_float_ops(self) -> None:
        """A chain through float arithmetic with constants, in float32, and
        in half where both operands are half."""
        out, tables = _tabulate(
            """\
            b = word >> 8 & 7
            h = as_type(half, static_cast(short, b << 9))
            v = static_cast(float, h * h) * k / 3.0
            r = v * scale""",
            {"k": 256.0},
        )
        self.assertEqual(out.splitlines()[1], "v = _helion_table_0[b]")
        b = torch.arange(8, dtype=torch.int32)
        h = (b << 9).to(torch.int16).view(torch.float16)
        ref = (h * h).float() * 256.0 / 3.0
        self.assertEqual(_entries(tables, "_helion_table_0"), ref.tolist())

    def test_kept(self) -> None:
        """Chains the evaluation doesn't model are kept as written."""
        for why, src in (
            ("a shift whose type isn't fixed", "q = x & 15\ny = q << 7\nz = y | w"),
            ("a byte field past the cap of 16", NIBBLE.replace("& m", "& 255")),
            ("a mask that isn't a low-bit field", NIBBLE.replace("& m", "& 8")),
            ("an int past 2**31", "q = x & 15\ny = q << 30\nz = y << 4\nr = z * w"),
            (
                "a bfloat result",
                (
                    "q = x & 15\n"
                    "v = static_cast(bfloat, static_cast(float, q) * 2.0)\n"
                    "r = v * w"
                ),
            ),
            (
                "a bfloat result of a long chain",
                (
                    "q = w & 15\n"
                    "v = as_type(bfloat, static_cast(short, q << 3 | 16256))\n"
                    "out = v * scale"
                ),
            ),
            (
                "a constant reassigned after the chain",
                (
                    "q = w & 15\nc = w + 1\na = (q << 2) + c\nb = a << 1 | 3\n"
                    "d = b << 1 | 1\ne = d * z\nc = 3"
                ),
            ),
            (
                "an integer whose promoted type isn't fixed",
                "q = w & 15\nv = (static_cast(uint, q) + 1 << 1) + 1\nr = v - z",
            ),
            (
                "an fma of an inf or a NaN",
                (
                    "q = w & 15\n"
                    "h = as_type(half, static_cast(ushort, q << 10 | 31744))\n"
                    "u = metal.fma(h, 2.0, 1.0)\n"
                    "r = u * z"
                ),
            ),
            (
                "a NaN converted to bfloat",
                (
                    "q = w & 15\n"
                    "f = as_type(float, static_cast(uint, q | 2147483392))\n"
                    "v = static_cast(float, static_cast(bfloat, f))\n"
                    "r = v * z"
                ),
            ),
            (
                "a NaN bitcast back to an integer",
                (
                    "q = w & 15\n"
                    "f = as_type(float, static_cast(uint, q << 16 | 2139095040))\n"
                    "b = as_type(uint, f) >> 16\n"
                    "r = static_cast(float, b) * z"
                ),
            ),
            (
                "a NaN of arithmetic bit cast to an integer",
                (
                    "q = w & 15\n"
                    "a = static_cast(float, q) * 1e38 * 1e38\n"
                    "b = a - a\n"
                    "c = as_type(uint, b)\n"
                    "e = c + w"
                ),
            ),
            (
                "an unknown call",
                "q = x & 15\nv = metal.exp(static_cast(float, q) * 2.0)\nr = v * w",
            ),
        ):
            with self.subTest(why):
                out, tables = _tabulate(src, {"m": 15})
                self.assertEqual(tables, {})
                kept = ast.parse(textwrap.dedent(src)).body
                self.assertEqual(out, "\n".join(ast.unparse(s) for s in kept))

    def test_fma_and_unmodeled_neighbors(self) -> None:
        """An fma rounds once; a statement the evaluation doesn't model,
        beside the chain, doesn't keep it."""
        out, tables = _tabulate(
            NIBBLE.replace("out = value * scale", "u = metal.fma(value, 0.5, 1.0)")
            + "\n    flag = q < 8\n    r = u * scale + flag",
            {"m": 15},
        )
        self.assertIn("u = _helion_table_0[q]", out)
        self.assertIn("flag = q < 8", out)
        q = torch.arange(16, dtype=torch.int32)
        ref = ((q & 7) << 9 | (q & 8) << 12).to(torch.int16).view(torch.float16)
        exact = ref.double() * 0.5 + 1.0
        self.assertEqual(_entries(tables, "_helion_table_0"), exact.float().tolist())

    def test_overflow(self) -> None:
        """A value past its type's range (inf) is kept computed; the finite
        values before it are tabulated."""
        out, tables = _tabulate(
            "q = w & 15\n"
            "h = as_type(half, static_cast(short, q << 6 | 30720))\n"
            "u = metal.fma(h, h, h)\n"
            "r = u * z"
        )
        self.assertEqual(
            out.splitlines()[1:3], ["h = _helion_table_0[q]", "u = metal.fma(h, h, h)"]
        )
        self.assertEqual(_entries(tables, "_helion_table_0")[:2], [32768.0, 34816.0])

    def test_signed_zero(self) -> None:
        """An fma of zeros keeps IEEE's sign: -0 only from two -0s."""
        out, tables = _tabulate(
            "q = w & 15\n"
            "h = as_type(half, static_cast(ushort, (q & 8) << 12))\n"
            "u = metal.fma(h, 1.0, h)\n"
            "r = u * z"
        )
        self.assertIn("u = _helion_table_0[q]", out)
        signs = [math.copysign(1.0, e) for e in _entries(tables, "_helion_table_0")]
        self.assertEqual(signs, [1.0] * 8 + [-1.0] * 8)

    def test_signed_zero_constants(self) -> None:
        """Constants 0.0 and -0.0 build different tables (a sign bit cast to
        an integer), though ``==`` takes one for the other."""
        src = (
            "q = w & 15\n"
            "f = static_cast(float, q) * c\n"
            "g = f * 2.0 + c\n"
            "b = as_type(uint, g)\n"
            "r = b ^ w"
        )
        for zero, bits in ((0.0, 0), (math.copysign(0.0, -1.0), 2**31)):
            with self.subTest(zero=zero):
                _, tables = _tabulate(src, {"c": zero})
                ((_, entries),) = tables
                self.assertEqual(entries, (str(bits),) * 16)

    def test_formats(self) -> None:
        """Not one format's decode: an E4M3 byte through half bits and a
        scale is one lookup in a 256-entry table; an int4 code with an affine
        scale keeps its decode, as only the conversion depends on the field
        alone (``(q - zero) * scale`` reads per-group values)."""
        out, tables = _tabulate(
            """\
            q = (w >> s) & m
            h = ((q & 127) << 7) | ((q & 128) << 8)
            v = as_type(half, static_cast(ushort, h))
            f = static_cast(float, v) * 256.0
            out = f * scale""",
            {"m": 255},
            256,
        )
        self.assertIn("f = _helion_table_0[q]", out)
        q = torch.arange(256, dtype=torch.int32)
        bits = ((q & 127) << 7) | ((q & 128) << 8)
        ref = bits.to(torch.int16).view(torch.float16).float() * 256.0
        e4m3 = q.to(torch.uint8).view(torch.float8_e4m3fn).float()
        finite = ~e4m3.isnan()
        self.assertEqual(ref[finite].tolist(), e4m3[finite].tolist())
        self.assertEqual(_entries(tables, "_helion_table_0"), ref.tolist())
        src = """\
            q = (w >> s) & m
            v = (static_cast(float, q) - zero) * scale
            out = v * x"""
        out, tables = _tabulate(src, {"m": 15}, 256)
        self.assertEqual(tables, {})

    def test_caps(self) -> None:
        """The cap admits fields of at most that many values; 0 none."""
        byte = NIBBLE.replace("& m", "& 255")
        for src, cap, tabulated in (
            (byte, 256, True),
            (NIBBLE, 256, True),
            (NIBBLE, 0, False),
        ):
            with self.subTest(cap=cap, field="byte" if src is byte else "nibble"):
                _, tables = _tabulate(src, {"m": 15}, cap)
                self.assertEqual(bool(tables), tabulated)

    def test_reassigned_field(self) -> None:
        """A field changed after its mask, in place or by an element binding,
        is no index of a table: ``q`` may reach 16 there."""
        for how, change in (
            ("in place", "q += 1"),
            ("rebound", "_helion_mpp_bind('q', w)"),
        ):
            with self.subTest(how):
                src = NIBBLE.replace("& m\n", f"& m\n    {change}\n")
                out, tables = _tabulate(src, {"m": 15})
                self.assertEqual(tables, {})

    def test_shared_tables(self) -> None:
        """Equal tables share a name across calls."""
        tables: dict[tuple[str, tuple[str, ...]], str] = {}
        for _ in range(2):
            tabulate(ast.parse(textwrap.dedent(NIBBLE)).body, {"m": 15}, tables, 16)
        self.assertEqual(list(tables.values()), ["_helion_table_0"])


if __name__ == "__main__":
    unittest.main()
