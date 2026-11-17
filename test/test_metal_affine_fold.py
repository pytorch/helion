"""Folding integer division and remainder out of index arithmetic over
known facts (``metal/affine_fold.py``): pure Python, no Metal device."""

from __future__ import annotations

import ast
import itertools
import textwrap
import unittest

import pytest

from helion._compiler.metal.affine_fold import Affine
from helion._compiler.metal.affine_fold import Fact
from helion._compiler.metal.affine_fold import floor_divide
from helion._compiler.metal.affine_fold import fold
from helion._compiler.metal.affine_fold import remainder
from helion._compiler.metal.affine_fold import render
from helion._compiler.metal.fragment_mma import kernel_constants
from helion._testing import skipUnlessBackends

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


FACTS = {
    "off": Fact(64, 0),  # a tile offset: a multiple of 64
    "lane": Fact(2, 0, 6),  # 0, 2, 4 or 6
    "row": Fact(1, 0, 7),
    "n": Fact(),  # nothing known
}


def _src(stmts: list[ast.stmt]) -> str:
    return "\n".join(ast.unparse(s) for s in stmts)


class TestAffineFold(unittest.TestCase):
    def test_floor_divide(self) -> None:
        """``(D + R) // d`` is ``D / d + q`` when D's terms are multiples of
        d and R stays within one multiple of d."""
        value = Affine.term("off") + Affine.term("lane") + Affine.of(8 * 3 + 1)
        self.assertEqual(
            ast.unparse(render(floor_divide(value, 8, dict(FACTS)))), "off / 8 + 3"
        )
        # R = lane + 25 spans 25..31: one multiple of 16 (16..31).
        self.assertEqual(
            ast.unparse(render(floor_divide(value, 16, dict(FACTS)))), "off / 16 + 1"
        )
        # A multiple's coefficient scales: 3 * off is a multiple of 192.
        self.assertEqual(
            ast.unparse(render(floor_divide(Affine.term("off", 3), 64, dict(FACTS)))),
            "off / 64 * 3",
        )

    def test_remainder(self) -> None:
        value = Affine.term("off") + Affine.term("lane") + Affine.of(8 * 3 + 1)
        self.assertEqual(
            ast.unparse(render(remainder(value, 8, dict(FACTS)))), "lane + 1"
        )
        self.assertEqual(
            ast.unparse(render(remainder(value, 16, dict(FACTS)))), "lane + 9"
        )

    def test_shared_floor(self) -> None:
        """When the rest spans multiples of the divisor, a constant below the
        rest's step can't carry it across one: values that differ only
        there share their quotient, ``(lane / 4)`` with ``lane`` even."""
        quotients = {
            ast.unparse(
                render(
                    floor_divide(
                        Affine.term("off") + Affine.term("lane") + Affine.of(c),
                        4,
                        dict(FACTS),
                    )
                )
            )
            for c in (16, 17)
        }
        self.assertEqual(quotients, {"lane / 4 + off / 4 + 4"})
        rem = remainder(
            Affine.term("off") + Affine.term("lane") + Affine.of(17), 4, dict(FACTS)
        )
        self.assertEqual(ast.unparse(render(rem)), "lane - lane / 4 * 4 + 1")

    def test_floor_term_of_several_names(self) -> None:
        """A floor term divides the whole of its dividend."""
        facts = {**FACTS, "off": Fact(8, 0)}
        value = Affine.term("off") + Affine.term("lane") + Affine.of(1)
        self.assertEqual(
            ast.unparse(render(floor_divide(value, 16, dict(facts)))),
            "(lane + off) / 16",
        )

    def test_large_integers(self) -> None:
        """Ranges and quotients are exact past a float's 53 bits."""
        rem = remainder(Affine.of(2**53 + 7), 8, {})
        self.assertEqual(ast.unparse(render(rem)), "7")

    def test_matches_floor_division(self) -> None:
        """Every fold of ``(a * off + b * lane + c * row + d) // q`` and
        ``% q``, and of a remainder's ``// q2`` and ``% q2`` in turn (its
        floor terms and negative coefficients, as later statements fold
        them), evaluates at each value of the names to Python's."""
        values = list(itertools.product(range(0, 129, 8), range(0, 7, 2), range(8)))
        facts = {"off": Fact(8, 0), "lane": Fact(2, 0, 6), "row": Fact(1, 0, 7)}

        def check(folded: Affine, expect: list[int]) -> None:
            # C's / is floor division here: every quotient's dividend is
            # nonnegative.
            expr = ast.unparse(render(folded)).replace("/", "//")
            code = compile(expr, "<fold>", "eval")
            for (off, lane, row), want in zip(values, expect, strict=True):
                env = {"off": off, "lane": lane, "row": row}
                self.assertEqual(eval(code, env), want, expr)

        folds = 0
        for a, b, c, d, q in itertools.product(
            (0, 1, 3, 8), (-1, 0, 1, 4), (-1, 0, 2), (0, 7, 33), (2, 8, 16, 128)
        ):
            value = (
                Affine.term("off", a)
                + Affine.term("lane", b)
                + Affine.term("row", c)
                + Affine.of(d)
            )
            exact = [a * off + b * lane + c * row + d for off, lane, row in values]
            known = dict(facts)
            quotient = floor_divide(value, q, known)
            if quotient is not None:
                folds += 1
                check(quotient, [v // q for v in exact])
            rem = remainder(value, q, known)
            if rem is None:
                continue
            check(rem, [v % q for v in exact])
            for q2 in (2, 4):
                again = floor_divide(rem, q2, dict(known))
                if again is not None:
                    folds += 1
                    check(again, [v % q // q2 for v in exact])
                again = remainder(rem, q2, dict(known))
                if again is not None:
                    check(again, [v % q % q2 for v in exact])
        self.assertGreater(folds, 300)

    def test_kernel_constants(self) -> None:
        """A kernel's constants are the names assigned once, a literal."""
        body = ast.parse(
            textwrap.dedent(
                """\
                a = 4
                b = 8
                if flag:
                    b = 9
                c = 1
                c += 1
                for d in range(3):
                    pass
                d = 2
                _helion_mpp_bind('e', 3)
                e = 5
                f = a + 1"""
            )
        ).body
        self.assertEqual(kernel_constants(body), {"a": 4})

    def test_no_fold(self) -> None:
        """A term unbounded below keeps the division; a nonnegative value
        whose range crosses a multiple divides as one floor term, C's ``/``
        (no floor_divide helper)."""
        self.assertIsNone(
            floor_divide(Affine.term("n") + Affine.term("lane"), 8, dict(FACTS))
        )
        # row + 4 spans 4..11, across 8.
        quotient = floor_divide(Affine.term("row") + Affine.of(4), 8, dict(FACTS))
        self.assertEqual(ast.unparse(render(quotient)), "(row + 4) / 8")
        # off is a multiple of 64, not of 128, and unbounded above.
        rem = remainder(Affine.term("off"), 128, dict(FACTS))
        self.assertEqual(ast.unparse(render(rem)), "off - off / 128 * 128")

    def test_signed_quotient(self) -> None:
        """A quotient of a signed index stays an ``int``: over a ``uint`` tile
        offset, ``off / 16 - 1`` at offset 0 would wrap to 0xFFFFFFFF where
        the floor division gives -1."""
        stmts = ast.parse(
            textwrap.dedent(
                """\
                idx = off + row
                signed = static_cast(int, idx)
                word = c10.metal.floor_divide(signed - 8, 16)
                nib = c10.metal.remainder(idx, 16)"""
            )
        ).body
        self.assertEqual(
            _src(fold(stmts, {}, dict(FACTS))).splitlines()[2:],
            ["word = static_cast(int, off / 16 - 1)", "nib = row"],
        )

    def test_fold_statements(self) -> None:
        """Statements fold in order: constants and bound names flow into
        later divisions, a ``uint3`` binding's components into subscripts
        and an index through its cast to ``int`` (folded as an ``int``), and
        statements that don't divide are kept as written."""
        stmts = ast.parse(
            textwrap.dedent(
                """\
                _helion_mpp_bind('tid', uint3(k / 2, 0, 0))
                idx = off + tid[0] * 2 + k % 2
                signed = static_cast(int, idx)
                word = c10.metal.floor_divide(signed, eight)
                nib = c10.metal.remainder(idx, eight)
                other = n // 8
                keep = word * 3"""
            )
        ).body
        env: dict[str, object] = {
            "eight": Affine.of(8),
            "k": Affine.term("lane") + Affine.of(17),
        }
        self.assertEqual(
            _src(fold(stmts, env, FACTS)),
            textwrap.dedent(
                """\
                _helion_mpp_bind('tid', uint3(lane / 2 + 8, 0, 0))
                idx = lane + off + 17
                signed = static_cast(int, idx)
                word = static_cast(int, off / 8 + 2)
                nib = lane + 1
                other = n // 8
                keep = word * 3"""
            ),
        )

    def test_c_division_needs_nonnegative(self) -> None:
        """C's ``/`` and ``%`` are floor division and remainder only at a
        dividend that can't be negative."""
        stmts = ast.parse("a = (row - 8) / 4\nb = row / 4\nc = (row - 8) // 4").body
        out = _src(fold(stmts, {}, FACTS))
        # Floor division pulls the constant's multiple of 4 out.
        self.assertEqual(out, "a = (row - 8) / 4\nb = row / 4\nc = row / 4 - 2")
        stmts = ast.parse("b = (row + 8) / 16").body
        self.assertEqual(_src(fold(stmts, {}, FACTS)), "b = 0")


if __name__ == "__main__":
    unittest.main()
