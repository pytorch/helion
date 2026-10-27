from __future__ import annotations

import collections
import textwrap
import unittest

from helion._testing import TestCase
from helion.runtime.triton import ttgir_tree as tree

_HEADER = """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
"""


def _module(body: str, params: str = "%p: !tt.ptr<i32>, %n: i32") -> str:
    return (
        _HEADER
        + f"  tt.func public @k({params}) attributes {{noinline = false}} {{\n"
        + textwrap.indent(textwrap.dedent(body), "    ")
        + "    tt.return loc(#loc)\n  } loc(#loc)\n} loc(#loc)\n"
        + '#loc = loc("k.py":1:1)\n'
    )


def _triton_ops(text: str) -> collections.Counter[str]:
    """Op names of ``text`` as Triton's own parser sees them (verified)."""
    names: list[str] = []
    tree.triton_parse(text).walk(lambda op: names.append(op.get_name()))
    return collections.Counter(names)


class TestTtgirTreeParse(TestCase):
    def assertMatchesTriton(self, text: str) -> tree.Module:
        m = tree.parse(text)
        self.assertEqual(m.text(), text)
        self.assertEqual([(op.name, nm) for op, nm in m.unresolved], [])
        tree.cross_check(m, tree.triton_parse(text))
        return m

    def test_round_trip_and_op_structure(self) -> None:
        text = _module(
            """\
            %c0_i32 = arith.constant 0 : i32 loc(#loc)
            %c1_i32 = arith.constant 1 : i32 loc(#loc)
            %r:2 = scf.for %iv = %c0_i32 to %n step %c1_i32 iter_args(%a = %c0_i32, %b = %c1_i32) -> (i32, i32)  : i32 {
              %s = arith.addi %a, %iv : i32 loc(#loc)
              scf.yield %s, %b : i32, i32 loc(#loc)
            } loc(#loc)
            %t = arith.addi %r#0, %r#1 : i32 loc(#loc)
            tt.store %p, %t : !tt.ptr<i32> loc(#loc)
            """
        )
        m = self.assertMatchesTriton(text)
        loop = next(op for op in m.walk() if op.name == "scf.for")
        self.assertEqual([(v.name, v.count) for v in loop.results], [("r", 2)])
        self.assertEqual(
            [v.name for v in loop.regions[0].header_arguments], ["iv", "a", "b"]
        )
        self.assertEqual(
            [u.value.name for u in loop.operands],
            ["c0_i32", "n", "c1_i32", "c0_i32", "c1_i32"],
        )
        add = next(op for op in m.walk() if op.first_line.lstrip().startswith("%t"))
        self.assertEqual(
            [(u.value.name, u.index) for u in add.operands], [("r", 0), ("r", 1)]
        )
        self.assertIs(add.operands[0].value.owner_op, loop)
        inner = next(op for op in m.walk() if op.first_line.lstrip().startswith("%s"))
        # %a, an argument of the loop body's entry block
        self.assertIs(inner.operands[0].value.owner_block, loop.regions[0].blocks[0])
        self.assertIs(inner.parent_op, loop)

    def test_names_reused_in_sibling_regions(self) -> None:
        reduce = """\
            %{r} = "tt.reduce"(%x) <{{axis = 0 : i32}}> ({{
            ^bb0(%u: i32 loc(#loc), %v: i32 loc(#loc)):
              %w = arith.addi %u, %v : i32 loc(#loc)
              tt.reduce.return %w : i32 loc(#loc)
            }}) : (tensor<32xi32, #blocked>) -> i32 loc(#loc)
            """
        text = _module(
            "%x = tt.splat %n : i32 -> tensor<32xi32, #blocked> loc(#loc)\n"
            + textwrap.dedent(reduce.format(r="r0"))
            + textwrap.dedent(reduce.format(r="r1"))
            + "%t = arith.addi %r0, %r1 : i32 loc(#loc)\n"
        )
        m = self.assertMatchesTriton(text)
        reds = [op for op in m.walk() if op.name == "tt.reduce"]
        adds = [op for op in m.walk() if op.first_line.lstrip().startswith("%w")]
        self.assertEqual(len(adds), 2)
        for red, add in zip(reds, adds, strict=True):
            owners = set()
            for u in add.operands:
                block = u.value.owner_block
                assert block is not None and block.parent is not None
                owners.add(block.parent.parent)
            self.assertEqual(owners, {red})

    def test_if_else_while_do_and_trailing_attributes(self) -> None:
        text = _module(
            """\
            %c0_i32 = arith.constant 0 : i32 loc(#loc)
            %cond = arith.cmpi sgt, %n, %c0_i32 : i32 loc(#loc)
            %v = scf.if %cond -> (i32) {
              scf.yield %n : i32 loc(#loc)
            } else {
              scf.yield %c0_i32 : i32 loc(#loc)
            } loc(#loc)
            %w = scf.while (%arg = %v) : (i32) -> i32 {
              %more = arith.cmpi sgt, %arg, %c0_i32 : i32 loc(#loc)
              scf.condition(%more) %arg : i32 loc(#loc)
            } do {
            ^bb0(%arg2: i32 loc(#loc)):
              %d = arith.subi %arg2, %n : i32 loc(#loc)
              scf.yield %d : i32 loc(#loc)
            } loc(#loc)
            %c1_i32 = arith.constant 1 : i32 loc(#loc)
            scf.for %i = %c0_i32 to %w step %c1_i32  : i32 {
              tt.store %p, %i : !tt.ptr<i32> loc(#loc)
            } {tt.num_stages = 2 : i32} loc(#loc)
            """
        )
        m = self.assertMatchesTriton(text)
        while_op = next(op for op in m.walk() if op.name == "scf.while")
        self.assertEqual(len(while_op.regions), 2)
        self.assertEqual(
            [v.name for v in while_op.regions[0].header_arguments], ["arg"]
        )
        self.assertEqual(
            [v.name for v in while_op.regions[1].blocks[0].arguments], ["arg2"]
        )
        loop = next(op for op in m.walk() if op.name == "scf.for")
        self.assertEqual(len(loop.regions), 1)
        self.assertIn("tt.num_stages", "".join(loop.parts[-1].lines))  # type: ignore[union-attr]

    def test_strings_hide_braces_and_percent_signs(self) -> None:
        text = _module(
            """\
            %x = tt.elementwise_inline_asm "{ .reg .u32 %r; mov.u32 $0, %tid.x; }" {constraints = "=r", packed_element = 1 : i32, pure = true}  -> i32 loc(#loc)
            tt.store %p, %x : !tt.ptr<i32> loc(#loc)
            """
        )
        m = self.assertMatchesTriton(text)
        asm = next(op for op in m.walk() if op.name == "tt.elementwise_inline_asm")
        self.assertEqual(asm.operands, [])
        self.assertEqual(asm.regions, [])

    def test_warp_specialize_partitions_continue_the_op(self) -> None:
        params = "%p: !tt.ptr<i32>, %n: i32"
        body = """\
            ttg.warp_specialize(%p, %n)
            default {
              tt.store %p, %n : !tt.ptr<i32> loc(#loc)
              ttg.warp_yield loc(#loc)
            }
            partition0(%q: !tt.ptr<i32> loc(#loc), %m: i32 loc(#loc)) num_warps(1) {
              tt.store %q, %m : !tt.ptr<i32> loc(#loc)
              ttg.warp_return loc(#loc)
            } : (!tt.ptr<i32>, i32) -> () loc(#loc)
            """
        m = self.assertMatchesTriton(_module(body, params))
        ws = next(op for op in m.walk() if op.name == "ttg.warp_specialize")
        self.assertEqual(len(ws.regions), 2)
        self.assertEqual([v.name for v in ws.regions[1].header_arguments], ["q", "m"])
        store = ws.regions[1].blocks[0].ops[0]
        self.assertEqual([u.value.name for u in store.operands], ["q", "m"])
        self.assertTrue(
            all(u.value.owner_block is ws.regions[1].blocks[0] for u in store.operands)
        )

    def test_parse_errors(self) -> None:
        with self.assertRaises(tree.ParseError):
            tree.parse("#a = 1\n")
        with self.assertRaises(tree.ParseError):
            tree.parse(_HEADER + '  tt.func @k() {\n    %x = "unterminated\n')


class TestTtgirTreeCrossCheck(TestCase):
    _STORE = """\
        %c0_i32 = arith.constant 0 : i32 loc(#loc)
        %v = arith.constant dense<0> : tensor<16x128xi32, #blocked2> loc(#loc)
        tt.descriptor_store %d[%c0_i32, %n], %v : DESC, tensor<16x128xi32, #blocked2> loc(#loc)
        """

    def _store_module(self) -> tuple[str, object]:
        layouts = (
            "#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], "
            "warpsPerCTA = [4, 1], order = [1, 0]}>\n"
            "#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, "
            "elementBitWidth = 32}>\n"
        )
        # Triton 3.8 spells the descriptor type without its tensor, 3.7 with.
        for desc in (
            "!tt.tensordesc<16x128xi32, #nvmma>",
            "!tt.tensordesc<tensor<16x128xi32, #nvmma>>",
        ):
            text = layouts + _module(
                self._STORE.replace("DESC", desc), f"%d: {desc}, %n: i32"
            )
            try:
                return text, tree.triton_parse(text)
            except tree.Mismatch:
                continue
        raise AssertionError("Triton parses neither descriptor spelling")

    def test_operands_come_in_triton_order(self) -> None:
        # The descriptor store prints its indices before the stored value;
        # Triton's operands are (descriptor, value, indices).
        text, mod = self._store_module()
        m = tree.parse(text)
        al = tree.cross_check(m, mod)
        store = next(op for op in m.walk() if op.name == "tt.descriptor_store")
        self.assertEqual(
            [u.value.name for u in store.operands], ["d", "c0_i32", "n", "v"]
        )
        self.assertEqual(
            [u.value.name for u in al.operands(store)], ["d", "v", "c0_i32", "n"]
        )
        self.assertTrue(al.type(al.operands(store)[0]).startswith("!tt.tensordesc<"))
        self.assertEqual(al.triton(store).get_name(), "tt.descriptor_store")

    def test_block_arguments_and_implicit_ops(self) -> None:
        # The loop's yield is elided (Triton still has it); its induction
        # variable is an argument of the body block.
        text = _module(
            """\
            %c0_i32 = arith.constant 0 : i32 loc(#loc)
            %c1_i32 = arith.constant 1 : i32 loc(#loc)
            scf.for %i = %c0_i32 to %n step %c1_i32  : i32 {
              tt.store %p, %i : !tt.ptr<i32> loc(#loc)
            } loc(#loc)
            """
        )
        m = tree.parse(text)
        al = tree.cross_check(m, tree.triton_parse(text))
        store = next(op for op in m.walk() if op.name == "tt.store")
        i = store.operands[1].value
        self.assertTrue(i.is_block_argument)
        self.assertEqual(al.type(i), "i32")
        self.assertEqual(al.type(store.operands[0]), "!tt.ptr<i32>")

    def test_disagreements_raise(self) -> None:
        body = """\
            %a = arith.addi %n, %n : i32 loc(#loc)
            %b = arith.muli %n, %n : i32 loc(#loc)
            tt.store %p, %a : !tt.ptr<i32> loc(#loc)
            """
        m = tree.parse(_module(body))
        tree.cross_check(m, tree.triton_parse(_module(body)))
        others = {
            "another op": body.replace("arith.muli", "arith.subi"),
            "an extra op": body + "%c = arith.addi %n, %n : i32 loc(#loc)\n",
            "another value": body.replace("tt.store %p, %a", "tt.store %p, %b"),
            "another result count": body.replace(
                "%b = arith.muli %n, %n : i32",
                "%b:2 = scf.if %t -> (i32, i32) {\n"
                "  scf.yield %n, %n : i32, i32 loc(#loc)\n"
                "} else {\n"
                "  scf.yield %n, %n : i32, i32 loc(#loc)\n"
                "}",
            ).replace("%a = ", "%t = arith.cmpi eq, %n, %n : i32 loc(#loc)\n%a = "),
        }
        for name, other in others.items():
            with self.subTest(name), self.assertRaises(tree.Mismatch):
                tree.cross_check(m, tree.triton_parse(_module(other)))
        with self.assertRaises(tree.Mismatch):
            tree.triton_parse(_module(body.replace("%n, %n", "%n, %missing", 1)))


class TestTtgirTreeEdit(TestCase):
    def test_move_before_keeps_the_module_valid(self) -> None:
        text = _module(
            """\
            %c1_i32 = arith.constant 1 : i32 loc(#loc)
            scf.for %i = %c1_i32 to %n step %c1_i32  : i32 {
              %slot = ttg.local_alloc : () -> !ttg.memdesc<32xi32, #shared, #smem, mutable> loc(#loc)
              tt.store %p, %i : !tt.ptr<i32> loc(#loc)
            } loc(#loc)
            """
        )
        m = tree.parse(text)
        loop = next(op for op in m.walk() if op.name == "scf.for")
        alloc = next(op for op in m.walk() if op.name == "ttg.local_alloc")
        alloc.move_before(loop)
        m.analyze()
        self.assertEqual([(op.name, nm) for op, nm in m.unresolved], [])
        self.assertIs(alloc.parent_op, loop.parent_op)
        out = m.text()
        self.assertIn("\n    %slot = ttg.local_alloc", out)  # re-indented
        self.assertLess(out.index("ttg.local_alloc"), out.index("scf.for"))
        self.assertEqual(_triton_ops(out), _triton_ops(text))

    def test_moving_a_use_above_its_definition_is_reported(self) -> None:
        text = _module(
            """\
            %c1_i32 = arith.constant 1 : i32 loc(#loc)
            scf.for %i = %c1_i32 to %n step %c1_i32  : i32 {
              %d = arith.addi %i, %n : i32 loc(#loc)
              tt.store %p, %d : !tt.ptr<i32> loc(#loc)
            } loc(#loc)
            """
        )
        m = tree.parse(text)
        loop = next(op for op in m.walk() if op.name == "scf.for")
        add = next(op for op in m.walk() if op.name == "arith.addi")
        add.move_before(loop)
        m.analyze()
        self.assertEqual(
            [(op.name, name) for op, name in m.unresolved], [("arith.addi", "i")]
        )

    def test_erase(self) -> None:
        text = _module("%c1_i32 = arith.constant 1 : i32 loc(#loc)\n")
        m = tree.parse(text)
        const = next(op for op in m.walk() if op.name == "arith.constant")
        const.erase()
        self.assertNotIn("arith.constant", m.text())
        self.assertEqual(m.text().count("\n"), text.count("\n") - 1)


if __name__ == "__main__":
    unittest.main()
