from __future__ import annotations

import ast
import contextlib
import importlib
import os
from pathlib import Path
import re
import sys
import textwrap
import threading
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest import mock

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.runtime.triton import ttgir_weight_prefetch as prefetch

if TYPE_CHECKING:
    from collections.abc import Callable

_HEADER = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
"""
_PARAMS = "%w: !tt.ptr<bf16>, %y: !tt.ptr<bf16>, %state: !tt.ptr<i32>"
_CONSTANTS = """\
%c0_i32 = arith.constant 0 : i32
%c1_i32 = arith.constant 1 : i32
%c16_i32 = arith.constant 16 : i32
%offs = arith.constant dense<1> : tensor<16x128xi32, #blocked>
%zero = arith.constant dense<0.000000e+00> : tensor<16x128xbf16, #blocked>
%ring = ttg.local_alloc : () -> !ttg.memdesc<2x16x128xbf16, #shared, #smem, mutable>
"""
_ACQUIRE = (
    'tt.elementwise_inline_asm "ld.acquire.gpu.global.u32 $0, [$1];" '
    '{constraints = "=r,l", packed_element = 1 : i32, pure = false} '
    "%state : !tt.ptr<i32> -> i32"
)


_GDC_WAIT = (
    '%gdc = tt.elementwise_inline_asm "griddepcontrol.wait; // dummy $0" '
    '{constraints = "=r", packed_element = 1 : i32, pure = false}  -> i32\n'
)
# The measurement harness's L2 prefetch and globaltimer stamp (allowlisted).
_L2_PREFETCH = (
    "{ .reg .pred p; .reg .u32 t; .reg .u64 a; mov.u32 t, %tid.x; "
    "setp.lt.u32 p, t, 8; mul.wide.u32 a, t, 16384; add.u64 a, a, $1; "
    "@p cp.async.bulk.prefetch.L2.global [a], 16384; mov.u32 $0, 0; }"
)
_STAMP = (
    "{ .reg .u64 t; mov.u64 t, %globaltimer; st.global.u64 [$1], t; mov.u32 $0, 0; }"
)


def _mlir_escape(text: str) -> str:
    """``text`` as MLIR prints a string literal's body."""
    return "".join(
        "\\\\"
        if c == "\\"
        else c
        if " " <= c <= "~" and c != '"'
        else f"\\{ord(c):02X}"
        for c in text
    )


def _wait(name: str = "poll") -> str:
    """A readiness spin as Helion emits it: poll, spin, warp sync."""
    return f"""\
%{name} = {_ACQUIRE}
scf.while (%{name}_arg = %{name}) : (i32) -> () {{
  %{name}_ne = arith.cmpi ne, %{name}_arg, %c16_i32 : i32
  scf.condition(%{name}_ne)
}} do {{
  %{name}_again = {_ACQUIRE}
  scf.yield %{name}_again : i32
}}
%{name}_lanes = tt.make_range {{end = 32 : i32, start = 0 : i32}} : tensor<32xi32, #blocked1>
%{name}_sync = tt.elementwise_inline_asm "bar.warp.sync 0xffffffff; mov.u32 $0, $1;" {{constraints = "=r,r", packed_element = 1 : i32, pure = false}} %{name}_lanes : tensor<32xi32, #blocked1> -> tensor<32xi32, #blocked1>
"""


_RELAXED = _ACQUIRE.replace("ld.acquire.gpu", "ld.relaxed.gpu")


def _two_counter_wait(relaxed: bool = True) -> str:
    """One spin polling two producers' counters, as Helion emits for a root
    with several incoming barriers: relaxed polls and one acquire fence (or,
    ``relaxed=False``, acquire polls)."""
    poll = _RELAXED if relaxed else _ACQUIRE
    fence = "fence.acquire.gpu; " if relaxed else ""
    return f"""\
%pa = {poll}
%pb = {poll}
scf.while (%pa_arg = %pa, %pb_arg = %pb) : (i32, i32) -> () {{
  %na = arith.cmpi ne, %pa_arg, %c16_i32 : i32
  %nb = arith.cmpi ne, %pb_arg, %c16_i32 : i32
  %either = arith.ori %na, %nb : i1
  scf.condition(%either)
}} do {{
  %pa_again = {poll}
  %pb_again = {poll}
  scf.yield %pa_again, %pb_again : i32, i32
}}
%both_lanes = tt.make_range {{end = 32 : i32, start = 0 : i32}} : tensor<32xi32, #blocked1>
%both_sync = tt.elementwise_inline_asm "{fence}bar.warp.sync 0xffffffff; mov.u32 $0, $1;" {{constraints = "=r,r", packed_element = 1 : i32, pure = false}} %both_lanes : tensor<32xi32, #blocked1> -> tensor<32xi32, #blocked1>
"""


def _copy(name: str, ptr: str, slot: int, buffer: str = "%ring") -> str:
    """One pipelined prologue copy of ``ptr`` into ``buffer[slot]``."""
    return f"""\
%{name}_ptrs = tt.splat {ptr} : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>, #blocked>
%{name}_addr = tt.addptr %{name}_ptrs, %offs : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>
%{name}_slot = ttg.memdesc_index {buffer}[%c{slot}_i32] : !ttg.memdesc<2x16x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xbf16, #shared, #smem, mutable>
%{name}_copy = ttg.async_copy_global_to_local %{name}_addr, %{name}_slot : tensor<16x128x!tt.ptr<bf16>, #blocked> -> <16x128xbf16, #shared, #smem, mutable>
%{name}_group = ttg.async_commit_group tokens %{name}_copy
"""


def _store(ptr: str, name: str = "dst") -> str:
    return f"""\
%{name} = tt.splat {ptr} : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>, #blocked>
tt.store %{name}, %zero : tensor<16x128x!tt.ptr<bf16>, #blocked>
"""


_LOOP = """\
scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
  %drained = ttg.async_wait {num = 0 : i32}
}
"""


def _module(body: str, params: str = _PARAMS) -> str:
    return (
        _HEADER
        + f"  tt.func public @kernel({params}) attributes {{noinline = false}} {{\n"
        + textwrap.indent(body, "    ")
        + "    tt.return\n  }\n}\n"
    )


def _line(text: str, pattern: str) -> int:
    for i, line in enumerate(text.splitlines()):
        if re.search(pattern, line):
            return i
    raise AssertionError(f"{pattern!r} not found")


def _poll_line(text: str, name: str = "poll") -> int:
    return _line(text, rf"_{name} = tt\.elementwise_inline_asm")


class TestTtgirWeightPrefetchRewrite(TestCase):
    def test_moves_read_only_copies_above_wait(self) -> None:
        text = _module(
            _CONSTANTS
            + _store("%y")
            + _wait()
            + _copy("w0", "%w", 0)
            + _copy("y0", "%y", 1)
            + _LOOP
        )
        out, report = prefetch.rewrite(text)
        self.assertEqual((report.hoisted, report.sites), (1, 1))
        poll = _poll_line(out)
        # The weight copy and its address and slot computation precede the wait.
        for name in ("w0_ptrs", "w0_addr", "w0_slot", "w0_copy"):
            self.assertLess(_line(out, rf"_{name} = "), poll, name)
        # The copy of the written tensor and every commit stay after it, in order.
        sync = _line(out, r"_poll_sync = ")
        groups = [_line(out, r"_w0_group = "), _line(out, r"_y0_group = ")]
        self.assertGreater(_line(out, r"_y0_copy = "), sync)
        self.assertGreater(groups[0], sync)
        self.assertLess(groups[0], groups[1])

    def test_unchanged_without_wait(self) -> None:
        text = _module(_CONSTANTS + _copy("w0", "%w", 0) + _LOOP)
        out, report = prefetch.rewrite(text)
        self.assertEqual(report.hoisted, 0)
        self.assertEqual(out, prefetch._uniquify(text))

    def test_resolves_names_reused_in_sibling_regions(self) -> None:
        # MLIR prints the same name in sibling regions; ``%p`` stores through
        # %y in the first branch and points into %w in the second.
        body = (
            _CONSTANTS
            + "%c = arith.cmpi eq, %c0_i32, %c1_i32 : i32\n"
            + "scf.if %c {\n"
            + textwrap.indent(_store("%y", name="p"), "  ")
            + "}\n"
            + "scf.if %c {\n"
            + textwrap.indent(
                "%p = tt.splat %w : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>, #blocked>\n"
                + _wait()
                + _copy("y0", "%y", 0)
                + _copy("w0", "%w", 1)
                + _LOOP,
                "  ",
            )
            + "}\n"
        )
        out, report = prefetch.rewrite(_module(body))
        self.assertEqual(report.hoisted, 1)
        poll = _poll_line(out)
        self.assertLess(_line(out, r"_w0_copy = "), poll)
        self.assertGreater(_line(out, r"_y0_copy = "), poll)

    def test_unattributed_store_writes_every_argument(self) -> None:
        params = _PARAMS + ", %table: !tt.ptr<!tt.ptr<bf16>>"
        body = (
            _CONSTANTS
            + "%anywhere = tt.load %table : !tt.ptr<!tt.ptr<bf16>>\n"
            + _store("%anywhere")
            + _wait()
            + _copy("w0", "%w", 0)
            + _LOOP
        )
        out, report = prefetch.rewrite(_module(body, params))
        self.assertEqual(report.hoisted, 0)
        self.assertGreater(_line(out, r"_w0_copy = "), _poll_line(out))

    def _asm_writer(self, op: str, after: str = "") -> str:
        """``op`` (an inline assembly or external call) on an integer address
        made from %w, before a spin and a copy of %w (``after``: more ops
        after the copy)."""
        return _module(
            _CONSTANTS
            + "%wi = tt.ptr_to_int %w : !tt.ptr<bf16> -> i64\n"
            + op
            + _wait()
            + _copy("w0", "%w", 0)
            + after
            + _LOOP
        )

    def test_assembly_store_through_integer_address_writes_its_argument(self) -> None:
        store = (
            '%st = tt.elementwise_inline_asm "st.global.u32 [$1], 7; mov.u32 $0, 0;" '
            '{constraints = "=r,l", packed_element = 1 : i32, pure = false} '
            "%wi : i64 -> i32\n"
        )
        _, report = prefetch.rewrite(self._asm_writer(store))
        self.assertEqual(report.hoisted, 0)
        # The allowlisted L2 prefetch (any decimal immediates) writes nothing;
        # the same prefetch spelled any other way is off the allowlist.
        text = "st.global.u32 [$1], 7; mov.u32 $0, 0;"
        l2 = store.replace(text, _L2_PREFETCH)
        self.assertEqual(self._hoisted(l2), 1)
        for other in (
            _L2_PREFETCH.replace("t, 8;", "t, 0x8;"),
            _L2_PREFETCH.replace("t, 8;", "t, -8;"),
            _L2_PREFETCH.replace("t, 8;", "t, $1;"),
            _L2_PREFETCH.replace("t, 8;", "t,  8;"),
            "@p cp.async.bulk.prefetch.L2.global [$1], 1024; mov.u32 $0, 0;",
        ):
            with self.subTest(other=other):
                self.assertEqual(self._hoisted(store.replace(text, other)), 0)
        # Mnemonics hidden behind a tab, a newline, a label or a comment.
        for hidden in (
            "\\09st.global.u32 [$1], 7;",
            "mov.u32 $0, 0;\\0Ast.global.u32 [$1], 7;",
            "L1: st.global.u32 [$1], 7;",
            "/* x */ st.global.u32 [$1], 7;",
        ):
            with self.subTest(hidden=hidden):
                op = store.replace("st.global.u32 [$1], 7;", hidden)
                _, report = prefetch.rewrite(self._asm_writer(op))
                self.assertEqual(report.hoisted, 0)

    def _hoisted(self, op: str, after: str = "") -> int:
        return prefetch.rewrite(self._asm_writer(op, after))[1].hoisted

    def test_writing_assembly_writes_every_argument(self) -> None:
        # Assembly can load or build the address it stores through, so one
        # that may write writes every argument, whatever its operands are.
        cases = {
            "store_through_y": (
                "st.global.u32 [$1], 7; mov.u32 $0, 0;",
                "=r,l",
                "%yi : i64",
            ),
            "address_loaded_from_y": (
                (
                    "{ .reg .b64 r; ld.global.u64 r, [$1]; st.global.u32 [r], 7; "
                    "mov.u32 $0, 0; }"
                ),
                "=r,l",
                "%yi : i64",
            ),
            "atomic_through_loaded_address": (
                (
                    "{ .reg .b64 r; .reg .b32 t; ld.global.u64 r, [$1]; "
                    "atom.global.add.u32 t, [r], 1; mov.u32 $0, t; }"
                ),
                "=r,l",
                "%yi : i64",
            ),
            "absolute_address_no_operand": (
                (
                    "{ .reg .b64 r; mov.b64 r, 140000000000000; "
                    "st.global.u32 [r], 7; mov.u32 $0, 0; }"
                ),
                "=r",
                "",
            ),
            "absolute_address_scalar_operand": (
                (
                    "{ .reg .b64 r; mov.b64 r, 140000000000000; "
                    "st.global.u32 [r], $1; mov.u32 $0, 0; }"
                ),
                "=r,r",
                "%c1_i32 : i32",
            ),
            "bulk_copy_from_y": (
                (
                    "cp.async.bulk.global.shared::cta.bulk_group [$1], [$1], 16; "
                    "mov.u32 $0, 0;"
                ),
                "=r,l",
                "%yi : i64",
            ),
        }
        yi = "%yi = tt.ptr_to_int %y : !tt.ptr<bf16> -> i64\n"
        for name, (text, constraints, operand) in cases.items():
            with self.subTest(name):
                op = (
                    f'%st = tt.elementwise_inline_asm "{text}" '
                    f'{{constraints = "{constraints}", packed_element = 1 : i32, '
                    f"pure = false}} {operand} -> i32\n"
                )
                self.assertEqual(self._hoisted(yi + op), 0)
        # Assembly off the allowlist writes every argument even if it only
        # loads; Helion's (allowlisted) poll through %y writes nothing.
        load = (
            '%ld = tt.elementwise_inline_asm "ld.global.u32 $0, [$1];" '
            '{constraints = "=r,l", packed_element = 1 : i32, pure = false} '
            "%yi : i64 -> i32\n"
        )
        self.assertEqual(self._hoisted(yi + load), 0)
        poll = load.replace("ld.global.u32", "ld.relaxed.gpu.global.u32")
        self.assertEqual(self._hoisted(yi + poll), 1)

    def test_assembly_whose_text_may_hide_instructions_writes(self) -> None:
        # A textual reading of the template must not miss instructions behind
        # a comment or a string: such assembly writes every argument, even if
        # every instruction it shows writes nothing.
        hidden = {
            "store_after_comment_with_block_start": (
                "mov.u32 $0, 0; // /*\\0A{ .reg .b16 h; mov.b16 h, 0x3F80; "
                "st.global.b16 [$1], h; } // */"
            ),
            "store_after_pragma_slashes": (
                ".pragma \\22//\\22; st.global.u32 [$1], 7; mov.u32 $0, 0;"
            ),
            "store_between_pragma_block_markers": (
                "mov.u32 $0, 0; .pragma \\22/*\\22; st.global.u32 [$1], 7; "
                ".pragma \\22*/\\22;"
            ),
            "store_after_carriage_return": (
                "mov.u32 $0, 0; // c\\0Dst.global.u32 [$1], 7;"
            ),
            "store_in_block_comment": "mov.u32 $0, 0; /* st.global.u32 [$1], 7; */",
            "loads_only_after_a_newline_comment": (
                "ld.global.u32 $0, [$1]; // x\\0Ald.global.u32 $0, [$1];"
            ),
            "loads_only_with_a_string": ".pragma \\22nounroll\\22; ld.global.u32 $0, [$1];",
            "loads_only_trailing_comment_with_tab": "ld.global.u32 $0, [$1]; // a\\09b",
        }
        for name, text in hidden.items():
            with self.subTest(name):
                op = (
                    f'%st = tt.elementwise_inline_asm "{text}" '
                    '{constraints = "=r,l", packed_element = 1 : i32, pure = false} '
                    "%wi : i64 -> i32\n"
                )
                self.assertEqual(self._hoisted(op), 0)
                self.assertEqual(
                    self._hoisted(op.replace("pure = false", "pure = true")), 0
                )
        # A comment of printable characters that runs to the end of the
        # template hides nothing (Triton's own gdc_launch_dependents).
        trailing = (
            '%st = tt.elementwise_inline_asm "griddepcontrol.launch_dependents; '
            '// dummy $0" {constraints = "=r", packed_element = 1 : i32, pure = false} '
            " -> i32\n"
        )
        self.assertEqual(self._hoisted(trailing), 1)

    def test_assembly_template_is_read_after_escape_decoding(self) -> None:
        # MLIR prints a quote, newline or tab in the template as two hex
        # digits and a backslash as two backslashes; the allowlist compares
        # the decoded template, exactly.
        string = (
            '%st = tt.elementwise_inline_asm ".pragma \\22x\\22; ld.global.u32 $0, [$1];" '
            '{constraints = "=r,l", packed_element = 1 : i32, pure = false} '
            "%wi : i64 -> i32\n"
        )
        self.assertEqual(self._hoisted(string), 0)
        fp4 = prefetch._HELION_FP4X2_TO_F32

        def convert(text: str) -> str:
            return (
                "%h = arith.constant 0 : i16\n"
                f'%cv:2 = tt.elementwise_inline_asm "{text}" '
                '{constraints = "=f,=f,h", packed_element = 1 : i32, pure = true} '
                "%h : i16 -> f32, f32\n"
            )

        self.assertEqual(self._hoisted(convert(_mlir_escape(fp4))), 1)
        # MLIR's lexer also reads \n as a newline.
        self.assertEqual(
            self._hoisted(convert(_mlir_escape(fp4).replace("\\0A", "\\n"))), 1
        )
        for name, other in {
            "backslash_n_instead_of_newline": fp4.replace("\n", "\\n"),
            "carriage_return_line_ends": fp4.replace("\n", "\r\n"),
            "one_more_space": fp4.replace("b0;", "b0 ;", 1),
            "trailing_newline_dropped": fp4.rstrip("\n"),
        }.items():
            with self.subTest(name):
                self.assertEqual(self._hoisted(convert(_mlir_escape(other))), 0)
        self.assertEqual(prefetch._mlir_string("a\\0Ab"), "a\nb")
        self.assertEqual(prefetch._mlir_string("a\\\\0Ab"), "a\\0Ab")
        self.assertEqual(prefetch._mlir_string('\\"\\t\\5C'), '"\t\\')
        self.assertIsNone(prefetch._mlir_string("a\\qb"))
        self.assertIsNone(prefetch._mlir_string("a\\"))
        self.assertIsNone(prefetch._mlir_string("a\\0"))

    def test_purity_is_read_from_the_attributes(self) -> None:
        # Purity is part of an allowlisted op: Helion's pure redux, declared
        # with side effects or without saying, is off the allowlist.
        asm = (
            '%st = tt.elementwise_inline_asm "redux.sync.min.s32 $0, $1, 0xffffffff;" '
            '{constraints = "=r,r", packed_element = 1 : i32, pure = SELF} '
            "%c1_i32 : i32 -> i32\n"
        )
        self.assertEqual(self._hoisted(asm.replace("SELF", "true")), 1)
        self.assertEqual(self._hoisted(asm.replace("SELF", "false")), 0)
        self.assertEqual(self._hoisted(asm.replace(", pure = SELF", "")), 0)
        self.assertEqual(
            self._hoisted(asm.replace("SELF", 'true, tag = "pure = true"')), 0
        )
        call = (
            '%st = tt.extern_elementwise %wi {libname = "", libpath = "", '
            'pure = SELF, symbol = "pure = OTHER"} : (i64) -> i32\n'
        )
        self.assertEqual(
            self._hoisted(call.replace("OTHER", "false").replace("SELF", "true")), 1
        )
        self.assertEqual(
            self._hoisted(call.replace("OTHER", "true").replace("SELF", "false")), 0
        )
        self.assertEqual(self._hoisted(call.replace("pure = SELF, ", "")), 0)

    def test_labels_and_predicates_do_not_hide_instructions(self) -> None:
        store = (
            '%st = tt.elementwise_inline_asm "PREFIX st.global.u32 [$1], 7; '
            'mov.u32 $0, 0;" {constraints = "=r,l", packed_element = 1 : i32, '
            "pure = true} %wi : i64 -> i32\n"
        )
        for prefix in (
            "L1:",
            "$L__spin:",
            "@p",
            "@!p",
            "@ p",
            ".loc 1 1 1",
            "{ L1: @%p1",
            "{ mov.u32 $0, 0 }",
            "{ $0 ",
        ):
            with self.subTest(prefix=prefix):
                self.assertEqual(self._hoisted(store.replace("PREFIX", prefix)), 0)
                impure = store.replace("PREFIX", prefix).replace(
                    "pure = true", "pure = false"
                )
                self.assertEqual(self._hoisted(impure), 0)
        # Loads behind a label or a predicate, or into a vector, are off the
        # allowlist: they write every argument.
        load = (
            '%st = tt.elementwise_inline_asm "{ .reg .pred p; $L__poll: '
            "ld.acquire.gpu.global.u32 $0, [$1]; setp.eq.u32 p, $0, 0; "
            '@p bra $L__poll; }" {constraints = "=r,l", packed_element = 1 : i32, '
            "pure = false} %wi : i64 -> i32\n"
        )
        self.assertEqual(self._hoisted(load), 0)
        vector = (
            '%st = tt.elementwise_inline_asm "{ .reg .b32 %a, %b; '
            'ld.global.v2.u32 {%a, %b}, [$1]; mov.u32 $0, %a; }" '
            '{constraints = "=r,l", packed_element = 1 : i32, pure = false} '
            "%wi : i64 -> i32\n"
        )
        self.assertEqual(self._hoisted(vector), 0)
        # Helion's own one-block readiness spin (label, predicate, branch) is
        # allowlisted.
        ready = (
            f'%st = tt.elementwise_inline_asm "{prefetch._READY_WAIT}mov.u32 $0, 0; }}" '
            '{constraints = "=r,l,r", packed_element = 1 : i32, pure = false} '
            "%wi, %c1_i32 : i64, i32 -> i32\n"
        )
        self.assertEqual(self._hoisted(ready), 1)

    def test_pure_assembly_with_any_memory_writing_instruction_writes(self) -> None:
        for text in (
            "sust.b.1d.b32.trap [$1, {$0}], {$0};",
            "sured.b.add.1d.u32.trap [$1, {$0}], $0;",
            (
                "wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [$1], "
                "{$0,$0,$0,$0,$0,$0,$0,$0};"
            ),
            "red.global.add.u32 [$1], 1;",
            "atom.global.exch.b32 $0, [$1], 1;",
            "cp.async.bulk.prefetch.L2.global [$1], 1024;",
            "st.bulk.weak [$1], 1024, 0;",
            "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [$1], [$1];",
        ):
            with self.subTest(text=text):
                op = (
                    f'%st = tt.elementwise_inline_asm "{text}" '
                    '{constraints = "=r,l", packed_element = 1 : i32, pure = true} '
                    "%wi : i64 -> i32\n"
                )
                self.assertEqual(self._hoisted(op), 0)

    def test_store_through_a_cast_pointer_writes_only_its_argument(self) -> None:
        # A pointer cast keeps naming its argument: a store through a cast of
        # %y writes %y only, one through a cast of %w writes %w.
        def store_through_cast(ptr: str) -> str:
            return (
                f"%cast = tt.bitcast {ptr} : !tt.ptr<bf16> -> !tt.ptr<i16>\n"
                "%cast_ptrs = tt.splat %cast : !tt.ptr<i16> -> tensor<16x128x!tt.ptr<i16>, #blocked>\n"
                "%cast_zero = arith.constant dense<0> : tensor<16x128xi16, #blocked>\n"
                "tt.store %cast_ptrs, %cast_zero : tensor<16x128x!tt.ptr<i16>, #blocked>\n"
            )

        self.assertEqual(self._hoisted(store_through_cast("%y")), 1)
        self.assertEqual(self._hoisted(store_through_cast("%w")), 0)

    def test_call_outside_the_module_writes_every_argument(self) -> None:
        call = "tt.call @elsewhere(%y) : (!tt.ptr<bf16>) -> ()\n"
        self.assertEqual(self._hoisted(call), 0)

    def test_called_function_gets_no_hoists(self) -> None:
        # A caller may pass one pointer for two parameters: %w and %y of a
        # called function may alias, so its copies of %w stay.
        callee = _module(
            _CONSTANTS + _store("%y") + _wait() + _copy("w0", "%w", 0) + _LOOP
        )
        out, report = prefetch.rewrite(callee)
        self.assertEqual(report.hoisted, 1)
        called = callee.replace(
            "    tt.return\n  }\n}",
            "    tt.return\n  }\n"
            "  tt.func public @caller(%a: !tt.ptr<bf16>, %s: !tt.ptr<i32>) "
            "attributes {noinline = false} {\n"
            "    tt.call @kernel(%a, %a, %s) : (!tt.ptr<bf16>, !tt.ptr<bf16>, "
            "!tt.ptr<i32>) -> ()\n    tt.return\n  }\n}",
        )
        out, report = prefetch.rewrite(called)
        self.assertEqual(report.hoisted, 0)
        self.assertTrue(any("called function" in s for s in report.skipped))

    def test_assembly_store_through_a_pointer_tensor_cast_to_integers(self) -> None:
        # A later root writes the weight through inline assembly on integer
        # addresses (``tt.ptr_to_int`` of pointer arithmetic on it), after
        # the copy: the copy must stay after the spin (the XR verifier's
        # CE3 / CE3b cases, for this pass).
        later = (
            "%wv = tt.splat %w : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>, #blocked>\n"
            "%wv_at = tt.addptr %wv, %offs : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>\n"
            "%wv_i = tt.ptr_to_int %wv_at : tensor<16x128x!tt.ptr<bf16>, #blocked> -> tensor<16x128xi64, #blocked>\n"
            '%wv_st = tt.elementwise_inline_asm "st.global.b16 [$1], 0; mov.u32 $0, 0;" '
            '{constraints = "=r,l", packed_element = 1 : i32, pure = false} '
            "%wv_i : tensor<16x128xi64, #blocked> -> tensor<16x128xi32, #blocked>\n"
        )
        out, report = prefetch.rewrite(self._asm_writer("", after=later))
        self.assertEqual(report.hoisted, 0)
        self.assertGreater(_line(out, r"_w0_copy = "), _poll_line(out))
        # Through a pointer bitcast too.
        cast = later.replace(
            "%wv_i = tt.ptr_to_int %wv_at",
            "%wv_c = tt.bitcast %wv_at : tensor<16x128x!tt.ptr<bf16>, #blocked> "
            "-> tensor<16x128x!tt.ptr<i16>, #blocked>\n%wv_i = tt.ptr_to_int %wv_c",
        ).replace(
            "tt.ptr_to_int %wv_c : tensor<16x128x!tt.ptr<bf16>, #blocked>",
            "tt.ptr_to_int %wv_c : tensor<16x128x!tt.ptr<i16>, #blocked>",
        )
        _, report = prefetch.rewrite(self._asm_writer("", after=cast))
        self.assertEqual(report.hoisted, 0)
        # The allowlisted stamp stores only through its operand: on (an
        # address in) %w it writes %w; on %y, only %y.
        stamp_w = later.replace("st.global.b16 [$1], 0; mov.u32 $0, 0;", _STAMP)
        _, report = prefetch.rewrite(self._asm_writer("", after=stamp_w))
        self.assertEqual(report.hoisted, 0)
        stamp_y = stamp_w.replace("%wv = tt.splat %w", "%wv = tt.splat %y")
        _, report = prefetch.rewrite(self._asm_writer("", after=stamp_y))
        self.assertEqual(report.hoisted, 1)
        self.assertTrue(any(w.endswith("_y") for w in report.written))
        self.assertFalse(any(w.endswith("_w") for w in report.written))
        # The same assembly reading (ld) is off the allowlist: it writes.
        load = later.replace("st.global.b16 [$1], 0;", "ld.global.b16 %rs1, [$1];")
        _, report = prefetch.rewrite(self._asm_writer("", after=load))
        self.assertEqual(report.hoisted, 0)

    def test_stamp_on_an_untraceable_address_writes_every_argument(self) -> None:
        # The stamp's address is an absolute integer, or integer arithmetic on
        # %y: it may lie in any argument.
        stamp = (
            f'%st = tt.elementwise_inline_asm "{_STAMP}" '
            '{constraints = "=r,l", packed_element = 1 : i32, pure = false} '
            "ADDR : i64 -> i32\n"
        )
        absolute = "%abs = arith.constant 1099511627776 : i64\n" + stamp.replace(
            "ADDR", "%abs"
        )
        self.assertEqual(self._hoisted(absolute), 0)
        offset = (
            "%yi = tt.ptr_to_int %y : !tt.ptr<bf16> -> i64\n"
            "%c8_i64 = arith.constant 8 : i64\n"
            "%yj = arith.addi %yi, %c8_i64 : i64\n" + stamp.replace("ADDR", "%yj")
        )
        self.assertEqual(self._hoisted(offset), 0)
        through_y = "%yi = tt.ptr_to_int %y : !tt.ptr<bf16> -> i64\n" + stamp.replace(
            "ADDR", "%yi"
        )
        self.assertEqual(self._hoisted(through_y), 1)

    def test_allowlist_entries_match_exactly(self) -> None:
        # Every entry is recognized as printed, and nothing else is: not
        # another template, constraint string, purity, packing or operand
        # count, nor another attribute or the generic op form.
        def line(
            template: str,
            constraints: str,
            pure: bool,
            operands: int,
            packed: int,
            extra: str = "",
        ) -> str:
            names = ", ".join(f"%a{k}" for k in range(operands))
            types = ", ".join("i32" for _ in range(operands))
            tail = f" {names} : {types} -> i32" if operands else "  -> i32"
            return (
                f'%r = tt.elementwise_inline_asm "{_mlir_escape(template)}" '
                f'{{constraints = "{_mlir_escape(constraints)}", packed_element = '
                f"{packed} : i32, pure = {str(pure).lower()}{extra}}}{tail}"
                ' loc("x")'
            )

        self.assertGreater(len(prefetch._ALLOWLIST), 10)
        for entry in prefetch._ALLOWLIST:
            template = entry.template.replace("{int}", "8")
            args = (
                template,
                entry.constraints,
                entry.pure,
                entry.operands,
                entry.packed,
            )
            with self.subTest(entry.source, template=template[:40]):
                self.assertIs(prefetch._asm_entry(line(*args)), entry)
                for changed in (
                    (template + " ", *args[1:]),
                    (" " + template, *args[1:]),
                    (template, entry.constraints + ",r", *args[2:]),
                    (*args[:2], not entry.pure, *args[3:]),
                    (*args[:3], entry.operands + 1, entry.packed),
                    (*args[:4], entry.packed + 1),
                ):
                    # Off the allowlist, or exactly another entry (smid is
                    # listed with both purities).
                    found = prefetch._asm_entry(line(*changed))
                    self.assertTrue(found is None or found.key() == changed)
                self.assertIsNone(prefetch._asm_entry(line(*args, extra=', tag = "x"')))
                generic = line(*args).replace(
                    "tt.elementwise_inline_asm", '"tt.elementwise_inline_asm"'
                )
                self.assertIsNone(prefetch._asm_entry(generic))
            if entry.kind == "writes":
                self.assertIsNotNone(entry.written_operand)
                self.assertLess(entry.written_operand, entry.operands)
            else:
                self.assertIsNone(entry.written_operand)
        keys = [e.key() for e in prefetch._ALLOWLIST]
        self.assertEqual(len(keys), len(set(keys)))
        # Only the measurement harness's prefetch has holes.
        self.assertEqual(
            [e.source for e in prefetch._ALLOWLIST if e.holes],
            [e.source for e in prefetch._ALLOWLIST if "_bulk_pf" in e.source],
        )
        self.assertEqual(
            [e.template for e in prefetch._ALLOWLIST if e.kind == "pdl_wait"],
            ["griddepcontrol.wait; // dummy $0"],
        )

    def test_side_effecting_external_call_writes_its_argument(self) -> None:
        call = (
            '%st = tt.extern_elementwise %wi {libname = "", libpath = "", '
            'pure = false, symbol = "store_word"} : (i64) -> i32\n'
        )
        _, report = prefetch.rewrite(self._asm_writer(call))
        self.assertEqual(report.hoisted, 0)
        _, report = prefetch.rewrite(
            self._asm_writer(call.replace("pure = false", "pure = true"))
        )
        self.assertEqual(report.hoisted, 1)

    def test_assembly_on_an_untraceable_address_writes_every_argument(self) -> None:
        store = (
            "%c8_i64 = arith.constant 8 : i64\n"
            "%wj = arith.addi %wi, %c8_i64 : i64\n"
            '%st = tt.elementwise_inline_asm "red.global.add.u32 [$1], 1; mov.u32 $0, 0;" '
            '{constraints = "=r,l", packed_element = 1 : i32, pure = false} '
            "%wj : i64 -> i32\n"
        )
        _, report = prefetch.rewrite(self._asm_writer(store))
        self.assertEqual(report.hoisted, 0)

    def test_pure_assembly_with_a_store_writes(self) -> None:
        store = (
            '%st = tt.elementwise_inline_asm "st.global.u32 [$1], 7; mov.u32 $0, 0;" '
            '{constraints = "=r,l", packed_element = 1 : i32, pure = true} '
            "%wi : i64 -> i32\n"
        )
        _, report = prefetch.rewrite(self._asm_writer(store))
        self.assertEqual(report.hoisted, 0)
        # Pure register arithmetic writes nothing only if it is allowlisted.
        _, report = prefetch.rewrite(
            self._asm_writer(store.replace("st.global.u32 [$1], 7; ", ""))
        )
        self.assertEqual(report.hoisted, 0)
        redux = (
            '%st = tt.elementwise_inline_asm "redux.sync.min.s32 $0, $1, 0xffffffff;" '
            '{constraints = "=r,r", packed_element = 1 : i32, pure = true} '
            "%c1_i32 : i32 -> i32\n"
        )
        self.assertEqual(self._hoisted(redux), 1)

    def test_callee_writes_count(self) -> None:
        callee = (
            "  tt.func private @helper(%dst: !tt.ptr<bf16>) attributes {noinline = true} {\n"
            "    %zero = arith.constant dense<0.000000e+00> : tensor<16x128xbf16, #blocked>\n"
            + textwrap.indent(_store("%dst"), "    ")
            + "    tt.return\n  }\n"
        )
        body = (
            _CONSTANTS
            + "tt.call @helper(%w) : (!tt.ptr<bf16>) -> ()\n"
            + _wait()
            + _copy("w0", "%w", 0)
            + _LOOP
        )
        text = _module(body).replace("  tt.func public", callee + "  tt.func public")
        _, report = prefetch.rewrite(text)
        self.assertEqual(report.hoisted, 0)

    def test_copy_after_side_effect_stays(self) -> None:
        text = _module(
            _CONSTANTS
            + _GDC_WAIT.replace("%gdc", "%entry_gdc")
            + _wait()
            + _copy("w0", "%w", 0)
            + _GDC_WAIT
            + _copy("w1", "%w", 1)
            + _LOOP
        )
        out, report = prefetch.rewrite(text)
        self.assertEqual(report.hoisted, 1)
        poll = _poll_line(out)
        self.assertLess(_line(out, r"_w0_copy = "), poll)
        self.assertGreater(_line(out, r"_w1_copy = "), _line(out, r"_gdc = "))

    def test_spin_without_a_dominating_pdl_wait_keeps_copies(self) -> None:
        # The module waits on its PDL predecessor, but not before this spin (only
        # in a sibling branch): a task that waits for an in-kernel producer may
        # rely on that producer's griddepcontrol.wait, so nothing moves.
        body = (
            _CONSTANTS
            + "%c = arith.cmpi eq, %c0_i32, %c1_i32 : i32\n"
            + "scf.if %c {\n"
            + textwrap.indent(_GDC_WAIT, "  ")
            + "}\n"
            + "scf.if %c {\n"
            + textwrap.indent(_wait() + _copy("w0", "%w", 0) + _LOOP, "  ")
            + "}\n"
        )
        out, report = prefetch.rewrite(_module(body))
        self.assertEqual(report.hoisted, 0)
        self.assertTrue(any("griddepcontrol" in s for s in report.skipped))
        # A wait at function entry dominates the spin.
        _, report = prefetch.rewrite(_module(_CONSTANTS + _GDC_WAIT + body))
        self.assertEqual(report.hoisted, 1)

    def test_only_an_unconditional_pdl_wait_dominates(self) -> None:
        # The module waits on its PDL predecessor after the copy; whether the
        # copy may move above the spin depends on what precedes the spin.
        def asm(text: str, pure: str = "false", name: str = "%gdc_top") -> str:
            return (
                f'{name} = tt.elementwise_inline_asm "{text}" {{constraints = "=r", '
                f"packed_element = 1 : i32, pure = {pure}}}  -> i32\n"
            )

        triton_wait = "griddepcontrol.wait; // dummy $0"

        def hoisted(top: str, later: str = _GDC_WAIT) -> int:
            body = _CONSTANTS + top + _wait() + _copy("w0", "%w", 0) + later + _LOOP
            return prefetch.rewrite(_module(body))[1].hoisted

        self.assertEqual(hoisted(""), 0)
        # Only Triton's own gdc_wait, exactly, is a wait.
        self.assertEqual(hoisted(asm(triton_wait)), 1)
        not_waits = {
            "plain": "griddepcontrol.wait; mov.u32 $0, 0;",
            "spaced": "griddepcontrol .wait; // dummy $0",
            "comment": "mov.u32 $0, 0; // griddepcontrol.wait",
            "after_newline_in_comment": "mov.u32 $0, 0; // x\\0Agriddepcontrol.wait;",
            "string": ".pragma \\22x\\22; griddepcontrol.wait; mov.u32 $0, 0;",
            "loc_hidden_branch": "{ .loc 1 1 1 bra $L_skip; griddepcontrol.wait; "
            ".loc 1 1 1 $L_skip: } mov.u32 $0, 0;",
            "predicated": "{ .reg .pred p; setp.eq.u32 p, 0, 1; @p griddepcontrol.wait; "
            "mov.u32 $0, 0; }",
            "branched_around": "{ bra $L_skip; griddepcontrol.wait; $L_skip: "
            "mov.u32 $0, 0; }",
        }
        for name, text in not_waits.items():
            with self.subTest(name):
                self.assertEqual(hoisted(asm(text)), 0)
        with self.subTest("pure"):
            self.assertEqual(hoisted(asm(triton_wait, "true")), 0)
        # Assembly off the allowlist anywhere in the function may branch
        # around a real gdc_wait (from one assembly block to a label in
        # another).
        around = (
            asm("bra L_x; mov.u32 $0, 0;", name="%t0")
            + asm(triton_wait, name="%t1")
            + asm("L_x: mov.u32 $0, 0;", name="%t2")
        )
        with self.subTest("branch_in_another_block"):
            self.assertEqual(hoisted(around), 0)
        # ... also from another function of the module (LLVM may inline it).
        body = (
            _CONSTANTS
            + asm(triton_wait)
            + _wait()
            + _copy("w0", "%w", 0)
            + _GDC_WAIT
            + _LOOP
        )
        callee = (
            "  tt.func private @helper() attributes {noinline = true} {\n"
            + textwrap.indent(asm("L_x: mov.u32 $0, 0;", name="%h"), "    ")
            + "    tt.return\n  }\n"
        )
        two = _module(body).replace("  tt.func public", callee + "  tt.func public")
        with self.subTest("label_in_another_function"):
            self.assertEqual(prefetch.rewrite(_module(body))[1].hoisted, 1)
            self.assertEqual(prefetch.rewrite(two)[1].hoisted, 0)
        # Assembly off the allowlist (whatever it says), or a wait, means the
        # module may wait on its predecessor: a dominating wait is required.
        for name in ("plain", "spaced", "predicated", "branched_around", "comment"):
            with self.subTest(later=name):
                self.assertEqual(hoisted("", asm(not_waits[name], name="%gdc")), 0)
        with self.subTest(later="pure_redux_off_the_list"):
            redux = (
                '%gdc = tt.elementwise_inline_asm "redux.sync.min.s32 $0, $1, 0xffffffff;" '
                '{constraints = "=r,r", packed_element = 1 : i32, pure = false} '
                "%c1_i32 : i32 -> i32\n"
            )
            self.assertEqual(hoisted("", redux), 0)
        self.assertEqual(hoisted("", ""), 1)
        # Other allowlisted assembly before the spin is no wait.
        with self.subTest("launch_dependents_on_top"):
            self.assertEqual(
                hoisted(asm("griddepcontrol.launch_dependents; // dummy $0")), 0
            )
        # Allowlisted assembly that does not wait leaves the module waitless.
        launch = asm("griddepcontrol.launch_dependents; // dummy $0", name="%gdc")
        self.assertEqual(hoisted("", launch), 1)
        # An external function with side effects may wait too, and so may a
        # function outside the module (each here in another function of the
        # module, so that the kernel's own written arguments do not change).
        body = _CONSTANTS + _wait() + _copy("w0", "%w", 0) + _LOOP

        def with_helper(op: str) -> str:
            helper = (
                "  tt.func private @helper(%v: i32) attributes {noinline = true} {\n"
                f"    {op}    tt.return\n  }}\n"
            )
            return _module(body).replace(
                "  tt.func public", helper + "  tt.func public"
            )

        extern = (
            '%e = tt.extern_elementwise %v {libname = "", libpath = "", '
            'pure = PURE, symbol = "f"} : (i32) -> i32\n'
        )
        self.assertEqual(prefetch.rewrite(_module(body))[1].hoisted, 1)
        self.assertEqual(
            prefetch.rewrite(with_helper(extern.replace("PURE", "false")))[1].hoisted, 0
        )
        self.assertEqual(
            prefetch.rewrite(with_helper(extern.replace("PURE", "true")))[1].hoisted, 1
        )
        call = "tt.call @elsewhere() : () -> ()\n"
        self.assertEqual(prefetch.rewrite(with_helper(call))[1].hoisted, 0)
        # And so may any assembly off the allowlist (a wait spelled with a
        # space, or a load), while allowlisted assembly that does not wait
        # leaves the module waitless.
        for text, expected in (
            ("griddepcontrol .wait; // dummy $0", 0),
            ("ld.global.u32 $0, [$1];", 0),
            ("griddepcontrol.launch_dependents; // dummy $0", 1),
        ):
            with self.subTest(helper_asm=text):
                op = asm(text, name="%h")
                self.assertEqual(prefetch.rewrite(with_helper(op))[1].hoisted, expected)

    def test_only_helion_polls_seed_a_spin(self) -> None:
        # A loop of other allowlisted reads (a globaltimer delay) followed by
        # Helion's warp sync is no readiness spin: nothing moves above it.
        timer = (
            'tt.elementwise_inline_asm "mov.u64 $0, %globaltimer;" '
            '{constraints = "=l", packed_element = 1 : i32, pure = false}  -> i64'
        )
        delay = _wait().replace(_ACQUIRE, timer).replace("(i32)", "(i64)")
        delay = delay.replace(": i32\n  scf.condition", ": i64\n  scf.condition")
        delay = delay.replace("%c16_i32 : i32", "%c16_i64 : i64").replace(
            "scf.yield %poll_again : i32", "scf.yield %poll_again : i64"
        )
        body = "%c16_i64 = arith.constant 16 : i64\n" + _CONSTANTS + delay
        self.assertEqual(self._hoisted_body(body + _copy("w0", "%w", 0) + _LOOP), 0)
        self.assertEqual(
            self._hoisted_body(_CONSTANTS + _wait() + _copy("w0", "%w", 0) + _LOOP), 1
        )

    def _hoisted_body(self, body: str) -> int:
        out, report = prefetch.rewrite(_module(body))
        return report.hoisted

    def test_copy_into_slot_in_use_stays(self) -> None:
        # A peeled first iteration reads slot 0 before the third copy.
        peeled = (
            "%ready = ttg.async_wait %w0_group {num = 0 : i32}\n"
            "%a = ttg.local_load %w0_slot token %ready : "
            "!ttg.memdesc<16x128xbf16, #shared, #smem, mutable> -> tensor<16x128xbf16, #blocked>\n"
        )
        text = _module(
            _CONSTANTS
            + _wait()
            + _copy("w0", "%w", 0)
            + peeled
            + _copy("w1", "%w", 1)
            + _copy("w2", "%w", 0)
            + _LOOP
        )
        out, report = prefetch.rewrite(text)
        self.assertEqual(report.hoisted, 2)
        poll = _poll_line(out)
        self.assertLess(_line(out, r"_w1_copy = "), poll)
        self.assertGreater(_line(out, r"_w2_copy = "), _line(out, r"_a = "))

    def test_copy_depending_on_wait_stays(self) -> None:
        dependent = """\
%seen = tt.splat %poll : i32 -> tensor<16x128xi32, #blocked>
%w0_ptrs = tt.splat %w : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>, #blocked>
%w0_addr = tt.addptr %w0_ptrs, %seen : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>
%w0_slot = ttg.memdesc_index %ring[%c0_i32] : !ttg.memdesc<2x16x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xbf16, #shared, #smem, mutable>
%w0_copy = ttg.async_copy_global_to_local %w0_addr, %w0_slot : tensor<16x128x!tt.ptr<bf16>, #blocked> -> <16x128xbf16, #shared, #smem, mutable>
"""
        _, report = prefetch.rewrite(_module(_CONSTANTS + _wait() + dependent + _LOOP))
        self.assertEqual(report.hoisted, 0)

    def test_back_to_back_waits_form_one_site(self) -> None:
        text = _module(
            _CONSTANTS + _wait("first") + _wait("second") + _copy("w0", "%w", 0) + _LOOP
        )
        out, report = prefetch.rewrite(text)
        self.assertEqual((report.hoisted, report.sites), (1, 1))
        self.assertLess(_line(out, r"_w0_copy = "), _poll_line(out, "first"))

    def test_spin_polling_several_counters_is_one_site(self) -> None:
        for relaxed in (True, False):
            with self.subTest(relaxed=relaxed):
                text = _module(
                    _CONSTANTS
                    + _two_counter_wait(relaxed)
                    + _copy("w0", "%w", 0)
                    + _LOOP
                )
                out, report = prefetch.rewrite(text)
                self.assertEqual((report.hoisted, report.sites), (1, 1))
                self.assertLess(
                    _line(out, r"_w0_copy = "),
                    _line(out, r"_pa = tt\.elementwise_inline_asm"),
                )

    def test_system_scope_function_is_skipped(self) -> None:
        peer = (
            "%one = arith.constant 1 : i32\n"
            "%true = arith.constant true\n"
            "%old = tt.atomic_rmw add, acq_rel, sys, %state, %one, %true : "
            "(!tt.ptr<i32>, i32, i1) -> i32\n"
        )
        text = _module(_CONSTANTS + peer + _wait() + _copy("w0", "%w", 0) + _LOOP)
        _, report = prefetch.rewrite(text)
        self.assertEqual(report.hoisted, 0)
        self.assertTrue(any("system-scope" in s for s in report.skipped))

    def test_moves_tma_weight_copies_with_their_stage_barriers(self) -> None:
        body = (
            _TMA_SETUP
            + _wait()
            + _tma_buffers()
            + _tma_stage(0, 0, "%bar0")
            + _peeled("%bar0", "%w0")
            + _tma_stage(1, 1, "%bar1")
            + _LOOP
        )
        out, report = prefetch.rewrite(_module(body, _TMA_PARAMS))
        self.assertEqual(report.hoisted, 2)
        poll = _poll_line(out)
        sync = _line(out, r"_poll_sync = ")
        for stage in (0, 1):
            init = _line(out, rf"init_barrier %u\d+_bar{stage},")
            expect = _line(out, rf"barrier_expect %u\d+_bar{stage},")
            copy = _tma_line(out, "w_desc", stage)
            self.assertLess(init, expect)
            self.assertLess(expect, copy)
            self.assertLess(copy, poll)
            # The activation copy still arrives on that barrier after the wait.
            self.assertGreater(_tma_line(out, "x_desc", stage), sync)
        self.assertGreater(_line(out, r"wait_barrier "), sync)

    def test_reused_tma_barrier_keeps_later_phase(self) -> None:
        body = (
            _TMA_SETUP
            + _wait()
            + _tma_buffers()
            + _tma_stage(0, 0, "%bar0")
            + _peeled("%bar0", "%w0")
            + _tma_stage(1, 1, "%bar0")
            + _LOOP
        )
        out, report = prefetch.rewrite(_module(body, _TMA_PARAMS))
        self.assertEqual(report.hoisted, 1)
        self.assertLess(_tma_line(out, "w_desc", 0), _poll_line(out))
        self.assertGreater(_tma_line(out, "w_desc", 1), _line(out, r"wait_barrier "))

    def test_copy_through_written_descriptor_stays(self) -> None:
        body = (
            _TMA_SETUP.replace(
                "%x_desc[%c0_i32, %c0_i32], %xval", "%w_desc[%c0_i32, %c0_i32], %xval"
            )
            + _wait()
            + _tma_buffers()
            + _tma_stage(0, 0, "%bar0")
            + _LOOP
        )
        out, report = prefetch.rewrite(_module(body, _TMA_PARAMS))
        self.assertEqual(report.hoisted, 1)
        self.assertGreater(_tma_line(out, "w_desc", 0), _line(out, r"_poll_sync = "))
        self.assertLess(_tma_line(out, "x_desc", 0), _poll_line(out))

    def test_stages_hook_keys_both_caches_with_the_pass_version(self) -> None:
        def previous(*args: object) -> object:
            return ("other-hook", "abc") if not args else None

        def make_ttgir(src: object, metadata: object) -> object:
            return src

        hook = prefetch._StagesHook(previous)
        # A compile outside an enabled() context (another thread, Triton's
        # async compile) gets neither the pass nor its key.
        self.assertEqual(hook(), ("other-hook", "abc"))
        stages: dict[str, object] = {"ttgir": make_ttgir}
        hook(None, stages, None, None, None)
        self.assertIs(stages["ttgir"], make_ttgir)
        with prefetch.enabled():
            key, digest = cast("tuple[str, str]", hook())
            self.assertTrue(key.startswith("other-hook"))
            self.assertIn(prefetch._DIGEST, key)
            alone_key, alone_digest = cast(
                "tuple[str, str]", prefetch._StagesHook(None)()
            )
            self.assertNotEqual(digest, alone_digest)
            self.assertIn(prefetch._DIGEST, alone_key)
            hook(None, stages, None, None, None)
            self.assertIsNot(stages["ttgir"], make_ttgir)
            # The launch's storage classes are part of both keys.
            with prefetch.enabled([("x_desc", "x"), ("w",)]):
                aliased_key, aliased_digest = cast("tuple[str, str]", hook())
                self.assertEqual(
                    prefetch._current_facts(), prefetch._Facts((("x", "x_desc"),))
                )
            self.assertNotEqual(aliased_key, key)
            self.assertNotEqual(aliased_digest, digest)
            self.assertEqual(hook(), (key, digest))
            # The kernel's ring setting (as its launch passes it) is too.
            with prefetch.enabled(ring=True):
                ring_key, ring_digest = cast("tuple[str, str]", hook())
                ring_stages: dict[str, object] = {
                    "ttgir": make_ttgir,
                    "llir": make_ttgir,
                    "ptx": make_ttgir,
                    "cubin": make_ttgir,
                }
                hook(None, ring_stages, None, None, None)
                self.assertIsInstance(
                    getattr(ring_stages["llir"], "__self__", None), prefetch._Fallback
                )
            self.assertTrue(key.endswith("-ring0"))
            self.assertTrue(ring_key.endswith("-ring1"))
            self.assertNotEqual(ring_digest, digest)
            elsewhere: list[object] = []
            thread = threading.Thread(target=lambda: elsewhere.append(hook()))
            thread.start()
            thread.join()
            self.assertEqual(elsewhere, [("other-hook", "abc")])

    def test_arguments_sharing_a_storage_are_written_together(self) -> None:
        # %x (stored through as a pointer) and %x_desc (read by TMA) view one
        # tensor when the launch says they share a storage.
        params = (
            '%w_desc: !tt.tensordesc<128x128xbf16, #shared> loc("w_desc"(#loc)), '
            '%x: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("x"(#loc)), '
            "%x_desc: !tt.tensordesc<16x128xbf16, #shared> loc(#loc7), "
            '%state: !tt.ptr<i32> loc("state"(#loc))'
        )
        setup = _TMA_SETUP.replace(
            "tt.descriptor_store %x_desc[%c0_i32, %c0_i32], %xval : "
            "!tt.tensordesc<16x128xbf16, #shared>, tensor<16x128xbf16, #blocked>",
            "%xs = tt.splat %x : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>, #blocked>\n"
            "tt.store %xs, %xval : tensor<16x128x!tt.ptr<bf16>, #blocked>",
        )
        body = setup + _wait() + _tma_buffers() + _tma_stage(0, 0, "%bar0") + _LOOP
        text = '#loc7 = loc("x_desc"(#loc))\n' + _module(body, params)
        _, report = prefetch.rewrite(text)
        self.assertEqual(report.hoisted, 2)
        self.assertEqual(report.written, ["x"])
        out, report = prefetch.rewrite(text, (("x", "x_desc"),))
        self.assertEqual(report.hoisted, 1)
        self.assertEqual(report.written, ["x", "x_desc"])
        self.assertLess(_tma_line(out, "w_desc", 0), _poll_line(out))
        self.assertGreater(_tma_line(out, "x_desc", 0), _line(out, r"_poll_sync = "))
        # A class naming an argument the function does not have: no hoists.
        _, report = prefetch.rewrite(text, (("x", "nowhere"),))
        self.assertEqual(report.hoisted, 0)
        self.assertTrue(any("storage classes" in r for r in report.skipped))
        # Without source names the classes cannot be resolved either.
        _, report = prefetch.rewrite(
            _module(body, params.replace(' loc("x"(#loc))', "")), (("x", "x_desc"),)
        )
        self.assertEqual(report.hoisted, 0)

    def test_uniquify_names_every_value_once(self) -> None:
        body = (
            _CONSTANTS
            + "%c = arith.cmpi eq, %c0_i32, %c1_i32 : i32\n"
            + "scf.if %c {\n  %v = arith.addi %c0_i32, %c1_i32 : i32\n}\n"
            + "scf.if %c {\n  %v = arith.muli %c0_i32, %c1_i32 : i32\n}\n"
        )
        out = prefetch._uniquify(_module(body))
        defs = [prefetch._results(line) for line in out.splitlines()]
        names = [n for d in defs for n in d]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum(1 for n in names if n.endswith("_v")), 2)


_TMA_PARAMS = (
    "%w_desc: !tt.tensordesc<128x128xbf16, #shared>, "
    "%x_desc: !tt.tensordesc<16x128xbf16, #shared>, %state: !tt.ptr<i32>"
)
_BAR = "!ttg.memdesc<1xi64, #shared1, #smem, mutable>"
_TMA_SETUP = """\
%c0_i32 = arith.constant 0 : i32
%c1_i32 = arith.constant 1 : i32
%c16_i32 = arith.constant 16 : i32
%c128_i32 = arith.constant 128 : i32
%true = arith.constant true
%xval = arith.constant dense<0.000000e+00> : tensor<16x128xbf16, #blocked>
tt.descriptor_store %x_desc[%c0_i32, %c0_i32], %xval : !tt.tensordesc<16x128xbf16, #shared>, tensor<16x128xbf16, #blocked>
"""


def _tma_buffers() -> str:
    return f"""\
%bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%bar0 = ttg.memdesc_index %bars[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> {_BAR}
ttng.init_barrier %bar0, 1 : {_BAR}
%bar1 = ttg.memdesc_index %bars[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> {_BAR}
ttng.init_barrier %bar1, 1 : {_BAR}
%wring = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>
%xring = ttg.local_alloc : () -> !ttg.memdesc<2x16x128xbf16, #shared, #smem, mutable>
"""


def _tma_stage(stage: int, slot: int, barrier: str) -> str:
    return f"""\
%w{stage} = ttg.memdesc_index %wring[%c{slot}_i32] : !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%x{stage} = ttg.memdesc_index %xring[%c{slot}_i32] : !ttg.memdesc<2x16x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect {barrier}, 36864, %true : {_BAR}
ttng.async_tma_copy_global_to_local %w_desc[%c0_i32, %c{slot}_i32] %w{stage}, {barrier}, %true : !tt.tensordesc<128x128xbf16, #shared>, {_BAR} -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.async_tma_copy_global_to_local %x_desc[%c0_i32, %c{slot}_i32] %x{stage}, {barrier}, %true : !tt.tensordesc<16x128xbf16, #shared>, {_BAR} -> !ttg.memdesc<16x128xbf16, #shared, #smem, mutable>
"""


def _peeled(barrier: str, slot_value: str) -> str:
    return f"""\
ttng.wait_barrier {barrier}, %c0_i32, %true : {_BAR}
%a = ttg.local_load {slot_value} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> tensor<128x128xbf16, #blocked>
"""


def _tma_line(text: str, desc: str, stage: int) -> int:
    return _line(
        text,
        rf"async_tma_copy_global_to_local %u\d+_{desc}\[.*%u\d+_{['w', 'x'][desc == 'x_desc']}{stage},",
    )


def _norm_gemv(x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    t_rows, k_dim = x.shape
    n_dim = w.size(0)
    y = torch.empty_like(x)
    out = torch.empty([n_dim, t_rows], dtype=torch.bfloat16, device=x.device)
    for t in hl.tile(t_rows, block_size=1):
        row = x[t, :].to(torch.float32)
        ms = torch.sum(row * row, dim=-1, keepdim=True) / k_dim
        y[t, :] = (row * torch.rsqrt(ms + 1e-6) * gamma[None, :].to(torch.float32)).to(
            torch.bfloat16
        )
    for n_t in hl.tile(n_dim, block_size=16):
        acc = hl.zeros([n_t, t_rows], dtype=torch.float32)
        for k_t in hl.tile(k_dim, block_size=128):
            acc = hl.dot(w[n_t, k_t], y[:, k_t].permute(1, 0), acc=acc)
        out[n_t, :] = acc.to(torch.bfloat16)
    return out


def _norm_gemv_out(
    x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    t_rows, k_dim = x.shape
    n_dim = w.size(0)
    out = torch.empty([n_dim, t_rows], dtype=torch.bfloat16, device=x.device)
    for t in hl.tile(t_rows, block_size=1):
        row = x[t, :].to(torch.float32)
        ms = torch.sum(row * row, dim=-1, keepdim=True) / k_dim
        y[t, :] = (row * torch.rsqrt(ms + 1e-6) * gamma[None, :].to(torch.float32)).to(
            torch.bfloat16
        )
    for n_t in hl.tile(n_dim, block_size=16):
        acc = hl.zeros([n_t, t_rows], dtype=torch.float32)
        for k_t in hl.tile(k_dim, block_size=128):
            acc = hl.dot(w[n_t, k_t], y[:, k_t].permute(1, 0), acc=acc)
        out[n_t, :] = acc.to(torch.bfloat16)
    return out


def _norm_gemv_masked(
    x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    tokens, k_dim = x.shape
    n_dim = w.size(0)
    out = torch.empty([n_dim, 16], dtype=torch.bfloat16, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        row = x[t, :].to(torch.float32)
        ms = torch.sum(row * row, dim=-1, keepdim=True) / k_dim
        y[t, :] = (row * torch.rsqrt(ms + 1e-6) * gamma[None, :].to(torch.float32)).to(
            torch.bfloat16
        )
    for n_t in hl.tile(n_dim, block_size=32):
        rows = hl.arange(16)
        live = rows < tokens
        acc = hl.zeros([n_t, 16], dtype=torch.float32)
        for k_t in hl.tile(k_dim, block_size=256):
            yk = hl.load(y, [rows, k_t], extra_mask=live[:, None])
            acc = hl.dot(w[n_t, k_t], yk.permute(1, 0), acc=acc)
        out[n_t, :] = acc.to(torch.bfloat16)
    return out


def _norm_gemv_wide(
    x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    t_rows, k_dim = x.shape
    n_dim = w.size(0)
    y = torch.empty_like(x)
    out = torch.empty([n_dim, t_rows], dtype=torch.bfloat16, device=x.device)
    for t in hl.tile(t_rows, block_size=1):
        row = x[t, :].to(torch.float32)
        ms = torch.sum(row * row, dim=-1, keepdim=True) / k_dim
        y[t, :] = (row * torch.rsqrt(ms + 1e-6) * gamma[None, :].to(torch.float32)).to(
            torch.bfloat16
        )
    for n_t in hl.tile(n_dim, block_size=128):
        acc = hl.zeros([n_t, t_rows], dtype=torch.float32)
        for k_t in hl.tile(k_dim, block_size=64):
            acc = hl.dot(w[n_t, k_t], y[:, k_t].permute(1, 0), acc=acc)
        out[n_t, :] = acc.to(torch.bfloat16)
    return out


def _norm_gemv_pdl(
    x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    t_rows, k_dim = x.shape
    n_dim = w.size(0)
    y = torch.empty_like(x)
    out = torch.empty([n_dim, t_rows], dtype=torch.bfloat16, device=x.device)
    hl.pdl_wait()
    for t in hl.tile(t_rows, block_size=1):
        row = x[t, :].to(torch.float32)
        ms = torch.sum(row * row, dim=-1, keepdim=True) / k_dim
        y[t, :] = (row * torch.rsqrt(ms + 1e-6) * gamma[None, :].to(torch.float32)).to(
            torch.bfloat16
        )
    for n_t in hl.tile(n_dim, block_size=16):
        acc = hl.zeros([n_t, t_rows], dtype=torch.float32)
        for k_t in hl.tile(k_dim, block_size=128):
            acc = hl.dot(w[n_t, k_t], y[:, k_t].permute(1, 0), acc=acc)
        out[n_t, :] = acc.to(torch.bfloat16)
    hl.pdl_launch_dependents()
    return out


def _norm_gemv_exit_trigger(
    x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    t_rows, k_dim = x.shape
    n_dim = w.size(0)
    y = torch.empty_like(x)
    out = torch.empty([n_dim, t_rows], dtype=torch.bfloat16, device=x.device)
    for t in hl.tile(t_rows, block_size=1):
        row = x[t, :].to(torch.float32)
        ms = torch.sum(row * row, dim=-1, keepdim=True) / k_dim
        y[t, :] = (row * torch.rsqrt(ms + 1e-6) * gamma[None, :].to(torch.float32)).to(
            torch.bfloat16
        )
    for n_t in hl.tile(n_dim, block_size=16):
        acc = hl.zeros([n_t, t_rows], dtype=torch.float32)
        for k_t in hl.tile(k_dim, block_size=128):
            acc = hl.dot(w[n_t, k_t], y[:, k_t].permute(1, 0), acc=acc)
        out[n_t, :] = acc.to(torch.bfloat16)
    hl.pdl_launch_dependents()
    return out


def _two_norm_gemv(
    x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    t_rows, k_dim = x.shape
    n_dim = w.size(0)
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    out = torch.empty([n_dim, t_rows], dtype=torch.bfloat16, device=x.device)
    for t in hl.tile(t_rows, block_size=1):
        row = x[t, :].to(torch.float32)
        y[t, :] = (row * gamma[None, :].to(torch.float32)).to(torch.bfloat16)
    for t in hl.tile(t_rows, block_size=1):
        row = x[t, :].to(torch.float32)
        z[t, :] = (row * row).to(torch.bfloat16)
    for n_t in hl.tile(n_dim, block_size=16):
        acc = hl.zeros([n_t, t_rows], dtype=torch.float32)
        for k_t in hl.tile(k_dim, block_size=128):
            acc = hl.dot(w[n_t, k_t], y[:, k_t].permute(1, 0), acc=acc)
            acc = hl.dot(w[n_t, k_t], z[:, k_t].permute(1, 0), acc=acc)
        out[n_t, :] = acc.to(torch.bfloat16)
    return out


# A 20 us delay of allowlisted assembly only (the pass stays engaged).
_SPIN_20US = """
t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
t1 = t0
while t1 - t0 < 20000:
    t1 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
0
"""


def _late_act_gemv(
    x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """The GEMV's activation ``y`` is written 20 us late by the first root:
    the ring's activation producer must wait for it like the compute warps."""
    t_rows, k_dim = x.shape
    n_dim = w.size(0)
    out = torch.empty([n_dim, t_rows], dtype=torch.bfloat16, device=x.device)
    for t in hl.tile(t_rows, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        y[t, :] = (x[t, :].to(torch.float32) * gamma[None, :].to(torch.float32)).to(
            torch.bfloat16
        )
    for n_t in hl.tile(n_dim, block_size=16):
        acc = hl.zeros([n_t, t_rows], dtype=torch.float32)
        for k_t in hl.tile(k_dim, block_size=128):
            acc = hl.dot(w[n_t, k_t], y[:, k_t].permute(1, 0), acc=acc)
        out[n_t, :] = acc.to(torch.bfloat16)
    return out


_CONFIG = helion.Config(
    block_sizes=[],
    cross_loop_pipeline="dynamic",
    num_warps=4,
    num_stages=2,
    pid_type="persistent_blocked",
    indexing="pointer",
    range_num_stages=[0, 0, 4],
)


_TMA_CONFIG = helion.Config(
    block_sizes=[],
    cross_loop_pipeline="dynamic",
    num_warps=4,
    num_stages=2,
    pid_type="persistent_blocked",
    indexing="tensor_descriptor",
    range_num_stages=[0, 0, 4],
)
_TWO_PRODUCER_CONFIG = helion.Config(
    block_sizes=[],
    cross_loop_pipeline="dynamic",
    num_warps=4,
    num_stages=2,
    pid_type="persistent_blocked",
    indexing="pointer",
    range_num_stages=[0, 0, 0, 4],
)


def _ring_parts(ttgir: str) -> tuple[str, list[str]]:
    """(default region, producer partitions) of the pass-built warp
    specialization in a compiled module's TTGIR."""
    body = ttgir[ttgir.index("ttg.warp_specialize") :]
    parts = re.split(r"\n\s*partition\d+\(", body)
    return parts[0], parts[1:]


def _compiled_ttgir(bound: object) -> list[str]:
    module = sys.modules[bound._run.__module__]  # pyrefly: ignore[missing-attribute]
    jit = module._helion__norm_gemv
    return [
        compiled.asm["ttgir"]
        for caches in jit.device_caches.values()
        for compiled in caches[0].values()
    ]


@onlyBackends(["triton"])
class TestTtgirWeightPrefetchCompile(TestCase):
    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        x = torch.randn(16, 512, device=DEVICE, dtype=torch.bfloat16)
        gamma = (1 + 0.1 * torch.randn(512, device=DEVICE)).bfloat16()
        w = (0.05 * torch.randn(256, 512, device=DEVICE)).bfloat16()
        return x, gamma, w

    def _kernel(self, enabled: bool, ring: bool = True) -> helion.Kernel:
        return helion.kernel(
            _norm_gemv,
            config=_CONFIG,
            static_shapes=True,
            ttgir_weight_prefetch=enabled,
            ttgir_weight_ring=ring,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((8, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_megakernel_issues_weight_copies_before_wait(self) -> None:
        # The hoist (the ring's fallback, and the pass with the ring setting
        # off): the weight copies move above the readiness spin.
        args = self._inputs()
        outputs = {}
        for enabled in (True, False):
            kernel = self._kernel(enabled, ring=False)
            bound = kernel.bind(args)
            code = bound.to_triton_code(_CONFIG)
            self.assertEqual("_ttgir_weight_prefetch=True" in code, enabled)
            self.assertNotIn("_ttgir_weight_ring", code)
            outputs[enabled] = kernel(*args)
            (ttgir,) = _compiled_ttgir(bound)
            lines = ttgir.splitlines()
            poll = next(i for i, line in enumerate(lines) if "ld.acquire" in line)
            early = [
                line for line in lines[:poll] if "async_copy_global_to_local" in line
            ]
            self.assertEqual(len(early), 3 if enabled else 0)
        x, gamma, w = (a.float() for a in args)
        y = (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * gamma).bfloat16()
        torch.testing.assert_close(
            outputs[True].float(), w @ y.float().T, rtol=2e-2, atol=2e-2
        )
        self.assertTrue(torch.equal(outputs[True], outputs[False]))

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((10, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_megakernel_streams_weight_copies_through_the_ring(self) -> None:
        # Default settings: the pass builds the warp-specialized ring, whose
        # weight producer issues the weight copies without waiting on any
        # readiness counter; the compute warps wait per stage on mbarriers.
        args = self._inputs()
        outputs = {}
        for prefetch_on, ring in ((True, True), (True, False), (False, True)):
            kernel = self._kernel(prefetch_on, ring=ring)
            bound = kernel.bind(args)
            code = bound.to_triton_code(_CONFIG)
            self.assertEqual("_ttgir_weight_prefetch=True" in code, prefetch_on)
            self.assertEqual("_ttgir_weight_ring=True" in code, prefetch_on and ring)
            outputs[prefetch_on, ring] = kernel(*args)
            (ttgir,) = _compiled_ttgir(bound)
            self.assertEqual("ttg.warp_specialize" in ttgir, prefetch_on and ring)
            if not (prefetch_on and ring):
                continue
            default, producers = _ring_parts(ttgir)
            weights = producers[0]
            self.assertIn("async_copy_global_to_local", weights)
            self.assertIn("async_copy_mbarrier_arrive", weights)
            self.assertNotIn("ld.acquire", weights)
            self.assertNotIn("ld.relaxed", weights)
            self.assertIn("ttng.wait_barrier", default)
        x, gamma, w = (a.float() for a in args)
        y = (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * gamma).bfloat16()
        torch.testing.assert_close(
            outputs[True, True].float(), w @ y.float().T, rtol=2e-2, atol=2e-2
        )
        self.assertTrue(torch.equal(outputs[True, True], outputs[True, False]))
        self.assertTrue(torch.equal(outputs[True, True], outputs[False, True]))

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((8, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_launch_with_aliasing_arguments_writes_their_class(self) -> None:
        x, gamma, w = self._inputs()
        kernel = helion.kernel(
            _norm_gemv_out,
            config=_CONFIG,
            static_shapes=True,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        y = torch.empty_like(x)
        bound = kernel.bind((x, gamma, w, y))
        self.assertIn("_ttgir_weight_prefetch=True", bound.to_triton_code(_CONFIG))
        expected = kernel(x, gamma, w, y)
        # The weight and the written activation as views of one buffer.  The
        # kernel's cache key includes which tensor inputs share storage, so
        # the call compiles a binding that knows: the weight shares a storage
        # with a written argument, so the pass treats it as written and
        # leaves its copies after the wait.
        both = torch.empty(w.numel() + y.numel(), device=DEVICE, dtype=torch.bfloat16)
        w_view = both[: w.numel()].view_as(w)
        y_view = both[w.numel() :].view_as(y)
        w_view.copy_(w)
        self.assertTrue(torch.equal(kernel(x, gamma, w_view, y_view), expected))
        aliased = kernel.bind((x, gamma, w_view, y_view))
        self.assertIsNot(aliased, bound)
        code = aliased.to_triton_code(_CONFIG)
        self.assertIn("_ttgir_weight_prefetch=True", code)
        self.assertIn("_tensor_storage_aliases=((", code)
        module = sys.modules[aliased._run.__module__]  # pyrefly: ignore[missing-attribute]
        (ttgir,) = [
            compiled.asm["ttgir"]
            for caches in module._helion__norm_gemv_out.device_caches.values()
            for compiled in caches[0].values()
        ]
        body = ttgir[ttgir.index("tt.func public") :]
        self.assertNotIn("async_copy_global_to_local", body[: body.index("ld.acquire")])
        # Calling the binding compiled for distinct storages re-dispatches to
        # that binding too; its compiled code alone (as exported, with no
        # compiler behind it) refuses the aliasing launch.
        self.assertTrue(torch.equal(bound(x, gamma, w_view, y_view), expected))
        run = bound._run
        assert run is not None
        with self.assertRaises(helion.exc.CrossLoopAliasedLaunch):
            run(x, gamma, w_view, y_view)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((8, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_static_root_waits_inside_its_task_for_the_pass(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(6, 512, device=DEVICE, dtype=torch.bfloat16)
        gamma = (1 + 0.1 * torch.randn(512, device=DEVICE)).bfloat16()
        w = (0.05 * torch.randn(512, 512, device=DEVICE)).bfloat16()
        y = torch.empty_like(x)
        config = helion.Config(
            block_sizes=[],
            cross_loop_pipeline="static",
            num_warps=4,
            num_stages=2,
            pid_type="persistent_blocked",
            indexing="pointer",
            range_num_stages=[0, 0, 4],
        )
        outputs = {}
        ring_on = torch.cuda.get_device_capability() >= (10, 0)
        for enabled, ring in ((True, False), (False, False), (True, True)):
            if ring and not ring_on:
                continue
            kernel = helion.kernel(
                _norm_gemv_masked,
                config=config,
                static_shapes=True,
                ttgir_weight_prefetch=enabled,
                ttgir_weight_ring=ring,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )
            bound = kernel.bind((x, gamma, w, y))
            lines = bound.to_triton_code(config).splitlines()
            wait = next(
                i for i, line in enumerate(lines) if "root_barrier_wait = " in line
            )
            loop = max(
                i for i, line in enumerate(lines[:wait]) if "for virtual_pid in" in line
            )
            # The single-trip GEMV root waits inside its task, right in front of
            # its K loop, not before the task loop.
            self.assertGreater(
                len(lines[wait]) - len(lines[wait].lstrip()),
                len(lines[loop]) - len(lines[loop].lstrip()),
            )
            outputs[enabled, ring] = kernel(x, gamma, w, y)
            module = sys.modules[bound._run.__module__]  # pyrefly: ignore[missing-attribute]
            (ttgir,) = [
                compiled.asm["ttgir"]
                for caches in module._helion__norm_gemv_masked.device_caches.values()
                for compiled in caches[0].values()
            ]
            if ring:
                # The ring: the weight producer streams the GEMV's weight
                # tiles without waiting; the wait stays in the compute warps'
                # task, in front of the K loop.
                default, producers = _ring_parts(ttgir)
                self.assertIn("async_copy_global_to_local", producers[0])
                self.assertNotIn("ld.acquire", producers[0])
                self.assertIn("ld.acquire", default)
                continue
            poll = ttgir.index("ld.acquire", ttgir.index("tt.func public"))
            early = ttgir[:poll].count("async_copy_global_to_local")
            self.assertEqual(early > 0, enabled)
        xf = x.float()
        ref = (
            xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * gamma.float()
        ).bfloat16()
        torch.testing.assert_close(
            outputs[True, False][:, :6].float(),
            w.float() @ ref.float().T,
            rtol=2e-2,
            atol=2e-2,
        )
        for out in outputs.values():
            self.assertTrue(torch.equal(out, outputs[False, False]))

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((10, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_static_tcgen05_megakernel_runs_two_programs_per_sm(self) -> None:
        # The occupancy API reports one block per SM for every kernel that uses
        # tensor memory; the residency check must count registers, shared memory
        # and TMEM columns instead, or a two-per-SM persistent grid is rejected.
        torch.manual_seed(0)
        x = torch.randn(16, 512, device=DEVICE, dtype=torch.bfloat16)
        gamma = (1 + 0.1 * torch.randn(512, device=DEVICE)).bfloat16()
        w = (0.05 * torch.randn(4096, 512, device=DEVICE)).bfloat16()
        config = helion.Config(
            block_sizes=[],
            cross_loop_pipeline="static",
            num_warps=4,
            num_stages=2,
            pid_type="persistent_blocked",
            num_sm_multiplier=2,
            indexing="pointer",
        )
        kernel = helion.kernel(
            _norm_gemv_wide,
            config=config,
            static_shapes=True,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        bound = kernel.bind((x, gamma, w))
        code = bound.to_triton_code(config)
        self.assertIn("_minimum_resident_programs=", code)
        out = kernel(x, gamma, w)
        module = sys.modules[bound._run.__module__]  # pyrefly: ignore[missing-attribute]
        (compiled,) = [
            compiled
            for caches in module._helion__norm_gemv_wide.device_caches.values()
            for compiled in caches[0].values()
        ]
        self.assertGreater(compiled.metadata.tmem_size, 0)
        xf = x.float()
        ref = (
            xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * gamma.float()
        ).bfloat16()
        torch.testing.assert_close(
            out.float(), w.float() @ ref.float().T, rtol=2e-2, atol=2e-2
        )

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((8, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_precompile_compiles_with_the_pass_under_the_launch_key(self) -> None:
        from helion.runtime.precompile_shim import make_precompiler

        args = self._inputs()
        ring_on = torch.cuda.get_device_capability() >= (10, 0)
        for ring in (True, False) if ring_on else (False,):
            kernel = self._kernel(True, ring=ring)
            bound = kernel.bind(args)
            bound.set_config(_CONFIG)
            extracted: dict[str, object] = {}

            def extract(
                triton_kernel: object,
                grid: object,
                *a: object,
                _sink: dict[str, object] = extracted,
                **kw: object,
            ) -> None:
                _sink.update(kernel=triton_kernel, args=a, kwargs=kw)

            bound._run(*args, _launcher=extract)  # pyrefly: ignore[missing-attribute]
            self.assertTrue(extracted["kwargs"]["_ttgir_weight_prefetch"])
            self.assertEqual(bool(extracted["kwargs"].get("_ttgir_weight_ring")), ring)
            jit = extracted["kernel"]
            compile_fn = make_precompiler(jit, _CONFIG, bound)(
                *extracted["args"], **extracted["kwargs"]
            )
            self.assertTrue(compile_fn(False))
            ((key, compiled),) = (
                item
                for caches in jit.device_caches.values()
                for item in caches[0].items()
            )
            ttgir = compiled.asm["ttgir"]
            if ring:
                # Precompile ran the ring: its weight producer issues the
                # copies, without waiting.
                _, producers = _ring_parts(ttgir)
                self.assertIn("async_copy_global_to_local", producers[0])
                self.assertNotIn("ld.acquire", producers[0])
            else:
                poll = ttgir.index("ld.acquire")
                self.assertIn("async_copy_global_to_local", ttgir[:poll])
            kernel(*args)
            keys = [k for caches in jit.device_caches.values() for k in caches[0]]
            pipeline = re.compile(r"\(\"custom_pipeline\", \w+\)")
            # Both keys name the same pass; the rest of the precompile key may
            # differ.
            self.assertEqual(
                {pipeline.search(k).group(0) for k in keys},
                {pipeline.search(key).group(0)},
            )

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((8, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_precompile_keys_the_storage_classes_like_the_launch(self) -> None:
        # The weight and the written activation share a storage: precompile
        # must compile with the same storage classes (and so the same key)
        # as the launch.
        from helion.runtime.precompile_shim import make_precompiler

        x, gamma, w = self._inputs()
        y = torch.empty_like(x)
        both = torch.empty(w.numel() + y.numel(), device=DEVICE, dtype=torch.bfloat16)
        w_view = both[: w.numel()].view_as(w)
        w_view.copy_(w)
        args = (x, gamma, w_view, both[w.numel() :].view_as(y))
        kernel = helion.kernel(
            _norm_gemv_out,
            config=_CONFIG,
            static_shapes=True,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        bound = kernel.bind(args)
        bound.set_config(_CONFIG)
        extracted = {}

        def extract(
            triton_kernel: object, grid: object, *a: object, **kw: object
        ) -> None:
            extracted.update(kernel=triton_kernel, args=a, kwargs=kw)

        bound._run(*args, _launcher=extract)  # pyrefly: ignore[missing-attribute]
        self.assertTrue(extracted["kwargs"]["_ttgir_weight_prefetch"])
        self.assertTrue(extracted["kwargs"]["_tensor_storage_aliases"])
        jit = extracted["kernel"]
        compile_fn = make_precompiler(jit, _CONFIG, bound)(
            *extracted["args"], **extracted["kwargs"]
        )
        self.assertTrue(compile_fn(False))
        ((key, compiled),) = (
            item for caches in jit.device_caches.values() for item in caches[0].items()
        )
        body = compiled.asm["ttgir"][compiled.asm["ttgir"].index("tt.func public") :]
        self.assertNotIn("async_copy_global_to_local", body[: body.index("ld.acquire")])
        kernel(*args)
        keys = [k for caches in jit.device_caches.values() for k in caches[0]]
        pipeline = re.compile(r"\(\"custom_pipeline\", \w+\)")
        self.assertEqual(
            {pipeline.search(k).group(0) for k in keys}, {pipeline.search(key).group(0)}
        )

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_compiled_megakernels_use_only_allowlisted_assembly(self) -> None:
        # Megakernels covering Helion's assembly paths: one-counter and
        # several-counter readiness polls, TMA proxy fences, PDL entry wait,
        # exit trigger (with its release fence and acquire epoch read).
        x, gamma, w = self._inputs()
        pointer = _CONFIG
        tma = _TMA_CONFIG
        two = _TWO_PRODUCER_CONFIG
        kinds: set[str] = set()
        for fn, config in (
            (_norm_gemv, pointer),
            (_norm_gemv, tma),
            (_norm_gemv_pdl, pointer),
            (_norm_gemv_exit_trigger, pointer),
            (_two_norm_gemv, two),
        ):
            with self.subTest(fn.__name__, indexing=config.config.get("indexing")):
                kernel = helion.kernel(
                    fn,
                    config=config,
                    static_shapes=True,
                    ignore_warnings=[helion.exc.TensorOperationInWrapper],
                )
                bound = kernel.bind((x, gamma, w))
                self.assertIn(
                    "_ttgir_weight_prefetch=True", bound.to_triton_code(config)
                )
                kernel(x, gamma, w)
                module = sys.modules[bound._run.__module__]  # pyrefly: ignore[missing-attribute]
                jit = getattr(module, f"_helion_{fn.__name__}")
                for caches in jit.device_caches.values():
                    for compiled in caches[0].values():
                        for line in compiled.asm["ttgir"].splitlines():
                            if "tt.elementwise_inline_asm" not in line:
                                continue
                            entry = prefetch._asm_entry(line.strip())
                            assert entry is not None, (
                                f"Helion emitted inline assembly off the allowlist: {line.strip()[:300]}"
                            )
                            kinds.add(entry.kind)
        self.assertEqual(kinds, {"poll", "sync", "pdl_wait", "quiet"})


@onlyBackends(["triton"])
class TestTtgirWeightRingSetting(TestCase):
    """The ring is a Helion setting (``ttgir_weight_ring``, default on): it
    reaches the pass through the launch, never through the process
    environment at compile or launch time."""

    def test_setting_defaults_on_and_reads_its_environment_variable(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("HELION_TTGIR_WEIGHT_RING", None)
            self.assertTrue(helion.Settings().ttgir_weight_ring)
            os.environ["HELION_TTGIR_WEIGHT_RING"] = "0"
            self.assertFalse(helion.Settings().ttgir_weight_ring)
            os.environ["HELION_TTGIR_WEIGHT_RING"] = "1"
            self.assertTrue(helion.Settings().ttgir_weight_ring)
            self.assertFalse(helion.Settings(ttgir_weight_ring=False).ttgir_weight_ring)
        # Documented like every setting (help(Settings)).
        self.assertIn(
            "HELION_TTGIR_WEIGHT_RING", helion.Settings.__slots__["ttgir_weight_ring"]
        )

    def test_launcher_passes_the_launch_flag_to_the_pass(self) -> None:
        from triton import knobs

        from helion.runtime.triton.launcher import default_launcher

        seen: list[object] = []

        class FakeJit:
            arg_names: tuple[str, ...] = ()

            def run(self, *args: object, **kwargs: object) -> None:
                hook = knobs.runtime.add_stages_inspection_hook
                seen.append(
                    (
                        prefetch._current_facts(),
                        None if hook is None else cast("tuple[str, str]", hook())[0],
                    )
                )

        saved = knobs.runtime.add_stages_inspection_hook
        knobs.runtime.add_stages_inspection_hook = None
        try:
            for env in ("0", "1"):
                with mock.patch.dict(os.environ, {"HELION_TTGIR_WEIGHT_RING": env}):
                    for pass_on, ring in ((True, True), (True, False), (False, True)):
                        seen.clear()
                        default_launcher(
                            FakeJit(),
                            (1,),
                            num_warps=4,
                            num_stages=1,
                            _ttgir_weight_prefetch=pass_on,
                            _ttgir_weight_ring=ring,
                        )
                        ((facts, key),) = seen
                        if not pass_on:  # no pass, so no ring either
                            self.assertIsNone(facts)
                            self.assertIsNone(key)
                            continue
                        assert isinstance(facts, prefetch._Facts)
                        self.assertEqual(facts.ring, ring)
                        assert isinstance(key, str)
                        self.assertTrue(key.endswith("-ring1" if ring else "-ring0"))
                    # Generated code from before the setting has no ring flag:
                    # the hoist alone.
                    seen.clear()
                    default_launcher(
                        FakeJit(),
                        (1,),
                        num_warps=4,
                        num_stages=1,
                        _ttgir_weight_prefetch=True,
                    )
                    self.assertFalse(cast("prefetch._Facts", seen[0][0]).ring)
        finally:
            knobs.runtime.add_stages_inspection_hook = saved

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((8, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_generated_launch_carries_the_setting(self) -> None:
        args = TestTtgirWeightPrefetchCompile._inputs(self)  # pyrefly: ignore[bad-argument-type]
        for prefetch_on, ring, flags in (
            (True, True, ("_ttgir_weight_prefetch=True", "_ttgir_weight_ring=True")),
            (True, False, ("_ttgir_weight_prefetch=True",)),
            (False, True, ()),
            (False, False, ()),
        ):
            kernel = helion.kernel(
                _norm_gemv,
                config=_CONFIG,
                static_shapes=True,
                ttgir_weight_prefetch=prefetch_on,
                ttgir_weight_ring=ring,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )
            code = kernel.bind(args).to_triton_code(_CONFIG)
            present = tuple(
                f
                for f in ("_ttgir_weight_prefetch=True", "_ttgir_weight_ring=True")
                if f in code
            )
            self.assertEqual(present, flags)
        # The environment variable is read when the kernel's settings are
        # made, not at launch.
        with mock.patch.dict(os.environ, {"HELION_TTGIR_WEIGHT_RING": "0"}):
            off = helion.kernel(
                _norm_gemv,
                config=_CONFIG,
                static_shapes=True,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )
        self.assertNotIn("_ttgir_weight_ring", off.bind(args).to_triton_code(_CONFIG))

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((10, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_kernels_with_either_setting_share_a_process(self) -> None:
        # Triton keys every launch with the stages hook again; the key comes
        # from each kernel's own launch flag, so flipping the environment
        # between launches neither recompiles a kernel nor swaps its binary.
        args = TestTtgirWeightPrefetchCompile._inputs(self)  # pyrefly: ignore[bad-argument-type]
        kernels = {
            ring: helion.kernel(
                _norm_gemv,
                config=_CONFIG,
                static_shapes=True,
                ttgir_weight_ring=ring,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )
            for ring in (True, False)
        }
        outputs: dict[bool, list[torch.Tensor]] = {True: [], False: []}
        for env in ("0", "1", "0", None):
            for ring, kernel in kernels.items():
                with mock.patch.dict(os.environ):
                    if env is None:
                        os.environ.pop("HELION_TTGIR_WEIGHT_RING", None)
                    else:
                        os.environ["HELION_TTGIR_WEIGHT_RING"] = env
                    outputs[ring].append(kernel(*args))
        for ring, kernel in kernels.items():
            bound = kernel.bind(args)
            (ttgir,) = _compiled_ttgir(bound)
            self.assertEqual("ttg.warp_specialize" in ttgir, ring)
        reference = outputs[False][0]
        for out in outputs[True] + outputs[False]:
            self.assertTrue(torch.equal(out, reference))


@onlyBackends(["triton"])
class TestTtgirWeightRingRace(TestCase):
    """The ring on GPU against a late producer of the activation it streams,
    with the private negative control that drops the activation producer's
    readiness spin (HELION_TTGIR_RING_NEG=nospin): that one must race."""

    def _run(
        self, ring: bool, neg: str | None, launches: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], str]:
        torch.manual_seed(0)
        x = torch.randn(16, 512, device=DEVICE, dtype=torch.bfloat16)
        gamma = (1 + 0.1 * torch.randn(512, device=DEVICE)).bfloat16()
        w = (0.05 * torch.randn(256, 512, device=DEVICE)).bfloat16()
        y = torch.empty_like(x)
        outs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        with mock.patch.dict(os.environ):
            # the hook is read at compile time and in every launch's key
            os.environ.pop("HELION_TTGIR_RING_NEG", None)
            if neg is not None:
                os.environ["HELION_TTGIR_RING_NEG"] = neg
            kernel = helion.kernel(
                _late_act_gemv,
                config=_CONFIG,
                static_shapes=True,
                ttgir_weight_ring=ring,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )
            bound = kernel.bind((x, gamma, w, y))
            bound.set_config(_CONFIG)
            module = sys.modules[bound._run.__module__]  # pyrefly: ignore[missing-attribute]

            def compiled() -> dict[object, str]:
                # Kernels with equal generated code share one JIT function;
                # the hook keys its binaries apart (ring0 / ring1 / ring1[neg]).
                return {
                    key: c.asm["ttgir"]
                    for caches in module._helion__late_act_gemv.device_caches.values()
                    for key, c in caches[0].items()
                }

            before = compiled()
            for _ in range(launches):
                y.fill_(float("nan"))  # poisoned until the late root writes it
                outs.append(kernel(x, gamma, w, y).clone())
                ys.append(y.clone())
            torch.cuda.synchronize()
        # one new binary, compiled once for all the launches
        (ttgir,) = [t for key, t in compiled().items() if key not in before]
        return outs, ys, ttgir

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((10, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_activation_producer_waits_for_a_late_writer(self) -> None:
        (reference,), (y_reference,), hoisted = self._run(False, None, 1)
        self.assertNotIn("ttg.warp_specialize", hoisted)
        self.assertFalse(reference.isnan().any())
        outs, ys, ttgir = self._run(True, None, 10)
        # The ring streams the weight and the activation: the activation
        # producer polls the readiness counter (its cloned spin), the weight
        # producer polls nothing.
        _, (weights, activation) = _ring_parts(ttgir)
        for producer in (weights, activation):
            self.assertIn("async_copy_mbarrier_arrive", producer)
        self.assertNotIn("ld.acquire", weights)
        self.assertIn("ld.acquire", activation)
        for out, y in zip(outs, ys, strict=True):
            self.assertTrue(torch.equal(y, y_reference))
            self.assertTrue(torch.equal(out, reference))
        # Negative control: without its spin the activation producer copies
        # the poisoned rows before the late root writes them.
        outs, _, ttgir = self._run(True, "nospin", 5)
        _, (weights, activation) = _ring_parts(ttgir)
        self.assertNotIn("ld.acquire", activation)
        self.assertIn("async_copy_mbarrier_arrive", activation)
        wrong = sum(not torch.equal(out, reference) for out in outs)
        self.assertGreater(wrong, 0)


@onlyBackends(["triton"])
class TestTtgirWeightRingDefaultPath(TestCase):
    """The ring's fallback chain on the default path (helion.kernel with
    default settings, through the launcher)."""

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((10, 0))
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_default_path_falls_back_when_the_ring_fails_to_lower(self) -> None:
        from triton import knobs

        args = TestTtgirWeightPrefetchCompile._inputs(self)  # pyrefly: ignore[bad-argument-type]
        off = helion.kernel(
            _norm_gemv,
            config=_CONFIG,
            static_shapes=True,
            ttgir_weight_ring=False,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        reference = off(*args)
        failed: list[str] = []

        def refuse_warp_specialized(*hook_args: object) -> object:
            # A previously installed hook (chained by the pass) whose
            # make_llir rejects every warp-specialized module.
            if not hook_args:
                return "ringdef-test-refuse-ws", "ringdef-test-refuse-ws"
            stages = cast("dict[str, object]", hook_args[1])
            inner = cast("Callable[[object, object], object]", stages["llir"])

            def llir(src: object, md: object) -> object:
                if "ttg.warp_specialize" in src.str():  # pyrefly: ignore[missing-attribute]
                    failed.append("ring")
                    raise RuntimeError("injected make_llir failure")
                return inner(src, md)

            stages["llir"] = llir
            return None

        with mock.patch.dict(os.environ):
            # the default settings, whatever the suite's environment says
            os.environ.pop("HELION_TTGIR_WEIGHT_RING", None)
            kernel = helion.kernel(
                _norm_gemv,
                config=_CONFIG,
                static_shapes=True,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )
        bound = kernel.bind(args)
        self.assertIn("_ttgir_weight_ring=True", bound.to_triton_code(_CONFIG))
        bound.set_config(_CONFIG)
        module = sys.modules[bound._run.__module__]  # pyrefly: ignore[missing-attribute]

        def compiled_by_key() -> dict[object, object]:
            return {
                k: c
                for caches in module._helion__norm_gemv.device_caches.values()
                for k, c in caches[0].items()
            }

        before = compiled_by_key()
        saved = knobs.runtime.add_stages_inspection_hook
        knobs.runtime.add_stages_inspection_hook = refuse_warp_specialized  # pyrefly: ignore[bad-assignment]
        try:
            out = kernel(*args)
        finally:
            knobs.runtime.add_stages_inspection_hook = saved
        # every ring form failed to lower; the binary is the hoist
        self.assertTrue(failed)
        (compiled,) = [c for k, c in compiled_by_key().items() if k not in before]
        self.assertEqual(compiled.metadata.num_warps, 4)
        self.assertNotIn("cp.async.mbarrier.arrive", compiled.asm["ptx"])
        self.assertTrue(torch.equal(out, reference))


def _asm_calls(source: str) -> list[tuple[str, str, bool, int, int]]:
    """``(template, constraints, pure, operands, pack)`` of every
    ``tl.inline_asm_elementwise`` call in Triton source text."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "inline_asm_elementwise"
        ):
            continue
        names = ("asm", "constraints", "args", "dtype", "is_pure", "pack")
        bound = dict(zip(names, node.args, strict=False))
        bound.update({k.arg: k.value for k in node.keywords if k.arg})
        args = bound["args"]
        assert isinstance(args, ast.List)
        found.append(
            (
                ast.literal_eval(bound["asm"]),
                ast.literal_eval(bound["constraints"]),
                ast.literal_eval(bound["is_pure"]),
                len(args.elts),
                ast.literal_eval(bound["pack"]),
            )
        )
    return found


# What a formatted value in the code generator's inline assembly f-strings can
# be (None: the literal is a helper's, checked through the helper's callers).
_FORMATTED: dict[str, list[str] | None] = {
    "fence": ["", "fence.proxy.async.global; "],
    "asm": None,
}


def _rendered(node: ast.Constant | ast.JoinedStr) -> list[str | None]:
    """The texts a string literal (an f-string: each formatted value replaced
    by its alternatives, or by a name for an argument expression) can be."""
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else []
    variants: list[str] = [""]
    for part in node.values:
        if isinstance(part, ast.Constant):
            alternatives = [str(part.value)]
        else:
            assert isinstance(part, ast.FormattedValue)
            value = part.value
            if isinstance(value, ast.IfExp) and all(
                isinstance(b, ast.Constant) for b in (value.body, value.orelse)
            ):
                alternatives = [value.body.value, value.orelse.value]  # pyrefly: ignore[missing-attribute]
            elif isinstance(value, ast.Name) and value.id in _FORMATTED:
                known = _FORMATTED[value.id]
                if known is None:
                    return [None]
                alternatives = known
            else:
                alternatives = ["x"]
        variants = [v + a for v in variants for a in alternatives]
    return list(variants)


def _ttgir_asm(
    template: str, constraints: str, pure: bool, operands: int, pack: int
) -> str:
    """An inline assembly op as Triton prints it in TTGIR."""
    names = ", ".join(f"%a{k}" for k in range(operands))
    tail = (
        f" {names} : {', '.join(['i32'] * operands)} -> i32" if operands else "  -> i32"
    )
    return (
        f'%r = tt.elementwise_inline_asm "{_mlir_escape(template)}" '
        f'{{constraints = "{_mlir_escape(constraints)}", packed_element = {pack} : i32, '
        f"pure = {str(pure).lower()}}}{tail}"
    )


# Every site in Helion that emits inline assembly into Triton kernels, by file:
# how many ``inline_asm_elementwise`` calls (or Triton gdc helper calls) its
# source spells.  A new site fails this census: put its template(s) on
# ttgir_weight_prefetch._ALLOWLIST (with their source) if they write nothing,
# or leave them off (the pass then treats them as writing every argument), and
# update the count.  Off the allowlist on purpose: distributed_ops.py and
# dist_utils.py (peer memory, system scope), inline_asm_ops.py (user assembly).
_ASM_SITES = {
    "_compiler/cross_loop_codegen.py": 9,
    "_compiler/device_function.py": 4,
    "_compiler/triton/distributed_ops.py": 2,
    "_compiler/triton/inline_asm_ops.py": 2,
    "_compiler/triton/quantized_ops.py": 3,
    "runtime/triton/dist_utils.py": 10,
    "runtime/triton/helpers.py": 2,
}


class TestAllowlistCoversCodegen(TestCase):
    """The allowlist must keep up with the assembly Helion emits: a template
    missing from it silently costs hoists (never soundness)."""

    def _assert_allowlisted(self, calls: list[tuple[str, str, bool, int, int]]) -> None:
        self.assertTrue(calls)
        for call in calls:
            with self.subTest(template=call[0][:60]):
                self.assertIsNotNone(
                    prefetch._asm_entry(_ttgir_asm(*call)),
                    f"Helion emits inline assembly that is not on the allowlist: {call}",
                )

    def test_cross_loop_waits_and_fences_are_allowlisted(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        from helion._compiler import cross_loop_codegen
        from helion._compiler.device_function import DeviceFunction

        names = iter(range(1000))
        for proxy in (False, True):
            stub = SimpleNamespace(
                new_var=lambda prefix, dce=False: f"{prefix}_{next(names)}",
                uses_async_proxy_global=proxy,
                config={"num_warps": 1},
                _async_store_drained=False,
                _check_async_store_drain=lambda: None,
            )
            stub._thread_asm = lambda prefix, asm, stub=stub: (
                DeviceFunction._thread_asm(
                    stub,  # pyrefly: ignore[bad-argument-type]
                    prefix,
                    asm,
                )
            )
            stub.async_load_fence = lambda stub=stub: DeviceFunction.async_load_fence(
                stub  # pyrefly: ignore[bad-argument-type]
            )
            statements = [
                *cross_loop_codegen._wait_for_counter(
                    device_function=stub,  # pyrefly: ignore[bad-argument-type]
                    counter="state + 0",
                    target="1",
                    prefix="wait",
                ),
                cross_loop_codegen._publication_sync(stub),  # pyrefly: ignore[bad-argument-type]
                *DeviceFunction.async_store_drain(stub),  # pyrefly: ignore[bad-argument-type]
            ]
            for fence in (False, True):
                with mock.patch.object(
                    cross_loop_codegen, "_has_acquire_fence", return_value=fence
                ):
                    for dependencies in (
                        (("state + 0", "1"),),
                        (("state + 0", "1"), ("state + 1", "2")),
                    ):
                        statements += cross_loop_codegen._wait_for_dependencies(
                            device_function=stub,  # pyrefly: ignore[bad-argument-type]
                            dependencies=dependencies,
                            prefix="wait",
                        )
            calls = _asm_calls("\n".join(ast.unparse(s) for s in statements))
            self._assert_allowlisted(calls)
            if proxy:
                self.assertIn(
                    "cp.async.bulk.wait_group 0; fence.proxy.async.global; mov.u32 $0, $1;",
                    [c[0] for c in calls],
                )

    def test_assembly_spelled_in_helion_sources_is_allowlisted(self) -> None:
        # Every inline assembly call that Helion's sources spell as a string
        # literal (code generator) or as a call (runtime helpers), with each
        # formatted value of an f-string replaced by what it can be.
        root = Path(helion.__file__).parent
        found: dict[str, set[tuple[str, str, bool, int, int]]] = {}
        for name in (
            "_compiler/cross_loop_codegen.py",
            "_compiler/device_function.py",
            "runtime/triton/helpers.py",
        ):
            calls = found.setdefault(name, set())
            for node in ast.walk(ast.parse((root / name).read_text())):
                texts = []
                if isinstance(node, (ast.Constant, ast.JoinedStr)):
                    texts = _rendered(node)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "inline_asm_elementwise"
                ):
                    texts = [ast.unparse(node)]
                for text in texts:
                    if text is None or "tl.inline_asm_elementwise(" not in text:
                        continue
                    start = text.index("tl.inline_asm_elementwise(")
                    with contextlib.suppress(SyntaxError, AssertionError, ValueError):
                        calls.update(_asm_calls(text[start:]))
        # Off the allowlist on purpose: the peer wait polls system scope (and
        # makes its function ineligible).
        for name, calls in found.items():
            for call in sorted(calls):
                if "ld.acquire.sys" in call[0]:
                    continue
                with self.subTest(name, template=call[0][:60]):
                    self.assertIsNotNone(
                        prefetch._asm_entry(_ttgir_asm(*call)),
                        f"{name} emits inline assembly off the allowlist: {call}",
                    )
        templates = {c[0] for calls in found.values() for c in calls}
        for expected in (
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "fence.acquire.gpu; bar.warp.sync 0xffffffff; mov.u32 $0, $1;",
            prefetch._READY_WAIT + "fence.proxy.async.global; mov.u32 $0, 0; }",
            "redux.sync.min.s32 $0, $1, 0xffffffff;",
            "fence.acq_rel.gpu; // $0",
            "bar.sync 0;",
        ):
            self.assertIn(expected, templates)
        quantized = importlib.import_module("helion._compiler.triton.quantized_ops")
        for name, constraints, operands in (
            ("_FP4X2_TO_F32_ASM", "=f,=f,h", 1),
            ("_FP4X16_TO_F16_ASM", ",".join(["=h"] * 16 + ["l"]), 1),
            ("_BF16X16_TO_F16_ASM", ",".join(["=h"] * 16 + ["l"] * 4), 4),
        ):
            with self.subTest(name):
                self._assert_allowlisted(
                    [(getattr(quantized, name), constraints, True, operands, 1)]
                )

    def test_triton_helpers_helion_uses_are_allowlisted(self) -> None:
        import inspect

        from triton.language.extra.cuda import gdc
        from triton.language.extra.cuda import utils

        for fn, kind in (
            (gdc.gdc_wait, "pdl_wait"),
            (gdc.gdc_launch_dependents, "quiet"),
            (utils.globaltimer, "quiet"),
            (utils.smid, "quiet"),
        ):
            source = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
            source = source.replace(
                "core.inline_asm_elementwise", "tl.inline_asm_elementwise"
            )
            (call,) = _asm_calls(source)
            with self.subTest(fn.__name__):
                entry = prefetch._asm_entry(_ttgir_asm(*call))
                assert entry is not None, (
                    f"{fn.__name__} is not on the allowlist: {call}"
                )
                self.assertEqual(entry.kind, kind)

    def test_every_assembly_emission_site_is_known(self) -> None:
        root = Path(helion.__file__).parent
        pattern = re.compile(
            r"\btl\.inline_asm_elementwise\b|gdc_wait\(\)|gdc_launch_dependents\(\)"
        )
        census = {}
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if relative.startswith(("_compiler/cute/", "_compiler/flydsl/")) or (
                relative == "runtime/triton/ttgir_weight_prefetch.py"
            ):
                continue
            count = len(pattern.findall(path.read_text()))
            if count:
                census[relative] = count
        self.assertEqual(
            census,
            _ASM_SITES,
            "Helion's inline assembly emission sites changed: check the new "
            "templates against ttgir_weight_prefetch._ALLOWLIST, then update "
            "_ASM_SITES",
        )


if __name__ == "__main__":
    unittest.main()
