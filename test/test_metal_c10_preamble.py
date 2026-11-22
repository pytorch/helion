"""The c10 preamble (``c10_preamble``): Metal kernels carry the c10
declarations they use, read statically from torch's headers, in place of the
whole-header includes."""

from __future__ import annotations

import collections
from pathlib import Path
import re
import tempfile
from unittest import mock

import pytest
import torch

from helion._compiler.metal import c10_preamble
from helion._testing import TestCase
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_fragment import fragment_config
from helion._testing.metal_preamble import bfloat_mul_add
from helion._testing.metal_preamble import erf_of
from helion._testing.metal_preamble import row_sums

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

_HEADER = """
#include <metal_stdlib>
#define TWICE(x) ((x) * 2)
#define UNUSED_MACRO 1
namespace c10 {
namespace metal {
template <typename T>
struct box {
  T v;
};
template <>
struct box<float> {
  float v;
};
template <typename T>
inline T pick(T x) {
  return x;
}
inline long pick(long x) {
  return x + 1;
}
inline float helper(float x) {
  return TWICE(x);
}
inline float uses_helper(float x) {
  return helper(x);
}
template <typename T>
inline bool operator==(box<T> a, box<T> b) {
  return a.v == b.v;
}
static_assert(sizeof(box<float>) == 4, "one float");
inline float unused(float x) {
  return x;
}
namespace detail {
constant constexpr int table[2] = {1, 2};
}
} // namespace metal
} // namespace c10
#include <metal_stdlib>
using namespace c10::metal;
using namespace metal;
"""


def _expanded(*includes: str) -> str:
    headers = c10_preamble._headers(includes, torch.__version__)
    assert headers is not None
    return headers.text


class TestC10Preamble(TestCase):
    def test_guards_and_conditionals(self) -> None:
        """The headers expand once each, without the stable-ABI guards each
        is wrapped in and with ``__METAL__``'s conditionals resolved; one on
        a macro whose value isn't known (the Metal version) stays."""
        text = _expanded("c10/metal/utils.h", "c10/metal/special_math.h")
        for gone in ("TORCH_STABLE_ONLY", "#error", "#include <c10/", "__METAL__)"):
            self.assertNotIn(gone, text)
        self.assertEqual(
            _expanded("c10/metal/utils.h", "c10/metal/utils.h"),
            _expanded("c10/metal/utils.h") + "\n",
        )
        resolved = c10_preamble._resolve(
            "#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)\n"
            "cuda\n#elif defined(__METAL__)\nmetal\n#else\nhost\n#endif\n"
            "#ifndef __METAL__\nnot metal\n#elif X > 1\nmaybe\n#else\nelse\n#endif"
        )
        self.assertEqual(
            resolved.split("\n"),
            ["metal", "#if X > 1", "maybe", "#else", "else", "#endif"],
        )

    def test_closure(self) -> None:
        """A name's overloads and specializations are kept together, with
        what they name (a helper, a macro), the operators naming a kept
        type and the assertions about it; the rest goes, and the namespaces
        are rebuilt around what stays."""
        kept = c10_preamble.prune_source(_HEADER, {"pick", "uses_helper", "box"})
        for present in (
            "inline T pick(T x)",
            "inline long pick(long x)",
            "inline float helper(float x)",
            "#define TWICE(x)",
            "struct box<float>",
            "operator==",
            "static_assert",
            "#include <metal_stdlib>",
        ):
            self.assertIn(present, kept)
        for absent in ("unused", "UNUSED_MACRO", "table"):
            self.assertNotIn(absent, kept)
        self.assertEqual(kept.count("namespace metal {"), 1)
        self.assertEqual(kept.count("#include <metal_stdlib>"), 1)
        self.assertIn("using namespace c10::metal;", kept)
        # A using-directive of a namespace nothing kept declares goes;
        # Metal's own stays; one before what it names declares it first.
        self.assertEqual(
            c10_preamble.prune_source(_HEADER, set()).split("\n"),
            ["#include <metal_stdlib>", "using namespace metal;"],
        )
        # ``::metal`` is Metal's, from inside c10 too.
        rooted = (
            "namespace c10 {\nnamespace metal {\nint g();\n}\n"
            "using namespace ::metal;\nint f();\n}"
        )
        self.assertIn(
            "using namespace ::metal;", c10_preamble.prune_source(rooted, {"f"})
        )
        early = "using namespace c10::metal;\nnamespace c10 {\nnamespace metal {\nint f();\n}\n}"
        self.assertEqual(
            c10_preamble.prune_source(early, {"f"}).split("\n"),
            [
                "namespace c10 {",
                "namespace metal {",
                "}",
                "}",
                "using namespace c10::metal;",
                "namespace c10 {",
                "namespace metal {",
                "int f();",
                "}",
                "}",
            ],
        )
        self.assertEqual(
            c10_preamble.prune_source(_HEADER, {"table"}).split("\n"),
            [
                "#include <metal_stdlib>",
                "namespace c10 {",
                "namespace metal {",
                "namespace detail {",
                "constant constexpr int table[2] = {1, 2};",
                "}",
                "}",
                "}",
                "using namespace c10::metal;",
                "using namespace metal;",
            ],
        )
        # From torch's headers: every piece declaring a used name with
        # several.
        headers = c10_preamble._headers(
            ("c10/metal/utils.h", "c10/metal/special_math.h"), torch.__version__
        )
        assert headers is not None
        counts = collections.Counter(n for piece in headers.pieces for n in piece.names)
        name, count = counts.most_common(1)[0]
        self.assertGreater(count, 1)
        kept = c10_preamble._select(headers.pieces, {name}, headers.notices)
        for piece in headers.pieces:
            if name in piece.names:
                self.assertIn(piece.text.strip("\n"), kept)

    def test_hard_declarations(self) -> None:
        """A default argument that calls, a ``>=`` in template arguments, an
        overload in a conditional on the Metal version and names past
        attributes, specialization arguments or a decltype split as
        declarations; headers that don't split (an inline namespace, braces
        split across a conditional's branches) leave the includes whole."""
        text = """
namespace c10 {
template <typename T>
inline T scaled(T x, T s = T(1)) {
  return x * s;
}
template <typename T, ::metal::enable_if_t<(sizeof(T) >= 4), bool> = true>
inline T wide(T x) {
  return x;
}
inline float helper(float x) {
  return x;
}
inline float pick(float x) {
  return x;
}
#if __METAL_VERSION__ >= 310
inline int pick(int x) {
  return int(helper(float(x)));
}
#endif
inline float last(float x) {
  return x;
}
}
"""
        pieces = c10_preamble._pieces(text)
        self.assertEqual(
            [sorted(piece.names) for piece in pieces],
            [["scaled"], ["wide"], ["helper"], ["pick"], ["pick"], ["last"]],
        )
        # Overloads and specializations whose names take reading past an
        # attribute, a specialization's arguments or a decltype.
        for declaration in (
            "__attribute__((always_inline)) inline half foo(half x) { return x; }",
            "[[clang::always_inline]] inline half foo(half x) { return x; }",
            "template <>\ninline float foo<float>(float x) { return x; }",
            "template <typename T>\ninline decltype(T(0) + 1) foo(T x) { return x; }",
            "template <>\nconstexpr constant bool foo<bfloat> = true;",
        ):
            with self.subTest(declaration=declaration):
                self.assertEqual(c10_preamble._names(declaration), ("decl", {"foo"}))
        # Where the split could be wrong, it refuses: an inline namespace, a
        # macro's use (with or without ``;``), two declarations on a line, a
        # name read as a keyword.
        self.assertEqual(c10_preamble._names("enum { A, B };"), ("decl", {"A", "B"}))
        for unread in (
            "namespace c10 {\ninline namespace v1 {\nint f();\n}\n}",
            (
                "#define MAKE(T) inline T twice(T x) { return x + x; }\nMAKE(float)\n"
                "inline float g(float x) { return x; }"
            ),
            "#define MAKE(T) inline T twice(T x) { return x + x; }\nMAKE(float);",
            "inline int f() { return 1; } inline int g(int x) { return x; }",
            "inline int (float x);",
        ):
            with self.subTest(unread=unread), self.assertRaises(c10_preamble._Unparsed):
                c10_preamble._pieces(unread)
        # An operator in a conditional goes with the type it names.
        boxed = (
            "struct box {\n  int v;\n};\n#if __METAL_VERSION__ >= 310\n"
            "inline box operator+(box a, box b) {\n  return {a.v + b.v};\n}\n#endif"
        )
        self.assertIn("operator+", c10_preamble.prune_source(boxed, {"box"}))
        kept = c10_preamble.prune_source(text, {"pick"})
        for present in ("inline int pick(int x)", "inline float helper", "#endif"):
            self.assertIn(present, kept)
        self.assertNotIn("last", kept)
        msl = "#include <c10/metal/utils.h>\nkernel void k() {}"
        with tempfile.TemporaryDirectory() as root:
            header = Path(root) / "c10" / "metal" / "utils.h"
            header.parent.mkdir(parents=True)
            header.write_text(
                "#if X\ninline float f(float x) {\n#else\ninline float f() {\n#endif\n  return 0;\n}\n"
            )
            c10_preamble._headers.cache_clear()
            try:
                with mock.patch.object(c10_preamble, "include_dir", lambda: Path(root)):
                    self.assertEqual(c10_preamble.prune(msl), msl)
            finally:
                c10_preamble._headers.cache_clear()

    def test_notices(self) -> None:
        """A license or attribution comment stays with what follows it in
        its header, first in the preamble; a header whose declarations all
        go takes its notices with it."""
        with tempfile.TemporaryDirectory() as root:
            metal = Path(root) / "c10" / "metal"
            metal.mkdir(parents=True)
            (metal / "utils.h").write_text(
                "#pragma once\n#include <c10/metal/vendored.h>\n"
                "// Copyright (c) Someone Else\nnamespace c10 {\n"
                "inline float unused(float x) {\n  return x;\n}\n}\n"
            )
            (metal / "vendored.h").write_text(
                "#pragma once\nnamespace c10 {\ninline float early(float x) {\n"
                "  return x;\n}\n// Copyright (c) Someone\n// All rights reserved.\n"
                "inline float vendored(float x) {\n  return x;\n}\n}\n"
            )
            msl = (
                "#include <c10/metal/utils.h>\nfloat k(float x) { return c10::%s(x); }"
            )
            c10_preamble._headers.cache_clear()
            c10_preamble._pruned.cache_clear()
            try:
                with mock.patch.object(c10_preamble, "include_dir", lambda: Path(root)):
                    kept = c10_preamble.prune(msl % "vendored")
                    early = c10_preamble.prune(msl % "early")
            finally:
                c10_preamble._headers.cache_clear()
                c10_preamble._pruned.cache_clear()
        self.assertTrue(
            kept.startswith("// Copyright (c) Someone\n// All rights reserved.")
        )
        self.assertNotIn("Someone Else", kept)
        self.assertNotIn("Copyright", early)

    def test_pasted_names(self) -> None:
        """A macro that pastes a name (``simd_##NAME``) keeps the names it
        can form."""
        names = frozenset({"simd_sum", "simd_max", "sum", "other"})
        text = "#define R(NAME) return ::c10::metal::simd_##NAME(v);\nR(sum)"
        self.assertEqual(c10_preamble._pasted(text, names), {"simd_sum", "simd_max"})
        self.assertEqual(c10_preamble._pasted("#define P(A, B) A##B", names), names)

    def test_kernels(self) -> None:
        """Representative kernels carry no c10 include and under a tenth of
        the headers they included: elementwise bfloat math, a special
        function, a reduction and an affine matmul on fragments."""
        x = torch.randn(64, 64)
        for kernel, args, config in (
            (bfloat_mul_add, (x.bfloat16(), x.bfloat16()), None),
            (erf_of, (x,), None),
            (row_sums, (torch.randn(32, 128),), None),
            (affine_mm, affine_args(4, 32, 8, 64, 512), fragment_config([16, 32, 64])),
        ):
            with self.subTest(kernel=kernel.name):
                config = config or kernel.bind(args).config_spec.default_config()
                msl = render_msl(kernel, args, config)
                full = render_msl(kernel, args, config, whole_c10=True)
                self.assertNotIn("#include <c10/", msl)
                lines = re.findall(r"#include <c10/[^>]+>\n", full)
                headers = _expanded(*(line[10:-2] for line in lines))
                # The declarations kept, against the headers they come from.
                preamble = len(msl) - len(full) + len("".join(lines))
                self.assertLess(preamble, len(headers) / 10)
                if kernel is row_sums:
                    # c10's SIMD reductions, which the reduction preamble
                    # names by pasting (``simd_##NAME``).
                    c10 = msl[: msl.index("namespace helion_red")]
                    self.assertRegex(c10, r"\bsimd_sum\b")
