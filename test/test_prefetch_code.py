from __future__ import annotations

import struct
import unittest

import torch

from helion.runtime.settings import _get_backend

if _get_backend() not in ("triton", "tileir"):
    raise unittest.SkipTest("triton not available")

import triton
import triton.language as tl

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
import helion.language as hl
from helion.runtime.triton import prefetch_code


def _table_sizes(cubin: bytes) -> list[int]:
    """The text sizes in a cubin's code prefetch table (zero: unresolved)."""
    header = struct.unpack_from("<HHIQQQIHHHHHH", cubin, 16)
    shoff, shentsize, shnum = header[5], header[10], header[11]
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", cubin, shoff + i * shentsize)
        for i in range(shnum)
    ]
    symtab = next(section for section in sections if section[1] == 2)
    strtab = sections[symtab[6]][4]
    for offset in range(symtab[4], symtab[4] + symtab[5], 24):
        name, _, _, index, value, size = struct.unpack_from("<IBBHQQ", cubin, offset)
        if cubin[strtab + name :].split(b"\0", 1)[0] == b"helion_prefetch_code_tab":
            words = struct.unpack_from(
                f"<{size // 8}Q", cubin, sections[index][4] + value
            )
            # (text address, text bytes) per function, then the sink word.
            return list(words[1::2])
    raise AssertionError("no code prefetch table")


@triton.jit
def _prefetched_copy(x, out):
    prefetch_code.prefetch_code(tl.program_id(0) == 0)
    offsets = tl.arange(0, 128)
    tl.store(out + offsets, tl.load(x + offsets))


@triton.jit
def _plain_copy(x, out):
    offsets = tl.arange(0, 128)
    tl.store(out + offsets, tl.load(x + offsets))


@helion.kernel(static_shapes=True, autotune_effort="none")
def two_roots(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    hl.prefetch_code()
    for tile_a in hl.tile(x.size(0), block_size=1):
        y[tile_a, :] = x[tile_a, :] + 1
    for tile_b in hl.tile(x.size(0), block_size=1):
        out[tile_b, :] = y[tile_b, :] * 2
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def two_roots_unmarked(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_a in hl.tile(x.size(0), block_size=1):
        y[tile_a, :] = x[tile_a, :] + 1
    for tile_b in hl.tile(x.size(0), block_size=1):
        out[tile_b, :] = y[tile_b, :] * 2
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def two_roots_prefetch_between(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_a in hl.tile(x.size(0), block_size=1):
        y[tile_a, :] = x[tile_a, :] + 1
    # The second root's code, prefetched when the kernel reaches that root (a
    # root of its own: column tiles, not a continuation of the row tiles).
    hl.prefetch_code()
    for tile_b in hl.tile(x.size(1), block_size=16):
        out[:, tile_b] = y[:, tile_b] * 2
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def independent_roots_prefetch_between(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_a in hl.tile(x.size(0), block_size=1):
        y[tile_a, :] = x[tile_a, :] + 1
    hl.prefetch_code()
    for tile_b in hl.tile(x.size(1), block_size=16):
        out[:, tile_b] = x[:, tile_b] * 2
    return y + out


@helion.kernel(static_shapes=True, autotune_effort="none")
def prefetch_code_after_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    hl.prefetch_code()
    return out


@onlyBackends(["triton"])
class TestPrefetchCode(TestCase):
    @skipIfCudaCapabilityLessThan((9, 0), reason="code prefetch marks sm_90+ kernels")
    @skipIfNotCUDA()
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_marked_kernel_prefetches_its_code(self) -> None:
        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        out = torch.empty_like(x)
        compiled = _prefetched_copy[(1,)](x, out)
        torch.testing.assert_close(out, x)
        ptx = compiled.asm["ptx"]
        self.assertIn(".global .align 8 .u64 helion_prefetch_code_tab[3]", ptx)
        # The keep-alive store targets the sink word, past the table entries.
        self.assertIn("st.global.u32 [helion_prefetch_code_tab+16]", ptx)
        # The cubin stage resolved the entry's text size.
        self.assertEqual(len(_table_sizes(compiled.asm["cubin"])), 1)
        self.assertTrue(all(_table_sizes(compiled.asm["cubin"])))
        self.assertIn("ld.global.cg.L2::cache_hint.u32", ptx)
        # The readers (the call's: the function selects none) split the lines:
        # a reader's lanes start at its index's share and stride by every
        # reader's lanes.
        prefetch = ptx[ptx.index(".func helion_prefetch_code_fn_1()\n{") :]
        self.assertIn("div.u32 %pcnw, %pcnw, 16;", prefetch)
        self.assertIn("mad.lo.u32 %pcw, %pcc, 32, %pcw;", prefetch)
        # The prefetch is a function the call site calls: inline, its loop costs
        # ptxas its uniform-register analysis across the whole kernel.
        self.assertIn("@%pcr call.uni helion_prefetch_code_fn_1;", ptx)
        # Unmarked kernels are unchanged.
        plain = _plain_copy[(1,)](x, out)
        self.assertNotIn("helion_prefetch_code_tab", plain.asm["ptx"])

    @skipIfCudaCapabilityLessThan((9, 0), reason="code prefetch marks sm_90+ kernels")
    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_source_marks_megakernel_for_prefetch_code(self) -> None:
        x = torch.randn((8, 64), device=DEVICE, dtype=torch.float32)
        for pipeline in ("static", "dynamic"):
            with self.subTest(pipeline=pipeline):
                code, out = code_and_output(
                    two_roots,
                    (x,),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                )
                torch.testing.assert_close(out, (x + 1) * 2)
                self.assertIn("helion_prefetch_code.prefetch_code(", code)
                # Without hl.prefetch_code() the megakernel stays unmarked.
                code, out = code_and_output(
                    two_roots_unmarked,
                    (x,),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                )
                torch.testing.assert_close(out, (x + 1) * 2)
                self.assertNotIn("helion_prefetch_code", code)

    @skipIfCudaCapabilityLessThan((9, 0), reason="code prefetch marks sm_90+ kernels")
    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_prefetch_code_between_roots(self) -> None:
        """hl.prefetch_code() between two loops runs where the megakernel's workers
        reach the second root (no prefetch at entry), read by every 16th worker
        counted from the last; where the launch runs no root as a block of its
        own (packet dispatch, a grid of waves) it prefetches at entry and says so,
        a grid of waves counting its readers from the first CTA."""
        x = torch.randn((8, 64), device=DEVICE, dtype=torch.float32)
        last = "helion_prefetch_code.prefetch_code((tl.num_programs(0) - 1 - tl.program_id(0)) % 16 == 0)"
        first = "helion_prefetch_code.prefetch_code(tl.program_id(0) % 16 == 0)"
        # A grid of waves takes roots without dependencies between them.
        for kernel, pid_type, pipeline, call in (
            (two_roots_prefetch_between, "persistent_blocked", "static", last),
            (two_roots_prefetch_between, "persistent_blocked", "dynamic", last),
            (independent_roots_prefetch_between, "flat", None, first),
        ):
            # The fallback to entry says so.
            logs = self.assertNoLogs if pipeline == "static" else self.assertLogs
            with (
                self.subTest(pid_type=pid_type, pipeline=pipeline),
                logs("helion._compiler.device_function", "INFO") as said,
            ):
                config = {"cross_loop_pipeline": pipeline} if pipeline else {}
                code, out = code_and_output(kernel, (x,), pid_type=pid_type, **config)
                torch.testing.assert_close(out, (x + 1) * 2 if pipeline else 3 * x + 1)
                self.assertEqual(code.count("helion_prefetch_code.prefetch_code("), 1)
                self.assertIn(call, code)
                device = code[code.index(f"def _helion_{kernel.name}(") :]
                if pipeline == "static":
                    # After the first root's tasks, before the second's.
                    at = device.index(call)
                    self.assertLess(device.index("tile_dependency_root_0"), at)
                    self.assertLess(at, device.index("tile_dependency_root_1"))
                else:
                    # Packet dispatch and waves run no root as a block: at entry.
                    lines = [line.strip() for line in device.splitlines()[1:]]
                    statements = [line for line in lines if not line.startswith("#")]
                    self.assertEqual(statements[0], call)
                    self.assertIn("before root 1 runs at entry", "".join(said.output))

    def test_dry_pass_runs_prefetch_on_its_dry_trip(self) -> None:
        """A dry pass of a root runs its code prefetch on the dry trip, ahead of
        the code it warms, and not on the live one (CPU only)."""
        import ast

        from helion._compiler import cross_loop_codegen

        body = [
            ast.parse("helion_prefetch_code.prefetch_code(tl.program_id(0) == 0)").body[
                0
            ],
            ast.parse("tl.store(p, v)").body[0],
        ]
        rewritten = cross_loop_codegen._dry_pass_rewrite(body, "live", frozenset(), [])
        text = ast.unparse(rewritten)
        self.assertIn("if not live:\n    helion_prefetch_code.prefetch_code(", text)
        # The store is masked live (unchanged).
        self.assertIn("mask=live", text)

    def test_every_call_copy_calls_the_prefetch(self) -> None:
        """The call site is the call: every copy LLVM makes of it (jump threading
        duplicates inline asm onto each incoming path) calls the one function the
        PTX stage declares ahead of the code and defines once per width (CPU
        only)."""
        body = prefetch_code._call_asm(1, first_warp=True).replace("$1", "%r1")
        part = prefetch_code._call_asm(4, first_warp=False).replace("$1", "%r1")
        ptx = (
            ".version 8.7\n.target sm_100a\n.address_size 64\n"
            ".func helper()\n{\n\tret;\n}\n"
            ".visible .entry k(\n\t.param .u64 k_param_0\n)\n{\n"
            "\t.reg .b32 %r<4>;\n\t.reg .b64 %rd<2>;\n"
            "\tld.param.u64 %rd1, [k_param_0];\n"
            f"\t// begin inline asm\n\t{body.replace('$0', '%r2')}\n"
            "\t// end inline asm\n\tst.global.u32 [%rd1], %r1;\n"
            f"\t// begin inline asm\n\t{body.replace('$0', '%r3')}\n"
            f"\t// end inline asm\n\t{part.replace('$0', '%r3')}\n"
            "\tst.global.u32 [%rd1], %r2;\n\tret;\n}\n"
        )
        out = prefetch_code.prefetch_code_ptx(ptx)
        # The call sites are untouched; one definition per width.
        self.assertEqual(out.count("call.uni helion_prefetch_code_fn_1;"), 2)
        self.assertEqual(out.count("call.uni helion_prefetch_code_fn_4;"), 1)
        self.assertEqual(out.count(".func helion_prefetch_code_fn_1()\n{"), 1)
        self.assertEqual(out.count(".func helion_prefetch_code_fn_4()\n{"), 1)
        # Declared before the first function, defined after the kernel.
        declared = out.index(".func helion_prefetch_code_fn_1();")
        self.assertLess(declared, out.index(".func helper()"))
        self.assertLess(
            out.index(".visible .entry k("),
            out.index(".func helion_prefetch_code_fn_1()\n{"),
        )
        # The table lists the kernel and its helper.
        self.assertIn(".u64 helion_prefetch_code_tab[5] = {k, 0, helper, 0, 0};", out)
        # The body's first warp alone; a partition's warps all walk the lines.
        self.assertIn("setp.lt.and.u32 %pcr, %pcq, 32, %pcr;", body)
        self.assertNotIn("%tid", part)
        prefetch4 = out[out.index(".func helion_prefetch_code_fn_4()\n{") :]
        self.assertIn("and.b32 %pcw, %pct, 127;", prefetch4)
        self.assertIn("mul.lo.u32 %pcnw, %pcnw, 128;", prefetch4)
        # A kernel without calls is unchanged.
        plain = ptx.replace("call.uni", "nop.uni")
        self.assertEqual(prefetch_code.prefetch_code_ptx(plain), plain)

    @skipIfRefEager("PdlPlacement is raised by compiled-mode type propagation")
    def test_prefetch_code_placement_is_checked(self) -> None:
        """After the last loop there is no code left to prefetch."""
        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(helion.exc.PdlPlacement):
            code_and_output(prefetch_code_after_loop, (x,))
