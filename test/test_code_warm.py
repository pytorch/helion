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
from helion.runtime.triton import code_warm


def _table_sizes(cubin: bytes) -> list[int]:
    """The text sizes in a cubin's code warm table (zero: unresolved)."""
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
        if cubin[strtab + name :].split(b"\0", 1)[0] == b"helion_codewarm_tab":
            words = struct.unpack_from(
                f"<{size // 8}Q", cubin, sections[index][4] + value
            )
            # (text address, text bytes) per function, then the sink word.
            return list(words[1::2])
    raise AssertionError("no code warm table")


@triton.jit
def _warmed_copy(x, out):
    code_warm.code_warm()
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
    hl.code_warm()
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
def warm_after_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    hl.code_warm()
    return out


@onlyBackends(["triton"])
class TestCodeWarm(TestCase):
    @skipIfCudaCapabilityLessThan((9, 0), reason="code warm marks sm_90+ kernels")
    @skipIfNotCUDA()
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_marked_kernel_warms_its_code(self) -> None:
        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        out = torch.empty_like(x)
        compiled = _warmed_copy[(1,)](x, out)
        torch.testing.assert_close(out, x)
        ptx = compiled.asm["ptx"]
        self.assertIn(".global .align 8 .u64 helion_codewarm_tab[3]", ptx)
        # The keep-alive store targets the sink word, past the table entries.
        self.assertIn("st.global.u32 [helion_codewarm_tab+16]", ptx)
        # The cubin stage resolved the entry's text size.
        self.assertEqual(len(_table_sizes(compiled.asm["cubin"])), 1)
        self.assertTrue(all(_table_sizes(compiled.asm["cubin"])))
        self.assertIn("ld.global.cg.L2::cache_hint.u32", ptx)
        # The warm is a function the entry calls: inline, its loop costs ptxas
        # its uniform-register analysis across the whole kernel.
        self.assertIn(".func helion_codewarm_fn()", ptx)
        self.assertIn("call.uni helion_codewarm_fn;", ptx)
        # Unmarked kernels are unchanged.
        plain = _plain_copy[(1,)](x, out)
        self.assertNotIn("helion_codewarm_tab", plain.asm["ptx"])

    @skipIfCudaCapabilityLessThan((9, 0), reason="code warm marks sm_90+ kernels")
    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_source_marks_megakernel_for_code_warm(self) -> None:
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
                self.assertIn("helion_code_warm.code_warm()", code)
                # Without hl.code_warm() the megakernel stays unmarked.
                code, out = code_and_output(
                    two_roots_unmarked,
                    (x,),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                )
                torch.testing.assert_close(out, (x + 1) * 2)
                self.assertNotIn("helion_code_warm", code)

    @skipIfRefEager("PdlPlacement is raised by compiled-mode type propagation")
    def test_code_warm_after_the_loops_is_rejected(self) -> None:
        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(helion.exc.PdlPlacement):
            code_and_output(warm_after_loop, (x,))
