"""The launcher's SASS checks (``helion.runtime.cute.sass_check``): the LDGSTS
encoding the CuTe DSL's assembler produced for the mma_dag emitter's hinted
``cp.async`` packets against the correct encoding of the same PTX, parsed
from real ``nvdisasm`` / ``cuobjdump`` lines; the cubin extraction from a
fat binary; the fail-closed verification of a compiled module.  Pure
Python: no GPU, no DSL."""

from __future__ import annotations

import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from helion import exc
from helion._compiler.cute import cutedsl_compat
from helion._compiler.cute.cute_mma_dag import MMA_DAG_HINTED_CP_ASYNC_CHECK
from helion.runtime.cute import sass_check

# The kda WY transform's autotune-fault config (``_WY_FAULT_CONFIG`` of the
# GPU tests), its one PTX assembled twice and disassembled with the same
# ``nvdisasm -c``; each excerpt is the first staging region's fourteen
# LDGSTS plus every instruction that writes a uniform register they name.
#
# cutlass-dsl 4.7.0's embedded assembler ("Cuda compilation tools, release
# 13.3, V13.3.27"): six packets split the shared address into ``[R42 + UR0
# + imm]`` beside ``desc[UR1]`` -- UR0 / UR1 are written nowhere (the
# operand fields were dropped); the kernel died with an illegal instruction.
_WY_DSL = """
        /*0070*/                   S2UR UR8, SR_CgaCtaId ;
        /*91f0*/                   LDCU.64 UR12, c[0x0][0x358] ;
        /*9220*/                   UMOV UR10, 0xf0 ;
        /*9510*/                   USHF.R.U32 UR10, UR10, 0x4, URZ ;
        /*9550*/                   UMOV UR10, URZ ;
        /*9850*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R6+0x800], desc[UR10][R16.64] ;
        /*9890*/                   UMOV UR10, UR8 ;
        /*9910*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x2800], desc[UR12][R44.64] ;
        /*9970*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR0+0xc00], desc[UR1][R12.64] ;
        /*9990*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x3000], desc[UR12][R44.64+0x20] ;
        /*99b0*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR0+0x1000], desc[UR1][R14.64] ;
        /*99c0*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x3800], desc[UR12][R44.64+0x40] ;
        /*99e0*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR0+0x1400], desc[UR1][R18.64] ;
        /*99f0*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x4000], desc[UR12][R44.64+0x60] ;
        /*9a10*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR0+0x1800], desc[UR1][R6.64] ;
        /*9a40*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x4800], desc[UR12][R44.64+0x80] ;
        /*9ad0*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR0+0x1c00], desc[UR1][R10.64] ;
        /*9af0*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x5000], desc[UR12][R44.64+0xa0] ;
        /*9b40*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR0+0x2000], desc[UR1][R12.64] ;
        /*9b70*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x5800], desc[UR12][R44.64+0xc0] ;
"""
# ptxas 13.4.59 (Triton's ``ptxas-blackwell``) on the same PTX, which runs
# correctly: the same fourteen packets in the same forms, the split ones
# naming UR10 (the shared base, ``UMOV UR10, UR8``) and UR11 (the policy
# word, ``UMOV / ULOP3 / USHF UR11``) -- every register written.
_WY_PTXAS = """
        /*0070*/                   S2UR UR8, SR_CgaCtaId ;
        /*91f0*/                   LDCU.64 UR12, c[0x0][0x358] ;
        /*9220*/                   UMOV UR10, 0xf0 ;
        /*9510*/                   USHF.R.U32 UR10, UR10, 0x4, URZ ;
        /*9520*/                   UMOV UR11, 0x140 ;
        /*9530*/                   ULOP3.LUT UR11, UR11, 0xf, UR10, 0xf8, !UPT ;
        /*9540*/                   USHF.L.U32 UR11, UR11, 0x14, URZ ;
        /*9550*/                   UMOV UR10, URZ ;
        /*9850*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R6+0x800], desc[UR10][R16.64] ;
        /*9890*/                   UMOV UR10, UR8 ;
        /*9910*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x2800], desc[UR12][R44.64] ;
        /*9970*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR10+0xc00], desc[UR11][R12.64] ;
        /*9990*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x3000], desc[UR12][R44.64+0x20] ;
        /*99b0*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR10+0x1000], desc[UR11][R14.64] ;
        /*99c0*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x3800], desc[UR12][R44.64+0x40] ;
        /*99e0*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR10+0x1400], desc[UR11][R18.64] ;
        /*99f0*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x4000], desc[UR12][R44.64+0x60] ;
        /*9a10*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR10+0x1800], desc[UR11][R6.64] ;
        /*9a40*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x4800], desc[UR12][R44.64+0x80] ;
        /*9ad0*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR10+0x1c00], desc[UR11][R10.64] ;
        /*9af0*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x5000], desc[UR12][R44.64+0xa0] ;
        /*9b40*/              @!P3 LDGSTS.E.BYPASS.LTC128B.128 [R42+UR10+0x2000], desc[UR11][R12.64] ;
        /*9b70*/                   LDGSTS.E.BYPASS.LTC128B.128 [R43+0x5800], desc[UR12][R44.64+0xc0] ;
"""
_WY_BROKEN_FIRST = "LDGSTS.E.BYPASS.LTC128B.128 [R42+UR0+0xc00], desc[UR1][R12.64]"
_WY_CORRECT_FIRST = "LDGSTS.E.BYPASS.LTC128B.128 [R42+UR10+0xc00], desc[UR11][R12.64]"
# The indexer decode's hinted kernel under the DSL's assembler (nvdisasm
# -c): every packet's shared address is a vector register plus an
# immediate and the descriptor's register is loaded.
_DECODE_OK = """
        /*0080*/                   LDCU.64 UR4, c[0x0][0x358] ;
        /*02c0*/                   LDGSTS.E.BYPASS.128 [R13], desc[UR4][R6.64] ;
        /*0300*/                   LDGSTS.E.BYPASS.128 [R15+0x200], desc[UR4][R6.64+0x200] ;
        /*0110*/                   LDG.E.128 R36, desc[UR4][R80.64] ;
        /*0120*/                   LDGDEPBAR ;
        /*0130*/                   DEPBAR.LE SB0, 0x0 ;
"""


def _line(text: str, address: int = 0x10) -> str:
    return f"        /*{address:04x}*/                   {text} ;\n"


def test_instruction_lines_drop_predicates_and_encodings() -> None:
    texts = sass_check.instructions(_WY_DSL)
    assert texts[:3] == [
        "S2UR UR8, SR_CgaCtaId",
        "LDCU.64 UR12, c[0x0][0x358]",
        "UMOV UR10, 0xf0",
    ]
    assert texts[8] == _WY_BROKEN_FIRST
    assert sass_check.instructions(_DECODE_OK)[-2:] == [
        "LDGDEPBAR",
        "DEPBAR.LE SB0, 0x0",
    ]
    assert sass_check.instructions("\t.target\tsm_100a\n// comment\n") == []
    # Functions are the listing's ``.text`` sections; a header-less excerpt
    # is one function.
    assert sass_check.functions(_DECODE_OK) == [sass_check.instructions(_DECODE_OK)]
    two = (
        '\t.section\t.text.kernel_a,"ax",@progbits\n'
        + _line("UMOV UR4, 0x1")
        + '\t.section\t.text.kernel_b,"ax",@progbits\n'
        + _line("UMOV UR6, 0x2")
    )
    assert sass_check.functions(two) == [["UMOV UR4, 0x1"], ["UMOV UR6, 0x2"]]


def test_uniform_register_destinations() -> None:
    written = sass_check.written_uniform_registers(sass_check.instructions(_WY_PTXAS))
    assert written == {"URZ", "UR8", "UR10", "UR11", "UR12", "UR13"}
    # A 64-bit or wide destination writes the pair; the register after
    # ``R2UR``'s completion predicate is its destination; a predicate
    # compare writes no uniform register; stores and the LDGSTS themselves
    # write none.
    assert sass_check.written_uniform_registers(
        [
            "UIMAD.WIDE.U32 UR4, UR6, 0x4, UR8",
            "R2UR P1, UR10, R4",
            "R2UR.OR P1, UR11, R5",
            "UISETP.GE.AND UP0, UPT, UR14, UR15, UPT",
            "STG.E [R2.64+UR16], R5",
            "LDGSTS.E.BYPASS.128 [R13+UR18], desc[UR20][R6.64]",
            "UIADD3 UR22, UP0, UR22, UR6, URZ",
        ]
    ) == {"URZ", "UR4", "UR5", "UR10", "UR11", "UR22"}


def test_the_defect_is_a_uniform_register_the_kernel_never_writes() -> None:
    # The DSL's listing: the first split packet reads UR0 and UR1, which no
    # instruction writes.  ptxas 13.4.59's listing of the same PTX names
    # written registers on every LDGSTS and passes.
    found = sass_check.unwritten_uniform_register_read(_WY_DSL)
    assert found is not None and found.startswith(_WY_BROKEN_FIRST)
    assert found.endswith("reads UR0, UR1, which no instruction of the kernel writes")
    assert sass_check.unwritten_uniform_register_read(_WY_PTXAS) is None
    assert sass_check.unwritten_uniform_register_read(_DECODE_OK) is None
    # The dropped fields on a packet whose shared address is ``[R + imm]``:
    # the descriptor's register alone is unwritten, and the rule refuses it
    # (no shared-address form would).
    dropped = _line("LDGSTS.E.BYPASS.128 [R13+0x200], desc[UR1][R6.64]")
    assert sass_check.unwritten_uniform_register_read(_DECODE_OK + dropped) == (
        "LDGSTS.E.BYPASS.128 [R13+0x200], desc[UR1][R6.64] reads UR1, which no "
        "instruction of the kernel writes"
    )
    assert sass_check.uniform_shared_address_beside_descriptor(dropped) is None
    # Written in another function does not count: each function is judged
    # on its own instructions.
    other = '\t.section\t.text.other,"ax",@progbits\n' + _line("UMOV UR1, 0x1")
    own = '\t.section\t.text.kernel,"ax",@progbits\n' + dropped
    assert sass_check.unwritten_uniform_register_read(other + own) is not None
    assert (
        sass_check.unwritten_uniform_register_read(_line("UMOV UR1, 0x1") + dropped)
        is None
    )
    # The zero register needs no writer.
    assert (
        sass_check.unwritten_uniform_register_read(
            _line("LDGSTS.E.BYPASS.128 [R13+URZ], desc[UR4][R6.64]")
            + _line("LDCU.64 UR4, c[0x0][0x358]")
        )
        is None
    )
    assert sass_check.unwritten_uniform_register_read("") is None


def test_the_form_condition_is_conservative_and_documented_as_such() -> None:
    # The split shared address beside a descriptor is the form the
    # misencoding took; ptxas 13.4.59 emits it correctly encoded for the
    # same PTX, and the form condition flags that listing too (the price,
    # until the DSL's assembler is validated on the form, is the hint on
    # kernels whose shared base the assembler keeps uniform).
    found = sass_check.uniform_shared_address_beside_descriptor(_WY_PTXAS)
    assert found is not None and found.startswith(_WY_CORRECT_FIRST)
    assert "refused conservatively" in found
    found = sass_check.uniform_shared_address_beside_descriptor(_WY_DSL)
    assert found is not None and found.startswith(_WY_BROKEN_FIRST)
    assert sass_check.uniform_shared_address_beside_descriptor(_DECODE_OK) is None
    # A uniform register in the shared address without a descriptor is a
    # plain packet's legal form (pre-descriptor architectures).
    assert (
        sass_check.uniform_shared_address_beside_descriptor(
            _line("LDGSTS.E.BYPASS.128 [R2+UR4+0x400], [R6.64]")
        )
        is None
    )
    # Only LDGSTS is judged: a hinted LDG with a descriptor is fine.
    assert (
        sass_check.uniform_shared_address_beside_descriptor(
            _line("LDG.E.128 R36, desc[UR4][R80.64+UR5]")
        )
        is None
    )
    assert sass_check.uniform_shared_address_beside_descriptor("") is None


def test_the_check_reports_the_defect_first_then_the_form() -> None:
    assert sass_check.hinted_cp_async_misencoding(_DECODE_OK) is None
    found = sass_check.hinted_cp_async_misencoding(_WY_DSL)
    assert found is not None and "which no instruction of the kernel writes" in found
    assert found.startswith(_WY_BROKEN_FIRST)
    found = sass_check.hinted_cp_async_misencoding(_WY_PTXAS)
    assert found is not None and "refused conservatively" in found
    assert found.startswith(_WY_CORRECT_FIRST)


def test_misencoding_names_its_check_and_rejects_unknown_checks() -> None:
    assert sass_check.misencoding(_DECODE_OK, [MMA_DAG_HINTED_CP_ASYNC_CHECK]) is None
    assert MMA_DAG_HINTED_CP_ASYNC_CHECK == sass_check.HINTED_CP_ASYNC_CHECK
    found = sass_check.misencoding(_WY_DSL, [sass_check.HINTED_CP_ASYNC_CHECK])
    assert found is not None and found.startswith("hinted_cp_async: LDGSTS")
    assert sass_check.misencoding(_DECODE_OK, ["no_such_check"]) == (
        "no_such_check: unknown SASS check 'no_such_check'"
    )


def _elf(payload: bytes) -> bytes:
    """A minimal ELF64 image: the header, ``payload``, then a one-entry
    section header table (which ends the image)."""
    body = payload
    shoff = 64 + len(body)
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    struct.pack_into("<QQ", header, 0x20, 0, shoff)
    struct.pack_into("<HHHH", header, 0x36, 0, 0, 64, 1)
    return bytes(header) + body + bytes(64)


_PRODUCER = b"Cuda compilation tools, release 13.3, V13.3.27"


def test_elf_images_come_out_of_a_fat_binary_sized_by_their_headers() -> None:
    first = _elf(b"first cubin " + _PRODUCER)
    second = _elf(b"second, longer cubin payload")
    assert sass_check.elf_images(first) == [first]
    fatbin = (
        b"\x50\xed\x55\xba"
        + bytes(12)
        + bytes(48)  # an entry header
        + first
        + bytes(16)  # padding between entries
        + bytes(48)
        + second
        + bytes(8)
    )
    assert sass_check.elf_images(fatbin) == [first, second]
    assert sass_check.elf_images(b"neither an ELF nor a fat binary") == []
    # A truncated image is clamped to the blob.
    cut = fatbin[: len(fatbin) - len(second) // 2 - 8]
    images = sass_check.elf_images(cut)
    assert images[0] == first and images[1] == second[: len(images[1])]
    # The producer a cubin names in its ``.note.nv.tkinfo``.
    assert sass_check.assembler(first) == _PRODUCER.decode()
    assert sass_check.assembler(second) is None


def test_verify_compiled_fails_closed() -> None:
    compiled = SimpleNamespace(ir_module=object())
    checks = (sass_check.HINTED_CP_ASYNC_CHECK,)
    cubin = _elf(_PRODUCER)
    with (
        patch.object(sass_check, "_cubin_blobs", return_value=[cubin]),
        patch.object(sass_check, "disassemble", return_value=_DECODE_OK),
    ):
        sass_check.verify_compiled(compiled, checks, "k")
        sass_check.verify_compiled(compiled, (), "k")
    # The refusal quotes the instruction and names the assembler.
    with (
        patch.object(sass_check, "_cubin_blobs", return_value=[cubin]),
        patch.object(sass_check, "disassemble", return_value=_WY_DSL),
        pytest.raises(
            exc.CuteSassCheckFailed,
            match=r"\[R42\+UR0\+0xc00\].*release 13\.3, V13\.3\.27",
        ),
    ):
        sass_check.verify_compiled(compiled, checks, "k")
    with (
        patch.object(sass_check, "_cubin_blobs", return_value=[b"cubin"]),
        patch.object(sass_check, "disassemble", return_value=_WY_PTXAS),
        pytest.raises(
            exc.CuteSassCheckFailed,
            match=r"refused conservatively.*an assembler the cubin does not name",
        ),
    ):
        sass_check.verify_compiled(compiled, checks, "k")
    # Two cubins: the second one's defect fails the kernel too.
    with (
        patch.object(sass_check, "_cubin_blobs", return_value=[b"a", b"b"]),
        patch.object(sass_check, "disassemble", side_effect=[_DECODE_OK, _WY_DSL]),
        pytest.raises(exc.CuteSassCheckFailed, match="hinted_cp_async"),
    ):
        sass_check.verify_compiled(compiled, checks, "k")
    # No cubin, an unreadable module, a failing disassembler: refused.
    with (
        patch.object(sass_check, "_cubin_blobs", return_value=[]),
        pytest.raises(exc.CuteSassCheckFailed, match="holds no cubin"),
    ):
        sass_check.verify_compiled(compiled, checks, "k")
    with (
        patch.object(sass_check, "_cubin_blobs", side_effect=RuntimeError("walk")),
        pytest.raises(exc.CuteSassCheckFailed, match="could not read the cubin: walk"),
    ):
        sass_check.verify_compiled(compiled, checks, "k")
    with (
        patch.object(sass_check, "_cubin_blobs", return_value=[b"cubin"]),
        patch.object(
            sass_check, "disassemble", side_effect=FileNotFoundError("nvdisasm")
        ),
        pytest.raises(exc.CuteSassCheckFailed, match="could not disassemble"),
    ):
        sass_check.verify_compiled(compiled, checks, "k")
    # The refusal's text must not read as a device fault: the autotuner
    # treats those as unrecoverable and restarts its worker.
    from helion.autotuner.logger import match_unrecoverable_runtime_error

    error = exc.CuteSassCheckFailed("k", "hinted_cp_async", "x")
    assert not match_unrecoverable_runtime_error(error)
    for listing in (_WY_DSL, _WY_PTXAS):
        found = sass_check.misencoding(listing, checks)
        assert found is not None
        assert not match_unrecoverable_runtime_error(
            exc.CuteSassCheckFailed(
                "k", "hinted_cp_async", f"{found} (assembled by {_PRODUCER.decode()})"
            )
        )


def test_disassemble_needs_the_disassembler() -> None:
    with (
        patch.object(cutedsl_compat, "nvdisasm_path", return_value=None),
        pytest.raises(FileNotFoundError),
    ):
        sass_check.disassemble(b"\x7fELF")
    # The availability gate the normalizer and the seeds consult is the
    # same probe.
    with patch.object(cutedsl_compat, "nvdisasm_path", return_value=None):
        assert not cutedsl_compat.hinted_cp_async_available()
    with patch.object(cutedsl_compat, "nvdisasm_path", return_value="/x/nvdisasm"):
        assert cutedsl_compat.hinted_cp_async_available()


def test_the_bundled_nvdisasm_is_found() -> None:
    # The CuTe DSL depends on the nvidia-cuda-nvdisasm wheel; where it is
    # installed the probe finds its binary (the environment this repository
    # tests the cute backend in).
    pytest.importorskip("cutlass")
    path = cutedsl_compat.nvdisasm_path()
    assert path is not None and path.endswith("nvdisasm")
