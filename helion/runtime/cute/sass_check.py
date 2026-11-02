"""SASS-level checks of compiled CuTe kernels.

A kernel declares the checks its instruction forms need as the
``_helion_cute_sass_checks`` attribute the device codegen emits beside
``_helion_cute_wrapper_plans``; the launcher runs them on the cubin
``cute.compile`` produced, before the kernel's first launch and before the
module is persisted to the on-disk cache (``_CompiledCuteLauncher``).  A
failed check raises :class:`helion.exc.CuteSassCheckFailed`: the kernel never
runs, and the autotuner drops the config as it drops any config that fails
to compile.

The one check today, ``hinted_cp_async``, guards the mma_dag emitter's
L2-hinted operand packets (``cp.async.cg.shared.global.L2::cache_hint``,
``cute_mma_dag_codegen.emit_issue_tile``).  The CuTe DSL's embedded
assembler (cutlass-dsl 4.7.0: "Cuda compilation tools, release 13.3,
V13.3.27", the producer its cubins name) has dropped the uniform-register
operand fields of that instruction whenever it keeps a uniform term of the
shared address in a uniform register: the SASS reads ``[R42+UR0+0xc00],
desc[UR1]`` where ptxas 13.4.59 writes ``[R42+UR10+0xc00], desc[UR11]`` for
the same PTX (UR10 the shared base, UR11 the policy word, both written
just before), and UR0 / UR1 are registers no instruction of the kernel
writes -- the kernel dies with an illegal instruction at launch.  Which
packets get that encoding is the assembler's register allocation (the same
render encodes differently across kernels and the first hinted packet of a
kernel is often fine), so nothing at plan time can rule it out; the check
reads the assembler's actual output instead (``hinted_cp_async_misencoding``:
the defect's signature, a uniform register the kernel never writes, plus a
documented conservative condition on the instruction form).
"""

from __future__ import annotations

import logging
import os
import re
import struct
import subprocess
import tempfile
from typing import TYPE_CHECKING

from ... import exc
from ..._compiler.cute import cutedsl_compat

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

log = logging.getLogger(__name__)

# The mma_dag emitter's check name (``MMA_DAG_HINTED_CP_ASYNC_CHECK``; a
# literal here so the launcher does not import the compiler's planner).
HINTED_CP_ASYNC_CHECK = "hinted_cp_async"
KNOWN_CHECKS: frozenset[str] = frozenset({HINTED_CP_ASYNC_CHECK})

# One SASS instruction line of ``nvdisasm -c``: ``/*0040*/ [@!P0] OPCODE
# operands ;`` (the trailing ``/* encoding */`` comment is optional).
_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?U?P[T0-9]+\s+)?(?P<text>[A-Z][A-Z0-9_.]*(?:\s[^;]*)?);"
)
# The ``.section .text.<function>`` header that opens each function of a
# listing.
_SECTION_RE = re.compile(r"^\s*\.section\s+\.text\.")
_UR_RE = re.compile(r"\bUR\d+\b")
# An instruction's uniform-register destination: its first operand
# (``UMOV UR10, UR8``, ``LDCU.64 UR12, c[0x0][0x358]``, ``S2UR UR8,
# SR_CgaCtaId``, ``UIADD3 UR4, UP0, UR4, UR6, URZ``), or the register after
# the completion predicate of ``R2UR P1, UR10, R4``.
_UR_DESTINATION_RE = re.compile(
    r"^(?P<opcode>[A-Z][A-Z0-9_.]*) (?:U?P[T0-9]+, )?(?P<ur>UR\d+)(?:,|$)"
)
# The assembler a cubin names in its ``.note.nv.tkinfo`` section.
_PRODUCER_RE = re.compile(rb"Cuda compilation tools, release \d+\.\d+, V[\d.]+")


def functions(sass: str) -> list[list[str]]:
    """The instruction texts of an ``nvdisasm`` listing (predicates and
    encodings stripped), in order, grouped by function (``.text`` section);
    lines ahead of the first section header form a group of their own."""
    groups: list[list[str]] = []
    current: list[str] = []
    for line in sass.splitlines():
        if _SECTION_RE.match(line):
            if current:
                groups.append(current)
            current = []
            continue
        match = _INSTRUCTION_RE.match(line)
        if match is not None:
            current.append(" ".join(match.group("text").split()))
    if current:
        groups.append(current)
    return groups


def instructions(sass: str) -> list[str]:
    """The instruction texts of an ``nvdisasm`` listing (predicates and
    encodings stripped), in order."""
    return [text for group in functions(sass) for text in group]


def written_uniform_registers(texts: Iterable[str]) -> set[str]:
    """The uniform registers the instructions ``texts`` write (``URZ``, the
    zero register, always counts).  A 64-bit destination (``LDCU.64 UR12``,
    ``UIMAD.WIDE.U32 UR4``) writes the next register too."""
    written = {"URZ"}
    for text in texts:
        match = _UR_DESTINATION_RE.match(text)
        if match is None:
            continue
        register = match.group("ur")
        written.add(register)
        modifiers = match.group("opcode").split(".")[1:]
        if "64" in modifiers or "WIDE" in modifiers:
            written.add(f"UR{int(register[2:]) + 1}")
    return written


def _ldgsts(texts: Iterable[str]) -> list[str]:
    return [text for text in texts if text.split(" ", 1)[0].startswith("LDGSTS")]


def unwritten_uniform_register_read(sass: str) -> str | None:
    """The defect's signature: the first LDGSTS of ``sass`` that names a
    uniform register no instruction of its function writes (the operand
    fields the DSL's assembler dropped decode as UR0 / UR1, which the
    kernel never writes; a correct encoding names only registers it loaded
    or computed), or None.  Reads only the LDGSTS: the misencoding is
    theirs, and a predicate-first destination form this parser does not
    know would otherwise refuse a correct kernel."""
    for texts in functions(sass):
        written = written_uniform_registers(texts)
        for text in _ldgsts(texts):
            unwritten = sorted(
                set(_UR_RE.findall(text)) - written,
                key=lambda register: int(register[2:]),
            )
            if unwritten:
                return (
                    f"{text} reads {', '.join(unwritten)}, which no instruction "
                    "of the kernel writes"
                )
    return None


def uniform_shared_address_beside_descriptor(sass: str) -> str | None:
    """The conservative condition: the first LDGSTS of ``sass`` whose shared
    address names a uniform register beside a cache-policy descriptor
    (``[R42+UR10+0xc00], desc[UR11]``), or None.

    This is the *form* the misencoding took, not the misencoding: ptxas
    13.4.59 emits the same form, correctly encoded, for the same PTX (the
    shared base kept uniform in a K loop), and the written-register rule
    above passes that listing.  The DSL's assembler dropped the operand
    fields in every observed instance of the form, and a kernel that
    happened to write UR0 and UR1 elsewhere would hide the defect from the
    written-register rule, so until a DSL release ships an assembler
    validated on this form the check refuses the form as well.  The cost is
    the hint on the kernels whose shared base the assembler keeps uniform
    (K-loop kernels such as the kda WY transform); retire this condition
    behind a gate on the producer the cubin names (``assembler``) once that
    assembler is validated.  A uniform register in the shared address
    without a descriptor is a plain packet's legal form.
    """
    for texts in functions(sass):
        for text in _ldgsts(texts):
            open_bracket = text.find("[")
            close_bracket = text.find("]", open_bracket)
            if open_bracket < 0 or close_bracket < 0:
                return f"unparsable LDGSTS operands: {text}"
            shared = text[open_bracket + 1 : close_bracket]
            if _UR_RE.search(shared) and "desc[" in text[close_bracket:]:
                return (
                    f"{text} keeps a uniform register in its shared address "
                    "beside the cache-policy descriptor (the form the "
                    "misencoding took; refused conservatively)"
                )
    return None


def hinted_cp_async_misencoding(sass: str) -> str | None:
    """The first finding against the hinted ``cp.async`` packets of ``sass``
    (``unwritten_uniform_register_read``, then the conservative
    ``uniform_shared_address_beside_descriptor``), or None when every LDGSTS
    passes both.  Every LDGSTS of an sm_90+ kernel carries a descriptor, so
    a kernel that declares this check (one with hinted packets) is judged on
    all of them."""
    found = unwritten_uniform_register_read(sass)
    if found is None:
        found = uniform_shared_address_beside_descriptor(sass)
    return found


_ELF_MAGIC = b"\x7fELF"
# NVIDIA fat binary container header magic (little-endian 0xBA55ED50): the
# form ``cute.compile`` embeds the device code in (one or more cubins and
# PTX entries behind per-entry headers).
_FATBIN_MAGIC = b"\x50\xed\x55\xba"


def elf_images(blob: bytes) -> list[bytes]:
    """The ELF images inside ``blob``: the blob itself when it is one, else
    every ELF of a fat binary container, each sized from its own header (the
    section header table ends an nvcc / ptxas cubin)."""
    if blob[:4] == _ELF_MAGIC:
        return [blob]
    if blob[:4] != _FATBIN_MAGIC:
        return []
    images: list[bytes] = []
    start = blob.find(_ELF_MAGIC)
    while start >= 0:
        header = blob[start : start + 64]
        if len(header) < 64:
            break
        # ELF64 little-endian: e_phoff / e_shoff at 0x20 / 0x28 (u64),
        # e_phentsize / e_phnum / e_shentsize / e_shnum at 0x36.. (u16).
        phoff, shoff = struct.unpack_from("<QQ", header, 0x20)
        phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", header, 0x36)
        end = max(phoff + phentsize * phnum, shoff + shentsize * shnum, 64)
        end = min(start + end, len(blob))
        images.append(blob[start:end])
        start = blob.find(_ELF_MAGIC, end)
    return images


def assembler(cubin: bytes) -> str | None:
    """The assembler that produced ``cubin``, as its ``.note.nv.tkinfo``
    names it (``"Cuda compilation tools, release 13.3, V13.3.27"``), or
    None."""
    match = _PRODUCER_RE.search(cubin)
    return match.group(0).decode("ascii") if match is not None else None


def _cubin_blobs(ir_module: object) -> list[bytes]:
    """The cubins embedded in a compiled module: ``cute.compile`` lowers the
    device code into an LLVM string global of the host module (the fat
    binary ``kernels_binary``), so every ELF of every string global that
    holds a fat binary or a bare cubin is one."""
    from cutlass._mlir import ir

    advance = ir.WalkResult.ADVANCE  # pyrefly: ignore[missing-attribute]
    string_attr = ir.StringAttr  # pyrefly: ignore[missing-attribute]
    blobs: list[bytes] = []
    failures: list[str] = []

    def visit(op: object) -> object:
        if op.name != "llvm.mlir.global":  # pyrefly: ignore[missing-attribute]
            return advance
        try:
            attributes = op.attributes  # pyrefly: ignore[missing-attribute]
            if "value" in attributes:
                value = attributes["value"]
                if isinstance(value, string_attr):
                    blobs.extend(elf_images(bytes(value.value_bytes)))
        except Exception as error:
            # (An exception escaping the walk callback loses its message.)
            failures.append(f"{type(error).__name__}: {error}")
        return advance

    ir_module.operation.walk(visit)  # pyrefly: ignore[missing-attribute]
    if failures and not blobs:
        raise RuntimeError("; ".join(failures[:3]))
    return blobs


def disassemble(cubin: bytes) -> str:
    """``nvdisasm -c`` of one cubin."""
    tool = cutedsl_compat.nvdisasm_path()
    if tool is None:
        raise FileNotFoundError("nvdisasm")
    fd, path = tempfile.mkstemp(suffix=".cubin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(cubin)
        result = subprocess.run(
            [tool, "-c", path], capture_output=True, text=True, check=False
        )
    finally:
        os.unlink(path)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvdisasm failed ({result.returncode}): {result.stderr.strip()[-500:]}"
        )
    return result.stdout


def misencoding(sass: str, checks: Iterable[str]) -> str | None:
    """The first finding of ``checks`` over ``sass`` as ``"<check>: <text>"``,
    or None when every check passes."""
    for check in checks:
        if check == HINTED_CP_ASYNC_CHECK:
            found = hinted_cp_async_misencoding(sass)
        else:
            found = f"unknown SASS check {check!r}"
        if found is not None:
            return f"{check}: {found}"
    return None


def verify_compiled(compiled: object, checks: Sequence[str], kernel_name: str) -> None:
    """Run ``checks`` on every cubin of the compiled module ``compiled``
    (``cute.compile``'s result, whose ``ir_module`` holds the kernels'
    binaries).  Fail closed: a cubin that cannot be found or disassembled
    fails the check like a misencoding does.  A refusal names the assembler
    the cubin was produced by."""
    if not checks:
        return
    try:
        blobs = _cubin_blobs(compiled.ir_module)  # pyrefly: ignore[missing-attribute]
    except Exception as error:
        raise exc.CuteSassCheckFailed(
            kernel_name, ", ".join(checks), f"could not read the cubin: {error}"
        ) from error
    if not blobs:
        raise exc.CuteSassCheckFailed(
            kernel_name, ", ".join(checks), "the compiled module holds no cubin"
        )
    for blob in blobs:
        try:
            sass = disassemble(blob)
        except Exception as error:
            raise exc.CuteSassCheckFailed(
                kernel_name, ", ".join(checks), f"could not disassemble: {error}"
            ) from error
        found = misencoding(sass, checks)
        if found is not None:
            producer = assembler(blob) or "an assembler the cubin does not name"
            raise exc.CuteSassCheckFailed(
                kernel_name, ", ".join(checks), f"{found} (assembled by {producer})"
            )
    log.debug(
        "SASS checks %s passed for %s (%d cubin(s), assembled by %s)",
        ", ".join(checks),
        kernel_name,
        len(blobs),
        assembler(blobs[0]),
    )


__all__ = [
    "HINTED_CP_ASYNC_CHECK",
    "KNOWN_CHECKS",
    "assembler",
    "disassemble",
    "functions",
    "hinted_cp_async_misencoding",
    "instructions",
    "misencoding",
    "uniform_shared_address_beside_descriptor",
    "unwritten_uniform_register_read",
    "verify_compiled",
    "written_uniform_registers",
]
