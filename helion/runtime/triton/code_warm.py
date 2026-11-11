"""Megakernel code warm.

A megakernel's native text outgrows the launch prefetch window, and the weights
it and the kernels between its launches stream evict that text from L2, so its
tasks first run their code from DRAM. Kernels that call ``code_warm()`` (what
``hl.code_warm()`` lowers to) L2-warm their own text at entry, before their PDL
wait.
Importing this module (a generated kernel calling ``code_warm()`` does)
registers a compiler stage hook: the PTX stage gives a marked kernel a table of
its entry and helper functions plus the entry loop, and the cubin stage turns the
table's function relocations into text addresses and writes the text sizes (when
it cannot, the sizes stay zero, the loop loads nothing and a warning says so).
Unmarked kernels are unchanged; the hook adds this module's source hash to the
cache key of the kernels compiled after it in the process.
"""

from __future__ import annotations

import functools
import hashlib
import logging
from pathlib import Path
import re
import struct
from typing import Any
from typing import cast

import triton
import triton.language as tl

# Compiled kernels are cached under a key that tracks this pass's source.
_KEY = "helion_code_warm_" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_R_CUDA_64 = 2
_R_CUDA_DESCRIPTOR = 35
log: logging.Logger = logging.getLogger(__name__)


_CW_TAB = "helion_codewarm_tab"
_CW_FN = "helion_codewarm_fn"
_CW_MARK = "// helion code warm"


@triton.jit
def code_warm():  # noqa: ANN201
    """Mark the calling kernel to L2-warm its own native text at entry."""
    tl.inline_asm_elementwise(
        "mov.u32 $0, 0; // helion code warm",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def code_warm_here(warps: int) -> str:
    """Device code warming the kernel's text where it sits, by the calling
    partition's ``warps`` warps (a warp-specialized kernel's idle partition)."""
    marker = f"mov.u32 $0, 0; {_CW_MARK} here {warps}"
    return f"tl.inline_asm_elementwise({marker!r}, '=r', [], tl.int32, False, 1)"


def codewarm_ptx(ptx: str) -> str:
    """Declare the text-range table (a (base, bytes) pair per function, then a
    sink word) and emit the warm (CTAs with blockIdx % 16 == 0 share the lines,
    evict_last loads): at a partition's code_warm_here<W> marker by its W warps,
    else at entry by one warp.  The warm is its own function: inline, its loop
    costs ptxas its uniform-register analysis across the whole kernel (every
    kernel argument then moves to a uniform register at each use).  It must not
    use bar.sync 0: a warp-specialized partition calls it."""
    if _CW_MARK not in ptx:
        return ptx
    head = re.search(r"\.visible \.entry (\w+)\([^)]*\)", ptx, re.DOTALL)
    body = re.search(
        r"\.visible \.entry \w+\([^)]*\)\s*(?:\.\w+[^{]*)?\{", ptx, re.DOTALL
    )
    if head is None or body is None:
        _warn_unresolved("no kernel entry in the PTX")
        return ptx
    funcs = re.findall(
        r"^(?:\.(?:visible|weak)\s+)?\.func\s+(?:\([^)]*\)\s*)?([\w$]+)\s*\(",
        ptx,
        re.MULTILINE,
    )
    names = list(dict.fromkeys([head[1], *funcs]))
    marker = r"(mov\.u32 \t?%r\d+, 0; " + re.escape(_CW_MARK) + r" here (\d+))"
    here = re.search(marker, ptx[body.end() :])
    if here is None and f"{_CW_MARK} here" in ptx:
        _warn_unresolved("a partition warm marker outside the entry; warming at entry")
    lanes = 32 * int(here[2]) if here is not None else 32
    slots = ", ".join(f"{n}, 0" for n in names)
    table = f".global .align 8 .u64 {_CW_TAB}[{2 * len(names) + 1}] = {{{slots}, 0}};"
    lane = (
        "setp.ge.u32 %cwp, %cwt, 32; @%cwp bra CW_DONE;"
        if here is None
        else f"and.b32 %cwt, %cwt, {lanes - 1};"
    )
    warm = f"""\
.func {_CW_FN}()
{{
\t{{
\t.reg .pred %cwp; .reg .b32 %cwt, %cwc, %cwnw, %cww, %cwn, %cwi, %cwk, %cwacc, %cwx;
\t.reg .b64 %cwbase, %cwbytes, %cwaddr, %cwpol;
\tmov.u32 %cwt, %tid.x; {lane}
\tmov.u32 %cwc, %ctaid.x; and.b32 %cwx, %cwc, 15; setp.ne.u32 %cwp, %cwx, 0;
\t@%cwp bra CW_DONE;
\tmov.u32 %cwnw, %nctaid.x; add.u32 %cwnw, %cwnw, 15; shr.u32 %cwnw, %cwnw, 4;
\tshr.u32 %cww, %cwc, 4; mad.lo.u32 %cww, %cww, {lanes}, %cwt; mul.lo.u32 %cwnw, %cwnw, {lanes};
\tcreatepolicy.fractional.L2::evict_last.b64 %cwpol, 1.0; mov.u32 %cwacc, 0;
"""
    for i in range(len(names)):
        warm += f"""\
\tld.global.u64 %cwbase, [{_CW_TAB}+{16 * i}];
\tld.global.u64 %cwbytes, [{_CW_TAB}+{16 * i + 8}];
\tadd.u64 %cwbytes, %cwbytes, 127; shr.u64 %cwbytes, %cwbytes, 7;
\tcvt.u32.u64 %cwn, %cwbytes; mov.u32 %cwi, %cww;
CW_LOOP{i}:
\tsetp.ge.u32 %cwp, %cwi, %cwn; @%cwp bra CW_NEXT{i};
"""
        # Eight lines in flight per lane, then fold their words.
        for j in range(8):
            warm += f"""\
\t{{ .reg .b32 %cwv; .reg .pred %cwq; mad.lo.u32 %cwk, %cwnw, {j}, %cwi;
\tsetp.lt.u32 %cwq, %cwk, %cwn; mul.wide.u32 %cwaddr, %cwk, 128;
\tadd.u64 %cwaddr, %cwaddr, %cwbase; mov.u32 %cwv, 0;
\t@%cwq ld.global.cg.L2::cache_hint.u32 %cwv, [%cwaddr], %cwpol;
\txor.b32 %cwacc, %cwacc, %cwv; }}
"""
        warm += f"\tmad.lo.u32 %cwi, %cwnw, 8, %cwi; bra CW_LOOP{i};\nCW_NEXT{i}:\n"
    # An impossible store keeps the loads live; their L2 fill is the only effect.
    warm += f"""\
\tsetp.eq.u32 %cwp, %cwacc, 0x9E3779B9;
\t@%cwp st.global.u32 [{_CW_TAB}+{16 * len(names)}], %cwacc;
CW_DONE:
\t}}
\tret;
}}"""
    i, j = head.start(), body.end()
    call = f"\tcall.uni {_CW_FN};"
    rest = ptx[j:]
    if here is not None:
        rest = "\n" + rest.replace(here[1], f"{here[1]}\n{call}", 1)
    else:
        rest = f"\n{call}\n{rest}"
    return (
        f"{ptx[:i]}{head[0]};\n{table}\n.func {_CW_FN}();\n{ptx[i:j]}{rest}\n{warm}\n"
    )


def codewarm_cubin(cubin: bytes) -> bytes:
    """Retype the table's descriptor relocations to text PCs; write text sizes."""
    if _CW_TAB.encode() not in cubin:
        return cubin
    try:
        return _patch_cubin(cubin)
    except (struct.error, ValueError, IndexError):
        # An ELF layout this parser does not expect: leave the table unresolved.
        return _unresolved(cubin, "unexpected cubin layout")


def _patch_cubin(cubin: bytes) -> bytes:
    data = bytearray(cubin)
    if data[:4] != b"\x7fELF":
        return _unresolved(cubin, "not an ELF cubin")
    hdr = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
    shoff, shentsize, shnum, shstrndx = hdr[5], hdr[10], hdr[11], hdr[12]
    secs = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        for i in range(shnum)
    ]

    def name_at(off: int, base: int) -> str:
        return data[base + off : data.index(b"\0", base + off)].decode()

    names = [name_at(sec[0], secs[shstrndx][4]) for sec in secs]
    if ".symtab" not in names or ".rela.nv.global.init" not in names:
        return _unresolved(cubin, "no table relocations")
    symtab = secs[names.index(".symtab")]
    sym_str = secs[symtab[6]][4]
    syms = []
    for off in range(symtab[4], symtab[4] + symtab[5], 24):
        n, _info, _other, shndx, value, size = struct.unpack_from("<IBBHQQ", data, off)
        syms.append((name_at(n, sym_str), shndx, value, size))
    tabs = [sym for sym in syms if sym[0] == _CW_TAB]
    if not tabs:
        return _unresolved(cubin, "no table symbol")
    _, tab_sec, tab_val, tab_size = tabs[0]
    init = secs[tab_sec]
    resolved = 0
    rela = secs[names.index(".rela.nv.global.init")]
    for k in range(rela[5] // 24):
        off, info, addend = struct.unpack_from("<QQq", data, rela[4] + 24 * k)
        sym, kind = info >> 32, info & 0xFFFFFFFF
        if not (tab_val <= off < tab_val + tab_size) or addend != 0:
            continue
        text_name = ".text." + syms[sym][0]
        if kind not in (_R_CUDA_DESCRIPTOR, _R_CUDA_64) or text_name not in names:
            continue
        struct.pack_into("<Q", data, rela[4] + 24 * k + 8, (sym << 32) | _R_CUDA_64)
        struct.pack_into("<Q", data, init[4] + off + 8, secs[names.index(text_name)][5])
        resolved += 1
    if not resolved:
        return _unresolved(cubin, "no function relocation")
    return bytes(data)


def _unresolved(cubin: bytes, reason: str) -> bytes:
    _warn_unresolved(reason)
    return cubin


@functools.cache
def _warn_unresolved(reason: str) -> None:
    log.warning(
        "code warm: cannot resolve the text table (%s); it loads nothing", reason
    )


def _wrap_stages(stages: dict[str, Any]) -> None:
    if "ptx" not in stages or "cubin" not in stages:
        return
    make_ptx, make_cubin = stages["ptx"], stages["cubin"]
    stages["ptx"] = lambda src, metadata: codewarm_ptx(make_ptx(src, metadata))
    stages["cubin"] = lambda src, metadata: codewarm_cubin(make_cubin(src, metadata))


def _install() -> None:
    # Triton calls the hook with no args for a cache key, else with
    # (backend, stages, options, language, capability); keep any earlier hook.
    prev: Any = triton.knobs.runtime.add_stages_inspection_hook
    if getattr(prev, "code_warm_key", None) == _KEY:
        return  # Installed already (a reload of this module).

    def hook(*args: object) -> tuple[str, str] | None:
        if not args:
            if prev is None:
                return _KEY, _KEY
            key, digest = prev()
            return key + _KEY, f"{digest}{_KEY}"
        if prev is not None:
            prev(*args)
        _wrap_stages(cast("dict[str, Any]", args[1]))
        return None

    hook.code_warm_key = _KEY  # pyrefly: ignore[missing-attribute]
    triton.knobs.runtime.add_stages_inspection_hook = hook  # pyrefly: ignore[bad-assignment]


_install()
