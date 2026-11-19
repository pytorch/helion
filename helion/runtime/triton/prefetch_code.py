"""Megakernel code prefetch.

A megakernel's native text outgrows the launch prefetch window, and the weights
it and the kernels between its launches stream evict that text from L2, so its
tasks first run their code from DRAM. Kernels that call ``prefetch_code()`` (what
``hl.prefetch_code()`` lowers to) prefetch their own text into L2 where the call
sits: at entry (before their PDL wait), or between two roots, so the roots after a
large weight stream find their code in L2.
Importing this module (a generated kernel calling ``prefetch_code()`` does)
registers a compiler stage hook: a call site is a call of a prefetch function the
PTX stage declares and defines, with a table of the kernel's entry and helper
functions it walks, and the cubin stage turns the table's function relocations
into text addresses and writes the text sizes (when it cannot, the sizes stay
zero, the loop loads nothing and a warning says so).  Kernels calling none are
unchanged; the hook adds this module's source hash to the cache key of the
kernels compiled after it in the process.
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
_KEY = "helion_prefetch_code_" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_R_CUDA_64 = 2
_R_CUDA_DESCRIPTOR = 35
log: logging.Logger = logging.getLogger(__name__)


_PC_TAB = "helion_prefetch_code_tab"
_PC_FN = "helion_prefetch_code_fn"  # _<W>: the prefetch by W warps
_CALL = re.compile(rf"call\.uni\s+{_PC_FN}_(\d+);")


def _call_asm(warps: int, first_warp: bool) -> str:
    """The call site: ``$1`` (the reader predicate, CTA uniform) calls the
    prefetch of ``warps`` warps (``_PC_FN``), only from the CTA's first warp if
    ``first_warp`` (the body of a kernel not warp-specialized there).  The asm is
    the call itself: wherever LLVM copies it (jump threading duplicates inline
    asm onto incoming paths), each copy calls."""
    lead = "mov.u32 %pcq, %tid.x; setp.lt.and.u32 %pcr, %pcq, 32, %pcr; "
    return (
        "{ .reg .pred %pcr; .reg .b32 %pcq; setp.ne.s32 %pcr, $1, 0; "
        f"{lead if first_warp else ''}@%pcr call.uni {_PC_FN}_{warps}; "
        "mov.u32 $0, 0; }"
    )


# The body's call site (the first warp of each reader CTA).
_BODY_CALL = tl.constexpr(_call_asm(1, first_warp=True))


@triton.jit
def prefetch_code(reader):  # noqa: ANN001, ANN201
    """Prefetch the calling kernel's native text into L2 here, by the first warp
    of each CTA ``reader`` holds."""
    tl.inline_asm_elementwise(
        _BODY_CALL,  # pyrefly: ignore [bad-argument-type]
        "=r,r",
        [reader.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def prefetch_code_here(warps: int, reader: str) -> str:
    """Device code prefetching the kernel's text where it sits, by the calling
    partition's ``warps`` warps (a warp-specialized kernel's idle partition), on
    the CTAs the device expression ``reader`` holds."""
    asm = _call_asm(warps, first_warp=False)
    return (
        f"tl.inline_asm_elementwise({asm!r}, '=r,r', [({reader}).to(tl.int32)], "
        "tl.int32, False, 1)"
    )


def prefetch_code_ptx(ptx: str) -> str:
    """Declare and define the prefetch functions a kernel's call sites call
    (``_call_asm``): a text-range table (a (base, bytes) pair per function, then
    a sink word) and, per partition width W, ``_PC_FN``_W: the calling warps
    (lanes ``%tid.x`` mod 32 W) load every line of the text with evict_last.  Each
    reader CTA reads every line: a two-die GPU's L2 keeps its own copy per die of
    what that die's SMs read, so a line one reader brought in serves the other
    die's SMs from afar.  The prefetch is its own function: inline, its loop costs
    ptxas its uniform-register analysis across the whole kernel (every kernel
    argument then moves to a uniform register at each use).  It must not use
    bar.sync 0: a warp-specialized partition calls it."""
    widths = sorted({int(w) for w in _CALL.findall(ptx)})
    if not widths:
        return ptx
    head = re.search(r"\.visible \.entry (\w+)\([^)]*\)", ptx, re.DOTALL)
    first = re.search(
        r"^(?:\.(?:visible|weak|extern)\s+)?\.(?:func|entry)\b", ptx, re.MULTILINE
    )
    assert first is not None  # the calls sit in a function
    declare = "".join(f".func {_PC_FN}_{w}();\n" for w in widths)
    if head is None:
        # No table to walk: the functions load nothing.
        _warn_unresolved("no kernel entry in the PTX")
        empty = "".join(f".func {_PC_FN}_{w}()\n{{\n\tret;\n}}\n" for w in widths)
        return f"{ptx[: first.start()]}{declare}{ptx[first.start() :]}\n{empty}"
    funcs = re.findall(
        r"^(?:\.(?:visible|weak)\s+)?\.func\s+(?:\([^)]*\)\s*)?([\w$]+)\s*\(",
        ptx,
        re.MULTILINE,
    )
    names = list(dict.fromkeys([head[1], *funcs]))
    slots = ", ".join(f"{n}, 0" for n in names)
    table = f".global .align 8 .u64 {_PC_TAB}[{2 * len(names) + 1}] = {{{slots}, 0}};"
    prefetch = "".join(_prefetch_function(w, len(names)) for w in widths)
    i = head.start()
    if first.start() < i:
        ptx = f"{ptx[: first.start()]}{declare}{ptx[first.start() :]}"
        i += len(declare)
        declare = ""
    return f"{ptx[:i]}{declare}{head[0]};\n{table}\n{ptx[i:]}\n{prefetch}"


def _prefetch_function(warps: int, functions: int) -> str:
    """``_PC_FN``_``warps``: its warps walk the table's ``functions`` text ranges,
    eight 128-byte lines in flight per lane, and fold their words into a store
    that never happens (it keeps the loads live; their L2 fill is the effect)."""
    lanes = 32 * warps
    prefetch = f"""\
.func {_PC_FN}_{warps}()
{{
\t{{
\t.reg .pred %pcp; .reg .b32 %pct, %pcnw, %pcw, %pcn, %pci, %pck, %pcacc;
\t.reg .b64 %pcbase, %pcbytes, %pcaddr, %pcpol;
\tmov.u32 %pct, %tid.x; and.b32 %pcw, %pct, {lanes - 1}; mov.u32 %pcnw, {lanes};
\tcreatepolicy.fractional.L2::evict_last.b64 %pcpol, 1.0; mov.u32 %pcacc, 0;
"""
    for i in range(functions):
        prefetch += f"""\
\tld.global.u64 %pcbase, [{_PC_TAB}+{16 * i}];
\tld.global.u64 %pcbytes, [{_PC_TAB}+{16 * i + 8}];
\tadd.u64 %pcbytes, %pcbytes, 127; shr.u64 %pcbytes, %pcbytes, 7;
\tcvt.u32.u64 %pcn, %pcbytes; mov.u32 %pci, %pcw;
PC_LOOP{i}:
\tsetp.ge.u32 %pcp, %pci, %pcn; @%pcp bra PC_NEXT{i};
"""
        for j in range(8):
            prefetch += f"""\
\t{{ .reg .b32 %pcv; .reg .pred %pcq; mad.lo.u32 %pck, %pcnw, {j}, %pci;
\tsetp.lt.u32 %pcq, %pck, %pcn; mul.wide.u32 %pcaddr, %pck, 128;
\tadd.u64 %pcaddr, %pcaddr, %pcbase; mov.u32 %pcv, 0;
\t@%pcq ld.global.cg.L2::cache_hint.u32 %pcv, [%pcaddr], %pcpol;
\txor.b32 %pcacc, %pcacc, %pcv; }}
"""
        prefetch += f"\tmad.lo.u32 %pci, %pcnw, 8, %pci; bra PC_LOOP{i};\nPC_NEXT{i}:\n"
    prefetch += f"""\
\tsetp.eq.u32 %pcp, %pcacc, 0x9E3779B9;
\t@%pcp st.global.u32 [{_PC_TAB}+{16 * functions}], %pcacc;
\t}}
\tret;
}}
"""
    return prefetch


def prefetch_code_cubin(cubin: bytes) -> bytes:
    """Retype the table's descriptor relocations to text PCs; write text sizes."""
    if _PC_TAB.encode() not in cubin:
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
    tabs = [sym for sym in syms if sym[0] == _PC_TAB]
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
        "prefetch code: cannot resolve the text table (%s); it loads nothing", reason
    )


def _wrap_stages(stages: dict[str, Any]) -> None:
    if "ptx" not in stages or "cubin" not in stages:
        return
    make_ptx, make_cubin = stages["ptx"], stages["cubin"]
    stages["ptx"] = lambda src, metadata: prefetch_code_ptx(make_ptx(src, metadata))
    stages["cubin"] = lambda src, metadata: prefetch_code_cubin(
        make_cubin(src, metadata)
    )


def _install() -> None:
    # Triton calls the hook with no args for a cache key, else with
    # (backend, stages, options, language, capability); keep any earlier hook.
    prev: Any = triton.knobs.runtime.add_stages_inspection_hook
    if getattr(prev, "prefetch_code_key", None) == _KEY:
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

    hook.prefetch_code_key = _KEY  # pyrefly: ignore[missing-attribute]
    triton.knobs.runtime.add_stages_inspection_hook = hook  # pyrefly: ignore[bad-assignment]


_install()
