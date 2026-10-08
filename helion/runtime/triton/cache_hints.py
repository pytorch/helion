"""L2 cache-policy hints for tensor-descriptor (TMA) loads, and L2 prefetch.

Triton keeps ``eviction_policy`` on ``tt.descriptor_load`` but drops it when
lowering to TMA. Importing this module registers a compiler stage hook that adds
``.L2::cache_hint`` to the PTX TMA loads of every host descriptor whose loads
all agree on one policy. Hints only steer L2 replacement, never values. The
hook also makes kernels that call ``code_warm()`` (megakernels) L2-warm their own
native text at entry.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import struct
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import triton
import triton.language as tl
from triton.language import core
from triton.language.extra.cuda.utils import num_threads

if TYPE_CHECKING:
    from triton.language.semantic import TritonSemantic

# Compiled kernels are cached under a key that tracks this pass's source.
_KEY = "helion_cache_hints_" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_POLICIES = ("evict_first", "evict_last")
_R_CUDA_64 = 2
_R_CUDA_DESCRIPTOR = 35
_NAME = r"%[\w$.-]+"
_SELECT = re.compile(
    rf"({_NAME}) = arith\.select {_NAME}, ({_NAME}), ({_NAME}) : !tt\.tensordesc"
)
_LOAD = re.compile(rf"tt\.descriptor_load ({_NAME})\[[^\]]*\]([^:]*):")
_DEF = re.compile(
    r"^\s*(?:@!?%p\d+ )?([a-z][\w.:]*)\s+(%rd\d+), ([^;]*);", re.MULTILINE
)
_TMA = re.compile(
    r"^(\s*)(@!?%p\d+ )?(cp\.async\.bulk\.tensor\.\dd\.shared::\w+\.global"
    r"\.mbarrier::complete_tx::bytes) (\[[^\]]*\], \[(%rd\d+), \{[^}]*\}\], "
    r"\[[^\]]*\]);$",
    re.MULTILINE,
)


@core.builtin
def descriptor_load(
    desc: core.tensor_descriptor_base,
    offsets: list[core.tensor],
    eviction_policy: core.constexpr,
    _semantic: TritonSemantic | None = None,
) -> core.tensor:
    """``desc.load(offsets)`` that keeps ``eviction_policy`` in the TTIR."""
    assert _semantic is not None
    return _semantic.descriptor_load(desc, offsets, "", str(eviction_policy.value))


@triton.jit
def prefetch_l2(base, offset: tl.constexpr, nbytes: tl.constexpr):  # noqa: ANN001, ANN201
    """One thread bulk-prefetches ``nbytes`` at byte ``offset`` past ``base`` into L2."""
    lane = tl.arange(0, num_threads())
    tl.inline_asm_elementwise(
        "{ .reg .pred %p; setp.eq.s32 %p, $2, 0; "
        f"@%p cp.async.bulk.prefetch.L2.global [$1], {nbytes}; mov.u32 $0, 0; }}",
        "=r,l,r",
        [base.to(tl.int64) + offset, lane],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def _arg_sources(
    name: str, args: dict[str, int], selects: dict[str, tuple[str, str]]
) -> set[int]:
    if name in args:
        return {args[name]}
    return set().union(*(_arg_sources(n, args, selects) for n in selects.get(name, ())))


def descriptor_policies(ttir: str) -> dict[int, str]:
    """Map kernel argument index to the policy every load through it uses."""
    start = ttir.index("tt.func public")
    end = ttir.find("tt.func", start + 1)
    body = ttir[start : end if end >= 0 else len(ttir)]
    header = body[: body.index("\n")]
    args = {name: i for i, name in enumerate(re.findall(rf"({_NAME}): ", header))}
    selects = {m[1]: (m[2], m[3]) for m in _SELECT.finditer(body)}
    seen: dict[int, set[str]] = {}
    for m in _LOAD.finditer(body):
        policy = re.search(r"evictionPolicy = (\w+)", m[2])
        for arg in _arg_sources(m[1], args, selects):
            seen.setdefault(arg, set()).add(policy[1] if policy else "")
    return {
        arg: next(iter(found))
        for arg, found in seen.items()
        if len(found) == 1 and next(iter(found)) in _POLICIES
    }


def _param_sources(reg: str, defs: dict[str, list[str] | None]) -> set[int]:
    # Kernel params a descriptor register may hold, through mov/cvta copies.
    found, todo, done = set(), [reg], set()
    while todo:
        name = todo.pop()
        if name in done:
            continue
        done.add(name)
        if m := re.fullmatch(r"\w+_param_(\d+)", name):
            found.add(int(m[1]))
            continue
        sources = defs.get(name)
        if sources is None:
            return set()
        todo += sources
    return found


def hint_tma_loads(ptx: str, policies: dict[int, str]) -> str:
    """Add an L2 cache hint to entry TMA loads whose descriptor params agree."""
    entry = re.search(r"\.visible \.entry (\w+)\(", ptx)
    if not policies or entry is None:
        return ptx
    # Inline asm braces sit at column 0, so the body ends at the next function.
    start = entry.start()
    after = re.compile(r"^\.(?:visible|weak|extern|func|entry)\b", re.MULTILINE)
    end = m.start() if (m := after.search(ptx, entry.end())) else len(ptx)
    body = ptx[start:end]
    defs: dict[str, list[str] | None] = {}
    for m in _DEF.finditer(body):
        srcs = m[3].split(", ")[: 2 if m[1] == "selp.b64" else None]
        copy = m[1] in ("mov.b64", "mov.u64", "cvta.param.u64", "selp.b64")
        if copy and all(re.fullmatch(rf"%rd\d+|{entry[1]}_param_\d+", s) for s in srcs):
            if (sources := defs.setdefault(m[2], [])) is not None:
                sources += srcs
        else:
            defs[m[2]] = None

    def hint(m: re.Match[str]) -> str:
        params = _param_sources(m[5], defs)
        found = {policies.get(p) for p in params}
        if not params or len(found) != 1 or None in found:
            return m[0]
        return (
            f"{m[1]}{{ .reg .b64 %hint; createpolicy.fractional.L2::{found.pop()}"
            f".b64 %hint, 1.0; {m[2] or ''}{m[3]}.L2::cache_hint {m[4]}, %hint; }}"
        )

    return ptx[:start] + _TMA.sub(hint, body) + ptx[end:]


# Code warm: a kernel L2-warms its own native text at entry. Text past the launch
# prefetch window is otherwise fetched from DRAM on first use in every launch.
_CW_TAB = "helion_codewarm_tab"
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


def codewarm_ptx(ptx: str) -> str:
    """Declare the text-range table and emit the entry warm (one warp, CTAs with
    blockIdx % 16 == 0 share the lines, evict_last loads)."""
    entry = re.search(r"\.visible \.entry (\w+)\(", ptx)
    if entry is None or _CW_MARK not in ptx:
        return ptx
    head = re.search(r"\.visible \.entry \w+\([^)]*\)", ptx, re.DOTALL)
    body = re.search(
        r"\.visible \.entry \w+\([^)]*\)\s*(?:\.\w+[^{]*)?\{", ptx, re.DOTALL
    )
    if head is None or body is None:
        return ptx
    funcs = list(
        dict.fromkeys(
            re.findall(r"^\.func (?:\([^)]*\) )?([\w$]+)\(", ptx, re.MULTILINE)
        )
    )
    names = [entry[1], *funcs]
    slots = ", ".join(f"{n}, 0" for n in names)
    table = f".global .align 8 .u64 {_CW_TAB}[{2 * len(names)}] = {{{slots}}};"
    lines = [
        "\t{",
        (
            "\t.reg .pred %cwp; "
            ".reg .b32 %cwt, %cwc, %cwnw, %cww, %cwn, %cwi, %cwk, %cwacc, %cwx;"
        ),
        "\t.reg .b64 %cwbase, %cwbytes, %cwaddr, %cwpol;",
        "\tmov.u32 %cwt, %tid.x; setp.ge.u32 %cwp, %cwt, 32; @%cwp bra CW_DONE;",
        "\tmov.u32 %cwc, %ctaid.x; and.b32 %cwx, %cwc, 15; setp.ne.u32 %cwp, %cwx, 0;",
        "\t@%cwp bra CW_DONE;",
        (
            "\tmov.u32 %cwnw, %nctaid.x; "
            "add.u32 %cwnw, %cwnw, 15; shr.u32 %cwnw, %cwnw, 4;"
        ),
        (
            "\tshr.u32 %cww, %cwc, 4; "
            "mad.lo.u32 %cww, %cww, 32, %cwt; shl.b32 %cwnw, %cwnw, 5;"
        ),
        "\tcreatepolicy.fractional.L2::evict_last.b64 %cwpol, 1.0; mov.u32 %cwacc, 0;",
    ]
    for s_i in range(len(names)):
        lines += [
            f"\tld.global.u64 %cwbase, [{_CW_TAB}+{16 * s_i}];",
            f"\tld.global.u64 %cwbytes, [{_CW_TAB}+{16 * s_i + 8}];",
            "\tadd.u64 %cwbytes, %cwbytes, 127; shr.u64 %cwbytes, %cwbytes, 7;",
            "\tcvt.u32.u64 %cwn, %cwbytes; mov.u32 %cwi, %cww;",
            f"CW_LOOP{s_i}:",
            "\tsetp.ge.u32 %cwp, %cwi, %cwn; @%cwp bra CW_NEXT" + str(s_i) + ";",
        ]
        # Eight lines in flight per lane, then fold their words.
        for j in range(8):
            lines += [
                (
                    "\t{ .reg .b32 %cwv; .reg .pred %cwq; "
                    f"mad.lo.u32 %cwk, %cwnw, {j}, %cwi;"
                ),
                "\tsetp.lt.u32 %cwq, %cwk, %cwn; mul.wide.u32 %cwaddr, %cwk, 128;",
                "\tadd.u64 %cwaddr, %cwaddr, %cwbase; mov.u32 %cwv, 0;",
                "\t@%cwq ld.global.cg.L2::cache_hint.u32 %cwv, [%cwaddr], %cwpol;",
                "\txor.b32 %cwacc, %cwacc, %cwv; }",
            ]
        lines += [
            "\tmad.lo.u32 %cwi, %cwnw, 8, %cwi; bra CW_LOOP" + str(s_i) + ";",
            f"CW_NEXT{s_i}:",
        ]
    lines += [
        # Impossible store keeps the loads live; their L2 fill is the only effect.
        "\tsetp.eq.u32 %cwp, %cwacc, 0x9E3779B9;",
        f"\t@%cwp st.global.u32 [{_CW_TAB}+4], %cwacc;",
        "CW_DONE:",
        "\t}",
    ]
    i, j = head.start(), body.end()
    warm = "\n".join(lines)
    return f"{ptx[:i]}{head[0]};\n{table}\n{ptx[i:j]}\n{warm}\n{ptx[j:]}"


def codewarm_cubin(cubin: bytes) -> bytes:
    """Retype the table's descriptor relocations to text PCs; write text sizes."""
    data = bytearray(cubin)
    if data[:4] != b"\x7fELF":
        return cubin
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
        return cubin
    symtab = secs[names.index(".symtab")]
    sym_str = secs[symtab[6]][4]
    syms = []
    for off in range(symtab[4], symtab[4] + symtab[5], 24):
        n, _info, _other, shndx, value, _size = struct.unpack_from("<IBBHQQ", data, off)
        syms.append((name_at(n, sym_str), shndx, value))
    tabs = [sym for sym in syms if sym[0] == _CW_TAB]
    if not tabs:
        return cubin
    _, tab_sec, tab_val = tabs[0]
    init = secs[tab_sec]
    rela = secs[names.index(".rela.nv.global.init")]
    for k in range(rela[5] // 24):
        off, info, addend = struct.unpack_from("<QQq", data, rela[4] + 24 * k)
        sym, kind = info >> 32, info & 0xFFFFFFFF
        if not (tab_val <= off < tab_val + init[5]) or addend != 0:
            continue
        text_name = ".text." + syms[sym][0]
        if kind not in (_R_CUDA_DESCRIPTOR, _R_CUDA_64) or text_name not in names:
            continue
        struct.pack_into("<Q", data, rela[4] + 24 * k + 8, (sym << 32) | _R_CUDA_64)
        struct.pack_into("<Q", data, init[4] + off + 8, secs[names.index(text_name)][5])
    return bytes(data)


def _wrap_stages(stages: dict[str, Any]) -> None:
    if "ttir" not in stages or "ptx" not in stages:
        return
    make_ttir, make_ptx = stages["ttir"], stages["ptx"]
    policies: dict[int, str] = {}

    def ttir(src: object, metadata: dict[str, object]) -> object:
        mod = make_ttir(src, metadata)
        text = str(mod)
        if "evictionPolicy = evict_" in text:
            policies.update(descriptor_policies(text))
        return mod

    def ptx(src: object, metadata: dict[str, object]) -> str:
        return codewarm_ptx(hint_tma_loads(make_ptx(src, metadata), policies))

    stages["ttir"], stages["ptx"] = ttir, ptx
    if "cubin" in stages:
        make_cubin = stages["cubin"]
        stages["cubin"] = lambda src, metadata: codewarm_cubin(
            make_cubin(src, metadata)
        )


def _install() -> None:
    # Triton calls the hook with no args for a cache key, else with
    # (backend, stages, options, language, capability); keep any earlier hook.
    prev: Any = triton.knobs.runtime.add_stages_inspection_hook

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

    triton.knobs.runtime.add_stages_inspection_hook = hook  # pyrefly: ignore[bad-assignment]


_install()
