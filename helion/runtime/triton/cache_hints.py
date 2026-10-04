"""L2 cache-policy hints for tensor-descriptor (TMA) loads, and L2 prefetch.

Triton keeps ``eviction_policy`` on ``tt.descriptor_load`` but drops it when
lowering to TMA. Importing this module registers a compiler stage hook that adds
``.L2::cache_hint`` to the PTX TMA loads of every host descriptor whose loads
all agree on one policy. Hints only steer L2 replacement, never values.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
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
        return hint_tma_loads(make_ptx(src, metadata), policies)

    stages["ttir"], stages["ptx"] = ttir, ptx


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
