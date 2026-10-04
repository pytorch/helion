"""Register budget for a warp-specialized epilogue partition.

Triton 3.7 requests 88 registers for every warp-specialized partition that holds
tensors and gives the rest of the register file to the default warp group, which
usually runs the epilogue. When the epilogue (the TMEM accumulator load) lands in
its own 4-warp partition instead, it spills while the default warps idle.
Importing this module registers a stage hook that hands such a partition the
budget the default warp group would get, up to 256, and the default the rest.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import tempfile
from typing import Any
from typing import cast

import triton
from triton._C.libtriton import ir  # pyrefly: ignore[missing-module-attribute]

# Compiled kernels are cached under a key that tracks this pass's source.
_KEY = "helion_ws_registers_" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_PAD_REGS = 16  # Triton's request for the partitions padding a warp group
_MAX_REGS = 256  # setmaxnreg ceiling
_REQUESTED = re.compile(
    r"(ttg\.warp_specialize\(.*requestedRegisters = array<i32: )([\d, ]+)>"
)
_HEADER = re.compile(
    r"^\s*(?:default|partition\d+\(.*\) num_warps\((\d+)\)) \{$", re.MULTILINE
)
_CLOSE = re.compile(r"^\s*\} : \(", re.MULTILINE)
_TMEM_LOAD = "ttng.tmem_load "


def _groups(
    warps: list[int], regs: list[int], base: int, groups: int
) -> list[list[int]]:
    # Partition indices per warp group, padded and placed like AllocateWarpGroups.
    pad = groups * 4 - sum(warps)
    while pad > 0:
        warps.append(1 << (pad.bit_length() - 1))
        regs.append(_PAD_REGS)
        pad -= warps[-1]
    found: list[list[int]] = []
    start = base
    for i in sorted(range(len(warps)), key=lambda i: -warps[i]):
        if start % 4 == 0:
            found.append([])
        found[-1].append(i)
        start += warps[i]
    return found


def _regions(segment: str) -> tuple[str, list[tuple[int, str]]]:
    # The default region's text and (num_warps, text) of each partition region.
    heads = list(_HEADER.finditer(segment))
    close = _CLOSE.search(segment, heads[-1].end())
    ends = [h.start() for h in heads[1:]] + [close.start() if close else len(segment)]
    texts = [segment[h.end() : e] for h, e in zip(heads, ends, strict=True)]
    return texts[0], [(int(h[1]), t) for h, t in zip(heads[1:], texts[1:], strict=True)]


def split_registers(ttgir: str) -> str:
    """Raise the request of a whole-warp-group epilogue partition."""
    ops = list(_REQUESTED.finditer(ttgir))
    num_warps = re.search(r'"ttg\.num-warps" = (\d+)', ttgir)
    if not ops or num_warps is None:
        return ttgir
    base = int(num_warps[1])
    ends = [m.start() for m in ops[1:]] + [len(ttgir)]
    regions = [_regions(ttgir[m.end() : end]) for m, end in zip(ops, ends, strict=True)]
    groups = math.ceil(max(sum(w for w, _ in parts) for _, parts in regions) / 4)
    total = base + groups * 4
    maxnreg = 64 * 1024 // total // 32 // 8 * 8
    out, pos = [], 0
    for m, (default, parts) in zip(ops, regions, strict=True):
        requested = [int(r) for r in m[2].split(",")]
        regs, sizes = list(requested), [w for w, _ in parts]
        epilogue, fixed = [], 0
        for group in _groups(sizes, regs, base, groups):
            first = group[0]
            if (
                len(group) == 1
                and first < len(parts)
                and sizes[first] % 4 == 0
                and _TMEM_LOAD in parts[first][1]
                and _TMEM_LOAD not in default
            ):
                epilogue.append(first)
            else:
                width = sum(sizes[i] for i in group)
                fixed += max(-(-regs[i] // 8) * 8 for i in group) * width
        if epilogue:
            # The default warp group keeps at least the epilogue's own request.
            floor = max(regs[i] for i in epilogue)
            width = sum(sizes[i] for i in epilogue)
            share = (maxnreg * total - fixed - base * floor) // width
            share = min(share // 8 * 8, _MAX_REGS)
            for i in epilogue:
                requested[i] = max(requested[i], share)
        out += [ttgir[pos : m.start(2)], ", ".join(map(str, requested))]
        pos = m.end(2)
    return "".join([*out, ttgir[pos:]])


def _wrap_stages(stages: dict[str, Any]) -> None:
    if "ttgir" not in stages:
        return
    make_ttgir = stages["ttgir"]

    def ttgir(src: object, metadata: dict[str, object]) -> object:
        mod = make_ttgir(src, metadata)
        text = str(mod)
        if "@_helion_" not in text or "requestedRegisters" not in text:
            return mod
        new = split_registers(text)
        if new == text:
            return mod
        with tempfile.NamedTemporaryFile("w", suffix=".ttgir") as f:
            f.write(new)
            f.flush()
            parsed = ir.parse_mlir_module(f.name, mod.context)
        parsed.context = mod.context
        return parsed

    stages["ttgir"] = ttgir


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
        backend, stages, options = cast("tuple[Any, dict[str, Any], Any]", args[:3])
        # A user maxnreg changes the budget AllocateWarpGroups splits.
        if backend.target.backend == "cuda" and options.maxnreg is None:
            _wrap_stages(stages)
        return None

    triton.knobs.runtime.add_stages_inspection_hook = hook  # pyrefly: ignore[bad-assignment]


_install()
