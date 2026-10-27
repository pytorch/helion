"""Race regression tests for multi-root (cross-loop) kernel schedules.

Each GPU case runs a kernel whose producer is 20 us late (a ``%globaltimer``
spin) on memory poisoned before every launch, so an access that the schedule
does not order after its producer shows up as a wrong result, and checks a
negative control that drops the ordering under test: it must race.  The cases
come from adversarial reviews of the schedule (transitive chains and WAR
behind pruned root-barrier edges, storage aliasing through views, exported
code, symbolic layouts and separately constructed storages over one buffer,
transitively reduced producers, warp-specialized inner loops).
"""

from __future__ import annotations

import contextlib
from functools import partial
import re
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from unittest import mock

import torch

import helion
from helion._compiler import cross_loop_codegen
from helion._compiler import input_storage_aliasing
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.runtime.triton import launcher
from helion.runtime.triton import ttgir_weight_prefetch

if TYPE_CHECKING:
    import ast

_SPIN_20US = """
t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
t1 = t0
while t1 - t0 < 20000:
    t1 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
0
"""


def chain3(
    x: torch.Tensor, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """A chain of three producers, each reading all of the previous one's
    output; the dot root reads the first one's ``a`` (written late) before
    the last one's ``c``: its edges from the first two are pruned."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = x[t, :] + torch.sum(a[:, :], dim=0)
    for t in hl.tile(tokens, block_size=1):
        c[t, :] = x[t, :] + torch.sum(b[:, :], dim=0)
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, va.permute(1, 0), out_dtype=torch.float32)
        vc = hl.load(c, [rows, hl.arange(k)], extra_mask=live[:, None])
        acc = hl.dot(wt, vc.permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


def war_trans(
    x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """The dot root overwrites ``s``, which the producer (late) of the
    producer it waits for reads: the WAR edge is pruned behind that one."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = s[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = x[t, :] + torch.sum(a[:, :], dim=0)
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        wt = w[n_t, :]
        hl.store(
            s,
            [rows, n_t.begin + hl.arange(16)],
            hl.full([16, 16], 100.0, dtype=s.dtype),
            extra_mask=live[:, None],
        )
        vb = hl.load(b, [rows, hl.arange(k)], extra_mask=live[:, None])
        out[n_t, :] = hl.dot(wt, vb.permute(1, 0), out_dtype=torch.float32)
    return out


def war_view(
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    s_v: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """The dot root overwrites ``s_v`` (the caller passes a view of ``s``),
    which its producer (late) reads as ``s``."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = s[t, :] + 1
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        wt = w[n_t, :]
        hl.store(
            s_v,
            [rows, n_t.begin + hl.arange(16)],
            hl.full([16, 16], 100.0, dtype=s_v.dtype),
            extra_mask=live[:, None],
        )
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        out[n_t, :] = hl.dot(wt, va.permute(1, 0), out_dtype=torch.float32)
    return out


def trans_view(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    a_v: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """``a`` (late) reaches the dot root through a producer that reads all of
    it as ``a_v`` (the caller passes a view of ``a``)."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = x[t, :] + torch.sum(a_v[:, :], dim=0)
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, va.permute(1, 0), out_dtype=torch.float32)
        vb = hl.load(b, [rows, hl.arange(k)], extra_mask=live[:, None])
        acc = hl.dot(wt, vb.permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


def alias_view(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    a_v: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """The dot root reads ``b``, then ``a`` (written late) through ``a_v``,
    which the caller may pass over ``a``'s memory."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = x[t, :] * 2
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        vb = hl.load(b, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, vb.permute(1, 0), out_dtype=torch.float32)
        va = hl.load(a_v, [rows, hl.arange(k)], extra_mask=live[:, None])
        acc = hl.dot(wt, va.permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


def two_root_alias(x: torch.Tensor, a: torch.Tensor, a_r: torch.Tensor) -> torch.Tensor:
    """Writes ``a`` (late); the next root reads ``a_r``, which the caller may
    pass as a view of ``a``: with distinct storages the roots are
    independent."""
    tokens, k = x.size()
    rows = a_r.size(0)
    out = torch.empty([rows, a_r.size(1)], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for r in hl.tile(rows, block_size=1):
        out[r, :] = a_r[r, :].to(torch.float32)
    return out


def barrier_alias(
    x: torch.Tensor, a: torch.Tensor, a_r: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Barrier phases: writes ``b``; reads it to write ``a`` (late); reads
    ``a_r``, which with distinct storages shares the second phase."""
    tokens, k = x.size()
    rows = a_r.size(0)
    out = torch.empty([rows, a_r.size(1)], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = x[t, :] * 2
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = b[t, :] + 1
    for r in hl.tile(rows, block_size=1):
        out[r, :] = a_r[r, :].to(torch.float32)
    return out


def two_same(x: torch.Tensor, a: torch.Tensor, a_v: torch.Tensor) -> torch.Tensor:
    """Writes ``a`` (late); the next root reads ``a_v`` (same shape)."""
    tokens, k = x.size()
    out = torch.empty([tokens, k], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        out[t, :] = a_v[t, :].to(torch.float32)
    return out


def barrier_same(
    x: torch.Tensor, a: torch.Tensor, a_v: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Writes ``b``; reads it to write ``a`` (late); reads ``a_v``."""
    tokens, k = x.size()
    out = torch.empty([tokens, k], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = x[t, :] * 2
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = b[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        out[t, :] = a_v[t, :].to(torch.float32)
    return out


def two_readers(
    x: torch.Tensor, a: torch.Tensor, a_v: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Two roots that only read ``a`` and ``a_v``: independent however the
    two alias."""
    tokens, k = x.size()
    out = torch.empty([tokens, k], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = a[t, :] * 2
    for t in hl.tile(tokens, block_size=1):
        out[t, :] = a_v[t, :].to(torch.float32) + x[t, :]
    return out


def transitive_gather(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    idx1: torch.Tensor,
    b: torch.Tensor,
    idx2: torch.Tensor,
    idx3: torch.Tensor,
    w2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """W (a slow GEMM) writes ``a``; Q gathers rows of ``a`` into ``b``; A
    gathers rows of ``a``, then of ``b``.  The edge W -> A is reduced away
    (W -> Q -> A): A's load of ``a`` must still follow a wait implying W."""
    for i in hl.tile(a.size(0), block_size=16):
        acc = hl.zeros([i, 64], dtype=torch.float32)
        for k in hl.tile(x.size(1), block_size=128):
            acc = hl.dot(x[i, k], w[k, :], acc=acc)
        a[i, :] = acc.to(torch.bfloat16)
    for j in hl.tile(b.size(0), block_size=1):
        r = idx1[j.begin, hl.arange(16)]
        g = hl.load(a, [r[:, None], hl.arange(64)[None, :]])
        b[j.begin, :] = (g.to(torch.float32).sum(0) + 1).to(torch.bfloat16)
    for c in hl.tile(out.size(0), block_size=1):
        ra_rows = idx2[c.begin, hl.arange(16)]
        ra = hl.load(a, [ra_rows[:, None], hl.arange(64)[None, :]])
        acc2 = hl.dot(ra, w2[:, :], out_dtype=torch.float32)
        rb_rows = idx3[c.begin, hl.arange(16)]
        rb = hl.load(b, [rb_rows[:, None], hl.arange(64)[None, :]])
        out[c.begin, :, :] = hl.dot(rb, w2[:, :], acc=acc2)
    return out


def transitive_gather2(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    idx1: torch.Tensor,
    b: torch.Tensor,
    xv: torch.Tensor,
    v: torch.Tensor,
    idx2: torch.Tensor,
    idx3: torch.Tensor,
    idx4: torch.Tensor,
    w2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """:func:`transitive_gather` plus an independent producer V of ``v``
    that A gathers last, so A waits per producer (Q and V)."""
    for i in hl.tile(a.size(0), block_size=16):
        acc = hl.zeros([i, 64], dtype=torch.float32)
        for k in hl.tile(x.size(1), block_size=128):
            acc = hl.dot(x[i, k], w[k, :], acc=acc)
        a[i, :] = acc.to(torch.bfloat16)
    for j in hl.tile(b.size(0), block_size=1):
        r = idx1[j.begin, hl.arange(16)]
        g = hl.load(a, [r[:, None], hl.arange(64)[None, :]])
        b[j.begin, :] = (g.to(torch.float32).sum(0) + 1).to(torch.bfloat16)
    for m in hl.tile(v.size(0), block_size=1):
        v[m.begin, :] = (xv[m.begin, :].to(torch.float32) * 2).to(torch.bfloat16)
    for c in hl.tile(out.size(0), block_size=1):
        ra_rows = idx2[c.begin, hl.arange(16)]
        ra = hl.load(a, [ra_rows[:, None], hl.arange(64)[None, :]])
        acc2 = hl.dot(ra, w2[:, :], out_dtype=torch.float32)
        rb_rows = idx3[c.begin, hl.arange(16)]
        rb = hl.load(b, [rb_rows[:, None], hl.arange(64)[None, :]])
        acc3 = hl.dot(rb, w2[:, :], acc=acc2)
        rv_rows = idx4[c.begin, hl.arange(16)]
        rv = hl.load(v, [rv_rows[:, None], hl.arange(64)[None, :]])
        out[c.begin, :, :] = hl.dot(rv, w2[:, :], acc=acc3)
    return out


def ws_inner_gather(x: torch.Tensor, w: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Writes ``a`` (late); the next root's single-trip tasks read all of it
    in a K loop that may be warp-specialized."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        acc = hl.zeros([n_t, 16], dtype=torch.float32)
        for k_t in hl.tile(k, block_size=64):
            va = hl.load(a, [rows, k_t.index], extra_mask=live[:, None])
            acc = hl.dot(w[n_t, k_t], va.permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


def tma_alias(x: torch.Tensor, w: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Writes ``a`` through pointers (late); the next root reads ``a`` and the
    weight ``w`` in a pipelined K loop, by TMA descriptors when indexing says
    so: ``a`` then reaches the kernel twice, as a pointer and a descriptor."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, tokens], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for n_t in hl.tile(n, block_size=64):
        acc = hl.zeros([n_t, tokens], dtype=torch.float32)
        for k_t in hl.tile(k, block_size=64):
            acc = hl.dot(w[n_t, k_t], a[:, k_t].permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


_STATIC: dict[str, Any] = {
    "block_sizes": [],
    "cross_loop_pipeline": "static",
    "pid_type": "persistent_blocked",
    "num_warps": 4,
    "num_stages": 1,
    "num_sm_multiplier": 1,
    "indexing": "pointer",
}
_BARRIER: dict[str, Any] = {**_STATIC, "cross_loop_pipeline": "barrier"}
_PLAIN: dict[str, Any] = {
    key: value for key, value in _STATIC.items() if key != "cross_loop_pipeline"
}


def _padded_rows(t: torch.Tensor) -> torch.Tensor:
    rows = torch.zeros(16, t.size(1), device=t.device, dtype=torch.float32)
    rows[: t.size(0)] = t.float()
    return rows


def _close(out: torch.Tensor, expected: torch.Tensor) -> bool:
    return not bool(torch.isnan(out).any()) and torch.allclose(
        out, expected, rtol=2e-2, atol=2e-2
    )


def _bad_launches(
    launch: Callable[[], object],
    poison: Callable[[], None],
    good: Callable[[Any], bool],
    launches: int = 20,
) -> int:
    """Launches whose result shows an access that ran before its producer
    finished (``poison`` resets the producers' memory before each)."""
    bad = 0
    for _ in range(launches):
        poison()
        out = launch()
        torch.cuda.synchronize()
        bad += not good(out)
    return bad


def _nan(*tensors: torch.Tensor) -> Callable[[], None]:
    def poison() -> None:
        for t in tensors:
            t.fill_(float("nan"))

    return poison


def _without_names(dropped: str) -> Callable[..., tuple[int, ...]]:
    """``_insert_producer_waits`` that ignores ``dropped`` as a use of any
    producer's memory (a negative control)."""
    original = cross_loop_codegen._insert_producer_waits

    def insert(
        statements: list[ast.stmt],
        names: tuple[frozenset[str] | None, ...],
        make_wait: Any,
    ) -> tuple[int, ...]:
        return original(
            statements,
            tuple(None if n is None else (n - {dropped}) or None for n in names),
            make_wait,
        )

    return insert


def _initial_polls(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line, indent, counter) of every root-barrier poll outside its spin."""
    pattern = re.compile(
        r"^(\s*)tile_dependency_root_barrier_wait(?:_\d+)? = tl\.inline_asm_elementwise"
        r"\(asm='ld\.(?:acquire|relaxed)\.gpu\.global\.u32 \$0, \[\$1\];', "
        r"constraints='=r,l', "
        r"args=\[([^\]]+)\]"
    )
    polls = []
    spin: int | None = None
    for index, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        if spin is not None and indent > spin:
            continue
        spin = indent if line.lstrip().startswith("while ") else None
        match = pattern.match(line)
        if match:
            polls.append((index, indent, match.group(2)))
    return polls


def _first(lines: list[str], prefix: str, after: int = -1) -> int:
    return next(
        i
        for i, line in enumerate(lines)
        if i > after and line.lstrip().startswith(prefix)
    )


def _task_loop_before(lines: list[str], index: int) -> int:
    return max(
        i for i, line in enumerate(lines[:index]) if "for virtual_pid in" in line
    )


def _polled_between(lines: list[str], start: int, stop: int) -> bool:
    return any(start < poll[0] < stop for poll in _initial_polls(lines))


def _symbolic_fakes(
    real: tuple[torch.Tensor, ...],
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    """Outer-trace fake tensors with symbolic layouts for ``real`` (as a
    ``torch.compile(dynamic=True)`` trace hands them to Helion); a view of
    another argument stays a view of its fake."""
    from torch._dynamo.source import LocalSource
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import DimDynamic
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.fx.experimental.symbolic_shapes import StatelessSymbolicContext

    mode = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
    bases = {}
    for index, tensor in enumerate(real):
        if tensor._base is None:
            context = StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC] * tensor.dim()
            )
            bases[id(tensor)] = mode.from_tensor(
                tensor.to("meta"),
                source=LocalSource(f"arg{index}"),
                symbolic_context=context,
            )
    with mode:
        fakes = {key: fake.to(DEVICE) for key, fake in bases.items()}
        args = tuple(
            fakes[id(t)]
            if t._base is None
            else fakes[id(t._base)].view(fakes[id(t._base)].size())
            for t in real
        )
    return mode, args


def _edges(bound: Any) -> set[tuple[int, int]]:
    graph = bound.host_function.device_ir.tile_dependency_graph
    assert graph is not None
    return {(edge.producer_root, edge.consumer_root) for edge in graph.edges}


def _early_descriptor_copies(ttgir: str, names: tuple[str, ...]) -> dict[str, int]:
    """TMA copies of each named descriptor that read ahead of a readiness poll:
    copies hoisted above the entry's first ``ld.acquire`` (outside any
    warp_specialize partition), plus copies a warp_specialize producer
    partition issues before any poll of its own (the partition runs ahead of
    the consumer's poll).  A partition sees a descriptor as the block argument
    at the descriptor's position in the warp_specialize capture list."""
    body = ttgir[ttgir.index("tt.func public") :]
    found = dict.fromkeys(names, 0)

    def tma_of(lines: list[str], operand: str) -> int:
        return sum(
            1
            for line in lines
            if "async_tma_copy_global_to_local" in line
            and re.search(rf"{re.escape(operand)}\[", line)
        )

    head = (
        body.split("ttg.warp_specialize(", 1)[0]
        if "ttg.warp_specialize(" in body
        else body
    )
    before = head.partition("ld.acquire")[0].splitlines()
    for n in names:
        found[n] += sum(
            1
            for line in before
            if "async_tma_copy_global_to_local" in line
            and re.search(rf"%{n}(_\d+)?\[", line)
        )
    m = re.search(r"ttg\.warp_specialize\(([^)]*)\)", body)
    if m is None:
        return found
    captures = [c.strip() for c in m.group(1).split(",")]
    for part in re.finditer(r"partition\d+\((.*?)\)\s*num_warps\(", body):
        block_args = re.findall(r"(%[\w.$-]+)\s*:", part.group(1))
        text = body[part.end() :]
        nxt = re.search(r"\n\s*(partition\d+\(|\} : \()", text)
        early = (
            (text[: nxt.start()] if nxt else text)
            .partition("ld.acquire")[0]
            .splitlines()
        )
        for n in names:
            for k, cap in enumerate(captures):
                if re.fullmatch(rf"%{n}(_\d+)?", cap) and k < len(block_args):
                    found[n] += tma_of(early, block_args[k])
    return found


class TestStorageClasses(TestCase):
    """CPU: separately constructed storages over one buffer are one class."""

    def test_overlapping_storages_share_a_class(self) -> None:
        buf = torch.zeros(16, 64)
        inner = torch.from_dlpack(buf[4:12])  # own storage inside buf's
        head = torch.from_dlpack(buf[0:6])  # own storage at buf's base
        tail = torch.from_dlpack(buf[12:16])  # ends where buf ends
        other = torch.zeros(4)
        partition = input_storage_aliasing.input_storage_partition
        # inner and tail only touch: disjoint.
        self.assertEqual(partition([inner, tail, other]), (0, 1, 2))
        self.assertEqual(partition([inner, head, tail, other]), (0, 0, 1, 2))
        # buf overlaps all three: one class, through it.
        self.assertEqual(partition([tail, inner, buf, other, 3]), (0, 0, 0, 1, -1))
        self.assertEqual(partition([buf[0:0], torch.empty(0)]), (0, -1))

    def test_launcher_merges_overlapping_storages(self) -> None:
        buf = torch.zeros(16, 64)
        inner = torch.from_dlpack(buf[4:12])
        tail = torch.from_dlpack(buf[12:16])
        other = torch.zeros(4)
        self.assertEqual(
            launcher._shared_storages((tail, inner, buf, other)), ((0, 1, 2),)
        )
        self.assertEqual(
            launcher._shared_storages((inner, other, tail, buf[1:])), ((0, 2, 3),)
        )
        self.assertTrue(launcher._distinct_storages((inner, tail, other)))
        self.assertFalse(launcher._distinct_storages((inner, torch.from_dlpack(buf))))

    def test_pass_compiles_for_the_storage_classes_the_compiler_saw(self) -> None:
        # A tensor passed as a pointer and as a TMA descriptor (which views
        # its base tensor) is one class: the launcher runs the pass with the
        # classes' argument names when the launch's sharing stays within the
        # compile-time classes, and without the pass otherwise.
        x, w = torch.zeros(8, 64), torch.zeros(4, 64)
        desc = SimpleNamespace(base=x)
        kernel = mock.Mock(arg_names=["x", "w", "x_desc"])
        kernel.run.return_value = "ran"
        classes = launcher.prefetch_alias_classes
        self.assertEqual(classes(kernel, (x, w, desc), ((0, 2),)), (("x", "x_desc"),))
        # Classes the launch does not share only make the pass write more.
        apart = SimpleNamespace(base=torch.zeros(8, 64))
        self.assertEqual(classes(kernel, (x, w, apart), ((0, 2),)), (("x", "x_desc"),))
        self.assertEqual(classes(kernel, (x, w, apart), ()), ())
        # Sharing no compile-time class covers: no pass.
        self.assertIsNone(classes(kernel, (x, w, desc), ()))
        self.assertIsNone(classes(kernel, (x, x[1:], desc), ((0, 2),)))
        self.assertIsNone(classes(kernel, (x, w, desc), ((0, 1),)))
        self.assertIsNone(classes(object(), (x, w, desc), ((0, 2),)))

        seen: list[object] = []

        def enabled(
            alias_classes: object = (),
        ) -> contextlib.AbstractContextManager[None]:
            seen.append(alias_classes)
            return contextlib.nullcontext()

        def launch(*args: object, aliases: tuple[tuple[int, ...], ...]) -> object:
            return helion.runtime.default_launcher(
                kernel,
                (1,),
                *args,
                num_warps=4,
                num_stages=1,
                _ttgir_weight_prefetch=True,
                _tensor_storage_aliases=aliases,
            )

        with mock.patch.object(ttgir_weight_prefetch, "enabled", enabled):
            self.assertEqual(launch(x, w, desc, aliases=((0, 2),)), "ran")
            self.assertEqual(seen, [(("x", "x_desc"),)])
            self.assertEqual(launch(x, w, desc, aliases=()), "ran")
            self.assertEqual(launch(x, w, apart, aliases=()), "ran")
            self.assertEqual(seen, [(("x", "x_desc"),), ()])

    def test_fused_roots_without_launch_state_check_storage_sharing(self) -> None:
        x = torch.zeros(8, 4)
        kernel = mock.Mock(arg_names=["x", "x_tail"])
        # The in-process launcher checks before it touches the kernel.
        with self.assertRaises(helion.exc.CrossLoopAliasedLaunch):
            helion.runtime.default_launcher(
                kernel,
                (1,),
                x,
                torch.from_dlpack(x[1:]),
                num_warps=4,
                num_stages=1,
                _check_storage_aliasing=True,
            )
        kernel.run.assert_not_called()


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan(
    (10, 0), reason="only sm100 keeps dot roots inline in the task loop"
)
class TestCrossLoopRaces(TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)
        self.x = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        self.w = (0.1 * torch.randn(64, 64, device=DEVICE)).bfloat16()

    def _fit(self, a: torch.Tensor, *more: torch.Tensor) -> torch.Tensor:
        expected = self.w.float() @ _padded_rows(a).T
        for t in more:
            expected = expected + self.w.float() @ _padded_rows(t).T
        return expected

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_wait_covers_a_chain_of_pruned_producers(self) -> None:
        x, w = self.x, self.w
        a, b, c = (torch.empty_like(x) for _ in range(3))
        config = helion.Config(**_STATIC)
        args = (x, w, a, b, c)
        poison = _nan(a, b, c)

        def good(out: torch.Tensor) -> bool:
            return _close(out, self._fit(a, c))

        kernel = helion.kernel(chain3, config=config, static_shapes=True)
        self.assertEqual(_bad_launches(partial(kernel, *args), poison, good), 0)
        lines = kernel.bind(args).to_triton_code(config).splitlines()
        va = _first(lines, "va = tl.load(a ")
        # The dot root waits for the last producer only, before reading the
        # first producer's `a`: that wait certifies both earlier producers.
        self.assertTrue(_polled_between(lines, _task_loop_before(lines, va), va))
        # Negative control (must race): the wait ignores `a`.
        with mock.patch.object(
            cross_loop_codegen, "_insert_producer_waits", _without_names("a")
        ):
            mutant = helion.kernel(chain3, config=config, static_shapes=True)
            self.assertGreater(_bad_launches(partial(mutant, *args), poison, good), 0)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_wait_precedes_overwriting_what_a_pruned_producer_reads(self) -> None:
        x, w = self.x, self.w
        s0 = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        s, a, b = s0.clone(), torch.empty_like(x), torch.empty_like(x)
        a_expected = (s0.float() + 1).bfloat16()
        config = helion.Config(**_STATIC)
        args = (x, w, s, a, b)

        def poison() -> None:
            s.copy_(s0)
            a.fill_(float("nan"))
            b.fill_(float("nan"))

        def good(out: torch.Tensor) -> bool:
            # The first producer read the original `s`, not the dot root's 100s.
            return torch.equal(a, a_expected) and _close(out, self._fit(b))

        kernel = helion.kernel(war_trans, config=config, static_shapes=True)
        self.assertEqual(_bad_launches(partial(kernel, *args), poison, good), 0)
        lines = kernel.bind(args).to_triton_code(config).splitlines()
        store = _first(lines, "tl.store(s ")
        self.assertTrue(_polled_between(lines, _task_loop_before(lines, store), store))
        with mock.patch.object(
            cross_loop_codegen, "_insert_producer_waits", _without_names("s")
        ):
            mutant = helion.kernel(war_trans, config=config, static_shapes=True)
            self.assertGreater(_bad_launches(partial(mutant, *args), poison, good), 0)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_views_carry_war_and_transitive_obligations(self) -> None:
        x, w = self.x, self.w
        s0 = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        s, a, b = s0.clone(), torch.empty_like(x), torch.empty_like(x)
        a_expected = (s0.float() + 1).bfloat16()
        config = helion.Config(**_STATIC)

        def war_poison() -> None:
            s.copy_(s0)
            a.fill_(float("nan"))

        cases: dict[str, tuple[Any, tuple[torch.Tensor, ...], Any, Any, str, str]] = {
            # (kernel, args, poison, good, line read or written early, dropped)
            "war_view": (
                war_view,
                (x, w, s, s.view(6, 64), a),
                war_poison,
                lambda out: torch.equal(a, a_expected) and _close(out, self._fit(a)),
                "tl.store(s_v ",
                "s_v",
            ),
            "trans_view": (
                trans_view,
                (x, w, a, a.view(6, 64), b),
                _nan(a, b),
                lambda out: _close(out, self._fit(a, b)),
                "va = tl.load(a ",
                "a",
            ),
        }
        for name, (fn, args, poison, good, early, dropped) in cases.items():
            with self.subTest(name=name):
                kernel = helion.kernel(fn, config=config, static_shapes=True)
                self.assertEqual(_bad_launches(partial(kernel, *args), poison, good), 0)
                lines = kernel.bind(args).to_triton_code(config).splitlines()
                at = _first(lines, early)
                self.assertTrue(
                    _polled_between(lines, _task_loop_before(lines, at), at)
                )
                with mock.patch.object(
                    cross_loop_codegen,
                    "_insert_producer_waits",
                    _without_names(dropped),
                ):
                    mutant = helion.kernel(fn, config=config, static_shapes=True)
                    self.assertGreater(
                        _bad_launches(partial(mutant, *args), poison, good), 0
                    )

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_exported_fused_roots_refuse_unseen_aliasing(self) -> None:
        x = self.x
        a, b = torch.empty_like(x), torch.empty_like(x)
        cases = {
            "plain": (
                two_root_alias,
                _PLAIN,
                (x, a, torch.zeros(3, 128, device=DEVICE, dtype=x.dtype)),
                (x, a, a.view(3, 128)),
                _nan(a),
            ),
            "barrier": (
                barrier_alias,
                _BARRIER,
                (x, a, torch.zeros(3, 128, device=DEVICE, dtype=x.dtype), b),
                (x, a, a.view(3, 128), b),
                _nan(a, b),
            ),
        }

        def good(out: torch.Tensor) -> bool:
            return _close(out, a.view(3, 128).float())

        for name, (fn, config, distinct, aliased, poison) in cases.items():
            with self.subTest(name=name):
                kernel = helion.kernel(
                    fn, config=helion.Config(**config), static_shapes=True
                )
                kernel(*distinct)
                # The kernel's cache key includes the storage aliasing of its
                # inputs: the aliasing call compiles for it.
                self.assertEqual(
                    _bad_launches(partial(kernel, *aliased), poison, good), 0
                )
                bound = kernel.bind(distinct)
                run = bound._run
                assert run is not None
                # The code compiled for distinct storages, run without the
                # compiler (as exported), refuses the aliasing launch.
                with self.assertRaises(helion.exc.CrossLoopAliasedLaunch):
                    run(*aliased)
                code = bound.to_triton_code(helion.Config(**config))
                self.assertIn("_check_storage_aliasing=True", code)
                self.assertNotIn("_persistent_state_specs", code)
                # Negative control (must race): without that check.
                with mock.patch.object(
                    launcher, "_check_compiled_storage_sharing", lambda *args: None
                ):
                    self.assertGreater(
                        _bad_launches(partial(run, *aliased), poison, good), 0
                    )

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_symbolic_aliasing_keeps_its_edges(self) -> None:
        x = self.x
        a, b = torch.empty_like(x), torch.empty_like(x)
        cases = {
            "plain": (two_same, _PLAIN, (x, a, a.view(6, 64)), (0, 1), _nan(a)),
            "barrier": (
                barrier_same,
                _BARRIER,
                (x, a, a.view(6, 64), b),
                (1, 2),
                _nan(a, b),
            ),
        }

        def good(out: torch.Tensor) -> bool:
            return _close(out, a.float())

        def compile_symbolic(fn: Any, config: Any, real: Any) -> tuple[Any, Any]:
            kernel = helion.kernel(fn, config=config, static_shapes=False)
            mode, fake_args = _symbolic_fakes(real)
            with mode:
                bound = kernel.bind(fake_args)
                return bound, bound.compile_config(config)

        for name, (fn, config, real, edge, poison) in cases.items():
            with self.subTest(name=name):
                config = helion.Config(**config)
                bound, run = compile_symbolic(fn, config, real)
                # Inputs sharing a storage with symbolic layouts share one fake
                # storage too: the edge through the alias is kept.
                self.assertEqual(_bad_launches(partial(run, *real), poison, good), 0)
                self.assertIn(edge, _edges(bound))
                self.assertFalse(bound.env.unrepresented_input_aliasing)
                # Negative control (must race): fakes kept apart and the
                # launch check off.
                with (
                    mock.patch.object(
                        CompileEnvironment,
                        "_alias_input_fake",
                        lambda env, alias, fake: fake,
                    ),
                    mock.patch.object(
                        launcher, "_check_compiled_storage_sharing", lambda *a: None
                    ),
                ):
                    mutant, mutant_run = compile_symbolic(fn, config, real)
                    self.assertNotIn(edge, _edges(mutant))
                    self.assertGreater(
                        _bad_launches(partial(mutant_run, *real), poison, good), 0
                    )
        # Roots that only read the aliased inputs stay independent: the
        # binding is not refused.
        config = helion.Config(**_PLAIN)
        real = (x, a, a.view(6, 64), b)
        a.copy_(x)
        bound, run = compile_symbolic(two_readers, config, real)
        self.assertEqual(_edges(bound), set())
        out = run(*real)
        torch.testing.assert_close(out, a.float() + x.float())
        torch.testing.assert_close(b, a * 2)
        # Backstop: sharing the fakes cannot represent refuses any multi-root
        # binding rather than drop its edges.
        with (
            mock.patch.object(
                CompileEnvironment,
                "_input_alias_storage",
                mock.Mock(side_effect=RuntimeError("unrepresentable")),
            ),
            self.assertRaises(helion.exc.CrossLoopSchedulingError),
        ):
            compile_symbolic(two_same, config, (x, a, a.view(6, 64)))

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_storages_over_one_buffer_are_one_allocation(self) -> None:
        x, w = self.x, self.w
        b = torch.empty_like(x)
        buf = torch.empty(12, 64, device=DEVICE, dtype=x.dtype)
        config = helion.Config(**_STATIC)
        # `other` is built over buf's memory with its own storage: at buf's
        # base address (full) or inside buf (partial overlap).
        cases = {
            "partial": (buf[6:12], torch.from_dlpack(buf[6:12])),
            "full": (buf[0:6], torch.from_dlpack(buf[0:6])),
        }

        def by_base_pointer(values: Any) -> list[Any]:
            """Storage classes before overlapping storages were merged."""
            classes = []
            for value in values:
                extent = (
                    input_storage_aliasing.input_storage_extent(value)
                    if isinstance(value, torch.Tensor)
                    else None
                )
                classes.append(
                    ("ptr", extent[0])
                    if extent is not None and isinstance(extent[0], int)
                    else extent
                )
            return classes

        for name, (av, other) in cases.items():
            with self.subTest(name=name):

                def good(out: torch.Tensor, av: torch.Tensor = av) -> bool:
                    return _close(out, self._fit(av, b))

                poison = _nan(buf, b)
                aliased = (x, w, av, other, b)
                kernel = helion.kernel(alias_view, config=config, static_shapes=True)
                self.assertEqual(
                    _bad_launches(partial(kernel, *aliased), poison, good), 0
                )
                # Code compiled for distinct storages refuses the launch.
                distinct = (x, w, av, torch.zeros_like(x), b)
                kernel(*distinct)
                run = kernel.bind(distinct)._run
                assert run is not None
                with self.assertRaises(helion.exc.CrossLoopAliasedLaunch):
                    run(*aliased)
                if name != "partial":
                    continue
                # Negative control (must race): storage classes by base
                # pointer (no merging of overlapping storages), launch check
                # off.
                with (
                    mock.patch.object(
                        input_storage_aliasing, "input_storage_classes", by_base_pointer
                    ),
                    mock.patch.object(
                        launcher, "_check_compiled_storage_sharing", lambda *a: None
                    ),
                ):
                    mutant = helion.kernel(
                        alias_view, config=config, static_shapes=True
                    )
                    self.assertGreater(
                        _bad_launches(partial(mutant, *aliased), poison, good), 0
                    )
        # Dynamic shapes: storages over one buffer get one fake storage too.
        a = torch.empty_like(x)
        for name, (fn, config, args, edge, poison) in {
            "plain": (
                two_same,
                _PLAIN,
                (x, a, torch.from_dlpack(a)),
                (0, 1),
                _nan(a),
            ),
            "barrier": (
                barrier_same,
                _BARRIER,
                (x, a, torch.from_dlpack(a), b),
                (1, 2),
                _nan(a, b),
            ),
        }.items():
            with self.subTest(name=f"dynamic_{name}"):
                kernel = helion.kernel(
                    fn, config=helion.Config(**config), static_shapes=False
                )
                self.assertEqual(
                    _bad_launches(
                        partial(kernel, *args), poison, lambda o: _close(o, a.float())
                    ),
                    0,
                )
                self.assertIn(edge, _edges(kernel.bind(args)))

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_transitively_reduced_producers_still_order_reads(self) -> None:
        torch.manual_seed(0)
        x = (0.05 * torch.randn(64, 8192, device=DEVICE)).bfloat16()
        w = (0.05 * torch.randn(8192, 64, device=DEVICE)).bfloat16()
        a = torch.empty(64, 64, device=DEVICE, dtype=torch.bfloat16)
        idx1 = torch.randint(0, 64, (32, 16), device=DEVICE, dtype=torch.int32)
        b = torch.empty(32, 64, device=DEVICE, dtype=torch.bfloat16)
        xv = torch.randn(32, 64, device=DEVICE).bfloat16()
        v = torch.empty(32, 64, device=DEVICE, dtype=torch.bfloat16)
        idx2 = torch.randint(0, 64, (96, 16), device=DEVICE, dtype=torch.int32)
        idx3 = torch.randint(0, 32, (96, 16), device=DEVICE, dtype=torch.int32)
        idx4 = torch.randint(0, 32, (96, 16), device=DEVICE, dtype=torch.int32)
        w2 = (0.1 * torch.randn(64, 16, device=DEVICE)).bfloat16()
        out = torch.empty(96, 16, 16, device=DEVICE)
        a_ref = (x.float() @ w.float()).bfloat16()
        b_ref = (a_ref[idx1.long()].float().sum(1) + 1).bfloat16()
        v_ref = (xv.float() * 2).bfloat16()
        ref = (
            a_ref[idx2.long()].float() @ w2.float()
            + b_ref[idx3.long()].float() @ w2.float()
        )
        cases = {
            # (kernel, args, range_num_stages of A's sink / no sink, expected)
            "one_incoming": (
                transitive_gather,
                (x, w, a, idx1, b, idx2, idx3, w2, out),
                lambda stages: [0, 2, 0, stages],
                ref,
            ),
            "two_incoming": (
                transitive_gather2,
                (x, w, a, idx1, b, xv, v, idx2, idx3, idx4, w2, out),
                lambda stages: [0, 2, 0, 0, stages],
                ref + v_ref[idx4.long()].float() @ w2.float(),
            ),
        }
        poison = _nan(a, b, v)
        for name, (fn, args, stages, expected) in cases.items():
            with self.subTest(name=name):

                def make(sink: bool, fn: Any = fn, stages: Any = stages) -> Any:
                    options: dict[str, Any] = {
                        **_STATIC,
                        "num_stages": 2,
                        "range_num_stages": stages(1 if sink else 2),
                    }
                    config = helion.Config(**options)
                    kernel = helion.kernel(
                        fn,
                        config=config,
                        static_shapes=True,
                        ignore_warnings=[helion.exc.TensorOperationInWrapper],
                    )
                    return kernel, config

                def good(result: torch.Tensor, expected: Any = expected) -> bool:
                    return _close(result, expected)

                kernel, config = make(sink=True)
                self.assertEqual(_bad_launches(partial(kernel, *args), poison, good), 0)
                lines = kernel.bind(args).to_triton_code(config).splitlines()
                ra = _first(lines, "ra = tl.load(a ")
                # A's wait (sunk into its task) precedes its read of `a`.
                self.assertTrue(
                    _polled_between(lines, _task_loop_before(lines, ra), ra)
                )
                # Control: the wait at the task's top (no sinking).
                control, _ = make(sink=False)
                self.assertEqual(
                    _bad_launches(partial(control, *args), poison, good), 0
                )
                # Negative control (must race): the sunk wait ignores `a`.
                with mock.patch.object(
                    cross_loop_codegen, "_insert_producer_waits", _without_names("a")
                ):
                    mutant, _ = make(sink=True)
                    self.assertGreater(
                        _bad_launches(partial(mutant, *args), poison, good), 0
                    )

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_warp_specialized_inner_loop_waits_inside_its_task(self) -> None:
        x = torch.randn(6, 256, device=DEVICE, dtype=torch.bfloat16)
        w = (0.1 * torch.randn(64, 256, device=DEVICE)).bfloat16()
        a = torch.empty_like(x)
        args = (x, w, a)
        options: dict[str, Any] = {
            **_STATIC,
            "num_stages": 3,
            "range_warp_specializes": [None, None, True],
        }
        config = helion.Config(**options)
        poison = _nan(a)

        def good(out: torch.Tensor) -> bool:
            return _close(out, w.float() @ _padded_rows(a).T)

        kernel = helion.kernel(ws_inner_gather, config=config, static_shapes=True)
        self.assertEqual(_bad_launches(partial(kernel, *args), poison, good), 0)
        lines = kernel.bind(args).to_triton_code(config).splitlines()
        inner = next(
            i for i, line in enumerate(lines) if "warp_specialize=True" in line
        )
        # Only the inner loop is specialized: the root's wait sits in its task,
        # in front of that loop (the specialized partitions start after it).
        self.assertTrue(_polled_between(lines, _task_loop_before(lines, inner), inner))
        self.assertNotIn("warp_specialize=True", lines[_task_loop_before(lines, inner)])

        def after_the_loop(
            statements: list[ast.stmt],
            names: tuple[frozenset[str] | None, ...],
            make_wait: Any,
        ) -> tuple[int, ...]:
            producers = tuple(range(len(names)))
            statements.extend(make_wait(producers))
            return producers

        # Negative control (must race): the wait after the specialized loop.
        with mock.patch.object(
            cross_loop_codegen, "_insert_producer_waits", after_the_loop
        ):
            mutant = helion.kernel(ws_inner_gather, config=config, static_shapes=True)
            self.assertGreater(_bad_launches(partial(mutant, *args), poison, good), 0)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_descriptor_of_a_written_tensor_is_written(self) -> None:
        # ``a`` is stored through pointers and read through a host TMA
        # descriptor: one storage class, so the pass runs (the launch's
        # sharing is the compiled one) and treats ``a_desc`` as written; the
        # weight's copies still move above the wait.
        x = torch.randn(16, 512, device=DEVICE, dtype=torch.bfloat16)
        w = (0.05 * torch.randn(256, 512, device=DEVICE)).bfloat16()
        a = torch.empty_like(x)
        args = (x, w, a)
        options: dict[str, Any] = {
            **_STATIC,
            "num_stages": 3,
            "indexing": "tensor_descriptor",
            "host_tensor_descriptors": True,
        }
        config = helion.Config(**options)
        poison = _nan(a)

        def good(out: torch.Tensor) -> bool:
            return _close(out, w.float() @ a.float().T)

        def early_copies(kernel: helion.Kernel) -> list[dict[str, int]]:
            """Per compiled binary of the kernel's Triton function (codegen is
            cached by content, so every kernel built here shares it): copies of
            each descriptor issued ahead of a readiness poll, in either form the
            pass produces (see ``_early_descriptor_copies``)."""
            bound = kernel.bind(args)
            module = sys.modules[bound._run.__module__]  # pyrefly: ignore[missing-attribute]
            return [
                _early_descriptor_copies(compiled.asm["ttgir"], ("w_desc", "a_desc"))
                for caches in module._helion_tma_alias.device_caches.values()
                for compiled in caches[0].values()
            ]

        kernel = helion.kernel(tma_alias, config=config, static_shapes=True)
        code = kernel.bind(args).to_triton_code(config)
        self.assertIn("_ttgir_weight_prefetch=True", code)
        self.assertRegex(code, r"_tensor_storage_aliases=\(\(\d+, \d+\),\)")
        self.assertEqual(_bad_launches(partial(kernel, *args), poison, good), 0)
        (copies,) = early_copies(kernel)
        self.assertGreater(copies["w_desc"], 0)
        self.assertEqual(copies["a_desc"], 0)
        # Negative control (must race): the pass told nothing shares a storage.
        with mock.patch.object(
            ttgir_weight_prefetch, "normalize_alias_classes", lambda classes: ()
        ):
            mutant = helion.kernel(tma_alias, config=config, static_shapes=True)
            self.assertGreater(_bad_launches(partial(mutant, *args), poison, good), 0)
            self.assertTrue(any(c["a_desc"] > 0 for c in early_copies(mutant)))

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_warp_specialized_task_loop_keeps_its_wait_in_front_of_it(self) -> None:
        # Triton cannot warp-specialize a loop holding the spin: a root whose
        # task loop is specialized waits before that loop; a root with only a
        # specialized inner loop waits inside its task, before that loop.
        x = torch.randn(6, 256, device=DEVICE, dtype=torch.bfloat16)
        w = (0.1 * torch.randn(64, 256, device=DEVICE)).bfloat16()
        args = (x, w, torch.empty_like(x))
        for specializes, inside in (
            ([None, True, None], False),
            ([None, True, True], False),
            ([None, None, True], True),
        ):
            with self.subTest(range_warp_specializes=specializes):
                options: dict[str, Any] = {
                    **_STATIC,
                    "num_stages": 3,
                    "range_warp_specializes": specializes,
                }
                config = helion.Config(**options)
                kernel = helion.kernel(
                    ws_inner_gather, config=config, static_shapes=True
                )
                lines = kernel.bind(args).to_triton_code(config).splitlines()
                loops = [
                    i
                    for i, line in enumerate(lines)
                    if "for virtual_pid in" in line or "for offset_" in line
                ]
                task, inner = loops[-2], loops[-1]
                self.assertEqual(
                    "warp_specialize=True" in lines[task], specializes[1] is True
                )
                # Triton does not nest specialized loops: a specialized task
                # loop leaves its inner loop alone.
                self.assertEqual(
                    "warp_specialize=True" in lines[inner],
                    specializes[2] is True and specializes[1] is not True,
                )
                self.assertEqual(len(_initial_polls(lines)), 1)
                self.assertEqual(_polled_between(lines, task, inner), inside)
                self.assertEqual(_polled_between(lines, loops[0], task), not inside)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_symbolic_storage_class_bounds_no_input_size(self) -> None:
        # Dynamic shapes: inputs over one buffer with their own storages get one
        # fake storage whose size is unbacked, so moving their symbolic layouts
        # onto it bounds (and guards) no input size: the input symbols keep the
        # ranges of a binding whose inputs share nothing.
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        def bind(*args: torch.Tensor) -> Any:
            kernel = helion.kernel(
                two_same, config=helion.Config(**_PLAIN), static_shapes=False
            )
            return kernel.bind(args).env

        def input_ranges(env: Any) -> list[str]:
            return sorted(
                f"{bounds.lower}..{bounds.upper}"
                for symbol, bounds in env.shape_env.var_to_range.items()
                if not free_unbacked_symbols(symbol)
            )

        a = torch.empty_like(self.x)
        env = bind(self.x, a, torch.from_dlpack(a))
        (alias,) = env._input_storage_alias_classes.values()
        self.assertIsNone(alias.nbytes)
        assert alias.storage is not None
        self.assertTrue(free_unbacked_symbols(alias.storage.nbytes()))
        distinct = bind(self.x, a, torch.empty_like(a))
        self.assertFalse(distinct._input_storage_alias_classes)
        self.assertEqual(input_ranges(env), input_ranges(distinct))
