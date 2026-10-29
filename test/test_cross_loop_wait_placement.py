"""Race tests for dynamic cross-loop roots that wait on a root barrier where
they first use a producer's memory.

Each case runs a kernel whose first producer is late (a ``%globaltimer``
spin) on memory poisoned before every launch, so an access that the schedule
does not order after its producer shows up as a wrong result; most also check
a negative control that drops one name from the sunk wait's guard: it must
race.  The cases are the transitive and write-after-read hazards behind
pruned root-barrier edges, directly and through host views, accesses the
dependency graph does not record, through kernel tensors and through
addresses given as values, and a gated root that publishes a root barrier.
"""

from __future__ import annotations

from functools import partial
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from unittest import mock

import torch

import helion
from helion._compiler import cross_loop_codegen
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessHostTensorDescriptor
import helion.language as hl

if TYPE_CHECKING:
    import ast

# Spins until %globaltimer has advanced by $1 ns, then yields 0.  A producer
# indexes its first access with the result, so that access runs late.  The
# race kernels declare it pure: assembly with side effects makes its root
# opaque, keeping every wait on it at task entry.
_GLOBALTIMER_SPIN = (
    "{ .reg .u64 t0, t1; .reg .pred p; mov.u64 t0, %globaltimer; "
    "SPIN: mov.u64 t1, %globaltimer; sub.u64 t1, t1, t0; "
    "setp.lt.u64 p, t1, $1; @p bra SPIN; mov.u32 $0, 0; }"
)

# Overwrites rows [0, {2}) of columns [{1}, {1} + 16) of a 64-column bf16
# tensor with 100s.
_STORE_ROWS = """
r = tl.arange(0, 16)
c = {1} + tl.arange(0, 16)
tl.store({0} + r[:, None] * 64 + c[None, :], tl.full([16, 16], 100.0, tl.bfloat16), mask=(r < {2})[:, None])
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
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [hl.full([1], 20000, dtype=torch.int64)],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        a[t, :] = x[t.index + spun, :] + 1
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
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [hl.full([1], 20000, dtype=torch.int64)],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        a[t, :] = s[t.index + spun, :] + 1
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
    x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    """The dot root overwrites ``s_v`` (a host view of ``s``), which its
    producer (late) reads as ``s``."""
    tokens, k = x.size()
    n = w.size(0)
    s_v = s.view(tokens, k)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [hl.full([1], 20000, dtype=torch.int64)],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        a[t, :] = s[t.index + spun, :] + 1
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
    x: torch.Tensor, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """``a`` (late) reaches the dot root through a producer that reads all of
    it as ``a_v`` (a host view of ``a``)."""
    tokens, k = x.size()
    n = w.size(0)
    a_v = a.view(tokens, k)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [hl.full([1], 20000, dtype=torch.int64)],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        a[t, :] = x[t.index + spun, :] + 1
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


def war_unrecorded(
    x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    """The dot root overwrites ``s``, which its producer (late) reads, through
    ``hl.inline_triton``: a write the dependency graph does not record."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [hl.full([1], 20000, dtype=torch.int64)],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        a[t, :] = s[t.index + spun, :] + 1
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        wt = w[n_t, :]
        hl.inline_triton(_STORE_ROWS, args=(s, n_t.begin, tokens), output_like=None)
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        out[n_t, :] = hl.dot(wt, va.permute(1, 0), out_dtype=torch.float32)
    return out


# Stores $2 at, and loads from, a global address given as a value.
_STORE_AT = "st.global.f32 [$1], $2; mov.u32 $0, 0;"
_LOAD_AT = "ld.global.f32 $0, [$1];"


def address_access(
    x: torch.Tensor, w: torch.Tensor, a: torch.Tensor, addr: int
) -> torch.Tensor:
    """The producer (late) writes a scratch buffer through assembly given its
    address ``addr`` as a value; the dot root reads it the same way before it
    reads ``a``: accesses the dependency graph does not record, through an
    address no kernel tensor names."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [hl.full([1], 20000, dtype=torch.int64)],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        at = (
            hl.full([1], addr, dtype=torch.int64)
            + t.index.to(torch.int64) * 4
            + spun.to(torch.int64)
        )
        stored = hl.inline_asm_elementwise(
            _STORE_AT,
            "=r,l,f",
            [at, hl.full([1], 1.0, dtype=torch.float32)],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        a[t, :] = x[t.index + stored, :] + 1
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        scratch = hl.inline_asm_elementwise(
            _LOAD_AT,
            "=f,l",
            [hl.full([16], addr, dtype=torch.int64) + rows.to(torch.int64) * 4],
            dtype=torch.float32,
            is_pure=False,
            pack=1,
        )
        scratch = torch.where(live, scratch, 0.0)
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, va.permute(1, 0), out_dtype=torch.float32)
        out[n_t, :] = acc + scratch[None, :]
    return out


def gated_overwrite(
    x: torch.Tensor, keep: torch.Tensor, out: torch.Tensor, delay: torch.Tensor
) -> torch.Tensor:
    """The producer zero-fills ``out`` (rows ``delay[row]`` ns late); the
    gated consumer (a counter on the producer) overwrites the rows ``keep``
    selects; the join reads all of ``out`` (a root barrier on the consumer).
    The dependency graph lets the consumer's store, which may not run, kill
    the producer's: only the consumer's wait orders the join after it."""
    total = torch.empty([x.size(0)], device=x.device, dtype=x.dtype)
    for producer_m in hl.tile(x.size(0), block_size=1):
        for producer_n in hl.tile(x.size(1), block_size=32):
            spun = hl.inline_asm_elementwise(
                _GLOBALTIMER_SPIN,
                "=r,l",
                [delay[producer_m]],
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            out[producer_m, producer_n] = (
                hl.zeros([producer_m, producer_n], dtype=torch.float32)
                + spun.to(torch.float32)[:, None]
            )
    for consumer_m, consumer_n in hl.tile(x.size(), block_size=[1, 32]):
        if keep[consumer_m.begin] > 0:
            out[consumer_m, consumer_n] = x[consumer_m, consumer_n] * 2
    for join_m in hl.tile(x.size(0), block_size=1):
        total[join_m] = (
            torch.sum(torch.sum(out[:, :], dim=-1), dim=0) + 0 * join_m.begin
        )
    return total


def gated_overwrite_after_read(
    x: torch.Tensor,
    keep: torch.Tensor,
    s: torch.Tensor,
    a: torch.Tensor,
    flag: torch.Tensor,
    delay: torch.Tensor,
) -> torch.Tensor:
    """The producer reads ``s`` into ``a`` (rows ``delay[row]`` ns late); the
    gated consumer marks ``flag`` and overwrites the rows of ``s`` that
    ``keep`` selects; the join reads all of ``flag`` (a root barrier on the
    consumer) and overwrites ``s``.  The consumer's write, which may not run,
    kills the producer's read: only the consumer's wait orders the join's
    write after it."""
    for producer_m in hl.tile(x.size(0), block_size=1):
        for producer_n in hl.tile(x.size(1), block_size=32):
            spun = hl.inline_asm_elementwise(
                _GLOBALTIMER_SPIN,
                "=r,l",
                [delay[producer_m]],
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            a[producer_m, producer_n] = (
                s[producer_m, producer_n] + 1 + spun.to(torch.float32)[:, None]
            )
    for consumer_m, consumer_n in hl.tile(x.size(), block_size=[1, 32]):
        flag[consumer_m, consumer_n] = x[consumer_m, consumer_n] * 0 + 1
        if keep[consumer_m.begin] > 0:
            s[consumer_m, consumer_n] = x[consumer_m, consumer_n] * 100
    for join_m in hl.tile(x.size(0), block_size=1):
        seen = torch.sum(torch.sum(flag[:, :], dim=-1), dim=0)
        s[join_m, :] = hl.zeros([join_m, x.size(1)], dtype=torch.float32) + 7 + 0 * seen
    return a


# Waits 20 us without touching memory.
_TENSORLESS_SPIN = """
t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
t1 = t0
while t1 - t0 < 20000:
    t1 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
0
"""


def opaque_kinds(
    x: torch.Tensor, q: torch.Tensor, flags: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """One root per kind of op the opaque-root rule classifies, in order."""
    n = x.size(0)
    for t in hl.tile(n):
        out[t] = x[t] + 1
    for t in hl.tile(n):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [hl.full([1], 20, dtype=torch.int64)],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        out[t.index + spun] = x[t] + 2
    for t in hl.tile(n):
        hl.atomic_add(flags, [t], 1)
        hl.atomic_add(flags, [t], 1, sem="release")
    for t in hl.tile(n):
        hl.atomic_add(flags, [t], 1, sem="acquire")
    for t in hl.tile(n):
        hl.inline_triton(_TENSORLESS_SPIN, args={}, output_like=None)
        out[t] = x[t] + 3
    for t in hl.tile(n):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [hl.full([1], 20, dtype=torch.int64)],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        out[t.index + spun] = x[t] + 4
    for t in hl.tile(n):
        hl.inline_triton(_STORE_ROWS, args=(out, t.begin, n), output_like=None)
    for t in hl.tile(n, block_size=16):
        lanes = hl.load_float4_e2m1fn_x16_to_float16(q, t.index)
        out[t] = lanes[0].to(torch.float32)
    return out


def tma_alias(x: torch.Tensor, w: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Writes ``a`` through pointers (late); the next root reads ``a`` and the
    weight ``w`` in a pipelined K loop, by TMA descriptors when indexing says
    so: ``a`` then reaches the kernel twice, as a pointer and a descriptor."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, tokens], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [hl.full([1], 20000, dtype=torch.int64)],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        a[t, :] = x[t.index + spun, :] + 1
    for n_t in hl.tile(n, block_size=64):
        acc = hl.zeros([n_t, tokens], dtype=torch.float32)
        for k_t in hl.tile(k, block_size=64):
            acc = hl.dot(w[n_t, k_t], a[:, k_t].permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


_DYNAMIC: dict[str, Any] = {
    "block_sizes": [],
    "cross_loop_pipeline": "dynamic",
    "pid_type": "persistent_blocked",
    "num_warps": 4,
    "num_stages": 1,
    "num_sm_multiplier": 1,
    "indexing": "pointer",
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


def _initial_polls(
    lines: list[str], wait: str = "root_barrier"
) -> list[tuple[int, int, str]]:
    """(line, indent, counter) of every poll outside its spin of a ``wait``
    of the kind (``root_barrier``, ``readiness``, ``nested_loop``) or kinds
    (a pattern) given."""
    pattern = re.compile(
        rf"^(\s*)tile_dependency_{wait}_wait(?:_\d+)? = tl\.inline_asm_elementwise"
        r"\(asm='ld\.acquire\.gpu\.global\.u32 \$0, \[\$1\];', "
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


def _without_names(dropped: str) -> Callable[..., bool]:
    """``_insert_before_first_use`` that ignores ``dropped`` as a use of the
    producer's memory (a negative control)."""
    original = cross_loop_codegen._insert_before_first_use

    def insert(
        statements: list[ast.stmt], names: frozenset[str], inserted: list[ast.stmt]
    ) -> bool:
        return original(statements, (names - {dropped}) or frozenset({"_"}), inserted)

    return insert


def _polled_before(lines: list[str], index: int, wait: str = "root_barrier") -> bool:
    """A ``wait`` poll precedes line ``index`` in its function."""
    start = max(i for i, line in enumerate(lines[:index]) if line.startswith("def "))
    return any(start < poll[0] < index for poll in _initial_polls(lines, wait))


@onlyBackends(["triton"])
@skipIfNotCUDA()
class TestDynamicWaitPlacementRaces(TestCase):
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

    def _check(
        self,
        fn: Any,
        args: tuple[torch.Tensor, ...],
        poison: Callable[[], None],
        good: Callable[[Any], bool],
        early: str,
        dropped: str,
        wait: str = "root_barrier",
    ) -> None:
        config = helion.Config(**_DYNAMIC)
        kernel = helion.kernel(fn, config=config, static_shapes=True)
        self.assertEqual(_bad_launches(partial(kernel, *args), poison, good), 0)
        lines = kernel.bind(args).to_triton_code(config).splitlines()
        self.assertTrue(_polled_before(lines, _first(lines, early), wait))
        # Negative control (must race): the sunk wait ignores ``dropped``.
        with mock.patch.object(
            cross_loop_codegen, "_insert_before_first_use", _without_names(dropped)
        ):
            mutant = helion.kernel(fn, config=config, static_shapes=True)
            self.assertGreater(_bad_launches(partial(mutant, *args), poison, good), 0)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_wait_covers_a_chain_of_pruned_producers(self) -> None:
        # The dot root waits for the last producer only, before reading the
        # first producer's ``a``: that wait certifies both earlier producers.
        x, w = self.x, self.w
        a, b, c = (torch.empty_like(x) for _ in range(3))
        self._check(
            chain3,
            (x, w, a, b, c),
            _nan(a, b, c),
            lambda out: _close(out, self._fit(a, c)),
            "va = tl.load(a ",
            "a",
        )

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_wait_precedes_overwriting_what_a_pruned_producer_reads(self) -> None:
        x, w = self.x, self.w
        s0 = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        s, a, b = s0.clone(), torch.empty_like(x), torch.empty_like(x)
        a_expected = (s0.float() + 1).bfloat16()

        def poison() -> None:
            s.copy_(s0)
            a.fill_(float("nan"))
            b.fill_(float("nan"))

        self._check(
            war_trans,
            (x, w, s, a, b),
            poison,
            # The first producer read the original ``s``, not the dot root's 100s.
            lambda out: torch.equal(a, a_expected) and _close(out, self._fit(b)),
            "tl.store(s ",
            "s",
        )

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_wait_precedes_overwriting_a_view_of_what_a_producer_reads(self) -> None:
        x, w = self.x, self.w
        s0 = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        s, a = s0.clone(), torch.empty_like(x)
        a_expected = (s0.float() + 1).bfloat16()

        def poison() -> None:
            s.copy_(s0)
            a.fill_(float("nan"))

        self._check(
            war_view,
            (x, w, s, a),
            poison,
            lambda out: torch.equal(a, a_expected) and _close(out, self._fit(a)),
            "tl.store(s_v ",
            "s_v",
        )

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_wait_covers_a_producer_that_reads_through_a_view(self) -> None:
        x, w = self.x, self.w
        a, b = torch.empty_like(x), torch.empty_like(x)
        self._check(
            trans_view,
            (x, w, a, b),
            _nan(a, b),
            lambda out: _close(out, self._fit(a, b)),
            "va = tl.load(a ",
            "a",
        )

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_unrecorded_write_keeps_the_wait_at_entry(self) -> None:
        # ``hl.inline_triton`` gets ``s`` itself, so the dot root writes
        # memory the dependency graph cannot see: its wait does not sink.
        x, w = self.x, self.w
        s0 = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        s, a = s0.clone(), torch.empty_like(x)
        a_expected = (s0.float() + 1).bfloat16()

        def poison() -> None:
            s.copy_(s0)
            a.fill_(float("nan"))

        def good(out: torch.Tensor) -> bool:
            return torch.equal(a, a_expected) and _close(out, self._fit(a))

        args = (x, w, s, a)
        config = helion.Config(**_DYNAMIC)
        kernel = helion.kernel(war_unrecorded, config=config, static_shapes=True)
        self.assertEqual(_bad_launches(partial(kernel, *args), poison, good), 0)
        lines = kernel.bind(args).to_triton_code(config).splitlines()
        self.assertTrue(_polled_before(lines, _first(lines, "tl.store(s ")))
        # Negative control (must race): no root counts as opaque.
        with mock.patch.object(
            cross_loop_codegen, "_opaque_roots", lambda *_args, **_kwargs: frozenset()
        ):
            mutant = helion.kernel(war_unrecorded, config=config, static_shapes=True)
            self.assertGreater(_bad_launches(partial(mutant, *args), poison, good), 0)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_access_through_an_address_keeps_the_wait_at_entry(self) -> None:
        # Assembly with side effects may reach any memory through the values
        # it is given: the dot root's read of the scratch buffer must follow
        # the producer's write, and no kernel tensor names it, so its wait
        # does not sink.
        x, w = self.x, self.w
        a = torch.empty_like(x)
        scratch = torch.empty(16, device=DEVICE)
        written = torch.zeros(16, device=DEVICE)
        written[: x.size(0)] = 1.0

        def good(out: torch.Tensor) -> bool:
            return _close(out, self._fit(a) + written[None, :])

        args = (x, w, a, scratch.data_ptr())
        config = helion.Config(**_DYNAMIC)
        kernel = helion.kernel(address_access, config=config, static_shapes=True)
        poison = _nan(a, scratch)
        self.assertEqual(_bad_launches(partial(kernel, *args), poison, good), 0)
        lines = kernel.bind(args).to_triton_code(config).splitlines()
        self.assertTrue(
            _polled_before(lines, _first(lines, "scratch = tl.inline_asm_elementwise"))
        )
        # Negative control (must race): the dot root's wait sinks as an
        # opaque root's whose accesses all name kernel tensors would.
        opaque_roots = cross_loop_codegen._opaque_roots

        def named_only(
            device_ir: Any, *, addresses_only: bool = False
        ) -> frozenset[int]:
            return frozenset() if addresses_only else opaque_roots(device_ir)

        with mock.patch.object(cross_loop_codegen, "_opaque_roots", named_only):
            mutant = helion.kernel(address_access, config=config, static_shapes=True)
            self.assertGreater(_bad_launches(partial(mutant, *args), poison, good), 0)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_publishing_root_waits_in_every_task(self) -> None:
        # The gated consumer publishes the join's root barrier, which also
        # certifies the producer: every consumer task waits, even where the
        # branch skips its stores.  The rows it skips are produced 50 us late.
        x = torch.randn(8, 128, device=DEVICE)
        keep = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], device=DEVICE, dtype=torch.int32)
        delay = torch.where(keep > 0, 0, 50000).to(torch.int64)
        out = torch.empty_like(x)
        total = torch.where(keep[:, None] > 0, x * 2, 0).sum().expand(8)
        s0 = torch.randn_like(x)
        s, a, flag = s0.clone(), torch.empty_like(x), torch.empty_like(x)

        def poison_read() -> None:
            s.copy_(s0)
            a.fill_(float("nan"))

        cases = (
            (
                gated_overwrite,
                (x, keep, out, delay),
                _nan(out),
                lambda result: _close(result, total),
            ),
            (
                gated_overwrite_after_read,
                (x, keep, s, a, flag, delay),
                poison_read,
                lambda _result: torch.equal(a, s0 + 1),
            ),
        )
        config = helion.Config(**_DYNAMIC)
        for fn, args, poison, good in cases:
            with self.subTest(fn=fn.__name__):
                kernel = helion.kernel(fn, config=config, static_shapes=True)
                self.assertEqual(_bad_launches(partial(kernel, *args), poison, good), 0)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    @skipUnlessHostTensorDescriptor("host tensor descriptors are required")
    def test_wait_precedes_reads_through_a_descriptor_of_a_written_tensor(
        self,
    ) -> None:
        # ``a`` is stored through pointers and read through a host TMA
        # descriptor over its storage.
        x = torch.randn(16, 512, device=DEVICE, dtype=torch.bfloat16)
        w = (0.05 * torch.randn(256, 512, device=DEVICE)).bfloat16()
        a = torch.empty_like(x)
        args = (x, w, a)
        options: dict[str, Any] = {
            **_DYNAMIC,
            "num_stages": 3,
            "indexing": "tensor_descriptor",
            "host_tensor_descriptors": True,
        }
        config = helion.Config(**options)

        def good(out: torch.Tensor) -> bool:
            return _close(out, w.float() @ a.float().T)

        kernel = helion.kernel(tma_alias, config=config, static_shapes=True)
        self.assertEqual(_bad_launches(partial(kernel, *args), _nan(a), good), 0)
        lines = kernel.bind(args).to_triton_code(config).splitlines()
        read = next(i for i, line in enumerate(lines) if "a_desc.load(" in line)
        self.assertTrue(
            _polled_before(lines, read, "(?:root_barrier|readiness|nested_loop)")
        )


@onlyBackends(["triton"])
@skipIfNotCUDA()
class TestOpaqueRoots(TestCase):
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_roots_with_effects_the_graph_does_not_record(self) -> None:
        n = 64
        args = (
            torch.zeros(n, device=DEVICE),
            torch.zeros(n, 8, device=DEVICE, dtype=torch.uint8),
            torch.zeros(n, device=DEVICE, dtype=torch.int32),
            torch.zeros(n, device=DEVICE),
        )
        kernel = helion.kernel(opaque_kinds, static_shapes=True)
        host_function = kernel.bind(args).host_function
        assert host_function is not None
        device_ir = host_function.device_ir
        # Loads, stores, pure inline assembly, and relaxed or release
        # atomics are recorded or have no effect on memory.  An acquire
        # atomic, inline Triton (with or without kernel tensors), inline
        # assembly with side effects and a fp4 load (the tensor's accesses
        # are not recorded) are opaque.
        self.assertEqual(cross_loop_codegen._opaque_roots(device_ir), {3, 4, 5, 6, 7})
        # Of those, the atomic and the fp4 load access only the tensors they
        # are passed.
        self.assertEqual(
            cross_loop_codegen._opaque_roots(device_ir, addresses_only=True),
            {4, 5, 6},
        )
