"""Code layout of static persistent cross-loop kernels (``cross_loop_code_layout``).

Every worker runs the root blocks it belongs to in textual order, so the layout
rule may only swap blocks that share no worker.  CPU tests check the rule
(:func:`code_layout_order`) on its own; GPU tests check the generated kernels
(block order, every worker's block sequence, outputs) and a race run of a
reordered kernel against a late producer, with a negative control that drops
the consumer's wait.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from unittest import mock

import torch

import helion
from helion._compiler.cross_loop_codegen import code_layout_order
from helion._compiler.cross_loop_codegen import static_root_workers
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl


def block_sequences(
    order: tuple[int, ...], workers: dict[int, frozenset[int] | None], width: int
) -> dict[int, list[int]]:
    """Each worker's sequence of the blocks it runs, in textual order ``order``
    (``None``: the root may run on any worker)."""
    return {
        w: [r for r in order if workers[r] is None or w in workers[r]]
        for w in range(width)
    }


class TestCodeLayoutRule(TestCase):
    def test_static_root_workers_follow_the_rotated_lanes(self) -> None:
        self.assertEqual(static_root_workers(6, 0, 148), frozenset(range(6)))
        # lane (w + 142) % 148 < 6: workers 6..11
        self.assertEqual(static_root_workers(6, 142, 148), frozenset(range(6, 12)))
        # wrapping: lane (w + 14) % 148 < 32 -> workers 134..147 and 0..17
        self.assertEqual(
            static_root_workers(32, 14, 148),
            frozenset([*range(134, 148), *range(18)]),
        )
        # multi-wave roots run on every worker
        self.assertEqual(static_root_workers(300, 5, 148), frozenset(range(148)))

    def test_every_worker_keeps_its_block_sequence(self) -> None:
        rng = random.Random(7)
        width = 24
        for _ in range(400):
            n = rng.randint(3, 9)
            roots = tuple(range(n))
            workers: dict[int, frozenset[int] | None] = {}
            for r in roots:
                if rng.random() < 0.08:
                    workers[r] = None
                else:
                    count = rng.randint(1, width)
                    workers[r] = static_root_workers(
                        count, rng.randrange(width), width
                    )
            deps = [
                (a, b) for a in roots for b in roots if a < b and rng.random() < 0.3
            ]
            costs = {r: rng.choice([0, 1, 8, 64, 512, 4096]) for r in roots}
            order = code_layout_order(roots, workers, deps, costs)
            self.assertEqual(sorted(order), list(roots))
            self.assertEqual(
                block_sequences(order, workers, width),
                block_sequences(roots, workers, width),
            )

    def test_sequence_check_catches_a_mutant_order(self) -> None:
        # The invariant the GPU tests also check has teeth: swapping two blocks
        # that share a worker changes that worker's sequence.
        workers: dict[int, frozenset[int] | None] = {
            0: frozenset({0, 1}),
            1: frozenset({2}),
            2: frozenset({1, 3}),
        }
        good = block_sequences((1, 0, 2), workers, 4)
        self.assertEqual(good, block_sequences((0, 1, 2), workers, 4))
        self.assertNotEqual(block_sequences((2, 1, 0), workers, 4), good)

    def _attention_shape(
        self,
    ) -> tuple[tuple[int, ...], dict[int, frozenset[int] | None], list[tuple[int, int]], dict[int, int]]:
        """The GLM-5 one-kernel attention megakernel's roots (R K Q P N A UV O),
        packed placement on 148 workers, bytes per task from its tiles."""
        width = 148
        tasks = (6, 6, 128, 8, 32, 102, 32, 96)
        rotations, cursor = [], 0
        for count in tasks:
            rotations.append(-cursor % width)
            cursor = (cursor + count) % width
        workers: dict[int, frozenset[int] | None] = {
            r: static_root_workers(tasks[r], rotations[r], width) for r in range(8)
        }
        R, K, Q, P, N, A, UV, O = range(8)
        deps = [(R, Q), (Q, P), (Q, N), (K, A), (P, A), (N, A), (A, UV), (UV, O)]
        costs = {
            R: 8 << 10,
            K: 3 << 10,
            Q: 132 << 10,
            P: 2 << 10,
            N: 99 << 10,
            A: 110 << 10,
            UV: 132 << 10,
            O: 520 << 10,
        }
        return tuple(range(8)), workers, deps, costs

    def test_slack_roots_move_behind_the_critical_path(self) -> None:
        roots, workers, deps, costs = self._attention_shape()
        R, K, Q, P, N, A, UV, O = roots
        order = code_layout_order(roots, workers, deps, costs)
        # P (short; N feeds A later) moves behind A; K shares workers 6..11
        # with N, so it is pulled in right before N.
        self.assertEqual(order, (R, Q, K, N, A, P, UV, O))
        self.assertEqual(
            block_sequences(order, workers, 148),
            block_sequences(roots, workers, 148),
        )

    def test_source_order_without_costs_or_with_unknown_workers(self) -> None:
        roots, workers, deps, costs = self._attention_shape()
        self.assertEqual(
            code_layout_order(roots, workers, deps, dict.fromkeys(roots, 0)), roots
        )
        # A root whose workers are unknown keeps its place relative to all.
        unknown = dict(workers)
        unknown[3] = None
        order = code_layout_order(roots, unknown, deps, costs)
        self.assertEqual(order.index(3), 3)
        self.assertEqual(set(order[:3]), {0, 1, 2})

    def test_disjoint_roots_are_laid_out_by_start_time(self) -> None:
        # Three roots on disjoint workers: the slack source goes last.
        workers: dict[int, frozenset[int] | None] = {
            0: frozenset({0}),
            1: frozenset({1, 2}),
            2: frozenset({3}),
        }
        order = code_layout_order((0, 1, 2), workers, [(1, 2)], {0: 1, 1: 100, 2: 10})
        self.assertEqual(order, (1, 2, 0))

    def test_layout_is_on_by_default(self) -> None:
        # (ATT7 verify, mutant L8) the rule is automatic: no setting or environment variable is needed to get it.
        env = {k: v for k, v in os.environ.items() if k != "HELION_CROSS_LOOP_CODE_LAYOUT"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(helion.Settings().cross_loop_code_layout)


@helion.kernel(static_shapes=True)
def _slack_root_kernel(
    x: torch.Tensor, z: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Root 0 is long (a K loop) and feeds root 2 (a different tile size, so it
    stays a separate root); root 1 is short and independent (slack). Packed
    placement gives the three roots disjoint workers; the layout rule lays out
    0, 2, then 1."""
    m, k = x.shape
    zo = torch.empty_like(z)
    for tm in hl.tile(m, block_size=1):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tk in hl.tile(k, block_size=128):
            acc = acc + torch.sum(x[tm, tk].to(torch.float32), dim=-1)
        y[tm] = acc
    for tz in hl.tile(z.size(0), block_size=1):
        zo[tz] = z[tz] * 2
    for tn in hl.tile(m, block_size=4):
        out[tn] = y[tn] + 1
    return zo


# A 20 us delay of allowlisted assembly (the producer root writes y late).
_SPIN_20US = """
t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
t1 = t0
while t1 - t0 < 20000:
    t1 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
0
"""


@helion.kernel(static_shapes=True)
def _late_producer_kernel(
    x: torch.Tensor, z: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """_slack_root_kernel with root 0 writing y 20 us late."""
    m, k = x.shape
    zo = torch.empty_like(z)
    for tm in hl.tile(m, block_size=1):
        hl.inline_triton(_SPIN_20US, args={}, output_like=None)
        acc = hl.zeros([tm], dtype=torch.float32)
        for tk in hl.tile(k, block_size=128):
            acc = acc + torch.sum(x[tm, tk].to(torch.float32), dim=-1)
        y[tm] = acc
    for tz in hl.tile(z.size(0), block_size=1):
        zo[tz] = z[tz] * 2
    for tn in hl.tile(m, block_size=4):
        out[tn] = y[tn] + 1
    return zo


_STATIC = {
    "block_sizes": [],
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "static",
    "num_sm_multiplier": 1,
    "num_warps": 4,
    "num_stages": 2,
    "indexing": "pointer",
}

_GUARD = re.compile(
    r"^    if (?:\(tl\.program_id\(0\) \+ (\d+)\) % (\d+)|tl\.program_id\(0\)) >= 0 and "
    r"(?:\(tl\.program_id\(0\) \+ \d+\) % \d+|tl\.program_id\(0\)) < (\d+):$"
)
_SEGMENT = re.compile(r"tl\.range\(.*? - 0 \+ (\d+), \d+, \d+\)")


def generated_blocks(code: str, width: int) -> list[tuple[int, frozenset[int] | None]]:
    """(root segment base, workers) of each top-level root block of the
    generated persistent kernel, in textual order."""
    lines = code.split("\n")
    k0 = next(i for i, line in enumerate(lines) if line.startswith("def _helion_"))
    k1 = next(
        (i for i in range(k0 + 1, len(lines)) if lines[i].startswith("def ")),
        len(lines),
    )
    out = []
    for i in range(k0 + 1, k1):
        line = lines[i]
        m = _GUARD.match(line)
        if m is None and not line.startswith("    for virtual_pid in tl.range("):
            continue
        j = i
        while _SEGMENT.search(lines[j]) is None:
            j += 1
        base = int(_SEGMENT.search(lines[j]).group(1))  # pyrefly: ignore[missing-attribute]
        if m is None:
            workers = None
        else:
            rot = int(m.group(1) or 0)
            mod = int(m.group(2) or width)
            workers = static_root_workers(int(m.group(3)), rot, mod)
        out.append((base, workers))
    return out


def _load(code: str, name: str, directory: str) -> object:
    path = Path(directory) / f"{name}.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@onlyBackends(["triton"])
class TestCodeLayoutGenerated(TestCase):
    def _inputs(self) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(0)
        x = torch.randn(16, 1024, device=DEVICE, dtype=torch.float32)
        z = torch.randn(4, device=DEVICE, dtype=torch.float32)
        y = torch.empty(16, device=DEVICE, dtype=torch.float32)
        out = torch.empty(16, device=DEVICE, dtype=torch.float32)
        return x, z, y, out

    def _code(
        self,
        fn: object,
        layout: bool | None,
        args: tuple[torch.Tensor, ...],
        **overrides: object,
    ) -> str:
        """Generated code with the rule forced on / off, or (``None``) with the
        environment variable unset (the default)."""
        env = {k: v for k, v in os.environ.items() if k != "HELION_CROSS_LOOP_CODE_LAYOUT"}
        if layout is not None:
            env["HELION_CROSS_LOOP_CODE_LAYOUT"] = "1" if layout else "0"
        config = {**_STATIC, **overrides}
        with mock.patch.dict(os.environ, env, clear=True):
            kernel = helion.kernel(
                fn.fn,  # pyrefly: ignore[missing-attribute]
                config=helion.Config(**config),  # pyrefly: ignore[bad-argument-type]
                static_shapes=True,
            )
            self.assertEqual(kernel.settings.cross_loop_code_layout, layout is not False)
            bound = kernel.bind(args)
            return bound.to_triton_code(helion.Config(**config))  # pyrefly: ignore[bad-argument-type]

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_default_codegen_is_laid_out(self) -> None:
        # (ATT7 verify, mutant L8) the default compile applies the rule.
        args = self._inputs()
        default = self._code(_slack_root_kernel, None, args)
        self.assertEqual(default, self._code(_slack_root_kernel, True, args))
        self.assertNotEqual(default, self._code(_slack_root_kernel, False, args))

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_packet_dispatch_keeps_source_order(self) -> None:
        # Packet dispatch hands tasks to whichever worker claims them, so no block
        # has a static worker set: the rule must leave the text alone.
        args = self._inputs()
        on = self._code(_slack_root_kernel, True, args, cross_loop_pipeline="dynamic")
        off = self._code(_slack_root_kernel, False, args, cross_loop_pipeline="dynamic")
        self.assertEqual(on, off)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_blocks_follow_the_layout_rule_and_keep_every_worker_sequence(self) -> None:
        width = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        args = self._inputs()
        source = self._code(_slack_root_kernel, False, args)
        laid = self._code(_slack_root_kernel, True, args)
        src_blocks = generated_blocks(source, width)
        new_blocks = generated_blocks(laid, width)
        self.assertEqual(len(src_blocks), 3)
        # source order 0, 1, 2 (segment bases increase); laid out 0, 2, 1
        bases = sorted(b for b, _ in src_blocks)
        self.assertEqual([b for b, _ in src_blocks], bases)
        self.assertEqual([b for b, _ in new_blocks], [bases[0], bases[2], bases[1]])
        # disjoint workers (packed placement): the reorder is free
        self.assertFalse(src_blocks[0][1] & src_blocks[1][1] or src_blocks[1][1] & src_blocks[2][1])
        workers = dict(src_blocks)
        self.assertEqual(
            block_sequences(tuple(b for b, _ in new_blocks), workers, width),
            block_sequences(tuple(bases), workers, width),
        )
        # the laid-out text is a permutation of the source-order blocks
        self.assertEqual(sorted(laid.split("\n")), sorted(source.split("\n")))
        x, z, y, out = args
        with tempfile.TemporaryDirectory() as d:
            mod = _load(laid, "_helion_test_layout_laid", d)
            for launch in range(3):
                zo = mod._slack_root_kernel(x + launch, z, y, out)  # pyrefly: ignore[missing-attribute]
                torch.cuda.synchronize()
                expect = (x + launch).sum(-1)
                torch.testing.assert_close(zo, z * 2)
                torch.testing.assert_close(y, expect, rtol=1e-5, atol=1e-4)
                torch.testing.assert_close(out, expect + 1, rtol=1e-5, atol=1e-4)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_reordered_consumer_waits_for_a_late_producer(self) -> None:
        width = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        args = self._inputs()
        laid = self._code(_late_producer_kernel, True, args)
        blocks = generated_blocks(laid, width)
        bases = sorted(b for b, _ in blocks)
        self.assertEqual([b for b, _ in blocks], [bases[0], bases[2], bases[1]])
        # Negative control: drop the consumer's readiness spin (root 2, laid out
        # second, is the only root with one): it reads y before the late
        # producer writes it.
        lines = laid.split("\n")
        spins = [
            i
            for i, line in enumerate(lines)
            if re.match(r"\s*while tile_dependency_readiness_wait != ", line)
        ]
        self.assertTrue(spins)
        for i in spins:
            lines[i] = re.sub(r"while (\w+) != .*:$", r"while \1 != \1:", lines[i])
        nowait = "\n".join(lines)
        x, z, y, out = args
        expect = x.sum(-1) + 1
        with tempfile.TemporaryDirectory() as d:
            for name, code, want_exact in (
                ("_helion_test_layout_late", laid, True),
                ("_helion_test_layout_nowait", nowait, False),
            ):
                mod = _load(code, name, d)
                wrong = 0
                for _ in range(20):
                    y.fill_(float("nan"))  # poisoned until the late root writes it
                    out.fill_(float("nan"))
                    mod._late_producer_kernel(x, z, y, out)  # pyrefly: ignore[missing-attribute]
                    torch.cuda.synchronize()
                    wrong += not torch.allclose(out, expect, rtol=1e-5, atol=1e-4)
                if want_exact:
                    self.assertEqual(wrong, 0)
                else:
                    self.assertGreater(wrong, 0)
