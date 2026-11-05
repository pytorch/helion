"""CPU-only native operand lifetime checks; run directly to avoid GPU conftest."""

from __future__ import annotations

import ast
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CUTE = ROOT / "helion/_compiler/cute"


def _source_module(name: str, path: Path, **namespace: object) -> ModuleType:
    """Load real pure emitters without importing Helion's GPU-aware package."""
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.level)
    ]
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(namespace)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


EXTENDED_STUB = type("_ExtendedAST", (ast.AST,), {})
RW = _source_module(
    "_cpu_native_rw",
    ROOT / "helion/_compiler/ast_read_writes.py",
    ExtendedAST=EXTENDED_STUB,
)
RECIPE = _source_module(
    "_cpu_native_recipe", CUTE / "scalar_recipe.py", ExtendedAST=EXTENDED_STUB
)
TMEM = _source_module("_cpu_tmem_operand", CUTE / "collective_tmem_operand.py")
NATIVE = _source_module(
    "_cpu_native_pipeline",
    CUTE / "collective_tcgen05.py",
    CollectiveTmemOperand=TMEM.CollectiveTmemOperand,
    ast_rename=RW.ast_rename,
    _clone=RECIPE._clone,
)
CONFIG = _source_module("_cpu_native_config", ROOT / "helion/runtime/config.py")


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _plan(*, stages: int = 2, bm: int = 128, bn: int = 64, bk: int = 32):
    return NATIVE.CollectiveTcgen05Plan(
        "p",
        "tid",
        bm,
        bn,
        bk,
        "cutlass.BFloat16",
        NATIVE.CollectiveTmemResource("r", bn),
        a_in_tmem=True,
        operand_stages=stages,
    )


def _schedule(plan, start: int, stop: int, *, asynchronous: bool = False):
    loop = ast.parse(
        f"for k in range(cutlass.Int32({start}), cutlass.Int32({stop}), {plan.bk}):\n"
        "    pass"
    ).body[0]
    return plan.pipeline_loop(
        loop,
        ast.parse("produce_a(k)").body[0],
        ast.parse("publish_a()").body,
        ast.parse("produce_b(k, p_b)").body[0],
        _expr(str(start)),
        _expr(str(stop)),
        async_copies=asynchronous,
    )


class _Int32(int):
    def __new__(cls, value):
        value = int(value) & ((1 << 32) - 1)
        return super().__new__(cls, value - (1 << 32) if value >= (1 << 31) else value)

    def __mul__(self, other):
        return _Int32(int(self) * int(other))


class _Stage:
    def __init__(self, slot: int):
        self.slot = slot

    def __getitem__(self, unused):
        return self


class _Ownership:
    """Worst case: MMA stays live through every next-tile producer operation."""

    def __init__(self, plan, start: int, stop: int, *, asynchronous: bool):
        self.plan, self.start, self.stop = plan, start, stop
        self.asynchronous = asynchronous
        self.register_a = None
        self.tmem_a = None
        self.b: dict[int, tuple[int, int]] = {}
        self.live = None
        self.phase = 0
        self.committed = False
        self.issued: list[int] = []
        self.read_a: list[int] = []
        self.read_b: list[int] = []
        self.overlaps = 0
        self.pending: list[tuple[int, tuple[int, int]]] = []
        self.groups: list[list[tuple[int, tuple[int, int]]]] = []
        self.namespace = {}

    def payload(self, k):
        if not isinstance(k, _Int32):
            raise AssertionError("recipe induction variable was widened")
        if not self.start <= int(k) < self.stop:
            raise AssertionError("prefetch escaped the original reduction range")
        return int(k), int(k * _Int32(2))

    def produce_a(self, k):
        self.register_a = self.payload(k)
        self.read_a.append(int(k))
        if self.live is not None:
            self.overlaps += 1
            if self.tmem_a[0] != self.live[1]:
                raise AssertionError("current TMEM A was overwritten during prefetch")

    def publish_a(self):
        if self.live is not None:
            raise AssertionError("TMEM A overwritten before MMA completion")
        self.tmem_a = self.register_a

    def produce_b(self, k, stage):
        if self.live is not None and self.live[0] == stage.slot:
            raise AssertionError("B stage overwritten before MMA completion")
        value = self.payload(k)
        self.read_b.append(int(k))
        if self.asynchronous:
            self.pending.append((stage.slot, value))
        else:
            self.b[stage.slot] = value

    def commit_copy(self):
        self.groups.append(self.pending[:])
        self.pending.clear()

    def wait_copy(self, count):
        if count != 0:
            raise AssertionError("this two-stage producer must drain its copy group")
        for group in self.groups:
            for slot, value in group:
                if self.live is not None and slot == self.live[0]:
                    raise AssertionError("async B copy completed into a live stage")
                self.b[slot] = value
        self.groups.clear()

    def gemm(self, mma, acc, a, b, acc_again):
        k = int(self.namespace["k"])
        if self.live is not None:
            if self.committed or self.live != (b.slot, k):
                raise AssertionError(
                    "dependent accumulator update before MMA completion"
                )
        else:
            if self.tmem_a[0] != k or self.b.get(b.slot, (None,))[0] != k:
                raise AssertionError("MMA read an unready or wrong operand stage")
            self.live = b.slot, k
            self.issued.append(k)

    def commit_mma(self, unused):
        if self.live is None:
            raise AssertionError("commit without MMA")
        self.committed = True

    def wait_mma(self, unused, phase):
        if self.live is None or not self.committed or phase != self.phase:
            raise AssertionError("incorrect MMA completion/phase ownership")
        self.live = None
        self.committed = False
        self.phase ^= 1

    def run(self, schedule):
        mma = SimpleNamespace(set=lambda *unused: None)
        thr = SimpleNamespace(
            partition_B=lambda tensor: tensor,
            make_fragment_B=lambda tensor: tensor,
        )
        arch = SimpleNamespace(
            fence_proxy=lambda **unused: None,
            sync_threads=lambda: None,
            make_warp_uniform=lambda value: value,
            warp_idx=lambda: 0,
            elect_one=nullcontext,
            mbarrier_wait=self.wait_mma,
            cp_async_commit_group=self.commit_copy,
            cp_async_wait_group=self.wait_copy,
        )
        cute = SimpleNamespace(
            arch=arch,
            make_tensor=lambda pointer, layout: _Stage(
                pointer // (self.plan.bn * self.plan.bk)
            ),
            recast_ptr=lambda pointer, *unused, **kw: pointer,
            gemm=self.gemm,
            nvgpu=SimpleNamespace(
                tcgen05=SimpleNamespace(
                    Field=SimpleNamespace(ACCUMULATE="accumulate"),
                    commit=self.commit_mma,
                )
            ),
        )
        self.namespace = {
            "cutlass": SimpleNamespace(
                Int32=_Int32,
                Int64=int,
                BFloat16="bf16",
                range=lambda *args, **kw: range(*args),
            ),
            "cute": cute,
            "p_bp": 0,
            "p_bl": SimpleNamespace(inner="swizzle", outer="layout"),
            "p_thr": thr,
            "p_mma": mma,
            "p_acc": object(),
            "p_ra": _Stage(0),
            "r_done": object(),
            "r_phase": 0,
            "produce_a": self.produce_a,
            "produce_b": self.produce_b,
            "publish_a": self.publish_a,
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=schedule, type_ignores=[])),
                "<native-stage-ownership>",
                "exec",
            ),
            self.namespace,
        )
        expected = list(range(self.start, self.stop, self.plan.bk))
        if (
            self.issued != expected
            or self.read_a != expected
            or self.read_b != expected
        ):
            raise AssertionError("prefetch changed the original read/compute sequence")
        if self.live is not None or self.pending or self.groups:
            raise AssertionError("pipeline returned with outstanding storage users")
        if self.overlaps != len(expected) - 1:
            raise AssertionError(
                "next A was not produced while the current MMA was live"
            )


class NativePipelineCPU(unittest.TestCase):
    def test_b_slot_stride_preserves_swizzle_atoms(self):
        for dtype in ("cutlass.Float16", "cutlass.BFloat16"):
            for bn in (32, 64):
                for bk in (16, 32, 64, 128):
                    for k_major in (False, True):
                        with self.subTest(dtype=dtype, bn=bn, bk=bk, k_major=k_major):
                            plan = replace(
                                _plan(bn=bn, bk=bk), dtype=dtype, b_k_major=k_major
                            )
                            atom_bytes = 8 * min(128, 2 * (bk if k_major else bn))
                            self.assertEqual((bn * bk * 2) % atom_bytes, 0)
                            self.assertTrue(plan.b_stage_stride_aligned())
        plan = _plan(stages=1)
        with patch.object(
            NATIVE.CollectiveTcgen05Plan, "b_stage_stride_aligned", return_value=False
        ):
            self.assertEqual(
                plan.pipeline_stages(
                    2,
                    _expr("0"),
                    _expr("128"),
                    static_shapes=True,
                    unmasked_recipes=True,
                    independent_recipes=True,
                    single_site=True,
                    no_other_smem=True,
                    capacity_bytes=1 << 20,
                ),
                1,
            )
            with self.assertRaisesRegex(ValueError, "swizzle atom"):
                replace(plan, operand_stages=2)

    def test_lowering_admission_rejects_loop_dependencies_and_masks(self):
        source = ast.parse((CUTE / "collective_matmul.py").read_text())
        lower = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "_lower_site"
        )
        gate = next(
            node
            for node in lower.body
            if isinstance(node, ast.If)
            and "cute_collective_stages" in ast.unparse(node.test)
        )
        for dependent, masked, multiple, allocated in (
            (False, False, False, False),
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (False, False, False, True),
        ):
            with self.subTest(
                dependent=dependent,
                masked=masked,
                multiple=multiple,
                allocated=allocated,
            ):
                a = "av = acc" if dependent else "av = A.load()"
                if masked:
                    a = "av = A.load() if k < 127 else 0"
                loop = ast.parse(
                    "for k in range(0, 128, 32):\n    acc = av * bv + acc"
                ).body[0]
                body = [loop]
                if allocated:
                    body += ast.parse("scratch = cute.arch.alloc_smem(dtype, 64)").body
                env = SimpleNamespace(
                    settings=SimpleNamespace(static_shapes=True),
                    device=None,
                    config_spec=SimpleNamespace(
                        _cute_tcgen05_config=SimpleNamespace(
                            per_cta_smem_capacity_bytes=lambda device: 1 << 20,
                        )
                    ),
                )
                namespace = {
                    "native": _plan(stages=1),
                    "ast": ast,
                    "replace": replace,
                    "a_statements": ast.parse(a).body,
                    "b_statements": ast.parse("bv = B.load()").body,
                    "a_value": _expr("av"),
                    "b_value": _expr("bv"),
                    "k_loop": loop,
                    "body": body,
                    "site": SimpleNamespace(k_offset="k"),
                    "ReadWrites": RW.ReadWrites,
                    "k_start": _expr("0"),
                    "k_stop": _expr("128"),
                    "df": SimpleNamespace(
                        config={"cute_collective_stages": 2},
                        cute_state=SimpleNamespace(
                            collective_mma_sites=[1, 2] if multiple else [1]
                        ),
                    ),
                    "CompileEnvironment": SimpleNamespace(current=lambda env=env: env),
                }
                exec(
                    compile(
                        ast.Module(body=[gate], type_ignores=[]),
                        "<admission-gate>",
                        "exec",
                    ),
                    namespace,
                )
                self.assertEqual(
                    namespace["native"].operand_stages,
                    1 if dependent or masked or multiple or allocated else 2,
                )

    def test_emitted_producer_wait_publication_order(self):
        plan = _plan()
        schedule = _schedule(plan, 0, 128, asynchronous=True)
        loop = schedule[-1]
        prefetch_index = next(
            i
            for i, node in enumerate(loop.body)
            if isinstance(node, ast.If) and "p_next_k" in ast.unparse(node.test)
        )
        wait_index = next(
            i
            for i, node in enumerate(loop.body)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "cute.arch.mbarrier_wait"
        )
        publish_index = next(
            i
            for i, node in enumerate(loop.body)
            if isinstance(node, ast.If) and "publish_a" in ast.unparse(node)
        )
        self.assertLess(prefetch_index, wait_index)
        self.assertLess(wait_index, publish_index)
        self.assertIn(
            "cute.gemm",
            ast.unparse(ast.Module(body=loop.body[:prefetch_index], type_ignores=[])),
        )
        self.assertNotIn("publish_a", ast.unparse(loop.body[prefetch_index]))
        self.assertIn("produce_a", ast.unparse(loop.body[prefetch_index]))
        self.assertIn("produce_b", ast.unparse(loop.body[prefetch_index]))

    def test_admission_and_budget_boundary(self):
        plan = _plan(stages=1, bk=128)
        facts = {
            "static_shapes": True,
            "unmasked_recipes": True,
            "independent_recipes": True,
            "single_site": True,
            "no_other_smem": True,
            "capacity_bytes": 2 * 64 * 128 * 2 + 128 * 64 * 4 + 4096,
        }
        self.assertEqual(plan.pipeline_stages(2, _expr("0"), _expr("256"), **facts), 2)
        facts["capacity_bytes"] -= 1
        self.assertEqual(plan.pipeline_stages(2, _expr("0"), _expr("256"), **facts), 1)
        facts["capacity_bytes"] = 1 << 20
        for flag in (
            "static_shapes",
            "unmasked_recipes",
            "independent_recipes",
            "single_site",
            "no_other_smem",
        ):
            with self.subTest(flag=flag):
                self.assertEqual(
                    plan.pipeline_stages(
                        2, _expr("0"), _expr("256"), **(facts | {flag: False})
                    ),
                    1,
                )
        for requested in (1, 3, 4, True):
            self.assertEqual(
                plan.pipeline_stages(requested, _expr("0"), _expr("256"), **facts), 1
            )
        for first, last in (
            ("0", "0"),
            ("0", "128"),
            ("0", "257"),
            ("-1", "255"),
            ("limit", "256"),
            ("0", "2147483648"),
        ):
            self.assertEqual(
                plan.pipeline_stages(2, _expr(first), _expr(last), **facts), 1
            )
        tf32 = replace(plan, dtype="cutlass.TFloat32", a_in_tmem=False)
        self.assertEqual(tf32.pipeline_stages(2, _expr("0"), _expr("256"), **facts), 1)

    def test_setup_changes_only_b_capacity(self):
        one, two = _plan(stages=1), _plan(stages=2)
        old, new = (
            ast.unparse(ast.Module(body=one.setup("tid"), type_ignores=[])),
            ast.unparse(ast.Module(body=two.setup("tid"), type_ignores=[])),
        )
        self.assertIn("alloc_smem(cutlass.BFloat16, 4096", new)
        self.assertEqual(
            old.replace(
                "BFloat16, 2048, alignment=1024", "BFloat16, 4096, alignment=1024"
            ),
            new,
        )
        self.assertEqual(
            one.resource.allocation_columns, two.resource.allocation_columns
        )
        pool = NATIVE.CollectiveOperandSmem("pool")
        replace(two, operands=pool).setup("tid")
        self.assertEqual(pool.a_bytes, 0)
        self.assertEqual(pool.b_bytes, two.bn * two.bk * 2 * 2)

    def test_stage_one_issue_wait_is_unchanged(self):
        plan = _plan(stages=1)
        source = ast.unparse(
            ast.Module(body=plan.compute(_expr("k"), _expr("0")), type_ignores=[])
        )
        self.assertLess(source.index("cute.gemm"), source.index("tcgen05.commit"))
        self.assertLess(source.index("tcgen05.commit"), source.index("mbarrier_wait"))
        self.assertTrue(source.endswith("cute.arch.sync_threads()"))
        with self.assertRaises(ValueError):
            _schedule(plan, 0, 128)

    def test_ownership_and_typed_read_order(self):
        for asynchronous in (False, True):
            for bm, bn, bk in ((64, 32, 16), (64, 64, 64), (128, 64, 128)):
                for tiles in (2, 3, 7, 32):
                    for start in (0, (1 << 31) - (tiles + 1) * bk):
                        with self.subTest(
                            async_copy=asynchronous,
                            tile=(bm, bn, bk),
                            tiles=tiles,
                            start=start,
                        ):
                            plan = _plan(bm=bm, bn=bn, bk=bk)
                            stop = start + tiles * bk
                            _Ownership(
                                plan, start, stop, asynchronous=asynchronous
                            ).run(
                                _schedule(plan, start, stop, asynchronous=asynchronous)
                            )

    def test_simulator_catches_early_tmem_publication(self):
        plan = _plan()
        schedule = _schedule(plan, 0, 128)
        loop = schedule[-1]
        prefetch = next(
            node
            for node in loop.body
            if isinstance(node, ast.If) and "p_next_k" in ast.unparse(node.test)
        )
        prefetch.body.append(ast.parse("publish_a()").body[0])
        with self.assertRaisesRegex(AssertionError, "TMEM A overwritten"):
            _Ownership(plan, 0, 128, asynchronous=False).run(schedule)

    def test_simulator_catches_live_b_overwrite(self):
        plan = _plan()
        schedule = _schedule(plan, 0, 128)
        loop = schedule[-1]
        prefetch = next(
            node
            for node in loop.body
            if isinstance(node, ast.If) and "p_next_k" in ast.unparse(node.test)
        )
        for node in ast.walk(prefetch):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
                node.right = ast.Constant(0)
        with self.assertRaisesRegex(AssertionError, "B stage overwritten"):
            _Ownership(plan, 0, 128, asynchronous=False).run(schedule)

    def test_simulator_catches_missing_mma_wait(self):
        plan = _plan()
        schedule = _schedule(plan, 0, 128)
        loop = schedule[-1]
        loop.body = [
            node
            for node in loop.body
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "cute.arch.mbarrier_wait"
            )
        ]
        with self.assertRaises(AssertionError):
            _Ownership(plan, 0, 128, asynchronous=False).run(schedule)

    def test_config_roundtrip_keeps_requested_stage_and_semantics(self):
        value = CONFIG.Config.from_dict(
            {
                "block_sizes": [128, 64, 32],
                "cute_collective_compute": "tcgen05",
                "cute_collective_tmem_a": True,
                "cute_collective_stages": 2,
            }
        )
        restored = CONFIG.Config.from_json(value.to_json())
        self.assertEqual(dict(value), dict(restored))
        self.assertEqual(restored["cute_collective_stages"], 2)
        old, new = _plan(stages=1), _plan(stages=2)
        for method in ("seed_from_shared",):
            self.assertEqual(
                ast.dump(ast.Module(body=getattr(old, method)(), type_ignores=[])),
                ast.dump(ast.Module(body=getattr(new, method)(), type_ignores=[])),
            )
        self.assertEqual(
            ast.dump(
                ast.Module(body=old.finish(_expr("0"), _expr("128")), type_ignores=[])
            ),
            ast.dump(
                ast.Module(body=new.finish(_expr("0"), _expr("128")), type_ignores=[])
            ),
        )


if __name__ == "__main__":
    if "torch" in sys.modules:
        raise RuntimeError(
            "CPU lifetime tests must not import torch or initialize CUDA"
        )
    unittest.main()
