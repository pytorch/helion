"""``hl.barrier()`` on Metal without a GPU: what codegen plans for ticketed
phases (``metal/ticketed_phases.py``) and which buffers it makes coherent
(``metal/msl_phases.py``).  Binds CPU tensors only; ``test_metal_phases.py``
runs the same kernels."""

from __future__ import annotations

import ast
import unittest

import pytest
import torch

import helion
from helion import exc
from helion._compiler.metal.msl_phases import PHASE_ACCESS_GLOBAL
from helion._compiler.metal.msl_phases import coherent_buffers
from helion._compiler.metal.msl_phases import coherent_param_prologue
from helion._compiler.metal.search_surface import kernel_facts
from helion._compiler.metal.ticketed_phases import launch_dims
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_phases import bounded_by_tensor
from helion._testing.metal_phases import matvec_then_norm
from helion._testing.metal_phases import pingpong
from helion._testing.metal_phases import row_sums_then_scale
from helion._testing.metal_phases import scale_by_first
from helion._testing.metal_phases import scale_then_matmul
from helion._testing.metal_phases import scale_then_reverse
from helion._testing.metal_phases import three_phases
from helion._testing.metal_phases import two_phase_matmuls
from helion._testing.metal_phases import write_then_matmul

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

Access = dict[str, tuple[tuple[int, ...], tuple[int, ...]]]


def _phase_access(code: str) -> Access:
    """The phases that read and that store each argument, as the generated
    module records them."""
    (line,) = [
        line
        for line in code.splitlines()
        if line.startswith(f"{PHASE_ACCESS_GLOBAL} = ")
    ]
    return ast.literal_eval(line.split(" = ", 1)[1])


def _coherent(access: Access, shared: tuple[str, ...] = ()) -> set[str]:
    """The coherent buffers when only the ``shared`` arguments share a
    storage."""
    storages = {name: 0 if name in shared else i + 1 for i, name in enumerate(access)}
    return coherent_buffers(access, storages)


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


class TestMetalPhaseCodegen(unittest.TestCase):
    def test_tickets_waits_arrivals(self) -> None:
        """One ticket per threadgroup; every phase but the first waits and
        every phase but the last arrives.  ``a`` and ``b`` are each written
        by one phase and read by the next, so both are coherent."""
        a = torch.randn(4096)
        code = pingpong.bind((a, torch.empty_like(a))).to_code(
            helion.Config(block_sizes=[64] * 4, pid_type="flat")
        )
        self.assertEqual(code.count("helion_phase.ticket("), 1)
        self.assertEqual(code.count("helion_phase.wait("), 3)
        self.assertEqual(code.count("helion_phase.arrive("), 3)
        self.assertEqual(_coherent(_phase_access(code)), {"a", "b"})

    def test_roots_and_views(self) -> None:
        """One arrival per root of the first phase, one wait in the second;
        an intermediate written through one view and read through another
        is one buffer, coherent through both."""
        x = torch.randn(64, 96)
        code = row_sums_then_scale.bind((x,)).to_code(
            helion.Config(block_sizes=[1, 128, 4, 32], pid_type="flat")
        )
        self.assertEqual(code.count("helion_phase.arrive("), 2)
        self.assertEqual(code.count("helion_phase.wait("), 1)
        access = _phase_access(code)
        self.assertEqual(access["y"], ((1,), ()))
        self.assertEqual(access["y2"], ((), (0,)))
        self.assertEqual(_coherent(access, shared=("y", "y2")), {"s", "y", "y2"})

    def test_reduction_widens_no_root(self) -> None:
        """The row sums' reduction over x widens the layout recorded for every
        root, but the stores of the root that writes ``y2`` depend only on
        y: its surplus threads along x don't write either."""
        x = torch.randn(64, 96)
        msl = render_msl(
            row_sums_then_scale,
            (x,),
            helion.Config(block_sizes=[1, 128, 4, 32], pid_type="flat"),
        )
        self.assertRegex(
            msl,
            r"if \(\(\(\(tid\[0\] < 1\) && \(tid\[2\] < 1\)\) && mask_\d+\)\) "
            r"\{\s*y2__coherent\[",
        )

    def test_root_layouts(self) -> None:
        """Roots with different thread layouts (32 x 8 and 256) share one flat
        256-thread launch, each laying it out its own way."""
        x = torch.randn(4, 512)
        w = torch.randn(1024, 512)
        code = matvec_then_norm.bind((x, w)).to_code(
            helion.Config(
                block_sizes=[32, 256, 4, 8, 1],
                num_threads=[32, 256, 1, 8, 1],
                num_warps=8,
                pid_type="flat",
            )
        )
        self.assertIn("_block_dims=(256, 1, 1)", code)
        self.assertEqual(code.count("helion_phase.layout("), 2)

    def test_empty_phase_branches(self) -> None:
        """Task counts known only at run time: phase 2 waits for phase 1 when
        it has tasks and else for phase 0 when that has any, and phase 1
        waits for phase 0 when it has tasks.  A phase without tasks raises no
        flag (an empty input, an MoE expert without tokens)."""
        code = three_phases.bind((torch.randn(64), torch.randn(32))).to_code(
            helion.Config(block_sizes=[64, 64, 64], pid_type="flat")
        )
        fn = next(
            node
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.FunctionDef) and node.name == "_helion_three_phases"
        )
        waits = {
            ast.unparse(node.body[0]): node
            for node in ast.walk(fn)
            if isinstance(node, ast.If)
            and len(node.body) == 1
            and "helion_phase.wait(" in ast.unparse(node.body[0])
        }
        wait_1 = waits["helion_phase.wait(_helion_phase_state, 3, 1, tid)"]
        self.assertIn("m", _names(wait_1.test))
        (wait_0,) = wait_1.orelse
        assert isinstance(wait_0, ast.If)
        self.assertEqual(
            ast.unparse(wait_0.body[0]),
            "helion_phase.wait(_helion_phase_state, 3, 0, tid)",
        )
        self.assertIn("n", _names(wait_0.test))
        self.assertEqual(wait_0.orelse, [])
        self.assertEqual(code.count("helion_phase.wait("), 3)

    def test_coherent_buffers(self) -> None:
        """A buffer is coherent when a phase writes it and a later phase reads
        or writes it: not when it is only read, only read before its
        writes, or written by the last phase that touches it."""
        access: Access = {
            "inputs": ((0, 1, 2), ()),
            "read_then_written": ((0,), (1,)),
            "handed_on": ((1,), (0,)),
            "rewritten": ((), (0, 2)),
            "updated_in_place": ((0, 2), (0,)),
            "output": ((), (2,)),
            "scratch": ((1,), (1,)),
        }
        handed_on = {"handed_on", "rewritten", "updated_in_place"}
        self.assertEqual(_coherent(access), handed_on)
        # Views of one storage are one buffer: written last, it stays plain;
        # written by two phases through two views, or handed on and read
        # through another view, every view is coherent.
        self.assertEqual(_coherent(access, shared=("inputs", "output")), handed_on)
        self.assertEqual(
            _coherent(access, shared=("read_then_written", "output")),
            handed_on | {"read_then_written", "output"},
        )
        self.assertEqual(
            _coherent(access, shared=("handed_on", "inputs")), handed_on | {"inputs"}
        )

    def test_partial_lines(self) -> None:
        """``x``, updated in place in phase 0, and ``y``, written there, are
        both read by phase 1: both coherent, ``out`` not."""
        x = torch.randn(1 << 12)
        code = scale_then_reverse.bind((x, torch.empty_like(x))).to_code(
            helion.Config(block_sizes=[16, 16], pid_type="flat")
        )
        self.assertEqual(_coherent(_phase_access(code)), {"x", "y"})

    def test_coherent_scalar_tensor(self) -> None:
        """A 0-d tensor that a phase writes and a later phase reads is read
        through its coherent alias, cast from its pointer, after the wait."""
        msl = render_msl(
            scale_by_first,
            (torch.randn(64), torch.ones(())),
            helion.Config(block_sizes=[1, 64], pid_type="flat"),
        )
        self.assertIn("device float* scale [[buffer(", msl)
        self.assertIn(
            "device coherent(device) float* scale__coherent = "
            "reinterpret_cast<device coherent(device) float*>(scale);",
            msl,
        )
        self.assertNotIn("_buf_scale", msl)

    def test_surplus_threads_dont_write(self) -> None:
        """Phase 0's root, laid out in one thread, runs on all 64 of the
        launch phase 1 needs: only its own thread stores, or the in-place
        update of ``x`` could land once per SIMD group.  Phase 1's root
        takes every thread, so its store needs no guard."""
        x = torch.randn(1 << 12)
        msl = render_msl(
            scale_then_reverse,
            (x, torch.empty_like(x)),
            helion.Config(block_sizes=[1, 64], pid_type="flat"),
        )
        for name in ("x", "y"):
            self.assertRegex(
                msl, rf"if \(\(tid\[0\] < 1\)\) \{{\s*{name}__coherent\[\w+\] = "
            )
        self.assertNotIn("tid[0] < 64", msl)
        self.assertNotRegex(msl, r"if \(\(tid\[0\] < 1\)\) \{\s*helion_phase::")

    def test_oversized_launch_refused(self) -> None:
        """Roots that each fit a threadgroup (512 x 1 and 32 x 8 threads)
        can share a per-axis maximum (512 x 8) past Metal's 1024 threads: the
        config is refused at codegen, not when Metal builds the pipeline."""
        x, w = torch.randn(4, 512), torch.randn(1024, 512)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "more than 1024 threads per threadgroup"
        ):
            matvec_then_norm.bind((x, w)).to_code(
                helion.Config(
                    block_sizes=[32, 512, 4, 8, 1],
                    num_threads=[32, 512, 1, 8, 1],
                    num_warps=32,
                    pid_type="flat",
                )
            )

    def test_narrow_matmul_roots(self) -> None:
        """An MPP matmul's result spreads over every SIMD group of the
        launch, so a phase whose matmul root lays out fewer threads than the
        launch compiles.  The split-K example at its default config: an MPP
        root of one thread beside a 128-thread sum, whose partials every
        thread stores, the same values on each.  And an 8 x 16 matmul root
        in a 64 x 16 launch, whose result is stored through the matmul's
        element loop, beside an elementwise root that lays out no threads
        along y and so takes ``tid[1] < 1``."""
        split_k_matmul = import_example("split_k_barrier").split_k_matmul
        a, b = torch.randn(16, 4096), torch.randn(4096, 16)
        code = split_k_matmul.bind((a, b)).to_code(
            helion.Config(
                block_sizes=[16, 16, 16, 1, 1],
                reduction_loops=[1024],
                split_k=64,
                pid_type="flat",
            )
        )
        self.assertIn("_block_dims=(128, 1, 1)", code)
        self.assertRegex(code, r"tl\.store\(tmp \+ [^\n]*, _mpp_v_\d+, None\)")
        x = torch.randn(64, 64, dtype=torch.float16)
        code = scale_then_matmul.bind((x, x.clone())).to_code(
            helion.Config(block_sizes=[1, 64, 32, 32, 32], pid_type="flat")
        )
        self.assertIn("_block_dims=(64, 16, 1)", code)
        self.assertRegex(code, r"tl\.store\(out \+ [^\n]*, acc, mask_\d+ & mask_\d+\)")
        self.assertRegex(
            code, r"tl\.store\(xs \+ [^\n]*, tid\[1\] < 1 and mask_\d+ & mask_\d+\)"
        )

    def test_alias_names(self) -> None:
        """Each coherent alias is named apart from every parameter and body
        name, and is cast from its own buffer."""
        body = [ast.parse("tl.store(a + i, tl.load(a__coherent + i))").body[0]]
        renamed, decls = coherent_param_prologue(
            body,
            {"a", "a__coherent"},
            {"a": "float", "a__coherent": "float", "a__coherent1": "float"},
            ["a", "a__coherent", "a__coherent1", "i"],
        )
        self.assertEqual(
            decls,
            [
                (
                    "device coherent(device) float* a__coherent2 = "
                    "reinterpret_cast<device coherent(device) float*>(a);"
                ),
                (
                    "device coherent(device) float* a__coherent__coherent = "
                    "reinterpret_cast<device coherent(device) float*>(a__coherent);"
                ),
            ],
        )
        self.assertEqual(
            ast.unparse(renamed[0]),
            "tl.store(a__coherent2 + i, tl.load(a__coherent__coherent + i))",
        )

    def test_only_flat(self) -> None:
        """The search keeps only the flat launch: a static persistent split
        would need every threadgroup co-resident."""
        a = torch.randn(256)
        spec = pingpong.bind((a, torch.empty_like(a))).config_spec
        self.assertEqual(spec.allowed_pid_types, ("flat",))

    def test_flat_ruled_out(self) -> None:
        """A data-dependent loop bound rules out the flat launch, the only
        one Metal runs phases in: binding refuses the kernel rather than
        leave it no pid type."""
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "pid_type='flat' ruled out"
        ):
            bounded_by_tensor.bind((torch.randn(100), torch.tensor(77)))

    def test_non_flat_pid_type(self) -> None:
        """A config naming another pid type is refused with one naming
        flat, the only one Metal orders phases under."""
        a = torch.randn(256)
        bound = pingpong.bind((a, torch.empty_like(a)))
        for pid_type in ("xyz", "persistent_blocked", "persistent_interleaved"):
            with (
                self.subTest(pid_type=pid_type),
                self.assertRaisesRegex(
                    exc.InvalidConfig, rf"use pid_type='flat' \(got '{pid_type}'\)"
                ),
            ):
                bound.to_code(helion.Config(block_sizes=[64] * 4, pid_type=pid_type))

    def test_coherent_view_refused(self) -> None:
        """Bound with separate tensors, the matmul views ``a`` in place; a
        call passing ``y`` as ``a`` makes it coherent, which a view can't
        read through, so the shader is refused rather than miscompiled."""
        x, w = (
            torch.randn(32, 64, dtype=torch.float16),
            torch.randn(64, 32, dtype=torch.float16),
        )
        y = torch.empty_like(x)
        config = helion.Config(
            block_sizes=[16, 16, 16, 16, 16], metal_matmul="mpp", pid_type="flat"
        )
        render_msl(write_then_matmul, (x, y, x.clone(), w), config)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "an MPP matmul operand viewed in place in a"
        ):
            render_msl(
                write_then_matmul, (x, y, x.clone(), w), config, run_args=(x, y, y, w)
            )

    def test_no_simdgroup_split(self) -> None:
        """SIMD groups don't split a phased kernel's fragment K loops: each
        threadgroup's first thread takes its ticket, wait and arrival, and
        each split SIMD group would have its own first thread.  The search
        keeps one SIMD group, and a config asking for two is refused."""
        x = torch.randn(16, 64, dtype=torch.float16)
        y = torch.randn(64, 32, dtype=torch.float16)
        bound = two_phase_matmuls.bind((x, y, y.clone()))
        self.assertFalse(kernel_facts(bound.config_spec).fragment_splits)
        config = helion.Config(
            block_sizes=[16] * 6, metal_matmul="fragment", num_warps=1, pid_type="flat"
        )
        bound.to_code(config)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"split over 2 SIMD groups \(num_warps=2\): the kernel has "
            r"hl.barrier\(\) phases; use num_warps=1",
        ):
            bound.to_code(helion.Config(**{**config.config, "num_warps": 2}))


class TestMetalPhasePlan(unittest.TestCase):
    """Launch and shader-key decisions."""

    def test_mpp_roots_keep_the_axis_maximum(self) -> None:
        """Roots laid out differently in the same thread count share one flat
        axis, unless the kernel has MPP matmuls, which run on the launch's own
        axes: then they share the per-axis maximum, as roots of different
        thread counts always do."""
        dims = [(32, 4, 1), (128, 1, 1)]
        self.assertEqual(launch_dims(dims, matmuls_own_axes=False), (128, 1, 1))
        self.assertEqual(launch_dims(dims, matmuls_own_axes=True), (128, 4, 1))
        self.assertEqual(
            launch_dims([(32, 4, 1), (64, 1, 1)], matmuls_own_axes=False), (64, 4, 1)
        )

    def test_shader_key_records_the_storage_grouping(self) -> None:
        """Which arguments share storage with which is part of the shader
        key: a phased kernel's coherent buffers depend on the grouping."""
        # Imported here: Linux CI imports Metal test modules only to skip them.
        from helion._compiler.metal.metal_jit import _shared_storage

        p, q = torch.zeros(8), torch.zeros(8)
        self.assertNotEqual(
            _shared_storage((p, q, p, q)), _shared_storage((p, p, q, q))
        )
        self.assertEqual(_shared_storage((p, q)), ())


if __name__ == "__main__":
    unittest.main()
