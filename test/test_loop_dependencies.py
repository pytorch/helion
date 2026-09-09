from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import code_and_output
from helion._testing import is_cuda
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfNotTriton
from helion._testing import skipIfRefEager
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl


@helion.kernel(autotune_effort="none")
def tile_dependency_info_across_barrier(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        tmp[tile] = x[tile] + 1
    hl.barrier()
    for tile in hl.tile(x.size(0)):
        out[tile] = tmp[tile] * 2
    return out


@helion.kernel(autotune_effort="none")
def implicit_tile_dependency_chain(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        tmp[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        out[tile] = tmp[tile] * 2
    return out


@helion.kernel(autotune_effort="none")
def single_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(autotune_effort="none")
def independent_loops(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out0[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        out1[tile] = x[tile] * 2
    return out0, out1


dynamic_implicit_tile_dependency_chain = helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    triton_do_not_specialize=True,
)(implicit_tile_dependency_chain.fn)


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    triton_do_not_specialize=True,
)
def dynamic_exact_tile_dependency_chain(x: torch.Tensor) -> torch.Tensor:
    """Keep scratch layout static while the scheduled task domain varies."""
    tmp = torch.empty((8192,), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        tmp[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        out[tile] = tmp[tile] * 2
    return out


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    triton_do_not_specialize=True,
)
def dynamic_exact_then_barrier_chain(x: torch.Tensor) -> torch.Tensor:
    """Mix a parameter-sized exact counter with a cumulative root barrier."""
    exact_tmp = torch.empty((8192,), dtype=x.dtype, device=x.device)
    dynamic_tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        exact_tmp[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        dynamic_tmp[tile] = exact_tmp[tile] * 2
    for tile in hl.tile(x.size(0)):
        out[tile] = dynamic_tmp[tile] - 3
    return out


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    triton_do_not_specialize=True,
)
def dynamic_exact_three_stage_chain(x: torch.Tensor) -> torch.Tensor:
    first_tmp = torch.empty((8192,), dtype=x.dtype, device=x.device)
    second_tmp = torch.empty((8192,), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        first_tmp[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        second_tmp[tile] = first_tmp[tile] * 2
    for tile in hl.tile(x.size(0)):
        out[tile] = second_tmp[tile] - 3
    return out


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    triton_do_not_specialize=True,
)
def two_dynamic_exact_chains(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exercise two symbolic counter sections whose aggregate size can match."""
    x_tmp = torch.empty((4096,), dtype=x.dtype, device=x.device)
    x_out = torch.empty_like(x)
    y_tmp = torch.empty((4096,), dtype=y.dtype, device=y.device)
    y_out = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        x_tmp[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        x_out[tile] = x_tmp[tile] * 2
    for tile in hl.tile(y.size(0)):
        y_tmp[tile] = y[tile] - 3
    for tile in hl.tile(y.size(0)):
        y_out[tile] = y_tmp[tile] * 4
    return x_out, y_out


@helion.kernel(autotune_effort="none")
def implicit_atomic_dependency(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.atomic_xchg(tmp, [tile], x[tile])
    for tile in hl.tile(x.size(0)):
        out[tile] = tmp[tile] + 1
    return out


@helion.kernel(
    autotune_effort="none",
)
def invalid_cross_root_value(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for source_tile in hl.tile(x.size(0)):
        carry = x[source_tile] + 1
    for output_tile in hl.tile(x.size(0)):
        out[output_tile] = carry  # pyrefly: ignore [unbound-name]
    return out


@helion.kernel(
    autotune_effort="none",
)
def implicit_tile_dependency_fanout(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tmp = torch.empty_like(x)
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        tmp[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        out0[tile] = tmp[tile] * 2
    for tile in hl.tile(x.size(0)):
        out1[tile] = tmp[tile] * 3
    return out0, out1


@helion.kernel(
    autotune_effort="none",
)
def implicit_tile_dependency_three_stage(x: torch.Tensor) -> torch.Tensor:
    tmp0 = torch.empty_like(x)
    tmp1 = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        tmp0[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        tmp1[tile] = tmp0[tile] * 2
    for tile in hl.tile(x.size(0)):
        out[tile] = tmp1[tile] - 3
    return out


@helion.kernel(
    autotune_effort="none",
)
def implicit_tile_dependency_matmul_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    _, p = c.size()
    tmp = torch.empty([m, n], device=a.device, dtype=a.dtype)
    out = torch.empty([m, p], device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        tmp[tile_m, tile_n] = acc
    for tile_m, tile_p in hl.tile([m, p]):
        acc = hl.zeros([tile_m, tile_p], dtype=torch.float32)
        for tile_reduce_n in hl.tile(n):
            acc = torch.addmm(
                acc,
                tmp[tile_m, tile_reduce_n],
                c[tile_reduce_n, tile_p],
            )
        out[tile_m, tile_p] = acc
    return out


class TestTileDependencyAnalysis(TestCase):
    def test_tile_dependency_schedule_has_no_separate_public_object(self) -> None:
        self.assertFalse(hasattr(helion, "TileDependencySchedule"))
        self.assertNotIn("cross_loop_num_workers", helion.Config())

    @skipIfRefEager("compiled HostFunction metadata is unavailable in ref eager mode")
    def test_source_barrier_remains_distinct_from_implicit_lowering(self) -> None:
        x = torch.empty(8, device=DEVICE)
        bound = tile_dependency_info_across_barrier.bind((x,))
        host_function = bound.host_function
        assert host_function is not None
        dependency_graph = host_function.device_ir.tile_dependency_graph
        assert dependency_graph is not None
        self.assertTrue(dependency_graph.accesses)
        self.assertEqual(dependency_graph.edges, ())
        self.assertEqual(
            tuple(host_function.device_ir.phase_for_root(root) for root in range(2)),
            (0, 1),
        )
        self.assertEqual(
            host_function.device_ir.implicit_dependency_starts, frozenset()
        )
        self.assertNotIn("cross_loop_schedule", bound.config_spec._flat_fields())

    @skipIfRefEager("compiled HostFunction metadata is unavailable in ref eager mode")
    def test_regular_kernels_do_not_expose_cross_loop_schedule(self) -> None:
        x = torch.empty(8, device=DEVICE)
        for kernel in (single_loop, independent_loops):
            bound = kernel.bind((x,))
            self.assertNotIn(
                "cross_loop_schedule",
                bound.config_spec._flat_fields(),
            )

    @skipIfRefEager("compiler validation is unavailable in ref eager mode")
    def test_cross_root_device_values_are_rejected(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        with pytest.raises(
            exc.CrossRootDeviceValue,
            match="cannot be carried between top-level loops",
        ):
            invalid_cross_root_value.bind((x,))

    @skipIfRefEager("Loop dependency checks are not performed in ref eager mode")
    def test_top_level_statement_between_loops_is_rejected(self) -> None:
        @helion.kernel
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size()):
                x[tile] += x[tile]

            x.sum()

            for tile in hl.tile(y.size()):
                y[tile] += y[tile]

            return x + y

        x = torch.randn(4, device=DEVICE)
        y = torch.randn(4, device=DEVICE)

        with pytest.raises(
            expected_exception=exc.TopLevelStatementBetweenLoops,
            match="Statements cannot appear between top level loops.",
        ):
            kernel.bind((x, y))

    @skipIfRefEager("Loop dependency checks are not performed in ref eager mode")
    def test_implicit_dependency_lowering_is_rejected_when_unsupported(self) -> None:
        if _get_backend() == "triton" and is_cuda():
            self.skipTest("implicit dependency lowering is supported")

        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        with pytest.raises(
            exc.LoopDependencyError,
            match="Loop dependency detected: 'tmp' was written in a previous loop.",
        ):
            implicit_tile_dependency_chain.bind((x,))


@skipIfNotTriton("tile-dependency lowering requires the Triton backend")
@skipIfNotCUDA()
@skipIfRefEager("tile-dependency lowering is unavailable in ref eager mode")
class TestTritonTileDependencyLowering(TestCase):
    def test_implicit_dependency_exposes_cross_loop_schedule(self) -> None:
        x = torch.empty(8, device=DEVICE)
        bound = implicit_tile_dependency_chain.bind((x,))
        fragment = bound.config_spec._flat_fields()["cross_loop_schedule"]
        self.assertIsInstance(fragment, EnumFragment)
        assert isinstance(fragment, EnumFragment)
        self.assertEqual(fragment.choices, ("barrier", "static_pipeline"))
        self.assertEqual(
            bound.config_spec.default_config()["cross_loop_schedule"],
            "barrier",
        )

    def test_implicit_dependency_defaults_to_grid_barrier(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            implicit_tile_dependency_chain,
            (x,),
            block_sizes=[8, 8],
            pid_type="persistent_blocked",
        )
        torch.testing.assert_close(output, (x + 1) * 2)
        self.assertIn("triton_helpers.x_grid_barrier(", code)
        self.assertIn("launch_cooperative_grid=True", code)
        self.assertNotIn("_minimum_resident_programs=", code)

    def test_atomic_dependency_defaults_to_grid_barrier(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            implicit_atomic_dependency,
            (x,),
            block_sizes=[8, 8],
            pid_type="persistent_blocked",
        )
        torch.testing.assert_close(output, x + 1)
        self.assertIn("triton_helpers.x_grid_barrier(", code)
        self.assertNotIn("_minimum_resident_programs=", code)

    def test_dynamic_shape_defaults_to_grid_barrier(self) -> None:
        x = torch.arange(65, device=DEVICE, dtype=torch.float32)
        bound = dynamic_implicit_tile_dependency_chain.bind((x,))
        host_function = bound.host_function
        assert host_function is not None
        dependency_graph = host_function.device_ir.tile_dependency_graph
        assert dependency_graph is not None
        self.assertTrue(dependency_graph.accesses)
        self.assertTrue(
            all(not access.layout_is_static for access in dependency_graph.accesses)
        )
        code, output = code_and_output(
            dynamic_implicit_tile_dependency_chain,
            (x,),
            block_sizes=[16, 32],
            pid_type="persistent_blocked",
            num_warps=1,
        )
        torch.testing.assert_close(output, (x + 1) * 2)
        self.assertIn("triton_helpers.x_grid_barrier(", code)
        self.assertIn("launch_cooperative_grid=True", code)
        self.assertNotIn("_minimum_resident_programs=", code)

    def test_implicit_dependency_static_pipeline(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            implicit_tile_dependency_chain,
            (x,),
            block_sizes=[8, 8],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
        )
        torch.testing.assert_close(output, (x + 1) * 2)
        self.assertIn("tile_dependency_root_barrier_wait", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)
        self.assertIn("_minimum_resident_programs=", code)

    def test_codegen_schedule_order_does_not_leak_barrier_state(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        bound = implicit_tile_dependency_chain.bind((x,))
        common = {
            "block_sizes": [8, 8],
            "pid_type": "persistent_blocked",
        }
        static_config = helion.Config(
            **common,
            cross_loop_schedule="static_pipeline",
        )
        barrier_config = helion.Config(
            **common,
            cross_loop_schedule="barrier",
        )

        static_before = bound.to_code(static_config)
        barrier = bound.to_code(barrier_config)
        static_after = bound.to_code(static_config)

        self.assertEqual(static_before, static_after)
        self.assertNotIn("launch_cooperative_grid=True", static_before)
        self.assertIn("launch_cooperative_grid=True", barrier)

    def test_one_task_event_synchronizes_multiple_consumers(self) -> None:
        x = torch.arange(64, device=DEVICE, dtype=torch.float32)
        code, outputs = code_and_output(
            implicit_tile_dependency_fanout,
            (x,),
            block_sizes=[8, 8, 8],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
        )
        out0, out1 = outputs
        torch.testing.assert_close(out0, (x + 1) * 2)
        torch.testing.assert_close(out1, (x + 1) * 3)
        self.assertEqual(code.count("sem='release'"), 1)
        self.assertGreaterEqual(code.count("tile_dependency_readiness_wait"), 2)
        self.assertIn("ld.acquire.gpu.global.u32", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)

    def test_three_stage_chain(self) -> None:
        x = torch.arange(64, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            implicit_tile_dependency_three_stage,
            (x,),
            block_sizes=[8, 8, 8],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_warps=1,
        )
        torch.testing.assert_close(output, (x + 1) * 2 - 3)
        self.assertNotIn("tl.atomic_", code)
        self.assertIn("tile_dependency_root_1(tmp0, tmp1", code)
        self.assertIn("tile_dependency_root_2(tmp1, out", code)
        self.assertNotIn("tile_dependency_root_barrier", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)
        self.assertNotIn("launch_cooperative_grid=True", code)

    def test_dynamic_shape_static_pipeline_reuses_compiled_binary(self) -> None:
        exemplar = torch.arange(65, device=DEVICE, dtype=torch.float32)
        bound = dynamic_implicit_tile_dependency_chain.bind((exemplar,))
        config = helion.Config(
            block_sizes=[16, 32],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_warps=1,
        )
        code = bound.to_code(config)
        compiled = bound.compile_config(config)
        worker_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count

        self.assertIn("tile_dependency_root_barrier_wait", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)
        self.assertNotIn("launch_cooperative_grid=True", code)
        self.assertNotIn("tile_dependency_dispatch_ticket", code)
        self.assertIn(
            f"* tl.cast({worker_count}, tl.uint32)",
            code,
        )
        self.assertTrue(
            any(
                "tl.range(" in line and "x_size_0" in line for line in code.splitlines()
            ),
            "the scheduled task range must use the runtime extent",
        )

        def compiled_cubin_hashes() -> set[str]:
            triton_kernel = compiled.__globals__.get(f"_helion_{bound.kernel.name}")
            self.assertIsNotNone(triton_kernel)
            device_caches = getattr(triton_kernel, "device_caches", None)
            self.assertIsInstance(device_caches, dict)
            assert isinstance(device_caches, dict)
            return {
                compiled_kernel.hash
                for cache_tuple in device_caches.values()
                for compiled_kernel in cache_tuple[0].values()
                if getattr(compiled_kernel, "hash", None) is not None
            }

        expected_hashes: set[str] | None = None
        lengths = (
            65,
            0,
            15,
            16,
            17,
            31,
            32,
            33,
            97,
            16 * (worker_count - 1),
            16 * worker_count,
            16 * (worker_count + 1),
            32 * (worker_count - 1),
            32 * worker_count,
            32 * (worker_count + 1),
            65,
            17,
        )
        for length in lengths:
            x = torch.arange(length, device=DEVICE, dtype=torch.float32)
            output = compiled(x)
            torch.testing.assert_close(output, (x + 1) * 2)
            hashes = compiled_cubin_hashes()
            self.assertEqual(len(hashes), 1)
            if expected_hashes is None:
                expected_hashes = hashes
            else:
                self.assertEqual(hashes, expected_hashes)

    def test_dynamic_exact_readiness_reuses_epoch_counter_across_shapes(self) -> None:
        exemplar = torch.arange(65, device=DEVICE, dtype=torch.float32)
        bound = dynamic_exact_tile_dependency_chain.bind((exemplar,))
        config = helion.Config(
            block_sizes=[16, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_warps=1,
        )
        code = bound.to_code(config)
        compiled = bound.compile_config(config)
        worker_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count

        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertIn("tl.atomic_xchg", code)
        self.assertNotIn("tl.atomic_add", code)
        self.assertNotIn("tile_dependency_root_barrier", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)
        self.assertNotIn("tile_dependency_dispatch_ticket", code)
        self.assertIn("32 * ((15 + x.size(0)) // 16)", code)
        self.assertEqual(
            code.count("for tile_dependency_event_frontier_task in tl.range"),
            1,
        )

        def compiled_cubin_hashes() -> set[str]:
            triton_kernel = compiled.__globals__.get(f"_helion_{bound.kernel.name}")
            self.assertIsNotNone(triton_kernel)
            device_caches = getattr(triton_kernel, "device_caches", None)
            self.assertIsInstance(device_caches, dict)
            assert isinstance(device_caches, dict)
            return {
                compiled_kernel.hash
                for cache_tuple in device_caches.values()
                for compiled_kernel in cache_tuple[0].values()
                if getattr(compiled_kernel, "hash", None) is not None
            }

        task_counts = (
            5,
            1,
            0,
            worker_count + 1,
            worker_count - 1,
            2 * worker_count + 3,
            5,
        )
        expected_hashes: set[str] | None = None
        for launch, task_count in enumerate(task_counts):
            length = 0 if task_count == 0 else task_count * 16 - launch % 7
            x = (
                torch.arange(length, device=DEVICE, dtype=torch.float32)
                + launch * 10000
            )
            output = compiled(x)
            torch.testing.assert_close(output, (x + 1) * 2)
            hashes = compiled_cubin_hashes()
            self.assertEqual(len(hashes), 1)
            if expected_hashes is None:
                expected_hashes = hashes
            else:
                self.assertEqual(hashes, expected_hashes)

        captured_input = torch.empty(65, device=DEVICE, dtype=torch.float32)
        compiled(captured_input)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = compiled(captured_input)
        for value in (3.0, 7.0, -2.0):
            captured_input.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured_output, (captured_input + 1) * 2)

    def test_dynamic_exact_three_stage_chain_uses_one_frontier_loop(self) -> None:
        x = torch.arange(2049, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            dynamic_exact_three_stage_chain,
            (x,),
            block_sizes=[16, 16, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_warps=1,
        )

        torch.testing.assert_close(output, (x + 1) * 2 - 3)
        self.assertEqual(code.count("tl.atomic_xchg"), 2)
        self.assertGreaterEqual(code.count("tile_dependency_readiness_wait"), 2)
        self.assertEqual(
            code.count("for tile_dependency_event_frontier_task in tl.range"),
            1,
        )
        loop = code.index("for tile_dependency_event_frontier_task in tl.range")
        root_0 = code.index("tile_dependency_root_0_scheduled_task(", loop)
        root_1 = code.index("tile_dependency_root_1_scheduled_task(", loop)
        root_2 = code.index("tile_dependency_root_2_scheduled_task(", loop)
        self.assertLess(root_0, root_1)
        self.assertLess(root_1, root_2)

    def test_dynamic_counter_tail_does_not_relocate_root_barrier(self) -> None:
        exemplar = torch.arange(65, device=DEVICE, dtype=torch.float32)
        bound = dynamic_exact_then_barrier_chain.bind((exemplar,))
        config = helion.Config(
            block_sizes=[16, 16, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_warps=1,
        )
        code = bound.to_code(config)
        compiled = bound.compile_config(config)
        worker_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        epoch_words = (worker_count + 31) // 32 * 32

        fixed_barrier = f"tile_dependency_state + {epoch_words} + 0 + 0"
        dynamic_counter = f"tile_dependency_state + {epoch_words} + 32 + 0"
        self.assertIn(f"tl.atomic_add({fixed_barrier}", code)
        self.assertIn("tile_dependency_root_barrier_wait", code)
        self.assertIn(f"tl.atomic_xchg({dynamic_counter}", code)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertIn("32 * ((15 + x.size(0)) // 16)", code)

        for launch, length in enumerate((65, 17, 0, 97, 65)):
            x = (
                torch.arange(length, device=DEVICE, dtype=torch.float32)
                + launch * 10000
            )
            output = compiled(x)
            torch.testing.assert_close(output, (x + 1) * 2 - 3)

    def test_dynamic_counter_sections_are_replay_safe_when_offsets_move(self) -> None:
        exemplar = (
            torch.arange(65, device=DEVICE, dtype=torch.float32),
            torch.arange(49, device=DEVICE, dtype=torch.float32),
        )
        bound = two_dynamic_exact_chains.bind(exemplar)
        config = helion.Config(
            block_sizes=[16, 16, 16, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_warps=1,
        )
        code = bound.to_code(config)
        compiled = bound.compile_config(config)

        self.assertEqual(code.count("tl.atomic_xchg"), 2)
        self.assertGreaterEqual(code.count("tile_dependency_readiness_wait"), 2)
        self.assertNotIn("tl.atomic_add", code)
        self.assertNotIn("tile_dependency_root_barrier", code)
        self.assertNotIn("tile_dependency_event_frontier_task", code)

        # The first two shapes have the same total key count but move the
        # boundary between the two event sections.  Absolute epoch xchg makes
        # stale values harmless even when the launcher reuses one state pad.
        shapes = ((16, 144), (0, 145), (16, 144))
        for launch, (x_length, y_length) in enumerate(shapes):
            x = (
                torch.arange(x_length, device=DEVICE, dtype=torch.float32)
                + launch * 10000
            )
            y = (
                torch.arange(y_length, device=DEVICE, dtype=torch.float32)
                + launch * 20000
            )
            x_out, y_out = compiled(x, y)
            torch.testing.assert_close(x_out, (x + 1) * 2)
            torch.testing.assert_close(y_out, (y - 3) * 4)

        graph_inputs = tuple(
            (
                torch.empty(x_length, device=DEVICE, dtype=torch.float32),
                torch.empty(y_length, device=DEVICE, dtype=torch.float32),
            )
            for x_length, y_length in shapes[:2]
        )
        capture_stream = torch.cuda.Stream(device=DEVICE)
        with torch.cuda.stream(capture_stream):
            for x, y in graph_inputs:
                compiled(x, y)
        capture_stream.synchronize()

        graphs: list[
            tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, torch.Tensor]]
        ] = []
        captured_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for inputs in graph_inputs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                outputs = compiled(*inputs)
            graphs.append((graph, inputs))
            captured_outputs.append(outputs)

        for launch, graph_index in enumerate((0, 1, 0, 1, 0)):
            graph, (x, y) = graphs[graph_index]
            x.fill_(launch * 10000 + 1)
            y.fill_(launch * 20000 + 2)
            graph.replay()
            torch.cuda.synchronize()
            x_out, y_out = captured_outputs[graph_index]
            torch.testing.assert_close(x_out, (x + 1) * 2)
            torch.testing.assert_close(y_out, (y - 3) * 4)

    def test_matmul_chain_allows_reused_accumulator_name(self) -> None:
        a = torch.arange(256, device=DEVICE, dtype=torch.float32).reshape(16, 16)
        b = torch.eye(16, device=DEVICE)
        c = torch.eye(16, device=DEVICE)
        code, output = code_and_output(
            implicit_tile_dependency_matmul_chain,
            (a, b, c),
            block_sizes=[16, 16, 16, 16, 16, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_warps=4,
        )
        torch.testing.assert_close(output, a, atol=0, rtol=0)
        self.assertIn("ld.acquire.gpu.global.u32", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)

    def test_outlined_matmul_root_threads_tensor_descriptor(self) -> None:
        a = torch.arange(256, device=DEVICE, dtype=torch.float32).reshape(16, 16)
        b = torch.eye(16, device=DEVICE)
        c = torch.eye(16, device=DEVICE)
        code, output = code_and_output(
            implicit_tile_dependency_matmul_chain,
            (a, b, c),
            block_sizes=[16, 16, 16, 16, 16, 16],
            indexing=[
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
            ],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_warps=4,
        )

        torch.testing.assert_close(output, a, atol=0, rtol=0)
        self.assertIn("b_desc = tl.make_tensor_descriptor", code)
        self.assertIn("def tile_dependency_root_0(a, tmp, b_desc):", code)


if __name__ == "__main__":
    unittest.main()
