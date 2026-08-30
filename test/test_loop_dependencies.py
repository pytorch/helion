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
)(implicit_tile_dependency_chain.fn)


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
        self.assertIn("_minimum_resident_programs=_NUM_SM", code)

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

    def test_dynamic_shape_rejects_static_pipeline(self) -> None:
        x = torch.arange(65, device=DEVICE, dtype=torch.float32)
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "requires concrete top-level task counts",
        ):
            code_and_output(
                dynamic_implicit_tile_dependency_chain,
                (x,),
                block_sizes=[16, 32],
                pid_type="persistent_blocked",
                cross_loop_schedule="static_pipeline",
                num_warps=1,
            )

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
