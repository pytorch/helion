from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from typing import cast
import unittest
from unittest import mock

import pytest
import torch

import helion
from helion import exc
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.device_function import DeviceFunction
from helion._compiler.program_id import _ast_fingerprint
from helion._compiler.program_id import _clone_opaque_loop_segment
from helion._compiler.program_id import _clone_opaque_statements
from helion._compiler.program_id import _clone_opaque_statements_with_scope_stages
from helion._compiler.tile_dependency import TILE_DEPENDENCY_SCOPE_ID_ATTR
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfRefEager
from helion.runtime import get_num_sm
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


dynamic_implicit_tile_dependency_chain = helion.kernel(
    static_shapes=False,
    autotune_effort="none",
)(implicit_tile_dependency_chain.fn)

reserved_sm_implicit_tile_dependency_chain = helion.kernel(
    autotune_effort="none",
    persistent_reserved_sms=1,
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


@helion.kernel(autotune_effort="none")
def nested_atomic_dependency(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.zeros_like(x)
    out = torch.empty(x.size(1), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        for column in hl.tile(x.size(1)):
            hl.atomic_add(tmp, [row, column], x[row, column])
    for column in hl.tile(x.size(1)):
        out[column] = tmp[:, column].sum(0) + 1
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


class TestTileDependencyScheduling(unittest.TestCase):
    def test_opaque_tile_body_clone_is_structurally_identical(self) -> None:
        body = ast.parse("value = value * 2\nout[index] = value\n").body
        cloned = _clone_opaque_statements(body)
        self.assertEqual(_ast_fingerprint(cloned), _ast_fingerprint(body))
        self.assertIsNot(cloned[0], body[0])

    def test_tile_dependency_loop_staging_preserves_computation(self) -> None:
        loop = ast.parse(
            "for k in tl.range(0, 128, 16):\n"
            "    partial = tl.load(pointer + k)\n"
            "    accumulator = accumulator + partial\n"
        ).body[0]
        assert isinstance(loop, ast.For)
        computation = _ast_fingerprint(loop.body)

        first = _clone_opaque_loop_segment(loop, end=ast.parse("64", mode="eval").body)
        second = _clone_opaque_loop_segment(
            loop, begin=ast.parse("64", mode="eval").body
        )
        setattr(loop, TILE_DEPENDENCY_SCOPE_ID_ATTR, 7)
        staged = _clone_opaque_statements_with_scope_stages(
            [loop],
            scope_id=7,
            split_iteration_offsets=(4,),
            stage_waits=(
                tuple(ast.parse("first_ready = tl.load(counter)\n").body),
                tuple(ast.parse("second_ready = tl.load(counter + 1)\n").body),
            ),
        )

        self.assertEqual(_ast_fingerprint(first.body), computation)
        self.assertEqual(_ast_fingerprint(second.body), computation)
        self.assertIsInstance(staged[1], ast.For)
        self.assertIsInstance(staged[3], ast.For)
        self.assertEqual(_ast_fingerprint(staged[1].body), computation)
        self.assertEqual(_ast_fingerprint(staged[3].body), computation)
        self.assertEqual(ast.unparse(staged[0]), "first_ready = tl.load(counter)")
        self.assertEqual(ast.unparse(staged[2]), "second_ready = tl.load(counter + 1)")

    def test_opaque_tile_body_can_be_outlined_without_rewriting(self) -> None:
        device_function = object.__new__(DeviceFunction)
        device_function.arguments = []
        device_function.wrapper_only_params = []
        device_function.preamble = []
        cast("Any", device_function).namespace = SimpleNamespace(
            create_name=lambda name, _value: name
        )
        device_function.triton_outlined_helpers = []
        device_function.triton_outlined_helper_constexprs = {}
        device_function._variable_renames = {}
        device_function.dce_vars = []
        cast("Any", device_function).codegen = SimpleNamespace(module_statements=[])
        cast("Any", device_function).helper_manager = SimpleNamespace(
            codegen_helper_functions=list
        )
        body = ast.parse("value = tl.load(pointer)\ntl.store(output, value)\n").body
        computation = _ast_fingerprint(body)
        environment = SimpleNamespace(backend_name="triton")
        with mock.patch.object(CompileEnvironment, "current", return_value=environment):
            helper_name, arguments = device_function.register_triton_outlined_helper(
                "opaque_tile", body, noinline=True
            )
            helper = device_function.codegen_helper_functions()[0]

        self.assertEqual(helper_name, "opaque_tile")
        self.assertEqual(arguments, ())
        self.assertIsInstance(helper, ast.FunctionDef)
        assert isinstance(helper, ast.FunctionDef)
        self.assertEqual(_ast_fingerprint(helper.body), computation)
        self.assertEqual(
            ast.unparse(helper.decorator_list[0]), "triton.jit(noinline=True)"
        )

    def test_outlined_tile_body_captures_compiler_preamble_values(self) -> None:
        device_function = object.__new__(DeviceFunction)
        device_function.arguments = []
        device_function.wrapper_only_params = []
        device_function.preamble = cast(
            "list[ast.AST]",
            ast.parse(
                "weight_desc = tl.make_tensor_descriptor(weight, [size], [1], [16])\n"
            ).body,
        )
        cast("Any", device_function).namespace = SimpleNamespace(
            create_name=lambda name, _value: name
        )
        device_function.triton_outlined_helpers = []
        device_function.triton_outlined_helper_constexprs = {}
        device_function._variable_renames = {}
        device_function.dce_vars = []
        cast("Any", device_function).codegen = SimpleNamespace(module_statements=[])
        cast("Any", device_function).helper_manager = SimpleNamespace(
            codegen_helper_functions=list
        )
        body = ast.parse("value = weight_desc.load([offset])\n").body
        environment = SimpleNamespace(backend_name="triton")

        with mock.patch.object(CompileEnvironment, "current", return_value=environment):
            helper_name, arguments = device_function.register_triton_outlined_helper(
                "descriptor_tile", body
            )
            helper = device_function.codegen_helper_functions()[0]

        self.assertEqual(helper_name, "descriptor_tile")
        self.assertEqual(arguments, ("weight_desc",))
        self.assertIsInstance(helper, ast.FunctionDef)
        assert isinstance(helper, ast.FunctionDef)
        self.assertEqual(
            [argument.arg for argument in helper.args.args], ["weight_desc"]
        )

    def test_tile_dependency_schedule_has_no_separate_public_object(self) -> None:
        self.assertFalse(hasattr(helion, "TileDependencySchedule"))
        self.assertEqual(helion.Config().cross_loop_num_workers, 0)
        self.assertEqual(
            helion.Config(cross_loop_num_workers=3).cross_loop_num_workers,
            3,
        )


@onlyBackends(["triton"])
class TestTileDependencyLowering(RefEagerTestBase, TestCase):
    @skipIfRefEager("compiled HostFunction metadata is unavailable in ref eager mode")
    def test_source_barrier_remains_distinct_from_implicit_lowering(self) -> None:
        x = torch.empty(8, device=DEVICE)
        bound = tile_dependency_info_across_barrier.bind((x,))
        host_function = bound.host_function
        assert host_function is not None
        dependency_plan = host_function.device_ir.tile_dependency_graph
        assert dependency_plan is not None
        self.assertTrue(dependency_plan.accesses)
        self.assertEqual(dependency_plan.edges, ())
        self.assertEqual(
            tuple(host_function.device_ir.phase_for_root(root) for root in range(2)),
            (0, 1),
        )
        self.assertEqual(
            host_function.device_ir.implicit_dependency_starts, frozenset()
        )

    @skipIfRefEager("persistent grid-barrier codegen is unavailable in ref eager mode")
    def test_implicit_dependency_is_scheduled_automatically(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            implicit_tile_dependency_chain,
            (x,),
            block_sizes=[8, 8],
            pid_type="persistent_blocked",
        )
        torch.testing.assert_close(output, (x + 1) * 2)
        if "_requires_clc=True" in code:
            self.assertIn("tile_dependency_keyed_event_wait", code)
        else:
            self.assertIn("tile_dependency_root_completion_wait", code)

    @skipIfRefEager("persistent grid-barrier codegen is unavailable in ref eager mode")
    def test_allocation_graph_tracks_atomics_as_writes(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        _code, output = code_and_output(
            implicit_atomic_dependency,
            (x,),
            block_sizes=[8, 8],
            pid_type="persistent_blocked",
        )
        torch.testing.assert_close(output, x + 1)

    @skipIfRefEager("persistent grid-barrier codegen is unavailable in ref eager mode")
    def test_schedule_rejects_cross_root_device_values(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        with pytest.raises(
            exc.CrossRootDeviceValue,
            match="cannot be carried between top-level loops",
        ):
            code_and_output(
                invalid_cross_root_value,
                (x,),
                block_sizes=[8, 8],
                pid_type="persistent_blocked",
            )

    @skipIfRefEager("persistent grid-barrier codegen is unavailable in ref eager mode")
    def test_one_task_event_synchronizes_multiple_consumers(self) -> None:
        x = torch.arange(64, device=DEVICE, dtype=torch.float32)
        code, outputs = code_and_output(
            implicit_tile_dependency_fanout,
            (x,),
            block_sizes=[8, 8, 8],
            pid_type="persistent_blocked",
        )
        out0, out1 = outputs
        torch.testing.assert_close(out0, (x + 1) * 2)
        torch.testing.assert_close(out1, (x + 1) * 3)
        self.assertEqual(code.count("sem='release'"), 1)
        self.assertGreaterEqual(code.count("tile_dependency_keyed_event_wait"), 2)
        self.assertIn("ld.acquire.gpu.global.u32", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable in ref eager")
    def test_three_stage_chain(self) -> None:
        x = torch.arange(64, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            implicit_tile_dependency_three_stage,
            (x,),
            block_sizes=[8, 8, 8],
            pid_type="persistent_blocked",
            num_warps=1,
        )
        torch.testing.assert_close(output, (x + 1) * 2 - 3)
        self.assertNotIn("tl.atomic_", code)
        self.assertIn("tile_dependency_root_1(tmp0, tmp1", code)
        self.assertIn("tile_dependency_root_2(tmp1, out", code)
        self.assertNotIn("tile_dependency_root_completion", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)

    @skipIfRefEager("persistent grid-barrier codegen is unavailable in ref eager")
    def test_dynamic_shape_schedule_uses_safe_phase_fallback(self) -> None:
        x = torch.arange(65, device=DEVICE, dtype=torch.float32)
        bound = dynamic_implicit_tile_dependency_chain.bind((x,))
        host_function = bound.host_function
        assert host_function is not None
        dependency_plan = host_function.device_ir.tile_dependency_graph
        assert dependency_plan is not None
        self.assertTrue(dependency_plan.accesses)
        self.assertTrue(
            all(not access.layout_is_static for access in dependency_plan.accesses)
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

    @skipIfRefEager("persistent grid-barrier codegen is unavailable in ref eager")
    def test_dynamic_schedule_does_not_leak_clc_launch_flags(self) -> None:
        x = torch.arange(65, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            dynamic_implicit_tile_dependency_chain,
            (x,),
            block_sizes=[16, 32],
            pid_type="persistent_blocked",
            num_warps=1,
        )
        torch.testing.assert_close(output, (x + 1) * 2)
        self.assertIn("launch_cooperative_grid=True", code)
        self.assertNotIn("_requires_clc=True", code)
        self.assertNotIn("launch_pdl=True", code)

    @skipIfRefEager("CLC codegen is unavailable in ref eager mode")
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="Cluster Launch Control requires CUDA capability >= 10.0"
    )
    def test_clc_ticket_dispatch_executes_for_multiple_warp_counts(self) -> None:
        x = torch.arange(64, device=DEVICE, dtype=torch.float32)
        for num_warps in (1, 4, 8):
            with self.subTest(num_warps=num_warps):
                code, output = code_and_output(
                    implicit_tile_dependency_chain,
                    (x,),
                    block_sizes=[8, 8],
                    pid_type="persistent_blocked",
                    num_warps=num_warps,
                    cross_loop_num_workers=8,
                )
                torch.testing.assert_close(output, (x + 1) * 2)
                self.assertIn("clusterlaunchcontrol.try_cancel", code)
                self.assertIn("_requires_clc=True", code)
                self.assertIn("launch_pdl=True", code)
                self.assertIn("_cross_loop_dispatch_kind='clc'", code)

    @skipIfRefEager("CLC codegen is unavailable in ref eager mode")
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="Cluster Launch Control requires CUDA capability >= 10.0"
    )
    def test_clc_rounds_non_sm_aligned_worker_count(self) -> None:
        num_sm = get_num_sm(DEVICE)
        worker_count = num_sm + 1
        x = torch.arange(worker_count * 2, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            implicit_tile_dependency_chain,
            (x,),
            block_sizes=[1, 2],
            pid_type="persistent_blocked",
            num_warps=1,
            num_sm_multiplier=2,
            cross_loop_num_workers=worker_count,
        )

        torch.testing.assert_close(output, (x + 1) * 2)
        self.assertIn("clusterlaunchcontrol.try_cancel", code)
        self.assertIn("_requires_clc=True", code)
        self.assertIn("_cross_loop_dispatch_kind='clc'", code)
        self.assertIn("_target_resident_programs_per_sm=2", code)

    @skipIfRefEager("CLC codegen is unavailable in ref eager mode")
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="Cluster Launch Control requires CUDA capability >= 10.0"
    )
    def test_clc_residency_uses_physical_sm_count(self) -> None:
        num_sm = get_num_sm(DEVICE)
        x = torch.arange(num_sm * 2, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            reserved_sm_implicit_tile_dependency_chain,
            (x,),
            block_sizes=[1, 2],
            pid_type="persistent_blocked",
            num_warps=1,
            num_sm_multiplier=2,
            cross_loop_num_workers=num_sm,
        )

        torch.testing.assert_close(output, (x + 1) * 2)
        self.assertIn("clusterlaunchcontrol.try_cancel", code)
        self.assertIn("_requires_clc=True", code)
        self.assertIn("_target_resident_programs_per_sm=1", code)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable in ref eager")
    def test_matmul_chain_allows_reused_accumulator_name(self) -> None:
        a = torch.arange(256, device=DEVICE, dtype=torch.float32).reshape(16, 16)
        b = torch.eye(16, device=DEVICE)
        c = torch.eye(16, device=DEVICE)
        code, output = code_and_output(
            implicit_tile_dependency_matmul_chain,
            (a, b, c),
            block_sizes=[16, 16, 16, 16, 16, 16],
            pid_type="persistent_blocked",
            num_warps=4,
        )
        torch.testing.assert_close(output, a, atol=0, rtol=0)
        self.assertIn("ld.acquire.gpu.global.u32", code)
        self.assertNotIn("triton_helpers.x_grid_barrier(", code)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable in ref eager")
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="Cluster Launch Control requires CUDA capability >= 10.0"
    )
    def test_clc_normalizes_tensor_descriptor_indexing_to_pointer(self) -> None:
        a = torch.arange(512, device=DEVICE, dtype=torch.float32).reshape(32, 16)
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
            num_warps=4,
        )

        torch.testing.assert_close(output, a, atol=0, rtol=0)
        self.assertNotIn("make_tensor_descriptor", code)
        self.assertIn("clusterlaunchcontrol.try_cancel", code)
        self.assertIn("_requires_clc=True", code)
        self.assertIn("_cross_loop_dispatch_kind='clc'", code)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable in ref eager")
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="Cluster Launch Control requires CUDA capability >= 10.0"
    )
    def test_clc_normalizes_atomic_tensor_descriptor_indexing_to_pointer(self) -> None:
        x = torch.arange(32, device=DEVICE, dtype=torch.float32)
        code, output = code_and_output(
            implicit_atomic_dependency,
            (x,),
            block_sizes=[8, 8],
            indexing="pointer",
            atomic_indexing="tensor_descriptor",
            pid_type="persistent_blocked",
            num_warps=4,
        )

        torch.testing.assert_close(output, x + 1)
        self.assertNotIn("make_tensor_descriptor", code)
        self.assertIn("clusterlaunchcontrol.try_cancel", code)
        self.assertIn("_requires_clc=True", code)
        self.assertIn("_cross_loop_dispatch_kind='clc'", code)

    @onlyBackends(["triton"])
    @skipIfRefEager("persistent tile-dependency codegen is unavailable in ref eager")
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="Cluster Launch Control requires CUDA capability >= 10.0"
    )
    def test_nested_atomic_descriptor_is_normalized_for_clc(self) -> None:
        x = torch.arange(512, device=DEVICE, dtype=torch.float32).reshape(16, 32)
        code, output = code_and_output(
            nested_atomic_dependency,
            (x,),
            block_sizes=[32, 32],
            indexing="pointer",
            atomic_indexing="tensor_descriptor",
            pid_type="persistent_blocked",
            num_warps=4,
        )

        torch.testing.assert_close(output, x.sum(0) + 1)
        self.assertNotIn("make_tensor_descriptor", code)
        self.assertIn("tile_dependency_keyed_event_wait", code)
        self.assertIn("clusterlaunchcontrol.try_cancel", code)
        self.assertIn("_requires_clc=True", code)


@onlyBackends(["cute"])
class TestLoops(RefEagerTestBase, TestCase):
    @skipIfRefEager("compiled HostFunction metadata is unavailable in ref eager mode")
    def test_device_ir_records_source_barrier_phases(self) -> None:
        x = torch.empty(8, device=DEVICE)
        bound = tile_dependency_info_across_barrier.bind((x,))
        host_function = bound.host_function
        assert host_function is not None
        self.assertEqual(
            tuple(host_function.device_ir.phase_for_root(root) for root in range(2)),
            (0, 1),
        )
        self.assertEqual(
            host_function.device_ir.implicit_dependency_starts, frozenset()
        )

    @skipIfRefEager("Loop dependency checks are not performed in ref eager mode")
    def test_loop_dependency_error1(self):
        @helion.kernel
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            for tile in hl.tile(x.size()):
                out[tile] += x[tile]

            for tile in hl.tile(y.size()):
                out[tile] += y[tile]

            return out

        x = torch.randn(4, device=DEVICE)
        y = torch.randn(4, device=DEVICE)

        with pytest.raises(
            expected_exception=exc.LoopDependencyError,
            match="Loop dependency detected: 'out' was written in a previous loop.",
        ):
            code_and_output(kernel, (x, y))

    @skipIfRefEager("Loop dependency checks are not performed in ref eager mode")
    def test_loop_dependency_error2(self):
        @helion.kernel
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            for tile in hl.tile(x.size()):
                y[tile] += x[tile]

            for tile in hl.tile(y.size()):
                out[tile] += y[tile]

            return out

        x = torch.randn(4, device=DEVICE)
        y = torch.randn(4, device=DEVICE)

        with pytest.raises(
            expected_exception=exc.LoopDependencyError,
            match="Loop dependency detected: 'y' was written in a previous loop.",
        ):
            code_and_output(kernel, (x, y))

    @skipIfRefEager("Loop dependency checks are not performed in ref eager mode")
    def test_loop_dependency_error3(self):
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
            code_and_output(kernel, (x, y))


if __name__ == "__main__":
    unittest.main()
