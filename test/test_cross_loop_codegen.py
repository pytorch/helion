from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest import mock

import torch

from test._cross_loop_test_kernels import cartesian_affine_chain
from test._cross_loop_test_kernels import cartesian_affine_join
from test._cross_loop_test_kernels import coalesced_multi_producer_join
from test._cross_loop_test_kernels import coalesced_single_producer_fanout
from test._cross_loop_test_kernels import direct_nested_continuation
from test._cross_loop_test_kernels import grouped_affine_chain
from test._cross_loop_test_kernels import mixed_radix_continuation
from test._cross_loop_test_kernels import multi_producer_join
from test._cross_loop_test_kernels import nested_load_store_chain
from test._cross_loop_test_kernels import nested_store_chain
from test._cross_loop_test_kernels import nested_two_axis_consumer
from test._cross_loop_test_kernels import offset_affine_chain
from test._cross_loop_test_kernels import partial_prefix_continuation
from test._cross_loop_test_kernels import partial_prefix_in_place_chain
from test._cross_loop_test_kernels import prewait_singleton_reduction
from test._cross_loop_test_kernels import readiness_counter_chain
from test._cross_loop_test_kernels import singleton_root_join
from test._cross_loop_test_kernels import size_one_view_chain
from test._cross_loop_test_kernels import specialized_quotient_chain
from test._cross_loop_test_kernels import streamed_sibling_reductions
from test._cross_loop_test_kernels import streamed_singleton_reduction
from test._cross_loop_test_kernels import three_way_affine_chain

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cross_loop_codegen import _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
from helion._compiler.cross_loop_codegen import _ast_fingerprint
from helion._compiler.cross_loop_codegen import _clone_opaque_loop_segment
from helion._compiler.cross_loop_codegen import _clone_opaque_statements
from helion._compiler.cross_loop_codegen import (
    _clone_opaque_statements_with_loop_segments,
)
from helion._compiler.device_function import DeviceFunction
from helion._compiler.tile_dependency import TILE_DEPENDENCY_SITE_ID_ATTR
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl


class TestCrossLoopCodegenHelpers(TestCase):
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
        setattr(loop, TILE_DEPENDENCY_SITE_ID_ATTR, 7)
        staged = _clone_opaque_statements_with_loop_segments(
            [loop],
            site_id=7,
            split_iteration_offsets=(4,),
            segment_waits=(
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


@onlyBackends(["triton"])
class TestCrossLoopCodegen(RefEagerTestBase, TestCase):
    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_nested_producer_iterations_publish_readiness(self) -> None:
        x = torch.arange(2 * 64, device=DEVICE, dtype=torch.float32).reshape(2, 64)
        for name, extra_config, expected_range_option in (
            ("default", {"num_warps": 1}, None),
            (
                "pipelined",
                {"num_warps": 4, "range_num_stages": [0, 4, 0]},
                "num_stages=4",
            ),
            (
                "unrolled",
                {"num_warps": 4, "range_unroll_factors": [0, 2, 0]},
                "loop_unroll_factor=2",
            ),
        ):
            with self.subTest(name=name):
                code, out = code_and_output(
                    nested_store_chain,
                    (x,),
                    pid_type="persistent_blocked",
                    cross_loop_schedule="static_pipeline",
                    num_sm_multiplier=1,
                    **extra_config,
                )

                torch.testing.assert_close(out, (x + 1) * 2)
                self.assertIn("tile_dependency_readiness_wait", code)
                self.assertNotIn("tile_dependency_root_barrier", code)
                if expected_range_option is not None:
                    self.assertIn(expected_range_option, code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_nested_loop_can_consume_and_publish_readiness(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        code, out = code_and_output(
            nested_load_store_chain,
            (x,),
            block_sizes=[1, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2 + 3)
        self.assertIn("tile_dependency_nested_loop_wait", code)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_two_axis_nested_loop_falls_back_to_root_barrier(self) -> None:
        x = torch.arange(32 * 32, device=DEVICE, dtype=torch.float32).reshape(32, 32)
        code, out = code_and_output(
            nested_two_axis_consumer,
            (x,),
            block_sizes=[8, 8],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertNotIn("tile_dependency_nested_loop_wait", code)
        self.assertIn("tile_dependency_root_barrier_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_cartesian_unequal_tiles_choose_proven_readiness(self) -> None:
        for batch, width, producer_width, consumer_width in (
            (2, 64, 16, 32),
            (4, 64, 16, 32),
            (2, 64, 32, 16),
        ):
            with self.subTest(
                batch=batch,
                width=width,
                producer_width=producer_width,
                consumer_width=consumer_width,
            ):
                x = torch.arange(
                    batch * width,
                    device=DEVICE,
                    dtype=torch.float32,
                ).reshape(batch, width)
                for launch in range(2):
                    code, out = code_and_output(
                        cartesian_affine_chain,
                        (x + launch,),
                        block_sizes=[1, producer_width, 1, consumer_width],
                        pid_type="persistent_blocked",
                        cross_loop_schedule="static_pipeline",
                        num_sm_multiplier=1,
                        num_warps=1,
                    )
                    torch.testing.assert_close(out, ((x + launch) + 1) * 2)
                self.assertNotIn("tile_dependency_root_barrier", code)
                if producer_width < consumer_width:
                    self.assertIn("tile_dependency_continuation_previous", code)
                    self.assertNotIn("tile_dependency_task_wait", code)
                else:
                    self.assertIn("tile_dependency_readiness_wait", code)
                    self.assertNotIn("tile_dependency_task_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_continuation_accepts_non_power_of_two_fanin(self) -> None:
        x = torch.arange(2 * 96, device=DEVICE, dtype=torch.float32).reshape(2, 96)
        for launch in range(2):
            code, out = code_and_output(
                three_way_affine_chain,
                (x + launch,),
                block_sizes=[1, 16, 1, 16],
                pid_type="persistent_blocked",
                cross_loop_schedule="static_pipeline",
                num_sm_multiplier=1,
                num_warps=1,
            )
            expected_input = x + launch + 1
            expected = (
                expected_input[:, :32]
                + expected_input[:, 32:64]
                + expected_input[:, 64:]
            )
            torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("* tl.cast(3, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_readiness_counter_supports_chaining(self) -> None:
        x = torch.arange(8 * 4, device=DEVICE, dtype=torch.float32).reshape(8, 4)
        for launch in range(2):
            code, out = code_and_output(
                readiness_counter_chain,
                (x + launch,),
                pid_type="persistent_blocked",
                cross_loop_schedule="static_pipeline",
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, torch.sum(x + launch + 1).reshape(1))
        self.assertGreaterEqual(code.count("tile_dependency_continuation_previous"), 2)
        continuation_lines = [
            line
            for line in code.splitlines()
            if "tile_dependency_continuation_previous" in line
            and "tl.atomic_add" in line
        ]
        self.assertEqual(len(continuation_lines), 2)
        for line in continuation_lines:
            self.assertIn(f"* {_CROSS_LOOP_COUNTER_ALIGNMENT_WORDS}", line)
        self.assertIn(
            f"+ {16 * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS} +",
            continuation_lines[1],
        )
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertIn("tile_dependency_root_barrier_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partial_tiles_keep_exact_task_readiness(self) -> None:
        x = torch.arange(140, device=DEVICE, dtype=torch.float32).reshape(2, 70)
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partial_prefix_uses_counter_continuation(self) -> None:
        x = torch.arange(96, device=DEVICE, dtype=torch.float32)
        for launch in range(2):
            code, (tmp, out) = code_and_output(
                partial_prefix_continuation,
                (x + launch,),
                block_sizes=[16, 32],
                pid_type="persistent_blocked",
                cross_loop_schedule="static_pipeline",
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(tmp, x + launch + 1)
            torch.testing.assert_close(out, (x[:64] + launch + 1) * 2)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("< 4", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_mixed_radix_dependency_uses_counter_continuation(self) -> None:
        x = torch.randn((8, 4096), device=DEVICE, dtype=torch.float32)
        for launch in range(2):
            code, out = code_and_output(
                mixed_radix_continuation,
                (x + launch,),
                pid_type="persistent_blocked",
                cross_loop_schedule="static_pipeline",
                num_sm_multiplier=1,
                num_warps=1,
            )
            gate_up = x + launch + 1
            gate, up = gate_up.chunk(2, dim=1)
            torch.testing.assert_close(out, gate * torch.sigmoid(gate) * up)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tl.cast(32, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_root_barrier", code)
        self.assertLessEqual(code.count("tl.where"), 2)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partial_in_place_preserves_unowned_reaching_definition(self) -> None:
        x = torch.arange(96, device=DEVICE, dtype=torch.float32)
        for launch in range(2):
            code, out = code_and_output(
                partial_prefix_in_place_chain,
                (x + launch,),
                block_sizes=[16, 32, 16],
                pid_type="persistent_blocked",
                cross_loop_schedule="static_pipeline",
                num_sm_multiplier=1,
                num_warps=1,
            )
            expected = x + launch + 1
            expected = torch.cat((expected[:64] * 2, expected[64:]))
            torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_multi_producer_join_uses_one_readiness_counter(self) -> None:
        x = torch.arange(128, device=DEVICE, dtype=torch.float32)
        y = torch.arange(128, device=DEVICE, dtype=torch.float32) + 3
        for launch in range(2):
            code, out = code_and_output(
                multi_producer_join,
                (x + launch, y + launch),
                block_sizes=[16, 16, 16],
                pid_type="persistent_blocked",
                cross_loop_schedule="static_pipeline",
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, x + launch + 1 + (y + launch) * 2)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tl.cast(2, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_repeated_join_waits_once_on_a_coalesced_key(self) -> None:
        x = torch.arange(8 * 4, device=DEVICE, dtype=torch.float32).reshape(8, 4)
        y = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            coalesced_multi_producer_join,
            (x, y),
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        expected = torch.stack([x + 1 + (y * 2)[:, None] + split for split in range(4)])
        torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertIn("tl.cast(5, tl.uint32)", code)
        wait_lines = [
            line
            for line in code.splitlines()
            if "tile_dependency_readiness_wait =" in line
        ]
        publication_lines = [
            line
            for line in code.splitlines()
            if "tl.atomic_add(tile_dependency_state" in line
        ]
        self.assertTrue(wait_lines)
        self.assertTrue(publication_lines)
        for line in wait_lines:
            self.assertIn(f"* {_CROSS_LOOP_COUNTER_ALIGNMENT_WORDS}]", line)
        for line in publication_lines:
            self.assertIn(f"* {_CROSS_LOOP_COUNTER_ALIGNMENT_WORDS}, 1", line)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_single_producer_fanout_waits_once_per_ready_group(self) -> None:
        x = torch.arange(8 * 4, device=DEVICE, dtype=torch.float32).reshape(8, 4)
        code, out = code_and_output(
            coalesced_single_producer_fanout,
            (x,),
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        expected = torch.stack([x + 1 + split for split in range(4)])
        torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertIn("tl.cast(4, tl.uint32)", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_fan_in_one_nested_continuation_needs_no_counter(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            direct_nested_continuation,
            (x,),
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1).reshape(4, 2).sum(dim=-1) * 2)
        self.assertIn("tl.cast(2, tl.uint32) - 1", code)
        self.assertNotIn("tl.cast(1, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_task_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_zero_task_roots_do_not_allocate_task_events(self) -> None:
        x = torch.empty((0, 64), device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        self.assertEqual(out.shape, x.shape)
        self.assertNotIn("tile_dependency_task_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_l2_remapped_roots_use_logical_task_readiness(self) -> None:
        for batch in (3, 4):
            with self.subTest(batch=batch):
                x = torch.arange(
                    batch * 64, device=DEVICE, dtype=torch.float32
                ).reshape(batch, 64)
                code, out = code_and_output(
                    cartesian_affine_chain,
                    (x,),
                    block_sizes=[1, 16, 1, 32],
                    l2_groupings=[2, 2],
                    pid_type="persistent_blocked",
                    cross_loop_schedule="static_pipeline",
                    num_sm_multiplier=1,
                    num_warps=1,
                )

                torch.testing.assert_close(out, (x + 1) * 2)
                self.assertNotIn("tile_dependency_task_wait", code)
                self.assertIn("tile_dependency_continuation_previous", code)
                self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_size_one_view_uses_task_readiness(self) -> None:
        x = torch.arange(32 * 128, device=DEVICE, dtype=torch.float32).reshape(32, 128)
        code, out = code_and_output(
            size_one_view_chain,
            (x,),
            block_sizes=[4, 1, 4, 32],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, ((x + 1) * 2).unsqueeze(0))
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_nonzero_grid_start_uses_root_barrier(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32)
        bound = offset_affine_chain.bind((x,))
        assert bound.host_function is not None
        dependency_graph = bound.host_function.device_ir.tile_dependency_graph
        assert dependency_graph is not None

        code, out = code_and_output(
            offset_affine_chain,
            (x,),
            block_sizes=[16, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x[32:] + 1) * 2)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_specialized_quotient_retains_static_task_geometry(self) -> None:
        x = torch.arange(4, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            specialized_quotient_chain,
            (x, 8, 2),
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertIn("tile_dependency_continuation_task", code)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_continuation_follows_each_roots_pid_order(self) -> None:
        x = torch.arange(64 * 1024, device=DEVICE, dtype=torch.float32).reshape(
            64, 1024
        )
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            loop_orders=[[1, 0], [0, 1]],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tile_dependency_scheduled_pid_task", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_cartesian_join_combines_both_producers(self) -> None:
        x = torch.arange(128, device=DEVICE, dtype=torch.float32).reshape(2, 64)
        code, out = code_and_output(
            cartesian_affine_join,
            (x,),
            block_sizes=[1, 16, 1, 16, 1, 32],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, x * 2)
        self.assertNotIn("tile_dependency_task_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tl.cast(4, tl.uint32) - 1", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_singleton_root_waits_for_multiple_producers(self) -> None:
        x = torch.arange(64, device=DEVICE, dtype=torch.float32).reshape(1, 64)
        code, out = code_and_output(
            singleton_root_join,
            (x,),
            block_sizes=[1, 16, 1, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, torch.sum(x * 2, dim=-1))
        self.assertGreaterEqual(code.count("tile_dependency_root_barrier_wait"), 2)
        self.assertIn("if tl.program_id(0) == 0:", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_singleton_stream_uses_nested_split_nested_loop_at_readiness(self) -> None:
        for batch in (1, 2):
            with self.subTest(batch=batch):
                x = torch.arange(
                    batch * 4096, device=DEVICE, dtype=torch.float32
                ).reshape(batch, 4096)
                code, out = code_and_output(
                    streamed_singleton_reduction,
                    (x,),
                    block_sizes=[1, 16],
                    pid_type="persistent_blocked",
                    cross_loop_schedule="static_pipeline",
                    num_sm_multiplier=1,
                    num_warps=1,
                )

                torch.testing.assert_close(out, torch.sum(x + 1, dim=-1) + x[:, 0] + 1)
                self.assertNotIn("tile_dependency_ordered_group", code)
                self.assertIn("tile_dependency_nested_loop_wait", code)
                self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_nested_wait_does_not_cover_an_earlier_access(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        code, out = code_and_output(
            prewait_singleton_reduction,
            (x,),
            block_sizes=[1, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, torch.sum(x + 1, dim=-1) + x[:, 0] + 1)
        self.assertIn("tile_dependency_root_barrier_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_multiple_nested_loops_share_one_scheduled_root_task(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        code, out = code_and_output(
            streamed_sibling_reductions,
            (x,),
            block_sizes=[1, 16, 1, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(
            out,
            torch.sum(x + 1, dim=-1) + torch.sum(x * 2, dim=-1),
        )
        self.assertGreaterEqual(code.count("tile_dependency_nested_loop_wait"), 2)
        self.assertNotIn("tile_dependency_root_barrier_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_task_events_are_capture_safe(self) -> None:
        x = torch.arange(128, device=DEVICE, dtype=torch.float32).reshape(2, 64)
        bound = cartesian_affine_chain.bind((x,))
        config = helion.Config(
            block_sizes=[1, 16, 1, 32],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=4,
            num_warps=8,
        )
        code = bound.to_triton_code(config)
        self.assertNotIn("launch_cooperative_grid=True", code)
        compiled = bound.compile_config(config)
        compiled(x)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(x)
        for value in (3.0, 7.0, -2.0):
            x.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, (x + 1) * 2)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_grouped_schedule_requires_the_proven_access_order(self) -> None:
        torch.manual_seed(0)
        block_sizes = [1, 16, 1, 16]

        for batch, intermediate, group_size, reverse_groups in (
            (1, 128, 32, False),
            (1, 128, 32, True),
            (2, 128, 32, False),
            (4, 128, 32, False),
            (2, 96, 32, False),
            (2, 128, 64, False),
        ):
            with self.subTest(
                batch=batch,
                intermediate=intermediate,
                group_size=group_size,
                reverse_groups=reverse_groups,
            ):
                # Positive inputs avoid cancellation-dominated relative error;
                # this test is intended to catch readiness failures.
                x = torch.rand((batch, 64), device=DEVICE, dtype=torch.float16)
                w13 = torch.rand(
                    (64, 2 * intermediate), device=DEVICE, dtype=torch.float16
                )
                w2 = torch.rand((intermediate, 64), device=DEVICE, dtype=torch.float16)
                kernel_args = (
                    x,
                    w13,
                    w2,
                    group_size,
                    hl.constexpr(reverse_groups),
                )
                if batch == 1 and not reverse_groups:
                    bound = grouped_affine_chain.bind(kernel_args)
                    assert bound.host_function is not None
                    dependency_graph = (
                        bound.host_function.device_ir.tile_dependency_graph
                    )
                    assert dependency_graph is not None
                    self.assertTrue(
                        all(
                            access.root in (0, 1, 2)
                            for access in dependency_graph.accesses
                        )
                    )
                    downstream_edges = dependency_graph.edges_between(1, 2)
                    self.assertEqual(len(downstream_edges), 2)
                    nested_site_ids = {
                        site.site_id
                        for edge in downstream_edges
                        for dependency in edge.access_dependencies
                        for site in dependency_graph.sites_for_access(
                            dependency.consumer_access_id
                        )
                        if not site.is_root
                    }
                    self.assertEqual(len(nested_site_ids), 1)
                code, out = code_and_output(
                    grouped_affine_chain,
                    kernel_args,
                    block_sizes=block_sizes,
                    pid_type="persistent_blocked",
                    cross_loop_schedule="static_pipeline",
                    num_sm_multiplier=1,
                    num_warps=4,
                    num_stages=2,
                )

                gate_up = (x.float() @ w13.float()).half()
                gate, up = gate_up.chunk(2, dim=-1)
                groups = intermediate // group_size
                if reverse_groups:
                    gate = (
                        gate.reshape(batch, groups, group_size)
                        .flip(1)
                        .reshape(batch, intermediate)
                    )
                    up = (
                        up.reshape(batch, groups, group_size)
                        .flip(1)
                        .reshape(batch, intermediate)
                    )
                activated = gate.float() * up.float()
                scale = (
                    activated.abs().reshape(batch, groups, group_size).amax(dim=-1) + 1
                )
                activation = activated.half()
                expected = (
                    activation.float().reshape(batch, groups, group_size)
                    * scale[:, :, None]
                ).reshape(batch, intermediate) @ w2.float()
                torch.testing.assert_close(out, expected, rtol=3e-2, atol=3e-2)

                if reverse_groups:
                    self.assertNotIn("tile_dependency_group_arrivals", code)
                    self.assertIn("tile_dependency_root_barrier", code)
                elif group_size != 32:
                    self.assertNotIn("tile_dependency_group_arrivals", code)
                    self.assertNotIn("tile_dependency_cohort_wait", code)
                    self.assertIn("tile_dependency_continuation_previous", code)
                    self.assertIn("tile_dependency_nested_loop_wait", code)
                else:
                    self.assertNotIn("tile_dependency_group_arrivals", code)
                    self.assertNotIn("tile_dependency_root_barrier", code)
                    self.assertNotIn("tile_dependency_task_wait", code)
                    self.assertTrue(
                        "tile_dependency_continuation_previous" in code
                        or "tile_dependency_readiness_wait" in code
                    )
                    self.assertNotIn("tile_dependency_cohort_wait", code)
                    self.assertIn("tile_dependency_nested_loop_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_static_pipeline_uses_exact_nested_loop_wait(self) -> None:
        torch.manual_seed(0)
        x = torch.rand((1, 64), device=DEVICE, dtype=torch.float16)
        w13 = torch.rand((64, 256), device=DEVICE, dtype=torch.float16)
        w2 = torch.rand((128, 64), device=DEVICE, dtype=torch.float16)
        kernel_args = (x, w13, w2, 32, hl.constexpr(False))
        bound = grouped_affine_chain.bind(kernel_args)
        self.assertNotIn(
            "cross_loop_num_workers",
            bound.config_spec.user_defined_tunables,
        )
        invalid_config = dict(bound.config_spec.default_config())
        invalid_config["cross_loop_num_workers"] = 3
        with self.assertRaisesRegex(helion.exc.InvalidConfig, "Invalid config keys"):
            bound.config_spec.normalize(invalid_config)

        code, out = code_and_output(
            grouped_affine_chain,
            kernel_args,
            block_sizes=[1, 16, 1, 16],
            pid_type="persistent_blocked",
            cross_loop_schedule="static_pipeline",
            num_sm_multiplier=1,
            num_warps=4,
            num_stages=2,
        )

        gate_up = (x.float() @ w13.float()).half()
        gate, up = gate_up.chunk(2, dim=-1)
        activated = gate.float() * up.float()
        scale = activated.abs().reshape(1, 4, 32).amax(dim=-1) + 1
        expected = (
            activated.half().float().reshape(1, 4, 32) * scale[:, :, None]
        ).reshape(1, 128) @ w2.float()
        torch.testing.assert_close(out, expected, rtol=3e-2, atol=3e-2)
        self.assertNotIn("tile_dependency_root_barrier", code)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tile_dependency_nested_loop_wait", code)
        self.assertNotIn("tile_dependency_cohort_wait", code)
