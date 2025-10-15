from __future__ import annotations

import unittest

import torch

import helion
from helion._compat import get_tensor_descriptor_fn_name
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
import helion.language as hl


# Global kernel definitions to avoid duplication
@helion.kernel(autotune_effort="none")
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = x.new_empty(x.size())
    for tile in hl.grid(x.size()):
        result[tile] = x[tile] + y[tile]
    return result


@helion.kernel(autotune_effort="none")
def matmul_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.size()
    K2, N = B.size()
    assert K == K2
    result = A.new_empty([M, N])

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc += A[tile_m, tile_k] @ B[tile_k, tile_n]
        result[tile_m, tile_n] = acc
    return result


@helion.kernel(autotune_effort="none")
def add_3d_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = x.new_empty(x.size())
    for tile in hl.grid(x.size()):
        result[tile] = x[tile] + y[tile]
    return result


@helion.kernel(autotune_effort="none")
def add1_kernel(x: torch.Tensor) -> torch.Tensor:
    result = x.new_empty(x.size())
    for tile in hl.tile(x.size(), block_size=[32, 16]):
        result[tile] = x[tile] + 1
    return result


class TestPersistentKernels(RefEagerTestBase, TestCase):
    """Test persistent kernel codegen with different PID strategies."""

    def test_persistent_blocked_simple_add(self):
        """Test persistent blocked kernel with simple addition."""

        args = (
            torch.randn([128, 256], device=DEVICE),
            torch.randn([128, 256], device=DEVICE),
        )

        # Test with persistent_blocked
        code, result = code_and_output(add_kernel, args, pid_type="persistent_blocked")
        self.assertExpectedJournal(code)

        # Check correctness
        expected = args[0] + args[1]
        torch.testing.assert_close(result, expected)

        # Check that code contains persistent kernel infrastructure
        self.assertIn("virtual_pid", code)
        self.assertIn("total_pids", code)

    def test_persistent_interleaved_simple_add(self):
        """Test persistent interleaved kernel with simple addition."""

        args = (
            torch.randn([128, 256], device=DEVICE),
            torch.randn([128, 256], device=DEVICE),
        )

        # Test with persistent_interleaved
        code, result = code_and_output(
            add_kernel, args, pid_type="persistent_interleaved"
        )
        self.assertExpectedJournal(code)

        # Check correctness
        expected = args[0] + args[1]
        torch.testing.assert_close(result, expected)

        # Check that code contains persistent kernel infrastructure
        self.assertIn("virtual_pid", code)
        self.assertIn("total_pids", code)

    def test_persistent_blocked_matmul(self):
        """Test persistent blocked kernel with matrix multiplication."""

        args = (
            torch.randn([64, 128], device=DEVICE),
            torch.randn([128, 96], device=DEVICE),
        )

        # Test with persistent_blocked
        code_persistent, result_persistent = code_and_output(
            matmul_kernel, args, pid_type="persistent_blocked", block_sizes=[32, 32, 32]
        )
        self.assertExpectedJournal(code_persistent)

        # Test with flat for comparison
        code_flat, result_flat = code_and_output(
            matmul_kernel, args, pid_type="flat", block_sizes=[32, 32, 32]
        )
        self.assertExpectedJournal(code_flat)

        # Persistent and flat should produce identical results
        torch.testing.assert_close(result_persistent, result_flat, atol=0, rtol=0)

        # Check correctness against PyTorch
        expected = torch.matmul(args[0], args[1])
        torch.testing.assert_close(result_persistent, expected, atol=1e-1, rtol=1e-2)

        # Check that code contains persistent loop structure
        self.assertIn("for virtual_pid in tl.range", code_persistent)
        self.assertIn("virtual_pid", code_persistent)

    def test_persistent_interleaved_matmul(self):
        """Test persistent interleaved kernel with matrix multiplication."""

        args = (
            torch.randn([64, 128], device=DEVICE),
            torch.randn([128, 96], device=DEVICE),
        )

        # Test with persistent_interleaved
        code_persistent, result_persistent = code_and_output(
            matmul_kernel,
            args,
            block_sizes=[16, 16, 32],
            pid_type="persistent_interleaved",
        )
        self.assertExpectedJournal(code_persistent)

        # Test with flat for comparison
        code_flat, result_flat = code_and_output(
            matmul_kernel,
            args,
            block_sizes=[16, 16, 32],
            pid_type="flat",
        )
        self.assertExpectedJournal(code_flat)

        # Persistent and flat should produce identical results
        torch.testing.assert_close(result_persistent, result_flat, atol=0, rtol=0)

        # Check correctness against PyTorch
        expected = torch.matmul(args[0], args[1])
        torch.testing.assert_close(result_persistent, expected, atol=1e-1, rtol=1e-2)

        # Check that code contains persistent loop structure
        self.assertIn("for virtual_pid in tl.range", code_persistent)
        self.assertIn("virtual_pid", code_persistent)

    def test_persistent_blocked_3d(self):
        """Test persistent blocked kernel with 3D tensor."""

        args = (
            torch.randn([32, 64, 48], device=DEVICE),
            torch.randn([32, 64, 48], device=DEVICE),
        )

        # Test with persistent_blocked
        code_persistent, result_persistent = code_and_output(
            add_3d_kernel, args, pid_type="persistent_blocked"
        )
        self.assertExpectedJournal(code_persistent)

        # Test with flat for comparison
        code_flat, result_flat = code_and_output(add_3d_kernel, args, pid_type="flat")
        self.assertExpectedJournal(code_flat)

        # Persistent and flat should produce identical results
        torch.testing.assert_close(result_persistent, result_flat, atol=0, rtol=0)

        # Check correctness against expected
        expected = args[0] + args[1]
        torch.testing.assert_close(result_persistent, expected)

        # Check that code contains persistent kernel infrastructure with 3D decomposition
        self.assertIn("virtual_pid", code_persistent)
        self.assertIn("num_blocks_0", code_persistent)
        self.assertIn("num_blocks_1", code_persistent)

    def test_persistent_interleaved_3d(self):
        """Test persistent interleaved kernel with 3D tensor."""

        args = (
            torch.randn([32, 64, 48], device=DEVICE),
            torch.randn([32, 64, 48], device=DEVICE),
        )

        # Test with persistent_interleaved
        code_persistent, result_persistent = code_and_output(
            add_3d_kernel,
            args,
            pid_type="persistent_interleaved",
        )
        self.assertExpectedJournal(code_persistent)

        # Test with flat for comparison
        code_flat, result_flat = code_and_output(
            add_3d_kernel,
            args,
            pid_type="flat",
        )
        self.assertExpectedJournal(code_flat)

        # Persistent and flat should produce identical results
        torch.testing.assert_close(result_persistent, result_flat, atol=0, rtol=0)

        # Check correctness against expected
        expected = args[0] + args[1]
        torch.testing.assert_close(result_persistent, expected)

        # Check that code contains persistent kernel infrastructure with 3D decomposition
        self.assertIn("virtual_pid", code_persistent)
        self.assertIn("num_blocks_0", code_persistent)
        self.assertIn("num_blocks_1", code_persistent)

    def test_flat_vs_persistent_blocked_equivalence(self):
        """Test that flat and persistent_blocked produce same results."""

        args = (
            torch.randn([64, 128], device=DEVICE),
            torch.randn([64, 128], device=DEVICE),
        )

        # Test with flat
        _, result_flat = code_and_output(add_kernel, args, pid_type="flat")

        # Test with persistent_blocked
        _, result_persistent = code_and_output(
            add_kernel, args, pid_type="persistent_blocked"
        )

        # Should produce identical results
        torch.testing.assert_close(result_flat, result_persistent)

    def test_xyz_vs_persistent_interleaved_equivalence(self):
        """Test that xyz and persistent_interleaved produce same results."""

        args = (
            torch.randn([64, 128], device=DEVICE),
            torch.randn([64, 128], device=DEVICE),
        )

        # Test with xyz
        _, result_xyz = code_and_output(add_kernel, args, pid_type="xyz")

        # Test with persistent_interleaved
        _, result_persistent = code_and_output(
            add_kernel, args, pid_type="persistent_interleaved"
        )

        # Should produce identical results
        torch.testing.assert_close(result_xyz, result_persistent)

    def test_persistent_kernels_with_shared_program_id(self):
        """Test persistent kernels with multiple top-level for loops to trigger ForEachProgramID.

        Note: In the current implementation, ForEachProgramID generates if statements at the top level,
        and persistent kernels work within each if branch. This is different from the ideal
        architecture where persistent kernels would generate while loops containing ForEachProgramID
        if statements, but it still provides the hierarchical functionality.
        """

        @helion.kernel(autotune_effort="none")
        def multi_loop_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result1 = x.new_empty(x.size())
            result2 = y.new_empty(y.size())

            # First top-level loop - will get its own PID
            for tile1 in hl.grid(x.size()):
                result1[tile1] = x[tile1] * 2

            # Second top-level loop - will trigger ForEachProgramID
            for tile2 in hl.grid(y.size()):
                result2[tile2] = y[tile2] * 3

            return result1, result2

        torch.manual_seed(42)  # Set seed for reproducible results
        args = (
            torch.randn([8, 12], device=DEVICE),
            torch.randn([8, 12], device=DEVICE),
        )

        # Test with persistent_blocked
        code_blocked, results_blocked = code_and_output(
            multi_loop_kernel, args, pid_type="persistent_blocked"
        )
        self.assertExpectedJournal(code_blocked)

        # Test with persistent_interleaved
        code_interleaved, results_interleaved = code_and_output(
            multi_loop_kernel, args, pid_type="persistent_interleaved"
        )
        self.assertExpectedJournal(code_interleaved)

        # Test with flat for comparison
        code_flat, results_flat = code_and_output(
            multi_loop_kernel, args, pid_type="flat"
        )
        self.assertExpectedJournal(code_flat)

        # First verify all strategies produce identical results (most important check)
        torch.testing.assert_close(results_blocked[0], results_flat[0], atol=0, rtol=0)
        torch.testing.assert_close(results_blocked[1], results_flat[1], atol=0, rtol=0)
        torch.testing.assert_close(
            results_interleaved[0], results_flat[0], atol=0, rtol=0
        )
        torch.testing.assert_close(
            results_interleaved[1], results_flat[1], atol=0, rtol=0
        )

        # Calculate expected results
        expected1 = args[0] * 2
        expected2 = args[1] * 3

        # Check correctness against expected (using flat as reference since all should be identical)
        torch.testing.assert_close(results_flat[0], expected1)
        torch.testing.assert_close(results_flat[1], expected2)

        # Check that generated code contains ForEachProgramID patterns (not virtual_pid since ForEachProgramID disables persistent loops)
        self.assertIn("pid_shared", code_blocked)
        self.assertIn("if pid_shared <", code_blocked)
        self.assertIn("pid_shared", code_interleaved)
        self.assertIn("if pid_shared <", code_interleaved)

    def test_persistent_shared_vs_flat_shared_equivalence(self):
        """Test that persistent+ForEachProgramID produces same results as flat+ForEachProgramID."""

        @helion.kernel(autotune_effort="none")
        def shared_loops_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            output1 = a.new_empty(a.size())
            output2 = b.new_empty(b.size())

            # Two top-level loops that will use ForEachProgramID
            for tile_a in hl.grid(a.size()):
                output1[tile_a] = a[tile_a] + 1.0

            for tile_b in hl.grid(b.size()):
                output2[tile_b] = b[tile_b] * 2.0

            return output1, output2

        torch.manual_seed(42)  # Set seed for reproducible results
        args = (
            torch.randn([8, 12], device=DEVICE),
            torch.randn([8, 12], device=DEVICE),
        )

        # Test all strategies with ForEachProgramID
        _, results_flat = code_and_output(shared_loops_kernel, args, pid_type="flat")

        _, results_persistent_blocked = code_and_output(
            shared_loops_kernel, args, pid_type="persistent_blocked"
        )

        _, results_persistent_interleaved = code_and_output(
            shared_loops_kernel, args, pid_type="persistent_interleaved"
        )

        # All strategies should produce identical results
        torch.testing.assert_close(results_flat[0], results_persistent_blocked[0])
        torch.testing.assert_close(results_flat[1], results_persistent_blocked[1])
        torch.testing.assert_close(results_flat[0], results_persistent_interleaved[0])
        torch.testing.assert_close(results_flat[1], results_persistent_interleaved[1])
        torch.testing.assert_close(
            results_persistent_blocked[0], results_persistent_interleaved[0]
        )
        torch.testing.assert_close(
            results_persistent_blocked[1], results_persistent_interleaved[1]
        )

        # Verify expected computation
        expected1 = args[0] + 1.0
        expected2 = args[1] * 2.0
        torch.testing.assert_close(results_flat[0], expected1)
        torch.testing.assert_close(results_flat[1], expected2)

    def test_persistent_kernels_complex_shared_scenario(self):
        """Test persistent kernels with a more complex ForEachProgramID scenario."""

        @helion.kernel(autotune_effort="none")
        def complex_shared_kernel(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result1 = x.new_empty(x.size())
            result2 = y.new_empty(y.size())

            # First loop: process first input
            for tile1 in hl.grid(x.size()):
                result1[tile1] = x[tile1] + y[tile1]

            # Second loop: process second input (independent from first)
            for tile2 in hl.grid(y.size()):
                result2[tile2] = y[tile2] * z[tile2]

            return result1, result2

        torch.manual_seed(42)  # Set seed for reproducible results
        args = (
            torch.randn([6, 8], device=DEVICE),
            torch.randn([6, 8], device=DEVICE),
            torch.randn([6, 8], device=DEVICE),
        )

        # Test persistent strategies
        code_blocked, result_blocked = code_and_output(
            complex_shared_kernel, args, pid_type="persistent_blocked"
        )
        self.assertExpectedJournal(code_blocked)

        code_interleaved, result_interleaved = code_and_output(
            complex_shared_kernel, args, pid_type="persistent_interleaved"
        )
        self.assertExpectedJournal(code_interleaved)

        # Test with flat for comparison
        code_flat, result_flat = code_and_output(
            complex_shared_kernel, args, pid_type="flat"
        )
        self.assertExpectedJournal(code_flat)

        # All strategies should produce identical results
        torch.testing.assert_close(result_blocked, result_flat, atol=0, rtol=0)
        torch.testing.assert_close(result_interleaved, result_flat, atol=0, rtol=0)

        # Calculate expected results manually
        expected1 = args[0] + args[1]
        expected2 = args[1] * args[2]

        # Check correctness against PyTorch
        torch.testing.assert_close(result_blocked[0], expected1, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result_blocked[1], expected2, atol=1e-6, rtol=1e-6)

        # Verify ForEachProgramID structure is working (not virtual_pid loop since ForEachProgramID disables it)
        self.assertIn("pid_shared", code_blocked)
        self.assertIn("if pid_shared <", code_blocked)
        self.assertIn("pid_shared", code_interleaved)
        self.assertIn("if pid_shared <", code_interleaved)

    def test_persistent_blocked_with_l2_grouping(self):
        """Test persistent blocked kernels work with L2 grouping."""

        args = (
            torch.randn([64, 128], device=DEVICE),
            torch.randn([64, 128], device=DEVICE),
        )

        # Test with persistent_blocked + l2_grouping=8
        code_persistent_l2, result_persistent_l2 = code_and_output(
            add_kernel, args, pid_type="persistent_blocked", l2_grouping=8
        )
        self.assertExpectedJournal(code_persistent_l2)

        # Test with flat + l2_grouping=8 for comparison
        code_flat_l2, result_flat_l2 = code_and_output(
            add_kernel, args, pid_type="flat", l2_grouping=8
        )
        self.assertExpectedJournal(code_flat_l2)

        # Test with persistent_blocked alone for comparison
        code_persistent, result_persistent = code_and_output(
            add_kernel, args, pid_type="persistent_blocked", l2_grouping=1
        )
        self.assertExpectedJournal(code_persistent)

        # All should produce identical results
        torch.testing.assert_close(result_persistent_l2, result_flat_l2, atol=0, rtol=0)
        torch.testing.assert_close(
            result_persistent_l2, result_persistent, atol=0, rtol=0
        )

        # Check correctness against expected
        expected = args[0] + args[1]
        torch.testing.assert_close(result_persistent_l2, expected)

        # Check that persistent + L2 grouping code contains both features
        self.assertIn("for virtual_pid in tl.range", code_persistent_l2)
        self.assertIn("num_pid_in_group", code_persistent_l2)
        self.assertIn("group_id", code_persistent_l2)
        # Check that NUM_SM is used in device code and get_num_sm() in host code
        self.assertIn("_NUM_SM: tl.constexpr", code_persistent_l2)
        self.assertIn("helion.runtime.get_num_sm(", code_persistent_l2)

    def test_shared_program_id_with_persistent_basic_functionality(self):
        """Test that ForEachProgramID + persistent kernels generate correct code structure."""

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def multi_add_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result1 = x.new_empty(x.size())
            result2 = y.new_empty(y.size())

            # Two top-level loops to trigger ForEachProgramID
            for tile1 in hl.grid(x.size()):
                result1[tile1] = x[tile1] + 1.0

            for tile2 in hl.grid(y.size()):
                result2[tile2] = y[tile2] * 2.0

            return result1, result2

        torch.manual_seed(42)  # Set seed for reproducible results
        args = (
            torch.randn([8, 8], device=DEVICE),
            torch.randn([8, 8], device=DEVICE),
        )

        # Test persistent + ForEachProgramID
        code_persistent_shared, result_persistent_shared = code_and_output(
            multi_add_kernel, args, pid_type="persistent_blocked"
        )
        self.assertExpectedJournal(code_persistent_shared)

        # Check correctness - both results should be correct
        expected1 = args[0] + 1.0
        expected2 = args[1] * 2.0

        # Note: When persistent kernels are used with ForEachProgramID (multiple loops),
        # the system correctly falls back to ForEachProgramID behavior for correctness.
        # Both results should be computed correctly.

        torch.testing.assert_close(result_persistent_shared[0], expected1)
        torch.testing.assert_close(result_persistent_shared[1], expected2)

        # Check that code contains persistent loop with ForEachProgramID structure
        # The new implementation correctly combines persistent kernels with ForEachProgramID
        self.assertIn(
            "for virtual_pid in tl.range(start_pid, end_pid)", code_persistent_shared
        )
        self.assertIn("pid_shared = virtual_pid", code_persistent_shared)
        self.assertIn("if pid_shared <", code_persistent_shared)
        # Should have the combined total calculation
        self.assertIn(
            "total_pids = x_size_0 * x_size_1 + y_size_0 * y_size_1",
            code_persistent_shared,
        )
        # Grid should use SM count for persistent kernels
        # Check that NUM_SM is used in device code and get_num_sm() in host code
        self.assertIn("_NUM_SM: tl.constexpr", code_persistent_shared)
        self.assertIn("helion.runtime.get_num_sm(", code_persistent_shared)

    def test_simple_persistent_kernels_work(self):
        """Test that simple persistent kernels compile and run correctly."""

        @helion.kernel(autotune_effort="none")
        def simple_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result = x.new_empty(x.size())
            for tile in hl.tile(x.size(), block_size=[32, 16]):
                result[tile] = x[tile] + y[tile]
            return result

        args = (
            torch.randn([8, 12], device=DEVICE),
            torch.randn([8, 12], device=DEVICE),
        )
        expected = args[0] + args[1]

        # Test persistent_blocked
        code_blocked, result_blocked = code_and_output(
            simple_add, args, pid_type="persistent_blocked"
        )
        self.assertExpectedJournal(code_blocked)
        torch.testing.assert_close(result_blocked, expected)

        # Verify correct grid size and loop structure
        # Check that NUM_SM is used in device code and get_num_sm() in host code
        self.assertIn("_NUM_SM: tl.constexpr", code_blocked)
        self.assertIn("helion.runtime.get_num_sm(", code_blocked)
        self.assertIn("for virtual_pid in tl.range", code_blocked)

        # Test persistent_interleaved
        code_interleaved, result_interleaved = code_and_output(
            simple_add, args, pid_type="persistent_interleaved"
        )
        self.assertExpectedJournal(code_interleaved)
        torch.testing.assert_close(result_interleaved, expected)

        # Verify correct grid size and loop structure
        # Check that NUM_SM is used in device code and get_num_sm() in host code
        self.assertIn("_NUM_SM: tl.constexpr", code_interleaved)
        self.assertIn("helion.runtime.get_num_sm(", code_interleaved)
        self.assertIn("for virtual_pid in tl.range", code_interleaved)

    def test_multi_loop_persistent_with_shared_program_id(self):
        """Test that multi-loop persistent kernels with ForEachProgramID work correctly.

        This is a regression test for the bug where multi-loop kernels with persistent
        strategies would generate incorrect code with variable scoping issues.
        """

        @helion.kernel(autotune_effort="none")
        def multi_loop_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result1 = x.new_empty(x.size())
            result2 = y.new_empty(y.size())

            # First loop - will get its own PID
            for tile1 in hl.tile(x.size(), block_size=[16, 8]):
                result1[tile1] = x[tile1] * 2

            # Second loop - will trigger ForEachProgramID
            for tile2 in hl.tile(y.size(), block_size=[16, 8]):
                result2[tile2] = y[tile2] * 3

            return result1, result2

        args = (torch.randn([4, 6], device=DEVICE), torch.randn([4, 6], device=DEVICE))
        expected1 = args[0] * 2
        expected2 = args[1] * 3

        # Test with persistent_blocked - this was failing before the fix
        code_blocked, results_blocked = code_and_output(
            multi_loop_kernel, args, pid_type="persistent_blocked"
        )
        self.assertExpectedJournal(code_blocked)
        torch.testing.assert_close(results_blocked[0], expected1)
        torch.testing.assert_close(results_blocked[1], expected2)

        # Verify ForEachProgramID structure is present
        self.assertIn("pid_shared", code_blocked)
        self.assertIn("if pid_shared <", code_blocked)

        # Test with persistent_interleaved
        code_interleaved, results_interleaved = code_and_output(
            multi_loop_kernel, args, pid_type="persistent_interleaved"
        )
        self.assertExpectedJournal(code_interleaved)
        torch.testing.assert_close(results_interleaved[0], expected1)
        torch.testing.assert_close(results_interleaved[1], expected2)

        # Verify ForEachProgramID structure is present
        self.assertIn("pid_shared", code_interleaved)
        self.assertIn("if pid_shared <", code_interleaved)

    @skipIfRefEager("Code pattern checking not applicable in ref eager mode")
    def test_persistent_grid_size_correctness(self):
        """Test that persistent kernels use NUM_SMS grid size, not full grid size."""

        @helion.kernel(autotune_effort="none")
        def test_kernel(x: torch.Tensor) -> torch.Tensor:
            result = x.new_empty(x.size())
            for tile in hl.tile(x.size(), block_size=[32, 16]):
                result[tile] = x[tile] + 1
            return result

        args = (torch.randn([64, 96], device=DEVICE),)

        # Get codes for different strategies
        code_flat, _ = code_and_output(test_kernel, args, pid_type="flat")
        self.assertExpectedJournal(code_flat)
        code_persistent_blocked, _ = code_and_output(
            test_kernel, args, pid_type="persistent_blocked"
        )
        self.assertExpectedJournal(code_persistent_blocked)
        code_persistent_interleaved, _ = code_and_output(
            test_kernel, args, pid_type="persistent_interleaved"
        )
        self.assertExpectedJournal(code_persistent_interleaved)

        # Extract grid sizes from kernel calls - look for the pattern _launcher(_kernel, grid, ...)
        import re

        # Look for _launcher(_kernel_name, (grid_size), ...) pattern
        flat_grid_match = re.search(r"_launcher\([^,]+,\s*\(([^)]+)\)", code_flat)
        persistent_blocked_grid_match = re.search(
            r"_launcher\([^,]+,\s*\(([^)]+)\)", code_persistent_blocked
        )
        persistent_interleaved_grid_match = re.search(
            r"_launcher\([^,]+,\s*\(([^)]+)\)", code_persistent_interleaved
        )

        self.assertIsNotNone(flat_grid_match, "Could not find grid size in flat code")
        self.assertIsNotNone(
            persistent_blocked_grid_match,
            "Could not find grid size in persistent blocked code",
        )
        self.assertIsNotNone(
            persistent_interleaved_grid_match,
            "Could not find grid size in persistent interleaved code",
        )

        flat_grid = flat_grid_match.group(1).rstrip(",")  # Remove trailing comma
        persistent_blocked_grid = persistent_blocked_grid_match.group(1).rstrip(",")
        persistent_interleaved_grid = persistent_interleaved_grid_match.group(1).rstrip(
            ","
        )

        # Flat should use the full grid size calculation
        self.assertIn("triton.cdiv", flat_grid)

        # Persistent kernels should use NUM_SMS
        self.assertEqual(
            persistent_blocked_grid,
            "_NUM_SM",
        )
        self.assertEqual(
            persistent_interleaved_grid,
            "_NUM_SM",
        )

    def test_persistent_loop_variable_names(self):
        """Test that persistent kernels use correct virtual_pid variable names."""

        @helion.kernel(autotune_effort="none")
        def test_kernel(x: torch.Tensor) -> torch.Tensor:
            result = x.new_empty(x.size())
            for tile in hl.tile(x.size(), block_size=[32, 16]):
                result[tile] = x[tile] + 1
            return result

        args = (torch.randn([32, 48], device=DEVICE),)
        expected = args[0] + 1

        # Test blocked strategy
        code_blocked, result_blocked = code_and_output(
            test_kernel, args, pid_type="persistent_blocked"
        )
        self.assertExpectedJournal(code_blocked)
        torch.testing.assert_close(result_blocked, expected)

        # Should have the correct loop structure
        self.assertIn("for virtual_pid in tl.range(start_pid, end_pid):", code_blocked)
        self.assertIn("pid_0 = virtual_pid %", code_blocked)
        self.assertIn("pid_1 = virtual_pid //", code_blocked)

        # Test interleaved strategy
        code_interleaved, result_interleaved = code_and_output(
            test_kernel, args, pid_type="persistent_interleaved"
        )
        self.assertExpectedJournal(code_interleaved)
        torch.testing.assert_close(result_interleaved, expected)

        # Should have the correct loop structure
        self.assertIn(
            "for virtual_pid in tl.range(tl.program_id(0), total_pids, _NUM_SM):",
            code_interleaved,
        )
        self.assertIn("pid_0 = virtual_pid %", code_interleaved)
        self.assertIn("pid_1 = virtual_pid //", code_interleaved)

    def test_persistent_1d_tiling(self):
        """Test persistent kernels with 1D tiling."""

        @helion.kernel(autotune_effort="none")
        def vector_add_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result = x.new_empty(x.size())
            for tile in hl.tile(x.size(), block_size=[128]):
                result[tile] = x[tile] + y[tile]
            return result

        args = (
            torch.randn([1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        expected = args[0] + args[1]

        # Test persistent_blocked with 1D
        code_blocked, result_blocked = code_and_output(
            vector_add_1d, args, pid_type="persistent_blocked"
        )
        self.assertExpectedJournal(code_blocked)
        torch.testing.assert_close(result_blocked, expected)

        # Verify 1D persistent loop structure
        self.assertIn("for virtual_pid in tl.range", code_blocked)
        self.assertIn("pid_0 = virtual_pid", code_blocked)
        self.assertNotIn("pid_1", code_blocked)  # Should not have pid_1 for 1D

        # Test persistent_interleaved with 1D
        code_interleaved, result_interleaved = code_and_output(
            vector_add_1d, args, pid_type="persistent_interleaved"
        )
        self.assertExpectedJournal(code_interleaved)
        torch.testing.assert_close(result_interleaved, expected)

        # Verify 1D persistent loop structure
        self.assertIn("for virtual_pid in tl.range", code_interleaved)
        self.assertIn("pid_0 = virtual_pid", code_interleaved)
        self.assertNotIn("pid_1", code_interleaved)  # Should not have pid_1 for 1D

        # Test correctness vs flat
        code_flat, result_flat = code_and_output(vector_add_1d, args, pid_type="flat")
        self.assertExpectedJournal(code_flat)
        torch.testing.assert_close(result_blocked, result_flat, atol=0, rtol=0)
        torch.testing.assert_close(result_interleaved, result_flat, atol=0, rtol=0)

    def test_persistent_interleaved_with_l2_grouping_single_loop(self):
        """Test persistent_interleaved with l2_grouping (2D iteration space) - single loop case."""

        @helion.kernel(autotune_effort="none")
        def single_loop_l2_kernel(x: torch.Tensor) -> torch.Tensor:
            result = x.new_empty(x.size())
            # Single top-level hl.tile loop with 2D iteration space
            for tile in hl.tile(x.size(), block_size=[16, 16]):
                result[tile] = x[tile] * 2.0
            return result

        args = (torch.randn([64, 128], device=DEVICE),)

        # Test with persistent_interleaved + l2_grouping=4 (requires 2D iteration space)
        code, result = code_and_output(
            single_loop_l2_kernel,
            args,
            pid_type="persistent_interleaved",
            l2_grouping=4,
        )
        self.assertExpectedJournal(code)

        # Check correctness
        expected = args[0] * 2.0
        torch.testing.assert_close(result, expected)

        # Verify code contains persistent_interleaved feature
        self.assertIn("for virtual_pid in tl.range", code)
        self.assertIn("_NUM_SM", code)

        # Verify L2 grouping features are present
        self.assertIn("num_pid_in_group", code)
        self.assertIn("group_id", code)

        # Verify 2D iteration space variables
        self.assertIn("pid_0 = ", code)
        self.assertIn("pid_1 = ", code)

        # Test against flat for correctness comparison
        code_flat, result_flat = code_and_output(
            single_loop_l2_kernel, args, pid_type="flat", l2_grouping=4
        )
        self.assertExpectedJournal(code_flat)
        torch.testing.assert_close(result, result_flat, atol=0, rtol=0)

    def test_persistent_interleaved_multiple_loops_without_l2_grouping(self):
        """Test persistent_interleaved with multiple top-level hl.tile loops (without l2_grouping)."""

        @helion.kernel(autotune_effort="none")
        def multi_loop_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result1 = x.new_empty(x.size())
            result2 = y.new_empty(y.size())

            # First top-level hl.tile loop
            for tile1 in hl.tile(x.size(), block_size=[16, 16]):
                result1[tile1] = x[tile1] * 2.0

            # Second top-level hl.tile loop - triggers ForEachProgramID
            for tile2 in hl.tile(y.size(), block_size=[16, 16]):
                result2[tile2] = y[tile2] + 1.0

            return result1, result2

        args = (
            torch.randn([32, 64], device=DEVICE),
            torch.randn([32, 64], device=DEVICE),
        )

        # Test with persistent_interleaved (no l2_grouping to avoid current limitations)
        code, result = code_and_output(
            multi_loop_kernel,
            args,
            pid_type="persistent_interleaved",
        )
        self.assertExpectedJournal(code)

        # Check correctness
        expected1 = args[0] * 2.0
        expected2 = args[1] + 1.0
        torch.testing.assert_close(result[0], expected1)
        torch.testing.assert_close(result[1], expected2)

        # Verify code contains persistent_interleaved features combined with ForEachProgramID
        self.assertIn("for virtual_pid in tl.range", code)

        # Verify ForEachProgramID features (multiple loops)
        self.assertIn("pid_shared", code)
        self.assertIn("if pid_shared <", code)

        # Test against flat for correctness comparison
        code_flat, result_flat = code_and_output(
            multi_loop_kernel, args, pid_type="flat"
        )
        self.assertExpectedJournal(code_flat)
        torch.testing.assert_close(result[0], result_flat[0], atol=0, rtol=0)
        torch.testing.assert_close(result[1], result_flat[1], atol=0, rtol=0)

    def test_persistent_interleaved_multiple_loops_with_l2_grouping(self):
        """Test persistent_interleaved with multiple top-level hl.tile loops AND l2_grouping (all 3 features combined)."""

        @helion.kernel(autotune_effort="none")
        def multi_loop_l2_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result1 = x.new_empty(x.size())
            result2 = y.new_empty(y.size())
            result3 = y.new_empty(y.size())
            for tile1 in hl.tile(x.size(), block_size=[16, 16]):
                result1[tile1] = x[tile1] * 2.0
            for tile2 in hl.tile(y.size(), block_size=[16, 16]):
                result2[tile2] = y[tile2] + 1.0
            for tile3 in hl.tile(y.size(), block_size=[16, 16]):
                result3[tile3] = y[tile3] + 2.0
            return result1, result2, result3

        args = (
            torch.randn([32, 64], device=DEVICE),
            torch.randn([32, 64], device=DEVICE),
        )

        # Test with persistent_interleaved + multiple loops + l2_grouping=4 (all 3 features)
        code, result = code_and_output(
            multi_loop_l2_kernel,
            args,
            pid_type="persistent_interleaved",
            l2_grouping=[2, 4, 2],
        )
        self.assertExpectedJournal(code)

        # Check correctness
        expected1 = args[0] * 2.0
        expected2 = args[1] + 1.0
        expected3 = args[1] + 2.0
        torch.testing.assert_close(result[0], expected1)
        torch.testing.assert_close(result[1], expected2)
        torch.testing.assert_close(result[2], expected3)

        # Verify code contains persistent_interleaved features
        self.assertIn("for virtual_pid in tl.range", code)
        self.assertIn("_NUM_SM", code)

        # Verify L2 grouping features are present
        self.assertIn("num_pid_in_group", code)
        self.assertIn("group_id", code)

        # Verify ForEachProgramID features (multiple loops)
        self.assertIn("pid_shared", code)
        self.assertIn("if pid_shared <", code)

        # Verify 2D iteration space variables
        self.assertIn("pid_0 = ", code)
        self.assertIn("pid_1 = ", code)

        # Test against flat for correctness comparison
        code_flat, result_flat = code_and_output(
            multi_loop_l2_kernel, args, pid_type="flat", l2_grouping=4
        )
        self.assertExpectedJournal(code_flat)
        torch.testing.assert_close(result[0], result_flat[0])
        torch.testing.assert_close(result[1], result_flat[1])

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptors not supported on this device"
    )
    def test_persistent_kernels_with_tensor_descriptor_indexing(self):
        """Test persistent kernels with indexing='tensor_descriptor'."""

        @helion.kernel(autotune_effort="none")
        def tensor_descriptor_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result = x.new_empty(x.size())
            for tile in hl.tile(x.size(), block_size=[32, 32]):
                result[tile] = x[tile] + y[tile]
            return result

        args = (
            torch.randn([64, 128], device=DEVICE),
            torch.randn([64, 128], device=DEVICE),
        )

        # Test with tensor_descriptor indexing + persistent_blocked
        code_blocked, result_blocked = code_and_output(
            tensor_descriptor_kernel,
            args,
            pid_type="persistent_blocked",
            indexing="tensor_descriptor",
        )
        self.assertExpectedJournal(code_blocked)

        # Test with tensor_descriptor indexing + persistent_interleaved
        code_interleaved, result_interleaved = code_and_output(
            tensor_descriptor_kernel,
            args,
            pid_type="persistent_interleaved",
            indexing="tensor_descriptor",
        )
        self.assertExpectedJournal(code_interleaved)

        # Check correctness
        expected = args[0] + args[1]
        torch.testing.assert_close(result_blocked, expected)
        torch.testing.assert_close(result_interleaved, expected)

        # Verify tensor descriptor features in code
        self.assertIn(get_tensor_descriptor_fn_name(), code_blocked)
        self.assertIn(get_tensor_descriptor_fn_name(), code_interleaved)

        # Verify persistent kernel features
        self.assertIn("for virtual_pid in tl.range", code_blocked)
        self.assertIn("for virtual_pid in tl.range", code_interleaved)

        # Verify both produce identical results
        torch.testing.assert_close(result_blocked, result_interleaved, atol=0, rtol=0)

    def test_persistent_kernels_with_range_config_options(self):
        """Test that range configuration options work with persistent kernels."""

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def test_kernel(x: torch.Tensor) -> torch.Tensor:
            result = x.new_empty(x.size())
            for tile in hl.tile(x.size(), block_size=[32, 16]):
                result[tile] = x[tile] + 1
            return result

        args = (torch.randn([64, 96], device=DEVICE),)
        expected = args[0] + 1

        # Test with persistent_blocked + range_unroll_factors
        code_unroll, result_unroll = code_and_output(
            test_kernel, args, pid_type="persistent_blocked", range_unroll_factors=[2]
        )
        self.assertExpectedJournal(code_unroll)
        torch.testing.assert_close(result_unroll, expected)
        self.assertIn("loop_unroll_factor=2", code_unroll)

        # Test with persistent_interleaved + range_num_stages
        code_stages, result_stages = code_and_output(
            test_kernel, args, pid_type="persistent_interleaved", range_num_stages=[3]
        )
        self.assertExpectedJournal(code_stages)
        torch.testing.assert_close(result_stages, expected)
        self.assertIn("num_stages=3", code_stages)

        # Test with persistent_blocked + range_multi_buffers
        code_buffer, result_buffer = code_and_output(
            test_kernel,
            args,
            pid_type="persistent_blocked",
            range_multi_buffers=[False],
        )
        self.assertExpectedJournal(code_buffer)
        torch.testing.assert_close(result_buffer, expected)
        self.assertIn("disallow_acc_multi_buffer=True", code_buffer)

        # Test with persistent_interleaved + range_flatten
        code_flatten, result_flatten = code_and_output(
            test_kernel, args, pid_type="persistent_interleaved", range_flattens=[True]
        )
        self.assertExpectedJournal(code_flatten)
        torch.testing.assert_close(result_flatten, expected)
        self.assertIn("flatten=True", code_flatten)

        # Test combined range options with persistent kernel
        code_combined, result_combined = code_and_output(
            test_kernel,
            args,
            pid_type="persistent_blocked",
            range_unroll_factors=[2],
            range_num_stages=[3],
            range_multi_buffers=[True],
            range_flattens=[False],
        )
        self.assertExpectedJournal(code_combined)
        torch.testing.assert_close(result_combined, expected)

        # Verify all options are present in the generated code
        self.assertIn("loop_unroll_factor=2", code_combined)
        self.assertIn("num_stages=3", code_combined)
        self.assertIn("disallow_acc_multi_buffer=False", code_combined)
        self.assertIn("flatten=False", code_combined)

    @unittest.skipIf(
        DEVICE.type != "cuda" or torch.cuda.get_device_capability() < (12, 0),
        "Warp specialization requires CUDA compute capability >= 12.0",
    )
    def test_persistent_kernels_with_warp_specialize(self):
        """Test that range_warp_specialize works with persistent kernels."""

        @helion.kernel(autotune_effort="none")
        def test_kernel(x: torch.Tensor) -> torch.Tensor:
            result = x.new_empty(x.size())
            for tile in hl.tile(x.size(), block_size=[32, 16]):
                result[tile] = x[tile] + 1
            return result

        args = (torch.randn([64, 96], device=DEVICE),)
        expected = args[0] + 1

        # Test with persistent_blocked + range_warp_specialize
        code_warp, result_warp = code_and_output(
            test_kernel,
            args,
            pid_type="persistent_blocked",
            range_warp_specializes=[True],
        )
        self.assertExpectedJournal(code_warp)
        torch.testing.assert_close(result_warp, expected)
        self.assertIn("warp_specialize=True", code_warp)


if __name__ == "__main__":
    unittest.main()
