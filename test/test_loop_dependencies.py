from __future__ import annotations

import ast
import textwrap
import unittest

import pytest
import torch

import helion
from helion import exc
from helion._compiler import ast_extension
from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.ast_extension import LoopType
from helion._compiler.loop_dependency_checker import TileDependencyAnalysis
from helion._compiler.loop_dependency_checker import TileDependencyKind
from helion._compiler.loop_dependency_checker import TileDependencySynchronization
from helion._compiler.loop_dependency_checker import analyze_top_level_tile_dependencies
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


def _analyze_source(source: str) -> TileDependencyAnalysis:
    module = ast_extension.convert(ast.parse(textwrap.dedent(source)))
    assert isinstance(module, ast.Module)
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    for statement in function.body:
        if isinstance(statement, ast.For):
            assert isinstance(statement, ExtendedAST)
            statement._loop_type = LoopType.GRID
    return analyze_top_level_tile_dependencies(function.body)


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


class TestTileDependencyAnalysis(unittest.TestCase):
    def test_unsynchronized_read_after_write(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp, out, n):
                for i in range(n):
                    tmp[i] = x[i]
                for j in range(n):
                    out[j] = tmp[j]
            """
        )
        self.assertEqual(analysis.root_count, 2)
        self.assertEqual(analysis.source_phase_starts, frozenset())
        self.assertEqual(len(analysis.tile_dependencies), 1)
        dependency = analysis.tile_dependencies[0]
        self.assertEqual((dependency.producer_root, dependency.consumer_root), (0, 1))
        self.assertEqual(dependency.name, "tmp")
        self.assertEqual(
            dependency.kinds, frozenset({TileDependencyKind.READ_AFTER_WRITE})
        )
        self.assertIs(
            dependency.synchronization,
            TileDependencySynchronization.UNSYNCHRONIZED,
        )

    def test_tracks_multiple_consumers_without_creating_a_schedule(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp, out0, out1, n):
                for i in range(n):
                    tmp[i] = x[i]
                for j in range(n):
                    out0[j] = tmp[j]
                for k in range(n):
                    out1[k] = tmp[k]
            """
        )
        self.assertEqual(
            [
                (dependency.producer_root, dependency.consumer_root)
                for dependency in analysis.tile_dependencies
            ],
            [(0, 1), (0, 2)],
        )
        self.assertEqual(len(analysis.unsynchronized_tile_dependencies), 2)

    def test_combines_read_and_write_kinds(self) -> None:
        analysis = _analyze_source(
            """
            def kernel(x, tmp, n):
                for i in range(n):
                    tmp[i] = x[i]
                for j in range(n):
                    tmp[j] += 1
            """
        )
        self.assertEqual(
            analysis.tile_dependencies[0].kinds,
            frozenset(
                {
                    TileDependencyKind.READ_AFTER_WRITE,
                    TileDependencyKind.WRITE_AFTER_WRITE,
                }
            ),
        )


@onlyBackends(["triton", "cute"])
class TestLoops(RefEagerTestBase, TestCase):
    @skipIfRefEager("compiled HostFunction metadata is unavailable in ref eager mode")
    def test_tile_dependency_analysis_is_attached_to_device_ir(self) -> None:
        x = torch.empty(8, device=DEVICE)
        bound = tile_dependency_info_across_barrier.bind((x,))
        host_function = bound.host_function
        assert host_function is not None
        analysis = host_function.compiler_state.tile_dependency_analysis
        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis.source_phase_starts, frozenset({1}))
        self.assertEqual(len(analysis.tile_dependencies), 1)
        dependency = analysis.tile_dependencies[0]
        self.assertIs(
            dependency.synchronization,
            TileDependencySynchronization.SOURCE_BARRIER,
        )
        self.assertEqual(
            host_function.device_ir.tile_dependencies,
            analysis.tile_dependencies,
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
