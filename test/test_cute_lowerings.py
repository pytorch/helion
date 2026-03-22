from __future__ import annotations

import ast
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.fx import Graph

from helion._compiler.aten_lowering import codegen_mm_cute
from helion._compiler.backend import CuteBackend
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.cute_mma import _choose_mma_impl
from helion._compiler.cute.cute_mma import _get_mma_k_loop_info
from helion._compiler.cute.cute_mma import _make_tcgen05_layout_plan_setup
from helion._compiler.cute.cute_mma import _mma_result_can_be_deferred
from helion._compiler.cute.cute_mma import _new_tcgen05_layout_plan
from helion._compiler.cute.cute_mma import _tcgen05_pipeline_arrive_count
from helion._compiler.cute.cute_mma import _tcgen05_tmem_barrier_thread_count
from helion._compiler.cute.cute_mma import can_codegen_cute_mma_aten
from helion._compiler.cute.cute_reshape import codegen_cute_permute
from helion._compiler.cute.cute_reshape import codegen_cute_reshape
from helion._compiler.cute.matmul_fallback import _emit_cute_grouped_sum_reduction
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.tile_strategy import DeviceGridState
from helion._compiler.tile_strategy import DeviceLoopState
from helion.language.memory_ops import _codegen_cute_store_permute_lane_loops
from helion.language.memory_ops import _cute_combined_mask
from helion.language.memory_ops import _cute_index_exprs


class _FakeBlockSize:
    def __init__(self, size: int, *, reduction: bool = False) -> None:
        self.size = size
        self.reduction = reduction

    def from_config(self, config: object) -> int:
        return self.size


class _FakeDeviceFunction:
    def __init__(self) -> None:
        self.config = object()
        self._counts: dict[str, int] = {}

    def new_var(self, prefix: str) -> str:
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        return f"{prefix}_{self._counts[prefix]}"


class _FakeGenerateAST:
    def __init__(
        self,
        active_block_ids: set[int],
        current_grid_state: object | None = None,
    ) -> None:
        self.device_function = _FakeDeviceFunction()
        self.active_device_loops = {
            block_id: [SimpleNamespace(block_thread_axes={block_id: block_id})]
            for block_id in active_block_ids
        }
        self.current_grid_state = current_grid_state
        self.statements: list[ast.AST] = []

    def add_statement(self, stmt: ast.AST) -> None:
        self.statements.append(stmt)

    def index_var(self, block_idx: int) -> str:
        return f"indices_{block_idx}"

    def offset_var(self, block_idx: int) -> str:
        return f"offset_{block_idx}"


class _FakeLoopStrategy:
    def __init__(self, block_ids: list[int]) -> None:
        self.block_ids = block_ids

    def offset_var(self, block_idx: int) -> str:
        return f"offset_{block_idx}"

    def index_var(self, block_idx: int) -> str:
        return f"indices_{block_idx}"

    def mask_var(self, block_idx: int) -> None:
        return None


class _FakeMaskedLoopStrategy(_FakeLoopStrategy):
    def mask_var(self, block_idx: int) -> str:
        return f"mask_{block_idx}"


class _FakeGenerateASTForLaneStore:
    def __init__(self, grid_state: DeviceGridState) -> None:
        self.current_grid_state = grid_state
        self.active_device_loops = {}
        self.statements: list[ast.AST] = []
        self.device_function = SimpleNamespace(
            config=object(),
            new_var=lambda prefix: prefix,
            tensor_arg=lambda tensor: SimpleNamespace(name="out"),
        )

    def add_statement(self, stmt: ast.AST) -> None:
        self.statements.append(stmt)

    def lift(self, expr: ast.AST, *, dce: bool = False, prefix: str = "tmp") -> ast.AST:
        return expr


class _FakeMaskCodegen:
    def __init__(self, strategy: _FakeLoopStrategy, block_ids: set[int]) -> None:
        self.active_device_loops = {
            block_id: [SimpleNamespace(strategy=strategy)] for block_id in block_ids
        }

    def lift(self, expr: ast.AST, *, dce: bool = False, prefix: str = "tmp") -> ast.AST:
        return SimpleNamespace(id=f"{prefix}_0")


class _FakeCuteReductionCodegen:
    def __init__(self) -> None:
        self.device_function = _FakeDeviceFunction()
        self.active_device_loops = {
            0: [
                SimpleNamespace(
                    thread_axis_sizes={0: 3},
                    block_thread_axes={0: 0},
                )
            ],
            1: [
                SimpleNamespace(
                    thread_axis_sizes={1: 16},
                    block_thread_axes={1: 1},
                )
            ],
        }
        self.current_grid_state = None
        self.max_thread_block_dims = (3, 16, 1)
        self.statements: list[object] = []

    def add_statement(self, stmt: object) -> None:
        self.statements.append(stmt)


def _fake_env(
    block_ids_by_size: dict[int, int | None],
) -> object:
    block_sizes = {
        block_id: _FakeBlockSize(size)
        for size, block_id in block_ids_by_size.items()
        if block_id is not None
    }
    return SimpleNamespace(
        backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Float16"),
        block_sizes=block_sizes,
        get_block_id=lambda size: block_ids_by_size.get(int(size)),
        known_equal=lambda lhs, rhs: int(lhs) == int(rhs),
        resolve_block_id=lambda size: block_ids_by_size.get(int(size)),
    )


def _fake_device_loop(block_id: int) -> DeviceLoopState:
    return DeviceLoopState(
        strategy=_FakeLoopStrategy([block_id]),
        block_id_to_info={},
        for_node=ast.For(
            target=ast.Name(id=f"i_{block_id}", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[ast.Constant(value=1)],
                keywords=[],
            ),
            body=[],
            orelse=[],
            type_comment=None,
        ),
        inner_statements=[],
        block_thread_axes={block_id: 0},
    )


class TestCuteLowerings(unittest.TestCase):
    def test_mma_k_loop_selection_uses_reduction_block(self) -> None:
        env = _fake_env({32: 0, 64: 1, 16: 2, 7: 3})
        r_loop = _fake_device_loop(3)
        k_loop = _fake_device_loop(1)
        cg = SimpleNamespace(
            active_device_loops={3: [r_loop], 1: [k_loop]},
            device_function=_FakeDeviceFunction(),
        )

        loop_info = _get_mma_k_loop_info(
            cg,
            env,
            torch.empty(32, 64),
            torch.empty(64, 16),
        )

        self.assertIsNotNone(loop_info)
        assert loop_info is not None
        device_loop, k_block_id, k_offset_var, bk = loop_info
        self.assertIs(device_loop, k_loop)
        self.assertEqual(k_block_id, 1)
        self.assertEqual(k_offset_var, "offset_1")
        self.assertEqual(bk, 64)

    def test_mma_result_defer_only_when_node_exits_loop(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        addmm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        graph.output(addmm)
        self.assertTrue(_mma_result_can_be_deferred(addmm))

        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        addmm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        relu = graph.call_function(torch.ops.aten.relu.default, args=(addmm,))
        graph.output(relu)
        self.assertFalse(_mma_result_can_be_deferred(addmm))

    def test_mma_k_loop_selection_prefers_node_graph_when_symbols_do_not_resolve(
        self,
    ) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        dot = graph.call_function(torch.matmul, args=(lhs, rhs))
        graph.output(dot)

        k_loop = _fake_device_loop(2)
        outer_loop = _fake_device_loop(3)
        env = SimpleNamespace(
            block_sizes={2: _FakeBlockSize(64), 3: _FakeBlockSize(7)},
            known_equal=lambda lhs, rhs: False,
            resolve_block_id=lambda size: None,
        )
        cg = SimpleNamespace(
            active_device_loops={3: [outer_loop], 2: [k_loop]},
            codegen_graphs=[
                ForLoopGraphInfo(graph_id=0, graph=graph, node_args=[], block_ids=[2])
            ],
            device_function=_FakeDeviceFunction(),
        )

        loop_info = _get_mma_k_loop_info(
            cg,
            env,
            torch.empty(32, 64),
            torch.empty(64, 16),
            fx_node=dot,
        )

        self.assertIsNotNone(loop_info)
        assert loop_info is not None
        device_loop, k_block_id, k_offset_var, bk = loop_info
        self.assertIs(device_loop, k_loop)
        self.assertEqual(k_block_id, 2)
        self.assertEqual(k_offset_var, "offset_2")
        self.assertEqual(bk, 64)

    def test_permute_codegen_materializes_non_store_use(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        permute = graph.call_function(
            torch.ops.aten.permute.default, args=(inp, [1, 0])
        )
        graph.call_function(torch.ops.aten.add.Tensor, args=(permute, permute))
        inp.meta["val"] = torch.empty(4, 8)
        permute.meta["val"] = torch.empty(8, 4)

        cg = _FakeGenerateAST({0, 1})
        ctx = SimpleNamespace(
            cg=cg,
            env={inp: ast.Name(id="load", ctx=ast.Load())},
        )
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            result = codegen_cute_permute(ctx, permute)

        self.assertNotEqual(ast.unparse(result), "load")
        emitted = "\n".join(ast.unparse(stmt) for stmt in cg.statements)
        self.assertIn("permute_smem", emitted)
        self.assertIn("cute.arch.sync_threads()", emitted)

    def test_reshape_codegen_materializes_nontrivial_view(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(inp, [2, 6]),
        )
        inp.meta["val"] = torch.empty(4, 3)
        reshape.meta["val"] = torch.empty(2, 6)

        cg = _FakeGenerateAST({0, 1, 2, 3})
        ctx = SimpleNamespace(
            cg=cg,
            env={inp: ast.Name(id="load", ctx=ast.Load())},
        )
        env = _fake_env({4: 0, 3: 1, 2: 2, 6: 3})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            result = codegen_cute_reshape(ctx, reshape)

        self.assertNotEqual(ast.unparse(result), "load")
        emitted = "\n".join(ast.unparse(stmt) for stmt in cg.statements)
        self.assertIn("reshape_smem", emitted)
        self.assertIn("cute.arch.sync_threads()", emitted)

    def test_reshape_codegen_includes_grid_lane_offsets(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(inp, [2, 8]),
        )
        graph.call_function(torch.ops.aten.add.Tensor, args=(reshape, reshape))
        inp.meta["val"] = torch.empty(4, 4)
        reshape.meta["val"] = torch.empty(2, 8)

        grid_strategy = SimpleNamespace(
            _lane_var_by_block={0: "lane_0", 1: "lane_1"},
            _elements_per_thread_for_block=lambda block_id: 2,
        )
        grid_state = SimpleNamespace(
            block_thread_axes={0: 0, 1: 1, 2: 0, 3: 1},
            has_lane_loops=lambda: True,
            strategy=grid_strategy,
        )
        cg = _FakeGenerateAST({2, 3}, current_grid_state=grid_state)
        ctx = SimpleNamespace(
            cg=cg,
            env={inp: ast.Name(id="load", ctx=ast.Load())},
        )
        env = _fake_env({4: 0, 2: 2, 8: 3})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            codegen_cute_reshape(ctx, reshape)

        emitted = "\n".join(ast.unparse(stmt) for stmt in cg.statements)
        self.assertIn("thread_idx()[0]) * cutlass.Int32(2)", emitted)
        self.assertIn("cutlass.Int32(lane_0)", emitted)

    def test_codegen_mm_cute_resolves_constant_k_block_ids(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        mm = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        graph.output(mm)
        lhs.meta["val"] = torch.empty(4, 8)
        rhs.meta["val"] = torch.empty(8, 4)
        mm.meta["val"] = torch.empty(4, 4)

        ctx = SimpleNamespace(
            cg=object(),
            env={
                lhs: ast.Name(id="lhs_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(
            get_block_id=lambda size: None,
            resolve_block_id=lambda size: 7 if int(size) == 8 else None,
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.aten_lowering._emit_cute_matmul",
                return_value=ast.Name(id="mm_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = codegen_mm_cute(ctx, mm)

        self.assertEqual(ast.unparse(result), "mm_result")
        self.assertEqual(emit.call_args.kwargs["k_block_id"], 7)

    def test_can_codegen_cute_mma_aten_requires_exclusive_loop_body(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        addmm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        graph.output(addmm)
        acc.meta["val"] = torch.empty(16, 8, dtype=torch.float32)
        lhs.meta["val"] = torch.empty(16, 64, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(64, 8, dtype=torch.float16)
        addmm.meta["val"] = torch.empty(16, 8, dtype=torch.float32)
        with patch(
            "helion._compiler.cute.cute_mma.is_mma_compatible_aten",
            return_value=True,
        ):
            self.assertTrue(can_codegen_cute_mma_aten(addmm, with_acc=True))

        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        addmm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        graph.call_function(torch.ops.aten.neg.default, args=(lhs,))
        graph.output(addmm)
        acc.meta["val"] = torch.empty(16, 8, dtype=torch.float32)
        lhs.meta["val"] = torch.empty(16, 64, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(64, 8, dtype=torch.float16)
        addmm.meta["val"] = torch.empty(16, 8, dtype=torch.float32)
        with patch(
            "helion._compiler.cute.cute_mma.is_mma_compatible_aten",
            return_value=True,
        ):
            self.assertFalse(can_codegen_cute_mma_aten(addmm, with_acc=True))

    def test_lane_loop_store_permute_codegen_stays_inline(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        permute = graph.call_function(
            torch.ops.aten.permute.default,
            args=(inp, [1, 0]),
        )
        inp.meta["val"] = torch.empty(2, 2)
        permute.meta["val"] = torch.empty(2, 2)

        grid_state = DeviceGridState(
            strategy=SimpleNamespace(block_ids=[0, 1]),
            block_id_to_info={},
            lane_loops=[("lane_0", 2)],
            lane_setup_statements=[],
        )
        codegen = _FakeGenerateASTForLaneStore(grid_state)
        state = SimpleNamespace(
            codegen=codegen,
            device_function=codegen.device_function,
        )
        env = SimpleNamespace(
            backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Float32"),
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.generate_ast.GenerateAST",
                _FakeGenerateASTForLaneStore,
            ),
            patch(
                "helion.language.memory_ops._cute_index_exprs",
                return_value=["i0", "i1"],
            ),
            patch("helion.language.memory_ops._cute_combined_mask", return_value=None),
            patch(
                "helion._compiler.cute.cute_reshape._store_permute_info",
                return_value=(inp, [1, 0]),
            ),
            patch(
                "helion._compiler.cute.cute_reshape._permute_reorders_active_dims",
                return_value=True,
            ),
            patch(
                "helion._compiler.cute.cute_reshape._shape_op_needs_materialization",
                return_value=False,
            ),
            patch(
                "helion._compiler.cute.cute_reshape._get_tile_shape",
                return_value=[2, 2],
            ),
            patch(
                "helion._compiler.cute.cute_reshape._get_dim_local_coord",
                return_value="0",
            ),
            patch(
                "helion._compiler.cute.cute_reshape._flat_index_from_coords",
                side_effect=["0", "1"],
            ),
            patch(
                "helion._compiler.cute.cute_reshape._coords_from_flat_index",
                return_value=["0", "1"],
            ),
        ):
            result = _codegen_cute_store_permute_lane_loops(
                state,
                torch.empty(2, 2),
                [slice(None), slice(None)],
                [slice(None), slice(None)],
                ast.Name(id="value", ctx=ast.Load()),
                None,
                permute,
            )

        assert result is not None
        code = ast.unparse(result)
        self.assertIn("cute.arch.sync_threads()", code)
        self.assertIn("permute_smem", code)
        self.assertIn("out.__setitem__((i0, i1)", code)
        self.assertEqual(grid_state.outer_suffix, [])

    def test_choose_mma_impl_forced_incompatible_override_falls_back(self) -> None:
        support = SimpleNamespace(
            supported_impls=("universal", "warp", "tcgen05"),
            warp_f16bf16=True,
            tcgen05_f16bf16=True,
        )

        with patch(
            "helion._compiler.cute.cute_mma.get_cute_mma_support",
            return_value=support,
        ):
            with patch.dict(
                "os.environ", {"HELION_CUTE_MMA_IMPL": "warp"}, clear=False
            ):
                self.assertEqual(
                    _choose_mma_impl(torch.float16, bm=64, bn=8, bk=16),
                    "universal",
                )
                self.assertEqual(
                    _choose_mma_impl(torch.float32, bm=16, bn=8, bk=16),
                    "universal",
                )
            with patch.dict(
                "os.environ", {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False
            ):
                self.assertEqual(
                    _choose_mma_impl(torch.float16, bm=16, bn=8, bk=16),
                    "universal",
                )
                self.assertEqual(
                    _choose_mma_impl(torch.float16, bm=64, bn=16, bk=16),
                    "universal",
                )

    def test_tcgen05_thread_counts_match_participants_and_cta(self) -> None:
        self.assertEqual(_tcgen05_pipeline_arrive_count(64), 4)
        self.assertEqual(_tcgen05_pipeline_arrive_count(128), 8)
        self.assertEqual(_tcgen05_tmem_barrier_thread_count(64, 8), 512)
        self.assertEqual(_tcgen05_tmem_barrier_thread_count(128, 8), 1024)

    def test_tcgen05_layout_plan_setup_uses_pipeline_thread_counts(self) -> None:
        df = _FakeDeviceFunction()
        plan = _new_tcgen05_layout_plan(df)
        stmts = _make_tcgen05_layout_plan_setup(
            plan,
            "tiled_mma",
            bm=128,
            bn=8,
            bk=16,
            input_dtype_str="cutlass.Float16",
            acc_dtype_str="cutlass.Float32",
        )

        emitted = "\n".join(ast.unparse(stmt) for stmt in stmts)
        self.assertIn("tcgen05_pipeline_arrive_count_1 = cutlass.Int32(8)", emitted)
        self.assertNotIn(
            "tcgen05_pipeline_arrive_count_1 = cutlass.Int32(256)", emitted
        )

    def test_cute_grouped_sum_reduction_uses_tree_for_non_warp_multiple_groups(
        self,
    ) -> None:
        cg = _FakeCuteReductionCodegen()
        env = SimpleNamespace(
            backend=CuteBackend(),
            index_dtype=torch.int32,
        )
        loop_state = SimpleNamespace(block_thread_axes={1: 1})

        with patch.object(CompileEnvironment, "current", return_value=env):
            result = _emit_cute_grouped_sum_reduction(
                cg,
                "dot_input",
                value_dtype=torch.float32,
                loop_state=loop_state,
                k_block_id=1,
            )

        self.assertEqual(result, "dot_reduce_result_1")
        emitted = "\n".join(
            ast.unparse(stmt) if isinstance(stmt, ast.AST) else str(stmt)
            for stmt in cg.statements
        )
        self.assertNotIn("cute.arch.warp_reduction_sum", emitted)
        self.assertIn("dot_reduce_smem_1[dot_lane_1] = dot_masked_input_0_1", emitted)
        self.assertIn("+ 1 < 48", emitted)

    def test_cute_index_exprs_skip_none_axes_and_zero_singletons(self) -> None:
        state = SimpleNamespace(
            codegen=SimpleNamespace(
                active_device_loops={
                    1: [SimpleNamespace(strategy=_FakeLoopStrategy([1]))],
                },
                lift=lambda expr, *, dce=False, prefix="tmp": SimpleNamespace(
                    id="lifted_index"
                ),
            ),
            sympy_expr=lambda expr: str(expr),
        )
        env = SimpleNamespace(
            get_block_id=lambda size: 1 if int(size) == 8 else None,
            known_equal=lambda lhs, rhs: int(lhs) == int(rhs),
            resolve_block_id=lambda size: 1 if int(size) == 8 else None,
            block_sizes=[SimpleNamespace(size=8, block_id=1)],
        )

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertEqual(
                _cute_index_exprs(
                    state,
                    [None, slice(None)],
                    tensor=torch.empty(8),
                    inactive_slice_expr="None",
                    inactive_singleton_slice_expr="0",
                ),
                ["indices_1"],
            )
            self.assertEqual(
                _cute_index_exprs(
                    state,
                    [slice(None), slice(None)],
                    tensor=torch.empty(1, 8),
                    inactive_slice_expr="None",
                    inactive_singleton_slice_expr="0",
                ),
                ["0", "indices_1"],
            )

    def test_cute_combined_mask_skips_none_axes(self) -> None:
        state = SimpleNamespace(
            codegen=_FakeMaskCodegen(_FakeMaskedLoopStrategy([1]), {1})
        )
        env = SimpleNamespace(
            get_block_id=lambda size: 1 if int(size) == 8 else None,
            known_equal=lambda lhs, rhs: int(lhs) == int(rhs),
            resolve_block_id=lambda size: 1 if int(size) == 8 else None,
            block_sizes=[SimpleNamespace(size=8, block_id=1)],
        )

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertEqual(
                _cute_combined_mask(
                    state,
                    [None, slice(None)],
                    None,
                    tensor=torch.empty(8),
                ),
                "(mask_1)",
            )


if __name__ == "__main__":
    unittest.main()
