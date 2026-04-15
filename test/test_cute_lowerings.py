from __future__ import annotations

import ast
import contextlib
import operator
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sympy
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph

import helion
from helion import exc
from helion._compiler.ast_extension import expr_from_string
from helion._compiler.aten_lowering import _pallas_argreduce
from helion._compiler.aten_lowering import _should_use_cute_argreduce_lowering
from helion._compiler.aten_lowering import _triton_argreduce
from helion._compiler.aten_lowering import codegen_iota_cute
from helion._compiler.aten_lowering import codegen_mm_cute
from helion._compiler.aten_lowering import codegen_squeeze_cute
from helion._compiler.aten_lowering import codegen_stack_cute
from helion._compiler.aten_lowering import codegen_unsqueeze_cute
from helion._compiler.aten_lowering import codegen_view_cute
from helion._compiler.backend import CuteBackend
from helion._compiler.backend import PallasBackend
from helion._compiler.backend import TritonBackend
from helion._compiler.backend import _detect_mma_loop
from helion._compiler.backend import _loop_contains_matmul
from helion._compiler.backend import _loop_may_use_mma
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.argreduce import codegen_cute_tile_argreduce
from helion._compiler.cute.cute_mma import _choose_mma_impl
from helion._compiler.cute.cute_mma import _get_mma_k_loop_info
from helion._compiler.cute.cute_mma import _make_tcgen05_layout_plan_setup
from helion._compiler.cute.cute_mma import _mma_result_can_be_deferred
from helion._compiler.cute.cute_mma import _new_tcgen05_layout_plan
from helion._compiler.cute.cute_mma import _tcgen05_pipeline_arrive_count
from helion._compiler.cute.cute_mma import _tcgen05_tmem_barrier_thread_count
from helion._compiler.cute.cute_mma import can_codegen_cute_mma_aten
from helion._compiler.cute.cute_reshape import _get_dim_local_coord
from helion._compiler.cute.cute_reshape import codegen_cute_permute
from helion._compiler.cute.cute_reshape import codegen_cute_reshape
from helion._compiler.cute.indexing import CutePackedAffineLoad
from helion._compiler.cute.indexing import CutePackedTerms
from helion._compiler.cute.indexing import CuteShapeChainView
from helion._compiler.cute.indexing import is_cute_shape_chain_target
from helion._compiler.cute.indexing import match_cute_affine_range_iota
from helion._compiler.cute.indexing import match_cute_duplicate_stack_reshape_rhs
from helion._compiler.cute.indexing import match_cute_stack_reshape_rhs
from helion._compiler.cute.matmul_fallback import _emit_cute_grouped_sum_reduction
from helion._compiler.cute.matmul_utils import cute_resolve_active_block_id
from helion._compiler.cute.matmul_utils import cute_resolve_active_matmul_k_block_id
from helion._compiler.cute.matmul_utils import cute_static_k_invariant_extent
from helion._compiler.cute.matmul_utils import cute_supports_scalar_matmul_fallback
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.host_function import HostFunction
from helion._compiler.reduction_strategy import PersistentReductionStrategy
from helion._compiler.tile_strategy import DeviceGridState
from helion._compiler.tile_strategy import DeviceLoopState
from helion._compiler.variable_origin import TileBeginOrigin
from helion._testing import DEVICE
from helion._testing import onlyBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.language._tracing_ops import _mask_to
from helion.language._tracing_ops import _new_var
from helion.language.matmul_ops import _cute_dot_outer_accumulates_result
from helion.language.memory_ops import _codegen_cute_store_permute_lane_loops
from helion.language.memory_ops import _cute_combined_mask
from helion.language.memory_ops import _cute_index_exprs
from helion.language.memory_ops import _maybe_codegen_cute_packed_affine_lhs_load
from helion.language.memory_ops import load


class _FakeBlockSize:
    def __init__(
        self, size: int, *, block_id: int | None = None, reduction: bool = False
    ) -> None:
        self.size = size
        self.block_id = block_id
        self.reduction = reduction

    def from_config(self, config: object) -> int:
        return self.size

    def from_config_assert(self, config: object) -> int:
        return self.size

    def is_flattened(self, config: object) -> bool:
        return False


class _FakeDeviceFunction:
    def __init__(self) -> None:
        self.config = object()
        self._counts: dict[str, int] = {}
        self.block_size_var_cache: dict[tuple[int, ...], str] = {}

    def new_var(self, prefix: str, **kwargs: object) -> str:
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
            block_id: [
                SimpleNamespace(
                    strategy=_FakeLoopStrategy([block_id]),
                    block_thread_axes={block_id: block_id},
                )
            ]
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


class _FakeArgreduceTileStrategy:
    def shape_str(self, shape: list[object]) -> str:
        return f"[{', '.join(map(str, shape))}]"

    def shape_dims(self, shape: list[object]) -> list[str]:
        return [str(dim) for dim in shape]


def _fake_env(
    block_ids_by_size: dict[int, int | None],
) -> object:
    block_sizes = {
        block_id: _FakeBlockSize(size)
        for size, block_id in block_ids_by_size.items()
        if block_id is not None
    }
    backend = SimpleNamespace(
        dtype_str=lambda dtype: (
            "cutlass.Int32" if dtype == torch.int32 else "cutlass.Float16"
        ),
        cast_expr=lambda expr, dtype_str: f"{dtype_str}({expr})",
    )
    return SimpleNamespace(
        backend=backend,
        block_sizes=block_sizes,
        canonical_block_id=lambda block_id: block_id,
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


@onlyBackends(["cute"])
class TestCuteLowerings(unittest.TestCase):
    def _argreduce_ctx(self, inp: torch.fx.Node) -> object:
        return SimpleNamespace(
            env={inp: ast.Name(id="x", ctx=ast.Load())},
            cg=SimpleNamespace(
                device_function=SimpleNamespace(
                    tile_strategy=_FakeArgreduceTileStrategy()
                )
            ),
        )

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

    def test_detect_mma_loop_allows_serialized_k_reduction_loop(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        graph.output(acc)

        fn = _FakeDeviceFunction()
        fn.codegen = SimpleNamespace(
            codegen_graphs=[
                ForLoopGraphInfo(
                    graph_id=0, graph=graph, node_args=[acc], block_ids=[2]
                )
            ]
        )

        with (
            patch(
                "helion._compiler.host_function.HostFunction.current",
                return_value=SimpleNamespace(
                    device_ir=SimpleNamespace(grid_block_ids=[[0, 1]])
                ),
            ),
            patch(
                "helion._compiler.cute.cute_mma.can_codegen_cute_mma_aten",
                return_value=True,
            ),
        ):
            self.assertTrue(
                _detect_mma_loop(
                    fn,
                    [2],
                    block_sizes=[16],
                    num_threads_config=[1],
                )
            )

    def test_detect_mma_loop_rejects_serialized_root_grid_axes(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        graph.output(acc)

        fn = _FakeDeviceFunction()
        fn.codegen = SimpleNamespace(
            codegen_graphs=[
                ForLoopGraphInfo(
                    graph_id=0, graph=graph, node_args=[acc], block_ids=[0]
                )
            ]
        )

        with (
            patch(
                "helion._compiler.host_function.HostFunction.current",
                return_value=SimpleNamespace(
                    device_ir=SimpleNamespace(grid_block_ids=[[0, 1]])
                ),
            ),
            patch(
                "helion._compiler.cute.cute_mma.can_codegen_cute_mma_aten",
                return_value=True,
            ),
        ):
            self.assertFalse(
                _detect_mma_loop(
                    fn,
                    [0],
                    block_sizes=[64],
                    num_threads_config=[1],
                )
            )

    def test_tcgen05_codegen_ignores_serialized_k_loop_threads(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma_codegen_only(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 8, device=DEVICE, dtype=torch.float16),
        )
        config = helion.Config(block_sizes=[64, 8, 16])
        support = SimpleNamespace(
            supported_impls=("universal", "warp", "tcgen05"),
            warp_f16bf16=True,
            tcgen05_f16bf16=True,
        )

        with (
            patch.dict("os.environ", {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
            patch(
                "helion._compiler.cute.cute_mma.get_cute_mma_support",
                return_value=support,
            ),
            patch(
                "helion._compiler.cute.mma_support.get_cute_mma_support",
                return_value=support,
            ),
        ):
            code = cute_matmul_mma_codegen_only.bind(args).to_triton_code(config)

        self.assertIn("cutlass.utils.blackwell_helpers.make_trivial_tiled_mma", code)
        self.assertIn("cute.nvgpu.tcgen05.make_umma_smem_desc", code)
        self.assertNotIn("cute.arch.warp_reduction_sum", code)

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

        self.assertEqual(ast.unparse(result), "load")
        self.assertEqual(cg.statements, [])

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
            coord = _get_dim_local_coord(cg, inp.meta["val"], 0)
            result = codegen_cute_reshape(ctx, reshape)

        self.assertIn("thread_idx()[0]) * cutlass.Int32(2)", coord)
        self.assertIn("cutlass.Int32(lane_0)", coord)
        self.assertEqual(ast.unparse(result), "load")
        self.assertEqual(cg.statements, [])

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
            cg=SimpleNamespace(
                current_grid_state=SimpleNamespace(block_ids=[7]),
                active_device_loops={},
            ),
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

    def test_codegen_mm_cute_does_not_pack_distinct_rhs_without_packed_lhs(
        self,
    ) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 4]))
        mm = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        graph.output(mm)
        lhs.meta["val"] = torch.empty(4, 16)
        lo.meta["val"] = torch.empty(8, 4)
        hi.meta["val"] = torch.empty(8, 4)
        stack.meta["val"] = torch.empty(8, 2, 4)
        rhs.meta["val"] = torch.empty(16, 4)
        mm.meta["val"] = torch.empty(4, 4)

        ctx = SimpleNamespace(
            cg=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={
                lhs: ast.Name(id="lhs_tile", ctx=ast.Load()),
                lo: ast.Name(id="lo_tile", ctx=ast.Load()),
                hi: ast.Name(id="hi_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(
            resolve_block_id=lambda size: 11 if int(size) == 8 else None
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.aten_lowering.cute_static_k_invariant_extent",
                return_value=16,
            ),
            patch(
                "helion._compiler.aten_lowering._emit_cute_matmul",
                return_value=ast.Name(id="mm_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = codegen_mm_cute(ctx, mm)

        self.assertEqual(ast.unparse(result), "mm_result")
        self.assertIsNone(emit.call_args.kwargs["k_block_id"])
        self.assertIsInstance(emit.call_args.args[2], ast.AST)

    def test_codegen_mm_cute_packs_distinct_rhs_for_packed_lhs(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 4]))
        mm = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        graph.output(mm)
        lhs.meta["val"] = torch.empty(4, 16)
        lo.meta["val"] = torch.empty(8, 4)
        hi.meta["val"] = torch.empty(8, 4)
        stack.meta["val"] = torch.empty(8, 2, 4)
        rhs.meta["val"] = torch.empty(16, 4)
        mm.meta["val"] = torch.empty(4, 4)

        ctx = SimpleNamespace(
            cg=SimpleNamespace(
                current_grid_state=SimpleNamespace(block_ids=[11]),
                active_device_loops={},
            ),
            env={
                lhs: CutePackedAffineLoad(
                    (
                        ast.Name(id="lhs_lo", ctx=ast.Load()),
                        ast.Name(id="lhs_hi", ctx=ast.Load()),
                    )
                ),
                lo: ast.Name(id="lo_tile", ctx=ast.Load()),
                hi: ast.Name(id="hi_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(
            resolve_block_id=lambda size: 11 if int(size) == 8 else None
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
        self.assertEqual(emit.call_args.kwargs["k_block_id"], 11)
        self.assertIsInstance(emit.call_args.args[2], CutePackedTerms)

    def test_codegen_mm_cute_preserves_default_out_dtype_under_outer_add(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        acc = graph.placeholder("acc")
        mm = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(acc, mm))
        graph.output(add)
        lhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        acc.meta["val"] = torch.empty(4, 4, dtype=torch.float16)
        mm.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        add.meta["val"] = torch.empty(4, 4, dtype=torch.float32)

        ctx = SimpleNamespace(
            cg=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={
                lhs: ast.Name(id="lhs_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(resolve_block_id=lambda size: None)

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.aten_lowering.cute_static_k_invariant_extent",
                return_value=8,
            ),
            patch(
                "helion._compiler.aten_lowering._emit_cute_matmul",
                return_value=ast.Name(id="mm_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = codegen_mm_cute(ctx, mm)

        self.assertEqual(ast.unparse(result), "mm_result")
        self.assertEqual(emit.call_args.kwargs["out_dtype"], torch.float32)

    def test_codegen_mm_cute_does_not_force_integer_outer_add_dtype(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        acc = graph.placeholder("acc")
        mm = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(acc, mm))
        graph.output(add)
        lhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        acc.meta["val"] = torch.empty(4, 4, dtype=torch.int32)
        mm.meta["val"] = torch.empty(4, 4, dtype=torch.float16)
        add.meta["val"] = torch.empty(4, 4, dtype=torch.float16)

        ctx = SimpleNamespace(
            cg=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={
                lhs: ast.Name(id="lhs_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(resolve_block_id=lambda size: None)

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.aten_lowering.cute_static_k_invariant_extent",
                return_value=8,
            ),
            patch(
                "helion._compiler.aten_lowering._emit_cute_matmul",
                return_value=ast.Name(id="mm_result", ctx=ast.Load()),
            ) as emit,
        ):
            codegen_mm_cute(ctx, mm)

        self.assertEqual(emit.call_args.kwargs["out_dtype"], torch.float16)

    def test_codegen_cute_dot_preserves_default_out_dtype_under_outer_add(
        self,
    ) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        acc = graph.placeholder("acc")
        dot_node = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(acc, dot_node))
        graph.output(add)
        lhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        acc.meta["val"] = torch.empty(4, 4, dtype=torch.float16)
        dot_node.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        add.meta["val"] = torch.empty(4, 4, dtype=torch.float32)

        mode = FakeTensorMode()
        fake_lhs = mode.from_tensor(torch.empty(4, 8, dtype=torch.float16))
        fake_rhs = mode.from_tensor(torch.empty(8, 4, dtype=torch.float16))
        state = SimpleNamespace(
            proxy_args=[fake_lhs, fake_rhs, None, None],
            ast_args=[
                ast.Name(id="lhs_tile", ctx=ast.Load()),
                ast.Name(id="rhs_tile", ctx=ast.Load()),
                ast.Constant(value=None),
                None,
            ],
            ast_arg=lambda idx: (
                ast.Name(id="lhs_tile", ctx=ast.Load())
                if idx == 0
                else ast.Name(id="rhs_tile", ctx=ast.Load())
                if idx == 1
                else ast.Constant(value=None)
            ),
            fx_node=dot_node,
            codegen=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={},
        )
        env = SimpleNamespace(resolve_block_id=lambda size: None)

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion.language.matmul_ops.cute_static_k_invariant_extent",
                return_value=8,
            ),
            patch(
                "helion.language.matmul_ops._cute_mma_matches_dot_semantics",
                return_value=False,
            ),
            patch(
                "helion.language.matmul_ops._emit_cute_matmul",
                return_value=ast.Name(id="dot_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = hl.dot._codegen["cute"](state)

        self.assertEqual(ast.unparse(result), "dot_result")
        self.assertEqual(emit.call_args.kwargs["out_dtype"], torch.float32)

    def test_codegen_cute_dot_does_not_force_integer_outer_add_dtype(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        acc = graph.placeholder("acc")
        dot_node = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(acc, dot_node))
        graph.output(add)
        lhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        acc.meta["val"] = torch.empty(4, 4, dtype=torch.int32)
        dot_node.meta["val"] = torch.empty(4, 4, dtype=torch.float16)
        add.meta["val"] = torch.empty(4, 4, dtype=torch.float16)

        mode = FakeTensorMode()
        fake_lhs = mode.from_tensor(torch.empty(4, 8, dtype=torch.float16))
        fake_rhs = mode.from_tensor(torch.empty(8, 4, dtype=torch.float16))
        state = SimpleNamespace(
            proxy_args=[fake_lhs, fake_rhs, None, None],
            ast_args=[
                ast.Name(id="lhs_tile", ctx=ast.Load()),
                ast.Name(id="rhs_tile", ctx=ast.Load()),
                ast.Constant(value=None),
                None,
            ],
            ast_arg=lambda idx: (
                ast.Name(id="lhs_tile", ctx=ast.Load())
                if idx == 0
                else ast.Name(id="rhs_tile", ctx=ast.Load())
                if idx == 1
                else ast.Constant(value=None)
            ),
            fx_node=dot_node,
            codegen=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={},
        )
        env = SimpleNamespace(resolve_block_id=lambda size: None)

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion.language.matmul_ops.cute_static_k_invariant_extent",
                return_value=8,
            ),
            patch(
                "helion.language.matmul_ops._cute_mma_matches_dot_semantics",
                return_value=False,
            ),
            patch(
                "helion.language.matmul_ops._emit_cute_matmul",
                return_value=ast.Name(id="dot_result", ctx=ast.Load()),
            ) as emit,
        ):
            hl.dot._codegen["cute"](state)

        self.assertEqual(emit.call_args.kwargs["out_dtype"], torch.float32)

    def test_codegen_cute_dot_does_not_pack_distinct_rhs_without_packed_lhs(
        self,
    ) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 4]))
        dot_node = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        graph.output(dot_node)
        lhs.meta["val"] = torch.empty(4, 16, dtype=torch.float16)
        lo.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        hi.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        stack.meta["val"] = torch.empty(8, 2, 4, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(16, 4, dtype=torch.float16)
        dot_node.meta["val"] = torch.empty(4, 4, dtype=torch.float16)

        mode = FakeTensorMode()
        fake_lhs = mode.from_tensor(torch.empty(4, 16, dtype=torch.float16))
        fake_rhs = mode.from_tensor(torch.empty(16, 4, dtype=torch.float16))
        state = SimpleNamespace(
            proxy_args=[fake_lhs, fake_rhs, None, None],
            ast_args=[
                ast.Name(id="lhs_tile", ctx=ast.Load()),
                ast.Name(id="rhs_tile", ctx=ast.Load()),
                ast.Constant(value=None),
                None,
            ],
            ast_arg=lambda idx: (
                ast.Name(id="lhs_tile", ctx=ast.Load())
                if idx == 0
                else ast.Name(id="rhs_tile", ctx=ast.Load())
                if idx == 1
                else ast.Constant(value=None)
            ),
            fx_node=dot_node,
            codegen=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={
                lo: ast.Name(id="lo_tile", ctx=ast.Load()),
                hi: ast.Name(id="hi_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(
            resolve_block_id=lambda size: 11 if int(size) == 8 else None
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion.language.matmul_ops.cute_static_k_invariant_extent",
                return_value=16,
            ),
            patch(
                "helion.language.matmul_ops._cute_mma_matches_dot_semantics",
                return_value=False,
            ),
            patch(
                "helion.language.matmul_ops._emit_cute_matmul",
                return_value=ast.Name(id="dot_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = hl.dot._codegen["cute"](state)

        self.assertEqual(ast.unparse(result), "dot_result")
        self.assertIsNone(emit.call_args.kwargs["k_block_id"])
        self.assertIsInstance(emit.call_args.args[2], ast.AST)

    def test_codegen_cute_dot_packs_distinct_rhs_for_packed_lhs(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 4]))
        dot_node = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        graph.output(dot_node)
        lhs.meta["val"] = torch.empty(4, 16, dtype=torch.float16)
        lo.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        hi.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        stack.meta["val"] = torch.empty(8, 2, 4, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(16, 4, dtype=torch.float16)
        dot_node.meta["val"] = torch.empty(4, 4, dtype=torch.float16)

        mode = FakeTensorMode()
        fake_lhs = mode.from_tensor(torch.empty(4, 16, dtype=torch.float16))
        fake_rhs = mode.from_tensor(torch.empty(16, 4, dtype=torch.float16))
        state = SimpleNamespace(
            proxy_args=[fake_lhs, fake_rhs, None, None],
            ast_arg=lambda idx: (
                ast.Name(id="rhs_tile", ctx=ast.Load())
                if idx == 1
                else ast.Constant(value=None)
            ),
            ast_args=[
                CutePackedAffineLoad(
                    (
                        ast.Name(id="lhs_lo", ctx=ast.Load()),
                        ast.Name(id="lhs_hi", ctx=ast.Load()),
                    )
                ),
                ast.Name(id="rhs_tile", ctx=ast.Load()),
                ast.Constant(value=None),
                None,
            ],
            fx_node=dot_node,
            codegen=SimpleNamespace(
                current_grid_state=SimpleNamespace(block_ids=[11]),
                active_device_loops={},
            ),
            env={
                lo: ast.Name(id="lo_tile", ctx=ast.Load()),
                hi: ast.Name(id="hi_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(
            resolve_block_id=lambda size: 11 if int(size) == 8 else None
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion.language.matmul_ops._cute_mma_matches_dot_semantics",
                return_value=False,
            ),
            patch(
                "helion.language.matmul_ops._emit_cute_matmul",
                return_value=ast.Name(id="dot_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = hl.dot._codegen["cute"](state)

        self.assertEqual(ast.unparse(result), "dot_result")
        self.assertEqual(emit.call_args.kwargs["k_block_id"], 11)
        self.assertIsInstance(emit.call_args.args[2], CutePackedTerms)

    def test_match_cute_affine_range_iota_and_duplicate_stack_rhs(self) -> None:
        graph = Graph()
        block_size = graph.placeholder("block_size")
        begin = graph.call_function(hl.tile_begin, args=(block_size,))
        start = graph.call_function(operator.mul, args=(begin, 2))
        length = graph.call_function(operator.mul, args=(block_size, 2))
        iota = graph.call_function(
            torch.ops.prims.iota.default,
            args=(length,),
            kwargs={
                "start": start,
                "step": 1,
                "dtype": torch.int32,
                "device": torch.device("cuda"),
                "requires_grad": False,
            },
        )
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default, args=([packed, packed], 1)
        )
        rhs = graph.call_function(
            torch.ops.aten.view.default,
            args=(stack, [length, 8]),
        )
        graph.output((iota, rhs))

        affine = match_cute_affine_range_iota(iota)
        self.assertIsNotNone(affine)
        assert affine is not None
        self.assertEqual(affine.base, block_size)
        self.assertEqual(affine.factor, 2)

        duplicate_rhs = match_cute_duplicate_stack_reshape_rhs(rhs)
        self.assertEqual(duplicate_rhs, (packed, 2))

    def test_match_cute_stack_reshape_rhs_allows_distinct_stack_inputs(self) -> None:
        graph = Graph()
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 8]))
        graph.output(rhs)
        lo.meta["val"] = torch.empty(8, 8)
        hi.meta["val"] = torch.empty(8, 8)
        stack.meta["val"] = torch.empty(8, 2, 8)
        rhs.meta["val"] = torch.empty(16, 8)

        matched = match_cute_stack_reshape_rhs(rhs)
        assert matched is not None
        tensors, factor = matched
        self.assertEqual(tensors, (lo, hi))
        self.assertEqual(factor, 2)

    def test_match_cute_duplicate_stack_reshape_rhs_rejects_distinct_stack_inputs(
        self,
    ) -> None:
        graph = Graph()
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 8]))
        graph.output(rhs)
        lo.meta["val"] = torch.empty(8, 8)
        hi.meta["val"] = torch.empty(8, 8)
        stack.meta["val"] = torch.empty(8, 2, 8)
        rhs.meta["val"] = torch.empty(16, 8)

        self.assertIsNone(match_cute_duplicate_stack_reshape_rhs(rhs))

    def test_codegen_stack_cute_returns_shape_view_and_reshape_resolves_it(
        self,
    ) -> None:
        graph = Graph()
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([packed, packed], -1),
        )
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(stack, [4, 8]),
        )
        graph.output(reshape)
        packed.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(4, 4, 2)
        reshape.meta["val"] = torch.empty(4, 8)

        cg = _FakeGenerateAST({0, 1})
        packed_ast = ast.Name(id="packed_tile", ctx=ast.Load())
        ctx = SimpleNamespace(cg=cg, env={packed: packed_ast})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            stack_view = codegen_stack_cute(ctx, stack)
            self.assertIsInstance(stack_view, CuteShapeChainView)
            ctx.env[stack] = stack_view
            result = codegen_cute_reshape(ctx, reshape)

        self.assertIsInstance(result, ast.AST)
        self.assertIn("packed_tile", ast.unparse(result))
        self.assertEqual(cg.statements, [])

    def test_codegen_stack_cute_rejects_direct_consumer(self) -> None:
        graph = Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        c = graph.placeholder("c")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([a, b, c], 1),
        )
        graph.call_function(torch.ops.aten.sum.dim_IntList, args=(stack, [0], False))
        graph.output(stack)
        a.meta["val"] = torch.empty(4, 4)
        b.meta["val"] = torch.empty(4, 4)
        c.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(4, 3, 4)

        cg = _FakeGenerateAST({0, 2})
        ctx = SimpleNamespace(
            cg=cg,
            env={
                a: ast.Name(id="a_tile", ctx=ast.Load()),
                b: ast.Name(id="b_tile", ctx=ast.Load()),
                c: ast.Name(id="c_tile", ctx=ast.Load()),
            },
        )
        env = _fake_env({4: 0})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            self.assertRaisesRegex(
                exc.BackendUnsupported,
                "virtual shape-chain direct consumers are not yet supported",
            ),
        ):
            codegen_stack_cute(ctx, stack)

    def test_codegen_stack_cute_rejects_transpose_user(self) -> None:
        graph = Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([a, b], 0),
        )
        graph.call_function(torch.ops.aten.transpose.int, args=(stack, 0, 1))
        graph.output(stack)
        a.meta["val"] = torch.empty(4, 4)
        b.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(2, 4, 4)

        cg = _FakeGenerateAST({1, 2})
        ctx = SimpleNamespace(
            cg=cg,
            env={
                a: ast.Name(id="a_tile", ctx=ast.Load()),
                b: ast.Name(id="b_tile", ctx=ast.Load()),
            },
        )
        env = _fake_env({4: 0})

        self.assertFalse(is_cute_shape_chain_target(torch.ops.aten.transpose.int))
        self.assertFalse(is_cute_shape_chain_target(torch.ops.aten.t.default))
        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            self.assertRaisesRegex(
                exc.BackendUnsupported,
                "virtual shape-chain direct consumers are not yet supported",
            ),
        ):
            codegen_stack_cute(ctx, stack)

    def test_codegen_shape_chain_stays_virtual_through_unsqueeze_until_reshape(
        self,
    ) -> None:
        graph = Graph()
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([packed, packed], -1),
        )
        unsqueeze = graph.call_function(
            torch.ops.aten.unsqueeze.default,
            args=(stack, 1),
        )
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(unsqueeze, [4, 8]),
        )
        graph.output(reshape)
        packed.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(4, 4, 2)
        unsqueeze.meta["val"] = torch.empty(4, 1, 4, 2)
        reshape.meta["val"] = torch.empty(4, 8)

        cg = _FakeGenerateAST({0, 1})
        packed_ast = ast.Name(id="packed_tile", ctx=ast.Load())
        ctx = SimpleNamespace(cg=cg, env={packed: packed_ast})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            stack_view = codegen_stack_cute(ctx, stack)
            self.assertIsInstance(stack_view, CuteShapeChainView)
            ctx.env[stack] = stack_view
            unsqueeze_view = codegen_unsqueeze_cute(ctx, unsqueeze)
            self.assertIsInstance(unsqueeze_view, CuteShapeChainView)
            ctx.env[unsqueeze] = unsqueeze_view
            result = codegen_cute_reshape(ctx, reshape)

        self.assertIsInstance(result, ast.AST)
        self.assertIn("packed_tile", ast.unparse(result))

    def test_codegen_unsqueeze_materializes_virtual_view_before_compute(self) -> None:
        graph = Graph()
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([packed, packed], -1),
        )
        view = graph.call_function(
            torch.ops.aten.view.default,
            args=(stack, [4, 8]),
        )
        unsqueeze = graph.call_function(
            torch.ops.aten.unsqueeze.default,
            args=(view, 0),
        )
        graph.call_function(torch.ops.aten.add.Tensor, args=(unsqueeze, unsqueeze))
        graph.output(unsqueeze)
        packed.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(4, 4, 2)
        view.meta["val"] = torch.empty(4, 8)
        unsqueeze.meta["val"] = torch.empty(1, 4, 8)

        cg = _FakeGenerateAST({0, 1})
        packed_ast = ast.Name(id="packed_tile", ctx=ast.Load())
        ctx = SimpleNamespace(cg=cg, env={packed: packed_ast})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            stack_view = codegen_stack_cute(ctx, stack)
            self.assertIsInstance(stack_view, CuteShapeChainView)
            ctx.env[stack] = stack_view
            view_value = codegen_view_cute(ctx, view)
            self.assertIsInstance(view_value, CuteShapeChainView)
            ctx.env[view] = view_value
            result = codegen_unsqueeze_cute(ctx, unsqueeze)

        self.assertIsInstance(result, ast.AST)
        self.assertIn("packed_tile", ast.unparse(result))

    def test_codegen_shape_chain_squeeze_dim_noop_preserves_values(self) -> None:
        graph = Graph()
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([packed, packed], -1),
        )
        squeeze = graph.call_function(
            torch.ops.aten.squeeze.dim,
            args=(stack, 0),
        )
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(squeeze, [4, 8]),
        )
        graph.output(reshape)
        packed.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(4, 4, 2)
        squeeze.meta["val"] = torch.empty(4, 4, 2)
        reshape.meta["val"] = torch.empty(4, 8)

        cg = _FakeGenerateAST({0, 1})
        packed_ast = ast.Name(id="packed_tile", ctx=ast.Load())
        ctx = SimpleNamespace(cg=cg, env={packed: packed_ast})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            stack_view = codegen_stack_cute(ctx, stack)
            self.assertIsInstance(stack_view, CuteShapeChainView)
            ctx.env[stack] = stack_view
            squeeze_view = codegen_squeeze_cute(ctx, squeeze)
            self.assertIsInstance(squeeze_view, CuteShapeChainView)
            ctx.env[squeeze] = squeeze_view
            result = codegen_cute_reshape(ctx, reshape)

        self.assertIsInstance(result, ast.AST)
        rendered = ast.unparse(result)
        self.assertIn("packed_tile", rendered)
        self.assertNotIn("cutlass.Int32(0), cutlass.Int32(0)", rendered)

    def test_codegen_cute_argreduce_gates_scan_on_reduced_lane_loops(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmax = graph.call_function(torch.ops.aten.argmax.default, args=(inp, 1))
        graph.output(argmax)
        inp.meta["val"] = torch.empty(4, 8)
        argmax.meta["val"] = torch.empty(4, dtype=torch.int64)

        grid_strategy = SimpleNamespace(
            _lane_var_by_block={0: "lane_0", 1: "lane_1"},
            _elements_per_thread_for_block=lambda block_id: 2,
        )
        grid_state = SimpleNamespace(
            block_thread_axes={0: 0, 1: 1},
            has_lane_loops=lambda: True,
            lane_loops=[("lane_0", 2), ("lane_1", 2)],
            strategy=grid_strategy,
        )
        cg = _FakeGenerateAST({0, 1}, current_grid_state=grid_state)
        ctx = SimpleNamespace(cg=cg, env={inp: ast.Name(id="inp_tile", ctx=ast.Load())})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            result = codegen_cute_tile_argreduce(
                ctx,
                argmax,
                "argmax",
                dim=1,
                keepdim=False,
            )

        self.assertIsInstance(result, ast.AST)
        emitted = "\n".join(ast.unparse(stmt) for stmt in cg.statements)
        self.assertIn("if lane_1 == 1:", emitted)
        self.assertNotIn("if lane_0 == 1:", emitted)

    def test_loop_contains_matmul_for_root_grid_phase(self) -> None:
        root_graph = Graph()
        x = root_graph.placeholder("x")
        root_loop = root_graph.call_function(_tracing_ops._for_loop, args=(1, x))
        root_graph.output(root_loop)

        nested_graph = Graph()
        acc = nested_graph.placeholder("acc")
        lhs = nested_graph.placeholder("lhs")
        rhs = nested_graph.placeholder("rhs")
        nested_graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        nested_graph.output(acc)

        fn = SimpleNamespace(
            codegen=SimpleNamespace(
                codegen_graphs=[
                    RootGraphInfo(graph_id=0, graph=root_graph, phase_index=0),
                    ForLoopGraphInfo(
                        graph_id=1,
                        graph=nested_graph,
                        node_args=[],
                        block_ids=[2],
                    ),
                ]
            )
        )

        with patch(
            "helion._compiler.host_function.HostFunction.current",
            return_value=SimpleNamespace(
                device_ir=SimpleNamespace(grid_block_ids=[[0, 1]])
            ),
        ):
            self.assertTrue(_loop_contains_matmul(fn, [0, 1]))

    def test_loop_may_use_mma_for_root_grid_phase(self) -> None:
        root_graph = Graph()
        x = root_graph.placeholder("x")
        root_loop = root_graph.call_function(_tracing_ops._for_loop, args=(1, x))
        root_graph.output(root_loop)

        nested_graph = Graph()
        acc = nested_graph.placeholder("acc")
        lhs = nested_graph.placeholder("lhs")
        rhs = nested_graph.placeholder("rhs")
        addmm = nested_graph.call_function(
            torch.ops.aten.addmm.default, args=(acc, lhs, rhs)
        )
        nested_graph.output(addmm)
        acc.meta["val"] = torch.empty(64, 8, dtype=torch.float32)
        lhs.meta["val"] = torch.empty(64, 16, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(16, 8, dtype=torch.float16)
        addmm.meta["val"] = torch.empty(64, 8, dtype=torch.float32)

        fn = SimpleNamespace(
            codegen=SimpleNamespace(
                codegen_graphs=[
                    RootGraphInfo(graph_id=0, graph=root_graph, phase_index=0),
                    ForLoopGraphInfo(
                        graph_id=1,
                        graph=nested_graph,
                        node_args=[],
                        block_ids=[2],
                    ),
                ]
            )
        )

        with (
            patch(
                "helion._compiler.host_function.HostFunction.current",
                return_value=SimpleNamespace(
                    device_ir=SimpleNamespace(grid_block_ids=[[0, 1]])
                ),
            ),
            patch(
                "helion._compiler.cute.cute_mma.can_codegen_cute_mma_aten",
                return_value=True,
            ),
        ):
            self.assertTrue(_loop_may_use_mma(fn, [0, 1]))

    def test_create_loop_strategy_preserves_auto_threads_for_mma_candidate(
        self,
    ) -> None:
        backend = CuteBackend()
        fn = _FakeDeviceFunction()
        fn.codegen = SimpleNamespace(
            codegen_graphs=[
                ForLoopGraphInfo(
                    graph_id=0, graph=Graph(), node_args=[], block_ids=[0, 1]
                )
            ]
        )
        env = SimpleNamespace(
            block_sizes={
                0: _FakeBlockSize(128, block_id=0),
                1: _FakeBlockSize(8, block_id=1),
                2: _FakeBlockSize(16, block_id=2, reduction=True),
            },
            config_spec=SimpleNamespace(
                num_threads=SimpleNamespace(config_get=lambda *args: 0),
                loop_orders=SimpleNamespace(config_get=lambda *args: None),
                l2_groupings=SimpleNamespace(config_get=lambda *args: 1),
            ),
        )
        config = SimpleNamespace(loop_orders=None, l2_groupings=None, num_threads=None)

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.host_function.HostFunction.current",
                return_value=SimpleNamespace(
                    device_ir=SimpleNamespace(grid_block_ids=[[2]])
                ),
            ),
            patch("helion._compiler.backend._loop_contains_matmul", return_value=True),
            patch(
                "helion._compiler.backend._detect_mma_loop",
                return_value=True,
            ) as detect_mma_loop,
            patch(
                "helion._compiler.backend._detect_specialized_mma_loop",
                return_value=True,
            ),
        ):
            strategy = backend.create_loop_strategy(fn, [0, 1], config)

        self.assertEqual(strategy.num_threads, [0, 0])
        self.assertTrue(strategy.mma_mode)
        self.assertEqual(
            detect_mma_loop.call_args_list[0].kwargs["num_threads_config"],
            [128, 8],
        )

    def test_should_use_cute_argreduce_lowering_recurses_nested_for_loops(
        self,
    ) -> None:
        outer_graph = Graph()
        inp = outer_graph.placeholder("inp")
        outer_loop = outer_graph.call_function(_tracing_ops._for_loop, args=(1, inp))
        argmax = outer_graph.call_function(
            torch.ops.aten.argmax.default, args=(outer_loop, 1)
        )
        outer_graph.output(argmax)
        inp.meta["val"] = torch.empty(4, 4)
        outer_loop.meta["val"] = torch.empty(4, 4)
        argmax.meta["val"] = torch.empty(4, dtype=torch.int64)
        argmax.meta["location"] = contextlib.nullcontext()

        mid_graph = Graph()
        mid_inp = mid_graph.placeholder("mid_inp")
        nested_loop = mid_graph.call_function(_tracing_ops._for_loop, args=(2, mid_inp))
        mid_graph.output(nested_loop)

        inner_graph = Graph()
        acc = inner_graph.placeholder("acc")
        lhs = inner_graph.placeholder("lhs")
        rhs = inner_graph.placeholder("rhs")
        addmm = inner_graph.call_function(
            torch.ops.aten.addmm.default, args=(acc, lhs, rhs)
        )
        inner_graph.output(addmm)

        fake_env = SimpleNamespace(backend_name="cute")
        fake_device_ir = SimpleNamespace(
            graphs=[
                SimpleNamespace(graph=outer_graph),
                SimpleNamespace(graph=mid_graph),
                SimpleNamespace(graph=inner_graph),
            ]
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=fake_env),
            patch(
                "helion._compiler.device_ir.DeviceIR.current",
                return_value=fake_device_ir,
            ),
        ):
            self.assertTrue(_should_use_cute_argreduce_lowering(argmax))

    def test_should_use_cute_argreduce_lowering_for_dim_none(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        mm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        argmax = graph.call_function(torch.ops.aten.argmax.default, args=(mm,))
        graph.output(argmax)
        acc.meta["val"] = torch.empty(4, 4)
        lhs.meta["val"] = torch.empty(4, 4)
        rhs.meta["val"] = torch.empty(4, 4)
        mm.meta["val"] = torch.empty(4, 4)
        argmax.meta["val"] = torch.empty((), dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend_name="cute"),
        ):
            self.assertTrue(_should_use_cute_argreduce_lowering(argmax))

    def test_should_use_cute_argreduce_lowering_for_torch_matmul(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        matmul = graph.call_function(torch.matmul, args=(lhs, rhs))
        argmax = graph.call_function(torch.ops.aten.argmax.default, args=(matmul, 1))
        graph.output(argmax)
        lhs.meta["val"] = torch.empty(4, 4)
        rhs.meta["val"] = torch.empty(4, 4)
        matmul.meta["val"] = torch.empty(4, 4)
        argmax.meta["val"] = torch.empty(4, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend_name="cute"),
        ):
            self.assertTrue(_should_use_cute_argreduce_lowering(argmax))

    def test_should_use_cute_argreduce_lowering_for_keepdim(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        mm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        argmax = graph.call_function(
            torch.ops.aten.argmax.default,
            args=(mm, 1, True),
        )
        graph.output(argmax)
        acc.meta["val"] = torch.empty(4, 4)
        lhs.meta["val"] = torch.empty(4, 4)
        rhs.meta["val"] = torch.empty(4, 4)
        mm.meta["val"] = torch.empty(4, 4)
        argmax.meta["val"] = torch.empty(4, 1, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend_name="cute"),
        ):
            self.assertTrue(_should_use_cute_argreduce_lowering(argmax))

    def test_codegen_cute_argreduce_tracks_validity_separately_from_value(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmax = graph.call_function(torch.ops.aten.argmax.default, args=(inp, 1))
        graph.output(argmax)
        inp.meta["val"] = torch.empty(4, 8)
        argmax.meta["val"] = torch.empty(4, dtype=torch.int64)

        cg = _FakeGenerateAST({0, 1})
        ctx = SimpleNamespace(cg=cg, env={inp: ast.Name(id="inp_tile", ctx=ast.Load())})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            result = codegen_cute_tile_argreduce(
                ctx,
                argmax,
                "argmax",
                dim=1,
                keepdim=False,
            )

        self.assertIsInstance(result, ast.AST)
        emitted = "\n".join(
            stmt if isinstance(stmt, str) else ast.unparse(stmt)
            for stmt in cg.statements
        )
        self.assertIn("argreduce_valid_smem", emitted)
        self.assertIn("argmax_best_valid", emitted)
        self.assertIn("candidate_valid", emitted)

    def test_triton_argreduce_supports_dim_none_keepdim(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmax = graph.call_function(
            torch.ops.aten.argmax.default,
            args=(inp, None, True),
        )
        graph.output(argmax)
        inp.meta["val"] = torch.empty(4, 4)
        argmax.meta["val"] = torch.empty(1, 1, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend=TritonBackend()),
        ):
            result = _triton_argreduce(self._argreduce_ctx(inp), argmax, "argmax")

        emitted = ast.unparse(result)
        self.assertIn("tl.reshape(x, [16])", emitted)
        self.assertIn("axis=0", emitted)
        self.assertIn("tl.full([1, 1]", emitted)

    def test_triton_argreduce_preserves_keepdim(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmax = graph.call_function(
            torch.ops.aten.argmax.default,
            args=(inp, 1, True),
        )
        graph.output(argmax)
        inp.meta["val"] = torch.empty(4, 8)
        argmax.meta["val"] = torch.empty(4, 1, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend=TritonBackend()),
        ):
            result = _triton_argreduce(self._argreduce_ctx(inp), argmax, "argmax")

        emitted = ast.unparse(result)
        self.assertIn("axis=1", emitted)
        self.assertIn("tl.reshape", emitted)
        self.assertIn("[4, 1]", emitted)

    def test_pallas_argreduce_supports_dim_none_keepdim(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmin = graph.call_function(
            torch.ops.aten.argmin.default,
            args=(inp, None, True),
        )
        graph.output(argmin)
        inp.meta["val"] = torch.empty(4, 4)
        argmin.meta["val"] = torch.empty(1, 1, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend=PallasBackend()),
        ):
            result = _pallas_argreduce(self._argreduce_ctx(inp), argmin, "argmin")

        emitted = ast.unparse(result)
        self.assertIn("jnp.reshape(x, [16])", emitted)
        self.assertIn("axis=0", emitted)
        self.assertIn("jnp.full([1, 1]", emitted)

    def test_pallas_argreduce_preserves_keepdim(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmin = graph.call_function(
            torch.ops.aten.argmin.default,
            args=(inp, 1, True),
        )
        graph.output(argmin)
        inp.meta["val"] = torch.empty(4, 8)
        argmin.meta["val"] = torch.empty(4, 1, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend=PallasBackend()),
        ):
            result = _pallas_argreduce(self._argreduce_ctx(inp), argmin, "argmin")

        emitted = ast.unparse(result)
        self.assertIn("axis=1", emitted)
        self.assertIn("jnp.reshape", emitted)
        self.assertIn("[4, 1]", emitted)

    def test_cute_dot_outer_accumulates_result_requires_augassign_source(self) -> None:
        graph = Graph()
        acc_in = graph.placeholder("acc_in")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        copied_acc = graph.call_function(_new_var, args=(acc_in,))
        dot = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(copied_acc, dot))
        add.meta["stack_trace"] = (
            '  File "/tmp/test.py", line 1, in kernel\n    acc += hl.dot(lhs, rhs)\n'
        )
        graph.output(add)
        self.assertTrue(
            _cute_dot_outer_accumulates_result(
                SimpleNamespace(fx_node=dot),
                is_acc_none=True,
            )
        )

        non_acc_graph = Graph()
        tmp_in = non_acc_graph.placeholder("tmp_in")
        lhs = non_acc_graph.placeholder("lhs")
        rhs = non_acc_graph.placeholder("rhs")
        copied_tmp = non_acc_graph.call_function(_new_var, args=(tmp_in,))
        dot = non_acc_graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        add = non_acc_graph.call_function(
            torch.ops.aten.add.Tensor, args=(dot, copied_tmp)
        )
        add.meta["stack_trace"] = (
            '  File "/tmp/test.py", line 2, in kernel\n'
            "    out = hl.dot(lhs, rhs) + tmp\n"
        )
        non_acc_graph.output(add)
        self.assertFalse(
            _cute_dot_outer_accumulates_result(
                SimpleNamespace(fx_node=dot),
                is_acc_none=True,
            )
        )

    def test_cute_dot_outer_accumulates_result_accepts_simple_assign_pattern(
        self,
    ) -> None:
        graph = Graph()
        acc_in = graph.placeholder("acc_in")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        copied_acc = graph.call_function(_new_var, args=(acc_in,))
        dot = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(copied_acc, dot))
        add.meta["stack_trace"] = (
            '  File "/tmp/test.py", line 1, in kernel\n'
            "    acc = acc + hl.dot(lhs, rhs)\n"
        )
        graph.output(add)
        self.assertTrue(
            _cute_dot_outer_accumulates_result(
                SimpleNamespace(fx_node=dot),
                is_acc_none=True,
            )
        )

    def test_cute_static_k_invariant_extent_recovers_specialized_full_matmul(self):
        graph = Graph()
        lhs = graph.call_function(
            hl.full,
            args=([32, 512], 1.0 / 512, torch.float32, None),
        )
        rhs = graph.call_function(
            hl.full,
            args=([512, 512], 1.0, torch.float32, None),
        )
        graph.output((lhs, rhs))
        lhs.meta["val"] = torch.empty(32, 512)
        rhs.meta["val"] = torch.empty(512, 512)

        self.assertEqual(cute_static_k_invariant_extent(lhs, rhs), 512)

    def test_cute_static_k_invariant_extent_rejects_nonuniform_inputs(self):
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.call_function(
            hl.full,
            args=([512, 512], 1.0, torch.float32, None),
        )
        graph.output((lhs, rhs))
        lhs.meta["val"] = torch.empty(32, 512)
        rhs.meta["val"] = torch.empty(512, 512)

        self.assertIsNone(cute_static_k_invariant_extent(lhs, rhs))

    def test_cute_static_k_invariant_extent_rejects_masked_inputs(self):
        graph = Graph()
        lhs_base = graph.call_function(
            hl.full,
            args=([32, 512], 1.0 / 512, torch.float32, None),
        )
        lhs = graph.call_function(_mask_to, args=(lhs_base, 0.0))
        rhs = graph.call_function(
            hl.full,
            args=([512, 512], 1.0, torch.float32, None),
        )
        graph.output((lhs, rhs))
        lhs.meta["val"] = torch.empty(32, 512)
        rhs.meta["val"] = torch.empty(512, 512)

        self.assertIsNone(cute_static_k_invariant_extent(lhs, rhs))

    def test_cute_scalar_matmul_fallback_rejects_thread_carried_n(self):
        cg = SimpleNamespace(
            current_grid_state=SimpleNamespace(
                block_ids=[0],
                thread_axis_sizes={0: 32, 1: 4},
            )
        )
        lhs = torch.empty(32, 128, dtype=torch.bfloat16)
        rhs = torch.empty(128, 128, dtype=torch.bfloat16)
        out = torch.empty(32, 128, dtype=torch.bfloat16)
        env = SimpleNamespace(resolve_block_id=lambda size: None)

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertFalse(
                cute_supports_scalar_matmul_fallback(
                    cg,
                    lhs,
                    rhs,
                    out,
                    k_block_id=None,
                )
            )

    def test_cute_resolve_active_block_id_ignores_inactive_size_match(self):
        cg = SimpleNamespace(
            current_grid_state=SimpleNamespace(block_ids=[3]),
            active_device_loops={},
        )
        env = _fake_env({128: 7, 32: 3})

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertIsNone(cute_resolve_active_block_id(cg, 128))
            self.assertEqual(cute_resolve_active_block_id(cg, 32), 3)

    def test_cute_resolve_active_matmul_k_block_id_rejects_n_alias(self):
        cg = SimpleNamespace(
            current_grid_state=SimpleNamespace(block_ids=[7]),
            active_device_loops={},
        )
        env = _fake_env({128: 7})

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertIsNone(cute_resolve_active_matmul_k_block_id(cg, 128, 128, 128))

    def test_cute_resolve_active_block_id_rejects_ambiguous_aliases(self):
        cg = SimpleNamespace(
            current_grid_state=SimpleNamespace(block_ids=[3, 5]),
            active_device_loops={},
        )
        env = _fake_env({64: 7})
        env.canonical_block_id = lambda block_id: {3: 11, 5: 11, 7: 11}.get(
            block_id, block_id
        )

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertIsNone(cute_resolve_active_block_id(cg, 64))

    def test_resolve_block_id_requires_symbolic_origin(self) -> None:
        fake_env = SimpleNamespace(
            specialize_expr=lambda expr: expr,
            get_block_id=lambda size: None,
        )

        self.assertIsNone(CompileEnvironment.resolve_block_id(fake_env, 128))

    def test_resolve_block_id_ignores_specialized_derived_constant(self) -> None:
        u0 = sympy.Symbol("u0")

        fake_env = SimpleNamespace(
            specialize_expr=lambda expr: (
                sympy.Integer(16)
                if sympy.simplify(expr - u0) == 0
                else sympy.Integer(17)
                if sympy.simplify(expr - (u0 + 1)) == 0
                else expr
            ),
            get_block_id=lambda size: 0 if size == u0 else None,
            canonical_block_id=lambda block_id: block_id,
        )

        self.assertEqual(CompileEnvironment.resolve_block_id(fake_env, u0), 0)
        self.assertIsNone(CompileEnvironment.resolve_block_id(fake_env, u0 + 1))

    def test_resolve_block_id_accepts_tile_origin_subclasses(self) -> None:
        sym = sympy.Symbol("tile_begin_0")
        fake_env = SimpleNamespace(
            specialize_expr=lambda expr: expr,
            get_block_id=lambda size: CompileEnvironment.get_block_id(fake_env, size),
        )
        fake_host_fn = SimpleNamespace(
            expr_to_origin={sym: SimpleNamespace(origin=TileBeginOrigin(block_id=3))}
        )

        with patch.object(HostFunction, "current", return_value=fake_host_fn):
            self.assertEqual(CompileEnvironment.resolve_block_id(fake_env, sym), 3)

    def test_persistent_cute_reduction_marks_synthetic_lane_loop_block(self) -> None:
        fake_env = SimpleNamespace(
            backend=SimpleNamespace(name="cute"),
            block_sizes={0: SimpleNamespace(numel=sympy.Integer(1024))},
            index_type=lambda: "cutlass.Int32",
        )
        current_grid = DeviceGridState(
            strategy=_FakeLoopStrategy([0]),
            block_id_to_info={},
        )
        state = SimpleNamespace(
            codegen=SimpleNamespace(
                current_grid_state=current_grid,
                host_statements=[],
                push_active_loops=lambda loop_state: None,
            ),
            device_function=SimpleNamespace(
                constexpr_arg=lambda name: False,
                deferred_rdim_defs=[],
            ),
            add_statement=lambda stmt: None,
        )
        fake_strategy = SimpleNamespace(
            block_index=0,
            _mask_var=None,
            _thread_count=256,
            _synthetic_cute_lane_var="synthetic_lane_0",
            _synthetic_cute_lane_extent=4,
            block_size_var=lambda block_idx: "_RDIM_SIZE_0",
            index_var=lambda block_idx: "indices_0",
            _get_thread_axis=lambda: 0,
            _index_init_expr=lambda block_size_var, dtype, block_idx: "base_idx",
            fn=SimpleNamespace(sympy_expr=lambda expr: str(expr)),
        )

        with patch.object(CompileEnvironment, "current", return_value=fake_env):
            PersistentReductionStrategy.codegen_preamble(fake_strategy, state)

        self.assertIn(0, current_grid.lane_loop_blocks)
        self.assertIn(("synthetic_lane_0", 4), current_grid.lane_loops)

    def test_baddbmm_packed_affine_lhs_load_fuses_on_cute(self) -> None:
        graph = Graph()
        block_size = graph.placeholder("block_size")
        batch_idx = graph.placeholder("batch_idx")
        row_idx = graph.placeholder("row_idx")
        begin = graph.call_function(hl.tile_begin, args=(block_size,))
        start = graph.call_function(operator.mul, args=(begin, 2))
        length = graph.call_function(operator.mul, args=(block_size, 2))
        iota = graph.call_function(
            torch.ops.prims.iota.default,
            args=(length,),
            kwargs={
                "start": start,
                "step": 1,
                "dtype": torch.int32,
                "device": torch.device("cuda"),
                "requires_grad": False,
            },
        )
        lhs = graph.call_function(
            hl.load,
            args=(graph.placeholder("A"), [batch_idx, row_idx, iota], None, None),
        )
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([packed, packed], 2),
        )
        rhs = graph.call_function(
            torch.ops.aten.view.default,
            args=(stack, [2, length, 8]),
        )
        acc = graph.placeholder("acc")
        graph.call_function(torch.ops.aten.baddbmm.default, args=(acc, lhs, rhs))
        graph.output(rhs)

        packed.meta["val"] = torch.empty(2, 4, 8)
        rhs.meta["val"] = torch.empty(2, 8, 8)

        loop_strategy = _FakeMaskedLoopStrategy([0])
        codegen = SimpleNamespace(
            active_device_loops={0: [SimpleNamespace(strategy=loop_strategy)]},
            current_grid_state=None,
            codegen_graphs=[
                ForLoopGraphInfo(graph_id=0, graph=graph, node_args=[], block_ids=[0])
            ],
        )
        state = SimpleNamespace(
            fx_node=lhs,
            codegen=codegen,
            device_function=SimpleNamespace(
                tensor_arg=lambda tensor: SimpleNamespace(name="A")
            ),
        )

        env = SimpleNamespace(
            backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Float32"),
            block_sizes={0: _FakeBlockSize(4)},
            get_block_id=lambda size: None,
            known_equal=operator.eq,
            resolve_block_id=lambda size: None,
        )

        with patch.object(CompileEnvironment, "current", return_value=env):
            result = _maybe_codegen_cute_packed_affine_lhs_load(
                state,
                torch.empty(2, 8, 8),
                [0, 0, torch.empty(8, dtype=torch.int32)],
                None,
            )
        self.assertIsInstance(result, CutePackedAffineLoad)

    def test_codegen_iota_cute_recovers_active_axis_from_matching_block_size(self):
        graph = Graph()
        block_size = graph.placeholder("block_size")
        iota = graph.call_function(
            torch.ops.prims.iota.default,
            args=(block_size,),
            kwargs={
                "start": 0,
                "step": 1,
                "dtype": torch.int32,
                "device": torch.device("cuda"),
                "requires_grad": False,
            },
        )
        graph.output(iota)
        iota.meta["val"] = torch.empty(64, dtype=torch.int32)

        cg = _FakeGenerateAST({1})
        strategy = SimpleNamespace(index_var=lambda block_id: f"idx_{block_id}")
        cg.codegen_graphs = [
            ForLoopGraphInfo(graph_id=0, graph=graph, node_args=[], block_ids=[1])
        ]
        cg.active_device_loops = {1: [SimpleNamespace(strategy=strategy)]}
        cg.current_grid_state = None
        ctx = SimpleNamespace(
            cg=cg,
            to_ast=lambda value: ast.Constant(value=value),
        )
        env = SimpleNamespace(
            index_dtype=torch.int32,
            backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Int32"),
            block_sizes={
                0: _FakeBlockSize(64),
                1: _FakeBlockSize(64),
            },
            get_block_id=lambda size: 1 if int(size) == 64 else None,
            resolve_block_id=lambda size: 0 if size is block_size else None,
            resolve_codegen_block_id=lambda block_id, cg, graph=None: (
                1 if block_id == 0 else block_id
            ),
            size_hint=lambda size: 64 if size is block_size else 1,
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            result = codegen_iota_cute(ctx, iota)

        self.assertEqual(ast.unparse(result), "indices_1 - offset_1")

    def test_codegen_iota_cute_shared_atomic_user_does_not_collapse_to_zero(self):
        graph = Graph()
        length = graph.placeholder("length")
        out = graph.placeholder("out")
        val = graph.placeholder("val")
        iota = graph.call_function(
            torch.ops.prims.iota.default,
            args=(length,),
            kwargs={
                "start": 0,
                "step": 1,
                "dtype": torch.int32,
                "device": torch.device("cuda"),
                "requires_grad": False,
            },
        )
        graph.call_function(hl.atomic_add, args=(out, [iota], val))
        ordinary_use = graph.call_function(operator.add, args=(iota, 1))
        graph.output(ordinary_use)
        iota.meta["val"] = torch.empty(64, dtype=torch.int32)

        cg = _FakeGenerateAST(set())
        cg.active_device_loops = {}
        cg.current_grid_state = None
        cg.codegen_graphs = []
        ctx = SimpleNamespace(
            cg=cg,
            to_ast=lambda value: ast.Constant(value=value),
        )
        env = SimpleNamespace(
            index_dtype=torch.int32,
            backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Int32"),
            block_sizes={},
            get_block_id=lambda size: None,
            resolve_block_id=lambda size: None,
            resolve_codegen_block_id=lambda block_id, cg, graph=None: block_id,
            size_hint=lambda size: 64 if size is length else 1,
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
            patch(
                "helion._compiler.cute.cute_reshape._get_dim_local_coord",
                return_value="cutlass.Int32(0)",
            ),
            self.assertRaises(exc.BackendUnsupported),
        ):
            codegen_iota_cute(ctx, iota)

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

    def test_mask_to_cute_casts_then_branch_to_tensor_dtype(self) -> None:
        state = SimpleNamespace(
            proxy_arg=lambda index: (
                torch.empty(16, 8, dtype=torch.float16) if index == 0 else 0
            ),
            ast_arg=lambda index: (
                expr_from_string("load + 1") if index == 0 else expr_from_string("0")
            ),
            codegen=SimpleNamespace(mask_var=lambda block_id: f"mask_{block_id}"),
            tile_strategy=SimpleNamespace(expand_str=lambda sizes, dim: ""),
        )
        env = SimpleNamespace(
            backend=CuteBackend(),
            resolve_block_id=lambda size: {16: 0, 8: 1}.get(int(size)),
        )

        with patch.object(CompileEnvironment, "current", return_value=env):
            result = _mask_to._codegen["cute"](state)

        self.assertEqual(
            ast.unparse(result),
            "cutlass.Float16(load + 1) if mask_0 and mask_1 else cutlass.Float16(0)",
        )

    def test_lane_loop_store_permute_masked_load_uses_materialization(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        mask = graph.placeholder("mask")
        load_node = graph.call_function(
            load,
            args=(inp, [slice(None), slice(None)], mask, ""),
        )
        permute = graph.call_function(
            torch.ops.aten.permute.default,
            args=(load_node, [1, 0]),
        )
        inp.meta["val"] = torch.empty(2, 2)
        mask.meta["val"] = torch.empty(2, 2, dtype=torch.bool)
        load_node.meta["val"] = torch.empty(2, 2)
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
                return_value=(load_node, [1, 0]),
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
