"""CPU semantic, admission, and codegen coverage for tiled FP32 accumulation."""

from __future__ import annotations

import ast
from contextlib import ExitStack
from contextlib import contextmanager
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.aot_example import matmul_custom_key
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.cute_mma import analyze_cute_mma_node
from helion._compiler.cute.fuse_mm_accumulation import fuse_mm_accumulation
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops
from helion.language.matmul_ops import MATMUL_DIM_BLOCK_IDS_META
from helion.language.matmul_ops import MATMUL_FACT_ID_META

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _cpu_target() -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(patch_cute_mma_support())
        stack.enter_context(_mock_cuda_unavailable())
        stack.enter_context(
            patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU test"))
        )
        for name, value in (
            ("helion.runtime.kernel.target_device_capability", (10, 0)),
            ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
            ("helion.runtime.get_num_sm", 148),
            ("helion._compat._is_hip", False),
            (
                (
                    "helion._compiler.cute.tcgen05_config."
                    "CuteTcgen05Config.per_cta_smem_budget_bytes"
                ),
                232448,
            ),
        ):
            stack.enter_context(patch(name, return_value=value))
        yield


def _program(
    dtype: torch.dtype, *, commute: bool = False, carry_copy: bool = False
) -> tuple[torch.fx.GraphModule, ForLoopGraphInfo]:
    graph = torch.fx.Graph()
    accumulator = graph.placeholder("accumulator")
    accumulator.meta["val"] = torch.empty((3, 5), dtype=torch.float32)
    if carry_copy:
        accumulator = graph.call_function(_tracing_ops._new_var, (accumulator,))
        accumulator.meta["val"] = torch.empty((3, 5), dtype=torch.float32)
    module = torch.nn.Module()
    generator = torch.Generator().manual_seed(830)
    operands = []
    for name, shape in (("lhs", (3, 7)), ("rhs", (7, 5))):
        value = torch.randn(shape, generator=generator, dtype=dtype)
        module.register_buffer(name, value)
        operand = graph.get_attr(name)
        operand.meta["val"] = value
        if dtype != torch.float32:
            operand = graph.call_function(
                torch.ops.prims.convert_element_type.default,
                (operand, torch.float32),
            )
            operand.meta["val"] = value.float()
        operands.append(operand)
    product = graph.call_function(torch.ops.aten.mm.default, tuple(operands))
    product.meta["val"] = torch.empty((3, 5), dtype=torch.float32)
    product.meta[MATMUL_FACT_ID_META] = 0
    product.meta[MATMUL_DIM_BLOCK_IDS_META] = (0, 1)
    addition = graph.call_function(
        torch.ops.aten.add.Tensor,
        (product, accumulator) if commute else (accumulator, product),
    )
    addition.meta = {"val": product.meta["val"], "location": "the original carry"}
    graph.output([addition])
    return torch.fx.GraphModule(module, graph), ForLoopGraphInfo(
        graph_id=0, graph=graph, node_args=[accumulator], block_ids=[2]
    )


def _fuse(info: ForLoopGraphInfo, *, backend: str = "cute") -> int:
    env = SimpleNamespace(
        backend_name=backend,
        settings=SimpleNamespace(static_shapes=True),
        known_equal=operator.eq,
        canonical_block_id=lambda block_id: block_id,
        resolve_block_id=lambda size: {3: 0, 5: 1, 7: 2}.get(size),
    )
    with patch.object(CompileEnvironment, "current", return_value=env):
        return fuse_mm_accumulation(info)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("commute", [False, True])
def test_fp32_contract_and_input_casts(dtype: torch.dtype, commute: bool) -> None:
    module, info = _program(dtype, commute=commute)
    acc = torch.randn((3, 5), generator=torch.Generator().manual_seed(941))
    expected = module(acc)
    product = next(
        iter(
            info.graph.find_nodes(op="call_function", target=torch.ops.aten.mm.default)
        )
    )
    operands = product.args
    assert _fuse(info) == 1
    module.recompile()
    torch.testing.assert_close(module(acc), expected, rtol=2e-6, atol=2e-6)
    fused = next(
        iter(
            info.graph.find_nodes(
                op="call_function", target=torch.ops.aten.addmm.default
            )
        )
    )
    assert fused.args[1:] == operands
    assert fused.meta["val"].dtype is torch.float32
    assert fused.meta["location"] == "the original carry"
    assert fused.meta[MATMUL_FACT_ID_META] == 0
    assert fused.meta[MATMUL_DIM_BLOCK_IDS_META] == (0, 1)
    assert module.lhs.dtype is dtype and module.rhs.dtype is dtype
    assert _fuse(info) == 0


def test_new_var_carry_is_recognized() -> None:
    module, info = _program(torch.bfloat16, carry_copy=True)
    assert _fuse(info) == 1
    fused = next(
        iter(
            info.graph.find_nodes(
                op="call_function", target=torch.ops.aten.addmm.default
            )
        )
    )
    assert fused.args[0].target is _tracing_ops._new_var


@pytest.mark.parametrize("commute", [False, True])
def test_batched_product_fuses_with_an_independent_index_capture(commute: bool) -> None:
    graph = torch.fx.Graph()
    rows = graph.placeholder("rows")
    rows.meta["val"] = torch.tensor([1, 0])
    accumulator = graph.placeholder("accumulator")
    accumulator.meta["val"] = torch.empty((2, 3, 5))
    module = torch.nn.Module()
    module.register_buffer("left", torch.randn((2, 3, 7)))
    module.register_buffer("right", torch.randn((2, 7, 5)))
    lhs = graph.get_attr("left")
    lhs.meta["val"] = module.left
    rhs = graph.get_attr("right")
    rhs.meta["val"] = module.right
    gathered = graph.call_function(torch.ops.aten.index_select.default, (lhs, 0, rows))
    gathered.meta["val"] = module.left
    product = graph.call_function(torch.ops.aten.bmm.default, (gathered, rhs))
    product.meta["val"] = accumulator.meta["val"]
    addition = graph.call_function(
        torch.ops.aten.add.Tensor,
        (product, accumulator) if commute else (accumulator, product),
    )
    addition.meta["val"] = accumulator.meta["val"]
    graph.output([addition])
    info = ForLoopGraphInfo(0, graph, [rows, accumulator], [2])
    program = torch.fx.GraphModule(module, graph)
    initial = torch.randn((2, 3, 5))
    expected = program(rows.meta["val"], initial)
    assert _fuse(info) == 1
    program.recompile()
    torch.testing.assert_close(
        program(rows.meta["val"], initial), expected, rtol=2e-6, atol=2e-6
    )
    assert len(graph.find_nodes(op="placeholder")) == 2
    assert (
        len(graph.find_nodes(op="call_function", target=torch.ops.aten.baddbmm.default))
        == 1
    )


@pytest.mark.parametrize(
    "change",
    [
        "alpha",
        "multiple_uses",
        "effect",
        "broadcast",
        "half_acc",
        "wrong_k",
        "extra_carry",
        "epilogue",
        "triton",
    ],
)
def test_refuses_noncanonical_accumulation(change: str) -> None:
    module, info = _program(torch.float32)
    graph = info.graph
    product = next(
        iter(graph.find_nodes(op="call_function", target=torch.ops.aten.mm.default))
    )
    addition = next(
        iter(graph.find_nodes(op="call_function", target=torch.ops.aten.add.Tensor))
    )
    accumulator = addition.args[0]
    if change == "alpha":
        addition.kwargs = {"alpha": 2}
    elif change == "multiple_uses":
        with graph.inserting_before(addition):
            graph.call_function(torch.ops.aten.relu.default, (product,))
    elif change == "effect":
        with graph.inserting_before(addition):
            graph.call_function(memory_ops.store, (accumulator, [0, 0], accumulator))
    elif change == "broadcast":
        accumulator.meta["val"] = torch.empty((1, 5), dtype=torch.float32)
    elif change == "half_acc":
        accumulator.meta["val"] = torch.empty((3, 5), dtype=torch.float16)
    elif change == "wrong_k":
        info.block_ids = [0]
    elif change == "extra_carry":
        info.node_args.append(accumulator)
    elif change == "epilogue":
        output = next(iter(graph.find_nodes(op="output")))
        with graph.inserting_before(output):
            relu = graph.call_function(torch.ops.aten.relu.default, (addition,))
        output.args = ([relu],)
    original = str(graph)
    assert _fuse(info, backend="triton" if change == "triton" else "cute") == 0
    assert str(graph) == original


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(128, 128, 64), (73, 41, 78)])
@skipUnlessBackends(["cute"])
def test_aot_source_uses_half_mma_with_fp32_accumulator(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    m, n, k = shape
    kernel = helion.kernel(
        matmul_custom_key.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        dot_precision="ieee",
    )
    with _cpu_target():
        bound = kernel._bind_isolated(
            (torch.empty((m, k), dtype=dtype), torch.empty((k, n), dtype=dtype))
        )
        assert bound.env.config_spec.cute_tcgen05_search_enabled
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            candidates = [
                candidate
                for info in bound.host_function.device_ir.graphs
                for node in info.graph.nodes
                if (candidate := analyze_cute_mma_node(node)) is not None
            ]
            assert len(candidates) == 1
            candidate = candidates[0]
            assert candidate.operands.lhs.source_fake.dtype is dtype
            assert candidate.operands.rhs.source_fake.dtype is dtype
            assert candidate.lhs.meta["val"].dtype is torch.float32
            assert candidate.rhs.meta["val"].dtype is torch.float32
        code = bound.to_code(helion.Config(block_sizes=[64, 32, 32]))
    ast.parse(code)
    assert "cute.gemm(" in code
    assert "make_trivial_tiled_mma(" in code
    assert "cutlass.TFloat32" not in code


@skipUnlessBackends(["cute"])
def test_true_fp32_inputs_keep_ieee_precision() -> None:
    kernel = helion.kernel(
        matmul_custom_key.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        dot_precision="ieee",
    )
    with _cpu_target():
        bound = kernel._bind_isolated((torch.empty((64, 64)), torch.empty((64, 32))))
        assert not bound.env.config_spec.cute_tcgen05_search_enabled
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            for info in bound.host_function.device_ir.graphs:
                for node in info.graph.nodes:
                    candidate = analyze_cute_mma_node(node)
                    if candidate is not None:
                        assert candidate.operands.lhs.source_fake.dtype is torch.float32
                        assert candidate.operands.rhs.source_fake.dtype is torch.float32
        code = bound.to_code(helion.Config(block_sizes=[64, 32, 32]))
    ast.parse(code)
    assert "cutlass.TFloat32" not in code
    assert "cutlass.Float16" not in code
    assert "cutlass.BFloat16" not in code
    assert "make_trivial_tiled_mma(" not in code


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mismatch_first", [False, True])
@skipUnlessBackends(["cute"])
def test_dynamic_inputs_key_both_k_sizes_and_output_extents(
    dtype: torch.dtype, mismatch_first: bool
) -> None:
    kernel = helion.kernel(
        matmul_custom_key.fn,
        key=matmul_custom_key._key_fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        dot_precision="ieee",
    )
    shapes = [(128, 64, 64, 128), (128, 64, 80, 128)]
    if mismatch_first:
        shapes.reverse()
    shapes.extend([(128, 80, 80, 128), (73, 64, 64, 41), (128, 64, 48, 128)])
    bounds = []
    with _cpu_target():
        for m, left_k, right_k, n in shapes:
            args = (
                torch.empty((m, left_k), dtype=dtype),
                torch.empty((right_k, n), dtype=dtype),
            )
            bound = kernel.bind(args)
            assert all(bound is not previous for previous in bounds)
            assert kernel.bind(tuple(torch.empty_like(arg) for arg in args)) is bound
            bounds.append(bound)
            assert bound.env.specialized_vars
            assert bound.env.config_spec.cute_tcgen05_search_enabled == (
                left_k == right_k
            )
            code = bound.to_code(helion.Config(block_sizes=[64, 32, 32]))
            assert ("cute.gemm(" in code) == (left_k == right_k)
            ast.parse(code)


@pytest.mark.parametrize("change", ["cast_chain", "masked_load", "computed_operand"])
@skipUnlessBackends(["cute"])
def test_mma_proof_keeps_cast_and_mask_semantics(change: str) -> None:
    kernel = helion.kernel(
        matmul_custom_key.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with _cpu_target():
        bound = kernel._bind_isolated(
            (
                torch.empty((64, 64), dtype=torch.bfloat16),
                torch.empty((64, 32), dtype=torch.bfloat16),
            )
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            candidates = [
                (node, candidate)
                for info in bound.host_function.device_ir.graphs
                for node in info.graph.nodes
                if (candidate := analyze_cute_mma_node(node)) is not None
            ]
            assert len(candidates) == 1
            node, candidate = candidates[0]
            conversion = candidate.lhs
            load = candidate.operands.lhs.load
            graph = node.graph
            if change == "masked_load":
                load.args = (*load.args[:2], False, None)
            else:
                with graph.inserting_before(conversion):
                    if change == "cast_chain":
                        inner = graph.call_function(
                            torch.ops.prims.convert_element_type.default,
                            (load, torch.float16),
                        )
                        inner.meta["val"] = load.meta["val"].to(torch.float16)
                    else:
                        inner = graph.call_function(
                            torch.ops.aten.mul.Tensor, (load, 0.5)
                        )
                        inner.meta["val"] = load.meta["val"]
                conversion.args = (inner, torch.float32)
            assert analyze_cute_mma_node(node) is None


@skipUnlessBackends(["cute"])
def test_computed_fp32_inputs_are_not_specialized_or_narrowed() -> None:
    @helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
    def computed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        n = b.size(1)
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                left = a[tile_m, tile_k].float() * 1.25
                acc += left @ b[tile_k, tile_n].float()
            out[tile_m, tile_n] = acc.to(out.dtype)
        return out

    with _cpu_target():
        bound = computed._bind_isolated(
            (
                torch.empty((64, 64), dtype=torch.bfloat16),
                torch.empty((64, 32), dtype=torch.bfloat16),
            )
        )
        assert not bound.env.specialized_vars
        assert not bound.env.config_spec.cute_tcgen05_search_enabled
        code = bound.to_code(helion.Config(block_sizes=[64, 32, 32]))
    assert "cute.gemm(" not in code
    assert "1.25" in code


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_zero_k_keeps_zero_accumulator(dtype: torch.dtype) -> None:
    kernel = helion.kernel(
        matmul_custom_key.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
    )
    with _cpu_target():
        bound = kernel._bind_isolated(
            (torch.empty((64, 0), dtype=dtype), torch.empty((0, 32), dtype=dtype))
        )
        assert not bound.env.config_spec.cute_tcgen05_search_enabled
        code = bound.to_code(helion.Config(block_sizes=[64, 32, 32]))
    ast.parse(code)
    assert "cute.gemm(" not in code
    assert "cutlass.Float32(0.0)" in code


@skipUnlessBackends(["cute"])
def test_role_lifecycle_owns_absorbed_cast_scaffolding() -> None:
    kernel = helion.kernel(
        matmul_custom_key.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        dot_precision="ieee",
    )
    config = helion.Config(
        block_sizes=[128, 256, 128],
        indexing=["tensor_descriptor"] * 3,
        l2_groupings=[1],
        loop_orders=[[0, 1]],
        num_stages=2,
        num_warps=4,
        pid_type="flat",
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=1,
        tcgen05_c_stages=2,
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_num_epi_warps=4,
        tcgen05_strategy="pure_matmul_role_lifecycle",
    )
    with _cpu_target():
        bound = kernel._bind_isolated(
            (
                torch.empty((128, 384), dtype=torch.bfloat16),
                torch.empty((384, 256), dtype=torch.bfloat16),
            )
        )
        code = bound.to_code(config)
    ast.parse(code)
    assert "cute.gemm(" in code
    assert "cutlass.Float32(load" not in code
