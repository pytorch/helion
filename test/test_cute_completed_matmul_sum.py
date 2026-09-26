"""Check completion provenance on the original full-slice kernel's typed IR."""

from __future__ import annotations

import ast
from contextlib import ExitStack
from contextlib import contextmanager
import copy
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch.fx.node import Node

from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.cute import completed_matmul_sum as completion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.generate_ast import GenerateAST
from helion._hardware import HardwareInfo
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.kernel import Kernel

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.aten_lowering import LoweringContext


@dataclass
class _Capture:
    ctx: LoweringContext
    node: Node
    value: ast.Name
    plan: completion.CompletedMatmulSum
    source: str = ""


@contextmanager
def _attribute(obj: object, key: str, value: object) -> Iterator[None]:
    original = getattr(obj, key)
    setattr(obj, key, value)
    try:
        yield
    finally:
        setattr(obj, key, original)


@contextmanager
def _value_escape(node: Node) -> Iterator[None]:
    with node.graph.inserting_after(node):
        extra = node.graph.call_function(torch.ops.aten.clone.default, (node,))
    try:
        yield
    finally:
        node.graph.erase_node(extra)


@pytest.fixture(scope="module")
def captured() -> Iterator[_Capture]:
    original = Path(__file__).with_name("test_indexing.py")
    tree = ast.parse(original.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TestIndexing"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_full_slice_in_reduction_loop"
    )
    function = next(node for node in method.body if isinstance(node, ast.FunctionDef))
    namespace: dict[str, object] = {
        "torch": torch,
        "helion": helion,
        "hl": hl,
        "__name__": __name__,
    }
    # Retain the original function and source locations, including its allocating
    # caller and sum. Only the CPU target metadata is held for source generation.
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(original), "exec"),
        namespace,
    )
    original_kernel = namespace["kernel"]
    assert isinstance(original_kernel, Kernel)
    kernel = helion.kernel(
        original_kernel.fn,
        backend="cute",
        static_shapes=True,
        cute_full_slice_matmul_tiling=False,
        cute_segmented_matmul_tiling=False,
        cute_flatten_nested_reductions=False,
    )
    consume = completion.completed_matmul_sum_input
    observations: list[_Capture] = []

    def observe(ctx: LoweringContext, node: Node) -> ast.AST | None:
        value = consume(ctx, node)
        assert isinstance(value, ast.Name)
        assert isinstance(ctx.cg, GenerateAST)
        snapshot = copy.copy(ctx)
        snapshot.env = dict(ctx.env)
        observations.append(
            _Capture(
                snapshot,
                node,
                value,
                ctx.cg.device_function.cute_state.completed_matmul_sums[node],
            )
        )
        return value

    with ExitStack() as stack:
        stack.enter_context(_mock_cuda_unavailable())
        stack.enter_context(
            patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden"))
        )
        stack.enter_context(patch_cute_mma_support())
        for name in ("per_cta_smem_budget_bytes", "per_cta_smem_capacity_bytes"):
            stack.enter_context(
                patch.object(CuteTcgen05Config, name, return_value=232448)
            )
        for name, value in (
            ("helion.runtime.kernel.target_device_capability", (10, 0)),
            ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
            ("helion.runtime.get_num_sm", 148),
            ("helion._compat._is_hip", False),
            (
                "helion._hardware.get_hardware_info",
                HardwareInfo(
                    device_kind="cuda",
                    hardware_name="NVIDIA B200",
                    runtime_version="13.0",
                    compute_capability="sm100",
                ),
            ),
        ):
            stack.enter_context(patch(name, return_value=value))
        bound = kernel._bind_isolated((torch.empty(16, 16, 16, dtype=torch.float32),))
        with patch.object(completion, "completed_matmul_sum_input", observe):
            source = bound.to_code(helion.Config(block_sizes=[16, 16]))
        assert len(observations) == 1
        result = observations[0]
        result.source = source
        with bound.env:
            assert (
                completion.completed_matmul_sum_input(result.ctx, result.node)
                is result.value
            )
            yield result


def _consume(captured: _Capture) -> ast.AST | None:
    return completion.completed_matmul_sum_input(captured.ctx, captured.node)


def test_original_loop_consumes_the_sum_emitted_by_this_codegen(
    captured: _Capture,
) -> None:
    assert _consume(captured) is captured.value
    path = captured.plan.path
    assert path.n_extent == 16
    assert ast.unparse(captured.plan.expression) == "attn_copy_0 + dot_n_acc"
    assert any(
        user.target == torch.ops.aten.sym_size.int for user in path.placeholder.users
    )
    assert "_helion_lane_reduce" not in captured.source


@pytest.mark.parametrize("axis", [[0], [1], [-2], [-1, -2], []])
def test_other_reduction_axes_are_not_marked_complete(
    captured: _Capture, axis: list[int]
) -> None:
    with _attribute(captured.node, "args", (captured.node.args[0], axis)):
        assert _consume(captured) is None
    assert _consume(captured) is captured.value


@pytest.mark.parametrize("initial", [1.0, -0.0, True, float("nan"), float("inf")])
def test_initial_matrix_must_be_positive_fp32_zero(
    captured: _Capture, initial: object
) -> None:
    node = captured.plan.path.initial
    with _attribute(node, "args", (node.args[0], initial, *node.args[2:])):
        assert _consume(captured) is None
    assert _consume(captured) is captured.value


@pytest.mark.parametrize(
    "edge",
    [
        "placeholder",
        "accumulator_copy",
        "producer",
        "initial",
        "result_item",
        "phi",
        "masked",
    ],
)
def test_matrix_values_cannot_escape_the_closed_carry(
    captured: _Capture, edge: str
) -> None:
    node = getattr(captured.plan.path, edge)
    assert isinstance(node, Node)
    with _value_escape(node):
        assert _consume(captured) is None
    assert _consume(captured) is captured.value


def test_a_cast_before_the_sum_cannot_use_fp32_completion(captured: _Capture) -> None:
    parent = captured.node.args[0]
    assert isinstance(parent, Node)
    with captured.node.graph.inserting_before(captured.node):
        converted = captured.node.graph.call_function(
            torch.ops.aten._to_copy.default, (parent,), {"dtype": torch.float16}
        )
    try:
        with _attribute(captured.node, "args", (converted, captured.node.args[1])):
            assert _consume(captured) is None
    finally:
        captured.node.graph.erase_node(converted)
    assert _consume(captured) is captured.value


@pytest.mark.parametrize("edge", ["producer", "placeholder", "phi", "consumer"])
def test_untyped_edges_do_not_reuse_a_completed_record(
    captured: _Capture, edge: str
) -> None:
    node = getattr(captured.plan.path, edge)
    with patch.dict(node.meta, {"val": None}):
        assert _consume(captured) is None
    assert _consume(captured) is captured.value


def test_actual_current_loop_input_is_used_instead_of_stale_node_args(
    captured: _Capture,
) -> None:
    path = captured.plan.path
    # GraphInfo retains the pre-copy graph's operands. Corrupting those stale
    # operands must not change the proof, but changing the current call must.
    with _attribute(path.body, "node_args", []):
        assert _consume(captured) is captured.value
        with _attribute(path.loop_call, "args", (*path.loop_call.args[:3], [])):
            assert _consume(captured) is None
    assert _consume(captured) is captured.value


@pytest.mark.parametrize("edge", ["result_item", "phi"])
def test_output_slot_and_phi_edges_are_exact(captured: _Capture, edge: str) -> None:
    path = captured.plan.path
    node = getattr(path, edge)
    wrong = (
        (path.loop_call, 1)
        if edge == "result_item"
        else (path.result_item, path.initial)
    )
    with _attribute(node, "args", wrong):
        assert _consume(captured) is None
    assert _consume(captured) is captured.value


@pytest.mark.parametrize("mask", [-0.0, 1.0, True])
def test_nonidentity_mask_is_not_removed(captured: _Capture, mask: object) -> None:
    node = captured.plan.path.masked
    assert node is not None
    with _attribute(node, "args", (node.args[0], mask)):
        assert _consume(captured) is None
    with patch.dict(
        captured.ctx.env, {node: ast.Name(id="different_bound_value", ctx=ast.Load())}
    ):
        assert _consume(captured) is None
    assert _consume(captured) is captured.value


def test_record_is_local_to_the_codegen_and_actual_expression(
    captured: _Capture,
) -> None:
    state = captured.ctx.cg.device_function.cute_state.completed_matmul_sums
    with patch.dict(state, clear=True):
        assert _consume(captured) is None
    with patch.dict(
        state,
        {
            captured.node: replace(
                captured.plan, codegen=copy.copy(captured.plan.codegen)
            )
        },
    ):
        assert _consume(captured) is None
    with patch.dict(
        captured.plan.path.producer.meta,
        {"codegen": ast.parse(ast.unparse(captured.plan.expression), mode="eval").body},
    ):
        assert _consume(captured) is None
    assert _consume(captured) is captured.value


def test_node_names_and_bound_variable_spelling_are_not_provenance(
    captured: _Capture,
) -> None:
    path = captured.plan.path
    assert path.masked is not None
    renamed = ast.Name(id="opaque_result", ctx=ast.Load())
    with ExitStack() as stack:
        for index, node in enumerate((path.producer, path.phi, path.consumer)):
            stack.enter_context(_attribute(node, "name", f"unrelated_{index}"))
        stack.enter_context(
            patch.dict(captured.ctx.env, {path.phi: renamed, path.masked: renamed})
        )
        assert _consume(captured) is renamed
    assert _consume(captured) is captured.value
