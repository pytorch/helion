from __future__ import annotations

import ast
from operator import eq
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import sympy
import torch
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import Pointwise
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from helion import exc
from helion._compiler import inductor_lowering as lowering
from helion._compiler.cute.backend import CuteBackend

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.generate_ast import GenerateAST
    from helion._compiler.helper_function import CodegenInterface


class Codegen:
    def __init__(self):
        self.device_function = SimpleNamespace(
            _constexpr_args=set(),
            tile_strategy=SimpleNamespace(supports_index_rank_expansion=lambda: False),
        )
        self.lines = []

    def lift(self, value):
        if isinstance(value, ast.Name):
            return value
        name = f"value_{len(self.lines)}"
        self.lines.append(f"{name} = {ast.unparse(value)}")
        return ast.Name(id=name, ctx=ast.Load())


@pytest.fixture
def environment():
    env = SimpleNamespace(
        settings=SimpleNamespace(fast_math=False),
        backend=CuteBackend(),
        backend_name="cute",
        device=torch.device("cpu"),
        shape_env=ShapeEnv(),
        known_equal=eq,
        resolve_block_id=lambda size: None,
    )
    with patch.object(lowering.CompileEnvironment, "current", return_value=env):
        yield env


def pointwise(target=torch.ops.aten.add.Tensor, *, inner=None, shapes=((8,), (8,))):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    x.meta["val"] = torch.empty(shapes[0])
    y.meta["val"] = torch.empty(shapes[1])
    node = graph.call_function(target, (x, y))
    node.meta["val"] = torch.empty(shapes[0])
    if inner is None:

        def inner(indices):
            return ops.add(ops.load("x", indices[0]), ops.load("y", indices[0]))

    data = Pointwise(
        device=torch.device("cpu"),
        dtype=torch.float32,
        inner_fn=inner,
        ranges=[sympy.Integer(8)],
    )
    buffer = ComputedBuffer(
        name="result",
        layout=FixedLayout(torch.device("cpu"), torch.float32, [8]),
        data=data,
    )
    return lowering.PointwiseLowering(buffer, ["x", "y"]), node


def emit(cg, item, scope=None, *, inputs=("left", "right"), policy=None):
    op, node = item
    ctx = SimpleNamespace(cg=cg)
    with V.set_current_node(node):
        return op.codegen_from_input_asts(
            ctx,
            node,
            [ast.Name(id=name, ctx=ast.Load()) for name in inputs],
            policy=policy,
            graph_reuse=scope,
        )


def test_exact_fragments_fresh_inputs_ops_kernel_cse_and_current_node(environment):
    def sequence(reuse):
        cg = Codegen()
        scope = (
            lowering.PointwiseGraphReuse(cast("CodegenInterface", cg))
            if reuse
            else None
        )
        seen = []
        outputs = []
        for index in range(3):
            policy = lowering.PointwiseCodegenPolicy(sigmoid_tanh=bool(index % 2))

            def inner(indices):
                seen.append((V.graph, V.ops, V.kernel, V.kernel.cse, V.current_node))
                return ops.add(ops.load("x", indices[0]), ops.load("y", indices[0]))

            item = pointwise(inner=inner)
            result = emit(
                cg, item, scope, inputs=(f"x{index}", f"y{index}"), policy=policy
            )
            outputs.append(ast.dump(result))
            assert seen[-1][4] is item[1]
            handler = seen[-1][1]
            assert isinstance(handler, lowering.GenerateASTFromInductor)
            assert handler.policy is policy
        return cg.lines, outputs, seen

    old_lines, old_outputs, old = sequence(False)
    new_lines, new_outputs, new = sequence(True)
    assert (old_lines, old_outputs) == (new_lines, new_outputs)
    assert len({id(item[0]) for item in old}) == 3
    assert len({id(item[0]) for item in new}) == 1
    for column in (1, 2, 3):
        assert len({id(item[column]) for item in new}) == 3
    assert new[0][0].operations == []
    assert new[0][0].buffers == []
    assert not new[0][0].constants


@pytest.mark.parametrize(
    "change",
    [
        "cg",
        "device_function",
        "env",
        "backend",
        "backend_name",
        "shape_env",
        "device",
        "fast_math",
    ],
)
def test_owner_and_environment_mismatch_use_original_fresh_graph(environment, change):
    cg = Codegen()
    scope = lowering.PointwiseGraphReuse(cast("CodegenInterface", cg))
    with lowering._patched_inductor_config():
        original = scope.graph(cast("CodegenInterface", cg))
        if change == "cg":
            cg = Codegen()
        elif change == "device_function":
            cg.device_function = Codegen().device_function
        elif change == "env":
            with patch.object(
                lowering.CompileEnvironment,
                "current",
                return_value=SimpleNamespace(**vars(environment)),
            ):
                assert scope.graph(cast("CodegenInterface", cg)) is not original
            return
        elif change == "backend":
            environment.backend = CuteBackend()
        elif change == "backend_name":
            environment.backend_name = "triton"
        elif change == "shape_env":
            environment.shape_env = ShapeEnv()
        elif change == "device":
            environment.device = torch.device("cpu", 1)
        else:
            environment.settings.fast_math = True
        assert scope.graph(cast("CodegenInterface", cg)) is not original
        assert scope.graph(cast("CodegenInterface", cg)) is not original


@pytest.mark.parametrize(
    "target",
    [
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.rand_like.default,
        torch.ops.aten.sum.default,
        torch.add,
    ],
)
def test_mutable_random_nonpointwise_and_python_targets_keep_fresh_graph(
    environment, target
):
    cg = Codegen()
    scope = lowering.PointwiseGraphReuse(cast("CodegenInterface", cg))
    seen = []

    def inner(indices):
        seen.append(V.graph)
        V.graph.removed_buffers.add("simulated stateful path")
        return ops.load("x", indices[0])

    item = pointwise(target, inner=inner)
    emit(cg, item, scope)
    emit(cg, item, scope)
    assert seen[0] is not seen[1]
    assert scope._graph is None


def test_subclass_ir_and_wrong_current_node_are_not_admitted(environment):
    op, node = pointwise()
    with V.set_current_node(node):
        assert lowering.PointwiseGraphReuse.accepts(op, node)

        class CustomPointwise(Pointwise):
            pass

        op.buffer.data = CustomPointwise(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inner_fn=lambda i: "x",
            ranges=[8],
        )
        assert not lowering.PointwiseGraphReuse.accepts(op, node)
    op, node = pointwise()
    with V.set_current_node(pointwise()[1]):
        assert not lowering.PointwiseGraphReuse.accepts(op, node)


def test_broadcast_and_input_count_checks_still_run_on_warm_graph(environment):
    cg = Codegen()
    scope = lowering.PointwiseGraphReuse(cast("CodegenInterface", cg))
    emit(cg, pointwise(), scope)
    with pytest.raises(exc.ShapeMismatch):
        emit(cg, pointwise(shapes=((8,), (7,))), scope)
    with pytest.raises(AssertionError):
        emit(cg, pointwise(), scope, inputs=("only_one",))


def test_exception_restores_handlers_and_discards_graph(environment):
    cg = Codegen()
    scope = lowering.PointwiseGraphReuse(cast("CodegenInterface", cg))
    emit(cg, pointwise(), scope)
    original = scope._graph
    before = (V.graph, V.ops, V.kernel)

    def fail(indices):
        raise ValueError("pointwise failed")

    with pytest.raises(ValueError, match="pointwise failed"):
        emit(cg, pointwise(inner=fail), scope)
    assert (V.graph, V.ops, V.kernel) == before
    assert scope._graph is None
    emit(cg, pointwise(), scope)
    assert scope._graph is not original


def test_each_operand_domain_call_owns_a_distinct_scope(environment):
    from helion._compiler.cute import chained_matmul as chain

    cg = Codegen()
    scopes = []

    class Domain:
        pointwise_graph: lowering.PointwiseGraphReuse

        def __init__(self, *args):
            pass

        def value(self, node, coordinates):
            scopes.append(self.pointwise_graph)

    with (
        patch.object(chain, "_DomainExpression", Domain),
        patch.object(chain, "_domain_bounds", return_value=["r < 8"]),
    ):
        assert chain._operand_domain(
            cast("GenerateAST", cg),
            pointwise()[1],
            ("r",),
            cast("ChainedMatmulPlan", object()),
        ) == ["r < 8"]
        assert chain._operand_domain(
            cast("GenerateAST", cg),
            pointwise()[1],
            ("r",),
            cast("ChainedMatmulPlan", object()),
        ) == ["r < 8"]
    assert scopes[0] is not scopes[1]
