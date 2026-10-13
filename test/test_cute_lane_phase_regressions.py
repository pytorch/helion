"""CPU regressions for binary subtiles and isolated reduction-stage effects."""

from __future__ import annotations

import ast
import math
import operator
from types import SimpleNamespace
from unittest.mock import patch

from examples.jagged_softmax import jagged_softmax_kernel
import pytest
import sympy
import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_lane_loop_distribution import _matmul_with_bias

import helion
from helion import exc
from helion._compiler import epilogue_subtiling
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.tile_strategy import _lane_invariant_tail_after_consume
from helion.language.view_ops import split


@pytest.mark.parametrize("physical_write", [False, True])
def test_tail_order_uses_emitted_stage_carry_effects(physical_write: bool) -> None:
    statements: list[ast.AST] = list(ast.parse("old = carry\nnext_value = x + 1").body)
    stage = ast.parse(
        "next_value = x + 1" if physical_write else "next_value_local = x + 1"
    ).body

    def order() -> set[int]:
        return _lane_invariant_tail_after_consume(
            statements,
            invariant={0},
            accumulated={1},
            consumed=set(),
            rename_groups={"next_value": "carry"},
            accumulated_statements={1: stage},
        )

    if physical_write:
        with pytest.raises(exc.BackendUnsupported, match="before the accumulate pass"):
            order()
    else:
        assert order() == set()


def test_stage_effect_projection_preserves_memory_order_rejection() -> None:
    statements: list[ast.AST] = list(
        ast.parse("snapshot = x.iterator.load()\nx.iterator.store(value)").body
    )
    with pytest.raises(exc.BackendUnsupported, match="before the accumulate pass"):
        _lane_invariant_tail_after_consume(
            statements,
            invariant={0},
            accumulated={1},
            consumed=set(),
            rename_groups={},
            accumulated_statements={1: [statements[1]]},
        )


@pytest.mark.parametrize("block", [32, 64])
@pytest.mark.parametrize("factor", [2, 4])
def test_binary_epilogue_leaves_preserve_exact_words(block: int, factor: int) -> None:
    original = epilogue_subtiling._reshape_and_split
    depth = 0
    checked: list[tuple[tuple[int, ...], int, int]] = []

    def capture(graph, node, dim, count, block_id):
        nonlocal depth
        depth += 1
        pieces = original(graph, node, dim, count, block_id)
        depth -= 1
        if depth:
            return pieces
        env = CompileEnvironment.current()
        substitutions = {info.var.node.expr: block for info in env.block_sizes}

        def size(value: object) -> int:
            if isinstance(value, torch.SymInt):
                resolved = value.node.expr.subs(substitutions)
                assert isinstance(resolved, sympy.Integer)
                return int(resolved)
            assert isinstance(value, int)
            return value

        shape = tuple(size(value) for value in node.meta["val"].shape)
        # Include signed zero, infinities and NaN payloads: splitting is a
        # pure data movement operation, so all original FP32 bits must survive.
        with unset_fake_temporarily():
            patterns = torch.tensor(
                [0, -2147483648, 0x7F800000, -8388608, 0x7FC01234, 0x3F800000],
                dtype=torch.int32,
                device="cpu",
            )
            words = patterns.repeat(math.ceil(math.prod(shape) / len(patterns)))[
                : math.prod(shape)
            ].reshape(shape)
            values: dict[torch.fx.Node, torch.Tensor | tuple[torch.Tensor, ...]] = {
                node: words.view(torch.float32)
            }

            def evaluate(
                value: torch.fx.Node,
            ) -> torch.Tensor | tuple[torch.Tensor, ...]:
                if value in values:
                    return values[value]
                source_node = value.args[0]
                assert isinstance(source_node, torch.fx.Node)
                source = evaluate(source_node)
                if value.target is torch.ops.aten.view.default:
                    shape_arg = value.args[1]
                    assert isinstance(source, torch.Tensor)
                    assert isinstance(shape_arg, (list, tuple))
                    result = source.reshape([size(item) for item in shape_arg])
                elif value.target is torch.ops.aten.permute.default:
                    dims = value.args[1]
                    assert isinstance(source, torch.Tensor)
                    assert isinstance(dims, (list, tuple))
                    result = source.permute([size(item) for item in dims])
                elif value.target is split:
                    assert isinstance(source, torch.Tensor)
                    result = source.unbind(-1)
                else:
                    assert value.target is operator.getitem
                    index = value.args[1]
                    assert isinstance(source, tuple) and isinstance(index, int)
                    result = source[index]
                values[value] = result
                return result

            assert len(pieces) == count
            for piece, expected in zip(pieces, words.chunk(count, dim), strict=True):
                actual = evaluate(piece)
                assert isinstance(actual, torch.Tensor)
                assert torch.equal(actual.contiguous().view(torch.int32), expected), (
                    shape,
                    dim,
                    count,
                )
        checked.append((shape, dim, count))
        return pieces

    with (
        _mock_cuda_unavailable(),
        patch.object(epilogue_subtiling, "_reshape_and_split", capture),
    ):
        bound = _cpu_bind(
            _matmul_with_bias,
            (torch.empty(128, 128), torch.empty(128, 128), torch.empty(128)),
        )
        code = bound.to_code(
            helion.Config(block_sizes=[block, block, block], epilogue_subtile=factor)
        )
    assert len(checked) == 2  # matrix accumulator and broadcast bias
    assert code.count(".store(") == factor
    assert "split_smem" in code


def _assert_jagged_lane_phase_equations(
    tree: ast.Module, stages: list[ast.For]
) -> None:
    # LICM can put the isolated new maximum (and its exp scale) between the
    # two lane passes. Check the original online-softmax recurrence, including
    # multi-tile carry updates and masked tails, rather than a temporary name.
    parents = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and all(stage in node.body for stage in stages)
    ]
    assert len(parents) == 1
    chunk = parents[0]
    assert chunk.body.index(stages[0]) < chunk.body.index(stages[1])
    allocations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
    ]
    program = compile(
        ast.Module(body=[*allocations, chunk], type_ignores=[]), "<lane-phases>", "exec"
    )

    class Pointer:
        def __init__(self, values: list[float], index: int = 0) -> None:
            self.values, self.index = values, index

        def __add__(self, index: int) -> Pointer:
            return Pointer(self.values, self.index + index)

        def load(self) -> float:
            return self.values[self.index]

    scalar_cutlass = SimpleNamespace(Int32=int, Int64=int, Float32=float)
    scalar_cute = SimpleNamespace(
        make_rmem_tensor=lambda size, dtype: [0.0] * size,
        math=SimpleNamespace(
            max=lambda a, b, **kwargs: max(a, b),
            exp2=math.exp2,
            fma=lambda a, b, c: a * b + c,
        ),
    )
    for length in (15, 16, 19, 32, 33):
        for descending in (False, True):
            values = [index / 7.0 - 2.0 for index in range(length)]
            if descending:
                values.reverse()
            namespace = {
                "cutlass": scalar_cutlass,
                "cute": scalar_cute,
                "x_flat": SimpleNamespace(
                    iterator=Pointer([value for value in values for _ in range(8)]),
                    layout=SimpleNamespace(stride=(1,)),
                ),
                "_BLOCK_SIZE_2": 16,
                "amax": length,
                "v_2_copy_0": length,
                "starts_copy_0": 0,
                "indices_1": 0,
                "block_max": float("-inf"),
                "block_new_max": float("-inf"),
                "block_L": 0.0,
            }
            exec(program, namespace)
            expected_max = max(values)
            expected_sum = sum(math.exp(value - expected_max) for value in values)
            assert namespace["block_max"] == expected_max
            assert namespace["block_L"] == pytest.approx(expected_sum, rel=1e-12)


@pytest.mark.parametrize("features", [4, 8])
def test_original_jagged_softmax_generates_complete_reduction_phases(
    features: int,
) -> None:
    kernel = helion.kernel(
        jagged_softmax_kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with (
        _mock_cuda_unavailable(),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(
            kernel,
            (torch.empty(4096, features), torch.arange(129, dtype=torch.int64) * 32),
        )
        code = bound.to_code(helion.Config(block_sizes=[16, features, 16, 16]))
    assert "_helion_lane_reduce(" not in code
    tree = ast.parse(code)
    stages = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "lane_2"
    ]
    if features == 4:
        # This geometry reduces across threads instead of serial lane stages.
        reductions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_cute_grouped_reduce_shared_two_stage"
        ]
        assert len(reductions) == 2
        for node, operation in zip(reductions, ("max", "sum"), strict=True):
            assert isinstance(node.args[1], ast.Constant)
            assert node.args[1].value == operation
        assert not stages
        return
    assert len(stages) == 2
    for stage in stages:
        writes = {
            node.id
            for node in ast.walk(stage)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        assert not writes & {"block_max", "block_new_max", "block_L"}
    _assert_jagged_lane_phase_equations(tree, stages)
