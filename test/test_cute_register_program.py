"""CPU execution of generic register routing and demanded scalar programs."""

from __future__ import annotations

import ast
from contextlib import nullcontext
import operator
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import map_arg

from helion._compiler.cute.register_program import RegisterProgramEmitter
from helion._compiler.cute.register_program import _Literal
from helion._compiler.cute.register_program import register_program_supported
from helion._compiler.cute.row_fragment import RowFragment
from helion._compiler.cute.row_fragment import RowFragmentLayout
from helion._compiler.cute.row_fragment import row_fragment_tensor_inputs

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.fx import GraphModule
    from torch.fx import Node


class _VectorizeConditions(ast.NodeTransformer):
    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="where",
                    ctx=ast.Load(),
                ),
                args=[node.test, node.body, node.orelse],
                keywords=[],
            ),
            node,
        )

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        node.test = ast.Call(
            func=ast.Name(id="warp_predicate", ctx=ast.Load()),
            args=[node.test],
            keywords=[],
        )
        return node


def _run(
    module: GraphModule, *inputs: torch.Tensor, fake_mode: bool = False
) -> tuple[list[torch.Tensor], str]:
    lanes = inputs[0].size(0)
    statements = []
    scalar_functions = {}
    counter = 0

    def new_var(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}_{counter}"

    types = {
        torch.int32: "cutlass.Int32",
        torch.int64: "cutlass.Int64",
        torch.uint8: "cutlass.Uint8",
        torch.bool: "cutlass.Boolean",
        torch.float16: "cutlass.Float16",
        torch.bfloat16: "cutlass.BFloat16",
        torch.float32: "cutlass.Float32",
    }
    calls = {
        torch.ops.aten.maximum.default: "np.maximum",
        torch.ops.aten.minimum.default: "np.minimum",
        torch.ops.aten.add.Tensor: "np.add",
        torch.ops.aten.sub.Tensor: "np.subtract",
        torch.ops.aten.mul.Tensor: "np.multiply",
        torch.ops.aten.where.self: "np.where",
        torch.ops.aten.gt.Scalar: "np.greater",
    }

    def scalar(node: Node, arguments: list[str], emit: Callable[[str], None]) -> str:
        if node.target is torch.ops.aten.gt.Scalar:
            arguments.append(repr(node.args[1]))
        arity = 3 if node.target is torch.ops.aten.where.self else 2
        if node.target in calls and len(arguments) == arity:
            return f"{calls[node.target]}({', '.join(arguments)})"
        name = f"scalar_{len(scalar_functions)}"
        operands = row_fragment_tensor_inputs(node)

        def operation(*values):
            env = {
                source: torch.as_tensor(
                    np.asarray(value).copy(), dtype=source.meta["val"].dtype
                )
                for source, value in zip(operands, values, strict=True)
            }
            args = map_arg(node.args, env.__getitem__)
            kwargs = map_arg(node.kwargs, env.__getitem__)
            assert isinstance(node.target, torch._ops.OpOverload)
            result = node.target(*args, **kwargs)
            return as_numpy(result)

        scalar_functions[name] = operation
        return f"{name}({', '.join(arguments)})"

    bound = {
        node: RowFragment(
            f"input{index}",
            tensor.dtype,
            tensor.numel(),
            RowFragmentLayout(lanes, tensor.size(1), "lane"),
        )
        for index, (node, tensor) in enumerate(
            zip(module.graph.find_nodes(op="placeholder"), inputs, strict=True)
        )
    }
    with FakeTensorMode() if fake_mode else nullcontext():
        outputs = RegisterProgramEmitter(
            module,
            bound,
            lanes=lanes,
            lane_expr="lane",
            new_var=new_var,
            emit=statements.append,
            scalar=scalar,
            dtype_str=types.__getitem__,
        ).run()
    source = "\n".join(statements)
    parsed = _VectorizeConditions().visit(ast.parse(source))

    def as_numpy(tensor):
        return (tensor.float() if tensor.dtype == torch.bfloat16 else tensor).numpy()

    def bfloat16(value):
        return as_numpy(torch.as_tensor(np.asarray(value).copy()).to(torch.bfloat16))

    constructors = SimpleNamespace(
        Int32=np.int32,
        Int64=np.int64,
        Uint8=np.uint8,
        Uint32=np.uint32,
        Uint64=np.uint64,
        Boolean=np.bool_,
        Float16=np.float16,
        BFloat16=bfloat16,
        Float32=np.float32,
    )

    def shuffle(value, *, offset, mask_and_clamp=None):
        return np.broadcast_to(value, (lanes,))[offset]

    def warp_predicate(value):
        values = np.asarray(value)
        assert np.all(values == values.flat[0]), "divergent collective branch"
        return bool(values.flat[0])

    namespace = {
        "np": np,
        "warp_predicate": warp_predicate,
        "lane": np.arange(lanes, dtype=np.int32),
        "cutlass": constructors,
        "cute": SimpleNamespace(
            make_rmem_tensor=lambda size, dtype: np.zeros(
                (size, lanes), dtype=np.float32 if dtype is bfloat16 else dtype
            ),
            arch=SimpleNamespace(
                shuffle_sync=shuffle,
                shuffle_sync_bfly=lambda value, *, offset: shuffle(
                    value, offset=np.arange(lanes) ^ offset
                ),
            ),
        ),
        **{f"input{index}": as_numpy(tensor).T for index, tensor in enumerate(inputs)},
        **scalar_functions,
    }
    exec(
        compile(ast.fix_missing_locations(parsed), "<register-program>", "exec"),
        namespace,
    )
    results = []
    for output in outputs:
        result = namespace[output.name]
        assert isinstance(result, np.ndarray)
        results.append(torch.from_numpy(result.T.copy()).to(output.dtype))
    return results, source


def _check(function: Callable[..., torch.Tensor], *inputs: torch.Tensor) -> str:
    module = make_fx(function)(*inputs)
    actual, source = _run(module, *inputs)
    expected = function(*inputs)
    torch.testing.assert_close(actual[0], expected)
    assert "alloc_smem" not in source
    assert "sync_threads" not in source
    return source


def test_static_local_permutation_and_dead_compare_outputs():
    def function(x):
        permutation = torch.tensor([1, 0, 3, 2])
        other = torch.gather(x, 1, permutation.expand(x.size(0), -1))
        result = torch.where(
            torch.tensor([True, False, True, False]),
            torch.maximum(x, other),
            torch.minimum(x, other),
        )
        return result[:, ::2]

    values = torch.arange(32, dtype=torch.int64).reshape(8, 4)
    source = _check(function, values)
    assert "np.minimum" not in source
    assert "shuffle" not in source
    assert source.count("np.maximum") == 2


def test_negative_permutation_dimensions():
    def function(x):
        return x.permute(-1, -2)

    _check(function, torch.arange(16).reshape(4, 4))


@pytest.mark.parametrize("affine", [False, True])
def test_lane_permutations_use_bounded_bit_expressions(affine):
    permutation = (
        torch.arange(32) ^ ((torch.arange(32) & 16) >> 3)
        if affine
        else torch.randperm(32, generator=torch.Generator().manual_seed(47))
    )

    def function(x):
        return torch.gather(x, 0, permutation[:, None].expand(-1, x.size(1)))

    source = _check(function, torch.arange(64).reshape(32, 2))
    assert "shuffle_sync(" in source
    assert " if " not in source
    assert ("cutlass.Uint32" not in source) == affine


def test_lane_permutation_selects_source_register_before_shuffle():
    def function(x):
        groups = torch.arange(x.size(0))[:, None]
        positions = torch.where(
            groups < 4, torch.tensor([[0, 1, 2, 3]]), torch.tensor([[3, 2, 1, 0]])
        )
        local = torch.gather(x, 1, positions)
        return torch.gather(local, 0, (groups ^ 7).expand(-1, 4))

    values = torch.randperm(32).reshape(8, 4)
    source = _check(function, values)
    assert source.count("shuffle_sync_bfly") == 4


def test_same_source_lane_with_different_registers():
    def function(x):
        positions = torch.arange(x.size(0))[:, None] % x.size(1)
        broadcast = torch.gather(x, 0, torch.zeros(x.shape, dtype=torch.int64))
        return torch.gather(broadcast, 1, positions)

    values = torch.randperm(32).reshape(8, 4)
    source = _check(function, values)
    assert "shuffle_sync" in source


def test_arbitrary_gather_and_dynamic_where():
    def function(x, y):
        positions = torch.tensor(
            [[0, 1], [2, 3], [1, 0], [3, 2], [0, 1], [2, 3], [1, 0], [3, 2]]
        )
        first = torch.gather(x, 1, positions)
        second = torch.gather(
            y, 0, torch.tensor([3, 2, 0, 1, 7, 6, 4, 5])[:, None].expand(-1, 4)
        )[:, :2]
        return torch.where(first > 3, first, second)

    generator = torch.Generator().manual_seed(7)
    values = [torch.randint(0, 10, (8, 4), generator=generator) for _ in range(2)]
    _check(function, *values)


def test_permuted_producer_is_computed_once_in_its_own_registers():
    def function(x, y):
        producer = x + y
        groups = torch.arange(x.size(0))[:, None]
        peer = torch.gather(producer, 0, groups ^ 4)
        return producer + peer

    values = [torch.randperm(8).reshape(8, 1) for _ in range(2)]
    source = _check(function, *values)
    assert source.count("np.add") == 2
    assert source.count("shuffle_sync_bfly") == 1


@pytest.mark.parametrize("conflicting_source", [False, True])
def test_distinct_producers_select_before_shuffle_when_routes_allow(conflicting_source):
    def function(x, y):
        groups = torch.arange(x.size(0))[:, None]
        first, second = x + y, x - y
        source = groups * 0 if conflicting_source else groups ^ 3
        first = torch.gather(first, 0, source)
        second = torch.gather(second, 0, source)
        return torch.where((groups & 1) == 0, first, second)

    values = torch.arange(32, dtype=torch.int32).reshape(32, 1)
    source = _check(function, values, values * 3 + 5)
    assert source.count("np.add") == 1 and source.count("np.subtract") == 1
    assert source.count("shuffle_sync") == (2 if conflicting_source else 1)


def test_distinct_producer_routing_preserves_mixed_dtype_selection():
    def function(x, y):
        groups = torch.arange(x.size(0))[:, None]
        first = torch.gather(x + 5, 0, groups ^ 3)
        second = torch.gather(y - 7, 0, groups ^ 3)
        return torch.where((groups & 1) == 0, first, second)

    x = torch.arange(32, dtype=torch.int32).reshape(32, 1)
    source = _check(function, x, x.float() / 4)
    assert source.count("shuffle_sync") == 2


def test_lane_varying_producer_select_is_resident_before_permutation():
    def function(x, y):
        groups = torch.arange(x.size(0))[:, None]
        producer = torch.where((groups & 1) == 0, x + y, x - y)
        peer = torch.gather(producer, 0, groups ^ 3)
        return producer + peer

    values = [torch.randperm(32).reshape(32, 1) for _ in range(2)]
    source = _check(function, *values)
    assert source.count("np.add") == 2
    assert source.count("np.subtract") == 1
    assert source.count("shuffle_sync_bfly") == 1


def test_lane_varying_select_of_one_producer_folds_into_permutation():
    def function(x):
        groups = torch.arange(x.size(0))[:, None]
        peer = torch.gather(x, 0, groups ^ 7)
        producer = torch.where((groups & 4) == 0, x, peer)
        return torch.gather(producer, 0, groups ^ 2)

    source = _check(function, torch.randperm(32).reshape(32, 1))
    assert source.count("shuffle_sync") == 1


def test_constant_evaluation_inside_an_ambient_fake_mode():
    def function(x):
        groups = torch.arange(x.size(0))[:, None]
        return torch.gather(x, 0, (groups ^ 1).expand(-1, x.size(1)))

    values = torch.randperm(32).reshape(8, 4)
    module = make_fx(function)(values)
    outputs, source = _run(module, values, fake_mode=True)
    torch.testing.assert_close(outputs[0], function(values))
    assert source.count("shuffle_sync_bfly") == 4


def test_runtime_register_gather():
    def function(x, indices):
        return torch.gather(x.reshape(-1), 0, indices.reshape(-1)).reshape(
            indices.shape
        )

    generator = torch.Generator().manual_seed(17)
    values = torch.randperm(128, generator=generator).reshape(32, 4)
    indices = torch.randint(0, 128, (32, 2), generator=generator)
    source = _check(function, values, indices)
    assert source.count("shuffle_sync(") == 8


@pytest.mark.parametrize("groups", [1, 4, 32])
def test_grouped_integer_reduction(groups):
    def function(x):
        reduced = x.reshape(groups, -1).sum(1, dtype=torch.int32)
        index = torch.arange(x.size(0)) // (x.size(0) // groups)
        return torch.gather(reduced, 0, index).reshape(x.size(0), 1)

    values = torch.arange(128, dtype=torch.int32).reshape(32, 4)
    _check(function, values)


def test_register_local_reduction_after_reshape():
    def function(x):
        return x.reshape(32, 2, 4).sum(2, dtype=torch.int32)

    values = torch.arange(256, dtype=torch.int32).reshape(32, 8)
    source = _check(function, values)
    assert "shuffle" not in source


@pytest.mark.parametrize(
    "dtype,row",
    [
        (torch.float16, [65504, 65504, -65504, -65504]),
        (torch.bfloat16, [256, 1, -256, 0]),
    ],
)
@pytest.mark.parametrize("across_lanes", [False, True])
def test_reduced_precision_sum_uses_wider_accumulation(dtype, row, across_lanes):
    def function(x):
        if across_lanes:
            return x.sum().expand(32, 1)
        return x.sum(1, keepdim=True)

    values = torch.tensor(row, dtype=dtype).expand(32, 4).contiguous()
    _check(function, values)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_explicit_reduced_precision_sum_converts_inputs_first(dtype):
    def function(x):
        return x.sum(1, keepdim=True, dtype=dtype)

    values = torch.tensor([1.0001, -1.0]).expand(32, 2).contiguous()
    _check(function, values)


def test_constant_cache_preserves_signed_zero():
    def function(x):
        return torch.where(x > 0, torch.full(x.shape, -0.0), torch.zeros(x.shape))

    values = torch.arange(-4, 4).reshape(4, 2).float()
    actual, _source = _run(make_fx(function)(values), values)
    assert torch.equal(torch.signbit(actual[0]), torch.signbit(function(values)))


@pytest.mark.parametrize("kind", ["any", "all"])
@pytest.mark.parametrize(
    "dtype,value", [(torch.float32, 0.5), (torch.int32, 2), (torch.uint8, 2)]
)
def test_boolean_reductions_convert_numeric_truth(kind, dtype, value):
    def function(x):
        result = x.any() if kind == "any" else x.all()
        return result.expand(32, 1)

    values = torch.full((32, 2), value, dtype=dtype)
    _check(function, values)
    values[0, 0] = 0
    _check(function, values)


def test_deep_functional_program_compiles_without_python_recursion():
    def function(x):
        for _step in range(1200):
            x = x + 1
        return x

    values = torch.arange(8, dtype=torch.int32).reshape(8, 1)
    _check(function, values)


@pytest.mark.parametrize("positive", [False, True])
def test_warp_uniform_conditional(positive):
    def function(x):
        return torch.cond(
            x.any(), lambda value: value + 2, lambda value: value - 3, (x,)
        )

    values = torch.full((32, 4), int(positive), dtype=torch.int32)
    source = _check(function, values)
    assert "if " in source and "else:" in source


@pytest.mark.parametrize("positive", [False, True])
def test_conditional_specializes_literal_captures_without_mutating_trace(positive):
    def function(x):
        return torch.ops.higher_order.cond(
            x.any(),
            lambda value, width, bias: (value + torch.arange(width)[None, :] + bias,),
            lambda value, width, bias: (value - bias,),
            (x, 8, 3),
        )[0]

    values = torch.full((32, 8), float(positive))
    module = make_fx(function)(values)
    branch = module.true_graph_0
    _value, width, bias = branch.graph.find_nodes(op="placeholder")
    arange = next(
        node
        for node in branch.graph.nodes
        if node.target is torch.ops.aten.arange.default
    )
    with branch.graph.inserting_before(arange):
        computed_width = branch.graph.call_function(operator.add, (width, 0))
        computed_width.meta["val"] = 8
    arange.args = (computed_width,)
    addition = next(
        node
        for node in branch.graph.nodes
        if node.target is torch.ops.aten.add.Tensor and node.args[1] == 3
    )
    addition.args = (addition.args[0], bias)
    branch.recompile()
    before = {name: str(child.graph) for name, child in module.named_modules()}
    assert register_program_supported(
        module, tuple(module.graph.find_nodes(op="placeholder")), lanes=32
    )
    actual, source = _run(module, values)
    torch.testing.assert_close(actual[0], function(values))
    assert "if " in source and "else:" in source
    assert {name: str(child.graph) for name, child in module.named_modules()} == before


@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize("fallback", [False, True])
def test_coarse_tensor_program(recovery, fallback):
    from helion._compiler.cute.selection_coarse import coarse_rank_selection

    def function(x):
        return coarse_rank_selection(x, 4, "batcher", "sequential", 5, 2, 8, recovery)

    generator = torch.Generator().manual_seed(18)
    values = (torch.rand((32, 4), generator=generator) + 1).view(torch.int32)
    if fallback:
        values[0, 0] = 0x7F800000
    source = _check(function, values)
    assert "if " in source and "else:" in source


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize(
    "lanes,registers,k", [(1, 8, 4), (4, 8, 8), (8, 4, 4), (8, 8, 16)]
)
def test_tensor_selection_program(distributed, lanes, registers, k):
    from helion._compiler.cute.selection_network import selection_network

    def function(x):
        return selection_network(
            x, k, mode="distributed" if distributed else "replicated"
        )

    if not distributed:
        registers = max(registers, k)
    values = torch.randperm(lanes * registers).reshape(lanes, registers)
    source = _check(function, values)
    assert "distributed_topk" not in source
    assert "local_topk" not in source


@pytest.mark.parametrize("dtype,bits", [(torch.int32, 32), (torch.int64, 64)])
@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16, 32])
def test_integer_lane_literals_preserve_signed_bits(dtype, bits, lanes):
    def signed(value, width):
        value = int(value) & ((1 << width) - 1)
        return value - (1 << width) if value & (1 << (width - 1)) else value

    constructors = SimpleNamespace(
        Int32=lambda value: signed(value, 32),
        Int64=lambda value: signed(value, 64),
        Uint32=lambda value: int(value) & ((1 << 32) - 1),
        Uint64=lambda value: int(value) & ((1 << 64) - 1),
    )
    emitter = RegisterProgramEmitter.__new__(RegisterProgramEmitter)
    emitter.lanes = lanes
    emitter.lane = "lane"
    emitter.dtype_str = {
        torch.int32: "cutlass.Int32",
        torch.int64: "cutlass.Int64",
    }.__getitem__
    rng = random.Random(146779 + bits + lanes)
    tables = [
        [255 - (lane % 16) * 8 for lane in range(lanes)],
        [((lane ^ 7) << 9) | 31 for lane in range(lanes)],
        [-(1 << (bits - 1)) + lane for lane in range(lanes)],
        [(1 << (bits - 1)) - 1 - lane for lane in range(lanes)],
        [-1] * lanes,
        *[
            [rng.randrange(-(1 << (bits - 1)), 1 << (bits - 1)) for _ in range(lanes)]
            for _ in range(12)
        ],
    ]
    for values in tables:
        for partial in (False, True):
            literals = tuple(
                _Literal(value, dtype)
                if not partial or lane == 0 or rng.randrange(2)
                else None
                for lane, value in enumerate(values)
            )
            expression = emitter._integer_literals(literals)
            assert expression is not None and " if " not in expression
            for lane, literal in enumerate(literals):
                if literal is not None:
                    assert (
                        eval(expression, {"cutlass": constructors, "lane": lane})
                        == literal.value
                    )


def test_noninteger_or_mixed_lane_literals_keep_generic_fallback():
    emitter = RegisterProgramEmitter.__new__(RegisterProgramEmitter)
    for literals in (
        (),
        (None, None),
        (_Literal(0.0, torch.float32), _Literal(1.0, torch.float32)),
        (_Literal(True, torch.bool), _Literal(False, torch.bool)),
        (_Literal(1, torch.int32), _Literal(2, torch.int64)),
    ):
        assert emitter._integer_literals(literals) is None
