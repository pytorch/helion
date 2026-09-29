from __future__ import annotations

import ast
import ctypes
from functools import cache
import itertools
import math
from types import CodeType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_cluster_online_pair import softmax_two_pass_kernel

import helion
from helion._compiler.cute.duplicate_reduction_carries import (
    eliminate_duplicate_cluster_maxima,
)
from helion._compiler.cute.hoist_warp_reduce import validate_cluster_reduce_placement
from helion._compiler.tile_strategy import NDTileStrategy
from helion.language._decorators import is_api_func
from helion.language.tile_ops import tile_begin

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from helion._compiler.inductor_lowering import CodegenState


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        # Complete lazy backend registration before tests patch handler helpers.
        assert is_api_func(tile_begin)
        assert callable(tile_begin._codegen["cute"])
        yield


def _emit_tile_begin(state: CodegenState) -> ast.expr:
    assert is_api_func(tile_begin)
    expression = tile_begin._codegen["cute"](state)
    assert isinstance(expression, ast.expr)
    return expression


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("active_loop", [False, True])
def test_nd_begin_uses_authoritative_offset_for_integer_lane_layouts(
    width: int, active_loop: bool
) -> None:
    # Exercise the real ND accessors without constructing a compiler/GPU state.
    # The lowering must not request a lane index solely to recover its boundary.
    strategy = object.__new__(NDTileStrategy)
    strategy.block_ids = [7]
    strategy.offset_vars = {7: "logical_begin"}
    strategy.index_vars = {7: "logical_index"}
    context = SimpleNamespace(strategy=strategy, block_thread_axes={7: 0})
    codegen = SimpleNamespace(
        active_device_loops={7: [context]} if active_loop else {},
        current_grid_state=context,
        index_var=Mock(side_effect=AssertionError("unused lane-index dependency")),
    )
    state = cast(
        "CodegenState", SimpleNamespace(codegen=codegen, proxy_arg=lambda _: 7)
    )
    with patch(
        "helion._compiler.cute.tile_ops._disable_flatten_get_tile", return_value=7
    ):
        expression = _emit_tile_begin(state)
    assert isinstance(expression, ast.expr)
    assert ast.unparse(expression) == "logical_begin"
    after = compile(
        ast.fix_missing_locations(ast.Expression(expression)), "<ND begin>", "eval"
    )
    before = compile(
        "logical_index - (logical_index - logical_begin)", "<old begin>", "eval"
    )
    dtype = np.int32 if width == 32 else np.int64

    def integer(value: int):
        return dtype((value + (1 << (width - 1))) % (1 << width) - (1 << (width - 1)))

    for layout, vector, cluster in itertools.product(
        ("blocked", "strided"), (1, 2, 8), (1, 2, 4)
    ):
        for offset in (0, 5, -7, (1 << (width - 1)) - 64, -(1 << (width - 1)) + 64):
            for cta, lane, element in itertools.product(
                range(cluster), range(32), range(vector)
            ):
                local = cta * 256 + (
                    lane * vector + element
                    if layout == "blocked"
                    else lane + element * 32
                )
                namespace = {
                    "logical_begin": integer(offset),
                    "logical_index": integer(offset + local),
                }
                with np.errstate(over="ignore"):
                    expected = eval(before, namespace)
                    actual = eval(after, namespace)
                assert actual == expected == integer(offset)
                assert type(actual) is dtype


@pytest.mark.parametrize("thread_axis", [None, 0, 2])
def test_non_nd_begin_retains_coordinate_subtraction(thread_axis: int | None) -> None:
    axes = {} if thread_axis is None else {7: thread_axis}
    codegen = SimpleNamespace(
        active_device_loops={},
        current_grid_state=SimpleNamespace(strategy=object(), block_thread_axes=axes),
        index_var=lambda _: "global_index",
        offset_var=lambda _: "fallback_offset",
        lift=Mock(side_effect=lambda value, **kwargs: value),
    )
    state = cast(
        "CodegenState", SimpleNamespace(codegen=codegen, proxy_arg=lambda _: 7)
    )
    with (
        patch(
            "helion._compiler.cute.tile_ops._disable_flatten_get_tile", return_value=7
        ),
        patch(
            "helion._compiler.cute.cute_reshape._grid_local_coord_expr",
            return_value="local_coordinate",
        ) as local,
    ):
        expression = _emit_tile_begin(state)
    if thread_axis is None:
        assert ast.unparse(expression) == "fallback_offset"
        local.assert_not_called()
        codegen.lift.assert_not_called()
    else:
        assert ast.unparse(expression) == "global_index - local_coordinate"
        local.assert_called_once_with(codegen, 7, thread_axis)
        assert codegen.lift.call_args.kwargs == {"dce": True, "prefix": "tile_begin"}


_DUPLICATE = """
buffer0 = cute.arch.alloc_smem(cutlass.Float32, 2)
barrier0 = cute.arch.alloc_smem(cutlass.Int64, 1)
if cutlass.Int32(cute.arch.thread_idx()[0]) == 0:
    cute.arch.mbarrier_init(barrier0, 1)
buffer1 = cute.arch.alloc_smem(cutlass.Float32, 2)
barrier1 = cute.arch.alloc_smem(cutlass.Int64, 1)
if cutlass.Int32(cute.arch.thread_idx()[0]) == 0:
    cute.arch.mbarrier_init(barrier1, 1)
maximum = cutlass.Float32(float('-inf'))
first = cutlass.Float32(float('-inf'))
for offset in range(cutlass.Int32(0), cutlass.Int32(N), cutlass.Int32(BLOCK)):
    acc0 = cutlass.Float32(float('-inf'))
    acc1 = cutlass.Float32(float('-inf'))
    for lane in range(2):
        packet = cute.arch.load(lane, ir.VectorType.get([4], cutlass.Float32.mlir_type))
        for v in cutlass.range_constexpr(4):
            value = cutlass.Float32(packet[v])
            acc0 = cute.arch.fmax(acc0, cutlass.Float32(value))
        first_tile = offset == 0
        predicate = cutlass.Boolean(first_tile)
        for v in cutlass.range_constexpr(4):
            value = cutlass.Float32(packet[v])
            selected = cutlass.Float32(value) if predicate else cutlass.Float32(float('-inf'))
            acc1 = cute.arch.fmax(acc1, cutlass.Float32(selected))
    lane0 = cutlass.Int32(cute.arch.thread_idx()[0])
    lane1 = cutlass.Int32(cute.arch.thread_idx()[0])
    r0 = _cute_grouped_reduce_cluster(acc0, 'max', cutlass.Float32(float('-inf')), lane0, buffer0, barrier0, group_span=32, cluster_n=2)
    typed0 = cutlass.Float32(r0)
    copy0 = maximum
    maximum = cute.math.max(cutlass.Float32(copy0), cutlass.Float32(typed0), propagate_nan=True)
    r1 = _cute_grouped_reduce_cluster(acc1, 'max', cutlass.Float32(float('-inf')), lane1, buffer1, barrier1, group_span=32, cluster_n=2)
    typed1 = cutlass.Float32(r1)
    copy1 = first
    first = cute.math.max(cutlass.Float32(copy1), cutlass.Float32(typed1), propagate_nan=True)
observe(maximum, first)
"""


def _eliminate(
    source: str, constants: dict[str, int] | None = None
) -> tuple[list[ast.stmt], list[ast.stmt]]:
    tree = ast.parse(source)
    before = ast.parse(ast.unparse(tree)).body
    return before, eliminate_duplicate_cluster_maxima(
        tree.body, {"N": 8, "BLOCK": 8} if constants is None else constants, {}
    )


def _execute(
    body: list[ast.stmt], values: np.ndarray
) -> tuple[tuple[np.float32, np.float32], int]:
    calls = 0
    result = []

    def collective(value, _op, _identity, _lane, _buffer, _barrier, **_keywords):
        nonlocal calls
        calls += 1
        return value

    def maximum(left, right, *, propagate_nan):
        assert propagate_nan
        return np.maximum(left, right)

    class Float32:
        mlir_type = "float32"

        def __call__(self, value):
            return np.float32(value)

    namespace = {
        "N": 8,
        "BLOCK": 8,
        "cutlass": SimpleNamespace(
            Float32=Float32(),
            Int32=int,
            Int64=int,
            Boolean=bool,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                alloc_smem=lambda _dtype, size: np.empty(size),
                mbarrier_init=lambda _barrier, _count: None,
                thread_idx=lambda: (0, 0, 0),
                load=lambda lane, _dtype: values[lane],
                fmax=np.fmax,
            ),
            math=SimpleNamespace(max=maximum),
        ),
        "ir": SimpleNamespace(VectorType=SimpleNamespace(get=lambda *args: args)),
        "_cute_grouped_reduce_cluster": collective,
        "observe": lambda *args: result.append(args),
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body, [])), "<fold>", "exec"),
        namespace,
    )
    assert len(result) == 1
    return result[0], calls


@pytest.mark.parametrize("integer_type", ["Int32", "Int64"])
def test_one_trip_duplicate_reduction_preserves_ieee_values_and_private_ownership(
    integer_type: str,
) -> None:
    source = _DUPLICATE.replace(
        "cutlass.Int32(0), cutlass.Int32(N), cutlass.Int32(BLOCK)",
        f"cutlass.{integer_type}(0), cutlass.{integer_type}(N), cutlass.{integer_type}(BLOCK)",
    )
    before, after = _eliminate(source)
    assert (
        ast.unparse(ast.Module(after, [])).count("_cute_grouped_reduce_cluster(") == 1
    )
    assert "buffer1" not in ast.unparse(ast.Module(after, []))
    assert "barrier1" not in ast.unparse(ast.Module(after, []))
    patterns = (-math.inf, -1.0, -0.0, 0.0, 1.0, math.inf, math.nan)
    for row in itertools.product(patterns, repeat=3):
        values = np.asarray(
            [*row, -math.inf, *row, -math.inf], dtype=np.float32
        ).reshape(2, 4)
        expected, original_calls = _execute(before, values)
        actual, calls = _execute(after, values)
        np.testing.assert_array_equal(actual, expected)
        assert original_calls == 2 and calls == 1


@pytest.mark.parametrize(
    "old,new",
    [
        ("cutlass.Int32(N)", "cutlass.Int32(N * 2)"),
        ("cutlass.Int32(N)", "cutlass.Float32(N)"),
        ("cutlass.Int32(0), cutlass.Int32(N)", "cutlass.Int32(8), cutlass.Int32(16)"),
        ("first_tile = offset == 0", "first_tile = offset == 8"),
        ("first_tile = offset == 0", "first_tile = lane == 0"),
        ("first_tile = offset == 0", "first_tile = external_predicate()"),
        ("first = cutlass.Float32(float('-inf'))", "first = cutlass.Float32(0)"),
        ("acc1 = cutlass.Float32(float('-inf'))", "acc1 = cutlass.Float32(0)"),
        ("selected = cutlass.Float32(value)", "selected = cutlass.Float16(value)"),
        ("first_tile = offset == 0", "packet[0] = 3\n        first_tile = offset == 0"),
        (
            "first_tile = offset == 0",
            "external_effect()\n        first_tile = offset == 0",
        ),
        (
            "first_tile = offset == 0",
            "other_packet = cute.arch.load(lane, ir.VectorType.get([4], cutlass.Float32.mlir_type))\n        first_tile = offset == 0",
        ),
        (
            "acc1 = cute.arch.fmax(acc1, cutlass.Float32(selected))",
            "acc1 = cute.arch.fmax(acc1, cutlass.Float32(selected))\n            acc1 = acc1 + 1",
        ),
        (
            "typed1 = cutlass.Float32(r1)",
            "external_effect(r1)\n    typed1 = cutlass.Float32(r1)",
        ),
        ("observe(maximum, first)", "first = 0\nobserve(maximum, first)"),
        ("observe(maximum, first)", "observe(barrier1, first)"),
        (
            "lane1 = cutlass.Int32(cute.arch.thread_idx()[0])",
            "lane1 = cutlass.Int32(cute.arch.thread_idx()[1])",
        ),
        ("buffer1, barrier1, group_span=32", "buffer1, barrier1, group_span=16"),
        (
            "cute.arch.mbarrier_init(barrier1, 1)",
            "external_effect()\n    cute.arch.mbarrier_init(barrier1, 1)",
        ),
        ("range(2)", "range(2, unexpected=side_effect())"),
        ("range_constexpr(4)", "range_constexpr(4, 9)"),
        ("range_constexpr(4)", "range_constexpr(4, unroll=side_effect())"),
        ("range(2)", "range(True)"),
        ("cute.arch.load(lane,", "cute.arch.load(side_effect(lane),"),
        (
            "group_span=32, cluster_n=2",
            "group_span=32, cluster_n=2, ignored=side_effect()",
        ),
        ("cutlass.Float32, 2)", "cutlass.Float32, side_effect())"),
        (
            "cute.arch.mbarrier_init(barrier1, 1)",
            "cute.arch.mbarrier_init(barrier1, side_effect())",
        ),
        ("first_tile = offset == 0", "offset = 0\n        first_tile = offset == 0"),
        ("first_tile = offset == 0", "lane = 0\n        first_tile = offset == 0"),
        (
            "typed1 = cutlass.Float32(r1)",
            "typed1 = cutlass.Float32(r1)\n    typed1 = cutlass.Float32(typed1)",
        ),
    ],
)
def test_duplicate_proof_declines_unknown_or_observable_cases(
    old: str, new: str
) -> None:
    source = _DUPLICATE.replace(old, new)
    if "other_packet" in source:
        # Only the second fold uses the new load, despite the identical address.
        start = source.index("other_packet")
        source = source[:start] + source[start:].replace("packet[v]", "other_packet[v]")
    before, after = _eliminate(source)
    assert ast.dump(ast.Module(after, []), include_attributes=False) == ast.dump(
        ast.Module(before, []), include_attributes=False
    )


@pytest.mark.parametrize(
    "constants",
    [
        {"BLOCK": 8},
        {"N": 0, "BLOCK": 8},
        {"N": 16, "BLOCK": 8},
        {"N": 2**32 + 8, "BLOCK": 2**32 + 8},
    ],
)
def test_trip_proof_requires_exactly_one_positive_iteration(
    constants: dict[str, int],
) -> None:
    before, after = _eliminate(_DUPLICATE, constants)
    assert ast.dump(ast.Module(after, [])) == ast.dump(ast.Module(before, []))


def test_scalar_alias_captures_value_before_an_overwrite() -> None:
    source = _DUPLICATE.replace(
        "acc0 = cute.arch.fmax(acc0, cutlass.Float32(value))",
        "captured = value\n            value = cutlass.Float32(0)\n"
        "            acc0 = cute.arch.fmax(acc0, cutlass.Float32(captured))",
    ).replace("selected = cutlass.Float32(value)", "selected = cutlass.Float32(0)")
    before, after = _eliminate(source)
    assert ast.dump(ast.Module(after, [])) == ast.dump(ast.Module(before, []))
    output, calls = _execute(after, np.arange(8, dtype=np.float32).reshape(2, 4))
    assert output == (7, 0) and calls == 2


def test_last_lane_vector_cannot_replace_the_complete_iteration_domain() -> None:
    source = _DUPLICATE.replace(
        "        first_tile = offset == 0",
        "    for lane in range(2):\n        first_tile = offset == 0",
    )
    before, after = _eliminate(source)
    assert ast.dump(ast.Module(after, [])) == ast.dump(ast.Module(before, []))
    values = np.asarray([[9, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    output, calls = _execute(after, values)
    assert output == (9, 1) and calls == 2


def test_parent_initializers_do_not_prove_loop_carried_scalar_inputs() -> None:
    source = _DUPLICATE.replace(
        "    for lane in range(2):",
        "    previous0 = cutlass.Float32(0)\n"
        "    previous1 = cutlass.Float32(0)\n"
        "    for lane in range(2):",
    )
    first_value = "value = cutlass.Float32(packet[v])"
    source = source.replace(first_value, "value = cutlass.Float32(previous0)", 1)
    source = source.replace(first_value, "value = cutlass.Float32(previous1)", 1)
    source = source.replace(
        "    lane0 =",
        "        previous0 = cutlass.Float32(packet[0])\n"
        "        previous1 = cutlass.Float32(packet[1])\n"
        "    lane0 =",
    )
    before, after = _eliminate(source)
    assert ast.dump(ast.Module(after, [])) == ast.dump(ast.Module(before, []))
    values = np.asarray([[9, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    output, calls = _execute(after, values)
    assert output == (9, 1) and calls == 2


def test_future_scalar_rename_alias_keeps_original_collectives() -> None:
    tree = ast.parse(_DUPLICATE.replace("selected =", "renamed_value ="))
    before = ast.dump(tree)
    after = eliminate_duplicate_cluster_maxima(
        tree.body, {"N": 8, "BLOCK": 8}, {"renamed_value": "value", "selected": "value"}
    )
    assert ast.dump(ast.Module(after, [])) == before


def test_local_write_shadows_an_enclosing_constexpr_fact() -> None:
    source = _DUPLICATE.replace(
        "    for lane in range(2):",
        "    shadow = cutlass.Float32(9)\n    for lane in range(2):",
    ).replace(
        "value = cutlass.Float32(packet[v])", "value = cutlass.Float32(shadow)", 1
    )
    source = source.replace(
        "selected = cutlass.Float32(value)", "selected = cutlass.Float32(0)"
    )
    before, after = _eliminate(source, {"N": 8, "BLOCK": 8, "shadow": 0})
    assert ast.dump(ast.Module(after, [])) == ast.dump(ast.Module(before, []))
    output, calls = _execute(after, np.zeros((2, 4), dtype=np.float32))
    assert output == (9, 0) and calls == 2


def test_iteration_placeholders_are_fresh_for_existing_names() -> None:
    tree = ast.parse(_DUPLICATE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "v":
            node.id = "_iteration_0"
    before, after = _eliminate(ast.unparse(tree))
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    assert _execute(before, values) == ((7, 7), 2)
    assert _execute(after, values) == ((7, 7), 1)


def test_duplicate_proof_preserves_backend_loop_metadata() -> None:
    tree = ast.parse(_DUPLICATE)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    for loop in loops:
        # pyrefly: ignore [missing-attribute]
        loop._helion_test_metadata = object()
    # pyrefly: ignore [missing-attribute]
    tokens = {id(loop): loop._helion_test_metadata for loop in loops}
    after = eliminate_duplicate_cluster_maxima(tree.body, {"N": 8, "BLOCK": 8}, {})
    assert after is not tree.body
    for node in ast.walk(ast.Module(after, [])):
        if isinstance(node, ast.For):
            # pyrefly: ignore [missing-attribute]
            assert node._helion_test_metadata is tokens[id(node)]


@cache
def _cluster_source(dtype: torch.dtype, columns: int, cluster: int) -> str:
    kernel = helion.kernel(
        softmax_two_pass_kernel.fn, backend="cute", static_shapes=True, fast_math=False
    )
    return kernel._bind_isolated((torch.empty(32, columns, dtype=dtype),)).to_code(
        helion.Config(
            block_sizes=[1, columns],
            num_threads=[0, 256],
            cute_vector_widths=[1, 8],
            cute_lane_layouts=["blocked", "strided"],
            cute_cluster_n=cluster,
        )
    )


@pytest.mark.parametrize(
    "dtype,columns,cluster", [(torch.bfloat16, 32768, 2), (torch.float16, 65536, 4)]
)
def test_static_cluster_keeps_one_exchange_with_global_first_tile_poison(
    dtype: torch.dtype, columns: int, cluster: int
) -> None:
    code = _cluster_source(dtype, columns, cluster)
    assert code.count("_cute_grouped_reduce_cluster_online_pair(") == 1
    assert "_cute_grouped_reduce_cluster(" not in code
    assert "_helion_first_tile_max" not in code
    assert "_pair_negative_inf_0" in code
    tree = ast.parse(code)
    kernel_body = next(
        node.body
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(decorator) == "cute.kernel" for decorator in node.decorator_list
        )
    )
    validate_cluster_reduce_placement(kernel_body, {"_BLOCK_SIZE_1": columns})
    pair_index = code.index("mi = _pair_gmax_0")
    poison_index = code.index("operator.eq(mi,", pair_index)
    assert poison_index > pair_index
    assert "fastmath=True" not in code


@cache
def _emitted_frame_expressions() -> tuple[dict[str, CodeType], CodeType, str]:
    source = ast.parse(_cluster_source(torch.bfloat16, 32768, 2))
    names = {
        "_pair_negative_inf_0",
        "_helion_scaled_0",
        "_helion_scaled_1",
        "_pair_rescale_0",
    }
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(source)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    }
    assert assignments.keys() == names
    exponentials = [
        node.value
        for node in ast.walk(source)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.math.exp2"
    ]
    assert len(exponentials) == 1
    inputs = {
        node.id for node in ast.walk(exponentials[0]) if isinstance(node, ast.Name)
    } - {"cute", "_helion_scaled_0"}
    assert len(inputs) == 1
    return (
        {
            name: compile(ast.Expression(expr), "<emitted frame>", "eval")
            for name, expr in assignments.items()
        },
        compile(ast.Expression(exponentials[0]), "<emitted exponential>", "eval"),
        next(iter(inputs)),
    )


@cache
def _host_fmaf() -> Callable[[float, float, float], float]:
    library = ctypes.CDLL("libm.so.6")
    function = library.fmaf
    function.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    function.restype = ctypes.c_float
    return cast("Callable[[float, float, float], float]", function)


def _fp32_array_fma(left: np.ndarray, right: float, addend: float) -> np.ndarray:
    # Execute the actual emitted operation with one FP32 rounding. A Python
    # multiply followed by add would model the old, uncontracted expression.
    fma = _host_fmaf()
    return np.array(
        [fma(float(value), float(right), float(addend)) for value in left],
        dtype=np.float32,
    )


def _cluster_softmax_oracle(values: torch.Tensor, *, repaired: bool) -> torch.Tensor:
    local_max = torch.amax(
        torch.nan_to_num(values, nan=-math.inf, neginf=-math.inf, posinf=math.inf),
        dim=1,
    )
    scale = 1.4426950408889634
    group_max = local_max.max()
    if repaired:
        expressions, exponential, input_name = _emitted_frame_expressions()
        rows = []
        rescales = []
        for index in range(values.shape[0]):
            namespace = {
                "cutlass": SimpleNamespace(Float32=np.float32),
                "cute": SimpleNamespace(
                    math=SimpleNamespace(exp2=np.exp2, fma=_fp32_array_fma)
                ),
                "mi": np.float32(local_max[index].item()),
                input_name: values[index].numpy(),
            }
            with np.errstate(invalid="ignore", over="ignore", under="ignore"):
                for name in ("_pair_negative_inf_0", "_helion_scaled_0"):
                    namespace[name] = eval(expressions[name], namespace)
                rows.append(torch.from_numpy(eval(exponential, namespace)))
                namespace["mi"] = np.float32(group_max.item())
                namespace["_helion_scaled_1"] = eval(
                    expressions["_helion_scaled_1"], namespace
                )
                rescale = eval(expressions["_pair_rescale_0"], namespace)
                rescales.append(float(rescale))
        exponentials = torch.stack(rows)
        scale_back = torch.tensor(rescales)
    else:
        exponentials = torch.exp2(values * scale - local_max[:, None] * scale)
        scale_back = torch.exp2(local_max * scale - group_max * scale)
    local_sum = exponentials.sum(dim=1)
    # This is the existing packed helper's rescale, with the true local maxima.
    group_sum = (local_sum * torch.exp2((local_max - group_max) * scale)).sum()
    return exponentials * (scale_back / group_sum)[:, None]


@pytest.mark.parametrize("finite", [-1000.0, -1.0, 0.0, 1.0])
@pytest.mark.parametrize("cluster", [2, 4, 8, 16])
def test_cluster_empty_slice_uses_zero_frame_without_changing_true_maximum(
    finite: float, cluster: int
) -> None:
    values = torch.full((cluster, 8), -math.inf)
    values[-1] = finite
    expected = torch.softmax(values.flatten(), dim=0).view_as(values)
    assert torch.isnan(_cluster_softmax_oracle(values, repaired=False)).all()
    actual = _cluster_softmax_oracle(values, repaired=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("special", [-math.inf, math.inf, math.nan])
@pytest.mark.parametrize("cluster", [2, 4, 8, 16])
def test_cluster_frame_keeps_nonfinite_rows_poisoned(
    special: float, cluster: int
) -> None:
    values = torch.full((cluster, 8), -math.inf)
    values[-1] = special
    actual = _cluster_softmax_oracle(values, repaired=True)
    assert torch.isnan(actual).all()
