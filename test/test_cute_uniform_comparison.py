from __future__ import annotations

import ast
import ctypes
from functools import lru_cache
import struct
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import uniform_comparison
from helion._compiler.cute.autotune.heuristics.cute import CutePointwiseVecHeuristic
from helion._compiler.cute.uniform_comparison import UniformComparisonLowering
from helion._compiler.cute.uniform_comparison import cutoff_function
from helion._compiler.cute.uniform_comparison import lower_uniform_comparisons
from helion._compiler.cute.uniform_comparison import match_uniform_float_gt
from helion._testing import import_example
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl
from helion.language.random_ops import _uint32_to_uniform_float

_examples_low_mem_dropout = import_example("low_mem_dropout")
low_mem_dropout = _examples_low_mem_dropout.low_mem_dropout
low_mem_dropout_bwd = _examples_low_mem_dropout.low_mem_dropout_bwd
pytestmark = [pytest.mark.cpu_only, pytest.mark.cpu_render]


def _trace(*, probability=0.25):
    def compare(words):
        uniform = _uint32_to_uniform_float(words)
        return uniform > probability, uniform, words

    env = SimpleNamespace(backend=SimpleNamespace(name="cute"))
    with patch.object(CompileEnvironment, "current", return_value=env):
        return make_fx(compare)(torch.zeros(17, dtype=torch.int64))


def _comparison(graph):
    return next(
        node
        for node in graph.graph.nodes
        if node.target in (torch.ops.aten.gt.Scalar, torch.ops.aten.gt.Tensor)
    )


def test_match_preserves_all_uniform_and_integer_users():
    graph = _trace()
    before = graph.code
    comparison = _comparison(graph)
    word = match_uniform_float_gt(comparison)
    assert word is not None and word.meta["val"].dtype is torch.int64
    assert graph.code == before
    assert len(comparison.args[0].users) == 2  # comparison and observable return
    inputs = torch.tensor([-(1 << 63), -1, 0, 1, (1 << 31) - 1, 1 << 31, (1 << 63) - 1])
    expected = _uint32_to_uniform_float_reference(inputs)
    actual_mask, actual_uniform, actual_words = graph(inputs)
    torch.testing.assert_close(actual_uniform, expected, rtol=0, atol=0)
    assert torch.equal(actual_mask, expected > 0.25)
    assert torch.equal(actual_words, inputs)


def _uint32_to_uniform_float_reference(words):
    unsigned = words & 0xFFFFFFFF
    magnitude = torch.minimum(unsigned, 0xFFFFFFFF - unsigned)
    return magnitude.float() * torch.tensor(0x2FFFFFFF, dtype=torch.int32).view(
        torch.float32
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "scale",
        "float64",
        "int32",
        "fold_mask",
        "sign_offset",
        "negative_offset",
        "sign_predicate",
        "terminal_predicate",
        "conversion_dtype",
        "alpha",
    ),
)
def test_pattern_declines_changed_arithmetic(mutation):
    graph = _trace()
    comparison = _comparison(graph)
    product = comparison.args[0]
    converted = product.args[0]
    magnitude = converted.args[0]
    condition, negative, signed = magnitude.args
    masked = signed.args[0]
    shifted = masked.args[0]
    if mutation == "scale":
        product.args = (converted, 2.0**-31)
    elif mutation == "float64":
        product.meta["val"] = torch.empty(17, dtype=torch.float64)
    elif mutation == "int32":
        shifted.meta["val"] = torch.empty(17, dtype=torch.int32)
    elif mutation == "fold_mask":
        masked.args = (shifted, 0x7FFFFFFF)
    elif mutation == "sign_offset":
        signed.args = (masked, (1 << 31) - 1)
    elif mutation == "negative_offset":
        negative.args = (negative.args[0], 2)
    elif mutation == "sign_predicate":
        condition.target = torch.ops.aten.le.Scalar
    elif mutation == "terminal_predicate":
        comparison.target = torch.ops.aten.ge.Scalar
    elif mutation == "conversion_dtype":
        converted.args = (magnitude, torch.float64)
    elif mutation == "alpha":
        shifted.kwargs = {"alpha": 2}
    assert match_uniform_float_gt(comparison) is None


@pytest.mark.parametrize("probability", (0, True, torch.tensor(0.25), torch.ones(17)))
def test_pattern_requires_scalar_fp32_threshold_contract(probability):
    graph = _trace(probability=probability)
    assert match_uniform_float_gt(_comparison(graph)) is None


def test_combine_helper_scope_keeps_original_comparison():
    node = _comparison(_trace())
    function = SimpleNamespace(codegen=object())
    context = SimpleNamespace(
        cg=SimpleNamespace(device_function=function),
        to_ast=lambda value: (
            ast.Name(id=value.name, ctx=ast.Load())
            if isinstance(value, torch.fx.Node)
            else ast.Constant(value=value)
        ),
    )
    result = UniformComparisonLowering().codegen(context, node)
    assert ast.unparse(result) == f"operator.gt({node.args[0].name}, 0.25)"


class _Uint32(int):
    def __new__(cls, value):
        return int.__new__(cls, int(value) & 0xFFFFFFFF)

    def __lshift__(self, count):
        # PTX clamps unsigned counts at the word width, including inactive
        # operations that a compiler hoists out of structured branches.
        return _Uint32(0 if count >= 32 else int(self) << int(count))

    def __rshift__(self, count):
        return _Uint32(0 if count >= 32 else int(self) >> int(count))


class _Int32(int):
    def __new__(cls, value):
        return int.__new__(cls, ((int(value) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31))


class _Float32(float):
    def __new__(cls, value):
        # Match the runtime Float32 ABI for huge Python doubles as well as
        # ordinary values; struct.pack('f', value) alone raises on overflow.
        return float.__new__(cls, ctypes.c_float(value).value)

    def bitcast(self, dtype):
        assert dtype is _Uint32
        return _Uint32(struct.unpack("<I", struct.pack("<f", self))[0])


@lru_cache(None)
def _cutoff():
    helper = cutoff_function("cutoff")
    helper.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace = {
        "cutlass": SimpleNamespace(Float32=_Float32, Uint32=_Uint32, Int32=_Int32)
    }
    exec(compile(module, "<uniform-cutoff>", "exec"), namespace)
    return namespace["cutoff"]


def _oracle_cutoffs(probabilities):
    low = np.full(probabilities.shape, -1, dtype=np.int64)
    high = np.full(probabilities.shape, 0x7FFFFFFF, dtype=np.int64)
    scale = np.array([0x2FFFFFFF], dtype=np.uint32).view(np.float32)[0]
    for _iteration in range(32):
        middle = (low + high + 1) // 2
        original = middle.astype(np.float32) * scale
        keep = original > probabilities
        high = np.where(keep, middle - 1, high)
        low = np.where(keep, low, middle)
    return low


def test_cutoff_matches_independent_binary_search_and_both_preimages():
    rng = np.random.default_rng(128730)
    mantissas = np.array([0, 1, 2, 3, 0x3FFFFF, 0x7FFFFD, 0x7FFFFE, 0x7FFFFF])
    boundaries = (
        np.arange(256, dtype=np.uint32)[:, None] * np.uint32(1 << 23)
        + mantissas.astype(np.uint32)[None, :]
    ).ravel()
    bits = np.concatenate(
        (
            boundaries,
            boundaries | np.uint32(1 << 31),
            rng.integers(0, 1 << 32, 8192, dtype=np.uint32),
        )
    )
    probabilities = bits.view(np.float32)
    expected = _oracle_cutoffs(probabilities)
    triples = np.array(
        [_cutoff()(float(value)) for value in probabilities], dtype=np.uint64
    )
    first = ((expected + 1) & 0xFFFFFFFF).astype(np.uint64)
    span = ((-2 * (expected + 1)) & 0xFFFFFFFF).astype(np.uint64)
    np.testing.assert_array_equal(triples[:, 0], first)
    np.testing.assert_array_equal(triples[:, 1], span)
    np.testing.assert_array_equal(triples[:, 2], expected < 0)
    scale = np.array([0x2FFFFFFF], dtype=np.uint32).view(np.float32)[0]
    for delta in (-1, 0, 1, 2):
        magnitude = np.clip(expected + delta, 0, 0x7FFFFFFF).astype(np.uint64)
        original = magnitude.astype(np.float32) * scale > probabilities
        for word in (magnitude, np.uint64(0xFFFFFFFF) - magnitude):
            transformed = (triples[:, 2] != 0) | (
                ((word - triples[:, 0]) & np.uint64(0xFFFFFFFF)) < triples[:, 1]
            )
            np.testing.assert_array_equal(transformed, original)


@pytest.mark.parametrize(
    "probability, expected",
    (
        (0.25, (536871008, 3221225280, False)),
        (0.0, (1, 4294967294, False)),
        (-0.0, (1, 4294967294, False)),
        (-1e-300, (1, 4294967294, False)),
        (-float("inf"), (0, 0, True)),
        (-1e300, (0, 0, True)),
        (float("inf"), (2147483648, 0, False)),
        (1e300, (2147483648, 0, False)),
        (float("nan"), (2147483648, 0, False)),
        (-float("nan"), (2147483648, 0, False)),
    ),
)
def test_special_probability_abi(probability, expected):
    assert _cutoff()(probability) == expected


def test_oversize_unsigned_shift_model():
    for count in (32, 33, 255, 0xFFFFFFFF):
        assert _Uint32(0xFFFFFF) << _Uint32(count) == 0
        assert _Uint32(0xFFFFFF) >> _Uint32(count) == 0


def _function(renames=None):
    counter = iter(range(100))
    return SimpleNamespace(
        name="test_kernel",
        cute_state=SimpleNamespace(
            uniform_comparison_marker="_marked", uniform_cutoff_dtypes={}
        ),
        _variable_renames={} if renames is None else renames,
        codegen=SimpleNamespace(module_statements=[]),
        new_var=lambda stem: f"{stem}_{next(counter)}",
    )


def _rewrite(body: str, *, floats=("p",), renames=None):
    function = _function(renames)
    result = lower_uniform_comparisons(
        ast.parse(body).body, function, float_scalar_names=set(floats)
    )
    return ast.Module(body=result, type_ignores=[]), function.codegen.module_statements


def test_cutoff_names_record_their_dsl_types():
    # The hoisted tuple assignment does not show its names' types, so the
    # function state records them (``sweep_values`` types a carried mask).
    function = _function()
    result = lower_uniform_comparisons(
        ast.parse("keep = _marked(word, uniform, p)").body,
        function,
        float_scalar_names={"p"},
    )
    assert ast.unparse(result[0]).startswith(
        "_helion_uniform_first_1, _helion_uniform_span_2, _helion_uniform_all_3 = "
    )
    assert function.cute_state.uniform_cutoff_dtypes == {
        "_helion_uniform_first_1": "cutlass.Uint32",
        "_helion_uniform_span_2": "cutlass.Uint32",
        "_helion_uniform_all_3": "cutlass.Boolean",
    }


def test_cutoff_is_shared_outside_packet_and_tail_loops():
    result, helpers = _rewrite("""
for packet in range(2):
    for lane in cutlass.range_constexpr(4):
        keep = _marked(word, uniform, p)
for tail in range(3):
    other = _marked(other_word, other_uniform, p)
""")
    assert len(helpers) == 1
    assert isinstance(result.body[0], ast.Assign)
    for loop in result.body[1:]:
        assert "uniform_cutoff" not in ast.unparse(loop)
        assert "_marked" not in ast.unparse(loop)
        assert "cutlass.Uint32" in ast.unparse(loop)


@pytest.mark.parametrize(
    "body, renames",
    (
        ("p = p + 1\nkeep = _marked(word, uniform, p)", {}),
        ("keep = _marked(word, uniform, p)\np += 1", {}),
        ("if condition:\n    p = 0.5\nkeep = _marked(word, uniform, p)", {}),
        ("for p in range(3):\n    keep = _marked(word, uniform, p)", {}),
        ("alias = 0.5\nkeep = _marked(word, uniform, p)", {"alias": ["p"]}),
        ("keep = _marked(word, uniform, loaded_threshold)", {}),
        ("keep = _marked(word, uniform, cutlass.Float64(p))", {}),
    ),
)
def test_unproved_uniform_scopes_restore_original_comparison(body, renames):
    result, helpers = _rewrite(body, renames=renames)
    assert not helpers
    assert "_marked" not in ast.unparse(result)
    assert "operator.gt(uniform," in ast.unparse(result)


@helion.kernel(
    backend="cute", static_shapes=False, autotune_effort="none", cute_rng_stream="word0"
)
def _observe_uniform(x: torch.Tensor, p: float, seed: int):
    random = torch.empty_like(x)
    selected = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = hl.rand([tile], seed=seed)
        random[tile] = value
        selected[tile] = torch.where(value > p, x[tile], 0.0)
    return random, selected


def _config(width=4):
    return helion.Config(
        block_sizes=[128 * width],
        num_threads=[128],
        cute_vector_widths=[width],
        cute_lane_layouts=["blocked"],
        cute_cluster_n=1,
    )


@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("width", (1, 4, 8))
def test_original_dropout_hoists_one_cutoff_and_preserves_host_and_philox(
    backward, width
):
    original = low_mem_dropout_bwd if backward else low_mem_dropout

    def code():
        kernel = helion.kernel(
            original.fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
            cute_rng_stream="word0",
        )
        bound = cpu_bind(kernel, (0.25, torch.empty(513), 123))
        return bound.to_code(_config(width))

    actual = ast.parse(code())
    with patch.object(uniform_comparison, "match_uniform_float_gt", return_value=None):
        control = ast.parse(code())
    actual_functions = {
        node.name: node for node in actual.body if isinstance(node, ast.FunctionDef)
    }
    control_functions = {
        node.name: node for node in control.body if isinstance(node, ast.FunctionDef)
    }
    assert ast.dump(actual_functions[original.fn.__name__]) == ast.dump(
        control_functions[original.fn.__name__]
    )
    kernel = actual_functions[f"_helion_{original.fn.__name__}"]
    assert "uniform_cutoff(p)" in ast.unparse(kernel.body[0])
    for loop in [node for node in ast.walk(kernel) if isinstance(node, ast.For)]:
        assert "uniform_cutoff(" not in ast.unparse(loop)
    if width > 1:
        assert "_helion_affine_load" in ast.unparse(kernel)
    assert "_helion_uniform_gt(" not in ast.unparse(actual)

    def philox_calls(module):
        return [
            ast.dump(node)
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_cute_inline_asm_elementwise"
        ]

    assert philox_calls(actual) == philox_calls(control)
    namespace = {"torch": torch, "_default_cute_launcher": lambda *args: None}
    namespace[kernel.name] = object()
    host = ast.Module(body=[actual_functions[original.fn.__name__]], type_ignores=[])
    exec(compile(host, "<dropout-host>", "exec"), namespace)
    with pytest.raises(ZeroDivisionError):
        namespace[original.fn.__name__](1.0, torch.ones(17), 123)


def test_observable_uniform_keeps_original_cast_scale_and_store():
    bound = cpu_bind(_observe_uniform, (torch.empty(513), 0.25, 123))
    source = bound.to_code(_config(1))
    assert "uniform_cutoff(p)" in source
    assert "4.6566127342e-10" in source
    assert "random" in source
    assert "cutlass.Float32(" in source
    assert bound.host_function is not None
    comparisons = [
        node
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if isinstance(node.meta.get("lowering"), UniformComparisonLowering)
    ]
    assert len(comparisons) == 1
    assert len(comparisons[0].args[0].users) == 2


def test_pointwise_seed_discovery_is_unchanged():
    def seeds():
        kernel = helion.kernel(
            low_mem_dropout.fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
            cute_rng_stream="word0",
        )
        bound = cpu_bind(kernel, (0.25, torch.empty(262144), 123))
        assert bound.config_spec.pointwise_facts
        with bound.env:
            return CutePointwiseVecHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )

    actual = seeds()
    with patch.object(uniform_comparison, "match_uniform_float_gt", return_value=None):
        control = seeds()
    assert actual and actual == control


CUDA_DEVICE = "cuda"


def _pair(function, arguments, config):
    def bind():
        kernel = helion.kernel(
            function,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
            cute_rng_stream="word0",
        )
        bound = kernel._bind_isolated(arguments)
        source = bound.to_code(config)
        bound.set_config(config)
        return bound, source

    with patch.object(uniform_comparison, "match_uniform_float_gt", return_value=None):
        control, control_source = bind()
    candidate, candidate_source = bind()
    assert "uniform_cutoff(" not in control_source
    assert "uniform_cutoff(" in candidate_source
    assert control_source != candidate_source
    return control, candidate


def _assert_exact(actual, expected):
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_exact(left, right)
        return
    assert actual.shape == expected.shape and actual.dtype is expected.dtype
    if actual.dtype is torch.bool:
        assert torch.equal(actual, expected)
        return
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    integer = torch.int32 if actual.element_size() == 4 else torch.int16
    assert torch.equal(actual[~nan].view(integer), expected[~nan].view(integer))


def _poison(value):
    if isinstance(value, tuple):
        for item in value:
            _poison(item)
    elif value.dtype is torch.bool:
        value.logical_not_()
    else:
        value.fill_(float("nan"))


def _graph(candidate, control, arguments, mutate):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            candidate(*arguments)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = candidate(*arguments)
    for iteration in range(3):
        mutate(iteration)
        expected = control(*arguments)
        _poison(result)
        graph.replay()
        torch.cuda.synchronize()
        _assert_exact(result, expected)


def _special_inputs(data):
    data[:5] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")],
        dtype=data.dtype,
        device=data.device,
    )


def _dropout(backward, length, dtype, width):
    original = low_mem_dropout_bwd if backward else low_mem_dropout
    data = torch.linspace(-3, 3, length, device=CUDA_DEVICE, dtype=dtype)
    _special_inputs(data)
    arguments = (0.25, data, 123)
    control, candidate = _pair(original.fn, arguments, _config(width))
    for seed, probability in ((123, 0.25), (-1, 0.6), (0x1234567812345678, -0.0)):
        _assert_exact(
            candidate(probability, data, seed), control(probability, data, seed)
        )
    replacement = data.clone()
    _assert_exact(candidate(0.25, replacement, 123), control(0.25, replacement, 123))
    for bound in (control, candidate):
        with pytest.raises(ZeroDivisionError):
            bound(1.0, data, 123)

    def mutate(iteration):
        data.add_(0.125 * (iteration + 1))
        _special_inputs(data)

    _graph(candidate, control, arguments, mutate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.cuda
@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("length", (262144, 4194304, 33554432))
def test_original_forward_backward_exact(backward, length):
    _dropout(backward, length, torch.float32, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.cuda
@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_half_precision_tail_and_poisoned_graph(backward, dtype):
    _dropout(backward, 517, dtype, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.cuda
def test_nonfinite_probability_and_scaling_contract():
    data = torch.linspace(-2, 2, 257, device=CUDA_DEVICE)
    _special_inputs(data)
    control, candidate = _pair(low_mem_dropout.fn, (0.25, data, 123), _config(4))
    for probability in (
        -float("inf"),
        -1e300,
        -0.5,
        -1e-300,
        -0.0,
        0.0,
        2.0**-32,
        0.25,
        1.0 - 2.0**-24,
        1.0 + 2.0**-30,
        2.0,
        1e300,
        float("inf"),
        float("nan"),
        -float("nan"),
    ):
        _assert_exact(candidate(probability, data, -1), control(probability, data, -1))


def _observable_uniforms(x: torch.Tensor, p: float, seed: int):
    random = torch.empty_like(x)
    selected = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = hl.rand([tile], seed=seed)
        random[tile] = value
        selected[tile] = torch.where(value > p, x[tile], 0.0)
    return random, selected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.cuda
def test_observable_random_values_and_mutated_graph():
    data = torch.linspace(-2, 2, 517, device=CUDA_DEVICE)
    arguments = (data, 0.25, 123)
    control, candidate = _pair(_observable_uniforms, arguments, _config(4))
    for seed in (123, -1, 0x1234567812345678):
        _assert_exact(candidate(data, 0.25, seed), control(data, 0.25, seed))
    _graph(candidate, control, arguments, lambda iteration: data.add_(iteration + 1))


def _converted_word_comparison(words: torch.Tensor, p: float):
    values = torch.empty(words.shape, dtype=torch.float32, device=words.device)
    selected = torch.empty(words.shape, dtype=torch.bool, device=words.device)
    for tile in hl.tile(words.numel()):
        word = words[tile]
        signed = ((word + 2147483648) & 4294967295) - 2147483648
        magnitude = torch.where(signed < 0, -signed - 1, signed)
        uniform = magnitude.to(torch.float32) * 4.6566127342e-10
        values[tile] = uniform
        selected[tile] = uniform > p
    return values, selected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.cuda
@pytest.mark.parametrize("width", (1, 4))
def test_arbitrary_integer_words_and_threshold_endpoints(width):
    magnitudes = torch.tensor(
        [0, 1, 2, 3, 16777215, 16777216, 536871006, 536871007, 536871008, 2147483647],
        dtype=torch.int64,
        device=CUDA_DEVICE,
    )
    words = torch.cat(
        (
            magnitudes,
            0xFFFFFFFF - magnitudes,
            magnitudes - (1 << 63),
            torch.tensor([-1, (1 << 63) - 1], device=CUDA_DEVICE, dtype=torch.int64),
        )
    )
    control, candidate = _pair(
        _converted_word_comparison, (words, 0.25), _config(width)
    )
    probabilities = torch.tensor(
        [0, 1, 0x2FFFFFFE, 0x2FFFFFFF, 0x30000000, 0x3E800000, 0x3F7FFFFF, 0x3F800000],
        dtype=torch.int32,
    ).view(torch.float32)
    for probability in [*probabilities.tolist(), float("nan"), -float("inf")]:
        _assert_exact(candidate(words, probability), control(words, probability))
    _graph(
        candidate,
        control,
        (words, 0.25),
        lambda iteration: words.bitwise_xor_(0xFFFFFFFF + iteration),
    )
