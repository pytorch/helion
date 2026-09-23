from __future__ import annotations

from unittest.mock import patch

from examples.low_mem_dropout import low_mem_dropout
from examples.low_mem_dropout import low_mem_dropout_bwd
import pytest
import torch

import helion
from helion._compiler.cute import uniform_comparison
from helion._testing import skipUnlessBackends
import helion.language as hl

CUDA_DEVICE = "cuda"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    skipUnlessBackends(["cute"]),
]


def _config(width=4):
    return helion.Config(
        block_sizes=[128 * width],
        num_threads=[128],
        cute_vector_widths=[width],
        cute_lane_layouts=["blocked"],
        cute_cluster_n=1,
    )


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


@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("length", (262144, 4194304, 33554432))
def test_original_forward_backward_exact(backward, length):
    _dropout(backward, length, torch.float32, 4)


@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_half_precision_tail_and_poisoned_graph(backward, dtype):
    _dropout(backward, 517, dtype, 8)


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
