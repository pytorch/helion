from __future__ import annotations

from typing import TYPE_CHECKING

from examples.low_mem_dropout import low_mem_dropout
from examples.low_mem_dropout import low_mem_dropout_bwd
import pytest
import torch

from test.test_cute_philox_stream import _oracle

import helion
from helion._compiler.rng_utils import philox_rand_ref
from helion._testing import skipUnlessBackends
import helion.language as hl

CUDA_DEVICE = "cuda"

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    skipUnlessBackends(["cute"]),
]


def _config(width: int, layout: str = "blocked") -> helion.Config:
    return helion.Config(
        block_sizes=[256 * width],
        num_threads=[128],
        cute_vector_widths=[width],
        cute_lane_layouts=[layout],
        cute_cluster_n=1,
    )


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    integer = torch.int32 if actual.element_size() == 4 else torch.int16
    assert torch.equal(actual[~nan].view(integer), expected[~nan].view(integer))


def _mutating_graph(
    invoke: Callable[[], torch.Tensor],
    change_inputs: Callable[[int], object],
    reference: Callable[[], torch.Tensor],
) -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = invoke()
    for iteration in range(3):
        change_inputs(iteration)
        expected = reference()
        graph.replay()
        torch.cuda.synchronize()
        _assert_exact(actual, expected)


@pytest.mark.parametrize("width", (4, 8))
@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.float16, torch.bfloat16), ids=("f32", "f16", "bf16")
)
@pytest.mark.parametrize("backward", (False, True))
def test_dropout_packets_exact(width: int, dtype: torch.dtype, backward: bool) -> None:
    function = low_mem_dropout_bwd if backward else low_mem_dropout
    kernel = helion.kernel(
        function.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_rng_stream="word0",
    )
    config = _config(width)
    for length in (8192, 8221):
        _dropout_case(kernel, config, length, dtype)


def _dropout_case(
    kernel: helion.Kernel, config: helion.Config, length: int, dtype: torch.dtype
) -> None:
    data = torch.linspace(-2, 2, length, device=CUDA_DEVICE, dtype=dtype)
    data[:5] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")],
        device=CUDA_DEVICE,
        dtype=dtype,
    )
    bound = kernel._bind_isolated((0.25, data, 123))
    source = bound.to_code(config)
    assert "_helion_affine_load" in source
    assert "_helion_affine_store" in source
    bound.set_config(config)

    def expected(seed: int, probability: float) -> torch.Tensor:
        random = philox_rand_ref(seed, torch.arange(length, dtype=torch.int64)).to(
            device=CUDA_DEVICE
        )
        return torch.where(
            random > probability,
            data.float() * (1 / (1 - probability)),
            0.0,
        ).to(dtype)

    for seed, probability in ((123, 0.25), (-1, 0.6), (0x1234567812345678, 0.0)):
        _assert_exact(bound(probability, data, seed), expected(seed, probability))
    replacement = data.clone()
    _assert_exact(bound(0.25, replacement, 123), expected(123, 0.25))

    def mutate(iteration: int) -> None:
        data.copy_(
            torch.linspace(
                -1 - iteration, 3 + iteration, length, device=CUDA_DEVICE, dtype=dtype
            )
        )

    _mutating_graph(lambda: bound(0.25, data, 123), mutate, lambda: expected(123, 0.25))


@pytest.mark.parametrize("policy", (None, "philox4"), ids=("default", "philox4"))
@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_philox4_masks_config_independent_and_poisoned_replay(
    policy: str | None, backward: bool, dtype: torch.dtype
) -> None:
    function = low_mem_dropout_bwd if backward else low_mem_dropout
    if policy is None:
        kernel = helion.kernel(
            function.fn, backend="cute", static_shapes=False, autotune_effort="none"
        )
    else:
        kernel = helion.kernel(
            function.fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
            cute_rng_stream=policy,
        )
    data = torch.linspace(-2, 2, 517, device=CUDA_DEVICE, dtype=dtype)

    def special_values() -> None:
        data[:5] = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan")],
            device=data.device,
            dtype=dtype,
        )

    special_values()
    # Independent integer Philox oracle, shared by all layouts and packet policies.
    randoms = {
        seed: torch.tensor(
            [_oracle(seed, i) for i in range(data.numel())], device=CUDA_DEVICE
        )
        for seed in (123, -(2**63))
    }

    def expected(seed: int, probability: float) -> torch.Tensor:
        return torch.where(
            randoms[seed] > probability,
            data.float() * (1 / (1 - probability)),
            0.0,
        ).to(dtype)

    for width, layout, packet in (
        (4, "blocked", False),
        (8, "blocked", True),
        (8, "strided", True),
    ):
        config = _config(width, layout)
        config.config["cute_rng_packet"] = packet
        bound = kernel._bind_isolated((0.25, data, 123))
        assert bound.settings.cute_rng_stream == ("auto" if policy is None else policy)
        bound.set_config(config)
        for seed in randoms:
            for probability in (
                0.0,
                0.25,
                0.6,
                float("nan"),
                float("inf"),
                -float("inf"),
            ):
                _assert_exact(
                    bound(probability, data, seed), expected(seed, probability)
                )
        with pytest.raises(ZeroDivisionError):
            bound(1.0, data, 123)
        first = bound(0.25, data, 123)
        second = bound(0.25, data, 123)
        assert first.data_ptr() != second.data_ptr()
        first.fill_(float("nan"))
        _assert_exact(second, expected(123, 0.25))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                bound(0.25, data, 123)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = bound(0.25, data, 123)
        for iteration in range(3):
            data.fill_(iteration + 1.0)
            special_values()
            result.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            _assert_exact(result, expected(123, 0.25))


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _twice_assembly(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.numel()):
        value = hl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $1;",
            "=f,f",
            [x[tile]],
            dtype=torch.float32,
            is_pure=True,
            pack=1,
        )
        out[tile] = value
    return out


@pytest.mark.parametrize("width", (4, 8))
@pytest.mark.parametrize("layout", ("blocked", "strided"))
def test_generic_packet_assembly_and_fallback(width: int, layout: str) -> None:
    config = _config(width, layout)
    for mode in ("aligned", "offset", "strided", "alias"):
        _generic_case(config, mode)


def _generic_case(config: helion.Config, mode: str) -> None:
    length = 4093
    storage = torch.linspace(-2, 2, 2 * length + 2, device=CUDA_DEVICE)
    data = (
        storage[1 : length + 1]
        if mode == "offset"
        else storage[::2][:length]
        if mode == "strided"
        else storage[:length]
    )
    output = data if mode == "alias" else torch.empty(length, device=CUDA_DEVICE)
    bound = _twice_assembly._bind_isolated((data, output))
    source = bound.to_code(config)
    assert ("_helion_affine_load" in source) is (mode == "aligned")
    bound.set_config(config)
    expected = data * 2
    _assert_exact(bound(data, output), expected)
    # Refresh before each call, including aliases, so the exact reference
    # always observes the input before this invocation writes its result.
    for iteration in range(2):
        data.fill_(float(iteration) - 0.75)
        expected = data * 2
        _assert_exact(bound(data, output), expected)
    _mutating_graph(
        lambda: bound(data, output),
        lambda iteration: data.fill_(float(iteration) + 0.5),
        lambda: data * 2,
    )


def _arbitrary_offset_uniform(offsets: torch.Tensor, seed: int) -> torch.Tensor:
    out = torch.empty_like(offsets, dtype=torch.float32)
    for tile in hl.tile(offsets.numel()):
        out[tile] = hl.rand([], seed=seed, offsets=offsets[tile])
    return out


@pytest.mark.parametrize("width,layout", ((4, "blocked"), (8, "strided")))
def test_default_philox4_arbitrary_offsets(width: int, layout: str) -> None:
    # Include every counter word, noncontiguous offsets, negative offsets,
    # signed boundaries and high counter bits. The input has a vector tail.
    values = [3 * index + 1 for index in range(257)]
    values[:13] = [
        -(2**63),
        -(2**32) - 1,
        -9,
        -1,
        0,
        1,
        2,
        3,
        4,
        2**31 - 1,
        2**32 + 3,
        2**40 + 1,
        2**63 - 1,
    ]
    patterns = [values, list(reversed(values))]
    offsets = torch.tensor(values, device=CUDA_DEVICE, dtype=torch.int64)
    kernel = helion.kernel(
        _arbitrary_offset_uniform,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((offsets, -1))
    assert bound.settings.cute_rng_stream == "auto"
    bound.set_config(_config(width, layout))
    for pattern in patterns:
        offsets.copy_(torch.tensor(pattern, device=CUDA_DEVICE, dtype=torch.int64))
        for seed in (-1, 2**32 + 123):
            expected = torch.tensor(
                [_oracle(seed, offset) for offset in pattern], device=CUDA_DEVICE
            )
            first = bound(offsets, seed)
            second = bound(offsets, seed)
            _assert_exact(first, expected)
            _assert_exact(second, expected)
            assert first.data_ptr() != second.data_ptr()
            first.fill_(float("nan"))
            _assert_exact(second, expected)

    current = [patterns[-1]]

    def mutate(iteration: int) -> None:
        current[0] = patterns[iteration % len(patterns)]
        offsets.copy_(torch.tensor(current[0], device=CUDA_DEVICE, dtype=torch.int64))

    _mutating_graph(
        lambda: bound(offsets, -1),
        mutate,
        lambda: torch.tensor(
            [_oracle(-1, offset) for offset in current[0]], device=CUDA_DEVICE
        ),
    )
