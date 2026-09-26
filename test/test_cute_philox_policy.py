from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import dataclasses
import pickle
import struct
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.rng_utils import philox_rand_ref
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.cute.rng import stage_explicit_seed

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.delenv("HELION_CUTE_RNG_STREAM", raising=False)
    initialized = torch.cuda.is_initialized()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with ExitStack() as stack:
        for name, value in (
            ("helion.runtime.kernel.target_device_capability", (10, 0)),
            ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
            ("helion.runtime.get_num_sm", 148),
            ("helion._compat._is_hip", False),
        ):
            stack.enter_context(patch(name, return_value=value))
        stack.enter_context(_mock_cuda_unavailable())
        stack.enter_context(
            patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden"))
        )
        stack.enter_context(
            patch(
                "cutlass.cute.compile", side_effect=AssertionError("native forbidden")
            )
        )
        yield
    torch.set_num_threads(previous_threads)
    assert torch.cuda.is_initialized() == initialized


def _oracle(seed: int, offset: int) -> float:
    def f32(value: float) -> float:
        return struct.unpack("<f", struct.pack("<f", value))[0]

    counter = (offset // 4) & ((1 << 64) - 1)
    c0, c1, c2, c3 = counter & 0xFFFFFFFF, counter >> 32, 0, 0
    k0, k1 = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF
    for _ in range(10):
        a, b = 0xCD9E8D57 * c2, 0xD2511F53 * c0
        c0, c1, c2, c3 = (
            ((a >> 32) ^ c1 ^ k0) & 0xFFFFFFFF,
            a & 0xFFFFFFFF,
            ((b >> 32) ^ c3 ^ k1) & 0xFFFFFFFF,
            b & 0xFFFFFFFF,
        )
        k0, k1 = (k0 + 0x9E3779B9) & 0xFFFFFFFF, (k1 + 0xBB67AE85) & 0xFFFFFFFF
    word = (c0, c1, c2, c3)[offset % 4]
    signed = word - (1 << 32) if word & (1 << 31) else word
    return f32(
        f32(float(-signed - 1 if signed < 0 else signed)) * f32(4.6566127342e-10)
    )


def test_effective_default_and_settings_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for backend in ("cute", "triton", "pallas", "metal"):
        settings = helion.Settings(backend=backend)
        assert settings.cute_rng_stream == ("auto" if backend == "cute" else "word0")
        assert helion.Settings(**settings.to_dict()) == settings
        assert dataclasses.replace(settings) == settings
        assert deepcopy(settings) == settings
        assert pickle.loads(pickle.dumps(settings)) == settings
    monkeypatch.setenv("HELION_BACKEND", "cute")
    assert helion.Settings().cute_rng_stream == "auto"
    assert helion.Settings(backend="triton").cute_rng_stream == "word0"
    for policy in ("auto", "word0", "philox4"):
        monkeypatch.setenv("HELION_CUTE_RNG_STREAM", policy)
        assert helion.Settings(backend="cute").cute_rng_stream == policy
        assert (
            helion.Settings(backend="cute", cute_rng_stream="word0").cute_rng_stream
            == "word0"
        )


@pytest.mark.parametrize("name", ("low_mem_dropout", "low_mem_dropout_bwd"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@skipUnlessBackends(["cute"])
def test_original_dropout_reference_mapping(name: str, dtype: torch.dtype) -> None:
    from examples import low_mem_dropout as example

    function = vars(example)[name].fn
    values = torch.linspace(-2, 2, 33, dtype=dtype)
    auto = helion.kernel(
        function, backend="cute", static_shapes=False, ref_mode=helion.RefMode.EAGER
    )._bind_isolated((0.25, values, 123))
    legacy = helion.kernel(
        function,
        backend="cute",
        static_shapes=False,
        cute_rng_stream="word0",
        ref_mode=helion.RefMode.EAGER,
    )._bind_isolated((0.25, values, 123))
    for seed in (-(2**63), -1, 123, 2**32 + 123):
        expected = torch.where(
            torch.tensor([_oracle(seed, i) for i in range(33)]) > 0.25,
            values.float() * (4 / 3),
            0.0,
        ).to(dtype)
        assert torch.equal(auto.run_ref(0.25, values, seed), expected)
        old = torch.where(
            philox_rand_ref(seed, torch.arange(33)) > 0.25,
            values.float() * (4 / 3),
            0.0,
        ).to(dtype)
        assert torch.equal(legacy.run_ref(0.25, values, seed), old)
    with (
        patch("torch.empty_like", side_effect=AssertionError("allocation before p=1")),
        pytest.raises(ZeroDivisionError),
    ):
        auto.run_ref(1.0, values, 123)


@pytest.mark.parametrize("name", ("low_mem_dropout", "low_mem_dropout_bwd"))
@pytest.mark.parametrize(
    "width,layout", ((1, "blocked"), (4, "blocked"), (8, "strided"))
)
@skipUnlessBackends(["cute"])
def test_original_codegen_default_matches_explicit_mapping(
    name: str, width: int, layout: str
) -> None:
    from examples import low_mem_dropout as example

    function = vars(example)[name].fn
    args = (0.25, torch.empty(1025), -(2**63))
    config = helion.Config(
        block_sizes=[128 * width],
        num_warps=4,
        num_threads=128,
        indexing="pointer",
        cute_vector_widths=[width],
        cute_lane_layouts=[layout],
    )
    sources = []
    for policy in ("auto", "philox4", "word0"):
        bound = helion.kernel(
            function, backend="cute", static_shapes=False, cute_rng_stream=policy
        )._bind_isolated(args)
        source = bound.to_code(config)
        assert "cute_rng_stream=" + repr(policy) in bound.format_kernel_decorator(
            config, bound.settings
        )
        sources.append(source)
    assert sources[0] == sources[1]
    assert (
        "shr.s64 counter" in sources[0] and "_cute_stage_explicit_seed(" in sources[0]
    )
    assert "shr.s64 counter" not in sources[2]


def _offsets(offsets: torch.Tensor, seed: int) -> torch.Tensor:
    out = torch.empty_like(offsets, dtype=torch.float32)
    for tile in hl.tile(offsets.size(0)):
        out[tile] = hl.rand([tile], seed, offsets=offsets[tile])
    return out


def _other_random(x: torch.Tensor, seed: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        a, b, c, d = hl.rand4x(seed=seed, offsets=tile.index)
        out[tile] = a + b + c + d + hl.randint([tile], 0, 8, seed)
    return out


@skipUnlessBackends(["cute"])
def test_signed_offsets_and_other_apis_keep_their_policy() -> None:
    offsets = torch.tensor(
        [-(2**63), -(2**32), -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 2**32 + 3, 2**63 - 1]
    )
    bound = helion.kernel(
        _offsets, backend="cute", static_shapes=False, ref_mode=helion.RefMode.EAGER
    )._bind_isolated((offsets, 123))
    assert torch.equal(
        bound.run_ref(offsets, 123),
        torch.tensor([_oracle(123, i) for i in offsets.tolist()]),
    )
    x = torch.empty(17)
    outputs = []
    for policy in ("auto", "word0"):
        bound = helion.kernel(
            _other_random,
            backend="cute",
            cute_rng_stream=policy,
            ref_mode=helion.RefMode.EAGER,
        )._bind_isolated((x, 123))
        outputs.append(bound.run_ref(x, 123))
    assert torch.equal(*outputs)
    with pytest.raises(helion.exc.BackendUnsupported):
        helion.kernel(
            _other_random, backend="cute", cute_rng_stream="philox4"
        )._bind_isolated((x, 123))


def test_staged_seed_is_fresh_and_signed() -> None:
    x = torch.empty(1)
    first = stage_explicit_seed(-(2**63), x)
    second = stage_explicit_seed(-(2**63), x)
    assert first.dtype == second.dtype == torch.int64
    assert first.shape == second.shape == (1,)
    assert first.data_ptr() != second.data_ptr()
    assert first.item() == second.item() == -(2**63)
