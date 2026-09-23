from __future__ import annotations

import ast
from copy import deepcopy
import dataclasses
import pickle
from unittest.mock import patch

import pytest
import torch

from test.test_cute_philox_stream import _config
from test.test_cute_philox_stream import _explicit_offsets
from test.test_cute_philox_stream import _implicit_normal
from test.test_cute_philox_stream import _implicit_uniform
from test.test_cute_philox_stream import _multiple_random
from test.test_cute_philox_stream import _oracle
from test.test_cute_philox_stream import _unsupported_rand4
from test.test_cute_philox_stream import cpu_only  # noqa: F401

import helion
from helion._compiler.rng_utils import philox_rand4x_ref
from helion._compiler.rng_utils import philox_randint_ref
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.ref_mode import RefMode

pytestmark = pytest.mark.usefixtures("cpu_only", "default_policy")


@pytest.fixture
def default_policy(monkeypatch):
    monkeypatch.delenv("HELION_CUTE_RNG_STREAM", raising=False)
    with patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")):
        yield


def _bind(function, args, **settings):
    return helion.kernel(
        function, backend="cute", static_shapes=False, **settings
    )._bind_isolated(args)


def test_backend_environment_explicit_precedence_and_roundtrip(monkeypatch):
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
    for policy in ("word0", "philox4", "auto"):
        monkeypatch.setenv("HELION_CUTE_RNG_STREAM", policy)
        assert helion.Settings(backend="cute").cute_rng_stream == policy
        assert (
            helion.Settings(backend="cute", cute_rng_stream="word0").cute_rng_stream
            == "word0"
        )
    monkeypatch.setenv("HELION_CUTE_RNG_STREAM", "invalid")
    assert (
        helion.Settings(backend="cute", cute_rng_stream="philox4").cute_rng_stream
        == "philox4"
    )
    with pytest.raises(ValueError, match="HELION_CUTE_RNG_STREAM"):
        helion.Settings(backend="cute")


@pytest.mark.parametrize("implementation", ("low_mem_dropout", "low_mem_dropout_bwd"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("shape", ((17,), (3, 129)))
@skipUnlessBackends(["cute"])
def test_default_original_emission_matches_explicit_philox4(
    implementation, dtype, shape
):
    from examples import low_mem_dropout as example

    function = vars(example)[implementation].fn
    args = (0.25, torch.empty(shape, dtype=dtype), -(2**63))
    default = _bind(function, args)
    explicit = _bind(function, args, cute_rng_stream="philox4")
    assert default.settings.cute_rng_stream == "auto"
    assert default.config_spec.cute_rng_packet_enabled
    for packet in (False, True):
        config = _config(8, packet)
        assert default.to_code(config) == explicit.to_code(config)
    for policy in ("auto", "word0", "philox4"):
        bound = _bind(function, args, cute_rng_stream=policy)
        text = bound.format_kernel_decorator(_config(8, None), bound.settings)
        tree = ast.parse(text + "\ndef reproduced(): pass")
        decorator = tree.body[0].decorator_list[0]
        assert (
            next(
                ast.literal_eval(k.value)
                for k in decorator.keywords
                if k.arg == "cute_rng_stream"
            )
            == policy
        )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("policy", (None, "philox4"), ids=("default", "philox4"))
@skipUnlessBackends(["cute"])
def test_default_view_dynamic_seed_and_probability_reference(dtype, policy):
    from examples.low_mem_dropout import low_mem_dropout
    from examples.low_mem_dropout import low_mem_dropout_bwd

    values = torch.linspace(-2, 2, 2 * 33, dtype=dtype)[1::2]
    assert not values.is_contiguous()
    for function in (low_mem_dropout.fn, low_mem_dropout_bwd.fn):
        settings = {} if policy is None else {"cute_rng_stream": policy}
        default = _bind(
            function, (0.25, values, 123), ref_mode=RefMode.EAGER, **settings
        )
        for seed in (-(2**63), -1, 123, 2**32 + 123):
            randoms = torch.tensor(
                [_oracle(seed, i) for i in range(values.numel())]
            ).reshape(values.shape)
            for probability in (
                0.0,
                0.25,
                -0.5,
                1.5,
                float("nan"),
                float("inf"),
                -float("inf"),
            ):
                expected = torch.where(
                    randoms > probability,
                    values.float() * (1.0 / (1 - probability)),
                    0.0,
                ).to(dtype)
                assert torch.equal(default.run_ref(probability, values, seed), expected)
        with (
            patch(
                "torch.empty_like", side_effect=AssertionError("allocation before p=1")
            ),
            pytest.raises(ZeroDivisionError),
        ):
            default.run_ref(1.0, values, 123)


@skipUnlessBackends(["cute"])
def test_original_nonflattenable_view_still_rejects():
    from examples.low_mem_dropout import low_mem_dropout

    values = torch.empty(33, 2).t()
    for policy in ("auto", "philox4", "word0"):
        bound = _bind(
            low_mem_dropout.fn,
            (0.25, values, 123),
            cute_rng_stream=policy,
            ref_mode=RefMode.EAGER,
        )
        with pytest.raises(RuntimeError, match="view size is not compatible"):
            bound.run_ref(0.25, values, 123)


def _integer(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.int32)
    for tile in hl.tile(x.numel()):
        out[tile] = hl.randint([tile], low=-5, high=17, seed=123)
    return out


@pytest.mark.parametrize(
    "function", (_implicit_uniform, _implicit_normal, _unsupported_rand4, _integer)
)
@skipUnlessBackends(["cute"])
def test_default_other_random_apis_keep_exact_word0_emission_and_reference(function):
    args = (torch.empty(17),)
    default = _bind(function, args)
    old = _bind(function, args, cute_rng_stream="word0")
    assert not default.config_spec.cute_rng_packet_enabled
    assert default.to_code(_config(4, None)) == old.to_code(_config(4, None))
    for policy in ("auto", "word0"):
        bound = _bind(function, args, cute_rng_stream=policy, ref_mode=RefMode.EAGER)
        torch.manual_seed(314)
        actual = bound.run_ref(*args)
        if policy == "auto":
            expected = actual
        else:
            assert torch.equal(actual, expected)
    with pytest.raises(helion.exc.BackendUnsupported, match="explicit uniform"):
        _bind(function, args, cute_rng_stream="philox4")


def _mixed(x: torch.Tensor, seed: int):
    uniform = torch.empty_like(x)
    integer = torch.empty_like(x, dtype=torch.int32)
    four = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        uniform[tile] = hl.rand([tile], seed=seed)
        integer[tile] = hl.randint([tile], low=-5, high=17, seed=seed)
        four[tile] = hl.rand4x(seed, hl.tile_index(tile).to(torch.int64))[0]
    return uniform, integer, four


@skipUnlessBackends(["cute"])
def test_mixed_random_apis_use_philox4_only_for_explicit_uniform():
    args = (torch.empty(33), -1)
    bound = _bind(_mixed, args)
    assert bound.config_spec.cute_rng_packet_enabled
    source = bound.to_code(_config(4, False))
    assert "shr.s64 counter" in source
    reference = _bind(_mixed, args, ref_mode=RefMode.EAGER)
    uniform, integer, four = reference.run_ref(*args)
    offsets = torch.arange(33, dtype=torch.int64)
    assert torch.equal(uniform, torch.tensor([_oracle(-1, i) for i in range(33)]))
    assert torch.equal(integer, philox_randint_ref(-1, offsets, -5, 17))
    assert torch.equal(four, philox_rand4x_ref(-1, offsets)[0])


@skipUnlessBackends(["cute"])
def test_default_explicit_tensor_offsets_and_multiple_sites():
    offsets = torch.tensor([-9, -1, 0, 1, 4, 7, 2**32 + 3], dtype=torch.int64)
    args = (torch.tensor([-(2**63)], dtype=torch.int64), offsets)
    default = _bind(_explicit_offsets, args)
    explicit = _bind(_explicit_offsets, args, cute_rng_stream="philox4")
    assert default.to_code(_config(4, True)) == explicit.to_code(_config(4, True))
    ref = _bind(_explicit_offsets, args, ref_mode=RefMode.EAGER)
    assert torch.equal(
        ref.run_ref(*args), torch.tensor([_oracle(-(2**63), i.item()) for i in offsets])
    )
    args = (torch.empty(33), -1, 2**32 + 123)
    default = _bind(_multiple_random, args)
    explicit = _bind(_multiple_random, args, cute_rng_stream="philox4")
    assert default.to_code(_config(4, True)) == explicit.to_code(_config(4, True))


@pytest.mark.parametrize("disabled", (False, True))
@skipUnlessBackends(["cute"])
def test_default_and_opt_in_keep_identical_seed_domains(disabled):
    from test.test_cute_philox_seed_coverage import _uniform_pointwise

    from helion.autotuner.config_generation import ConfigGeneration

    args = (torch.empty(4096, dtype=torch.float16), 123)
    domains = []
    for policy in ("auto", "philox4"):
        bound = _bind(
            _uniform_pointwise,
            args,
            cute_rng_stream=policy,
            disable_autotuner_heuristics=disabled,
        )
        assert bound.config_spec.cute_rng_packet_enabled
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            domains.append(
                [config.config for _, config in generation.seed_flat_config_pairs()]
            )
            indices, _ = generation._key_to_flat_indices["cute_rng_packet"]
            assert generation.flat_spec[indices[0]].search_values() == [False, True]
    assert domains[0] == domains[1]
    assert bool(domains[0]) is not disabled
