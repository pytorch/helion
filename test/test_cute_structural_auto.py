from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Sequence
import copy
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_shared_rhs_grouped import _target
from test.test_cute_structural_compatibility import _pointwise

import helion
from helion._testing import skipUnlessBackends
from helion.runtime.cute_structural_config import CuteStructuralConfig
from helion.runtime.cute_structural_config import StructuralPolicyError
from helion.runtime.cute_structural_policy import CUTE_STRUCTURAL_SETTINGS
from helion.runtime.cute_structural_policy import CuteStructuralPolicy
from helion.runtime.kernel import Kernel
from helion.runtime.settings import Settings

ALL_PASSES = CuteStructuralPolicy(**dict.fromkeys(CUTE_STRUCTURAL_SETTINGS, True))


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in CUTE_STRUCTURAL_SETTINGS:
        monkeypatch.delenv(f"HELION_{name.upper()}", raising=False)
    monkeypatch.delenv("HELION_BACKEND", raising=False)
    monkeypatch.setenv("HELION_AUTOTUNE_RANDOM_SEED", "0")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _target(),
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("entrypoint", ["function", "decorator", "constructor"])
@skipUnlessBackends(["cute"])
def test_ordinary_default_selects_before_binding(entrypoint: str) -> None:
    settings = Settings(backend="cute")
    original = copy.deepcopy(settings.to_dict())
    if entrypoint == "function":
        kernel = helion.kernel(_pointwise, settings=settings)
    elif entrypoint == "decorator":
        kernel = helion.kernel(settings=settings)(_pointwise)
    else:
        kernel = Kernel(_pointwise, settings=settings)
    assert kernel.cute_structural_policy == ALL_PASSES
    assert kernel.settings.get_cute_structural_policy() == ALL_PASSES
    assert settings.to_dict() == original
    assert settings.get_cute_structural_policy() == CuteStructuralPolicy()
    bound = kernel.bind((torch.empty(37),))
    assert bound._structural_policy == ALL_PASSES
    legacy = helion.kernel(
        _pointwise, settings=settings, cute_structural_policy=None
    ).bind((torch.empty(37),))
    assert bound.to_code(bound.config_spec.default_config()) == legacy.to_code(
        legacy.config_spec.default_config()
    )


@pytest.mark.parametrize("policy_request", ["omitted", None, "auto"])
@pytest.mark.parametrize("entrypoint", ["ordinary", "aot"])
def test_public_request_boundary(policy_request: str | None, entrypoint: str) -> None:
    kwargs: dict[str, Any] = {"backend": "cute"}
    if policy_request != "omitted":
        kwargs["cute_structural_policy"] = policy_request
    factory = helion.kernel if entrypoint == "ordinary" else helion.aot_kernel
    kernel = factory(_pointwise, **kwargs)
    enabled = policy_request == "auto" or (
        policy_request == "omitted" and entrypoint == "ordinary"
    )
    assert kernel.cute_structural_policy == (ALL_PASSES if enabled else None)
    if entrypoint == "aot":
        assert kernel._key_fn.structural_policy == kernel.cute_structural_policy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"autotune_cache": "AOTAutotuneCache"},
        {"autotune_effort": "none"},
        {"ref_mode": "eager"},
    ],
)
def test_omitted_nonsearch_paths_remain_legacy(kwargs: dict[str, Any]) -> None:
    settings = Settings(backend="cute", **kwargs)
    legacy = helion.kernel(_pointwise, settings=settings)
    assert legacy.cute_structural_policy is None
    assert legacy.settings is settings
    explicit = helion.kernel(
        _pointwise, settings=settings, cute_structural_policy="auto"
    )
    assert explicit.cute_structural_policy == ALL_PASSES
    assert explicit.settings.autotune_effort == settings.autotune_effort
    assert explicit.settings.ref_mode == settings.ref_mode


def test_other_backend_does_not_receive_a_cute_policy() -> None:
    settings = Settings(backend="triton")
    kernel = helion.kernel(_pointwise, settings=settings)
    assert kernel.cute_structural_policy is None
    assert kernel.settings is settings
    with pytest.raises(StructuralPolicyError, match="backend"):
        helion.kernel(_pointwise, settings=settings, cute_structural_policy="auto")


@pytest.mark.parametrize("name", CUTE_STRUCTURAL_SETTINGS)
@pytest.mark.parametrize("origin", ["explicit", "environment", "blank", "copy"])
def test_auto_retains_captured_vetoes(
    name: str, origin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = f"HELION_{name.upper()}"
    if origin in ("environment", "blank"):
        monkeypatch.setenv(environment, "0" if origin == "environment" else " ")
    settings = Settings(
        backend="cute", **({name: False} if origin in ("explicit", "copy") else {})
    )
    if origin == "copy":
        settings = settings.copy(autotune_effort="quick")
    before = copy.deepcopy(settings.to_dict())
    monkeypatch.setenv(environment, "1")
    kernel = helion.kernel(_pointwise, settings=settings)
    policy = kernel.cute_structural_policy
    assert policy is not None
    assert getattr(policy, name) is (origin == "blank")
    assert all(
        getattr(policy, other) for other in CUTE_STRUCTURAL_SETTINGS if other != name
    )
    assert settings.to_dict() == before


def test_all_false_automatic_policy_keeps_a_versioned_namespace() -> None:
    settings = Settings(
        backend="cute", **dict.fromkeys(CUTE_STRUCTURAL_SETTINGS, False)
    )
    automatic = helion.kernel(_pointwise, settings=settings)
    legacy = helion.kernel(_pointwise, settings=settings, cute_structural_policy=None)
    assert automatic.cute_structural_policy == CuteStructuralPolicy()
    assert legacy.cute_structural_policy is None
    assert automatic.settings.to_dict() == legacy.settings.to_dict()


@pytest.mark.parametrize("policy_request", ["omitted", None, "auto"])
@pytest.mark.parametrize("location", ["config", "configs", "seed", "override"])
def test_recorded_policy_wins_at_every_declaration_boundary(
    policy_request: str | None, location: str
) -> None:
    recorded = CuteStructuralConfig(
        helion.Config(block_sizes=[16]), CuteStructuralPolicy()
    )
    kwargs: dict[str, Any] = {"backend": "cute"}
    if policy_request != "omitted":
        kwargs["cute_structural_policy"] = policy_request
    if location == "config":
        kwargs["config"] = recorded
    elif location == "configs":
        kwargs["configs"] = [recorded]
    elif location == "seed":
        kwargs["autotune_seed_configs"] = [recorded]
    else:
        kwargs["autotune_config_overrides"] = recorded
    kernel = helion.kernel(_pointwise, **kwargs)
    assert kernel.cute_structural_policy == recorded.policy
    assert kernel.settings.get_cute_structural_policy() == recorded.policy
    kwargs["cute_region_fission"] = True
    with pytest.raises(StructuralPolicyError, match="conflicts"):
        helion.kernel(_pointwise, **kwargs)


class _ForbiddenSeeds(Sequence):
    def __len__(self) -> int:
        raise AssertionError("custom seed length must not be read")

    def __getitem__(self, index: int) -> object:
        raise AssertionError("custom seed item must not be read")

    def __iter__(self) -> Iterator:
        raise AssertionError("custom seeds must not be consumed")


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize(
    "declaration",
    ["config", "empty_config", "seed", "opaque_seeds", "override", "filter", "tuner"],
)
def test_unversioned_declarations_keep_legacy_axes(
    declaration: str, force: bool
) -> None:
    kwargs: dict[str, Any] = {"backend": "cute", "force_autotune": force}
    config = helion.Config(block_sizes=[16])
    if declaration in ("config", "empty_config"):
        kwargs["config"] = config if declaration == "config" else helion.Config()
    elif declaration == "seed":
        kwargs["autotune_seed_configs"] = config
    elif declaration == "opaque_seeds":
        kwargs["autotune_seed_configs"] = _ForbiddenSeeds()
    elif declaration == "override":
        kwargs["autotune_config_overrides"] = {"block_sizes": [16]}
    elif declaration == "filter":
        kwargs["autotune_config_filter"] = lambda config: True
    else:
        kwargs["autotuner_fn"] = lambda bound, args, **kwargs: None
    kernel = helion.kernel(_pointwise, cute_structural_policy="auto", **kwargs)
    assert kernel.cute_structural_policy is None


@pytest.mark.parametrize("seeds", [None, [], ()])
def test_empty_builtin_declarations_allow_auto(seeds: object) -> None:
    kernel = helion.kernel(
        _pointwise,
        backend="cute",
        autotune_seed_configs=seeds,
        autotune_config_overrides={},
    )
    assert kernel.cute_structural_policy == ALL_PASSES


class _OnePassConfigs(Sequence[helion.Config]):
    def __init__(self) -> None:
        self.iterations = 0
        self.config = helion.Config(block_sizes=[16])

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> helion.Config:
        if index != 0:
            raise IndexError(index)
        return self.config

    def __iter__(self) -> Iterator[helion.Config]:
        self.iterations += 1
        assert self.iterations == 1
        yield self.config


def test_custom_configs_are_only_consumed_once() -> None:
    configs = _OnePassConfigs()
    kernel = helion.kernel(
        _pointwise, configs=configs, backend="cute", cute_structural_policy="auto"
    )
    assert kernel.cute_structural_policy is None
    assert configs.iterations == 1
    assert kernel.configs == [configs.config]


@pytest.mark.parametrize("policy_request", [False, 1, "invalid", {}])
def test_invalid_request_fails_at_construction(policy_request: Any) -> None:
    with pytest.raises(TypeError, match="cute_structural_policy"):
        helion.kernel(_pointwise, backend="cute", cute_structural_policy=policy_request)
