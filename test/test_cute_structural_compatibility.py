from __future__ import annotations

import ast
from collections.abc import Iterator
from collections.abc import Sequence
import copy
import dataclasses
import hashlib
import json
import pickle
import random
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import checked_initial_population
from test.test_cute_full_slice_matmul import _matmul
from test.test_cute_fuse_mm_accumulation import _cpu_target as _split_k_target
from test.test_cute_repro_settings import _case
from test.test_cute_repro_settings import _host
from test.test_cute_repro_settings import _replay
from test.test_cute_shared_rhs_grouped import _target
from test.test_cute_split_k_workspace import _args as _split_k_args
from test.test_cute_split_k_workspace import _config as _split_k_config
from test.test_cute_split_k_workspace import _embedded_sources
from test.test_cute_split_k_workspace import _fp32_bias

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import skipUnlessBackends
from helion.autotuner.base_search import normalize_autotune_seed_configs
from helion.autotuner.pattern_search import PatternSearch
import helion.language as hl
from helion.runtime.cute_structural_config import CuteStructuralConfig
from helion.runtime.cute_structural_config import StructuralPolicyError
from helion.runtime.cute_structural_config import resolve_cute_structural_policy
from helion.runtime.cute_structural_policy import CUTE_STRUCTURAL_SETTINGS
from helion.runtime.cute_structural_policy import CuteStructuralPolicy
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path

    from helion.runtime.kernel import BoundKernel


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
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield
    torch.set_num_threads(previous_threads)


def _pointwise(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


def _pointwise_bound(policy: CuteStructuralPolicy | None) -> BoundKernel[Any]:
    return helion.kernel(
        _pointwise,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_structural_policy=policy,
    )._bind_isolated((torch.empty(37),))


def test_envelope_is_a_detached_versioned_round_trip() -> None:
    settings_fields = dataclasses.fields(Settings)
    config = helion.Config.from_dict(
        {"block_sizes": [16, 32], "loop_orders": [[0, 1]], "num_warps": None}
    )
    policy = CuteStructuralPolicy(cute_full_slice_matmul_tiling=True)
    envelope = CuteStructuralConfig(config, policy)
    saved = envelope.to_json()
    config.block_sizes[0] = 8
    config.loop_orders[0][0] = 1
    for result in (
        envelope,
        copy.deepcopy(envelope),
        pickle.loads(pickle.dumps(envelope)),
        CuteStructuralConfig.from_json(saved),
        eval(repr(envelope), {"helion": helion}),
    ):
        assert result.to_json() == saved
        assert result.identity() == envelope.identity()
        assert result.config.config["num_warps"] is None
        unpacked = result.config
        unpacked.block_sizes[0] = 4
        unpacked.loop_orders[0][0] = 1
        assert result.config.block_sizes == [16, 32]
        assert result.config.loop_orders == [[0, 1]]
    assert dataclasses.fields(Settings) == settings_fields


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "other"},
        {"version": 2},
        {"version": True},
        {"unknown": 1},
        {"config": [16]},
        {"policy": {}},
    ],
)
def test_bad_envelope_never_becomes_config_knobs(change: dict[str, object]) -> None:
    payload = CuteStructuralConfig(helion.Config(), CuteStructuralPolicy()).to_dict()
    with pytest.raises(StructuralPolicyError):
        CuteStructuralConfig.from_dict(payload | change)


@pytest.mark.parametrize("name", CUTE_STRUCTURAL_SETTINGS)
@pytest.mark.parametrize("origin", ["default", "environment", "explicit"])
@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_auto_resolution_honors_captured_origins(
    name: str, origin: str, enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if origin == "environment":
        monkeypatch.setenv(f"HELION_{name.upper()}", str(int(enabled)))
    kwargs = {name: enabled} if origin == "explicit" else {}
    settings = Settings(backend="cute", **kwargs)
    snapshot = copy.deepcopy(settings.to_dict())
    monkeypatch.setenv(f"HELION_{name.upper()}", str(int(not enabled)))
    automatic = CuteStructuralPolicy(**dict.fromkeys(CUTE_STRUCTURAL_SETTINGS, True))
    resolved = resolve_cute_structural_policy(settings, automatic=automatic)
    assert getattr(resolved, name) is (True if origin == "default" else enabled)
    for other in CUTE_STRUCTURAL_SETTINGS:
        if other != name:
            assert getattr(resolved, other)
    assert settings.to_dict() == snapshot


@pytest.mark.parametrize(
    "kind", ["config", "configs", "seed", "seeds", "override", "empty_override"]
)
def test_legacy_positional_inputs_block_an_automatic_policy(kind: str) -> None:
    settings = Settings(backend="cute")
    config = helion.Config(block_sizes=[16, 32])
    kwargs: dict[str, Any] = {}
    if kind in ("config", "configs"):
        kwargs["configs"] = [config] * (1 if kind == "config" else 2)
    elif kind in ("seed", "seeds"):
        settings.autotune_seed_configs = config if kind == "seed" else [config]
    elif kind == "override":
        settings.autotune_config_overrides = {"block_sizes": [16, 32]}
    else:
        kwargs["overrides"] = {}
    automatic = CuteStructuralPolicy(**dict.fromkeys(CUTE_STRUCTURAL_SETTINGS, True))
    assert (
        resolve_cute_structural_policy(settings, automatic=automatic, **kwargs)
        == CuteStructuralPolicy()
    )


def test_recorded_policy_conflicts_fail_before_binding() -> None:
    enabled = CuteStructuralPolicy(cute_full_slice_matmul_tiling=True)
    disabled = CuteStructuralPolicy()
    record = CuteStructuralConfig(helion.Config(block_sizes=[16, 32, 16]), enabled)
    with patch(
        "helion.runtime.kernel.CompileEnvironment",
        side_effect=AssertionError("must fail before tracing"),
    ):
        with pytest.raises(StructuralPolicyError, match="conflicts"):
            helion.kernel(
                _matmul,
                config=record,
                backend="cute",
                cute_full_slice_matmul_tiling=False,
            )
        with pytest.raises(StructuralPolicyError, match="different"):
            helion.kernel(
                _matmul, config=record, backend="cute", cute_structural_policy=disabled
            )
        with pytest.raises(StructuralPolicyError, match="different"):
            helion.kernel(
                _matmul,
                configs=[record, CuteStructuralConfig(helion.Config(), disabled)],
                backend="cute",
            )
        with pytest.raises(StructuralPolicyError, match="backend"):
            helion.kernel(_matmul, config=record, backend="triton")


@pytest.mark.parametrize("name", CUTE_STRUCTURAL_SETTINGS)
@pytest.mark.parametrize("enabled", [False, True])
def test_recorded_policy_rejects_opposite_captured_environment(
    name: str, enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(f"HELION_{name.upper()}", str(int(enabled)))
    settings = Settings(backend="cute")
    monkeypatch.setenv(f"HELION_{name.upper()}", str(int(not enabled)))
    with pytest.raises(StructuralPolicyError, match=name):
        resolve_cute_structural_policy(
            settings, requested=CuteStructuralPolicy(**{name: not enabled})
        )
    assert resolve_cute_structural_policy(
        settings, requested=CuteStructuralPolicy(**{name: enabled})
    ) == CuteStructuralPolicy(**{name: enabled})


class _ObservedConfigs(Sequence[helion.Config | CuteStructuralConfig]):
    def __init__(self, config: helion.Config | CuteStructuralConfig) -> None:
        self.config = config
        self.iterations = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> helion.Config | CuteStructuralConfig:
        if index != 0:
            raise IndexError(index)
        return self.config

    def __iter__(self) -> Iterator[helion.Config | CuteStructuralConfig]:
        self.iterations += 1
        yield self.config


def test_default_off_path_preserves_settings_config_and_custom_seed_identity() -> None:
    config = helion.Config(block_sizes=[16])
    configs = _ObservedConfigs(config)
    seeds = _ObservedConfigs(config)
    settings = Settings(backend="cute", autotune_seed_configs=seeds)
    kernel = helion.kernel(_pointwise, configs=configs, settings=settings)
    assert kernel.settings is settings
    assert kernel.configs[0] is config
    assert kernel.cute_structural_policy is None
    assert configs.iterations == 1
    assert seeds.iterations == 0
    assert normalize_autotune_seed_configs(settings)[0] is config
    assert seeds.iterations == 1
    settings.cute_full_slice_matmul_tiling = True
    assert kernel.settings is settings


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("where", ["requested", "config"])
def test_new_policy_snapshots_custom_seeds_once(recorded: bool, where: str) -> None:
    policy = CuteStructuralPolicy()
    config = helion.Config(block_sizes=[16])
    record = CuteStructuralConfig(config, policy)
    seeds = _ObservedConfigs(record if recorded else config)
    settings = Settings(backend="cute", autotune_seed_configs=seeds)
    kwargs = (
        {"cute_structural_policy": policy}
        if where == "requested"
        else {"config": record}
    )
    kernel = helion.kernel(_pointwise, settings=settings, **kwargs)
    assert seeds.iterations == 1
    assert settings.autotune_seed_configs is seeds
    assert kernel.settings is not settings
    assert normalize_autotune_seed_configs(kernel.settings) == (config,)
    assert normalize_autotune_seed_configs(kernel.settings) == (config,)
    assert seeds.iterations == 1


@pytest.mark.parametrize(
    "where", ["config", "custom_configs", "seed", "seeds", "overrides", "requested"]
)
@skipUnlessBackends(["cute"])
def test_record_selects_policy_before_real_matmul_binding(where: str) -> None:
    policy = CuteStructuralPolicy(cute_full_slice_matmul_tiling=True)
    config = helion.Config(block_sizes=[16, 32, 16])
    record = CuteStructuralConfig(config, policy)
    settings = Settings(backend="cute", static_shapes=True, autotune_effort="none")
    kwargs: dict[str, Any] = {}
    if where == "config":
        kwargs["config"] = record
    elif where == "custom_configs":
        kwargs["configs"] = _ObservedConfigs(record)
    elif where in ("seed", "seeds"):
        settings.autotune_seed_configs = record if where == "seed" else [record]
    elif where == "overrides":
        # The envelope is consumed before the ordinary override dictionary is used.
        settings.autotune_config_overrides = record  # type: ignore[assignment]
    else:
        kwargs["cute_structural_policy"] = policy
    kernel = helion.kernel(_matmul, settings=settings, **kwargs)
    args = (
        torch.empty((32, 16), dtype=torch.float16),
        torch.empty((16, 64), dtype=torch.float16),
    )
    bound = kernel._bind_isolated(args)
    assert kernel.cute_structural_policy == policy
    assert kernel.settings is not settings
    assert settings.cute_full_slice_matmul_tiling is False
    assert "_helion_fullslice_k" in _host(bound)
    assert len(bound.config_spec.default_config().block_sizes) == 3
    assert "def _helion" in bound.to_code(config)
    if where in ("seed", "seeds"):
        assert normalize_autotune_seed_configs(kernel.settings) == (config,)
    if where == "overrides":
        assert kernel.settings.autotune_config_overrides == config.config
    config.block_sizes[0] = 8
    assert record.config.block_sizes[0] == 16


@pytest.mark.parametrize("name", CUTE_STRUCTURAL_SETTINGS)
@skipUnlessBackends(["cute"])
def test_actual_structural_repro_rebinds_under_conflicting_environment(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    function, args, marker = _case(name)
    policy = CuteStructuralPolicy(**{name: True})
    bound = helion.kernel(
        function,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_structural_policy=policy,
    )._bind_isolated(args)
    config = bound.config_spec.default_config()
    source = bound.to_code(config)
    ordinary = helion.kernel(
        function,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        **{setting: getattr(policy, setting) for setting in CUTE_STRUCTURAL_SETTINGS},
    )._bind_isolated(args)
    assert source == ordinary.to_code(config)
    assert _host(bound) == _host(ordinary)
    assert (
        bound.config_spec.cache_fingerprint_hash()
        == ordinary.config_spec.cache_fingerprint_hash()
    )
    decorator = bound.format_kernel_decorator(config, bound.settings)
    assert "helion.CuteStructuralConfig(" in decorator
    monkeypatch.setenv("HELION_BACKEND", "triton")
    for setting in CUTE_STRUCTURAL_SETTINGS:
        monkeypatch.setenv(
            f"HELION_{setting.upper()}", str(int(not getattr(policy, setting)))
        )
    restored = _replay(decorator, function)
    rebound = restored._bind_isolated(args)
    assert restored.cute_structural_policy == policy
    assert _host(rebound) == _host(bound)
    assert rebound.to_code() == source
    if marker == "fission":
        assert rebound.env.cute_fission_plan is not None
    else:
        assert marker in _host(rebound)
    (tmp_path / "resolved.py").write_text(source)
    (tmp_path / "ordinary.py").write_text(ordinary.to_code(config))
    (tmp_path / "replayed.py").write_text(rebound.to_code())
    (tmp_path / "decorator.txt").write_text(decorator + "\n")
    (tmp_path / "config-envelope.json").write_text(
        bound.config_envelope(config).to_json() + "\n"
    )
    (tmp_path / "witness.json").write_text(
        json.dumps(
            {
                "setting": name,
                "policy": policy.to_dict(),
                "config": config.config,
                "config_fingerprint": bound.config_spec.cache_fingerprint_hash(),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "ordinary_and_replayed_source_equal": True,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("explicit", [False, True])
@skipUnlessBackends(["cute"])
def test_late_envelope_checks_policy_even_when_config_axes_and_code_are_equal(
    explicit: bool,
) -> None:
    off = CuteStructuralPolicy()
    on = CuteStructuralPolicy(cute_full_slice_matmul_tiling=True)
    bound = _pointwise_bound(off if explicit else None)
    config = bound.config_spec.default_config()
    source = bound.to_code(config)
    assert source == _pointwise_bound(on).to_code(config)
    matching = bound.config_envelope(config)
    assert bound.to_code(matching) == source
    mismatch = CuteStructuralConfig(config, on)
    with (
        patch.object(
            bound, "compile_config", side_effect=AssertionError("compile forbidden")
        ),
        pytest.raises(StructuralPolicyError, match="different"),
    ):
        bound.set_config(mismatch)
    with (
        patch(
            "helion.runtime.kernel.generate_ast",
            side_effect=AssertionError("codegen forbidden"),
        ),
        pytest.raises(StructuralPolicyError, match="different"),
    ):
        bound.to_code(mismatch)
    bound.set_config(matching)
    assert bound._config == config
    assert bound._run is not None


@skipUnlessBackends(["cute"])
def test_matching_late_seeds_and_mismatched_late_seeds() -> None:
    bound = _pointwise_bound(CuteStructuralPolicy())
    config = bound.config_spec.default_config()
    bound.settings.autotune_seed_configs = [bound.config_envelope(config)]
    assert normalize_autotune_seed_configs(bound.settings) == (config,)
    bound.settings.autotune_seed_configs = CuteStructuralConfig(
        config, CuteStructuralPolicy(cute_flatten_nested_reductions=True)
    )
    with pytest.raises(StructuralPolicyError, match="Late autotune seed"):
        normalize_autotune_seed_configs(bound.settings)


@pytest.mark.parametrize("matching", [False, True])
@skipUnlessBackends(["cute"])
def test_late_override_envelopes_fail_before_config_interpretation(
    matching: bool,
) -> None:
    bound = _pointwise_bound(CuteStructuralPolicy())
    config = bound.config_spec.default_config()
    bound.settings.autotune_config_overrides = CuteStructuralConfig(  # type: ignore[assignment]
        config, CuteStructuralPolicy(cute_region_fission=not matching)
    )
    with (
        patch.object(
            type(bound.config_spec),
            "create_config_generation",
            side_effect=AssertionError("config interpretation forbidden"),
        ),
        pytest.raises(StructuralPolicyError, match="before creating a new Kernel"),
    ):
        PatternSearch(bound, (torch.empty(37),), initial_population=8)


@skipUnlessBackends(["cute"])
def test_policy_seed_and_override_preserve_actual_full_initial_population() -> None:
    policy = CuteStructuralPolicy(cute_full_slice_matmul_tiling=True)
    seed = helion.Config(block_sizes=[16, 32, 16])
    args = (
        torch.empty((32, 16), dtype=torch.float16),
        torch.empty((16, 64), dtype=torch.float16),
    )
    results = []
    for recorded in (False, True):
        kwargs = {} if recorded else {"cute_full_slice_matmul_tiling": True}
        settings = Settings(
            backend="cute",
            static_shapes=True,
            force_autotune=True,
            autotune_seed_configs=[
                CuteStructuralConfig(seed, policy) if recorded else seed
            ],
            autotune_config_overrides={"cute_proven_bounds": False},
            **kwargs,
        )
        bound = helion.kernel(_matmul, settings=settings)._bind_isolated(args)
        with bound.env:
            search = PatternSearch(bound, args, initial_population=100)
            random.seed(1807)
            population = [
                search.config_gen.unflatten(flat)
                for flat in checked_initial_population(search)
            ]
            user_pair = search.config_gen.user_seed_flat_config_pairs(
                search._autotune_seed_configs()
            )[0]
        # The helper checks 100 base rows and accounts for each coverage addition.
        assert population[0] == search.config_gen.unflatten(
            search.config_gen.default_flat()
        )
        assert population[1] == user_pair[1]
        assert all(not config.cute_proven_bounds for config in population)
        results.append([config.to_json() for config in population])
    assert results[0] == results[1]


@skipUnlessBackends(["cute"])
def test_late_seed_uses_captured_binding_policy_after_legacy_settings_mutation() -> (
    None
):
    bound = _pointwise_bound(None)
    config = bound.config_spec.default_config()
    bound.settings.cute_full_slice_matmul_tiling = True
    bound.settings.autotune_seed_configs = [
        CuteStructuralConfig(
            config, CuteStructuralPolicy(cute_full_slice_matmul_tiling=True)
        )
    ]
    search = PatternSearch(bound, (torch.empty(37),), initial_population=8)
    with pytest.raises(StructuralPolicyError, match="Late autotune seed envelope"):
        search._generate_initial_population_flat()


@skipUnlessBackends(["cute"])
def test_selected_settings_cannot_change_an_already_bound_schema() -> None:
    bound = _pointwise_bound(CuteStructuralPolicy())
    config = bound.config_spec.default_config()
    bound.settings.cute_flatten_nested_reductions = True
    with pytest.raises(StructuralPolicyError, match="Kernel settings"):
        bound.to_code(config)
    with pytest.raises(StructuralPolicyError, match="Kernel settings"):
        bound.kernel.bind((torch.empty(37),))
    with pytest.raises(StructuralPolicyError, match="Kernel settings"):
        bound.kernel(torch.empty(37))


@skipUnlessBackends(["cute"])
def test_real_python_module_cache_and_in_memory_binding_are_policy_scoped() -> None:
    off = CuteStructuralPolicy()
    on = CuteStructuralPolicy(cute_full_slice_matmul_tiling=True)
    args = (torch.empty(37),)
    kernels = [
        helion.kernel(
            _pointwise,
            backend="cute",
            static_shapes=True,
            cute_structural_policy=policy,
        )
        for policy in (off, off, on, None)
    ]
    bounds = [kernel.bind(args) for kernel in kernels]
    for kernel, bound in zip(kernels, bounds, strict=True):
        assert kernel.bind(args) is bound
    configs = [bound.config_spec.default_config() for bound in bounds]
    sources = [
        bound.to_code(config) for bound, config in zip(bounds, configs, strict=True)
    ]
    assert len(set(sources)) == 1
    paths = []
    original = PyCodeCache.load

    def capture(*args: object, **kwargs: object) -> Any:
        module = original(*args, **kwargs)
        paths.append(module.__file__)
        return module

    with patch.object(PyCodeCache, "load", side_effect=capture):
        for bound, config in zip(bounds, configs, strict=True):
            bound.compile_config(config)
    assert paths[0] == paths[1]
    assert len({paths[0], paths[2], paths[3]}) == 3
    assert bounds[0]._base_spec_key == bounds[1]._base_spec_key
    assert bounds[0]._base_spec_key != bounds[2]._base_spec_key
    assert bounds[3].extra_cache_key() == ""
    assert ast.parse(sources[0])


@skipUnlessBackends(["cute"])
def test_actual_split_k_child_preserves_parent_setting_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELION_CUTE_FLATTEN_NESTED_REDUCTIONS", "0")
    settings = Settings(
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_region_fission=False,
    )
    assert settings.cute_structural_origins.cute_region_fission == "explicit"
    assert (
        settings.cute_structural_origins.cute_flatten_nested_reductions == "environment"
    )
    assert settings.cute_structural_origins.cute_full_slice_matmul_tiling == "default"
    args = (*_split_k_args(), torch.empty(128, dtype=torch.float32))
    children = []
    original_init = CompileEnvironment.__init__

    def record_child(
        child: CompileEnvironment,
        device: torch.device,
        child_settings: Settings,
        **kwargs: Any,
    ) -> None:
        children.append(child_settings)
        original_init(child, device, child_settings, **kwargs)

    with _split_k_target():
        bound = helion.kernel(_fp32_bias.fn, settings=settings)._bind_isolated(args)
        with patch.object(CompileEnvironment, "__init__", record_child):
            source = bound.to_code(_split_k_config())
    assert len(_embedded_sources(source)) == 2
    assert len(children) == 1
    child = children[0]
    assert child is not settings
    assert child.static_shapes is True
    assert child.get_cute_structural_policy() == settings.get_cute_structural_policy()
    assert child.cute_structural_origins == settings.cute_structural_origins
