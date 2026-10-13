from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from test._cute_aux import _cpu_codegen
from test.test_cute_horizontal_state_update import _affine_rows
from test.test_cute_horizontal_state_update import _args
from test.test_cute_horizontal_state_update import _pointwise_rows

import helion
from helion._compiler.cute import horizontal_state_config as policy

if TYPE_CHECKING:
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

KEYS = (
    "num_threads",
    "cute_vector_widths",
    "cute_lane_layouts",
    "cute_reduction_reloads",
    "load_eviction_policies",
)


def unchanged(spec: ConfigSpec, config: dict[str, object]) -> None:
    pass


def selected(bound: BoundKernel) -> helion.Config:
    return next(
        copy.deepcopy(c)
        for c in bound.config_spec.compiler_seed_configs
        if c.config.get(policy.CONFIG_KEY) == policy.HORIZONTAL
    )


def mutations(config: helion.Config) -> tuple[dict[str, object], ...]:
    values = config.config
    return (
        {"num_threads": [8, 0]},
        {"cute_vector_widths": [8] * len(values["cute_vector_widths"])},
        {"cute_lane_layouts": ["strided"] * len(values["cute_lane_layouts"])},
        {"cute_reduction_reloads": ["gmem"] * len(values["cute_reduction_reloads"])},
        {"load_eviction_policies": ["last"] * len(values["load_eviction_policies"])},
    )


@pytest.mark.parametrize("kernel", [_affine_rows, _pointwise_rows])
def test_complete_source_equivalence(kernel: Kernel) -> None:
    with _cpu_codegen():
        args = _args()
        bound = kernel._bind_isolated(args)
        reference = selected(bound)
        expected = bound.to_code(reference)
        variants = list(mutations(reference))
        variants.append(dict(item for update in variants for item in update.items()))
        for update in variants:
            request = helion.Config.from_dict(reference.config | update)
            with patch.object(policy, "canonicalize_inactive", unchanged):
                original = bound.to_code(request)
            assert original == expected
            actual = bound.to_code(request)
            assert actual == original
            normalized = bound.config_spec.normalized_config(request)
            assert normalized == reference
            assert bound.config_spec.normalized_config(normalized) == normalized


def test_ordinary_and_unavailable_unchanged() -> None:
    with _cpu_codegen():
        args = _args()
        bound = _affine_rows._bind_isolated(args)
        default = bound.config_spec.default_config()
        for update in mutations(selected(bound)):
            request = helion.Config.from_dict(default.config | update)
            with patch.object(policy, "canonicalize_inactive", unchanged):
                old = bound.config_spec.normalized_config(request)
            assert bound.config_spec.normalized_config(request) == old
        raw = {policy.CONFIG_KEY: policy.HORIZONTAL, "num_threads": [8, 0]}
        with patch.object(bound.config_spec, "cute_horizontal_state_available", False):
            before = copy.deepcopy(raw)
            policy.canonicalize_inactive(bound.config_spec, raw)
            assert raw == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_threads", [3, 0]),
        ("cute_vector_widths", [3]),
        ("cute_lane_layouts", ["unknown"]),
        ("cute_reduction_reloads", ["unknown"]),
    ],
)
def test_original_invalid_values_are_not_hidden(key: str, value: object) -> None:
    with _cpu_codegen():
        args = _args()
        bound = _affine_rows._bind_isolated(args)
        request = helion.Config.from_dict(selected(bound).config | {key: value})
        with (
            patch.object(policy, "canonicalize_inactive", unchanged),
            pytest.raises(Exception) as before,
        ):
            bound.config_spec.normalized_config(request)
        with pytest.raises(type(before.value)):
            bound.config_spec.normalized_config(request)


def test_active_geometry_and_launch_values_retained() -> None:
    with _cpu_codegen():
        args = _args()
        bound = _affine_rows._bind_isolated(args)
        reference = selected(bound)
        source = bound.to_code(reference)
        request = helion.Config.from_dict(reference.config | {"block_sizes": [16]})
        normalized = bound.config_spec.normalized_config(request)
        assert normalized.config["block_sizes"] == [16]
        assert bound.to_code(request) != source
        for key, value in (
            ("cute_pdl", True),
            ("cute_min_blocks_per_mp", 2),
            ("cute_cluster_n", 2),
        ):
            request = helion.Config.from_dict(reference.config | {key: value})
            with patch.object(policy, "canonicalize_inactive", unchanged):
                before = bound.config_spec.normalized_config(request)
            assert bound.config_spec.normalized_config(request) == before


def test_cache_version_only_for_available_horizontal() -> None:
    with _cpu_codegen():
        args = _args()
        bound = _affine_rows._bind_isolated(args)
        spec = bound.config_spec
        value = spec.cache_fingerprint_hash()
        with patch.object(policy, "NORMALIZATION_VERSION", 2):
            assert spec.cache_fingerprint_hash() != value
        with patch.object(spec, "cute_horizontal_state_available", False):
            value = spec.cache_fingerprint_hash()
            with patch.object(policy, "NORMALIZATION_VERSION", 2):
                assert spec.cache_fingerprint_hash() == value


def test_seed_prefix_and_default_unchanged() -> None:
    with _cpu_codegen():
        args = _args()
        current = _affine_rows._bind_isolated(args)
        default = current.config_spec.default_config()
        seeds = current.config_spec.create_config_generation().seed_flat_config_pairs()
        with patch.object(policy, "canonicalize_inactive", unchanged):
            original = _affine_rows._bind_isolated(args)
            assert default == original.config_spec.default_config()
            assert (
                seeds
                == original.config_spec.create_config_generation().seed_flat_config_pairs()
            )


def test_unknown_eviction_request_is_not_erased() -> None:
    with _cpu_codegen():
        args = _args()
        bound = _affine_rows._bind_isolated(args)
        request = selected(bound)
        request.config["load_eviction_policies"] = ["unknown"]
        normalized = bound.config_spec.normalized_config(request)
        assert normalized.config["load_eviction_policies"] == ["unknown"]
