from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch

import helion
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.observed_heuristics import classify_runtime_kernel
from helion.autotuner.observed_heuristics import observed_heuristic_seed_configs
from helion.runtime.settings import Settings


def _row_config_spec() -> SimpleNamespace:
    return SimpleNamespace(
        reduction_loops=[object()],
        allowed_pid_types=("flat",),
        default_config=lambda: helion.Config(
            block_sizes=[1],
            num_warps=4,
            num_stages=1,
            pid_type="flat",
        ),
        _flat_fields=lambda: {
            "block_sizes": ListOf(IntegerFragment(1, 4096, 1), length=1),
            "num_warps": PowerOfTwoFragment(1, 32, 4),
            "num_stages": IntegerFragment(1, 8, 1),
            "pid_type": EnumFragment(("flat",)),
        },
        normalize=lambda raw, _fix_invalid=False: None,
    )


def test_observed_heuristic_seeds_are_default_on_and_disableable() -> None:
    config_spec = _row_config_spec()
    x = torch.empty((2048, 8192), dtype=torch.bfloat16)

    with patch.dict(os.environ, {}, clear=True):
        assert observed_heuristic_seed_configs(
            (x,),
            workload_traits=frozenset({"reduction", "softmax"}),
            config_spec=config_spec,
            max_configs=3,
        )

    with patch.dict(
        os.environ, {"HELION_AUTOTUNE_OBSERVED_HEURISTICS": "0"}, clear=True
    ):
        assert (
            observed_heuristic_seed_configs(
                (x,),
                workload_traits=frozenset({"reduction", "softmax"}),
                config_spec=config_spec,
                max_configs=3,
            )
            == []
        )


def test_observed_heuristics_generate_valid_matmul_seeds() -> None:
    config_spec = SimpleNamespace(
        reduction_loops=[],
        allowed_pid_types=("flat",),
        default_config=lambda: helion.Config(
            block_sizes=[1, 1, 1],
            l2_groupings=[1],
            num_warps=4,
            num_stages=1,
            pid_type="flat",
        ),
        _flat_fields=lambda: {
            "block_sizes": ListOf(IntegerFragment(1, 4096, 1), length=3),
            "l2_groupings": ListOf(IntegerFragment(1, 64, 1), length=1),
            "num_warps": PowerOfTwoFragment(1, 32, 4),
            "num_stages": IntegerFragment(1, 8, 1),
            "pid_type": EnumFragment(("flat",)),
        },
        normalize=lambda raw, _fix_invalid=False: None,
    )

    x = torch.empty((4, 16384), dtype=torch.bfloat16)
    y = torch.empty((16384, 8192), dtype=torch.bfloat16)
    seeds = observed_heuristic_seed_configs(
        (x, y),
        workload_traits=frozenset({"matmul"}),
        config_spec=config_spec,
        max_configs=3,
    )

    assert len(seeds) == 1
    seed = dict(seeds[0])
    assert len(seed["block_sizes"]) == 3
    assert all(isinstance(block, int) and block > 0 for block in seed["block_sizes"])
    assert seed["num_warps"] in {1, 2, 4, 8, 16, 32}
    assert 1 <= seed["num_stages"] <= 8


def test_autotune_effort_none_uses_observed_heuristic_config() -> None:
    from helion.runtime.kernel import BoundKernel

    observed_config = helion.Config(num_warps=8)
    settings = Settings(autotune_effort="none")
    config_spec = SimpleNamespace(
        default_config=lambda: helion.Config(num_warps=4),
    )
    bound = BoundKernel.__new__(BoundKernel)
    bound.kernel = SimpleNamespace(
        configs=[],
        settings=settings,
    )
    bound._env = SimpleNamespace(config_spec=config_spec)
    bound.fake_args = []
    assert bound.env is bound._env

    with (
        patch(
            "helion.runtime.kernel.observed_heuristic_default_config",
            return_value=observed_config,
        ),
        patch("helion.runtime.kernel.is_ref_mode_enabled", return_value=True),
    ):
        config = BoundKernel._user_provided_config(bound, ())

    assert config is observed_config


def test_row_reduction_classification_uses_runtime_structure() -> None:
    config_spec = SimpleNamespace(
        reduction_loops=[object()],
        default_config=lambda: helion.Config(block_sizes=[1]),
        _flat_fields=lambda: {
            "block_sizes": ListOf(IntegerFragment(1, 4096, 1), length=1),
        },
    )

    x = torch.empty((2048, 8192), dtype=torch.bfloat16)
    weight = torch.empty((8192,), dtype=torch.bfloat16)
    bias = torch.empty((8192,), dtype=torch.bfloat16)
    labels = torch.empty((2048,), dtype=torch.int64)

    assert (
        classify_runtime_kernel(
            (x, weight, bias),
            workload_traits=frozenset({"reduction"}),
            config_spec=config_spec,
        )
        == "row_norm_layer"
    )
    assert (
        classify_runtime_kernel(
            (x, weight),
            workload_traits=frozenset({"reduction"}),
            config_spec=config_spec,
        )
        == "row_norm_rms"
    )
    assert (
        classify_runtime_kernel(
            (x, labels),
            workload_traits=frozenset({"reduction", "exp", "sum_reduction"}),
            config_spec=config_spec,
        )
        == "row_cross_entropy"
    )
    assert (
        classify_runtime_kernel(
            (x,),
            workload_traits=frozenset({"reduction", "exp", "sum_reduction"}),
            config_spec=config_spec,
        )
        == "row_softmax"
    )
