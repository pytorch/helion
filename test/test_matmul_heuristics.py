from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

import helion
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_spec import MatmulFact
import helion.autotuner.matmul_heuristics as matmul_heuristics
from helion.autotuner.matmul_heuristics import _RUNTIME_HEURISTICS_PATH
from helion.autotuner.matmul_heuristics import matmul_heuristic_seed_configs
from helion.autotuner.matmul_heuristics import matmul_heuristic_seed_configs_for_kernel

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

_SHAPE_BUCKET_KEYS = {
    "dtype",
    "k_bucket",
    "m_bucket",
    "n_bucket",
    "k_value",
    "m_value",
    "n_value",
}


def _fake_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
) -> SimpleNamespace:
    return SimpleNamespace(shape=shape, dtype=dtype)


def _supported_b200_device() -> AbstractContextManager[object]:
    return patch.object(
        matmul_heuristics,
        "matmul_heuristics_supported_on_args",
        return_value=True,
    )


def _matmul_fact(
    *,
    static_m: int = 1024,
    static_n: int = 1024,
    static_k: int = 1024,
    lhs_ndim: int = 2,
    rhs_ndim: int = 2,
) -> MatmulFact:
    return MatmulFact(
        lhs_ndim=lhs_ndim,
        rhs_ndim=rhs_ndim,
        m_block_id=0,
        n_block_id=1,
        k_block_id=2,
        static_m=static_m,
        static_n=static_n,
        static_k=static_k,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )


def _matmul_config_spec(
    *,
    reduction_loops: list[object] | None = None,
    matmul_facts: list[MatmulFact] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        reduction_loops=[] if reduction_loops is None else reduction_loops,
        matmul_facts=[] if matmul_facts is None else matmul_facts,
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


def test_matmul_seeds_are_default_on_and_disableable() -> None:
    config_spec = _matmul_config_spec()
    x = _fake_tensor((1024, 1024))
    y = _fake_tensor((1024, 1024))

    with patch.dict(os.environ, {}, clear=True):
        assert matmul_heuristic_seed_configs(
            (x, y),
            config_spec=config_spec,
            max_configs=3,
        )

    with patch.dict(
        os.environ,
        {"HELION_AUTOTUNE_MATMUL_HEURISTICS": "0"},
        clear=True,
    ):
        assert (
            matmul_heuristic_seed_configs(
                (x, y),
                config_spec=config_spec,
                max_configs=3,
            )
            == []
        )


def test_matmul_heuristics_generate_valid_seed() -> None:
    config_spec = _matmul_config_spec()
    x = _fake_tensor((1024, 1024))
    y = _fake_tensor((1024, 1024))

    seeds = matmul_heuristic_seed_configs(
        (x, y),
        config_spec=config_spec,
        max_configs=10,
    )

    assert len(seeds) == 2
    seed = dict(seeds[0])
    assert seed["block_sizes"] == [128, 64, 64]
    assert seed["l2_groupings"] == [2]
    assert seed["num_warps"] == 4
    assert seed["num_stages"] == 4
    assert seed["pid_type"] == "flat"
    assert dict(seeds[1])["block_sizes"] == [64, 64, 64]


def test_matmul_heuristic_rules_have_unique_shape_buckets() -> None:
    data = json.loads(_RUNTIME_HEURISTICS_PATH.read_text())
    keys = [
        (rule["kernel_class"], json.dumps(rule["shape_bucket"], sort_keys=True))
        for rule in data["rules"]
    ]

    assert set(data) == {"rules"}
    assert len(keys) == len(set(keys))
    assert {rule["kernel_class"] for rule in data["rules"]} == {
        "matmul",
        "matmul_int4",
    }
    for rule in data["rules"]:
        assert set(rule) == {"kernel_class", "shape_bucket", "templates"}
        assert set(rule["shape_bucket"]).issubset(_SHAPE_BUCKET_KEYS)
        for key in ("k_bucket", "m_bucket", "n_bucket"):
            value = rule["shape_bucket"].get(key)
            if value is not None:
                values = value if isinstance(value, list) else [value]
                assert all(isinstance(item, str) for item in values)
                assert all(item.startswith("(") for item in values)
                assert all(item.endswith(("]", ")")) for item in values)
        for key in ("k_value", "m_value", "n_value"):
            value = rule["shape_bucket"].get(key)
            if value is not None:
                values = value if isinstance(value, list) else [value]
                assert all(isinstance(item, int) for item in values)
        assert rule["templates"]
        assert all("template" not in template for template in rule["templates"])


def test_matmul_kernel_detection_uses_matmul_facts() -> None:
    config_spec = _matmul_config_spec(
        reduction_loops=[object()],
        matmul_facts=[_matmul_fact()],
    )
    x = _fake_tensor((1024, 1024))
    y = _fake_tensor((1024, 1024))

    with _supported_b200_device():
        seeds = matmul_heuristic_seed_configs_for_kernel(
            None,
            (x, y),
            config_spec=config_spec,
            max_configs=10,
        )

    assert len(seeds) == 2
    assert dict(seeds[0])["block_sizes"] == [128, 64, 64]
    assert dict(seeds[1])["block_sizes"] == [64, 64, 64]


def test_dense_matmul_exact_ci_seed_buckets() -> None:
    cases = (
        ((4096, 1024, 1024), [256, 128, 64], [64]),
        ((2048, 4096, 2048), [256, 256, 64], [4]),
    )

    config_spec = _matmul_config_spec()
    for (m, n, k), expected_block_sizes, expected_l2 in cases:
        seeds = matmul_heuristic_seed_configs(
            (_fake_tensor((m, k)), _fake_tensor((k, n))),
            config_spec=config_spec,
            max_configs=10,
            kernel_class="matmul",
        )

        seed = dict(seeds[0])
        assert seed["block_sizes"] == expected_block_sizes
        assert seed["l2_groupings"] == expected_l2


def test_matmul_kernel_detection_skips_multiple_matmuls() -> None:
    config_spec = _matmul_config_spec(
        reduction_loops=[object()],
        matmul_facts=[_matmul_fact(), _matmul_fact()],
    )
    x = _fake_tensor((1024, 1024))
    y = _fake_tensor((1024, 1024))

    with _supported_b200_device():
        seeds = matmul_heuristic_seed_configs_for_kernel(
            None,
            (x, y),
            config_spec=config_spec,
            max_configs=3,
        )

    assert seeds == []


def test_int4_matmul_exact_ci_seed_buckets() -> None:
    cases = (
        ((1, 1280, 8192), [1024, 1, 16], [64]),
        ((65536, 1280, 8192), [16, 128, 256], [32]),
    )

    config_spec = _matmul_config_spec()
    for (m, n, k), expected_block_sizes, expected_l2 in cases:
        seeds = matmul_heuristic_seed_configs(
            (
                _fake_tensor((m, k)),
                _fake_tensor((k // 2, n), dtype=torch.int8),
            ),
            config_spec=config_spec,
            max_configs=10,
            kernel_class="matmul_int4",
        )

        seed = dict(seeds[0])
        assert seed["block_sizes"] == expected_block_sizes
        assert seed["l2_groupings"] == expected_l2


def test_int4_matmul_bad_ffn_w2_shapes_fall_back() -> None:
    config_spec = _matmul_config_spec()

    for m_value in (1, 4):
        assert (
            matmul_heuristic_seed_configs(
                (
                    _fake_tensor((m_value, 3584)),
                    _fake_tensor((1792, 8192), dtype=torch.int8),
                ),
                config_spec=config_spec,
                max_configs=10,
                kernel_class="matmul_int4",
            )
            == []
        )


def test_matmul_heuristics_skip_non_matching_shape() -> None:
    config_spec = _matmul_config_spec()
    x = _fake_tensor((128, 128))
    y = _fake_tensor((128, 128))

    assert (
        matmul_heuristic_seed_configs(
            (x, y),
            config_spec=config_spec,
            max_configs=3,
        )
        == []
    )
