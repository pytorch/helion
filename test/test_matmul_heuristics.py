from __future__ import annotations

import json
import os
from types import SimpleNamespace
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

_SHAPE_BUCKET_KEYS = {"aspect", "dtype", "k_bin", "m_bin", "n_bin"}


def _clear_heuristic_caches() -> None:
    matmul_heuristics._runtime_heuristics.cache_clear()
    matmul_heuristics._heuristic_rules.cache_clear()


def _matmul_fact(
    *,
    static_m: int = 4,
    static_n: int = 8192,
    static_k: int = 16384,
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
    x = torch.empty((4, 16384), dtype=torch.bfloat16)
    y = torch.empty((16384, 8192), dtype=torch.bfloat16)

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
    x = torch.empty((4, 16384), dtype=torch.bfloat16)
    y = torch.empty((16384, 8192), dtype=torch.bfloat16)

    seeds = matmul_heuristic_seed_configs(
        (x, y),
        config_spec=config_spec,
        max_configs=3,
    )

    assert len(seeds) == 1
    seed = dict(seeds[0])
    assert seed["block_sizes"] == [8, 64, 256]
    assert seed["l2_groupings"] == [1]
    assert seed["num_warps"] == 8
    assert seed["num_stages"] == 5
    assert seed["pid_type"] == "flat"


def test_matmul_heuristic_rules_have_unique_shape_buckets() -> None:
    data = json.loads(_RUNTIME_HEURISTICS_PATH.read_text())
    keys = [
        (rule["kernel_class"], json.dumps(rule["shape_bucket"], sort_keys=True))
        for rule in data["rules"]
    ]

    assert set(data) == {"rules"}
    assert len(keys) == len(set(keys))
    for rule in data["rules"]:
        assert set(rule) == {"kernel_class", "shape_bucket", "templates"}
        assert set(rule["shape_bucket"]).issubset(_SHAPE_BUCKET_KEYS)
        assert rule["templates"]
        assert all("template" not in template for template in rule["templates"])


def test_matmul_heuristics_prefer_more_specific_shape_bucket(tmp_path) -> None:
    config_spec = _matmul_config_spec()
    x = torch.empty((4, 16384), dtype=torch.bfloat16)
    y = torch.empty((16384, 8192), dtype=torch.bfloat16)
    heuristics_path = tmp_path / "matmul_heuristics.json"
    heuristics_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "kernel_class": "matmul",
                        "shape_bucket": {
                            "aspect": "skinny_m",
                            "dtype": "fp16_bf16",
                        },
                        "templates": [
                            {
                                "block_sizes": [16, 64, 64],
                                "num_stages": 2,
                                "num_warps": 1,
                            }
                        ],
                    },
                    {
                        "kernel_class": "matmul",
                        "shape_bucket": {
                            "aspect": "skinny_m",
                            "dtype": "fp16_bf16",
                            "k_bin": "<=32768",
                            "m_bin": "<=4",
                            "n_bin": ">4096",
                        },
                        "templates": [
                            {
                                "block_sizes": [8, 64, 256],
                                "num_stages": 5,
                                "num_warps": 8,
                            }
                        ],
                    },
                ]
            }
        )
    )

    with patch.dict(
        os.environ,
        {
            "HELION_AUTOTUNE_MATMUL_HEURISTICS": "1",
            "HELION_AUTOTUNE_MATMUL_HEURISTICS_PATH": str(heuristics_path),
        },
    ):
        _clear_heuristic_caches()
        try:
            seeds = matmul_heuristic_seed_configs(
                (x, y),
                config_spec=config_spec,
                max_configs=2,
            )
        finally:
            _clear_heuristic_caches()

    assert [dict(seed)["block_sizes"] for seed in seeds] == [
        [8, 64, 256],
        [16, 64, 64],
    ]


def test_matmul_kernel_detection_uses_matmul_facts() -> None:
    config_spec = _matmul_config_spec(
        reduction_loops=[object()],
        matmul_facts=[_matmul_fact()],
    )
    x = torch.empty((4, 16384), dtype=torch.bfloat16)
    y = torch.empty((16384, 8192), dtype=torch.bfloat16)

    seeds = matmul_heuristic_seed_configs_for_kernel(
        None,
        (x, y),
        config_spec=config_spec,
        max_configs=3,
    )

    assert len(seeds) == 1
    assert dict(seeds[0])["block_sizes"] == [8, 64, 256]


def test_dense_matmul_square_aot_seed_buckets() -> None:
    cases = (
        (256, [[32, 32, 64]]),
        (512, [[16, 64, 128], [32, 64, 256]]),
        (1024, [[128, 64, 64], [64, 64, 64]]),
        (
            4096,
            [
                [128, 128, 64],
                [128, 128, 64],
                [128, 256, 64],
                [256, 256, 32],
                [256, 256, 64],
            ],
        ),
    )

    for size, expected_block_sizes in cases:
        config_spec = _matmul_config_spec(
            matmul_facts=[_matmul_fact(static_m=size, static_n=size, static_k=size)],
        )
        x = torch.empty((size, size), dtype=torch.bfloat16)
        y = torch.empty((size, size), dtype=torch.bfloat16)

        seeds = matmul_heuristic_seed_configs_for_kernel(
            None,
            (x, y),
            config_spec=config_spec,
            max_configs=10,
        )

        assert [dict(seed)["block_sizes"] for seed in seeds] == expected_block_sizes


def test_matmul_kernel_detection_skips_multiple_matmuls() -> None:
    config_spec = _matmul_config_spec(
        reduction_loops=[object()],
        matmul_facts=[_matmul_fact(), _matmul_fact()],
    )
    x = torch.empty((4, 16384), dtype=torch.bfloat16)
    y = torch.empty((16384, 8192), dtype=torch.bfloat16)

    assert (
        matmul_heuristic_seed_configs_for_kernel(
            None,
            (x, y),
            config_spec=config_spec,
            max_configs=3,
        )
        == []
    )


def test_matmul_family_uses_kernel_class_in_rule_lookup() -> None:
    config_spec = _matmul_config_spec()
    x = torch.empty((4096, 4096), dtype=torch.bfloat16)
    packed_weight = torch.empty((2048, 4096), dtype=torch.int8)

    int4_seed = dict(
        matmul_heuristic_seed_configs(
            (x, packed_weight),
            config_spec=config_spec,
            max_configs=3,
            kernel_class="matmul_int4",
        )[0]
    )
    fp4_seed = dict(
        matmul_heuristic_seed_configs(
            (x, packed_weight),
            config_spec=config_spec,
            max_configs=3,
            kernel_class="matmul_fp4",
        )[0]
    )

    assert int4_seed["block_sizes"] == [16, 128, 128]
    assert int4_seed["l2_groupings"] == [2]
    assert fp4_seed["block_sizes"] == [8, 128, 128]
    assert fp4_seed["l2_groupings"] == [1]


def test_matmul_heuristics_skip_non_matching_shape() -> None:
    config_spec = _matmul_config_spec()
    x = torch.empty((128, 128), dtype=torch.bfloat16)
    y = torch.empty((128, 128), dtype=torch.bfloat16)

    assert (
        matmul_heuristic_seed_configs(
            (x, y),
            config_spec=config_spec,
            max_configs=3,
        )
        == []
    )
