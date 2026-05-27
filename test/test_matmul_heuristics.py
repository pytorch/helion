from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

import helion
from helion._compiler.autotuner_heuristics.triton import _B200_MATMUL_HEURISTICS_PATH
from helion._compiler.autotuner_heuristics.triton import TritonB200MatmulHeuristic
from helion._compiler.autotuner_heuristics.triton import _seed_config_for_bucket
from helion._compiler.autotuner_heuristics.triton import _seed_config_for_config_spec
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_spec import MatmulFact

_SHAPE_BUCKET_KEYS = {
    "dtype",
    "k_bucket",
    "m_bucket",
    "n_bucket",
    "k_value",
    "m_value",
    "n_value",
}


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
    matmul_facts: list[MatmulFact] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        matmul_facts=[] if matmul_facts is None else matmul_facts,
        block_sizes=[object(), object(), object()],
        allowed_pid_types=("flat",),
        _base_default_config=lambda: helion.Config(
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
        _shrink_for_numel_constraints=lambda config: None,
    )


def _bucket(m: int, n: int, k: int) -> dict[str, object]:
    return {
        "dtype": "fp16_bf16",
        "m_value": m,
        "n_value": n,
        "k_value": k,
    }


def test_matmul_heuristic_rules_have_unique_shape_buckets() -> None:
    data = json.loads(_B200_MATMUL_HEURISTICS_PATH.read_text())
    keys = [json.dumps(rule["shape_bucket"], sort_keys=True) for rule in data["rules"]]

    assert set(data) == {"rules"}
    assert len(keys) == len(set(keys))
    for rule in data["rules"]:
        assert set(rule) == {"shape_bucket", "templates"}
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


def test_matmul_bucket_matching_generates_seed_config() -> None:
    seed = _seed_config_for_bucket(
        _bucket(1024, 1024, 1024),
        config_spec=_matmul_config_spec(),
    )

    assert seed is not None
    assert dict(seed)["block_sizes"] == [128, 64, 64]
    assert dict(seed)["l2_groupings"] == [2]

    assert (
        _seed_config_for_bucket(
            _bucket(128, 128, 128),
            config_spec=_matmul_config_spec(),
        )
        is None
    )


def test_matmul_fact_generates_compiler_seed_config() -> None:
    config_spec = _matmul_config_spec(matmul_facts=[_matmul_fact()])

    seed = _seed_config_for_config_spec(config_spec)

    assert seed is not None
    assert dict(seed)["block_sizes"] == [128, 64, 64]

    config_spec = _matmul_config_spec(
        matmul_facts=[_matmul_fact(), _matmul_fact()],
    )

    assert _seed_config_for_config_spec(config_spec) is None


def test_triton_b200_matmul_heuristic_gates_on_hardware() -> None:
    env = SimpleNamespace(device=None, config_spec=_matmul_config_spec())
    env.config_spec.matmul_facts.append(_matmul_fact())
    b200 = SimpleNamespace(
        device_kind="cuda",
        hardware_name="NVIDIA B200",
        compute_capability="sm100",
    )
    h100 = SimpleNamespace(
        device_kind="cuda",
        hardware_name="NVIDIA H100",
        compute_capability="sm90",
    )

    with patch(
        "helion._hardware.get_hardware_info",
        return_value=b200,
    ):
        assert TritonB200MatmulHeuristic.is_eligible(env, SimpleNamespace())
        seed = TritonB200MatmulHeuristic.get_seed_config(env, SimpleNamespace())

    assert seed is not None
    assert dict(seed)["block_sizes"] == [128, 64, 64]

    with patch(
        "helion._hardware.get_hardware_info",
        return_value=h100,
    ):
        assert not TritonB200MatmulHeuristic.is_eligible(env, SimpleNamespace())
