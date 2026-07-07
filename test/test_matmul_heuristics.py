from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

import helion
from helion._compiler.autotuner_heuristics.triton import _B200_MATMUL_HEURISTICS_PATH
from helion._compiler.autotuner_heuristics.triton import (
    TritonB200FormulaMatmulHeuristic,
)
from helion._compiler.autotuner_heuristics.triton import TritonB200MatmulHeuristic
from helion._compiler.autotuner_heuristics.triton import TritonH100MatmulHeuristic
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


def test_b200_formula_subsumes_table_promotion_wiring() -> None:
    # The sm100 FORMULA owns the compiler default; the TABLE is demoted to a search seed.
    assert TritonB200FormulaMatmulHeuristic.promote_seed_to_default is True
    assert TritonB200MatmulHeuristic.promote_seed_to_default is False
    assert TritonB200FormulaMatmulHeuristic.HARDWARE_TARGETS == (("cuda", "sm100"),)
    # The formula is a subclass of the H100 budget formula (inherits _matmul_tile).
    assert issubclass(TritonB200FormulaMatmulHeuristic, TritonH100MatmulHeuristic)
    # Registered AFTER the table so it wins the last-promote-wins default loop.
    from helion._compiler.autotuner_heuristics import get_heuristics

    order = [h.__name__ for h in get_heuristics("triton")]
    assert order.index("TritonB200FormulaMatmulHeuristic") > order.index(
        "TritonB200MatmulHeuristic"
    )


def test_h100_base_tile_is_unchanged_by_tmem_budget() -> None:
    # TMEM_ACC_BUDGET is None on the sm90 base (no TMEM accumulator path), so the H100 formula is
    # byte-identical: a big compute-bound cube stays the wide-N [128, 256, 64] w8 s4 tile (num_sm=132).
    assert TritonH100MatmulHeuristic.TMEM_ACC_BUDGET is None
    assert TritonH100MatmulHeuristic._matmul_tile(4096, 4096, 4096, 2, 132, 1) == (
        128,
        256,
        64,
        8,
        4,
        1,
    )


def test_b200_tmem_budget_grows_to_square_tile() -> None:
    # On sm100 the accumulator lives in TMEM (65536 fp32 capacity), so a SINGLE (pinned_grid==1) GEMM
    # that fills a wave grows to the [T,T] square that fills it: T = isqrt(65536) = 256 (derived, not
    # hardcoded). bk auto-shrinks to fit SMEM.
    from math import isqrt

    cls = TritonB200FormulaMatmulHeuristic
    assert cls.TMEM_ACC_BUDGET == 65536
    assert (
        isqrt(cls.TMEM_ACC_BUDGET) == 256
    )  # the square side is derived from the TMEM capacity
    sm = 148
    # large compute-bound cube (fills the machine at 256x256) -> grows to the square, bk shrinks 64->32
    bm, bn, bk, _w, _s, _l2 = cls._matmul_tile(4096, 4096, 4096, 2, sm, 1)
    assert (bm, bn, bk) == (256, 256, 32)
    assert cls._matmul_tile(8192, 8192, 8192, 2, sm, 1)[:3] == (256, 256, 32)
    # small M*N (2048^2 = 64 tiles << 148 SMs, wave_eff 0.43 < 0.8) -> keeps the wide [128, 256] tile
    assert cls._matmul_tile(2048, 2048, 2048, 2, sm, 1)[:2] == (128, 256)
    # a BATCHED dot (pinned_grid > 1, e.g. bmm batch=4) NEVER grows the square (the win inverts there)
    assert cls._matmul_tile(4096, 4096, 4096, 2, sm, 4)[:2] == (128, 256)
    assert cls._matmul_tile(2048, 2048, 2048, 2, sm, 8)[:2] == (128, 256)
