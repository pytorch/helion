from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmarks.cute import compare_kda_recurrent_backends as recurrent
import pytest

if TYPE_CHECKING:
    from pathlib import Path

ORDINARY_CONFIGS = {
    "t3-lower-bound-n8-h16": {
        "block_sizes": [32],
        "num_threads": [2, 32],
        "loop_orders": [[2, 1, 0]],
        "num_warps": 2,
        "num_stages": 1,
        "indexing": "pointer",
        "pid_type": "flat",
    },
    "t5-precomputed-n32-h16-hv32": {
        "block_sizes": [128],
        "num_threads": [8, 32],
        "loop_orders": [[2, 1, 0]],
        "num_warps": 8,
        "num_stages": 1,
        "indexing": "pointer",
        "pid_type": "flat",
    },
}


def test_reportable_t5_case_is_in_default_scope() -> None:
    case = recurrent.CASES["t5-precomputed-n32-h16-hv32"]

    assert tuple(recurrent.CASES) == recurrent.DEFAULT_CASES
    assert (
        case.num_sequences,
        case.num_tokens,
        case.num_heads,
        case.num_value_heads,
    ) == (32, 5, 16, 32)
    assert case.use_gate_in_kernel is False
    assert case.gate_mode == "precomputed"
    assert recurrent._case_semantics(case, 0)["input_distribution"] == {
        "q_k_v": "normal-std-0.25-bfloat16",
        "beta_logits": "normal-std-1.0-bfloat16",
        "state": "normal-std-0.02-bfloat16",
        "precomputed_gate": "logsigmoid-normal-float32-to-bfloat16",
    }


def test_recommended_configs_round_trip_through_external_map(
    tmp_path: Path,
) -> None:
    config_map = tmp_path / "recurrent-configs.json"
    config_map.write_text(json.dumps(ORDINARY_CONFIGS), encoding="utf-8")

    loaded = recurrent._load_helion_config_map(config_map)

    assert {name: json.loads(config) for name, config in loaded.items()} == (
        ORDINARY_CONFIGS
    )


def test_ordinary_eligible_recurrent_config_keeps_fixed_token_provenance() -> None:
    plan_kind, markers, patterns, forbidden_patterns = (
        recurrent._helion_codegen_expectation(
            recurrent.CASES["t3-lower-bound-n8-h16"],
            {
                "block_sizes": [32],
                "num_warps": 2,
                "pid_type": "flat",
            },
        )
    )

    assert plan_kind == "fixed-token-rank1"
    assert "fixed_rank1_codegen_abi_version = 3" in markers
    assert patterns == ()
    assert forbidden_patterns == ()


@pytest.mark.parametrize(
    ("case_name", "config"),
    [
        (
            "t5-precomputed-n32-h16-hv32",
            {},
        ),
        (
            "t3-lower-bound-n8-h16",
            {
                "block_sizes": [64],
                "num_warps": 4,
                "pid_type": "flat",
            },
        ),
    ],
)
def test_ordinary_ineligible_recurrent_config_tracks_generic_codegen(
    case_name: str, config: dict[str, object]
) -> None:
    plan_kind, markers, patterns, forbidden_patterns = (
        recurrent._helion_codegen_expectation(recurrent.CASES[case_name], config)
    )

    assert plan_kind == "ordinary-cute"
    assert markers == ()
    assert patterns == ()
    assert any("fixed_rank1" in pattern for pattern in forbidden_patterns)
