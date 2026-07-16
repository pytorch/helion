"""Validation for user-configurable RAG thresholds."""

from __future__ import annotations

import pytest

import helion_rag._util as util
from helion_rag._util import DEFAULT_SIM_THRESHOLD


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0.0),
        ("0.85", 0.85),
        ("0.9", 0.9),
        ("1", 1.0),
        ("1e-3", 0.001),
    ],
)
def test_parse_sim_threshold_accepts_finite_unit_interval(
    raw: str, expected: float
) -> None:
    assert util._parse_sim_threshold(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "-0.5", "2", "nan", "inf", "not-a-number"],
)
def test_parse_sim_threshold_uses_default_for_invalid_values(raw: str) -> None:
    assert util._parse_sim_threshold(raw) == DEFAULT_SIM_THRESHOLD
