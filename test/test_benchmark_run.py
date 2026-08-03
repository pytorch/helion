from __future__ import annotations

from benchmarks import run


def test_default_accuracy_tolerances() -> None:
    args: list[str] = []

    run.add_default_accuracy_tolerances("softmax", args)

    assert args == ["--atol", "0.01", "--rtol", "0.01"]


def test_welford_uses_bf16_aware_absolute_tolerance() -> None:
    args: list[str] = []

    run.add_default_accuracy_tolerances("welford", args)

    assert args == ["--atol", "0.02", "--rtol", "0.01"]


def test_explicit_accuracy_tolerances_take_precedence() -> None:
    args = ["--atol=0.5", "--rtol", "0.25"]

    run.add_default_accuracy_tolerances("welford", args)

    assert args == ["--atol=0.5", "--rtol", "0.25"]
