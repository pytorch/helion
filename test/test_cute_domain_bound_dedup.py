from __future__ import annotations

from unittest.mock import patch

import pytest

from helion._compiler.cute import chained_matmul


def domain(
    bounds: list[str], definitions: dict[str, str], unavailable: set[str]
) -> chained_matmul._DomainExpression:
    expression = object.__new__(chained_matmul._DomainExpression)
    expression.bounds = bounds
    expression.definitions = definitions
    expression.unavailable_values = unavailable
    return expression


def test_duplicate_domains_keep_order_without_repeated_expansion() -> None:
    expression = domain(
        ["0 <= index < 16", "0 <= row < 32"] * 10000,
        {"index": "row + 1"},
        set(),
    )
    with patch.object(
        chained_matmul,
        "expr_from_string",
        wraps=chained_matmul.expr_from_string,
    ) as parse:
        assert chained_matmul._domain_bounds(expression) == [
            "0 <= row + 1 < 16",
            "0 <= row < 32",
        ]
    # Two unique predicates and the one referenced definition suffice.
    assert parse.call_count == 3


def test_normalized_domains_still_deduplicate() -> None:
    expression = domain(
        ["0 <= index < 16", "0 <= row + 1 < 16", "0 <= other < 16"],
        {"index": "row + 1", "other": "index"},
        set(),
    )
    assert chained_matmul._domain_bounds(expression) == ["0 <= row + 1 < 16"]


def test_duplicate_domains_do_not_hide_data_dependent_indices() -> None:
    expression = domain(
        ["0 <= row < 32"] * 10000 + ["0 <= index < 16"],
        {"index": "indirect + 1", "indirect": "loaded_position"},
        {"loaded_position"},
    )
    with pytest.raises(
        chained_matmul._UnsupportedChain,
        match="data-dependent contraction operand domain",
    ):
        chained_matmul._domain_bounds(expression)
