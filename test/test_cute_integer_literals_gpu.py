"""Native generic comparisons narrow scalar constants before taking predicates."""

from __future__ import annotations

import torch

import helion
from helion._testing.cute_flash import compile_and_source
from helion._testing.cute_integer_literals import integer_literal_comparisons
from helion._testing.cute_integer_literals import integer_literal_inputs
from helion._testing.cute_integer_literals import integer_literal_reference


def test_integer_literal_comparisons_match_eager() -> None:
    values = integer_literal_inputs(torch.device("cuda"))
    bound = integer_literal_comparisons.bind((values,))
    compiled, _ = compile_and_source(bound, helion.Config(block_sizes=[32]))
    for offset in (0, 1):
        current = values + offset
        torch.testing.assert_close(
            compiled(current), integer_literal_reference(current)
        )
