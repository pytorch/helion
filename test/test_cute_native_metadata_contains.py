from __future__ import annotations

import itertools
import random
import textwrap
from unittest.mock import patch

from helion._compiler.cute import native_matmul_metadata as metadata


def original(body: tuple[str, ...], segment: tuple[str, ...]) -> bool:
    expected = textwrap.dedent("\n".join(segment))
    return bool(segment) and any(
        textwrap.dedent("\n".join(body[i : i + len(segment)])) == expected
        for i in range(len(body) - len(segment) + 1)
    )


def test_exhaustive_indentation_and_multiline_equivalence() -> None:
    atoms = ("", " ", "\t", "x", " x", "\tx", "x\ny", "\n x")
    segments = [
        (),
        *itertools.product(atoms, repeat=1),
        *itertools.product(atoms, repeat=2),
    ]
    for body in itertools.product(atoms, repeat=3):
        for segment in segments:
            assert metadata._contains(body, segment) == original(body, segment)


def test_random_windows_and_unusual_whitespace_match_original() -> None:
    rng = random.Random(56017)
    atoms = (
        "",
        " ",
        "\t",
        "a",
        " a",
        "\ta",
        " a\n  b",
        "a\nb",
        "\r",
        " \r",
        "\v",
        "\f",
        "\u2003a",
        "a\n",
        "\n\n \t",
    )
    for _ in range(3000):
        body = tuple(rng.choice(atoms) for _ in range(rng.randrange(10)))
        if body and rng.randrange(2):
            start = rng.randrange(len(body))
            segment = body[start : start + rng.randrange(5)]
        else:
            segment = tuple(rng.choice(atoms) for _ in range(rng.randrange(6)))
        assert metadata._contains(body, segment) == original(body, segment)


def test_relative_control_scope_still_requires_complete_match() -> None:
    segment = ("if ready:", "    publish()", "consume()")
    assert metadata._contains(tuple("    " + line for line in segment), segment)
    assert not metadata._contains(("if ready:", "publish()", "consume()"), segment)
    assert not metadata._contains(("if ready:", "    consume()", "publish()"), segment)
    assert not metadata._contains(
        ("if ready:", "    publish()", "different()"), segment
    )
    # Joined contents can match even when tuple element boundaries differ.
    assert metadata._contains(("x", "y\nz"), ("x\ny", "z"))


def test_whole_window_dedent_is_limited_to_matching_stage_names() -> None:
    body = tuple(f"value_{i} = {i}" for i in range(1000))
    segment = body[600:900]
    dedent = textwrap.dedent
    with patch.object(metadata.textwrap, "dedent", wraps=dedent) as measured:
        assert metadata._contains(body, segment)
    # One normalization for the expected stage and one surviving window.
    assert measured.call_count == 2
