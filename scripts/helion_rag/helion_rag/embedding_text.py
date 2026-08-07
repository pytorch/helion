"""Shared index/query text composition for deployable embedding variants."""

from __future__ import annotations

import ast

DEPLOYABLE_VARIANTS = ("source", "cleaned", "comprehensive", "minimalist")


def _clean_source(source: str) -> str:
    return ast.unparse(ast.parse(source)).strip()


def query_text(
    source: str,
    shapes: str,
    dtypes: str,
    kernel_name: str,
    variant: str,
) -> str:
    """Compose the text embedded for a runtime lookup."""
    if variant == "source":
        return source.strip()
    if variant == "cleaned":
        return _clean_source(source)
    if variant == "comprehensive":
        return f"{source.strip()}\n# shapes: {shapes}\n# dtypes: {dtypes}"
    if variant == "minimalist":
        return f"{kernel_name} shapes={shapes}"
    raise ValueError(
        f"embedding variant {variant!r} is not runtime-deployable; "
        f"choose one of {', '.join(DEPLOYABLE_VARIANTS)}"
    )


def index_text(record: dict, variant: str) -> str:
    """Compose index text using the same contract as runtime queries."""
    return query_text(
        record["embed_text"],
        record["input_shapes"],
        record["dtypes"],
        record["kernel_name"],
        variant,
    )
