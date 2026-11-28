"""Ordinary shared launches that can execute a gathered fragment phase."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


# Number of physical threads proved for the ordinary gathered seed.
GATHERED_SEED_THREADS = 128


def gathered_launch_compatible(
    config: Mapping[str, object],
    *,
    actual_threads: int | None = None,
    required_threads: int | None = None,
) -> bool:
    return (
        config.get("cross_loop_pipeline") == "barrier"
        and str(config.get("pid_type", "")).startswith("persistent")
        and not config.get("cute_pdl")
        and (required_threads is None or actual_threads == required_threads)
    )
