"""Runtime validation for the stdlib-only grouped-GEMM reviewed profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from . import reviewed_profiles

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


def effective_reviewed_config(
    bound: BoundKernel[Any],
    profile: reviewed_profiles.ReviewedWorklistProfile,
) -> dict[str, dict[str, object]]:
    """Validate the reviewed request and return its effective config."""

    requested = reviewed_profiles.reviewed_config_values(profile.config_name)
    actual = bound._config
    if actual is None or actual.config != requested:
        raise RuntimeError(
            "AOT evaluation did not select the exact reviewed config "
            f"{profile.config_name}"
        )
    effective = bound.config_spec.normalized_config(actual)
    return {
        "requested": dict(actual.config),
        "effective": dict(effective.config),
    }
