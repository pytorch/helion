"""Utilities for working with packaged PTXAS control files."""

from __future__ import annotations

from functools import cache
from pathlib import Path

_ADVANCED_COMPILER_CONFIGURATIONS: dict[int, str] = {
    # 1: "fp8fatt_0.bin",  # caused timeouts
    # 2: "fp8fatt_1.bin",
    # 3: "fp8fatt_2.bin",
    # 4: "fp8fatt_3.bin",
    # 5: "matmul_0.bin",
    # 6: "matmul_1.bin",
    # 7: "matmul_2.bin",
    # 8: "matmul_3.bin",
    # 9: "matmul_4.bin",
    # 10: "matmul_5.bin",
    11: "traveling-newt-499.bin",
    12: "brawny-crane-613.bin",
    13: "marvelous-stork-298.bin",
    14: "useful-moose-382.bin",
    15: "wise-slug-1.bin",
}


def _config_root() -> Path:
    return Path(__file__).resolve().parent


@cache
def search_ptxas_configs() -> tuple[int, ...]:
    """Return the sorted tuple of available PTXAS config IDs."""

    return tuple(sorted(_ADVANCED_COMPILER_CONFIGURATIONS))


def _advanced_compiler_configuration_path(config_id: int) -> str:
    """Return the absolute path to the advanced compiler configuration for ``config_id``."""

    try:
        filename = _ADVANCED_COMPILER_CONFIGURATIONS[config_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown advanced compiler configuration id: {config_id}"
        ) from exc
    resolved = (_config_root() / filename).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Missing advanced compiler configuration file: {resolved}"
        )
    return str(resolved)


@cache
def get_ptxas_option(config_value: int) -> str | None:
    """Translate a config enum value into a PTXAS option string."""

    if config_value == 0:
        return None
    return f"--apply-controls {_advanced_compiler_configuration_path(config_value)}"
