"""Optional radix coordinates without importing compiler graph ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec

ALGORITHM_KEY = "cute_selection_algorithm"
ALGORITHMS = ("legacy", "radix")
GEOMETRY_CHOICES: dict[str, tuple[int, ...]] = {
    "cute_radix_threads": (256, 512, 1024),
    "cute_radix_cluster_ctas": (1, 2, 4, 8),
    "cute_radix_candidate_capacity": (0, 1024, 2048, 4096, 8192, 16384),
}
PIPELINE_CHOICES: dict[str, tuple[int, ...]] = {
    "cute_radix_packet_bits": (128, 256),
    "cute_radix_scan_unroll": (1, 2, 4),
}
HISTOGRAM_KEY = "cute_radix_histogram_mode"
HISTOGRAM_MODES = ("direct", "match")
BOOLEAN_KEYS = (
    "cute_radix_candidate_key_cache",
    "cute_radix_scalar_candidates",
    "cute_radix_valid_domain",
    "cute_radix_ticket_stores",
    "cute_radix_cached_payload_inverse",
)
CONFIG_KEYS = frozenset(
    (ALGORITHM_KEY, HISTOGRAM_KEY, *GEOMETRY_CHOICES, *BOOLEAN_KEYS, *PIPELINE_CHOICES)
)


def normalize_radix_selection(
    spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    if spec.cute_radix_selection_graph is None:
        if config.get(ALGORITHM_KEY, "legacy") != "legacy" and not fix_invalid:
            raise exc.InvalidConfig("radix requires a proved unordered selection root")
        for key in CONFIG_KEYS:
            if key in config and key != ALGORITHM_KEY and not fix_invalid:
                raise exc.InvalidConfig(f"{key} requires a radix selection capability")
            config.pop(key, None)
        return
    mode = config.setdefault(ALGORITHM_KEY, "legacy")
    if type(mode) is not str or mode not in ALGORITHMS:
        if not fix_invalid:
            raise exc.InvalidConfig(f"{ALGORITHM_KEY} must be one of {ALGORITHMS!r}")
        config[ALGORITHM_KEY] = mode = "legacy"
    for key, choices in (*GEOMETRY_CHOICES.items(), *PIPELINE_CHOICES.items()):
        value = config.setdefault(key, choices[0])
        if type(value) is not int or value not in choices:
            if not fix_invalid:
                raise exc.InvalidConfig(f"{key} must be one of {choices!r}")
            config[key] = choices[0]
    histogram = config.setdefault(HISTOGRAM_KEY, HISTOGRAM_MODES[0])
    if type(histogram) is not str or histogram not in HISTOGRAM_MODES:
        if not fix_invalid:
            raise exc.InvalidConfig(
                f"{HISTOGRAM_KEY} must be one of {HISTOGRAM_MODES!r}"
            )
        config[HISTOGRAM_KEY] = HISTOGRAM_MODES[0]
    for key in BOOLEAN_KEYS:
        value = config.setdefault(key, False)
        if type(value) is not bool:
            if not fix_invalid:
                raise exc.InvalidConfig(f"{key} must be a Boolean")
            config[key] = False
        if mode == "legacy" and config[key]:
            if not fix_invalid:
                raise exc.InvalidConfig(f"{key} requires the radix algorithm")
            config[key] = False
    if mode == "radix":
        # ConfigSpec is imported before Helion's graph modules during startup.
        from .radix_selection import validate_radix_schedule

        try:
            validate_radix_schedule(spec, config)
        except exc.InvalidConfig:
            if not fix_invalid:
                raise
            config[ALGORITHM_KEY] = "legacy"
            for key in BOOLEAN_KEYS:
                config[key] = False
