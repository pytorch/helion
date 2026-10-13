from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field
import json
from typing import TYPE_CHECKING

from ..exc import InvalidConfig
from ..runtime.config import Config
from .config_fragment import EnumFragment

if TYPE_CHECKING:
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig


@dataclass(frozen=True, init=False)
class CompilerSeedWitness:
    """Supplemental carrier; never part of the legacy random seed prefix."""

    mechanism: str
    _config_json: str = field(repr=False)

    def __init__(self, mechanism: str, config: Config) -> None:
        object.__setattr__(self, "mechanism", mechanism)
        object.__setattr__(
            self,
            "_config_json",
            json.dumps(config.config, sort_keys=True, allow_nan=False),
        )

    @property
    def config(self) -> Config:
        return Config.from_dict(json.loads(self._config_json))

    def policy(self) -> tuple[str, str]:
        return self.mechanism, self._config_json


@dataclass
class SeedCoverageOutcome:
    origin: str
    requested: Config
    effective: Config | None
    status: str


def append_seed_coverage(
    population: list[FlatConfig], generation: ConfigGeneration
) -> tuple[list[FlatConfig], list[SeedCoverageOutcome]]:
    """Retain all base rows/draws; append strict, deduplicated seed carriers."""
    result = copy.deepcopy(population)
    seen: set[Config] = set()
    for flat in population:
        try:
            seen.add(generation.canonicalize_flat(copy.deepcopy(flat))[1])
        except InvalidConfig:
            continue
    outcomes: list[SeedCoverageOutcome] = []
    requests = [
        (f"compiler_seed:{index}", seed, False)
        for index, seed in enumerate(generation.config_spec.compiler_seed_configs)
    ]
    requests.extend(
        (witness.mechanism, witness.config, True)
        for witness in generation.config_spec.compiler_seed_witnesses
    )
    for origin, requested, supplemental in requests:
        try:
            carrier = copy.deepcopy(requested)
            # Flat-only roots omit this fixed default from flat decoding.
            # Remove only that redundant spelling; never repair a tunable or
            # a non-flat PID request. The ordinary strict validator still runs.
            if (
                "pid_type" not in generation._flat_fields()
                and carrier.config.get("pid_type") == "flat"
            ):
                carrier.config.pop("pid_type")
            flat, effective = generation.strict_config_pair(carrier)
            if supplemental:
                for fragment, value in zip(generation.flat_spec, flat, strict=True):
                    if isinstance(fragment, EnumFragment) and not any(
                        type(value) is type(choice) and value == choice
                        for choice in fragment._active_choices()
                    ):
                        raise InvalidConfig(
                            "Supplemental witness is outside an active enum domain"
                        )
        except InvalidConfig as error:
            outcomes.append(
                SeedCoverageOutcome(
                    origin, copy.deepcopy(requested), None, f"invalid: {error}"
                )
            )
            continue
        # Strict transfer honors caller overrides. Never label a changed
        # supplemental recipe as coverage of its original requested modes.
        if not supplemental:
            changed = False
        else:
            changed = any(
                key not in effective.config or effective.config[key] != value
                for key, value in carrier.config.items()
            )
        if changed:
            status = "overridden"
        elif effective in seen:
            status = "already_present"
        else:
            seen.add(effective)
            result.append(copy.deepcopy(flat))
            status = "pending"
        outcomes.append(
            SeedCoverageOutcome(origin, copy.deepcopy(requested), effective, status)
        )
    return result, outcomes
